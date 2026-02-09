#!/usr/bin/env python3
"""
BAZINGA Local Model - On-Device Inference Without External APIs

Supports multiple backends:
- llama.cpp (via llama-cpp-python)
- ExLlama (for faster GPU inference)
- Transformers (fallback)
- MLX (Apple Silicon optimized)

Models:
- Phi-2 (2.7B params, ~3GB quantized)
- TinyLlama (1.1B params, ~1GB quantized)
- Mistral-7B (7B params, ~4GB quantized)

"Your AI, your hardware, your rules."
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Generator, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


class InferenceBackend(Enum):
    """Available inference backends."""
    LLAMA_CPP = "llama_cpp"      # llama-cpp-python
    EXLLAMA = "exllama"          # ExLlama/ExLlamaV2
    TRANSFORMERS = "transformers" # HuggingFace Transformers
    MLX = "mlx"                  # Apple MLX
    MOCK = "mock"                # For testing


class QuantizationType(Enum):
    """Model quantization types."""
    NONE = "none"           # Full precision (FP32)
    FP16 = "fp16"           # Half precision
    INT8 = "int8"           # 8-bit quantization
    INT4 = "int4"           # 4-bit quantization (GPTQ/AWQ)
    GGUF = "gguf"           # llama.cpp format


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    type: QuantizationType = QuantizationType.INT4
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True  # Descending activation order

    # Memory estimates (GB)
    @property
    def memory_factor(self) -> float:
        """Memory reduction factor compared to FP32."""
        factors = {
            QuantizationType.NONE: 1.0,
            QuantizationType.FP16: 0.5,
            QuantizationType.INT8: 0.25,
            QuantizationType.INT4: 0.125,
            QuantizationType.GGUF: 0.15,
        }
        return factors.get(self.type, 0.25)


@dataclass
class ModelConfig:
    """Configuration for local model inference."""
    # Model selection
    model_id: str = "microsoft/phi-2"
    model_path: Optional[str] = None  # Local path if already downloaded

    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Inference settings
    backend: InferenceBackend = InferenceBackend.LLAMA_CPP
    context_length: int = 2048
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1

    # Hardware
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    n_threads: int = 4
    use_mmap: bool = True
    use_mlock: bool = False

    # phi-coherence
    phi_temperature: bool = True  # Use phi-scaled temperature
    coherence_threshold: float = 0.5

    # Paths
    cache_dir: str = "~/.bazinga/models"

    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)

        # phi-scaled temperature
        if self.phi_temperature:
            self.temperature *= PHI / (1 + PHI)  # ~0.618 factor

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['quantization'] = {
            'type': self.quantization.type.value,
            'bits': self.quantization.bits,
            'group_size': self.quantization.group_size,
        }
        d['backend'] = self.backend.value
        return d

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        data['quantization'] = QuantizationConfig(
            type=QuantizationType(data['quantization']['type']),
            bits=data['quantization']['bits'],
            group_size=data['quantization']['group_size'],
        )
        data['backend'] = InferenceBackend(data['backend'])
        return cls(**data)


# Pre-defined model configurations
PRESET_MODELS = {
    "phi-2": ModelConfig(
        model_id="microsoft/phi-2",
        context_length=2048,
        n_gpu_layers=-1,
    ),
    "phi-2-gguf": ModelConfig(
        model_id="TheBloke/phi-2-GGUF",
        quantization=QuantizationConfig(type=QuantizationType.GGUF),
        context_length=2048,
    ),
    "tinyllama": ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_length=2048,
    ),
    "tinyllama-gguf": ModelConfig(
        model_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        quantization=QuantizationConfig(type=QuantizationType.GGUF),
        context_length=2048,
    ),
    "mistral-7b": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        context_length=4096,
        n_threads=8,
    ),
    "mistral-7b-gguf": ModelConfig(
        model_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        quantization=QuantizationConfig(type=QuantizationType.GGUF),
        context_length=4096,
    ),
}


class LocalModel:
    """
    Local model inference for BAZINGA.

    Provides on-device inference without external API dependencies.
    Supports multiple backends and quantization methods.

    Usage:
        model = LocalModel.from_preset("phi-2-gguf")
        response = model.generate("What is the meaning of life?")

        # Streaming
        for token in model.stream("Tell me a story"):
            print(token, end="", flush=True)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize local model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._loading = False
        self._lock = threading.Lock()

        # Stats
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.inference_count = 0

        print(f"LocalModel initialized:")
        print(f"  Model: {config.model_id}")
        print(f"  Backend: {config.backend.value}")
        print(f"  Quantization: {config.quantization.type.value}")

    @classmethod
    def from_preset(cls, preset_name: str) -> 'LocalModel':
        """
        Create model from preset configuration.

        Args:
            preset_name: One of: phi-2, phi-2-gguf, tinyllama, mistral-7b, etc.
        """
        if preset_name not in PRESET_MODELS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_MODELS.keys())}")
        return cls(PRESET_MODELS[preset_name])

    def load(self):
        """Load model into memory."""
        with self._lock:
            if self._loaded:
                return
            if self._loading:
                return

            self._loading = True

        try:
            print(f"Loading model {self.config.model_id}...")
            start = time.time()

            if self.config.backend == InferenceBackend.LLAMA_CPP:
                self._load_llama_cpp()
            elif self.config.backend == InferenceBackend.MLX:
                self._load_mlx()
            elif self.config.backend == InferenceBackend.TRANSFORMERS:
                self._load_transformers()
            elif self.config.backend == InferenceBackend.MOCK:
                self._load_mock()
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")

            elapsed = time.time() - start
            print(f"Model loaded in {elapsed:.2f}s")

            self._loaded = True

        finally:
            self._loading = False

    def _load_llama_cpp(self):
        """Load model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        # Download or locate model
        model_path = self._get_model_path()

        self.model = Llama(
            model_path=model_path,
            n_ctx=self.config.context_length,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=self.config.n_threads,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            verbose=False,
        )

    def _load_mlx(self):
        """Load model using Apple MLX."""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
        except ImportError:
            raise ImportError(
                "MLX not installed. Run: pip install mlx mlx-lm"
            )

        self.model, self.tokenizer = load(self.config.model_id)
        self._mlx_generate = generate

    def _load_transformers(self):
        """Load model using HuggingFace Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Load with quantization if specified
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_id,
            "trust_remote_code": True,
        }

        if self.config.quantization.type == QuantizationType.INT8:
            load_kwargs["load_in_8bit"] = True
        elif self.config.quantization.type == QuantizationType.INT4:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float16 if device != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if device != "cpu" and self.config.quantization.type in [QuantizationType.NONE, QuantizationType.FP16]:
            self.model = self.model.to(device)

        self._device = device

    def _load_mock(self):
        """Load mock model for testing."""
        self.model = "mock"
        self.tokenizer = "mock"

    def _get_model_path(self) -> str:
        """Get local path to model file, downloading if needed."""
        if self.config.model_path and os.path.exists(self.config.model_path):
            return self.config.model_path

        # Check cache
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Model-specific filename
        model_hash = hashlib.md5(self.config.model_id.encode()).hexdigest()[:8]
        quant_suffix = self.config.quantization.type.value
        model_file = cache_dir / f"{model_hash}_{quant_suffix}.gguf"

        if model_file.exists():
            return str(model_file)

        # Download from HuggingFace
        print(f"Downloading model to {model_file}...")
        try:
            from huggingface_hub import hf_hub_download

            # Find GGUF file in repo
            downloaded = hf_hub_download(
                repo_id=self.config.model_id,
                filename="*Q4_K_M.gguf",  # Common quantization
                local_dir=str(cache_dir),
            )
            return downloaded

        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Generated text
        """
        if not self._loaded:
            self.load()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        stop = stop or []

        start = time.time()

        if self.config.backend == InferenceBackend.LLAMA_CPP:
            result = self._generate_llama_cpp(prompt, max_tokens, temperature, stop)
        elif self.config.backend == InferenceBackend.MLX:
            result = self._generate_mlx(prompt, max_tokens, temperature)
        elif self.config.backend == InferenceBackend.TRANSFORMERS:
            result = self._generate_transformers(prompt, max_tokens, temperature)
        elif self.config.backend == InferenceBackend.MOCK:
            result = self._generate_mock(prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        elapsed = time.time() - start
        self.total_inference_time += elapsed
        self.inference_count += 1
        self.total_tokens_generated += len(result.split())  # Approximate

        return result

    def _generate_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: List[str],
    ) -> str:
        """Generate using llama-cpp."""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=stop,
        )
        return output['choices'][0]['text']

    def _generate_mlx(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using MLX."""
        return self._mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        )

    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Transformers."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self, '_device'):
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repeat_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_mock(self, prompt: str, max_tokens: int) -> str:
        """Generate mock response for testing."""
        return f"[MOCK RESPONSE to '{prompt[:50]}...'] This is a test response with phi={PHI:.3f} coherence."

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream generated text token by token.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            callback: Optional callback for each token

        Yields:
            Generated tokens
        """
        if not self._loaded:
            self.load()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        if self.config.backend == InferenceBackend.LLAMA_CPP:
            yield from self._stream_llama_cpp(prompt, max_tokens, temperature, callback)
        elif self.config.backend == InferenceBackend.MOCK:
            yield from self._stream_mock(prompt, callback)
        else:
            # Fallback to non-streaming
            result = self.generate(prompt, max_tokens, temperature)
            for word in result.split():
                token = word + " "
                if callback:
                    callback(token)
                yield token

    def _stream_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
    ) -> Generator[str, None, None]:
        """Stream using llama-cpp."""
        for output in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            stream=True,
        ):
            token = output['choices'][0]['text']
            if callback:
                callback(token)
            yield token

    def _stream_mock(
        self,
        prompt: str,
        callback: Optional[Callable[[str], None]],
    ) -> Generator[str, None, None]:
        """Stream mock response."""
        response = f"[MOCK] Streaming response for: {prompt[:30]}..."
        for word in response.split():
            token = word + " "
            if callback:
                callback(token)
            yield token
            time.sleep(0.05)  # Simulate delay

    def compute_phi_coherence(self, text: str) -> float:
        """
        Compute phi-coherence of generated text.

        Returns score 0-1 indicating alignment with phi patterns.
        """
        if not text:
            return 0.0

        words = text.split()
        if len(words) < 3:
            return 0.5

        # Check sentence length ratios
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Ideal: consecutive sentence lengths follow phi ratio
        ratios = []
        for i in range(len(sentences) - 1):
            len1 = len(sentences[i].split())
            len2 = len(sentences[i + 1].split())
            if len2 > 0:
                ratios.append(len1 / len2)

        if not ratios:
            return 0.5

        # Score based on proximity to phi
        phi_scores = [1 - min(1, abs(r - PHI) / PHI) for r in ratios]

        return sum(phi_scores) / len(phi_scores)

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = self.total_inference_time / max(1, self.inference_count)
        tokens_per_sec = self.total_tokens_generated / max(0.001, self.total_inference_time)

        return {
            'model_id': self.config.model_id,
            'backend': self.config.backend.value,
            'quantization': self.config.quantization.type.value,
            'loaded': self._loaded,
            'inference_count': self.inference_count,
            'total_tokens': self.total_tokens_generated,
            'total_time': self.total_inference_time,
            'avg_time_per_inference': avg_time,
            'tokens_per_second': tokens_per_sec,
        }

    def unload(self):
        """Unload model from memory."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            self._loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            print("Model unloaded from memory")


def get_available_backends() -> List[str]:
    """Get list of available inference backends."""
    available = []

    try:
        import llama_cpp
        available.append("llama_cpp")
    except ImportError:
        pass

    try:
        import mlx
        available.append("mlx")
    except ImportError:
        pass

    try:
        import transformers
        available.append("transformers")
    except ImportError:
        pass

    available.append("mock")  # Always available

    return available


def estimate_memory_usage(model_id: str, quantization: QuantizationType) -> float:
    """
    Estimate memory usage in GB.

    Args:
        model_id: Model identifier
        quantization: Quantization type

    Returns:
        Estimated memory in GB
    """
    # Base sizes (approximate, FP32)
    model_sizes = {
        "phi-2": 5.4,      # 2.7B params
        "tinyllama": 2.2,  # 1.1B params
        "mistral-7b": 14,  # 7B params
        "llama-7b": 14,
        "llama-13b": 26,
    }

    # Find matching model
    base_size = 5.0  # Default
    for name, size in model_sizes.items():
        if name in model_id.lower():
            base_size = size
            break

    # Apply quantization factor
    config = QuantizationConfig(type=quantization)
    return base_size * config.memory_factor


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Local Model Test")
    print("=" * 60)

    # Check available backends
    print(f"\nAvailable backends: {get_available_backends()}")

    # Memory estimates
    print("\nMemory estimates:")
    for model in ["phi-2", "tinyllama", "mistral-7b"]:
        for quant in [QuantizationType.NONE, QuantizationType.INT4, QuantizationType.GGUF]:
            mem = estimate_memory_usage(model, quant)
            print(f"  {model} ({quant.value}): {mem:.1f} GB")

    # Test with mock backend
    print("\nTesting mock backend:")
    config = ModelConfig(
        model_id="test/mock-model",
        backend=InferenceBackend.MOCK,
    )
    model = LocalModel(config)

    # Generate
    response = model.generate("What is consciousness?")
    print(f"Response: {response}")

    # Coherence
    coherence = model.compute_phi_coherence(response)
    print(f"phi-coherence: {coherence:.3f}")

    # Stream
    print("\nStreaming:")
    for token in model.stream("Tell me about phi"):
        print(token, end="", flush=True)
    print()

    # Stats
    print(f"\nStats: {model.get_stats()}")

    print("\n" + "=" * 60)
    print("Local Model module ready!")
    print("=" * 60)
