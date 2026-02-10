"""
BAZINGA Local LLM - Offline AI
==============================
Run AI completely offline using llama-cpp-python.

First run downloads a small model (~700MB), then works forever offline.

Models (auto-download on first use):
- tinyllama (1.1B) - Fast, 700MB - Good for simple queries
- phi-2 (2.7B) - Balanced, 1.6GB - Good general purpose
- mistral (7B) - Best quality, 4GB - Best responses

Usage:
    from bazinga.local_llm import LocalLLM

    llm = LocalLLM()  # Auto-downloads model on first use
    response = llm.generate("What is consciousness?")
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Model configs - HuggingFace GGUF models
MODELS = {
    "tinyllama": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 700,
        "context": 2048,
    },
    "phi-2": {
        "repo": "TheBloke/phi-2-GGUF",
        "file": "phi-2.Q4_K_M.gguf",
        "size_mb": 1600,
        "context": 2048,
    },
    "mistral": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_mb": 4000,
        "context": 4096,
    },
}

DEFAULT_MODEL = "tinyllama"


class LocalLLM:
    """
    Local LLM using llama-cpp-python.

    Auto-downloads model on first use, then works offline forever.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, models_dir: Optional[str] = None):
        self.model_name = model_name
        self.models_dir = Path(models_dir or Path.home() / ".bazinga" / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.llm = None
        self.available = False

        # Check if llama-cpp-python is installed
        try:
            from llama_cpp import Llama
            self._llama_cpp_available = True
        except ImportError:
            self._llama_cpp_available = False

    def _get_model_path(self) -> Path:
        """Get path to model file."""
        config = MODELS.get(self.model_name, MODELS[DEFAULT_MODEL])
        return self.models_dir / config["file"]

    def _download_model(self) -> bool:
        """Download model from HuggingFace."""
        config = MODELS.get(self.model_name, MODELS[DEFAULT_MODEL])
        model_path = self._get_model_path()

        if model_path.exists():
            return True

        print(f"Downloading {self.model_name} ({config['size_mb']}MB)...")
        print("This only happens once, then BAZINGA works offline forever.")
        print()

        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=config["repo"],
                filename=config["file"],
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )

            print(f"Model downloaded to: {model_path}")
            return True

        except ImportError:
            print("Error: huggingface_hub not installed.")
            print("Run: pip install huggingface_hub")
            return False
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

    def load(self) -> bool:
        """Load the model into memory."""
        if not self._llama_cpp_available:
            print("Local LLM not available. Install with:")
            print("  pip install llama-cpp-python")
            return False

        model_path = self._get_model_path()

        # Download if needed
        if not model_path.exists():
            if not self._download_model():
                return False

        # Load model
        try:
            from llama_cpp import Llama

            config = MODELS.get(self.model_name, MODELS[DEFAULT_MODEL])

            print(f"Loading {self.model_name}...")
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=config["context"],
                n_threads=4,
                verbose=False,
            )
            self.available = True
            print("Model loaded. Ready for offline AI!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate response from prompt."""
        if not self.available:
            if not self.load():
                return "Local LLM not available. Use GROQ_API_KEY for cloud AI."

        try:
            # Format as chat
            formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

            response = self.llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|user|>", "<|end|>"],
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            return f"Error generating response: {e}"

    def chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Chat with conversation history."""
        if not self.available:
            if not self.load():
                return "Local LLM not available."

        # Format messages
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|{role}|>\n{content}\n"
        formatted += "<|assistant|>\n"

        try:
            response = self.llm(
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|user|>", "<|end|>"],
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def is_available() -> bool:
        """Check if local LLM dependencies are installed."""
        try:
            from llama_cpp import Llama
            return True
        except ImportError:
            return False

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """List available models."""
        return MODELS


# Singleton
_local_llm: Optional[LocalLLM] = None

def get_local_llm(model_name: str = DEFAULT_MODEL) -> LocalLLM:
    """Get global LocalLLM instance."""
    global _local_llm
    if _local_llm is None or _local_llm.model_name != model_name:
        _local_llm = LocalLLM(model_name)
    return _local_llm
