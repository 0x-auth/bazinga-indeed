"""
BAZINGA Inference - Phase 4: On-Device & Distributed Inference

Local model inference without external API dependencies:
- On-device models (Phi-2, TinyLlama, Mistral)
- Ollama support (localhost:11434)
- Quantization (4-bit GPTQ/AWQ)
- Distributed inference across P2P network
- Specialized node routing
- PHI trust bonus for local model nodes

"Intelligence that lives where you live."
"""

from .local_model import LocalModel, ModelConfig, QuantizationConfig
from .distributed_inference import DistributedInference, InferenceNode
from .model_router import ModelRouter, DomainExpert
from .ollama_detector import (
    detect_ollama,
    detect_any_local_model,
    LocalModelStatus,
    LocalModelType,
    generate_with_ollama_sync,
    compute_pob_with_local_latency,
    print_local_status,
)

__all__ = [
    'LocalModel',
    'ModelConfig',
    'QuantizationConfig',
    'DistributedInference',
    'InferenceNode',
    'ModelRouter',
    'DomainExpert',
    # Ollama detector
    'detect_ollama',
    'detect_any_local_model',
    'LocalModelStatus',
    'LocalModelType',
    'generate_with_ollama_sync',
    'compute_pob_with_local_latency',
    'print_local_status',
]
