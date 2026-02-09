"""
BAZINGA Inference - Phase 4: On-Device & Distributed Inference

Local model inference without external API dependencies:
- On-device models (Phi-2, TinyLlama, Mistral)
- Quantization (4-bit GPTQ/AWQ)
- Distributed inference across P2P network
- Specialized node routing

"Intelligence that lives where you live."
"""

from .local_model import LocalModel, ModelConfig, QuantizationConfig
from .distributed_inference import DistributedInference, InferenceNode
from .model_router import ModelRouter, DomainExpert

__all__ = [
    'LocalModel',
    'ModelConfig',
    'QuantizationConfig',
    'DistributedInference',
    'InferenceNode',
    'ModelRouter',
    'DomainExpert',
]
