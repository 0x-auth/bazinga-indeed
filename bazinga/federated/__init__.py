"""
BAZINGA Federated Learning - Phase 3

Collaborative learning without sharing data:
- LoRA adapters for efficient fine-tuning
- Differential privacy for gradient protection
- Secure aggregation with homomorphic encryption
- Trust-weighted model updates

"Share learning, not data."
"""

from .lora_adapter import LoRAAdapter, LoRAConfig, create_lora_model
from .local_trainer import LocalTrainer, TrainingConfig
from .differential_privacy import DifferentialPrivacy, PrivacyConfig
from .secure_aggregation import SecureAggregator, PaillierKeyPair
from .federated_coordinator import FederatedCoordinator

__all__ = [
    'LoRAAdapter',
    'LoRAConfig',
    'create_lora_model',
    'LocalTrainer',
    'TrainingConfig',
    'DifferentialPrivacy',
    'PrivacyConfig',
    'SecureAggregator',
    'PaillierKeyPair',
    'FederatedCoordinator',
]
