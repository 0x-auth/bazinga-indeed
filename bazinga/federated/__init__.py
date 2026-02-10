"""
BAZINGA Federated Learning - Collective Intelligence
=====================================================

The network learns TOGETHER without sharing raw data.

How it works:
1. Each node trains locally on their own data
2. Nodes share GRADIENTS (learning), not data (privacy!)
3. φ-weighted aggregation based on trust scores
4. Network becomes smarter than any single node

"Intelligence emerges from relationship, not size."

Components:
- LoRAAdapter: Small trainable layers for local learning
- GradientSharer: Privacy-preserving gradient exchange
- FederatedAggregator: φ-weighted collective learning
- CollectiveLearner: Main interface

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

from .lora import LoRAAdapter, create_lora_adapter
from .gradients import GradientPackage, GradientSharer
from .aggregator import FederatedAggregator, phi_weighted_average
from .learner import CollectiveLearner, create_learner

__all__ = [
    'LoRAAdapter',
    'create_lora_adapter',
    'GradientPackage',
    'GradientSharer',
    'FederatedAggregator',
    'phi_weighted_average',
    'CollectiveLearner',
    'create_learner',
]
