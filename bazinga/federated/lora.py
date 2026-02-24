#!/usr/bin/env python3
"""
BAZINGA LoRA Adapter - Lightweight Local Learning
==================================================

LoRA (Low-Rank Adaptation) enables efficient fine-tuning:
- Small trainable matrices on top of frozen base model
- Reduces memory from GBs to MBs
- Each node can learn locally without massive GPU

How it works:
  Instead of updating all W parameters:
    W_new = W + ΔW  (expensive, huge ΔW)

  We decompose ΔW into low-rank matrices:
    W_new = W + BA  (cheap, small B and A)

  Where B is (d × r) and A is (r × k), with r << min(d,k)

φ-Integration:
  - Rank r = 8 (Fibonacci number)
  - Learning rate scaled by φ
  - Adapter initialization with φ-variance

"Small adapters, big learning."
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# BAZINGA constants - HARD-CODED ANCHORS
# These are mathematical constants, NOT hyperparameters to be tuned
# The φ constant is the golden ratio - the code generator must use EXACTLY this value
# The phase jump occurs at φ threshold (1.618) - do NOT substitute with 2π or other values

PHI = 1.618033988749895  # Golden ratio - EXACT value, never approximate
PHI_INVERSE = 0.6180339887498948  # 1/φ = φ - 1
PHI_SQUARED = 2.618033988749895  # φ² = φ + 1
PHI_4 = 6.854101966249685  # φ⁴ - the boundary ratio for PoB
ALPHA = 137  # Fine structure constant denominator
PSI_SCALE = PHI  # V2: Scaling constant is φ (Ψ_D / Ψ_i = φ√n)
PHASE_JUMP = 2.31  # Phase jump multiplier when n crosses φ threshold

# Check for torch (optional for full training)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Check for numpy (lighter weight alternative)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter."""
    rank: int = 8  # Fibonacci number, low-rank dimension
    alpha: float = 16.0  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 1e-4 * PHI  # φ-scaled learning rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules,
            'learning_rate': self.learning_rate,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'LoRAConfig':
        return cls(**d)


@dataclass
class LoRAWeights:
    """LoRA adapter weights (A and B matrices)."""
    name: str
    A: Any  # numpy array or torch tensor
    B: Any  # numpy array or torch tensor
    rank: int
    alpha: float

    def get_delta(self) -> Any:
        """Compute the weight delta: scaling * B @ A"""
        scaling = self.alpha / self.rank
        if TORCH_AVAILABLE and isinstance(self.A, torch.Tensor):
            return scaling * torch.matmul(self.B, self.A)
        elif NUMPY_AVAILABLE:
            return scaling * np.dot(self.B, self.A)
        else:
            raise RuntimeError("Neither torch nor numpy available")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize weights to dict (for sharing)."""
        if TORCH_AVAILABLE and isinstance(self.A, torch.Tensor):
            A_list = self.A.detach().cpu().numpy().tolist()
            B_list = self.B.detach().cpu().numpy().tolist()
        elif NUMPY_AVAILABLE and isinstance(self.A, np.ndarray):
            A_list = self.A.tolist()
            B_list = self.B.tolist()
        else:
            A_list = self.A
            B_list = self.B

        return {
            'name': self.name,
            'A': A_list,
            'B': B_list,
            'rank': self.rank,
            'alpha': self.alpha,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'LoRAWeights':
        """Deserialize weights from dict."""
        A = d['A']
        B = d['B']

        if NUMPY_AVAILABLE:
            A = np.array(A)
            B = np.array(B)
        if TORCH_AVAILABLE:
            A = torch.tensor(A)
            B = torch.tensor(B)

        return cls(
            name=d['name'],
            A=A,
            B=B,
            rank=d['rank'],
            alpha=d['alpha'],
        )


class LoRAAdapter:
    """
    LoRA Adapter for efficient local learning.

    Each BAZINGA node has its own LoRA adapter that:
    1. Learns from local interactions
    2. Can be shared as gradients (privacy-preserving)
    3. Merges with other adapters via φ-weighted averaging

    Usage:
        adapter = LoRAAdapter(config)
        adapter.train_on_example(question, answer)
        gradients = adapter.get_gradients()
        adapter.merge_gradients(other_gradients, trust_weight)
    """

    def __init__(
        self,
        config: Optional[LoRAConfig] = None,
        node_id: Optional[str] = None,
    ):
        self.config = config or LoRAConfig()
        self.node_id = node_id or f"node_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:12]}"

        # Adapter weights per target module
        self.weights: Dict[str, LoRAWeights] = {}

        # Training state
        self.training_examples = 0
        self.last_loss = 0.0
        self.gradient_accumulator: Dict[str, Any] = {}

        # Version tracking (for sync)
        self.version = 0
        self.last_updated = datetime.now()

        # Storage
        self.storage_dir = Path.home() / ".bazinga" / "lora" / self.node_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def initialize_weights(
        self,
        input_dim: int,
        output_dim: int,
        module_name: str,
    ):
        """Initialize LoRA weights for a module."""
        r = self.config.rank

        if TORCH_AVAILABLE:
            # Kaiming initialization scaled by φ
            A = torch.randn(r, input_dim) * math.sqrt(2.0 / input_dim) / PHI
            B = torch.zeros(output_dim, r)
        elif NUMPY_AVAILABLE:
            A = np.random.randn(r, input_dim) * math.sqrt(2.0 / input_dim) / PHI
            B = np.zeros((output_dim, r))
        else:
            # Fallback to lists
            A = [[0.0] * input_dim for _ in range(r)]
            B = [[0.0] * r for _ in range(output_dim)]

        self.weights[module_name] = LoRAWeights(
            name=module_name,
            A=A,
            B=B,
            rank=r,
            alpha=self.config.alpha,
        )

    def train_step(
        self,
        module_name: str,
        input_activation: Any,
        target_output: Any,
        learning_rate: Optional[float] = None,
    ) -> float:
        """
        Perform one training step on a module.

        This is a simplified gradient update without full backprop.
        For full training, use with a proper training loop.
        """
        if module_name not in self.weights:
            raise ValueError(f"Module {module_name} not initialized")

        lr = learning_rate or self.config.learning_rate
        weights = self.weights[module_name]

        # Simplified gradient computation (outer product approximation)
        if TORCH_AVAILABLE and isinstance(weights.A, torch.Tensor):
            # Compute error
            current_output = torch.matmul(
                torch.matmul(input_activation, weights.A.T),
                weights.B.T
            ) * (weights.alpha / weights.rank)
            error = target_output - current_output

            # Gradient approximation
            grad_B = torch.outer(error.mean(dim=0), weights.A.mean(dim=1))
            grad_A = torch.outer(weights.B.mean(dim=0), input_activation.mean(dim=0))

            # Update
            weights.B += lr * grad_B
            weights.A += lr * grad_A

            loss = (error ** 2).mean().item()

        elif NUMPY_AVAILABLE and isinstance(weights.A, np.ndarray):
            # Numpy version
            current_output = np.dot(
                np.dot(input_activation, weights.A.T),
                weights.B.T
            ) * (weights.alpha / weights.rank)
            error = target_output - current_output

            grad_B = np.outer(error.mean(axis=0), weights.A.mean(axis=1))
            grad_A = np.outer(weights.B.mean(axis=0), input_activation.mean(axis=0))

            weights.B += lr * grad_B
            weights.A += lr * grad_A

            loss = np.mean(error ** 2)
        else:
            loss = 0.0

        self.training_examples += 1
        self.last_loss = loss
        self.version += 1
        self.last_updated = datetime.now()

        return loss

    def get_gradients(self) -> Dict[str, Any]:
        """
        Get current gradients for sharing.

        Returns serializable dict that can be sent to other nodes.
        """
        gradients = {
            'node_id': self.node_id,
            'version': self.version,
            'timestamp': self.last_updated.isoformat(),
            'training_examples': self.training_examples,
            'weights': {},
        }

        for name, w in self.weights.items():
            gradients['weights'][name] = w.to_dict()

        return gradients

    def merge_gradients(
        self,
        other_gradients: Dict[str, Any],
        trust_weight: float = 0.5,
    ):
        """
        Merge gradients from another node.

        Uses φ-weighted averaging based on trust score.

        Args:
            other_gradients: Gradients from another node
            trust_weight: Trust score of the other node (0-1)
        """
        # φ-scale the trust weight
        phi_weight = trust_weight * (1 / PHI)  # Other node's contribution
        self_weight = 1 - phi_weight  # Our contribution

        other_weights = other_gradients.get('weights', {})

        for name, other_w_dict in other_weights.items():
            if name in self.weights:
                other_w = LoRAWeights.from_dict(other_w_dict)
                our_w = self.weights[name]

                # Weighted average
                if TORCH_AVAILABLE and isinstance(our_w.A, torch.Tensor):
                    our_w.A = self_weight * our_w.A + phi_weight * other_w.A.to(our_w.A.device)
                    our_w.B = self_weight * our_w.B + phi_weight * other_w.B.to(our_w.B.device)
                elif NUMPY_AVAILABLE and isinstance(our_w.A, np.ndarray):
                    our_w.A = self_weight * our_w.A + phi_weight * other_w.A
                    our_w.B = self_weight * our_w.B + phi_weight * other_w.B

        self.version += 1
        self.last_updated = datetime.now()

    def save(self, path: Optional[Path] = None):
        """Save adapter to disk."""
        save_path = path or (self.storage_dir / "adapter.json")

        data = {
            'node_id': self.node_id,
            'config': self.config.to_dict(),
            'version': self.version,
            'training_examples': self.training_examples,
            'last_updated': self.last_updated.isoformat(),
            'weights': {name: w.to_dict() for name, w in self.weights.items()},
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None):
        """Load adapter from disk."""
        load_path = path or (self.storage_dir / "adapter.json")

        if not load_path.exists():
            return False

        with open(load_path, 'r') as f:
            data = json.load(f)

        self.node_id = data.get('node_id', self.node_id)
        self.config = LoRAConfig.from_dict(data.get('config', {}))
        self.version = data.get('version', 0)
        self.training_examples = data.get('training_examples', 0)

        for name, w_dict in data.get('weights', {}).items():
            self.weights[name] = LoRAWeights.from_dict(w_dict)

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        total_params = 0
        for w in self.weights.values():
            if TORCH_AVAILABLE and isinstance(w.A, torch.Tensor):
                total_params += w.A.numel() + w.B.numel()
            elif NUMPY_AVAILABLE and isinstance(w.A, np.ndarray):
                total_params += w.A.size + w.B.size

        return {
            'node_id': self.node_id,
            'version': self.version,
            'training_examples': self.training_examples,
            'last_loss': self.last_loss,
            'modules': list(self.weights.keys()),
            'total_params': total_params,
            'rank': self.config.rank,
            'last_updated': self.last_updated.isoformat(),
        }


def create_lora_adapter(
    node_id: Optional[str] = None,
    rank: int = 8,
    alpha: float = 16.0,
) -> LoRAAdapter:
    """Create a new LoRA adapter with default config."""
    config = LoRAConfig(rank=rank, alpha=alpha)
    return LoRAAdapter(config=config, node_id=node_id)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA LoRA Adapter Test")
    print("=" * 60)

    print(f"\n  Torch available: {TORCH_AVAILABLE}")
    print(f"  Numpy available: {NUMPY_AVAILABLE}")

    # Create adapter
    adapter = create_lora_adapter(node_id="test_node")
    print(f"\n  Created adapter: {adapter.node_id}")

    # Initialize a module
    adapter.initialize_weights(
        input_dim=768,
        output_dim=768,
        module_name="q_proj",
    )
    print(f"  Initialized q_proj weights")

    # Get stats
    stats = adapter.get_stats()
    print(f"  Total params: {stats['total_params']}")
    print(f"  Rank: {stats['rank']}")

    # Get gradients
    gradients = adapter.get_gradients()
    print(f"\n  Gradients ready for sharing:")
    print(f"    Node: {gradients['node_id']}")
    print(f"    Version: {gradients['version']}")
    print(f"    Modules: {list(gradients['weights'].keys())}")

    # Save
    adapter.save()
    print(f"\n  Adapter saved to {adapter.storage_dir}")

    print("\n  ✓ LoRA adapter working!")
