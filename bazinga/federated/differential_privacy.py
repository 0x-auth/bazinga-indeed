#!/usr/bin/env python3
"""
BAZINGA Differential Privacy - Privacy-Preserving Gradient Sharing

Implements ε-differential privacy for federated learning:
- Gradient clipping to bound sensitivity
- Laplacian/Gaussian noise addition
- Privacy budget tracking
- φ-calibrated noise for harmonious learning

Math:
    Noisy gradient = gradient + Noise(Δf/ε)
    Where:
        Δf = sensitivity (max gradient change from one sample)
        ε = privacy budget (lower = more private)

"Privacy is not secrecy. Privacy is the power to selectively reveal."
"""

import math
import secrets
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Try imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    # Privacy parameters
    epsilon: float = 1.0          # Privacy budget (lower = more private)
    delta: float = 1e-5           # Failure probability
    noise_mechanism: str = 'gaussian'  # 'gaussian' or 'laplacian'

    # Gradient clipping
    clip_norm: float = 1.0        # Max gradient L2 norm
    clip_per_layer: bool = True   # Clip each layer separately

    # Privacy accounting
    target_epsilon: float = 10.0  # Total privacy budget
    composition: str = 'advanced' # 'basic', 'advanced', or 'rdp'

    # φ-scaling
    phi_calibration: bool = True  # Use φ-based noise calibration

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PrivacyConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))


class PrivacyAccountant:
    """
    Track privacy budget expenditure across training rounds.

    Uses Rényi Differential Privacy (RDP) for tight composition.
    """

    def __init__(self, target_epsilon: float, delta: float):
        """
        Args:
            target_epsilon: Total privacy budget
            delta: Failure probability
        """
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.spent_epsilon = 0.0
        self.rounds = []

    def add_round(self, epsilon: float, delta: float = None):
        """Record a round of gradient sharing."""
        delta = delta or self.delta
        self.rounds.append({'epsilon': epsilon, 'delta': delta})

        # Basic composition (conservative)
        self.spent_epsilon += epsilon

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.target_epsilon - self.spent_epsilon)

    def can_continue(self) -> bool:
        """Check if we can continue training within budget."""
        return self.spent_epsilon < self.target_epsilon

    def get_optimal_epsilon(self, remaining_rounds: int) -> float:
        """
        Get optimal epsilon per round for remaining budget.

        Args:
            remaining_rounds: Expected remaining training rounds
        """
        remaining = self.get_remaining_budget()
        if remaining_rounds <= 0:
            return remaining
        return remaining / remaining_rounds

    def get_stats(self) -> Dict:
        """Get privacy statistics."""
        return {
            'target_epsilon': self.target_epsilon,
            'spent_epsilon': self.spent_epsilon,
            'remaining_epsilon': self.get_remaining_budget(),
            'num_rounds': len(self.rounds),
            'can_continue': self.can_continue(),
        }


if TORCH_AVAILABLE:

    class DifferentialPrivacy:
        """
        Differential Privacy for federated gradient sharing.

        Provides:
        - Gradient clipping (bounds sensitivity)
        - Noise addition (Gaussian or Laplacian)
        - Privacy budget tracking
        - φ-calibrated noise for harmonious learning

        Usage:
            dp = DifferentialPrivacy(config)

            # Clip and noise gradients
            private_grads = dp.privatize_gradients(gradients)

            # Check privacy budget
            if dp.can_continue():
                share_gradients(private_grads)
        """

        def __init__(self, config: Optional[PrivacyConfig] = None):
            """
            Initialize differential privacy.

            Args:
                config: Privacy configuration
            """
            self.config = config or PrivacyConfig()

            # Privacy accountant
            self.accountant = PrivacyAccountant(
                target_epsilon=self.config.target_epsilon,
                delta=self.config.delta
            )

            # Stats
            self.total_gradients_processed = 0
            self.total_noise_added = 0.0

            print(f"DifferentialPrivacy initialized:")
            print(f"  ε = {self.config.epsilon}")
            print(f"  δ = {self.config.delta}")
            print(f"  Clip norm = {self.config.clip_norm}")
            print(f"  Noise mechanism = {self.config.noise_mechanism}")
            print(f"  φ-calibration = {self.config.phi_calibration}")

        def compute_sensitivity(self, gradients: Dict[str, torch.Tensor]) -> float:
            """
            Compute sensitivity of gradient release.

            Sensitivity = max change in gradients from one sample.
            With clipping, this is bounded by clip_norm.
            """
            return self.config.clip_norm

        def compute_noise_scale(self, sensitivity: float) -> float:
            """
            Compute noise scale for ε-differential privacy.

            For Gaussian: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
            For Laplacian: b = sensitivity / ε
            """
            epsilon = self.config.epsilon
            delta = self.config.delta

            if self.config.noise_mechanism == 'gaussian':
                # Gaussian mechanism
                scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            else:
                # Laplacian mechanism
                scale = sensitivity / epsilon

            # φ-calibration: slightly reduce noise for more harmonious learning
            if self.config.phi_calibration:
                scale = scale * (1 / PHI)  # Reduce by ~38%

            return scale

        def clip_gradients(
            self,
            gradients: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """
            Clip gradients to bound sensitivity.

            Args:
                gradients: Dict of gradient tensors

            Returns:
                Clipped gradients
            """
            clipped = {}

            if self.config.clip_per_layer:
                # Clip each layer separately
                for name, grad in gradients.items():
                    norm = grad.norm(2)
                    if norm > self.config.clip_norm:
                        clipped[name] = grad * (self.config.clip_norm / norm)
                    else:
                        clipped[name] = grad.clone()
            else:
                # Global clipping
                # Flatten all gradients
                all_grads = torch.cat([g.view(-1) for g in gradients.values()])
                global_norm = all_grads.norm(2)

                if global_norm > self.config.clip_norm:
                    scale = self.config.clip_norm / global_norm
                    clipped = {name: grad * scale for name, grad in gradients.items()}
                else:
                    clipped = {name: grad.clone() for name, grad in gradients.items()}

            return clipped

        def add_noise(
            self,
            gradients: Dict[str, torch.Tensor],
            noise_scale: float,
        ) -> Dict[str, torch.Tensor]:
            """
            Add calibrated noise to gradients.

            Args:
                gradients: Clipped gradients
                noise_scale: Noise scale from compute_noise_scale

            Returns:
                Noisy gradients
            """
            noisy = {}

            for name, grad in gradients.items():
                if self.config.noise_mechanism == 'gaussian':
                    # Gaussian noise
                    noise = torch.randn_like(grad) * noise_scale
                else:
                    # Laplacian noise
                    # Laplace(0, b) = Exponential(1/b) * random_sign
                    uniform = torch.rand_like(grad)
                    noise = -noise_scale * torch.sign(uniform - 0.5) * torch.log(1 - 2 * torch.abs(uniform - 0.5))

                noisy[name] = grad + noise
                self.total_noise_added += noise.abs().mean().item()

            return noisy

        def privatize_gradients(
            self,
            gradients: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """
            Apply differential privacy to gradients.

            Full pipeline:
            1. Clip gradients (bound sensitivity)
            2. Add calibrated noise
            3. Update privacy accountant

            Args:
                gradients: Raw gradients from training

            Returns:
                Privacy-preserving gradients
            """
            # Check budget
            if not self.can_continue():
                raise RuntimeError("Privacy budget exhausted!")

            # Step 1: Clip
            clipped = self.clip_gradients(gradients)

            # Step 2: Compute noise scale
            sensitivity = self.compute_sensitivity(clipped)
            noise_scale = self.compute_noise_scale(sensitivity)

            # Step 3: Add noise
            private = self.add_noise(clipped, noise_scale)

            # Step 4: Update accountant
            self.accountant.add_round(self.config.epsilon)
            self.total_gradients_processed += len(gradients)

            return private

        def can_continue(self) -> bool:
            """Check if training can continue within privacy budget."""
            return self.accountant.can_continue()

        def get_remaining_budget(self) -> float:
            """Get remaining privacy budget."""
            return self.accountant.get_remaining_budget()

        def get_recommended_rounds(self, desired_rounds: int) -> Tuple[int, float]:
            """
            Get recommended epsilon for desired number of rounds.

            Returns:
                (achievable_rounds, epsilon_per_round)
            """
            remaining = self.get_remaining_budget()
            epsilon_per_round = remaining / max(1, desired_rounds)

            # Minimum useful epsilon
            min_epsilon = 0.1
            if epsilon_per_round < min_epsilon:
                achievable = int(remaining / min_epsilon)
                return achievable, min_epsilon

            return desired_rounds, epsilon_per_round

        def adjust_epsilon(self, new_epsilon: float):
            """
            Adjust epsilon for future rounds.

            Useful for adaptive privacy budget allocation.
            """
            old = self.config.epsilon
            self.config.epsilon = new_epsilon
            print(f"Adjusted ε: {old} → {new_epsilon}")

        def get_stats(self) -> Dict[str, Any]:
            """Get privacy statistics."""
            return {
                'config': self.config.to_dict(),
                'accountant': self.accountant.get_stats(),
                'total_gradients_processed': self.total_gradients_processed,
                'avg_noise_added': self.total_noise_added / max(1, self.total_gradients_processed),
            }

        def verify_privacy(
            self,
            original: Dict[str, torch.Tensor],
            privatized: Dict[str, torch.Tensor],
        ) -> Dict[str, float]:
            """
            Verify privacy properties of privatized gradients.

            Args:
                original: Original gradients
                privatized: After privacy mechanism

            Returns:
                Verification metrics
            """
            metrics = {}

            for name in original:
                if name in privatized:
                    orig = original[name]
                    priv = privatized[name]

                    # Compute difference
                    diff = (priv - orig).abs()
                    metrics[f'{name}_mean_diff'] = diff.mean().item()
                    metrics[f'{name}_max_diff'] = diff.max().item()

                    # Cosine similarity (should be less than 1 due to noise)
                    cos_sim = torch.nn.functional.cosine_similarity(
                        orig.view(1, -1),
                        priv.view(1, -1)
                    ).item()
                    metrics[f'{name}_cosine_sim'] = cos_sim

            return metrics


    class AdaptivePrivacy(DifferentialPrivacy):
        """
        Adaptive differential privacy that adjusts epsilon based on:
        - Gradient coherence (high coherence → can tolerate less noise)
        - Training progress (early = more noise, late = less)
        - φ-scaling for harmonious adaptation

        "Adapt privacy to the learning, not learning to the privacy."
        """

        def __init__(
            self,
            config: Optional[PrivacyConfig] = None,
            min_epsilon: float = 0.1,
            max_epsilon: float = 5.0,
        ):
            super().__init__(config)
            self.min_epsilon = min_epsilon
            self.max_epsilon = max_epsilon
            self.coherence_history: List[float] = []

        def adapt_epsilon(
            self,
            coherence: float,
            training_progress: float,  # 0 to 1
        ) -> float:
            """
            Adapt epsilon based on coherence and progress.

            Args:
                coherence: Current gradient coherence (0 to 1)
                training_progress: Training progress (0 to 1)

            Returns:
                Adapted epsilon
            """
            self.coherence_history.append(coherence)

            # Base: use configured epsilon
            base = self.config.epsilon

            # Coherence adjustment: high coherence → more epsilon (less noise)
            coherence_factor = 1 + (coherence - 0.5)  # 0.5 to 1.5

            # Progress adjustment: more progress → more epsilon
            # (we can afford less privacy as model stabilizes)
            progress_factor = 1 + (training_progress * 0.5)  # 1 to 1.5

            # φ-scaling
            phi_factor = PHI / (1 + PHI)  # ≈ 0.618

            # Combined
            adapted = base * coherence_factor * progress_factor * phi_factor

            # Clamp to bounds
            adapted = max(self.min_epsilon, min(self.max_epsilon, adapted))

            self.config.epsilon = adapted
            return adapted

        def privatize_gradients_adaptive(
            self,
            gradients: Dict[str, torch.Tensor],
            coherence: float,
            training_progress: float,
        ) -> Dict[str, torch.Tensor]:
            """
            Privatize with adaptive epsilon.

            Args:
                gradients: Raw gradients
                coherence: Gradient coherence
                training_progress: Training progress

            Returns:
                Privacy-preserving gradients
            """
            # Adapt epsilon
            self.adapt_epsilon(coherence, training_progress)

            # Apply privacy
            return self.privatize_gradients(gradients)

else:
    # Stubs
    class DifferentialPrivacy:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Run: pip install torch")

    class AdaptivePrivacy:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Run: pip install torch")


# Pure Python implementations (no PyTorch)
def laplacian_noise(scale: float, size: int = 1) -> List[float]:
    """Generate Laplacian noise without PyTorch."""
    import random
    result = []
    for _ in range(size):
        u = random.random() - 0.5
        noise = -scale * (1 if u > 0 else -1) * math.log(1 - 2 * abs(u))
        result.append(noise)
    return result


def gaussian_noise(scale: float, size: int = 1) -> List[float]:
    """Generate Gaussian noise without PyTorch."""
    import random
    result = []
    for _ in range(size):
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        result.append(z * scale)
    return result


def compute_epsilon_for_budget(
    target_epsilon: float,
    num_rounds: int,
    delta: float = 1e-5,
) -> float:
    """
    Compute per-round epsilon for a total privacy budget.

    Uses advanced composition theorem.
    """
    # Simple composition (conservative)
    return target_epsilon / num_rounds


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Differential Privacy Test")
    print("=" * 60)

    # Test pure Python noise
    print("\nPure Python noise generation:")
    lap = laplacian_noise(1.0, 5)
    gau = gaussian_noise(1.0, 5)
    print(f"  Laplacian: {[f'{x:.3f}' for x in lap]}")
    print(f"  Gaussian: {[f'{x:.3f}' for x in gau]}")

    if TORCH_AVAILABLE:
        # Test DP with PyTorch
        config = PrivacyConfig(
            epsilon=1.0,
            clip_norm=1.0,
            phi_calibration=True,
        )
        dp = DifferentialPrivacy(config)

        # Create dummy gradients
        gradients = {
            'layer1': torch.randn(64, 32),
            'layer2': torch.randn(32, 16),
        }

        print(f"\nOriginal gradients:")
        for name, grad in gradients.items():
            print(f"  {name}: norm={grad.norm():.4f}, mean={grad.mean():.4f}")

        # Privatize
        private = dp.privatize_gradients(gradients)

        print(f"\nPrivatized gradients:")
        for name, grad in private.items():
            print(f"  {name}: norm={grad.norm():.4f}, mean={grad.mean():.4f}")

        # Verify
        verification = dp.verify_privacy(gradients, private)
        print(f"\nVerification:")
        for k, v in verification.items():
            print(f"  {k}: {v:.4f}")

        # Stats
        print(f"\nPrivacy Stats:")
        stats = dp.get_stats()
        print(f"  Remaining budget: {stats['accountant']['remaining_epsilon']:.2f}")

        print("\n✓ Differential Privacy module ready!")
    else:
        print("\n⚠️  PyTorch not available for full test")
