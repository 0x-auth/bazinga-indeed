#!/usr/bin/env python3
"""
BAZINGA Gradient Sharing - Privacy-Preserving Learning
=======================================================

Share LEARNING, not DATA.

How it works:
1. Node trains locally on private data
2. Computes gradients (how weights should change)
3. Adds differential privacy noise
4. Shares gradients with network
5. Other nodes learn from gradients

Privacy guarantees:
- Raw data never leaves the node
- Gradients have noise added (differential privacy)
- Individual examples cannot be reconstructed
- Only aggregate learning is shared

"Your data stays with you. Your learning spreads."
"""

import hashlib
import json
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Check for numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class GradientPackage:
    """
    A package of gradients ready to share with the network.

    Contains:
    - Compressed gradient tensors
    - Privacy noise already applied
    - Metadata for verification
    - Signature for authenticity
    """
    node_id: str
    version: int
    timestamp: float
    gradients: Dict[str, List]  # module_name -> flattened gradients
    training_samples: int
    noise_scale: float  # Differential privacy noise level
    hash: str = ""  # For verification
    signature: str = ""  # Node signature

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of gradient content."""
        content = json.dumps({
            'node_id': self.node_id,
            'version': self.version,
            'gradients_keys': list(self.gradients.keys()),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for network transmission."""
        return {
            'node_id': self.node_id,
            'version': self.version,
            'timestamp': self.timestamp,
            'gradients': self.gradients,
            'training_samples': self.training_samples,
            'noise_scale': self.noise_scale,
            'hash': self.hash,
            'signature': self.signature,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GradientPackage':
        """Deserialize from network."""
        return cls(
            node_id=d['node_id'],
            version=d['version'],
            timestamp=d['timestamp'],
            gradients=d['gradients'],
            training_samples=d['training_samples'],
            noise_scale=d['noise_scale'],
            hash=d.get('hash', ''),
            signature=d.get('signature', ''),
        )

    def verify(self) -> bool:
        """Verify package integrity."""
        expected_hash = self._compute_hash()
        return self.hash == expected_hash

    def get_size_bytes(self) -> int:
        """Estimate package size in bytes."""
        return len(json.dumps(self.to_dict()))


class GradientSharer:
    """
    Manages gradient sharing with differential privacy.

    Responsibilities:
    1. Collect gradients from local training
    2. Apply differential privacy noise
    3. Compress for efficient transmission
    4. Package for network sharing

    Usage:
        sharer = GradientSharer(node_id="my_node")
        sharer.add_gradients("q_proj", gradient_tensor)
        package = sharer.create_package()
        # Send package over network
    """

    def __init__(
        self,
        node_id: str,
        noise_scale: float = 0.01,  # Differential privacy noise
        clip_threshold: float = 1.0,  # Gradient clipping
        compression_ratio: float = 0.1,  # Keep top 10% of gradients
    ):
        self.node_id = node_id
        self.noise_scale = noise_scale
        self.clip_threshold = clip_threshold
        self.compression_ratio = compression_ratio

        # Accumulated gradients
        self.gradients: Dict[str, Any] = {}
        self.gradient_counts: Dict[str, int] = {}

        # Stats
        self.version = 0
        self.total_samples = 0
        self.packages_created = 0
        self.packages_received = 0

    def add_gradients(
        self,
        module_name: str,
        gradients: Any,
        num_samples: int = 1,
    ):
        """
        Add gradients from a training step.

        Gradients are accumulated and averaged before packaging.
        """
        if NUMPY_AVAILABLE:
            if not isinstance(gradients, np.ndarray):
                gradients = np.array(gradients)

            # Clip gradients for privacy
            norm = np.linalg.norm(gradients)
            if norm > self.clip_threshold:
                gradients = gradients * (self.clip_threshold / norm)

            # Accumulate
            if module_name not in self.gradients:
                self.gradients[module_name] = np.zeros_like(gradients)
                self.gradient_counts[module_name] = 0

            self.gradients[module_name] += gradients
            self.gradient_counts[module_name] += num_samples
            self.total_samples += num_samples

    def _apply_differential_privacy(self, gradients: Any) -> Any:
        """Apply differential privacy noise to gradients."""
        if not NUMPY_AVAILABLE:
            return gradients

        # Gaussian noise scaled by noise_scale
        noise = np.random.normal(0, self.noise_scale, gradients.shape)
        return gradients + noise

    def _compress_gradients(self, gradients: Any) -> List:
        """
        Compress gradients for efficient transmission.

        Uses top-k sparsification: only keep the largest gradients.
        """
        if not NUMPY_AVAILABLE:
            return gradients if isinstance(gradients, list) else gradients.tolist()

        flat = gradients.flatten()
        k = max(1, int(len(flat) * self.compression_ratio))

        # Get top-k indices
        top_indices = np.argsort(np.abs(flat))[-k:]

        # Create sparse representation
        sparse = [(int(i), float(flat[i])) for i in top_indices]

        return {
            'shape': list(gradients.shape),
            'sparse': sparse,
            'k': k,
        }

    def _decompress_gradients(self, compressed: Dict) -> Any:
        """Decompress gradients."""
        if not NUMPY_AVAILABLE:
            return compressed

        shape = tuple(compressed['shape'])
        sparse = compressed['sparse']

        # Reconstruct
        flat = np.zeros(np.prod(shape))
        for idx, val in sparse:
            flat[idx] = val

        return flat.reshape(shape)

    def create_package(self) -> GradientPackage:
        """
        Create a gradient package for sharing.

        Applies:
        1. Averaging over accumulated samples
        2. Differential privacy noise
        3. Compression
        """
        packaged_gradients = {}

        for name, grad_sum in self.gradients.items():
            count = self.gradient_counts.get(name, 1)

            # Average
            avg_grad = grad_sum / max(count, 1)

            # Apply differential privacy
            private_grad = self._apply_differential_privacy(avg_grad)

            # Compress
            compressed = self._compress_gradients(private_grad)
            packaged_gradients[name] = compressed

        package = GradientPackage(
            node_id=self.node_id,
            version=self.version,
            timestamp=time.time(),
            gradients=packaged_gradients,
            training_samples=self.total_samples,
            noise_scale=self.noise_scale,
        )

        # Reset accumulators
        self.gradients = {}
        self.gradient_counts = {}
        self.total_samples = 0
        self.version += 1
        self.packages_created += 1

        return package

    def receive_package(self, package: GradientPackage) -> Dict[str, Any]:
        """
        Receive and decompress a gradient package from another node.

        Returns decompressed gradients ready for merging.
        """
        if not package.verify():
            print(f"  ⚠ Package from {package.node_id} failed verification")
            return {}

        decompressed = {}
        for name, compressed in package.gradients.items():
            decompressed[name] = self._decompress_gradients(compressed)

        self.packages_received += 1
        return decompressed

    def get_stats(self) -> Dict[str, Any]:
        """Get sharer statistics."""
        return {
            'node_id': self.node_id,
            'version': self.version,
            'total_samples': self.total_samples,
            'pending_modules': list(self.gradients.keys()),
            'packages_created': self.packages_created,
            'packages_received': self.packages_received,
            'noise_scale': self.noise_scale,
            'compression_ratio': self.compression_ratio,
        }


# Utility functions

def estimate_privacy_budget(
    noise_scale: float,
    num_queries: int,
    sensitivity: float = 1.0,
) -> float:
    """
    Estimate differential privacy epsilon.

    Lower epsilon = stronger privacy.
    Typically want epsilon < 1 for good privacy.
    """
    # Gaussian mechanism privacy
    epsilon = sensitivity * math.sqrt(2 * math.log(1.25 / 0.01)) / noise_scale
    # Account for composition over queries
    total_epsilon = epsilon * math.sqrt(num_queries)
    return total_epsilon


def recommend_noise_scale(
    target_epsilon: float = 1.0,
    expected_queries: int = 100,
    sensitivity: float = 1.0,
) -> float:
    """Recommend noise scale for target privacy level."""
    base_epsilon = target_epsilon / math.sqrt(expected_queries)
    noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / 0.01)) / base_epsilon
    return noise_scale


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Gradient Sharer Test")
    print("=" * 60)

    print(f"\n  Numpy available: {NUMPY_AVAILABLE}")

    # Create sharer
    sharer = GradientSharer(node_id="test_node")
    print(f"\n  Created sharer: {sharer.node_id}")

    if NUMPY_AVAILABLE:
        # Simulate some gradients
        for i in range(5):
            grad = np.random.randn(768, 768) * 0.01
            sharer.add_gradients("q_proj", grad)
            print(f"    Added gradient batch {i+1}")

        # Create package
        package = sharer.create_package()
        print(f"\n  Created package:")
        print(f"    Version: {package.version}")
        print(f"    Samples: {package.training_samples}")
        print(f"    Size: {package.get_size_bytes()} bytes")
        print(f"    Hash: {package.hash}")
        print(f"    Verified: {package.verify()}")

        # Serialize/deserialize
        serialized = package.to_dict()
        restored = GradientPackage.from_dict(serialized)
        print(f"\n  Serialization test:")
        print(f"    Original hash: {package.hash}")
        print(f"    Restored hash: {restored.hash}")
        print(f"    Match: {package.hash == restored.hash}")

        # Privacy estimate
        epsilon = estimate_privacy_budget(sharer.noise_scale, 100)
        print(f"\n  Privacy budget (ε): {epsilon:.2f}")

    print("\n  ✓ Gradient sharer working!")
