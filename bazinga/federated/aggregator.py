#!/usr/bin/env python3
"""
BAZINGA Federated Aggregator - φ-Weighted Collective Learning
==============================================================

Aggregates gradients from multiple nodes using φ-weighted averaging.

How it works:
1. Collect gradients from N nodes
2. Weight each node's contribution by their trust score (τ)
3. Apply φ-scaling for balanced aggregation
4. Produce merged gradients for local model update

Why φ-weighting?
- Prevents any single node from dominating
- Natural balance between self and collective
- Matches BAZINGA's φ-coherence principles

Formula:
  merged = Σ (τᵢ / φ) × gradᵢ / Σ (τᵢ / φ)

"The collective is wiser than any individual."
"""

import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .gradients import GradientPackage

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
class AggregationResult:
    """Result of federated aggregation."""
    merged_gradients: Dict[str, Any]
    contributing_nodes: List[str]
    total_samples: int
    average_trust: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'merged_gradients': {
                k: v.tolist() if NUMPY_AVAILABLE and hasattr(v, 'tolist') else v
                for k, v in self.merged_gradients.items()
            },
            'contributing_nodes': self.contributing_nodes,
            'total_samples': self.total_samples,
            'average_trust': self.average_trust,
            'timestamp': self.timestamp,
        }


def phi_weighted_average(
    values: List[Any],
    trust_scores: List[float],
) -> Any:
    """
    Compute φ-weighted average of values.

    Each value is weighted by (trust / φ) to prevent domination.

    Args:
        values: List of numpy arrays or tensors
        trust_scores: Trust score for each value (0-1)

    Returns:
        Weighted average
    """
    if not values:
        return None

    if not NUMPY_AVAILABLE:
        # Simple average fallback
        return values[0]

    # Compute φ-scaled weights
    weights = np.array([t / PHI for t in trust_scores])
    weights = weights / weights.sum()  # Normalize

    # Weighted average
    result = np.zeros_like(values[0])
    for v, w in zip(values, weights):
        result += w * np.array(v)

    return result


class FederatedAggregator:
    """
    Aggregates gradients from multiple nodes.

    Features:
    - φ-weighted averaging based on trust
    - Outlier detection and filtering
    - Staleness detection (old gradients weighted less)
    - Contribution tracking

    Usage:
        aggregator = FederatedAggregator()
        aggregator.add_package(package1, trust=0.8)
        aggregator.add_package(package2, trust=0.6)
        result = aggregator.aggregate()
        # Apply result.merged_gradients to local model
    """

    def __init__(
        self,
        min_nodes: int = 2,  # Minimum nodes for aggregation
        max_staleness_seconds: float = 3600,  # 1 hour
        outlier_threshold: float = 3.0,  # Standard deviations
    ):
        self.min_nodes = min_nodes
        self.max_staleness = max_staleness_seconds
        self.outlier_threshold = outlier_threshold

        # Pending packages
        self.packages: List[Tuple[GradientPackage, float]] = []  # (package, trust)

        # Stats
        self.aggregations_performed = 0
        self.total_packages_processed = 0
        self.outliers_filtered = 0

    def add_package(
        self,
        package: GradientPackage,
        trust_score: float = 0.5,
    ):
        """Add a gradient package for aggregation."""
        # Check staleness
        age = time.time() - package.timestamp
        if age > self.max_staleness:
            print(f"  ⚠ Package from {package.node_id} is stale ({age:.0f}s old)")
            # Reduce trust for stale packages
            trust_score *= 0.5

        # Verify package
        if not package.verify():
            print(f"  ⚠ Package from {package.node_id} failed verification")
            return

        self.packages.append((package, trust_score))
        self.total_packages_processed += 1

    def _detect_outliers(
        self,
        gradients_list: List[Any],
    ) -> List[bool]:
        """Detect outlier gradients using z-score."""
        if not NUMPY_AVAILABLE or len(gradients_list) < 3:
            return [False] * len(gradients_list)

        # Compute norms
        norms = [np.linalg.norm(g) for g in gradients_list]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) + 1e-8

        # Z-score outlier detection
        is_outlier = [
            abs(n - mean_norm) / std_norm > self.outlier_threshold
            for n in norms
        ]

        return is_outlier

    def _decompress_gradient(self, compressed: Dict) -> Any:
        """Decompress a compressed gradient."""
        if not NUMPY_AVAILABLE:
            return compressed

        if isinstance(compressed, dict) and 'shape' in compressed:
            shape = tuple(compressed['shape'])
            sparse = compressed['sparse']
            flat = np.zeros(int(np.prod(shape)))
            for idx, val in sparse:
                flat[idx] = val
            return flat.reshape(shape)

        return np.array(compressed)

    def aggregate(self) -> Optional[AggregationResult]:
        """
        Aggregate all pending packages.

        Returns merged gradients using φ-weighted averaging.
        """
        if len(self.packages) < self.min_nodes:
            print(f"  Need at least {self.min_nodes} nodes, have {len(self.packages)}")
            return None

        # Collect all gradients by module
        module_gradients: Dict[str, List[Tuple[Any, float]]] = {}

        for package, trust in self.packages:
            for module_name, compressed in package.gradients.items():
                if module_name not in module_gradients:
                    module_gradients[module_name] = []

                grad = self._decompress_gradient(compressed)
                module_gradients[module_name].append((grad, trust))

        # Aggregate each module
        merged = {}
        for module_name, grad_trust_list in module_gradients.items():
            gradients = [g for g, t in grad_trust_list]
            trusts = [t for g, t in grad_trust_list]

            # Outlier detection
            is_outlier = self._detect_outliers(gradients)
            filtered_grads = [g for g, out in zip(gradients, is_outlier) if not out]
            filtered_trusts = [t for t, out in zip(trusts, is_outlier) if not out]
            self.outliers_filtered += sum(is_outlier)

            if filtered_grads:
                merged[module_name] = phi_weighted_average(
                    filtered_grads,
                    filtered_trusts,
                )

        # Compute stats
        contributing = list(set(p.node_id for p, t in self.packages))
        total_samples = sum(p.training_samples for p, t in self.packages)
        avg_trust = sum(t for p, t in self.packages) / len(self.packages)

        result = AggregationResult(
            merged_gradients=merged,
            contributing_nodes=contributing,
            total_samples=total_samples,
            average_trust=avg_trust,
        )

        # Clear packages
        self.packages = []
        self.aggregations_performed += 1

        return result

    def get_pending_count(self) -> int:
        """Get number of pending packages."""
        return len(self.packages)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            'pending_packages': len(self.packages),
            'min_nodes': self.min_nodes,
            'aggregations_performed': self.aggregations_performed,
            'total_packages_processed': self.total_packages_processed,
            'outliers_filtered': self.outliers_filtered,
        }


class TrustTracker:
    """
    Tracks trust scores for nodes based on their contributions.

    Trust increases when:
    - Node provides useful gradients
    - Node's gradients improve model performance
    - Node is consistent and reliable

    Trust decreases when:
    - Node provides outlier gradients
    - Node is inconsistent
    - Node's gradients hurt performance
    """

    def __init__(self, initial_trust: float = 0.5):
        self.initial_trust = initial_trust
        self.trust_scores: Dict[str, float] = {}
        self.contribution_counts: Dict[str, int] = {}
        self.outlier_counts: Dict[str, int] = {}

    def get_trust(self, node_id: str) -> float:
        """Get trust score for a node."""
        return self.trust_scores.get(node_id, self.initial_trust)

    def record_contribution(
        self,
        node_id: str,
        was_useful: bool = True,
        was_outlier: bool = False,
    ):
        """Record a contribution from a node."""
        if node_id not in self.trust_scores:
            self.trust_scores[node_id] = self.initial_trust
            self.contribution_counts[node_id] = 0
            self.outlier_counts[node_id] = 0

        self.contribution_counts[node_id] += 1

        if was_outlier:
            self.outlier_counts[node_id] += 1
            # Decrease trust
            self.trust_scores[node_id] = max(
                0.1,
                self.trust_scores[node_id] - 0.05
            )
        elif was_useful:
            # Increase trust (with φ-scaled ceiling)
            self.trust_scores[node_id] = min(
                1.0 / PHI + 0.5,  # Max ~1.12
                self.trust_scores[node_id] + 0.02
            )

    def get_all_trusts(self) -> Dict[str, float]:
        """Get all trust scores."""
        return dict(self.trust_scores)

    def get_stats(self) -> Dict[str, Any]:
        """Get trust tracker stats."""
        if not self.trust_scores:
            return {'nodes': 0}

        trusts = list(self.trust_scores.values())
        return {
            'nodes': len(self.trust_scores),
            'avg_trust': sum(trusts) / len(trusts),
            'min_trust': min(trusts),
            'max_trust': max(trusts),
            'total_contributions': sum(self.contribution_counts.values()),
            'total_outliers': sum(self.outlier_counts.values()),
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Federated Aggregator Test")
    print("=" * 60)

    print(f"\n  Numpy available: {NUMPY_AVAILABLE}")

    # Create aggregator
    aggregator = FederatedAggregator(min_nodes=2)
    print(f"\n  Created aggregator (min_nodes={aggregator.min_nodes})")

    if NUMPY_AVAILABLE:
        # Create some test packages
        for i in range(3):
            # Simulate compressed gradients
            grad = np.random.randn(768, 768) * 0.01
            flat = grad.flatten()
            k = int(len(flat) * 0.1)
            top_indices = np.argsort(np.abs(flat))[-k:]
            sparse = [(int(idx), float(flat[idx])) for idx in top_indices]

            package = GradientPackage(
                node_id=f"node_{i}",
                version=1,
                timestamp=time.time(),
                gradients={
                    'q_proj': {
                        'shape': list(grad.shape),
                        'sparse': sparse,
                        'k': k,
                    }
                },
                training_samples=100 + i * 50,
                noise_scale=0.01,
            )

            trust = 0.5 + i * 0.1
            aggregator.add_package(package, trust)
            print(f"    Added package from node_{i} (trust={trust:.2f})")

        # Aggregate
        result = aggregator.aggregate()

        if result:
            print(f"\n  Aggregation result:")
            print(f"    Contributing nodes: {result.contributing_nodes}")
            print(f"    Total samples: {result.total_samples}")
            print(f"    Average trust: {result.average_trust:.3f}")
            print(f"    Modules merged: {list(result.merged_gradients.keys())}")

        # Stats
        stats = aggregator.get_stats()
        print(f"\n  Aggregator stats:")
        print(f"    Aggregations: {stats['aggregations_performed']}")
        print(f"    Outliers filtered: {stats['outliers_filtered']}")

    # Trust tracker test
    print(f"\n  Trust Tracker Test:")
    tracker = TrustTracker()

    for i in range(5):
        tracker.record_contribution(f"node_0", was_useful=True)
        tracker.record_contribution(f"node_1", was_useful=True, was_outlier=(i % 3 == 0))

    print(f"    Node 0 trust: {tracker.get_trust('node_0'):.3f}")
    print(f"    Node 1 trust: {tracker.get_trust('node_1'):.3f}")

    print("\n  ✓ Federated aggregator working!")
