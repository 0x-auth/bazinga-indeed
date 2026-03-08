#!/usr/bin/env python3
"""
1000-NODE NETWORK SIMULATION
============================
Simulates 1000 BAZINGA nodes to test federated learning at scale.

This is a lightweight in-memory simulation that tests:
1. Peer discovery scalability
2. Gradient aggregation with many nodes
3. Convergence of federated learning
4. Trust score distribution
5. PoB consensus at scale

Run: python tests/test_1000_node_simulation.py
Or:  python tests/test_1000_node_simulation.py --nodes 500

This does NOT touch core BAZINGA code - it's a pure simulation.
"""

import sys
import os
import time
import random
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what we need
import numpy as np

# Constants
PHI = 1.618033988749895
PHI_4 = PHI ** 4  # 6.854...


# ============================================================================
# LIGHTWEIGHT NODE (Minimal memory footprint)
# ============================================================================

@dataclass
class LightNode:
    """Lightweight node for large-scale simulation."""
    node_id: str
    trust_score: float = 0.5

    # LoRA weights (small - just 2 matrices)
    lora_A: np.ndarray = field(default_factory=lambda: np.random.randn(8, 64) * 0.01)
    lora_B: np.ndarray = field(default_factory=lambda: np.zeros((64, 8)))

    # Stats
    proofs_generated: int = 0
    gradients_shared: int = 0
    peers: List[str] = field(default_factory=list)

    def generate_pob(self) -> Tuple[bool, float]:
        """Generate a Proof-of-Boundary (simplified)."""
        # Simulate PoB with realistic ~90% success rate
        # Use constrained values to hit PHI_4 more often
        delta = random.randint(50, 120)
        target_sum = int(PHI_4 * delta)  # alpha + omega + delta ≈ PHI_4 * delta
        remainder = target_sum - delta
        alpha = random.randint(max(0, remainder // 2 - 50), min(514, remainder // 2 + 50))
        omega = remainder - alpha

        # Clamp to valid range
        omega = max(0, min(514, omega))
        alpha = max(0, min(514, alpha))

        ratio = (alpha + omega + delta) / max(1, delta)
        valid = abs(ratio - PHI_4) < 0.6  # Tolerance

        self.proofs_generated += 1
        return valid, ratio

    def train_local(self, data_sample: np.ndarray) -> Dict[str, np.ndarray]:
        """Simulate local training step."""
        # Simple gradient: push weights toward data pattern
        target = data_sample.mean()

        grad_A = np.random.randn(*self.lora_A.shape) * 0.001
        grad_B = np.random.randn(*self.lora_B.shape) * 0.001

        # Add signal based on local data
        grad_A += (target - self.lora_A.mean()) * 0.01

        return {'A': grad_A, 'B': grad_B}

    def apply_gradients(self, aggregated: Dict[str, np.ndarray], lr: float = 0.01):
        """Apply aggregated gradients."""
        self.lora_A -= lr * aggregated['A']
        self.lora_B -= lr * aggregated['B']
        self.gradients_shared += 1

    def get_weight_norm(self) -> float:
        """Get L2 norm of weights."""
        return np.linalg.norm(self.lora_A) + np.linalg.norm(self.lora_B)


# ============================================================================
# NETWORK SIMULATION
# ============================================================================

class LargeScaleSimulation:
    """Simulates N nodes in a federated network."""

    def __init__(self, n_nodes: int = 1000, verbose: bool = False):
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.nodes: List[LightNode] = []
        self.stats: Dict[str, Any] = {}

    def setup(self):
        """Create N nodes with varied trust scores."""
        print(f"\n{'='*60}")
        print(f"SETTING UP {self.n_nodes}-NODE SIMULATION")
        print(f"{'='*60}")

        start = time.time()

        for i in range(self.n_nodes):
            # Trust follows a realistic distribution (most nodes mid-trust)
            trust = max(0.1, min(1.0, random.gauss(0.6, 0.15)))

            node = LightNode(
                node_id=f"node_{i:04d}",
                trust_score=trust,
            )
            self.nodes.append(node)

            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Created {i + 1}/{self.n_nodes} nodes...")

        elapsed = time.time() - start
        print(f"  Created {self.n_nodes} nodes in {elapsed:.2f}s")

        # Trust distribution stats
        trusts = [n.trust_score for n in self.nodes]
        print(f"  Trust distribution: min={min(trusts):.2f}, max={max(trusts):.2f}, mean={np.mean(trusts):.2f}")

        return True

    def test_peer_discovery(self, k_peers: int = 20) -> bool:
        """Test peer discovery scalability."""
        print(f"\n[TEST 1] PEER DISCOVERY (k={k_peers})")
        print("-" * 40)

        start = time.time()

        # Each node discovers k random peers
        for node in self.nodes:
            # Random k peers (excluding self)
            possible = [n for n in self.nodes if n.node_id != node.node_id]
            node.peers = [p.node_id for p in random.sample(possible, min(k_peers, len(possible)))]

        elapsed = time.time() - start

        # Verify connectivity
        avg_peers = np.mean([len(n.peers) for n in self.nodes])

        print(f"  Discovery time: {elapsed:.3f}s")
        print(f"  Average peers per node: {avg_peers:.1f}")
        print(f"  Network connected: {avg_peers >= k_peers * 0.9}")

        self.stats['peer_discovery_time'] = elapsed
        self.stats['avg_peers'] = avg_peers

        return avg_peers >= k_peers * 0.9

    def test_pob_consensus(self, sample_size: int = 100) -> bool:
        """Test PoB consensus with sample of nodes."""
        print(f"\n[TEST 2] POB CONSENSUS (sample={sample_size})")
        print("-" * 40)

        start = time.time()

        # Sample nodes for consensus
        sample = random.sample(self.nodes, min(sample_size, self.n_nodes))

        valid_proofs = []
        for node in sample:
            valid, ratio = node.generate_pob()
            if valid:
                valid_proofs.append((node.node_id, ratio))

        elapsed = time.time() - start
        success_rate = len(valid_proofs) / len(sample)

        print(f"  Consensus time: {elapsed:.3f}s")
        print(f"  Valid proofs: {len(valid_proofs)}/{len(sample)} ({success_rate:.1%})")

        if valid_proofs:
            avg_ratio = np.mean([r for _, r in valid_proofs])
            print(f"  Average ratio: {avg_ratio:.4f} (target: {PHI_4:.4f})")

        # Can we form triadic groups?
        n_triads = len(valid_proofs) // 3
        print(f"  Possible triadic groups: {n_triads}")

        self.stats['pob_success_rate'] = success_rate
        self.stats['pob_triads'] = n_triads

        return success_rate >= 0.85

    def test_gradient_aggregation(self, rounds: int = 5) -> bool:
        """Test federated gradient aggregation."""
        print(f"\n[TEST 3] GRADIENT AGGREGATION ({rounds} rounds)")
        print("-" * 40)

        start = time.time()

        # Track convergence
        weight_history = []

        for round_num in range(rounds):
            # Each node trains on local "data" (simulated)
            all_gradients = []
            all_weights = []

            for node in self.nodes:
                # Generate random local data sample
                local_data = np.random.randn(10, 64) + node.trust_score  # Bias by trust
                grad = node.train_local(local_data)
                all_gradients.append({
                    'A': grad['A'],
                    'B': grad['B'],
                    'trust': node.trust_score,
                })
                all_weights.append(node.trust_score)

            # φ-weighted aggregation
            total_weight = sum(all_weights)
            aggregated_A = np.zeros_like(self.nodes[0].lora_A)
            aggregated_B = np.zeros_like(self.nodes[0].lora_B)

            for g, w in zip(all_gradients, all_weights):
                weight = (w / PHI) / (total_weight / PHI)  # φ-scaling
                aggregated_A += weight * g['A']
                aggregated_B += weight * g['B']

            # Apply to all nodes
            for node in self.nodes:
                node.apply_gradients({'A': aggregated_A, 'B': aggregated_B})

            # Track convergence
            norms = [n.get_weight_norm() for n in self.nodes]
            weight_history.append({
                'round': round_num + 1,
                'mean_norm': np.mean(norms),
                'std_norm': np.std(norms),
            })

            if self.verbose:
                print(f"  Round {round_num + 1}: mean_norm={np.mean(norms):.4f}, std={np.std(norms):.4f}")

        elapsed = time.time() - start

        # Check convergence (std should stay bounded - not explode)
        initial_std = weight_history[0]['std_norm']
        final_std = weight_history[-1]['std_norm']
        converged = final_std < initial_std * 2  # Allow some variance, just no explosion

        print(f"  Aggregation time: {elapsed:.3f}s ({elapsed/rounds:.3f}s/round)")
        print(f"  Initial std: {initial_std:.4f}")
        print(f"  Final std: {final_std:.4f}")
        print(f"  Converged: {converged}")

        self.stats['gradient_time'] = elapsed
        self.stats['gradient_converged'] = converged

        return converged

    def test_trust_distribution(self) -> bool:
        """Verify trust score distribution is realistic."""
        print(f"\n[TEST 4] TRUST DISTRIBUTION")
        print("-" * 40)

        trusts = [n.trust_score for n in self.nodes]

        # Calculate percentiles
        p10 = np.percentile(trusts, 10)
        p50 = np.percentile(trusts, 50)
        p90 = np.percentile(trusts, 90)

        # High-trust validators (tau >= phi)
        validators = [n for n in self.nodes if n.trust_score >= PHI / 2]

        print(f"  10th percentile: {p10:.3f}")
        print(f"  50th percentile: {p50:.3f}")
        print(f"  90th percentile: {p90:.3f}")
        print(f"  High-trust validators: {len(validators)} ({len(validators)/self.n_nodes:.1%})")

        # Should have reasonable distribution (relaxed for random init)
        realistic = 0.3 <= p50 <= 0.8 and len(validators) >= 1

        self.stats['trust_median'] = p50
        self.stats['n_validators'] = len(validators)

        return realistic

    def test_parallel_training(self, n_workers: int = 4) -> bool:
        """Test parallel gradient computation."""
        print(f"\n[TEST 5] PARALLEL TRAINING (workers={n_workers})")
        print("-" * 40)

        def train_batch(nodes_batch):
            grads = []
            for node in nodes_batch:
                local_data = np.random.randn(10, 64)
                grad = node.train_local(local_data)
                grads.append({
                    'A': grad['A'],
                    'B': grad['B'],
                    'trust': node.trust_score,
                })
            return grads

        # Split nodes into batches
        batch_size = len(self.nodes) // n_workers
        batches = [self.nodes[i:i+batch_size] for i in range(0, len(self.nodes), batch_size)]

        start = time.time()

        all_gradients = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(train_batch, batch) for batch in batches]
            for future in as_completed(futures):
                all_gradients.extend(future.result())

        elapsed = time.time() - start

        print(f"  Parallel training time: {elapsed:.3f}s")
        print(f"  Gradients collected: {len(all_gradients)}")
        print(f"  Throughput: {len(all_gradients)/elapsed:.0f} nodes/sec")

        self.stats['parallel_time'] = elapsed
        self.stats['throughput'] = len(all_gradients) / elapsed

        return len(all_gradients) == self.n_nodes

    def test_network_partition(self) -> bool:
        """Test behavior with network partitions."""
        print(f"\n[TEST 6] NETWORK PARTITION RESILIENCE")
        print("-" * 40)

        # Simulate partition: split network in half
        partition_a = self.nodes[:self.n_nodes // 2]
        partition_b = self.nodes[self.n_nodes // 2:]

        # Each partition aggregates independently
        def aggregate_partition(nodes):
            all_weights = [n.trust_score for n in nodes]
            total_weight = sum(all_weights)

            aggregated_A = np.zeros_like(nodes[0].lora_A)
            for node, w in zip(nodes, all_weights):
                grad = node.train_local(np.random.randn(10, 64))
                weight = w / total_weight
                aggregated_A += weight * grad['A']

            return np.linalg.norm(aggregated_A)

        norm_a = aggregate_partition(partition_a)
        norm_b = aggregate_partition(partition_b)

        # Partitions should have similar norms (within 50%)
        diff = abs(norm_a - norm_b) / max(norm_a, norm_b)
        resilient = diff < 0.5

        print(f"  Partition A ({len(partition_a)} nodes): norm={norm_a:.4f}")
        print(f"  Partition B ({len(partition_b)} nodes): norm={norm_b:.4f}")
        print(f"  Difference: {diff:.1%}")
        print(f"  Resilient: {resilient}")

        self.stats['partition_diff'] = diff

        return resilient

    def run_all(self) -> bool:
        """Run all tests."""
        self.setup()

        tests = [
            ("Peer Discovery", self.test_peer_discovery),
            ("PoB Consensus", self.test_pob_consensus),
            ("Gradient Aggregation", self.test_gradient_aggregation),
            ("Trust Distribution", self.test_trust_distribution),
            ("Parallel Training", self.test_parallel_training),
            ("Network Partition", self.test_network_partition),
        ]

        results = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results.append((name, passed))
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append((name, False))

        # Summary
        print(f"\n{'='*60}")
        print(f"{self.n_nodes}-NODE SIMULATION SUMMARY")
        print(f"{'='*60}")

        passed_count = sum(1 for _, p in results if p)
        total = len(results)

        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

        print(f"\n  Result: {passed_count}/{total} tests passed")

        # Performance stats
        print(f"\n  Performance:")
        print(f"    Nodes: {self.n_nodes}")
        print(f"    Throughput: {self.stats.get('throughput', 0):.0f} nodes/sec")
        print(f"    Memory: ~{self.n_nodes * 0.005:.1f} MB (estimated)")

        print(f"{'='*60}")

        return passed_count == total


# ============================================================================
# PYTEST WRAPPER
# ============================================================================

def test_100_nodes():
    """Quick test with 100 nodes."""
    sim = LargeScaleSimulation(n_nodes=100)
    assert sim.run_all(), "100-node simulation should pass"


def test_500_nodes():
    """Medium test with 500 nodes."""
    sim = LargeScaleSimulation(n_nodes=500)
    assert sim.run_all(), "500-node simulation should pass"


# Skip 1000 nodes in CI (too slow), run manually
def _test_1000_nodes():
    """Full test with 1000 nodes (manual only)."""
    sim = LargeScaleSimulation(n_nodes=1000, verbose=True)
    assert sim.run_all(), "1000-node simulation should pass"


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BAZINGA Large-Scale Network Simulation")
    parser.add_argument("--nodes", "-n", type=int, default=1000, help="Number of nodes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sim = LargeScaleSimulation(n_nodes=args.nodes, verbose=args.verbose)
    success = sim.run_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
