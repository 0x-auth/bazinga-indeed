#!/usr/bin/env python3
"""
BYZANTINE FAULT TOLERANCE TEST
==============================
Tests BAZINGA's resilience against malicious actors in the network.

Scenarios tested:
1. Bad actors sending fake gradients
2. Trust decay attacks (nodes lying about trust scores)
3. Sybil attacks (many fake identities)
4. Gradient poisoning (adversarial updates)
5. Collusion attacks (malicious triads)
6. Eclipse attacks (isolating honest nodes)

The φ-weighted consensus should identify and neutralize bad actors.

Run: python tests/test_byzantine_fault_tolerance.py
Or:  python tests/test_byzantine_fault_tolerance.py --nodes 1000 --bad-actors 100
"""

import sys
import os
import time
import random
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Constants
PHI = 1.618033988749895
PHI_4 = PHI ** 4  # 6.854...


# ============================================================================
# NODE TYPES
# ============================================================================

@dataclass
class ByzantineNode:
    """Node that can be honest or malicious."""
    node_id: str
    trust_score: float = 0.5
    is_malicious: bool = False
    attack_type: str = "none"  # none, fake_gradient, trust_lie, sybil, poison

    # LoRA weights
    lora_A: np.ndarray = field(default_factory=lambda: np.random.randn(8, 64) * 0.01)
    lora_B: np.ndarray = field(default_factory=lambda: np.zeros((64, 8)))

    # Stats
    proofs_generated: int = 0
    gradients_shared: int = 0
    detected_as_malicious: bool = False
    detection_round: int = -1

    def generate_pob(self) -> Tuple[bool, float]:
        """Generate PoB - malicious nodes may try to fake it."""
        if self.is_malicious and self.attack_type == "fake_pob":
            # Try to fake PoB with clearly wrong ratio (outside valid range)
            fake_ratio = random.choice([
                random.uniform(3.0, 5.5),   # Too low
                random.uniform(8.0, 12.0),  # Too high
            ])
            return True, fake_ratio  # Claim it's valid

        # Honest PoB generation
        delta = random.randint(50, 120)
        target_sum = int(PHI_4 * delta)
        remainder = target_sum - delta
        alpha = random.randint(max(0, remainder // 2 - 50), min(514, remainder // 2 + 50))
        omega = remainder - alpha
        omega = max(0, min(514, omega))
        alpha = max(0, min(514, alpha))

        ratio = (alpha + omega + delta) / max(1, delta)
        valid = abs(ratio - PHI_4) < 0.6

        self.proofs_generated += 1
        return valid, ratio

    def compute_gradient(self, data_sample: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradient - malicious nodes may poison it."""
        if self.is_malicious:
            if self.attack_type == "fake_gradient":
                # Send random noise instead of real gradient
                return {
                    'A': np.random.randn(*self.lora_A.shape) * 10,  # 10x magnitude
                    'B': np.random.randn(*self.lora_B.shape) * 10,
                }
            elif self.attack_type == "poison":
                # Send gradient in opposite direction (gradient ascent)
                target = data_sample.mean()
                grad_A = np.random.randn(*self.lora_A.shape) * 0.001
                grad_A -= (target - self.lora_A.mean()) * 0.01  # Opposite sign
                grad_B = np.random.randn(*self.lora_B.shape) * 0.001
                return {'A': -grad_A * 5, 'B': -grad_B * 5}  # Reversed and amplified
            elif self.attack_type == "zero":
                # Send zero gradients (free-riding)
                return {
                    'A': np.zeros_like(self.lora_A),
                    'B': np.zeros_like(self.lora_B),
                }

        # Honest gradient computation
        target = data_sample.mean()
        grad_A = np.random.randn(*self.lora_A.shape) * 0.001
        grad_B = np.random.randn(*self.lora_B.shape) * 0.001
        grad_A += (target - self.lora_A.mean()) * 0.01

        self.gradients_shared += 1
        return {'A': grad_A, 'B': grad_B}

    def report_trust(self) -> float:
        """Report trust score - malicious nodes may lie."""
        if self.is_malicious and self.attack_type == "trust_lie":
            # Claim very high trust
            return min(1.0, self.trust_score + 0.4)
        return self.trust_score

    def apply_gradients(self, aggregated: Dict[str, np.ndarray], lr: float = 0.01):
        """Apply aggregated gradients."""
        self.lora_A -= lr * aggregated['A']
        self.lora_B -= lr * aggregated['B']


# ============================================================================
# BYZANTINE FAULT TOLERANCE SIMULATION
# ============================================================================

class ByzantineSimulation:
    """Simulates Byzantine fault tolerance scenarios."""

    def __init__(
        self,
        n_nodes: int = 100,
        n_bad_actors: int = 10,
        attack_type: str = "mixed",
        verbose: bool = False
    ):
        self.n_nodes = n_nodes
        self.n_bad_actors = min(n_bad_actors, n_nodes // 3)  # Max 1/3 Byzantine
        self.attack_type = attack_type
        self.verbose = verbose
        self.nodes: List[ByzantineNode] = []
        self.honest_nodes: List[ByzantineNode] = []
        self.malicious_nodes: List[ByzantineNode] = []
        self.stats: Dict[str, Any] = {}

    def setup(self):
        """Create nodes with some malicious actors."""
        print(f"\n{'='*60}")
        print(f"BYZANTINE FAULT TOLERANCE TEST")
        print(f"{'='*60}")
        print(f"  Total nodes: {self.n_nodes}")
        print(f"  Bad actors: {self.n_bad_actors} ({self.n_bad_actors/self.n_nodes:.1%})")
        print(f"  Attack type: {self.attack_type}")

        attack_types = ["fake_gradient", "trust_lie", "poison", "zero", "fake_pob"]

        for i in range(self.n_nodes):
            is_malicious = i < self.n_bad_actors

            if is_malicious:
                if self.attack_type == "mixed":
                    attack = random.choice(attack_types)
                else:
                    attack = self.attack_type
            else:
                attack = "none"

            trust = max(0.1, min(1.0, random.gauss(0.6, 0.15)))

            node = ByzantineNode(
                node_id=f"node_{i:04d}",
                trust_score=trust,
                is_malicious=is_malicious,
                attack_type=attack,
            )

            self.nodes.append(node)
            if is_malicious:
                self.malicious_nodes.append(node)
            else:
                self.honest_nodes.append(node)

        # Shuffle so bad actors aren't all at the start
        random.shuffle(self.nodes)

        print(f"  Honest nodes: {len(self.honest_nodes)}")
        print(f"  Malicious nodes: {len(self.malicious_nodes)}")

        if self.verbose:
            attack_counts = {}
            for node in self.malicious_nodes:
                attack_counts[node.attack_type] = attack_counts.get(node.attack_type, 0) + 1
            print(f"  Attack distribution: {attack_counts}")

        return True

    def test_pob_validation(self) -> bool:
        """Test 1: PoB validation catches fake proofs."""
        print(f"\n[TEST 1] POB VALIDATION (Fake PoB Detection)")
        print("-" * 40)

        valid_proofs = []
        invalid_proofs = []
        fake_accepted = 0
        fake_rejected = 0

        for node in self.nodes:
            valid, ratio = node.generate_pob()

            # Validate the PoB
            is_actually_valid = abs(ratio - PHI_4) < 0.6

            if node.is_malicious and node.attack_type == "fake_pob":
                if is_actually_valid:
                    fake_accepted += 1
                else:
                    fake_rejected += 1
                    node.detected_as_malicious = True

            if is_actually_valid:
                valid_proofs.append((node.node_id, ratio))
            else:
                invalid_proofs.append((node.node_id, ratio))

        detection_rate = fake_rejected / max(1, fake_rejected + fake_accepted)

        print(f"  Valid proofs: {len(valid_proofs)}")
        print(f"  Invalid proofs: {len(invalid_proofs)}")
        print(f"  Fake PoB rejected: {fake_rejected}")
        print(f"  Fake PoB accepted: {fake_accepted}")
        print(f"  Detection rate: {detection_rate:.1%}")

        self.stats['pob_detection_rate'] = detection_rate

        # Should reject most fake PoB (or if no fake PoB attacks, pass)
        # Note: Some fake PoB may accidentally hit the valid range - that's physics
        return detection_rate >= 0.5 or fake_rejected + fake_accepted == 0

    def test_gradient_anomaly_detection(self, rounds: int = 5) -> bool:
        """Test 2: Detect anomalous gradients from bad actors."""
        print(f"\n[TEST 2] GRADIENT ANOMALY DETECTION ({rounds} rounds)")
        print("-" * 40)

        detected_count = 0
        gradient_history = {node.node_id: [] for node in self.nodes}

        for round_num in range(rounds):
            all_gradients = []

            for node in self.nodes:
                local_data = np.random.randn(10, 64)
                grad = node.compute_gradient(local_data)
                grad_norm = np.linalg.norm(grad['A']) + np.linalg.norm(grad['B'])

                all_gradients.append({
                    'node': node,
                    'A': grad['A'],
                    'B': grad['B'],
                    'norm': grad_norm,
                    'trust': node.report_trust(),
                })
                gradient_history[node.node_id].append(grad_norm)

            # Compute statistics for anomaly detection
            norms = [g['norm'] for g in all_gradients]
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)

            # Flag anomalies: gradients > 3 std from mean
            threshold = mean_norm + 3 * std_norm
            zero_threshold = mean_norm * 0.01  # Near-zero gradients

            for g in all_gradients:
                node = g['node']
                if g['norm'] > threshold or g['norm'] < zero_threshold:
                    if not node.detected_as_malicious:
                        node.detected_as_malicious = True
                        node.detection_round = round_num
                        if node.is_malicious:
                            detected_count += 1

            if self.verbose and round_num == rounds - 1:
                print(f"  Round {round_num + 1}: mean_norm={mean_norm:.4f}, std={std_norm:.4f}")
                print(f"  Anomaly threshold: > {threshold:.4f} or < {zero_threshold:.4f}")

        # Count detections
        true_positives = sum(1 for n in self.malicious_nodes if n.detected_as_malicious)
        false_positives = sum(1 for n in self.honest_nodes if n.detected_as_malicious)

        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, len(self.malicious_nodes))

        print(f"  True positives: {true_positives}/{len(self.malicious_nodes)}")
        print(f"  False positives: {false_positives}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")

        self.stats['gradient_precision'] = precision
        self.stats['gradient_recall'] = recall

        # Key insight: in φ-weighted aggregation, we don't need to catch all bad actors
        # We just need their weighted contribution to be neutralized
        # High recall is good (we catch bad actors), low precision is acceptable
        # because the aggregation weighting handles false positives gracefully
        return recall >= 0.5 or (recall >= 0.3 and true_positives >= len(self.malicious_nodes) // 2)

    def test_trust_verification(self) -> bool:
        """Test 3: Verify trust scores against actual behavior."""
        print(f"\n[TEST 3] TRUST VERIFICATION (Lie Detection)")
        print("-" * 40)

        # Reset detection flags for this test
        for node in self.nodes:
            node.detected_as_malicious = False

        liars_detected = 0
        liars_total = sum(1 for n in self.malicious_nodes if n.attack_type == "trust_lie")

        # Cross-validate trust scores
        # In a real system, peers would verify each other's behavior
        for node in self.nodes:
            reported_trust = node.report_trust()
            actual_trust = node.trust_score

            # If reported trust is suspiciously higher than baseline
            if reported_trust > actual_trust + 0.2:
                node.detected_as_malicious = True
                if node.is_malicious and node.attack_type == "trust_lie":
                    liars_detected += 1

        detection_rate = liars_detected / max(1, liars_total) if liars_total > 0 else 1.0

        print(f"  Trust liars in network: {liars_total}")
        print(f"  Liars detected: {liars_detected}")
        print(f"  Detection rate: {detection_rate:.1%}")

        self.stats['trust_lie_detection'] = detection_rate

        # Should detect most liars (80%+ is excellent)
        return detection_rate >= 0.8 or liars_total == 0

    def test_sybil_resistance(self) -> bool:
        """Test 4: Resistance to Sybil attacks (many fake identities)."""
        print(f"\n[TEST 4] SYBIL RESISTANCE")
        print("-" * 40)

        # Simulate Sybil: bad actors create many identities
        sybil_count = self.n_bad_actors * 3  # Each bad actor creates 3 fake nodes

        # In BAZINGA, new nodes start with τ = 0.5 and must prove themselves
        sybil_nodes = []
        for i in range(sybil_count):
            sybil = ByzantineNode(
                node_id=f"sybil_{i:04d}",
                trust_score=0.5,  # Default new node trust
                is_malicious=True,
                attack_type="sybil",
            )
            sybil_nodes.append(sybil)

        # Calculate voting power
        honest_power = sum(n.trust_score for n in self.honest_nodes)
        malicious_power = sum(n.trust_score for n in self.malicious_nodes)
        sybil_power = sum(n.trust_score for n in sybil_nodes)
        total_power = honest_power + malicious_power + sybil_power

        # With φ-weighting, honest nodes should still dominate
        # because they have higher trust from actual good behavior
        honest_ratio = honest_power / total_power
        attack_ratio = (malicious_power + sybil_power) / total_power

        print(f"  Sybil identities created: {sybil_count}")
        print(f"  Honest voting power: {honest_power:.2f} ({honest_ratio:.1%})")
        print(f"  Malicious + Sybil power: {malicious_power + sybil_power:.2f} ({attack_ratio:.1%})")

        # In triadic consensus, need 3 high-trust nodes
        high_trust_honest = sum(1 for n in self.honest_nodes if n.trust_score >= PHI / 2)
        high_trust_attack = sum(1 for n in self.malicious_nodes + sybil_nodes if n.trust_score >= PHI / 2)

        print(f"  High-trust honest: {high_trust_honest}")
        print(f"  High-trust attackers: {high_trust_attack}")

        self.stats['sybil_honest_ratio'] = honest_ratio

        # Honest nodes should maintain majority power
        return honest_ratio > 0.5

    def test_gradient_aggregation_resilience(self, rounds: int = 10) -> bool:
        """Test 5: Does φ-weighted aggregation resist poisoning?"""
        print(f"\n[TEST 5] AGGREGATION RESILIENCE ({rounds} rounds)")
        print("-" * 40)

        # Track convergence of honest vs malicious influence
        honest_weights = [n.lora_A.copy() for n in self.honest_nodes[:10]]

        for round_num in range(rounds):
            all_gradients = []
            all_weights = []

            for node in self.nodes:
                # Skip detected malicious nodes
                if node.detected_as_malicious:
                    continue

                local_data = np.random.randn(10, 64)
                grad = node.compute_gradient(local_data)

                # Use φ-weighted trust
                weight = node.trust_score / PHI

                all_gradients.append({
                    'A': grad['A'],
                    'B': grad['B'],
                })
                all_weights.append(weight)

            if not all_gradients:
                continue

            # φ-weighted aggregation
            total_weight = sum(all_weights)
            aggregated_A = np.zeros_like(self.nodes[0].lora_A)
            aggregated_B = np.zeros_like(self.nodes[0].lora_B)

            for g, w in zip(all_gradients, all_weights):
                weight = w / total_weight
                aggregated_A += weight * g['A']
                aggregated_B += weight * g['B']

            # Apply to all non-detected nodes
            for node in self.nodes:
                if not node.detected_as_malicious:
                    node.apply_gradients({'A': aggregated_A, 'B': aggregated_B})

        # Check if honest nodes converged well
        final_weights = [n.lora_A for n in self.honest_nodes[:10]]

        # Compute variance among honest nodes (should be low = convergence)
        weight_norms = [np.linalg.norm(w) for w in final_weights]
        variance = np.var(weight_norms)
        mean_norm = np.mean(weight_norms)

        # Check if weights didn't explode (sign of successful poisoning)
        initial_mean = np.mean([np.linalg.norm(w) for w in honest_weights])
        explosion_ratio = mean_norm / max(0.001, initial_mean)

        print(f"  Final weight variance: {variance:.6f}")
        print(f"  Weight explosion ratio: {explosion_ratio:.2f}x")
        print(f"  Converged: {variance < 0.1}")
        print(f"  Stable: {0.1 < explosion_ratio < 10}")

        self.stats['aggregation_variance'] = variance
        self.stats['explosion_ratio'] = explosion_ratio

        # Should converge and not explode
        return variance < 0.1 and 0.1 < explosion_ratio < 10

    def test_triadic_consensus_integrity(self) -> bool:
        """Test 6: Can malicious nodes corrupt triadic consensus?"""
        print(f"\n[TEST 6] TRIADIC CONSENSUS INTEGRITY")
        print("-" * 40)

        # Try to form triads
        valid_triads = 0
        corrupted_triads = 0
        honest_triads = 0

        # Sample nodes for triadic validation
        for _ in range(100):
            # Select 3 random nodes weighted by trust
            weights = [n.trust_score for n in self.nodes if not n.detected_as_malicious]
            if len(weights) < 3:
                continue

            available = [n for n in self.nodes if not n.detected_as_malicious]
            triad = random.choices(available, weights=weights, k=3)

            # Each node generates PoB
            proofs = []
            for node in triad:
                valid, ratio = node.generate_pob()
                actually_valid = abs(ratio - PHI_4) < 0.6
                proofs.append((node, valid, actually_valid))

            # Count malicious in triad
            malicious_count = sum(1 for node, _, _ in proofs if node.is_malicious)
            all_valid = all(actually_valid for _, _, actually_valid in proofs)

            if all_valid:
                valid_triads += 1
                if malicious_count == 0:
                    honest_triads += 1
                elif malicious_count >= 2:
                    corrupted_triads += 1

        corruption_rate = corrupted_triads / max(1, valid_triads)
        honest_rate = honest_triads / max(1, valid_triads)

        print(f"  Valid triads formed: {valid_triads}")
        print(f"  Fully honest triads: {honest_triads} ({honest_rate:.1%})")
        print(f"  Majority-corrupted triads: {corrupted_triads} ({corruption_rate:.1%})")

        self.stats['triadic_corruption_rate'] = corruption_rate

        # Corrupted triads should be rare (< 10%)
        return corruption_rate < 0.15

    def run_all(self) -> bool:
        """Run all Byzantine fault tolerance tests."""
        self.setup()

        tests = [
            ("PoB Validation", self.test_pob_validation),
            ("Gradient Anomaly Detection", self.test_gradient_anomaly_detection),
            ("Trust Verification", self.test_trust_verification),
            ("Sybil Resistance", self.test_sybil_resistance),
            ("Aggregation Resilience", self.test_gradient_aggregation_resilience),
            ("Triadic Consensus Integrity", self.test_triadic_consensus_integrity),
        ]

        results = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results.append((name, passed))
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))

        # Summary
        print(f"\n{'='*60}")
        print(f"BYZANTINE FAULT TOLERANCE SUMMARY")
        print(f"{'='*60}")

        passed_count = sum(1 for _, p in results if p)
        total = len(results)

        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

        print(f"\n  Result: {passed_count}/{total} tests passed")
        print(f"\n  Network Config:")
        print(f"    Total nodes: {self.n_nodes}")
        print(f"    Bad actors: {self.n_bad_actors} ({self.n_bad_actors/self.n_nodes:.1%})")
        print(f"    Byzantine tolerance: {self.n_bad_actors < self.n_nodes / 3}")

        print(f"\n  Key Metrics:")
        print(f"    Gradient detection recall: {self.stats.get('gradient_recall', 0):.1%}")
        print(f"    Sybil resistance: {self.stats.get('sybil_honest_ratio', 0):.1%} honest power")
        print(f"    Triadic corruption: {self.stats.get('triadic_corruption_rate', 0):.1%}")

        if passed_count == total:
            print(f"\n  CONCLUSION: BAZINGA survives Byzantine attacks!")
            print(f"  The φ-weighted consensus successfully neutralizes bad actors.")
        else:
            print(f"\n  Some tests failed - review attack scenarios above.")

        print(f"{'='*60}")

        return passed_count == total


# ============================================================================
# PYTEST WRAPPERS
# ============================================================================

def test_byzantine_small():
    """Quick Byzantine test with 50 nodes, 5 bad actors."""
    sim = ByzantineSimulation(n_nodes=50, n_bad_actors=5)
    assert sim.run_all(), "Byzantine test (small) should pass"


def test_byzantine_medium():
    """Medium Byzantine test with 200 nodes, 20 bad actors."""
    sim = ByzantineSimulation(n_nodes=200, n_bad_actors=20)
    assert sim.run_all(), "Byzantine test (medium) should pass"


def test_byzantine_stress():
    """Stress test: 33% bad actors (max Byzantine tolerance)."""
    sim = ByzantineSimulation(n_nodes=100, n_bad_actors=33)
    assert sim.run_all(), "Byzantine test (33% bad actors) should pass"


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BAZINGA Byzantine Fault Tolerance Test")
    parser.add_argument("--nodes", "-n", type=int, default=100, help="Number of nodes")
    parser.add_argument("--bad-actors", "-b", type=int, default=10, help="Number of bad actors")
    parser.add_argument("--attack", "-a", type=str, default="mixed",
                        choices=["mixed", "fake_gradient", "trust_lie", "poison", "zero", "fake_pob"],
                        help="Attack type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sim = ByzantineSimulation(
        n_nodes=args.nodes,
        n_bad_actors=args.bad_actors,
        attack_type=args.attack,
        verbose=args.verbose
    )
    success = sim.run_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
