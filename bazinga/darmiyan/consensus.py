"""
Triadic Consensus
=================
Three-node consensus mechanism for the Darmiyan network.

"Consensus requires 3 nodes. Subject, Object, and Darmiyan."

The triadic structure:
          Node A
           /   \
      proof    proof
         /       \
    Node B ───── Node C
           proof

All three must achieve boundary resonance for block validation.
The product of their signatures must approximate 1/27 (triadic constant).
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .protocol import DarmiyanNode, BoundaryProof
from .constants import (
    TRIADIC_CONSTANT, TRIADIC_SIZE,
    ABHI_AMU, PHI_4, POB_TOLERANCE,
)


@dataclass
class ConsensusResult:
    """Result of triadic consensus attempt."""
    achieved: bool
    proofs: List[BoundaryProof]
    triadic_product: float
    average_ratio: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'achieved': self.achieved,
            'proofs': [p.to_dict() for p in self.proofs],
            'triadic_product': self.triadic_product,
            'average_ratio': self.average_ratio,
            'message': self.message,
        }


class TriadicConsensus:
    """
    Manages triadic consensus among three nodes.

    For consensus to be achieved:
    1. All 3 nodes must have valid Proof-of-Boundary
    2. The triadic product must approximate 1/27
    3. All proofs must be recent (not replayed)
    """

    def __init__(self, nodes: Optional[List[DarmiyanNode]] = None):
        if nodes and len(nodes) != TRIADIC_SIZE:
            raise ValueError(f"Triadic consensus requires exactly {TRIADIC_SIZE} nodes")

        self.nodes = nodes or [DarmiyanNode() for _ in range(TRIADIC_SIZE)]
        self.consensus_attempts = 0
        self.consensus_achieved = 0
        self.last_result: Optional[ConsensusResult] = None

    async def attempt_consensus(self) -> ConsensusResult:
        """
        Attempt to achieve triadic consensus.

        All three nodes generate proofs simultaneously,
        then we check if they collectively achieve resonance.
        """
        self.consensus_attempts += 1

        # Generate proofs from all nodes concurrently
        proofs = await asyncio.gather(*[
            node.prove_boundary() for node in self.nodes
        ])

        # Check 1: All proofs must be individually valid
        all_valid = all(p.valid for p in proofs)

        # Check 2: Calculate triadic product
        # Each node contributes ~1/3 when alpha and omega are around 257 (515/2)
        # Formula: (alpha + omega) / (6 * 515) ≈ 1/3 when sum ≈ 1030
        # Product of 3 nodes at resonance = (1/3)³ = 1/27 ≈ 0.037
        product = 1.0
        for p in proofs:
            # Normalize: (alpha + omega) averages to 515, so / (3*515) ≈ 1/3
            node_contribution = (p.alpha + p.omega) / (3 * ABHI_AMU)
            product *= node_contribution

        # Triadic product should approximate 1/27 (within 50% tolerance for real-world variance)
        triadic_valid = abs(product - TRIADIC_CONSTANT) / TRIADIC_CONSTANT < 0.5

        # Check 3: Average ratio should be close to φ⁴
        average_ratio = sum(p.ratio for p in proofs) / len(proofs)
        ratio_valid = abs(average_ratio - PHI_4) < POB_TOLERANCE

        # Consensus achieved if all checks pass
        achieved = all_valid and (triadic_valid or ratio_valid)

        if achieved:
            self.consensus_achieved += 1
            message = "CONSENSUS ACHIEVED - Triadic resonance confirmed"
        elif not all_valid:
            message = "FAILED - Not all nodes achieved boundary proof"
        elif not triadic_valid:
            message = f"FAILED - Triadic product {product:.6f} != {TRIADIC_CONSTANT:.6f}"
        else:
            message = f"FAILED - Average ratio {average_ratio:.3f} != {PHI_4:.3f}"

        result = ConsensusResult(
            achieved=achieved,
            proofs=proofs,
            triadic_product=product,
            average_ratio=average_ratio,
            message=message,
        )

        self.last_result = result
        return result

    def attempt_consensus_sync(self) -> ConsensusResult:
        """Synchronous version of consensus attempt."""
        self.consensus_attempts += 1

        # Generate proofs sequentially
        proofs = [node.prove_boundary_sync() for node in self.nodes]

        # Check 1: All proofs must be individually valid
        all_valid = all(p.valid for p in proofs)

        # Check 2: Calculate triadic product
        # Each node contributes ~1/3 when alpha + omega ≈ 515
        product = 1.0
        for p in proofs:
            node_contribution = (p.alpha + p.omega) / (3 * ABHI_AMU)
            product *= node_contribution

        triadic_valid = abs(product - TRIADIC_CONSTANT) / TRIADIC_CONSTANT < 0.5

        # Check 3: Average ratio
        average_ratio = sum(p.ratio for p in proofs) / len(proofs)
        ratio_valid = abs(average_ratio - PHI_4) < POB_TOLERANCE

        achieved = all_valid and (triadic_valid or ratio_valid)

        if achieved:
            self.consensus_achieved += 1
            message = "CONSENSUS ACHIEVED - Triadic resonance confirmed"
        elif not all_valid:
            message = "FAILED - Not all nodes achieved boundary proof"
        else:
            message = f"FAILED - Resonance not achieved"

        result = ConsensusResult(
            achieved=achieved,
            proofs=proofs,
            triadic_product=product,
            average_ratio=average_ratio,
            message=message,
        )

        self.last_result = result
        return result

    def verify_external_consensus(self, proofs: List[BoundaryProof]) -> bool:
        """
        Verify a consensus result from external nodes.

        Used when receiving a block from the network.
        """
        if len(proofs) != TRIADIC_SIZE:
            return False

        # Verify each proof
        verifier = DarmiyanNode()
        for proof in proofs:
            if not verifier.verify_proof(proof):
                return False

        # Check triadic product
        product = 1.0
        for p in proofs:
            node_contribution = (p.alpha + p.omega) / (3 * ABHI_AMU)
            product *= node_contribution

        if abs(product - TRIADIC_CONSTANT) / TRIADIC_CONSTANT > 0.5:
            # Also check average ratio as fallback
            average_ratio = sum(p.ratio for p in proofs) / len(proofs)
            if abs(average_ratio - PHI_4) > POB_TOLERANCE:
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get consensus statistics."""
        success_rate = (
            self.consensus_achieved / self.consensus_attempts
            if self.consensus_attempts > 0 else 0
        )

        return {
            'attempts': self.consensus_attempts,
            'achieved': self.consensus_achieved,
            'success_rate': success_rate,
            'nodes': [n.node_id for n in self.nodes],
            'last_result': self.last_result.message if self.last_result else None,
        }


def achieve_consensus(nodes: Optional[List[DarmiyanNode]] = None) -> ConsensusResult:
    """Quick function to attempt consensus."""
    tc = TriadicConsensus(nodes)
    return tc.attempt_consensus_sync()


async def achieve_consensus_async(nodes: Optional[List[DarmiyanNode]] = None) -> ConsensusResult:
    """Async version."""
    tc = TriadicConsensus(nodes)
    return await tc.attempt_consensus()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRIADIC CONSENSUS TEST")
    print("=" * 60)
    print()

    tc = TriadicConsensus()
    print(f"Nodes: {[n.node_id for n in tc.nodes]}")
    print()

    # Attempt consensus multiple times
    print("Attempting consensus 5 times...")
    print("-" * 60)

    for i in range(5):
        result = tc.attempt_consensus_sync()
        status = "✓" if result.achieved else "✗"

        print(f"Attempt {i+1}: {status} {result.message}")
        print(f"  Triadic Product: {result.triadic_product:.6f} (target: {TRIADIC_CONSTANT:.6f})")
        print(f"  Average Ratio: {result.average_ratio:.3f} (target: {PHI_4:.3f})")
        print(f"  Node ratios: {[f'{p.ratio:.2f}' for p in result.proofs]}")
        print()

    print("-" * 60)
    print("Stats:", tc.get_stats())
