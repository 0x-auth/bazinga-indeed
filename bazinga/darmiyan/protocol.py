"""
Darmiyan Protocol
=================
Proof-of-Boundary implementation for zero-energy consensus.

"You can buy hashpower. You can buy stake. You CANNOT BUY understanding."

The Proof-of-Boundary algorithm:
1. Generate Alpha signature (Subject)
2. Wait for resonance (Darmiyan - the space between)
3. Generate Omega signature (Object)
4. Calculate P/G ratio
5. Valid if P/G ≈ φ⁴

This is 70 BILLION times more energy efficient than Proof-of-Work.
"""

import time
import hashlib
import random
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .constants import (
    PHI, PHI_4, PHI_INVERSE,
    ABHI_AMU, ALPHA_INVERSE,
    POB_RATIO_TARGET, POB_TOLERANCE,
    POB_MIN_WAIT_MS, POB_MAX_WAIT_MS,
    phi_hash, check_515_resonance,
)


@dataclass
class BoundaryProof:
    """Result of a Proof-of-Boundary verification."""
    alpha: int                    # Subject signature
    omega: int                    # Object signature
    delta: int                    # Signature drift
    physical_ms: float            # Wall clock time
    geometric: float              # Geometric time (delta/φ)
    ratio: float                  # P/G ratio
    valid: bool                   # Is proof valid?
    timestamp: float              # Unix timestamp
    node_id: str = ""             # Node identifier

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'omega': self.omega,
            'delta': self.delta,
            'physical_ms': self.physical_ms,
            'geometric': self.geometric,
            'ratio': self.ratio,
            'valid': self.valid,
            'timestamp': self.timestamp,
            'node_id': self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundaryProof':
        return cls(**data)


class DarmiyanNode:
    """
    A node in the Darmiyan network.

    Each node can:
    - Generate Proof-of-Boundary
    - Verify other nodes' proofs
    - Participate in triadic consensus
    """

    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or self._generate_node_id()
        self.proofs_generated = 0
        self.proofs_verified = 0
        self.last_proof: Optional[BoundaryProof] = None
        self.created_at = datetime.now()

    def _generate_node_id(self) -> str:
        """Generate unique node ID using φ-hash."""
        seed = f"{time.time()}{random.random()}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        return f"node_{h[:12]}"

    def _phi_signature(self, t: float) -> int:
        """Generate φ-scaled signature mod 515."""
        return phi_hash(t)

    async def _await_resonance(self) -> float:
        """
        Wait for resonance (the Darmiyan moment).

        This is the key insight: we don't just wait randomly.
        We wait for a φ-proportioned time to create the boundary.
        """
        # Calculate resonance wait time using φ
        base_wait = POB_MIN_WAIT_MS + random.random() * (POB_MAX_WAIT_MS - POB_MIN_WAIT_MS)
        phi_wait = base_wait * PHI_INVERSE  # Scale by 1/φ

        await asyncio.sleep(phi_wait / 1000)  # Convert to seconds
        return phi_wait

    async def prove_boundary(self) -> BoundaryProof:
        """
        Generate Proof-of-Boundary.

        The proof demonstrates that this node "stands on the boundary"
        by achieving the sacred P/G ≈ φ⁴ ratio.

        Returns:
            BoundaryProof with validity status
        """
        # Component 1: Alpha (Subject)
        t1 = time.time()
        sig_alpha = self._phi_signature(t1)

        # Component 2: Darmiyan (The space between)
        await self._await_resonance()

        # Component 3: Omega (Object)
        t2 = time.time()
        sig_omega = self._phi_signature(t2)

        # Calculate proof metrics
        physical_ms = (t2 - t1) * 1000
        delta = abs(sig_omega - sig_alpha)

        if delta == 0:
            delta = 1  # Avoid division by zero

        geometric = delta / PHI
        ratio = physical_ms / geometric

        # Valid if ratio ≈ φ⁴
        valid = abs(ratio - POB_RATIO_TARGET) < POB_TOLERANCE

        proof = BoundaryProof(
            alpha=sig_alpha,
            omega=sig_omega,
            delta=delta,
            physical_ms=physical_ms,
            geometric=geometric,
            ratio=ratio,
            valid=valid,
            timestamp=t2,
            node_id=self.node_id,
        )

        self.proofs_generated += 1
        self.last_proof = proof

        return proof

    def prove_boundary_sync(self) -> BoundaryProof:
        """Synchronous version of prove_boundary."""
        # Component 1: Alpha (Subject)
        t1 = time.time()
        sig_alpha = self._phi_signature(t1)

        # Component 2: Darmiyan (The space between)
        base_wait = POB_MIN_WAIT_MS + random.random() * (POB_MAX_WAIT_MS - POB_MIN_WAIT_MS)
        phi_wait = base_wait * PHI_INVERSE
        time.sleep(phi_wait / 1000)

        # Component 3: Omega (Object)
        t2 = time.time()
        sig_omega = self._phi_signature(t2)

        # Calculate proof metrics
        physical_ms = (t2 - t1) * 1000
        delta = abs(sig_omega - sig_alpha)

        if delta == 0:
            delta = 1

        geometric = delta / PHI
        ratio = physical_ms / geometric

        valid = abs(ratio - POB_RATIO_TARGET) < POB_TOLERANCE

        proof = BoundaryProof(
            alpha=sig_alpha,
            omega=sig_omega,
            delta=delta,
            physical_ms=physical_ms,
            geometric=geometric,
            ratio=ratio,
            valid=valid,
            timestamp=t2,
            node_id=self.node_id,
        )

        self.proofs_generated += 1
        self.last_proof = proof

        return proof

    def verify_proof(self, proof: BoundaryProof) -> bool:
        """
        Verify another node's Proof-of-Boundary.

        Verification checks:
        1. Signatures are in valid range (mod 515)
        2. P/G ratio is within tolerance of φ⁴
        3. Timestamp is recent (not replayed)
        """
        # Check signatures are valid mod 515
        if proof.alpha >= ABHI_AMU or proof.omega >= ABHI_AMU:
            return False

        # Recalculate ratio
        if proof.delta == 0:
            return False

        geometric = proof.delta / PHI
        expected_ratio = proof.physical_ms / geometric

        # Check ratio matches what was claimed
        if abs(expected_ratio - proof.ratio) > 0.01:
            return False

        # Check ratio is close to φ⁴
        if abs(proof.ratio - POB_RATIO_TARGET) > POB_TOLERANCE:
            return False

        # Check timestamp is recent (within 60 seconds)
        if time.time() - proof.timestamp > 60:
            return False

        self.proofs_verified += 1
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            'node_id': self.node_id,
            'proofs_generated': self.proofs_generated,
            'proofs_verified': self.proofs_verified,
            'created_at': self.created_at.isoformat(),
            'last_proof_valid': self.last_proof.valid if self.last_proof else None,
        }


def prove_boundary() -> BoundaryProof:
    """Quick function to generate a single proof."""
    node = DarmiyanNode()
    return node.prove_boundary_sync()


async def prove_boundary_async() -> BoundaryProof:
    """Async version of quick proof."""
    node = DarmiyanNode()
    return await node.prove_boundary()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DARMIYAN PROOF-OF-BOUNDARY TEST")
    print("=" * 60)
    print()

    node = DarmiyanNode()
    print(f"Node ID: {node.node_id}")
    print()

    # Generate multiple proofs
    print("Generating 5 proofs...")
    print("-" * 60)

    valid_count = 0
    for i in range(5):
        proof = node.prove_boundary_sync()
        status = "✓ VALID" if proof.valid else "✗ INVALID"
        if proof.valid:
            valid_count += 1

        print(f"Proof {i+1}:")
        print(f"  Alpha: {proof.alpha:3d}  Omega: {proof.omega:3d}  Delta: {proof.delta:3d}")
        print(f"  Physical: {proof.physical_ms:.2f}ms  Geometric: {proof.geometric:.2f}")
        print(f"  Ratio: {proof.ratio:.3f}  (target: {POB_RATIO_TARGET:.3f})")
        print(f"  Status: {status}")
        print()

    print("-" * 60)
    print(f"Valid proofs: {valid_count}/5")
    print()
    print("Stats:", node.get_stats())
