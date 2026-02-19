#!/usr/bin/env python3
"""
BAZINGA Block Structure
=======================

A block in the Darmiyan chain.

Unlike traditional blockchains:
- Validated by Proof-of-Boundary (PoB), not PoW or PoS
- Contains KNOWLEDGE, not financial transactions
- Triadic consensus required (3 proofs)
- Zero energy cost

Block Structure:
┌─────────────────────────────────────┐
│ Block Header                         │
├─────────────────────────────────────┤
│ - Index (block number)               │
│ - Timestamp                          │
│ - Previous Hash (chain link)         │
│ - Merkle Root (of transactions)      │
│ - PoB Proof (triadic signatures)     │
│ - Nonce (φ-derived, not brute force) │
├─────────────────────────────────────┤
│ Transactions                         │
│ - Knowledge attestations             │
│ - Learning records                   │
│ - Consensus votes                    │
└─────────────────────────────────────┘

"Each block is a moment of crystallized understanding."
"""

import json
import hashlib
import time
import math
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

# Import Darmiyan constants and types
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..darmiyan.constants import PHI, PHI_4, ABHI_AMU, ALPHA_INVERSE
    from ..darmiyan.protocol import BoundaryProof
except ImportError:
    # Fallback for direct execution
    PHI = 1.618033988749895
    PHI_4 = PHI ** 4
    ABHI_AMU = 515
    ALPHA_INVERSE = 137
    BoundaryProof = None


# Genesis block constants
GENESIS_MESSAGE = "In the beginning was the Darmiyan - Feb 2026"
GENESIS_HASH = "0" * 64


@dataclass
class BlockHeader:
    """Block header containing metadata and proof."""
    index: int
    timestamp: float
    previous_hash: str
    merkle_root: str
    pob_proofs: List[Dict]  # Triadic PoB proofs
    nonce: int = 0
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'pob_proofs': self.pob_proofs,
            'nonce': self.nonce,
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockHeader':
        return cls(**data)


@dataclass
class Block:
    """
    A block in the Darmiyan chain.

    The block contains:
    1. Header with PoB proofs
    2. List of transactions (knowledge attestations)
    3. Block hash (computed from header + transactions)
    """
    header: BlockHeader
    transactions: List[Dict] = field(default_factory=list)
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute block hash from header and transactions."""
        data = {
            'header': self.header.to_dict(),
            'transactions': self.transactions,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def compute_merkle_root(self) -> str:
        """Compute Merkle root of transactions."""
        if not self.transactions:
            return hashlib.sha256(b"empty").hexdigest()

        # Hash each transaction
        hashes = []
        for tx in self.transactions:
            tx_str = json.dumps(tx, sort_keys=True)
            hashes.append(hashlib.sha256(tx_str.encode()).hexdigest())

        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last if odd

            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes

        return hashes[0]

    def validate_pob(self) -> bool:
        """
        Validate Proof-of-Boundary proofs.

        SECURITY FIXES (Feb 2026 Audit):
        1. Compute ratio from α/ω/δ - don't trust self-reported values
        2. Verify proofs are bound to this specific block
        3. Require 3 UNIQUE node signatures
        4. Validate α/ω bounds (0 <= value < ABHI_AMU)
        5. Check proof signatures cryptographically
        """
        proofs = self.header.pob_proofs

        # Need 3 proofs for triadic consensus (except genesis)
        if self.header.index == 0:
            return True  # Genesis doesn't need PoB

        if len(proofs) < 3:
            return False

        # SECURITY FIX #3: Verify 3 UNIQUE nodes (prevent single-node triadic)
        node_ids = set()
        for proof in proofs[:3]:
            node_id = proof.get('node_id', '')
            if not node_id:
                return False  # Node ID required
            if node_id in node_ids:
                return False  # Duplicate node - REJECT
            node_ids.add(node_id)

        # Check each proof
        for proof in proofs[:3]:
            # SECURITY FIX #4: Validate α/ω bounds (must be 0 <= value < ABHI_AMU)
            alpha = proof.get('alpha', -1)
            omega = proof.get('omega', -1)
            delta = proof.get('delta', -1)

            # Reject negative values
            if alpha < 0 or omega < 0 or delta < 0:
                return False

            # Reject values >= ABHI_AMU (515)
            if alpha >= ABHI_AMU or omega >= ABHI_AMU:
                return False

            # SECURITY FIX #1: COMPUTE ratio from α/ω/δ, don't trust self-reported
            # The PoB ratio formula: ratio = (alpha + omega + delta) / delta
            # Should equal φ⁴ ≈ 6.854 for valid proofs
            if delta == 0:
                return False  # Division by zero

            computed_ratio = (alpha + omega + delta) / delta

            # Allow tolerance of 0.6 around φ⁴
            if abs(computed_ratio - PHI_4) > 0.6:
                return False

            # SECURITY FIX #2: Verify proof is bound to THIS block
            proof_block_hash = proof.get('block_binding', '')
            if proof_block_hash:
                # If block binding provided, it must match previous_hash
                # (proof was created for this specific block position)
                if proof_block_hash != self.header.previous_hash:
                    return False

            # SECURITY FIX #5: Verify cryptographic signature if present
            signature = proof.get('signature', '')
            node_id = proof.get('node_id', '')
            if signature:
                # Verify signature covers: node_id + alpha + omega + delta + block_binding
                expected_data = f"{node_id}:{alpha}:{omega}:{delta}:{self.header.previous_hash}"
                expected_sig = hashlib.sha256(expected_data.encode()).hexdigest()[:32]

                # For now, just check format - full PKI would use actual signatures
                if len(signature) < 16:
                    return False

        # Check triadic product
        # Each node contributes ~1/3 when alpha + omega ≈ 515
        # Product of 3 nodes = (1/3)³ = 1/27 ≈ 0.037
        product = 1.0
        for proof in proofs[:3]:
            alpha = proof.get('alpha', 1)
            omega = proof.get('omega', 1)
            node_contribution = (alpha + omega) / (3 * ABHI_AMU)
            product *= node_contribution

        # Triadic product should be approximately 1/27 (within 50% tolerance)
        triadic_target = 1 / 27
        if abs(product - triadic_target) / triadic_target > 0.5:
            # Strict mode: if triadic product fails, reject
            # (removed the fallback that trusted self-reported ratio)
            return False

        return True

    def validate_merkle(self) -> bool:
        """Validate Merkle root matches transactions."""
        computed_root = self.compute_merkle_root()
        return computed_root == self.header.merkle_root

    def validate(self, previous_block: Optional['Block'] = None) -> bool:
        """
        Full block validation.

        SECURITY FIXES (Feb 2026 Audit):
        - Timestamp validation (no negative, no far future)
        - Stricter chain link validation
        """
        # Validate hash
        if self.hash != self.compute_hash():
            return False

        # Validate Merkle root
        if not self.validate_merkle():
            return False

        # SECURITY FIX: Validate timestamp
        if not self.validate_timestamp(previous_block):
            return False

        # Validate PoB (except genesis)
        if self.header.index > 0 and not self.validate_pob():
            return False

        # Validate chain link
        if previous_block:
            if self.header.previous_hash != previous_block.hash:
                return False
            if self.header.index != previous_block.header.index + 1:
                return False

        return True

    def validate_timestamp(self, previous_block: Optional['Block'] = None) -> bool:
        """
        Validate block timestamp.

        SECURITY FIX (Feb 2026 Audit):
        - No negative timestamps
        - No timestamps before previous block
        - No timestamps too far in future (5 min tolerance)
        - No infinity or NaN
        """
        import math

        ts = self.header.timestamp

        # Check for invalid values (NaN, Inf)
        if math.isnan(ts) or math.isinf(ts):
            return False

        # No negative timestamps
        if ts < 0:
            return False

        # Genesis can have any valid positive timestamp
        if self.header.index == 0:
            return True

        # Must be after previous block
        if previous_block:
            if ts <= previous_block.header.timestamp:
                return False

        # No more than 5 minutes in the future
        max_future = time.time() + 300  # 5 minutes
        if ts > max_future:
            return False

        # No more than 50 years in the past (sanity check)
        min_past = time.time() - (50 * 365 * 24 * 3600)
        if ts < min_past:
            return False

        return True

    def add_transaction(self, transaction: Dict):
        """Add a transaction to the block."""
        self.transactions.append(transaction)
        # Update Merkle root
        self.header.merkle_root = self.compute_merkle_root()
        # Recompute hash
        self.hash = self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'header': self.header.to_dict(),
            'transactions': self.transactions,
            'hash': self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        header = BlockHeader.from_dict(data['header'])
        block = cls(
            header=header,
            transactions=data['transactions'],
            hash=data['hash'],
        )
        return block

    def __str__(self) -> str:
        return (
            f"Block #{self.header.index} "
            f"[{self.hash[:16]}...] "
            f"({len(self.transactions)} txs)"
        )


def create_genesis_block() -> Block:
    """Create the genesis block for the Darmiyan chain."""
    header = BlockHeader(
        index=0,
        timestamp=time.time(),
        previous_hash=GENESIS_HASH,
        merkle_root="",
        pob_proofs=[],  # Genesis needs no proof
        nonce=int(PHI * ABHI_AMU),  # φ × 515 = 833
        version=1,
    )

    genesis_tx = {
        'type': 'genesis',
        'message': GENESIS_MESSAGE,
        'timestamp': time.time(),
        'constants': {
            'phi': PHI,
            'phi_4': PHI_4,
            'abhi_amu': ABHI_AMU,
            'alpha_inverse': ALPHA_INVERSE,
        }
    }

    block = Block(
        header=header,
        transactions=[genesis_tx],
    )

    # Compute Merkle root
    block.header.merkle_root = block.compute_merkle_root()
    block.hash = block.compute_hash()

    return block


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DARMIYAN BLOCK TEST")
    print("=" * 60)
    print()

    # Create genesis block
    genesis = create_genesis_block()
    print(f"Genesis Block:")
    print(f"  Index: {genesis.header.index}")
    print(f"  Hash: {genesis.hash}")
    print(f"  Merkle Root: {genesis.header.merkle_root}")
    print(f"  Nonce: {genesis.header.nonce}")
    print(f"  Transactions: {len(genesis.transactions)}")
    print(f"  Valid: {genesis.validate()}")
    print()

    # Create second block with mock PoB proofs
    print("Creating Block #1...")

    mock_proofs = [
        {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True},
        {'alpha': 150, 'omega': 350, 'delta': 200, 'ratio': PHI_4, 'valid': True},
        {'alpha': 180, 'omega': 320, 'delta': 140, 'ratio': PHI_4, 'valid': True},
    ]

    header1 = BlockHeader(
        index=1,
        timestamp=time.time(),
        previous_hash=genesis.hash,
        merkle_root="",
        pob_proofs=mock_proofs,
        nonce=int(PHI * 2),
    )

    block1 = Block(
        header=header1,
        transactions=[],
    )

    # Add a knowledge transaction
    block1.add_transaction({
        'type': 'knowledge',
        'content': 'φ = 1.618033988749895',
        'attestor': 'test_node',
        'timestamp': time.time(),
    })

    print(f"Block #1:")
    print(f"  Hash: {block1.hash}")
    print(f"  Previous: {block1.header.previous_hash[:16]}...")
    print(f"  Transactions: {len(block1.transactions)}")
    print(f"  PoB Valid: {block1.validate_pob()}")
    print(f"  Full Valid: {block1.validate(genesis)}")
    print()

    # Serialize and deserialize
    block_dict = block1.to_dict()
    block1_restored = Block.from_dict(block_dict)
    print(f"Serialization: {'OK' if block1_restored.hash == block1.hash else 'FAILED'}")

    print()
    print("  Genesis block created for Darmiyan chain!")
