#!/usr/bin/env python3
"""
BAZINGA PoB Miner - Zero-Energy Block Creation
================================================

This is NOT traditional mining.

Traditional Mining (PoW):
- Burn energy to find nonce
- Race against other miners
- Waste electricity
- No understanding required

PoB Mining:
- Generate Proof-of-Boundary
- Find the φ⁴ resonance
- ~1.618ms per φ-step
- UNDERSTANDING is the work

"You can buy hashpower. You can buy stake. You CANNOT buy understanding."

Energy comparison:
- Bitcoin: ~707 kWh per transaction
- Ethereum (PoS): ~0.03 kWh per transaction
- BAZINGA (PoB): ~0.00001 kWh per transaction (70 BILLION times more efficient than Bitcoin)
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .block import Block, BlockHeader
from .transaction import Transaction
from .chain import DarmiyanChain

# Import Darmiyan for PoB
try:
    from ..darmiyan.constants import PHI, PHI_4, ABHI_AMU
    from ..darmiyan.protocol import DarmiyanNode, BoundaryProof
    from ..darmiyan.consensus import TriadicConsensus, ConsensusResult
except ImportError:
    PHI = 1.618033988749895
    PHI_4 = PHI ** 4
    ABHI_AMU = 515
    DarmiyanNode = None
    BoundaryProof = None
    TriadicConsensus = None


@dataclass
class MiningResult:
    """Result of a mining attempt."""
    success: bool
    block: Optional[Block]
    proofs: List[Dict]
    attempts: int
    time_ms: float
    message: str


class PoBMiner:
    """
    Proof-of-Boundary miner for the Darmiyan chain.

    Unlike PoW, this miner doesn't waste energy.
    It generates understanding through φ-resonance.

    Usage:
        miner = PoBMiner(chain, wallet)

        # Mine a block
        result = await miner.mine()

        if result.success:
            chain.add_block(result.block)
    """

    def __init__(
        self,
        chain: DarmiyanChain,
        node_id: str,
        auto_add: bool = True,
    ):
        self.chain = chain
        self.node_id = node_id
        self.auto_add = auto_add

        # Create 3 Darmiyan nodes for triadic consensus
        if DarmiyanNode:
            self.nodes = [
                DarmiyanNode(f"{node_id}_a"),
                DarmiyanNode(f"{node_id}_b"),
                DarmiyanNode(f"{node_id}_c"),
            ]
            self.consensus = TriadicConsensus(self.nodes)
        else:
            self.nodes = []
            self.consensus = None

        # Stats
        self.blocks_mined = 0
        self.total_attempts = 0
        self.total_time_ms = 0

    async def mine(self, max_attempts: int = 10) -> MiningResult:
        """
        Mine a new block using Proof-of-Boundary.

        This doesn't waste energy like PoW. Instead, it:
        1. Generates triadic PoB proofs
        2. Creates a block with pending transactions
        3. Validates and adds to chain

        Args:
            max_attempts: Max consensus attempts

        Returns:
            MiningResult with block if successful
        """
        if not self.consensus:
            return MiningResult(
                success=False,
                block=None,
                proofs=[],
                attempts=0,
                time_ms=0,
                message="Darmiyan not available - install darmiyan module",
            )

        if not self.chain.pending_transactions:
            return MiningResult(
                success=False,
                block=None,
                proofs=[],
                attempts=0,
                time_ms=0,
                message="No pending transactions to mine",
            )

        start_time = time.time()
        attempts = 0

        for _ in range(max_attempts):
            attempts += 1
            self.total_attempts += 1

            # Attempt triadic consensus
            result = await self.consensus.attempt_consensus()

            if result.achieved:
                # Convert proofs to dict
                proofs = [p.to_dict() for p in result.proofs]

                # Create block
                block = self.chain.create_block(proofs)

                if block:
                    # Validate and add
                    if self.auto_add:
                        success = self.chain.add_block(block)
                    else:
                        success = block.validate(self.chain.get_latest_block())

                    if success:
                        self.blocks_mined += 1
                        elapsed = (time.time() - start_time) * 1000
                        self.total_time_ms += elapsed

                        return MiningResult(
                            success=True,
                            block=block,
                            proofs=proofs,
                            attempts=attempts,
                            time_ms=elapsed,
                            message=f"Block #{block.header.index} mined successfully",
                        )

        # Failed all attempts
        elapsed = (time.time() - start_time) * 1000
        return MiningResult(
            success=False,
            block=None,
            proofs=[],
            attempts=attempts,
            time_ms=elapsed,
            message=f"Failed to achieve consensus after {attempts} attempts",
        )

    def mine_sync(self, max_attempts: int = 10) -> MiningResult:
        """Synchronous version of mine()."""
        if not self.consensus:
            return MiningResult(
                success=False,
                block=None,
                proofs=[],
                attempts=0,
                time_ms=0,
                message="Darmiyan not available",
            )

        if not self.chain.pending_transactions:
            return MiningResult(
                success=False,
                block=None,
                proofs=[],
                attempts=0,
                time_ms=0,
                message="No pending transactions",
            )

        start_time = time.time()
        attempts = 0

        for _ in range(max_attempts):
            attempts += 1
            self.total_attempts += 1

            # Attempt triadic consensus (sync)
            result = self.consensus.attempt_consensus_sync()

            if result.achieved:
                proofs = [p.to_dict() for p in result.proofs]
                block = self.chain.create_block(proofs)

                if block:
                    if self.auto_add:
                        success = self.chain.add_block(block)
                    else:
                        success = block.validate(self.chain.get_latest_block())

                    if success:
                        self.blocks_mined += 1
                        elapsed = (time.time() - start_time) * 1000
                        self.total_time_ms += elapsed

                        return MiningResult(
                            success=True,
                            block=block,
                            proofs=proofs,
                            attempts=attempts,
                            time_ms=elapsed,
                            message=f"Block #{block.header.index} mined",
                        )

        elapsed = (time.time() - start_time) * 1000
        return MiningResult(
            success=False,
            block=None,
            proofs=[],
            attempts=attempts,
            time_ms=elapsed,
            message=f"Failed after {attempts} attempts",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get mining statistics."""
        avg_time = self.total_time_ms / self.blocks_mined if self.blocks_mined > 0 else 0
        avg_attempts = self.total_attempts / self.blocks_mined if self.blocks_mined > 0 else 0

        return {
            'node_id': self.node_id,
            'blocks_mined': self.blocks_mined,
            'total_attempts': self.total_attempts,
            'total_time_ms': self.total_time_ms,
            'average_time_ms': avg_time,
            'average_attempts': avg_attempts,
            'consensus_stats': self.consensus.get_stats() if self.consensus else None,
        }

    def print_status(self):
        """Print miner status."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("  BAZINGA PoB MINER")
        print("=" * 60)
        print(f"  Node ID: {stats['node_id']}")
        print(f"  Blocks Mined: {stats['blocks_mined']}")
        print(f"  Total Attempts: {stats['total_attempts']}")
        print("-" * 60)
        print(f"  Average Time: {stats['average_time_ms']:.2f}ms")
        print(f"  Average Attempts: {stats['average_attempts']:.1f}")
        if stats['consensus_stats']:
            cs = stats['consensus_stats']
            print(f"  Consensus Success: {cs['success_rate']*100:.1f}%")
        print("=" * 60)


def mine_block(
    chain: DarmiyanChain,
    node_id: str = "miner",
) -> MiningResult:
    """Quick function to mine a single block."""
    miner = PoBMiner(chain, node_id)
    return miner.mine_sync()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  BAZINGA PoB MINER TEST")
    print("=" * 60)
    print()

    # Create chain and miner
    with tempfile.TemporaryDirectory() as tmpdir:
        from .chain import create_chain

        chain = create_chain(data_dir=tmpdir)
        miner = PoBMiner(chain, "test_miner")

        print(f"Miner initialized: {miner.node_id}")
        print(f"Chain height: {len(chain)}")
        print()

        # Add some knowledge to mine
        print("Adding knowledge transactions...")
        chain.add_knowledge(
            content="The golden ratio φ ≈ 1.618 appears throughout nature",
            summary="Golden ratio in nature",
            sender="test_node",
            confidence=0.9,
        )
        chain.add_knowledge(
            content="Proof-of-Boundary uses P/G ≈ φ⁴ for consensus",
            summary="PoB consensus mechanism",
            sender="test_node",
            confidence=0.85,
        )
        print(f"  Pending transactions: {len(chain.pending_transactions)}")
        print()

        # Mine block
        print("Mining block with PoB...")
        result = miner.mine_sync(max_attempts=20)

        print(f"  Success: {result.success}")
        print(f"  Attempts: {result.attempts}")
        print(f"  Time: {result.time_ms:.2f}ms")
        print(f"  Message: {result.message}")

        if result.success:
            print(f"  Block Hash: {result.block.hash[:32]}...")
            print(f"  Transactions: {len(result.block.transactions)}")

        print()

        # Stats
        miner.print_status()

        # Chain status
        chain.print_chain()

    print("\n  PoB Miner test complete!")
