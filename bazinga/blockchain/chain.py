#!/usr/bin/env python3
"""
BAZINGA Darmiyan Chain - The Knowledge Ledger
==============================================

The immutable record of collective understanding.

This chain is different:
1. No financial transactions - only knowledge
2. No mining competition - PoB is about understanding
3. No energy waste - verification uses ~1.618ms per proof
4. Triadic consensus - 3 nodes must agree

Chain Structure:
Genesis → Block 1 → Block 2 → ... → Block N
   ↑         ↑          ↑              ↑
  PoB       PoB        PoB           PoB

Each block is validated by 3 Proof-of-Boundary proofs.
The chain grows as the network learns together.

"The chain is not a ledger of value. It's a ledger of UNDERSTANDING."
"""

import json
import hashlib
import time
import os
import fcntl
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .block import Block, BlockHeader, create_genesis_block
from .transaction import Transaction, TransactionType

# Import Darmiyan constants
try:
    from ..darmiyan.constants import PHI, PHI_4, ABHI_AMU
    from ..darmiyan.protocol import BoundaryProof
    from ..darmiyan.consensus import TriadicConsensus, ConsensusResult
except ImportError:
    PHI = 1.618033988749895
    PHI_4 = PHI ** 4
    ABHI_AMU = 515


@dataclass
class ChainStats:
    """Statistics about the chain."""
    height: int = 0
    total_transactions: int = 0
    knowledge_attestations: int = 0
    learning_records: int = 0
    consensus_votes: int = 0
    identity_registrations: int = 0
    alpha_seeds: int = 0  # Knowledge that is α-SEED
    average_phi_coherence: float = 0.0
    total_pob_proofs: int = 0


class DarmiyanChain:
    """
    The Darmiyan blockchain - a chain of understanding.

    Usage:
        chain = DarmiyanChain()

        # Add knowledge
        chain.add_knowledge(content, summary, sender)

        # Add block (requires PoB)
        chain.add_block(proofs)

        # Query chain
        chain.get_block(index)
        chain.search_knowledge(query)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".bazinga" / "chain"
        self.auto_save = auto_save

        # Chain storage
        self.blocks: List[Block] = []
        self.pending_transactions: List[Transaction] = []

        # Indices for fast lookup
        self.block_by_hash: Dict[str, Block] = {}
        self.tx_by_hash: Dict[str, Transaction] = {}
        self.knowledge_index: Dict[str, List[str]] = {}  # content_hash -> tx_hashes

        # SECURITY FIX: Track used proof hashes to prevent replay attacks
        self.used_proof_hashes: set = set()

        # Stats
        self.stats = ChainStats()

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize chain with genesis block or load from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        chain_file = self.data_dir / "chain.json"
        if chain_file.exists():
            self._load()
        else:
            # Create genesis block
            genesis = create_genesis_block()
            self._add_block_internal(genesis)

            if self.auto_save:
                self._save()

    def _add_block_internal(self, block: Block):
        """Add block to internal structures (no validation)."""
        self.blocks.append(block)
        self.block_by_hash[block.hash] = block

        # Index transactions
        for tx_data in block.transactions:
            if isinstance(tx_data, dict):
                tx_hash = tx_data.get('hash', '')
                if tx_hash:
                    self.tx_by_hash[tx_hash] = tx_data

                    # Index knowledge
                    if tx_data.get('tx_type') == TransactionType.KNOWLEDGE.value:
                        content_hash = tx_data.get('data', {}).get('content_hash', '')
                        if content_hash:
                            if content_hash not in self.knowledge_index:
                                self.knowledge_index[content_hash] = []
                            self.knowledge_index[content_hash].append(tx_hash)

        # Update stats
        self._update_stats()

    def _update_stats(self):
        """Update chain statistics."""
        self.stats.height = len(self.blocks)
        self.stats.total_transactions = sum(len(b.transactions) for b in self.blocks)
        self.stats.total_pob_proofs = sum(
            len(b.header.pob_proofs) for b in self.blocks if b.header.index > 0
        )

        # Count by type
        for block in self.blocks:
            for tx in block.transactions:
                if isinstance(tx, dict):
                    tx_type = tx.get('tx_type', tx.get('type', ''))
                    if tx_type == TransactionType.KNOWLEDGE.value or tx_type == 'knowledge':
                        self.stats.knowledge_attestations += 1
                        data = tx.get('data', tx)
                        if data.get('alpha_seed'):
                            self.stats.alpha_seeds += 1
                    elif tx_type == TransactionType.LEARNING.value or tx_type == 'learning':
                        self.stats.learning_records += 1
                    elif tx_type == TransactionType.CONSENSUS.value or tx_type == 'consensus':
                        self.stats.consensus_votes += 1
                    elif tx_type == TransactionType.IDENTITY.value or tx_type == 'identity':
                        self.stats.identity_registrations += 1

    def add_transaction(self, transaction: Transaction) -> str:
        """
        Add a transaction to the pending pool.

        Returns the transaction hash.
        """
        tx_hash = transaction.hash
        self.pending_transactions.append(transaction)
        return tx_hash

    def add_knowledge(
        self,
        content: str,
        summary: str,
        sender: str,
        confidence: float = 0.8,
        source_type: str = "rag",
        phi_coherence: float = 0.0,
    ) -> str:
        """
        Add a knowledge attestation to pending transactions.

        SECURITY FIX (Feb 2026 Audit):
        - Check for duplicate knowledge before adding

        Returns the transaction hash, or empty string if duplicate.
        """
        import hashlib
        from .transaction import create_knowledge_tx

        # SECURITY FIX: Check for duplicate knowledge
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check if this exact content already exists on chain
        if content_hash in self.knowledge_index:
            # Already attested - don't add duplicate
            return ""

        # Check pending transactions for duplicate
        for pending_tx in self.pending_transactions:
            if hasattr(pending_tx, 'data'):
                pending_hash = pending_tx.data.get('content_hash', '')
                if pending_hash == content_hash:
                    return ""  # Already pending

        tx = create_knowledge_tx(
            content=content,
            summary=summary,
            sender=sender,
            confidence=confidence,
            source_type=source_type,
        )
        return self.add_transaction(tx)

    def create_block(
        self,
        pob_proofs: List[Dict],
    ) -> Optional[Block]:
        """
        Create a new block from pending transactions.

        Requires triadic PoB proofs for validation.

        Args:
            pob_proofs: List of at least 3 Proof-of-Boundary proofs

        Returns:
            New block if created, None if no transactions pending
        """
        if not self.pending_transactions:
            return None

        if len(pob_proofs) < 3:
            raise ValueError("Need at least 3 PoB proofs for triadic consensus")

        # Get previous block
        previous_block = self.blocks[-1]

        # Create header
        header = BlockHeader(
            index=len(self.blocks),
            timestamp=time.time(),
            previous_hash=previous_block.hash,
            merkle_root="",  # Will be computed
            pob_proofs=pob_proofs,
            nonce=int(time.time() * PHI) % ABHI_AMU,
        )

        # Create block with pending transactions
        tx_dicts = [tx.to_dict() for tx in self.pending_transactions]

        block = Block(
            header=header,
            transactions=tx_dicts,
        )

        # Compute Merkle root
        block.header.merkle_root = block.compute_merkle_root()
        block.hash = block.compute_hash()

        return block

    def add_block(
        self,
        block: Optional[Block] = None,
        pob_proofs: Optional[List[Dict]] = None,
    ) -> bool:
        """
        Add a block to the chain.

        Either provide a block directly, or provide PoB proofs
        to create a block from pending transactions.

        SECURITY FIXES (Feb 2026 Audit):
        - Check for proof replay attacks
        - Validate proofs are bound to this block

        Returns True if block was added.
        """
        import hashlib

        if block is None:
            if pob_proofs is None:
                raise ValueError("Must provide either block or pob_proofs")
            block = self.create_block(pob_proofs)

            if block is None:
                return False

        # SECURITY FIX: Check for replay attack (reused proofs)
        for proof in block.header.pob_proofs:
            # Create unique hash for this proof
            proof_data = f"{proof.get('node_id', '')}:{proof.get('alpha', 0)}:{proof.get('omega', 0)}:{proof.get('delta', 0)}:{proof.get('signature', '')}"
            proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()

            if proof_hash in self.used_proof_hashes:
                return False  # REPLAY DETECTED - reject block

        # Validate block
        previous_block = self.blocks[-1] if self.blocks else None
        if not block.validate(previous_block):
            return False

        # SECURITY FIX: Mark proofs as used (prevent replay)
        for proof in block.header.pob_proofs:
            proof_data = f"{proof.get('node_id', '')}:{proof.get('alpha', 0)}:{proof.get('omega', 0)}:{proof.get('delta', 0)}:{proof.get('signature', '')}"
            proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()
            self.used_proof_hashes.add(proof_hash)

        # Add to chain
        self._add_block_internal(block)

        # Clear pending transactions that were included
        included_hashes = {
            tx.get('hash', '') for tx in block.transactions
            if isinstance(tx, dict)
        }
        self.pending_transactions = [
            tx for tx in self.pending_transactions
            if tx.hash not in included_hashes
        ]

        # Save
        if self.auto_save:
            self._save()

        return True

    def get_block(self, index: int) -> Optional[Block]:
        """Get block by index."""
        if 0 <= index < len(self.blocks):
            return self.blocks[index]
        return None

    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Get block by hash."""
        return self.block_by_hash.get(block_hash)

    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction by hash."""
        return self.tx_by_hash.get(tx_hash)

    def get_latest_block(self) -> Block:
        """Get the latest block."""
        return self.blocks[-1]

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for knowledge in the chain.

        Simple keyword search for now.
        """
        results = []

        for block in reversed(self.blocks):
            for tx in block.transactions:
                if isinstance(tx, dict):
                    tx_type = tx.get('tx_type', tx.get('type', ''))
                    if tx_type in [TransactionType.KNOWLEDGE.value, 'knowledge']:
                        data = tx.get('data', tx)
                        summary = data.get('content_summary', data.get('message', ''))
                        if query.lower() in summary.lower():
                            results.append({
                                'block': block.header.index,
                                'tx_hash': tx.get('hash', ''),
                                'summary': summary,
                                'confidence': data.get('confidence', 1.0),
                                'alpha_seed': data.get('alpha_seed', False),
                            })
                            if len(results) >= limit:
                                return results

        return results

    def validate_chain(self) -> bool:
        """Validate the entire chain."""
        for i, block in enumerate(self.blocks):
            previous = self.blocks[i - 1] if i > 0 else None
            if not block.validate(previous):
                return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return {
            'height': self.stats.height,
            'total_transactions': self.stats.total_transactions,
            'knowledge_attestations': self.stats.knowledge_attestations,
            'learning_records': self.stats.learning_records,
            'consensus_votes': self.stats.consensus_votes,
            'identity_registrations': self.stats.identity_registrations,
            'alpha_seeds': self.stats.alpha_seeds,
            'total_pob_proofs': self.stats.total_pob_proofs,
            'pending_transactions': len(self.pending_transactions),
            'valid': self.validate_chain(),
        }

    def __iter__(self) -> Iterator[Block]:
        """Iterate over blocks."""
        return iter(self.blocks)

    def __len__(self) -> int:
        """Number of blocks."""
        return len(self.blocks)

    def _save(self):
        """Save chain to disk with file locking to prevent concurrent write conflicts."""
        chain_file = self.data_dir / "chain.json"
        lock_file = self.data_dir / "chain.lock"

        data = {
            'blocks': [b.to_dict() for b in self.blocks],
            'pending': [tx.to_dict() for tx in self.pending_transactions],
            'saved_at': time.time(),
        }

        # Use exclusive file lock to prevent race conditions between --sync and --mine
        try:
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                try:
                    with open(chain_file, 'w') as f:
                        json.dump(data, f, indent=2)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)  # Release lock
        except (IOError, OSError):
            # Fallback: write without lock (better than failing)
            with open(chain_file, 'w') as f:
                json.dump(data, f, indent=2)

    def _load(self):
        """Load chain from disk with file locking to prevent read during write."""
        chain_file = self.data_dir / "chain.json"
        lock_file = self.data_dir / "chain.lock"

        # Use shared lock for reading (allows multiple readers, blocks writers)
        try:
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_SH)  # Shared lock
                try:
                    with open(chain_file, 'r') as f:
                        data = json.load(f)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)  # Release lock
        except (IOError, OSError):
            # Fallback: read without lock
            with open(chain_file, 'r') as f:
                data = json.load(f)

        # Restore blocks
        for block_data in data.get('blocks', []):
            block = Block.from_dict(block_data)
            self._add_block_internal(block)

        # Restore pending
        for tx_data in data.get('pending', []):
            tx = Transaction.from_dict(tx_data)
            self.pending_transactions.append(tx)

    def print_chain(self, last_n: int = 5):
        """Print chain summary."""
        print("\n" + "=" * 60)
        print("  DARMIYAN CHAIN")
        print("=" * 60)
        print(f"  Height: {len(self.blocks)} blocks")
        print(f"  Transactions: {self.stats.total_transactions}")
        print(f"  Knowledge: {self.stats.knowledge_attestations}")
        print(f"  α-SEEDs: {self.stats.alpha_seeds}")
        print(f"  Pending: {len(self.pending_transactions)}")
        print("-" * 60)

        start = max(0, len(self.blocks) - last_n)
        for block in self.blocks[start:]:
            print(f"  Block #{block.header.index}: {block.hash[:24]}...")
            print(f"    Transactions: {len(block.transactions)}")
            if block.header.pob_proofs:
                avg_ratio = sum(
                    p.get('ratio', PHI_4) for p in block.header.pob_proofs
                ) / len(block.header.pob_proofs)
                print(f"    PoB Avg Ratio: {avg_ratio:.3f}")
        print("=" * 60)


def create_chain(data_dir: Optional[str] = None) -> DarmiyanChain:
    """Create a new Darmiyan chain."""
    return DarmiyanChain(data_dir=data_dir)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  DARMIYAN CHAIN TEST")
    print("=" * 60)
    print()

    # Create chain in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        chain = create_chain(data_dir=tmpdir)

        print(f"Chain initialized with genesis block")
        print(f"  Genesis hash: {chain.blocks[0].hash[:32]}...")
        print()

        # Add some knowledge
        print("Adding knowledge attestations...")
        tx1 = chain.add_knowledge(
            content="φ (phi) is the golden ratio = 1.618033988749895",
            summary="Golden ratio definition",
            sender="test_node",
            confidence=0.95,
        )
        print(f"  Added: {tx1[:16]}...")

        tx2 = chain.add_knowledge(
            content="Proof-of-Boundary uses P/G ≈ φ⁴ for validation",
            summary="PoB mechanism",
            sender="test_node",
            confidence=0.9,
        )
        print(f"  Added: {tx2[:16]}...")

        tx3 = chain.add_knowledge(
            content="Darmiyan means 'in between' - the space where understanding happens",
            summary="Darmiyan etymology",
            sender="test_node",
            confidence=0.85,
        )
        print(f"  Added: {tx3[:16]}...")
        print()

        # Create block with mock PoB proofs
        print("Creating block with PoB proofs...")
        mock_proofs = [
            {'alpha': 200, 'omega': 300, 'delta': 100, 'ratio': PHI_4, 'valid': True, 'node_id': 'node_a'},
            {'alpha': 150, 'omega': 350, 'delta': 200, 'ratio': PHI_4, 'valid': True, 'node_id': 'node_b'},
            {'alpha': 180, 'omega': 320, 'delta': 140, 'ratio': PHI_4, 'valid': True, 'node_id': 'node_c'},
        ]

        success = chain.add_block(pob_proofs=mock_proofs)
        print(f"  Block added: {success}")
        print()

        # Search
        print("Searching for 'golden'...")
        results = chain.search_knowledge("golden")
        for r in results:
            print(f"  Block #{r['block']}: {r['summary']}")
        print()

        # Print chain
        chain.print_chain()

        # Stats
        print("\nStats:", chain.get_stats())

    print("\n  Darmiyan Chain test complete!")
