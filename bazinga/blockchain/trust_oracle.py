#!/usr/bin/env python3
"""
BAZINGA Trust Oracle - Bridge Between Chain and AI
====================================================

The Trust Oracle reads the blockchain and computes trust scores
for each node. These scores feed into AI routing decisions.

"Trust is EARNED through understanding, not bought."

How it works:
1. Read all PoB proofs from the chain for a node
2. Weight by recency (recent proofs matter more)
3. Apply φ-decay: score = Σ(success × φ^(-age)) / Σ(φ^(-age))
4. Feed trust scores to AI routing layer

This creates the connection:
  Blockchain (proofs) → Trust Oracle → AI Router → Better responses

A node that consistently proves boundary gets higher trust.
A node that fails proofs loses trust over time.
Trust decays with φ if you stop contributing.

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Import constants
try:
    from ..darmiyan.constants import PHI, PHI_INVERSE, ALPHA_INVERSE
except ImportError:
    PHI = 1.618033988749895
    PHI_INVERSE = 0.6180339887498948
    ALPHA_INVERSE = 137


@dataclass
class TrustRecord:
    """Record of a node's trust-relevant activity."""
    node_address: str
    block_number: int
    timestamp: float
    activity_type: str  # "pob", "knowledge", "gradient", "inference"
    success: bool
    score: float = 1.0  # Activity-specific score (e.g., coherence)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeTrust:
    """Computed trust for a node."""
    node_address: str
    trust_score: float  # 0-1 overall trust
    pob_score: float    # PoB success rate
    contribution_score: float  # Knowledge/gradient contributions
    recency_score: float  # How recently active
    total_activities: int
    last_active_block: int


class TrustOracle:
    """
    Reads blockchain → computes trust → feeds AI routing.

    The oracle maintains trust scores for all nodes based on
    their on-chain activity. These scores determine:
    - Whose knowledge to prioritize
    - Whose gradients to accept
    - Whose responses to trust

    Trust Formula:
        score = Σ (success_i × weight_i × φ^(-age_i/decay_rate)) / Σ (φ^(-age_i/decay_rate))

    Where:
        - success_i: 1 if activity succeeded, 0 otherwise
        - weight_i: Activity weight (PoB=1, knowledge=φ, gradient=φ²)
        - age_i: Blocks since activity
        - decay_rate: 100 blocks (configurable)
    """

    # Activity weights (some contributions worth more)
    WEIGHT_POB = 1.0           # Proving boundary
    WEIGHT_KNOWLEDGE = PHI     # Contributing knowledge (1.618)
    WEIGHT_GRADIENT = PHI ** 2 # Validating gradients (2.618)
    WEIGHT_INFERENCE = PHI_INVERSE  # Providing inference (0.618)

    # Trust thresholds
    THRESHOLD_TRUSTED = 0.7    # Above this = trusted node
    THRESHOLD_NEUTRAL = 0.3    # Above this = neutral
    # Below 0.3 = untrusted

    def __init__(
        self,
        chain=None,
        decay_rate: int = 100,  # Blocks for φ-decay
        min_activities: int = 3,  # Minimum activities for trust
    ):
        self.chain = chain
        self.decay_rate = decay_rate
        self.min_activities = min_activities

        # Cache of trust records by node
        self.records: Dict[str, List[TrustRecord]] = defaultdict(list)

        # Cache of computed trust scores
        self.trust_cache: Dict[str, NodeTrust] = {}
        self.cache_block: int = 0  # Block at which cache was computed

    def set_chain(self, chain):
        """Set the blockchain to read from."""
        self.chain = chain
        self.invalidate_cache()

    def invalidate_cache(self):
        """Invalidate trust cache (call when chain updates)."""
        self.trust_cache = {}
        self.cache_block = 0

    def record_activity(
        self,
        node_address: str,
        activity_type: str,
        success: bool,
        block_number: int = 0,
        score: float = 1.0,
        metadata: Dict = None,
    ):
        """
        Record an activity for a node.

        Called when:
        - Node generates PoB (from darmiyan/protocol.py)
        - Node contributes knowledge (from blockchain/chain.py)
        - Node validates gradient (from federated/aggregator.py)
        - Node provides inference (from inference/distributed.py)
        """
        if block_number == 0:
            block_number = len(self.chain.blocks) if self.chain else 0

        record = TrustRecord(
            node_address=node_address,
            block_number=block_number,
            timestamp=time.time(),
            activity_type=activity_type,
            success=success,
            score=score,
            metadata=metadata or {},
        )

        self.records[node_address].append(record)
        self.invalidate_cache()

        return record

    def get_trust_score(self, node_address: str) -> float:
        """
        Get the trust score for a node.

        Returns a value between 0 and 1:
        - 1.0 = Fully trusted (consistent successful proofs)
        - 0.5 = Neutral (new node or mixed record)
        - 0.0 = Untrusted (consistent failures)
        """
        trust = self.get_node_trust(node_address)
        return trust.trust_score if trust else 0.5

    def get_node_trust(self, node_address: str) -> Optional[NodeTrust]:
        """
        Get full trust details for a node.

        Returns NodeTrust with:
        - trust_score: Overall 0-1 score
        - pob_score: PoB success rate
        - contribution_score: Knowledge/gradient contributions
        - recency_score: How recently active
        """
        # Check cache
        current_block = len(self.chain.blocks) if self.chain else 0
        if node_address in self.trust_cache and self.cache_block == current_block:
            return self.trust_cache[node_address]

        # Get records for this node
        records = self.records.get(node_address, [])

        # Also scan chain for on-chain records
        if self.chain:
            chain_records = self._scan_chain_for_node(node_address)
            records = records + chain_records

        if not records:
            return None

        # Compute trust
        trust = self._compute_trust(node_address, records, current_block)

        # Cache
        self.trust_cache[node_address] = trust
        self.cache_block = current_block

        return trust

    def _scan_chain_for_node(self, node_address: str) -> List[TrustRecord]:
        """Scan blockchain for records of this node's activity."""
        records = []

        for block in self.chain.blocks:
            # Check PoB proofs
            for proof in block.header.pob_proofs:
                if proof.get('node_id', '').startswith(node_address[:12]):
                    records.append(TrustRecord(
                        node_address=node_address,
                        block_number=block.header.index,
                        timestamp=block.header.timestamp,
                        activity_type='pob',
                        success=proof.get('valid', False),
                        score=1.0 if proof.get('valid') else 0.0,
                    ))

            # Check transactions
            for tx in block.transactions:
                if isinstance(tx, dict):
                    sender = tx.get('sender', '')
                    if sender == node_address or sender.startswith(node_address[:12]):
                        tx_type = tx.get('tx_type', tx.get('type', ''))

                        if tx_type == 'knowledge':
                            records.append(TrustRecord(
                                node_address=node_address,
                                block_number=block.header.index,
                                timestamp=tx.get('timestamp', block.header.timestamp),
                                activity_type='knowledge',
                                success=True,
                                score=tx.get('data', {}).get('confidence', 0.8),
                            ))
                        elif tx_type == 'learning':
                            records.append(TrustRecord(
                                node_address=node_address,
                                block_number=block.header.index,
                                timestamp=tx.get('timestamp', block.header.timestamp),
                                activity_type='gradient',
                                success=True,
                                score=tx.get('data', {}).get('phi_weight', 0.618),
                            ))

        return records

    def _compute_trust(
        self,
        node_address: str,
        records: List[TrustRecord],
        current_block: int,
    ) -> NodeTrust:
        """Compute trust score from records."""
        if not records:
            return NodeTrust(
                node_address=node_address,
                trust_score=0.5,
                pob_score=0.5,
                contribution_score=0.0,
                recency_score=0.0,
                total_activities=0,
                last_active_block=0,
            )

        # Separate by type
        pob_records = [r for r in records if r.activity_type == 'pob']
        knowledge_records = [r for r in records if r.activity_type == 'knowledge']
        gradient_records = [r for r in records if r.activity_type == 'gradient']
        inference_records = [r for r in records if r.activity_type == 'inference']

        # Compute φ-weighted scores
        def weighted_score(recs: List[TrustRecord], weight: float) -> float:
            if not recs:
                return 0.0

            weighted_sum = 0.0
            weight_total = 0.0

            for r in recs:
                age = max(0, current_block - r.block_number)
                decay = PHI ** (-age / self.decay_rate)

                weighted_sum += r.success * r.score * weight * decay
                weight_total += weight * decay

            return weighted_sum / weight_total if weight_total > 0 else 0.0

        # Compute component scores
        pob_score = weighted_score(pob_records, self.WEIGHT_POB)
        knowledge_score = weighted_score(knowledge_records, self.WEIGHT_KNOWLEDGE)
        gradient_score = weighted_score(gradient_records, self.WEIGHT_GRADIENT)
        inference_score = weighted_score(inference_records, self.WEIGHT_INFERENCE)

        # Contribution score (knowledge + gradients)
        contribution_score = (knowledge_score + gradient_score) / 2 if (knowledge_records or gradient_records) else 0.0

        # Recency score (how recently active)
        last_block = max(r.block_number for r in records)
        blocks_since = current_block - last_block
        recency_score = PHI ** (-blocks_since / self.decay_rate)

        # Overall trust score
        # Weight: PoB matters most, then contributions, then recency
        if len(records) < self.min_activities:
            # Not enough data - neutral trust
            trust_score = 0.5
        else:
            trust_score = (
                pob_score * 0.5 +           # 50% from PoB
                contribution_score * 0.3 +   # 30% from contributions
                recency_score * 0.2          # 20% from recency
            )

        return NodeTrust(
            node_address=node_address,
            trust_score=trust_score,
            pob_score=pob_score,
            contribution_score=contribution_score,
            recency_score=recency_score,
            total_activities=len(records),
            last_active_block=last_block,
        )

    def get_trusted_nodes(self, min_trust: float = None) -> List[NodeTrust]:
        """Get all nodes above trust threshold."""
        if min_trust is None:
            min_trust = self.THRESHOLD_TRUSTED

        trusted = []

        # Get all known nodes
        all_nodes = set(self.records.keys())
        if self.chain:
            for block in self.chain.blocks:
                for proof in block.header.pob_proofs:
                    node_id = proof.get('node_id', '')
                    if node_id:
                        all_nodes.add(node_id)

        # Compute trust for each
        for node in all_nodes:
            trust = self.get_node_trust(node)
            if trust and trust.trust_score >= min_trust:
                trusted.append(trust)

        # Sort by trust score descending
        trusted.sort(key=lambda t: t.trust_score, reverse=True)

        return trusted

    def is_trusted(self, node_address: str) -> bool:
        """Check if a node is trusted."""
        return self.get_trust_score(node_address) >= self.THRESHOLD_TRUSTED

    def get_routing_weight(self, node_address: str) -> float:
        """
        Get routing weight for AI inference.

        Higher trust = more likely to route queries to this node.
        Used by inference/model_router.py.
        """
        trust = self.get_trust_score(node_address)

        # Scale trust to routing weight
        # Trust 0.5 → weight 0.5
        # Trust 1.0 → weight 1.0
        # Trust 0.0 → weight 0.1 (never fully exclude)
        return max(0.1, trust)

    def get_gradient_acceptance_threshold(self, node_address: str) -> float:
        """
        Get acceptance threshold for federated gradients.

        Higher trust = lower threshold (more lenient).
        Used by federated/aggregator.py.
        """
        trust = self.get_trust_score(node_address)

        # Trust 1.0 → threshold 0.3 (accept easily)
        # Trust 0.5 → threshold 0.6 (moderate scrutiny)
        # Trust 0.0 → threshold 0.9 (high scrutiny)
        return 0.9 - (trust * 0.6)

    def get_stats(self) -> Dict[str, Any]:
        """Get oracle statistics."""
        all_nodes = set(self.records.keys())

        trusted_count = sum(1 for n in all_nodes if self.is_trusted(n))
        total_records = sum(len(r) for r in self.records.values())

        return {
            'total_nodes': len(all_nodes),
            'trusted_nodes': trusted_count,
            'total_records': total_records,
            'decay_rate': self.decay_rate,
            'cache_valid': self.cache_block == (len(self.chain.blocks) if self.chain else 0),
        }

    def print_status(self):
        """Print oracle status."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("  BAZINGA TRUST ORACLE")
        print("=" * 60)
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Trusted Nodes: {stats['trusted_nodes']}")
        print(f"  Total Records: {stats['total_records']}")
        print(f"  φ-Decay Rate: {stats['decay_rate']} blocks")
        print("-" * 60)

        # Show top trusted nodes
        trusted = self.get_trusted_nodes()[:5]
        if trusted:
            print("  Top Trusted Nodes:")
            for t in trusted:
                print(f"    {t.node_address[:16]}... : {t.trust_score:.3f}")
        print("=" * 60)


# Convenience function
def create_trust_oracle(chain=None) -> TrustOracle:
    """Create a new trust oracle."""
    return TrustOracle(chain=chain)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA TRUST ORACLE TEST")
    print("=" * 60)
    print()

    import tempfile
    from .chain import create_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create chain
        chain = create_chain(data_dir=tmpdir)

        # Create oracle
        oracle = create_trust_oracle(chain)

        # Record some activities
        print("Recording activities for test nodes...")

        # Node A: Good node (mostly successes)
        for i in range(10):
            oracle.record_activity(
                node_address="node_good_001",
                activity_type="pob",
                success=True,
                block_number=i,
            )
        oracle.record_activity(
            node_address="node_good_001",
            activity_type="knowledge",
            success=True,
            score=0.9,
            block_number=10,
        )

        # Node B: Mixed node
        for i in range(5):
            oracle.record_activity(
                node_address="node_mixed_002",
                activity_type="pob",
                success=i % 2 == 0,  # 50% success
                block_number=i,
            )

        # Node C: Bad node (failures)
        for i in range(5):
            oracle.record_activity(
                node_address="node_bad_003",
                activity_type="pob",
                success=False,
                block_number=i,
            )

        print()

        # Get trust scores
        print("Trust Scores:")
        for node in ["node_good_001", "node_mixed_002", "node_bad_003", "node_unknown"]:
            trust = oracle.get_node_trust(node)
            if trust:
                print(f"  {node}:")
                print(f"    Trust: {trust.trust_score:.3f}")
                print(f"    PoB: {trust.pob_score:.3f}")
                print(f"    Contribution: {trust.contribution_score:.3f}")
                print(f"    Activities: {trust.total_activities}")
            else:
                print(f"  {node}: No data (trust=0.5)")
        print()

        # Check routing weights
        print("Routing Weights:")
        for node in ["node_good_001", "node_mixed_002", "node_bad_003"]:
            weight = oracle.get_routing_weight(node)
            print(f"  {node}: {weight:.3f}")
        print()

        # Show status
        oracle.print_status()

    print("\n  Trust Oracle test complete!")
