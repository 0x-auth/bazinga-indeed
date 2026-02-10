#!/usr/bin/env python3
"""
BAZINGA Transactions - Knowledge Attestations
==============================================

Transactions in the Darmiyan chain are NOT money transfers.
They are KNOWLEDGE ATTESTATIONS.

Transaction Types:
1. KnowledgeAttestation - Verify understanding
2. LearningRecord - Record federated learning contribution
3. ConsensusVote - Vote on network decisions
4. IdentityRegistration - Register node identity

"You can't buy understanding. You can only ATTEST to it."
"""

import json
import hashlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import constants
try:
    from ..darmiyan.constants import PHI, ABHI_AMU, ALPHA_INVERSE
except ImportError:
    PHI = 1.618033988749895
    ABHI_AMU = 515
    ALPHA_INVERSE = 137


class TransactionType(Enum):
    """Types of transactions in the Darmiyan chain."""
    KNOWLEDGE = "knowledge"           # Knowledge attestation
    LEARNING = "learning"             # Federated learning record
    CONSENSUS = "consensus"           # Consensus vote
    IDENTITY = "identity"             # Identity registration
    GENESIS = "genesis"               # Genesis transaction


@dataclass
class Transaction:
    """
    Base transaction in the Darmiyan chain.

    All transactions have:
    - type: What kind of transaction
    - sender: Who created it (node_id)
    - timestamp: When it was created
    - data: Transaction-specific payload
    - signature: Cryptographic signature
    - hash: Transaction identifier
    """
    tx_type: str
    sender: str
    timestamp: float
    data: Dict[str, Any]
    signature: str = ""
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute transaction hash."""
        payload = {
            'type': self.tx_type,
            'sender': self.sender,
            'timestamp': self.timestamp,
            'data': self.data,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def sign(self, private_key: str):
        """Sign the transaction with private key."""
        # Simple signature for now (would use real crypto in production)
        sign_data = self.hash + private_key
        self.signature = hashlib.sha256(sign_data.encode()).hexdigest()

    def verify_signature(self, public_key: str) -> bool:
        """Verify transaction signature."""
        # Simple verification (would use real crypto in production)
        expected = hashlib.sha256((self.hash + public_key).encode()).hexdigest()
        return self.signature == expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_type': self.tx_type,
            'sender': self.sender,
            'timestamp': self.timestamp,
            'data': self.data,
            'signature': self.signature,
            'hash': self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        return cls(**data)


@dataclass
class KnowledgeAttestation:
    """
    Attest that a piece of knowledge has been verified.

    This is the core transaction type. When BAZINGA generates
    understanding, nodes attest to it. The attestation is
    permanently recorded on the chain.

    Fields:
    - content_hash: Hash of the knowledge being attested
    - content_summary: Brief description of what's attested
    - confidence: How confident is the attester (0-1)
    - source_type: Where the knowledge came from
    - phi_coherence: φ-coherence score of the understanding
    - alpha_seed: Whether this is an α-seed (hash % 137 == 0)
    """
    content_hash: str
    content_summary: str
    confidence: float
    source_type: str  # "rag", "llm", "human", "consensus"
    phi_coherence: float = 0.0
    alpha_seed: bool = False
    attestor: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Check α-seed status
        if not self.alpha_seed:
            hash_int = int(self.content_hash[:16], 16)
            self.alpha_seed = (hash_int % ALPHA_INVERSE == 0)

    def to_transaction(self, sender: str) -> Transaction:
        """Convert to transaction."""
        return Transaction(
            tx_type=TransactionType.KNOWLEDGE.value,
            sender=sender,
            timestamp=self.timestamp,
            data={
                'content_hash': self.content_hash,
                'content_summary': self.content_summary,
                'confidence': self.confidence,
                'source_type': self.source_type,
                'phi_coherence': self.phi_coherence,
                'alpha_seed': self.alpha_seed,
            }
        )

    @classmethod
    def create(
        cls,
        content: str,
        summary: str,
        confidence: float,
        source_type: str = "rag",
        phi_coherence: float = 0.0,
    ) -> 'KnowledgeAttestation':
        """Create a knowledge attestation."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return cls(
            content_hash=content_hash,
            content_summary=summary,
            confidence=confidence,
            source_type=source_type,
            phi_coherence=phi_coherence,
        )


@dataclass
class LearningRecord:
    """
    Record a federated learning contribution.

    When a node contributes gradients to the network,
    this records that contribution permanently.
    """
    node_id: str
    gradient_hash: str
    training_samples: int
    modules_updated: List[str]
    noise_scale: float  # Differential privacy noise
    phi_weight: float   # φ-weighted contribution
    timestamp: float = field(default_factory=time.time)

    def to_transaction(self, sender: str) -> Transaction:
        """Convert to transaction."""
        return Transaction(
            tx_type=TransactionType.LEARNING.value,
            sender=sender,
            timestamp=self.timestamp,
            data={
                'node_id': self.node_id,
                'gradient_hash': self.gradient_hash,
                'training_samples': self.training_samples,
                'modules_updated': self.modules_updated,
                'noise_scale': self.noise_scale,
                'phi_weight': self.phi_weight,
            }
        )


@dataclass
class ConsensusVote:
    """
    Record a vote in network consensus.

    For governance decisions, proposals, and disputes.
    """
    proposal_hash: str
    proposal_type: str  # "governance", "dispute", "upgrade"
    vote: str          # "accept", "reject", "abstain"
    weight: float       # φ-weighted vote power
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_transaction(self, sender: str) -> Transaction:
        """Convert to transaction."""
        return Transaction(
            tx_type=TransactionType.CONSENSUS.value,
            sender=sender,
            timestamp=self.timestamp,
            data={
                'proposal_hash': self.proposal_hash,
                'proposal_type': self.proposal_type,
                'vote': self.vote,
                'weight': self.weight,
                'reason': self.reason,
            }
        )


@dataclass
class IdentityRegistration:
    """
    Register a node identity on the chain.

    This creates a permanent identity record.
    """
    node_id: str
    public_key: str
    node_type: str  # "full", "light", "validator"
    capabilities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_transaction(self, sender: str) -> Transaction:
        """Convert to transaction."""
        return Transaction(
            tx_type=TransactionType.IDENTITY.value,
            sender=sender,
            timestamp=self.timestamp,
            data={
                'node_id': self.node_id,
                'public_key': self.public_key,
                'node_type': self.node_type,
                'capabilities': self.capabilities,
                'metadata': self.metadata,
            }
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_knowledge_tx(
    content: str,
    summary: str,
    sender: str,
    confidence: float = 0.8,
    source_type: str = "rag",
) -> Transaction:
    """Quick function to create knowledge attestation transaction."""
    attestation = KnowledgeAttestation.create(
        content=content,
        summary=summary,
        confidence=confidence,
        source_type=source_type,
    )
    return attestation.to_transaction(sender)


def create_learning_tx(
    node_id: str,
    gradient_hash: str,
    training_samples: int,
    modules: List[str],
) -> Transaction:
    """Quick function to create learning record transaction."""
    record = LearningRecord(
        node_id=node_id,
        gradient_hash=gradient_hash,
        training_samples=training_samples,
        modules_updated=modules,
        noise_scale=0.1,
        phi_weight=1 / PHI,
    )
    return record.to_transaction(node_id)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DARMIYAN TRANSACTION TEST")
    print("=" * 60)
    print()

    # Create knowledge attestation
    print("Creating Knowledge Attestation...")
    ka = KnowledgeAttestation.create(
        content="The golden ratio φ = 1.618033988749895",
        summary="Mathematical constant definition",
        confidence=0.95,
        source_type="human",
        phi_coherence=0.9,
    )
    print(f"  Content Hash: {ka.content_hash[:32]}...")
    print(f"  α-SEED: {ka.alpha_seed}")
    print(f"  Confidence: {ka.confidence}")
    print()

    # Convert to transaction
    tx = ka.to_transaction(sender="node_test_001")
    print(f"Transaction:")
    print(f"  Type: {tx.tx_type}")
    print(f"  Sender: {tx.sender}")
    print(f"  Hash: {tx.hash[:32]}...")
    print()

    # Create learning record
    print("Creating Learning Record...")
    lr = LearningRecord(
        node_id="node_test_001",
        gradient_hash="abc123" * 10 + "abcd",
        training_samples=100,
        modules_updated=["q_proj", "v_proj"],
        noise_scale=0.1,
        phi_weight=1 / PHI,
    )
    ltx = lr.to_transaction("node_test_001")
    print(f"  Transaction Hash: {ltx.hash[:32]}...")
    print(f"  Samples: {lr.training_samples}")
    print(f"  Modules: {lr.modules_updated}")
    print()

    # Sign and verify
    print("Testing Signature...")
    tx.sign("secret_key_123")
    print(f"  Signature: {tx.signature[:32]}...")
    print(f"  Verified: {tx.verify_signature('secret_key_123')}")
    print(f"  Wrong Key: {tx.verify_signature('wrong_key')}")
    print()

    # Serialize and deserialize
    tx_dict = tx.to_dict()
    tx_restored = Transaction.from_dict(tx_dict)
    print(f"Serialization: {'OK' if tx_restored.hash == tx.hash else 'FAILED'}")

    print()
    print("  Transactions ready for Darmiyan chain!")
