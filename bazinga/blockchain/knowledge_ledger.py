"""
BAZINGA Knowledge Ledger - AI contributions on-chain
=====================================================

Every knowledge contribution becomes a transaction.
NOT storing the knowledge itself on-chain (too large).
Storing: hash of contribution + metadata + provenance.

"Who contributed WHAT, WHEN, and HOW VALUABLE."

Transaction format:
{
    sender: node_address,
    type: "KNOWLEDGE_CONTRIBUTION",
    payload_hash: SHA256(embedding_vector × φ),
    payload_type: "embedding" | "gradient" | "pattern" | "answer",
    coherence_score: float,  # φ-coherence of contribution
}

The actual knowledge lives in ChromaDB / local KB.
The blockchain just proves provenance and quality.

Author: Space (Abhishek/Abhilasia) 
License: MIT
"""

import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Import constants
try:
    from ..constants import PHI, ALPHA, ABHI_AMU
except ImportError:
    PHI = 1.618033988749895
    ALPHA = 137.035999084
    ABHI_AMU = 515

# φ⁻¹ threshold for coherence
PHI_INVERSE = 1.0 / PHI  # 0.618...


@dataclass
class KnowledgeContribution:
    """A knowledge contribution recorded on-chain."""

    contributor: str  # Node address
    contribution_type: str  # embedding, gradient, pattern, answer
    payload_hash: str  # SHA256(content × φ)
    coherence_score: float  # φ-coherence (0-1)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Provenance chain
    derived_from: Optional[str] = None  # Hash of parent contribution
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeContribution':
        """Create from dictionary."""
        return cls(**data)

    def get_credit_value(self) -> float:
        """
        Credit value based on type and coherence.

        Base rates (from Claude Web's plan):
        - PoB success: 1 credit
        - Knowledge contribution: φ credits (1.618)
        - Gradient validation: φ² credits (2.618)

        Scaled by coherence score.
        """
        base_rates = {
            "embedding": PHI,      # 1.618
            "pattern": PHI,        # 1.618
            "answer": PHI,         # 1.618
            "gradient": PHI ** 2,  # 2.618
        }
        base = base_rates.get(self.contribution_type, 1.0)
        return base * self.coherence_score


class PhiCoherenceFilter:
    """
    Simple φ-coherence calculator for contributions.

    Coherence = how well the contribution aligns with
    golden ratio principles (pattern recognition).
    """

    def score(self, content: Any) -> float:
        """
        Calculate φ-coherence of content.

        Uses entropy-based scoring scaled by φ.
        Real implementations would use embeddings.
        """
        if content is None:
            return 0.0

        # Convert to string for hashing
        content_str = str(content)
        if not content_str:
            return 0.0

        # Simple entropy-based coherence
        # More sophisticated implementations use semantic analysis
        hash_bytes = hashlib.sha256(content_str.encode()).digest()

        # Count unique byte values (entropy proxy)
        unique_bytes = len(set(hash_bytes))
        entropy = unique_bytes / 256.0

        # Scale by φ-function: coherent content has mid-range entropy
        # Not too random (noise), not too uniform (trivial)
        target = PHI_INVERSE  # 0.618
        distance = abs(entropy - target)
        coherence = 1.0 - (distance / target)

        return max(0.0, min(1.0, coherence))

    def is_coherent(self, content: Any) -> bool:
        """Check if content exceeds φ⁻¹ threshold."""
        return self.score(content) >= PHI_INVERSE


class KnowledgeLedger:
    """
    Tracks knowledge contributions as on-chain transactions.

    This is the bridge between:
    - AI layer (embeddings, patterns, gradients)
    - Blockchain layer (permanent provenance record)

    Every valuable contribution gets:
    1. Hashed (φ-scaled)
    2. Coherence-checked
    3. Recorded on-chain
    4. Credited to contributor
    """

    def __init__(self, chain=None):
        """
        Initialize Knowledge Ledger.

        Args:
            chain: Optional DarmiyanChain instance
        """
        self.chain = chain
        self.phi_filter = PhiCoherenceFilter()
        self.contributions: List[KnowledgeContribution] = []
        self.contributor_credits: Dict[str, float] = {}

    def compute_payload_hash(self, content: Any) -> str:
        """
        Hash content with φ-scaling.

        The hash includes φ multiplication to ensure
        the hash space is φ-distributed.
        """
        # Convert content to hashable string
        content_str = str(content)

        # Apply φ-scaling
        base_hash = hashlib.sha256(content_str.encode()).hexdigest()
        scaled = f"{base_hash}:{PHI:.15f}"

        # Final hash
        return hashlib.sha256(scaled.encode()).hexdigest()

    def record_contribution(
        self,
        contributor: str,
        content: Any,
        contribution_type: str = "knowledge",
        metadata: Optional[Dict[str, Any]] = None,
        derived_from: Optional[str] = None
    ) -> Optional[KnowledgeContribution]:
        """
        Record a knowledge contribution.

        Args:
            contributor: Node address/ID
            content: The actual content (embedding, pattern, etc.)
            contribution_type: Type of contribution
            metadata: Additional metadata
            derived_from: Hash of parent contribution (for versioning)

        Returns:
            KnowledgeContribution if accepted, None if rejected
        """
        # Check φ-coherence
        coherence = self.phi_filter.score(content)

        if coherence < PHI_INVERSE:
            # Below φ⁻¹ (0.618) = not coherent enough
            return None

        # Compute payload hash
        payload_hash = self.compute_payload_hash(content)

        # Create contribution record
        contribution = KnowledgeContribution(
            contributor=contributor,
            contribution_type=contribution_type,
            payload_hash=payload_hash,
            coherence_score=coherence,
            metadata=metadata or {},
            derived_from=derived_from
        )

        # Store locally
        self.contributions.append(contribution)

        # Credit contributor
        credit = contribution.get_credit_value()
        self.contributor_credits[contributor] = \
            self.contributor_credits.get(contributor, 0.0) + credit

        # If chain is connected, submit as transaction
        if self.chain is not None:
            self._submit_to_chain(contribution)

        return contribution

    def _submit_to_chain(self, contribution: KnowledgeContribution) -> bool:
        """Submit contribution as blockchain transaction."""
        try:
            # Import transaction type
            from .transaction import KnowledgeAttestation

            # Create attestation
            attestation = KnowledgeAttestation(
                knowledge_hash=contribution.payload_hash,
                knowledge_type=contribution.contribution_type,
                phi_signature=contribution.coherence_score,
                attester=contribution.contributor,
                metadata={
                    "coherence": contribution.coherence_score,
                    "timestamp": contribution.timestamp,
                    **contribution.metadata
                }
            )

            # Add to chain
            self.chain.add_knowledge(attestation)
            return True

        except Exception as e:
            # Chain submission failed, but local record kept
            return False

    def get_contributor_credits(self, contributor: str) -> float:
        """Get total credits earned by a contributor."""
        return self.contributor_credits.get(contributor, 0.0)

    def get_contributions_by_type(
        self,
        contribution_type: str
    ) -> List[KnowledgeContribution]:
        """Get all contributions of a specific type."""
        return [
            c for c in self.contributions
            if c.contribution_type == contribution_type
        ]

    def get_contribution_by_hash(
        self,
        payload_hash: str
    ) -> Optional[KnowledgeContribution]:
        """Find contribution by payload hash."""
        for c in self.contributions:
            if c.payload_hash == payload_hash:
                return c
        return None

    def get_provenance_chain(
        self,
        payload_hash: str
    ) -> List[KnowledgeContribution]:
        """
        Get the full provenance chain for a contribution.

        Traces back through derived_from links.
        """
        chain = []
        current = self.get_contribution_by_hash(payload_hash)

        while current is not None:
            chain.append(current)
            if current.derived_from:
                current = self.get_contribution_by_hash(current.derived_from)
            else:
                current = None

        return chain

    def verify_provenance(self, payload_hash: str) -> Dict[str, Any]:
        """
        Verify the provenance of a contribution.

        Returns verification details including:
        - Chain depth
        - All contributors
        - Total coherence
        """
        chain = self.get_provenance_chain(payload_hash)

        if not chain:
            return {
                "valid": False,
                "reason": "Contribution not found"
            }

        contributors = list(set(c.contributor for c in chain))
        total_coherence = sum(c.coherence_score for c in chain) / len(chain)

        return {
            "valid": True,
            "depth": len(chain),
            "contributors": contributors,
            "avg_coherence": total_coherence,
            "original_hash": chain[-1].payload_hash if chain else None,
            "latest_hash": chain[0].payload_hash if chain else None
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        type_counts = {}
        for c in self.contributions:
            type_counts[c.contribution_type] = \
                type_counts.get(c.contribution_type, 0) + 1

        total_credits = sum(self.contributor_credits.values())

        return {
            "total_contributions": len(self.contributions),
            "contributors": len(self.contributor_credits),
            "total_credits_issued": total_credits,
            "by_type": type_counts,
            "avg_coherence": sum(c.coherence_score for c in self.contributions) / len(self.contributions) if self.contributions else 0.0
        }


def create_ledger(chain=None) -> KnowledgeLedger:
    """Create a new Knowledge Ledger."""
    return KnowledgeLedger(chain=chain)


# CLI integration
def show_ledger_status(ledger: KnowledgeLedger) -> None:
    """Display ledger status for CLI."""
    stats = ledger.get_stats()

    print(f"\n📚 BAZINGA Knowledge Ledger")
    print(f"   Contributions: {stats['total_contributions']}")
    print(f"   Contributors: {stats['contributors']}")
    print(f"   Credits Issued: {stats['total_credits_issued']:.3f}")
    print(f"   Avg Coherence: {stats['avg_coherence']:.3f}")

    if stats['by_type']:
        print(f"\n   By Type:")
        for typ, count in stats['by_type'].items():
            print(f"     {typ}: {count}")

    print()
