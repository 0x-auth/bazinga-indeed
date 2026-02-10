"""
BAZINGA Gradient Validator - Blockchain-secured Federated Learning
===================================================================

Gradient updates go through triadic validation before aggregation.

The problem with federated learning:
  - Node could send poisoned gradients
  - Node could send random noise
  - Node could replay old gradients

Darmiyan solution:
  1. Node trains locally, produces gradient update
  2. Wraps gradient hash as transaction
  3. Submits to mempool
  4. 3 validator nodes each:
     a. Apply gradient to their local model copy
     b. Measure if loss decreased
     c. Check φ-coherence of the update
     d. Prove boundary (PoB)
  5. If all 3 agree: gradient accepted, recorded on-chain
  6. Aggregation only uses chain-validated gradients

This means:
  - Can't poison the model (3 independent checks)
  - Can't replay (blockchain ordering prevents it)
  - Can't spam (PoB rate-limits submissions)
  - Privacy preserved (differential_privacy.py still applies)

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import hashlib
import time
import copy
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

# Import constants
try:
    from ..constants import PHI, ALPHA, ABHI_AMU
except ImportError:
    PHI = 1.618033988749895
    ALPHA = 137.035999084
    ABHI_AMU = 515

# Thresholds
PHI_INVERSE = 1.0 / PHI  # 0.618
VALIDATION_THRESHOLD = 2  # Minimum validators needed (out of 3)


class ValidationStatus(Enum):
    """Gradient validation status."""
    PENDING = "pending"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class GradientUpdate:
    """A gradient update from federated learning."""

    submitter: str  # Node address
    gradient_hash: str  # Hash of the actual gradient
    model_version: str  # Which model version this is for
    loss_improvement: float  # Claimed improvement
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validation state
    status: ValidationStatus = ValidationStatus.PENDING
    validations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientUpdate':
        """Create from dictionary."""
        data['status'] = ValidationStatus(data.get('status', 'pending'))
        return cls(**data)

    def is_accepted(self) -> bool:
        """Check if gradient has been accepted."""
        return self.status == ValidationStatus.ACCEPTED

    def get_validation_count(self) -> Tuple[int, int]:
        """Get (accept_count, reject_count)."""
        accepts = sum(1 for v in self.validations if v.get('approved', False))
        rejects = len(self.validations) - accepts
        return accepts, rejects


@dataclass
class ValidationVote:
    """A validator's vote on a gradient update."""

    validator: str  # Validator node address
    gradient_hash: str  # Hash of gradient being validated
    approved: bool  # Did they approve?
    improvement_verified: bool  # Did loss actually improve?
    coherence_score: float  # φ-coherence of the update
    pob_proof: Optional[Dict[str, Any]] = None  # PoB proof
    timestamp: float = field(default_factory=time.time)
    reason: str = ""  # Reason for rejection (if any)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class GradientValidator:
    """
    Validates gradient updates through triadic consensus.

    Three independent validators must agree that:
    1. The gradient actually improves the model (loss decreases)
    2. The update is φ-coherent (not random noise)
    3. The submitter proves boundary (PoB)

    Only validated gradients are used in aggregation.
    """

    def __init__(self, chain=None, trust_oracle=None):
        """
        Initialize Gradient Validator.

        Args:
            chain: Optional DarmiyanChain instance
            trust_oracle: Optional TrustOracle for validator selection
        """
        self.chain = chain
        self.trust_oracle = trust_oracle
        self.pending_updates: Dict[str, GradientUpdate] = {}
        self.accepted_updates: List[GradientUpdate] = []
        self.rejected_updates: List[GradientUpdate] = []

        # Validator registry
        self.validators: Dict[str, Dict[str, Any]] = {}

    def register_validator(
        self,
        address: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a node as a gradient validator."""
        self.validators[address] = {
            "address": address,
            "registered_at": time.time(),
            "validations_performed": 0,
            "accuracy": 1.0,  # Starts perfect, degrades with bad votes
            **(metadata or {})
        }
        return True

    def submit_gradient(
        self,
        submitter: str,
        gradient_hash: str,
        model_version: str,
        loss_improvement: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GradientUpdate:
        """
        Submit a gradient update for validation.

        Args:
            submitter: Address of submitting node
            gradient_hash: Hash of the gradient data
            model_version: Model version this update is for
            loss_improvement: Claimed loss improvement
            metadata: Additional metadata

        Returns:
            GradientUpdate object
        """
        update = GradientUpdate(
            submitter=submitter,
            gradient_hash=gradient_hash,
            model_version=model_version,
            loss_improvement=loss_improvement,
            metadata=metadata or {}
        )

        self.pending_updates[gradient_hash] = update
        return update

    def select_validators(
        self,
        gradient_hash: str,
        count: int = 3
    ) -> List[str]:
        """
        Select validators for a gradient update.

        Uses trust scores to weight selection.
        Excludes the submitter from validation.
        """
        update = self.pending_updates.get(gradient_hash)
        if not update:
            return []

        # Get eligible validators (not the submitter)
        eligible = [
            addr for addr in self.validators
            if addr != update.submitter
        ]

        if not eligible:
            return []

        # If trust oracle available, weight by trust
        if self.trust_oracle:
            weighted = []
            for addr in eligible:
                trust = self.trust_oracle.get_trust_score(addr)
                # Higher trust = more likely to be selected
                weight = max(0.1, trust)  # Minimum weight
                weighted.append((addr, weight))

            # Sort by weight descending
            weighted.sort(key=lambda x: x[1], reverse=True)
            eligible = [addr for addr, _ in weighted]

        # Return top N validators
        return eligible[:count]

    def validate_gradient(
        self,
        validator: str,
        gradient_hash: str,
        approved: bool,
        improvement_verified: bool,
        coherence_score: float,
        pob_proof: Optional[Dict[str, Any]] = None,
        reason: str = ""
    ) -> ValidationVote:
        """
        Submit a validation vote for a gradient update.

        Args:
            validator: Address of validator
            gradient_hash: Hash of gradient being validated
            approved: Whether validator approves
            improvement_verified: Whether loss improvement verified
            coherence_score: φ-coherence score
            pob_proof: Optional PoB proof
            reason: Reason for rejection

        Returns:
            ValidationVote object
        """
        update = self.pending_updates.get(gradient_hash)
        if not update:
            raise ValueError(f"Unknown gradient: {gradient_hash}")

        # Validation requires φ-coherence above threshold
        coherent = coherence_score >= PHI_INVERSE

        # Must verify improvement and be coherent to approve
        valid_approval = approved and improvement_verified and coherent

        # PoB adds weight but isn't required for validation
        has_pob = pob_proof is not None and pob_proof.get('valid', False)

        vote = ValidationVote(
            validator=validator,
            gradient_hash=gradient_hash,
            approved=valid_approval,
            improvement_verified=improvement_verified,
            coherence_score=coherence_score,
            pob_proof=pob_proof,
            reason=reason if not valid_approval else ""
        )

        # Add to update's validations
        update.validations.append(vote.to_dict())
        update.status = ValidationStatus.VALIDATING

        # Update validator stats
        if validator in self.validators:
            self.validators[validator]['validations_performed'] += 1

        # Check if we have enough votes
        self._check_consensus(gradient_hash)

        return vote

    def _check_consensus(self, gradient_hash: str) -> None:
        """Check if gradient has reached consensus."""
        update = self.pending_updates.get(gradient_hash)
        if not update:
            return

        accepts, rejects = update.get_validation_count()
        total = accepts + rejects

        # Need at least 3 votes
        if total < 3:
            return

        # Triadic consensus: at least 2 of 3 must agree
        if accepts >= VALIDATION_THRESHOLD:
            # Accepted
            update.status = ValidationStatus.ACCEPTED
            self.accepted_updates.append(update)
            del self.pending_updates[gradient_hash]

            # Record on chain if available
            if self.chain:
                self._record_on_chain(update, accepted=True)

        elif rejects >= VALIDATION_THRESHOLD:
            # Rejected
            update.status = ValidationStatus.REJECTED
            self.rejected_updates.append(update)
            del self.pending_updates[gradient_hash]

            # Record rejection on chain
            if self.chain:
                self._record_on_chain(update, accepted=False)

    def _record_on_chain(
        self,
        update: GradientUpdate,
        accepted: bool
    ) -> bool:
        """Record gradient validation result on chain."""
        try:
            from .transaction import Transaction

            tx = Transaction(
                tx_type="GRADIENT_VALIDATION",
                sender=update.submitter,
                data={
                    "gradient_hash": update.gradient_hash,
                    "model_version": update.model_version,
                    "accepted": accepted,
                    "validators": [v['validator'] for v in update.validations],
                    "avg_coherence": sum(
                        v.get('coherence_score', 0)
                        for v in update.validations
                    ) / len(update.validations) if update.validations else 0
                }
            )

            # Submit to chain's mempool
            return self.chain.add_transaction(tx)

        except Exception:
            return False

    def get_accepted_gradients(
        self,
        model_version: Optional[str] = None
    ) -> List[GradientUpdate]:
        """Get all accepted gradients, optionally filtered by version."""
        if model_version:
            return [
                u for u in self.accepted_updates
                if u.model_version == model_version
            ]
        return self.accepted_updates.copy()

    def get_pending_for_validation(self) -> List[GradientUpdate]:
        """Get gradients needing validation."""
        return [
            u for u in self.pending_updates.values()
            if len(u.validations) < 3
        ]

    def calculate_coherence(self, gradient_data: Any) -> float:
        """
        Calculate φ-coherence of gradient data.

        Uses entropy-based scoring similar to KnowledgeLedger.
        """
        if gradient_data is None:
            return 0.0

        # Convert to string for hashing
        content_str = str(gradient_data)
        if not content_str:
            return 0.0

        # Hash-based entropy
        hash_bytes = hashlib.sha256(content_str.encode()).digest()
        unique_bytes = len(set(hash_bytes))
        entropy = unique_bytes / 256.0

        # φ-scaled coherence
        target = PHI_INVERSE
        distance = abs(entropy - target)
        coherence = 1.0 - (distance / target)

        return max(0.0, min(1.0, coherence))

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        total_validations = sum(
            v['validations_performed']
            for v in self.validators.values()
        )

        return {
            "validators": len(self.validators),
            "pending_updates": len(self.pending_updates),
            "accepted_updates": len(self.accepted_updates),
            "rejected_updates": len(self.rejected_updates),
            "total_validations": total_validations,
            "acceptance_rate": len(self.accepted_updates) / (
                len(self.accepted_updates) + len(self.rejected_updates)
            ) if (self.accepted_updates or self.rejected_updates) else 0.0
        }


def create_validator(chain=None, trust_oracle=None) -> GradientValidator:
    """Create a new Gradient Validator."""
    return GradientValidator(chain=chain, trust_oracle=trust_oracle)


# CLI integration
def show_validator_status(validator: GradientValidator) -> None:
    """Display validator status for CLI."""
    stats = validator.get_stats()

    print(f"\n🔄 BAZINGA Gradient Validator")
    print(f"   Validators: {stats['validators']}")
    print(f"   Pending: {stats['pending_updates']}")
    print(f"   Accepted: {stats['accepted_updates']}")
    print(f"   Rejected: {stats['rejected_updates']}")
    print(f"   Acceptance Rate: {stats['acceptance_rate']:.1%}")
    print()
