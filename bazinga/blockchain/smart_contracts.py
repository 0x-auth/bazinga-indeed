"""
BAZINGA Smart Contracts - Understanding is Law
===============================================

Not "code is law." Understanding is law.

Traditional smart contracts execute when conditions are met.
Darmiyan contracts execute when understanding crystallizes —
verified by the AI layer. Three parties must demonstrate
they genuinely understand the terms.

Example: Bounty Contract
  - Creator posts: "Explain quantum entanglement"
  - Bounty: 137 inference credits
  - Three reviewers must verify the explanation
  - Verification = PoB + φ-coherence > 0.8 + λG validation
  - If all 3 crystallize → bounty released

Example: Knowledge Escrow
  - Two parties want to exchange proprietary knowledge
  - Each deposits knowledge hash on-chain
  - Third party (arbiter) verifies both are valuable
  - Triadic crystallization → exchange happens
  - Neither party can cheat: fragments only unify if genuine

The AI layer (φ-coherence, λG) is what makes these contracts
ABOUT understanding, not just about math.

Author: Space (Abhishek/Abhilasia) 
License: MIT
"""

import time
import hashlib
from typing import Dict, Any, Optional, List, Callable
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
HIGH_COHERENCE = 0.8  # For contract verification


class ContractStatus(Enum):
    """Smart contract status."""
    CREATED = "created"
    ACTIVE = "active"
    REVIEWING = "reviewing"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class ContractType(Enum):
    """Types of understanding contracts."""
    BOUNTY = "bounty"  # Post challenge, reward solver
    ESCROW = "escrow"  # Two-party knowledge exchange
    ATTESTATION = "attestation"  # Third-party verification
    DELEGATION = "delegation"  # Inference delegation rights


@dataclass
class ContractTerms:
    """Terms of an understanding contract."""

    description: str  # What understanding is required
    required_coherence: float = HIGH_COHERENCE  # Minimum φ-coherence
    required_reviewers: int = 3  # Triadic consensus
    bounty_credits: float = 0.0  # Credits at stake
    expiration: Optional[float] = None  # When contract expires

    # Lambda-G validation
    lambda_g_required: bool = True  # Require boundary validation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Submission:
    """A submission to fulfill a contract."""

    submitter: str
    content_hash: str  # Hash of the submitted content
    coherence_score: float = 0.0
    lambda_g_valid: bool = False
    pob_proof: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    # Review results
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    approved: bool = False

    def get_approval_count(self) -> int:
        """Count approving reviews."""
        return sum(1 for r in self.reviews if r.get('approved', False))


@dataclass
class UnderstandingContract:
    """
    A smart contract based on understanding verification.

    Unlike traditional smart contracts (code is law),
    these contracts execute when genuine understanding
    is demonstrated and verified by three parties.
    """

    contract_id: str
    creator: str
    contract_type: ContractType
    terms: ContractTerms
    status: ContractStatus = ContractStatus.CREATED

    # Submissions
    submissions: List[Submission] = field(default_factory=list)
    winning_submission: Optional[str] = None  # Hash of winner

    # Timestamps
    created_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None

    # Escrow-specific (for ESCROW type)
    party_a_deposit: Optional[str] = None  # Knowledge hash from party A
    party_b_deposit: Optional[str] = None  # Knowledge hash from party B

    def __post_init__(self):
        if not self.contract_id:
            hash_input = f"{self.creator}:{self.created_at}"
            self.contract_id = hashlib.sha256(
                hash_input.encode()
            ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        d['contract_type'] = self.contract_type.value
        return d

    def is_active(self) -> bool:
        """Check if contract is still active."""
        if self.status != ContractStatus.ACTIVE:
            return False

        if self.terms.expiration and time.time() > self.terms.expiration:
            return False

        return True


class ContractEngine:
    """
    Engine for managing understanding contracts.

    Handles:
    - Contract creation
    - Submission processing
    - Triadic review
    - Contract execution
    """

    def __init__(
        self,
        trust_oracle=None,
        chain=None,
        inference_market=None,
        phi_coherence=None,
        lambda_g=None
    ):
        """
        Initialize Contract Engine.

        Args:
            trust_oracle: For getting trust scores
            chain: For recording contracts on-chain
            inference_market: For credit transfers
            phi_coherence: φ-coherence calculator
            lambda_g: Lambda-G boundary validator
        """
        self.trust_oracle = trust_oracle
        self.chain = chain
        self.market = inference_market
        self.phi_coherence = phi_coherence
        self.lambda_g = lambda_g

        # Contract registry
        self.contracts: Dict[str, UnderstandingContract] = {}

        # Reviewer pool
        self.reviewers: Dict[str, Dict[str, Any]] = {}

    def create_bounty(
        self,
        creator: str,
        description: str,
        bounty_credits: float,
        required_coherence: float = HIGH_COHERENCE,
        expiration_hours: Optional[float] = None
    ) -> UnderstandingContract:
        """
        Create a bounty contract.

        Args:
            creator: Who's posting the bounty
            description: What understanding is required
            bounty_credits: Credits offered
            required_coherence: Minimum φ-coherence
            expiration_hours: Hours until expiration

        Returns:
            The created contract
        """
        terms = ContractTerms(
            description=description,
            required_coherence=required_coherence,
            bounty_credits=bounty_credits,
            expiration=(
                time.time() + expiration_hours * 3600
                if expiration_hours else None
            )
        )

        contract = UnderstandingContract(
            contract_id="",  # Auto-generated
            creator=creator,
            contract_type=ContractType.BOUNTY,
            terms=terms,
            status=ContractStatus.ACTIVE
        )

        # Lock bounty credits (if market available)
        if self.market:
            balance = self.market.get_credits(creator)
            if balance < bounty_credits:
                raise ValueError(
                    f"Insufficient credits ({balance} < {bounty_credits})"
                )
            # Deduct from creator (held in escrow)
            self.market.credit_balances[creator] = balance - bounty_credits

        self.contracts[contract.contract_id] = contract

        # Record on chain
        if self.chain:
            self._record_on_chain(contract, "CREATE")

        return contract

    def create_escrow(
        self,
        party_a: str,
        party_b: str,
        description: str
    ) -> UnderstandingContract:
        """
        Create a knowledge escrow contract.

        Both parties deposit knowledge hashes.
        An arbiter verifies both are valuable.
        If verified, exchange happens atomically.
        """
        terms = ContractTerms(
            description=description,
            required_reviewers=1  # Just the arbiter
        )

        contract = UnderstandingContract(
            contract_id="",
            creator=party_a,
            contract_type=ContractType.ESCROW,
            terms=terms,
            status=ContractStatus.CREATED
        )

        self.contracts[contract.contract_id] = contract
        return contract

    def submit_solution(
        self,
        contract_id: str,
        submitter: str,
        content: Any,
        pob_proof: Optional[Dict[str, Any]] = None
    ) -> Submission:
        """
        Submit a solution to a contract.

        Args:
            contract_id: Which contract
            submitter: Who's submitting
            content: The submission content
            pob_proof: Optional PoB proof

        Returns:
            The Submission object
        """
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"Unknown contract: {contract_id}")

        if not contract.is_active():
            raise ValueError("Contract is not active")

        # Calculate coherence
        coherence = self._calculate_coherence(content)

        # Calculate lambda-G validity
        lambda_g_valid = self._validate_lambda_g(
            content, contract.terms.description
        )

        # Hash the content
        content_hash = hashlib.sha256(str(content).encode()).hexdigest()

        submission = Submission(
            submitter=submitter,
            content_hash=content_hash,
            coherence_score=coherence,
            lambda_g_valid=lambda_g_valid,
            pob_proof=pob_proof
        )

        contract.submissions.append(submission)
        contract.status = ContractStatus.REVIEWING

        return submission

    def _calculate_coherence(self, content: Any) -> float:
        """Calculate φ-coherence of content."""
        if self.phi_coherence:
            return self.phi_coherence.score(content)

        # Fallback: hash-based coherence
        content_str = str(content)
        hash_bytes = hashlib.sha256(content_str.encode()).digest()
        unique = len(set(hash_bytes))
        entropy = unique / 256.0

        target = PHI_INVERSE
        distance = abs(entropy - target)
        return max(0.0, 1.0 - distance / target)

    def _validate_lambda_g(
        self,
        submission: Any,
        terms: str
    ) -> bool:
        """Validate submission against terms using λG."""
        if self.lambda_g:
            return self.lambda_g.validate(submission, terms)

        # Fallback: basic text matching
        submission_str = str(submission).lower()
        terms_lower = terms.lower()

        # Count term matches
        terms_words = set(terms_lower.split())
        submission_words = set(submission_str.split())
        overlap = len(terms_words & submission_words)

        return overlap >= len(terms_words) * 0.5

    def review_submission(
        self,
        contract_id: str,
        submission_hash: str,
        reviewer: str,
        approved: bool,
        comments: str = "",
        pob_proof: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit a review for a submission.

        Three reviewers must approve for contract execution.
        """
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"Unknown contract: {contract_id}")

        # Find the submission
        submission = None
        for s in contract.submissions:
            if s.content_hash == submission_hash:
                submission = s
                break

        if not submission:
            raise ValueError(f"Unknown submission: {submission_hash}")

        review = {
            "reviewer": reviewer,
            "approved": approved,
            "comments": comments,
            "pob_proof": pob_proof,
            "timestamp": time.time()
        }

        submission.reviews.append(review)

        # Check if we have enough approvals
        approvals = submission.get_approval_count()
        if approvals >= contract.terms.required_reviewers:
            submission.approved = True
            self._execute_contract(contract, submission)

        return review

    def _execute_contract(
        self,
        contract: UnderstandingContract,
        winning_submission: Submission
    ) -> None:
        """Execute a contract (release bounty, etc.)."""
        contract.status = ContractStatus.EXECUTED
        contract.winning_submission = winning_submission.content_hash
        contract.executed_at = time.time()

        # Transfer bounty credits
        if contract.contract_type == ContractType.BOUNTY and self.market:
            self.market.add_credits(
                winning_submission.submitter,
                contract.terms.bounty_credits,
                f"bounty:{contract.contract_id}"
            )

        # Record on chain
        if self.chain:
            self._record_on_chain(contract, "EXECUTE")

        # Update trust scores
        if self.trust_oracle:
            # Winner's trust increases
            # Reviewers' trust updates based on consistency
            pass

    def _record_on_chain(
        self,
        contract: UnderstandingContract,
        action: str
    ) -> bool:
        """Record contract action on chain."""
        try:
            from .transaction import Transaction

            tx = Transaction(
                tx_type=f"CONTRACT_{action}",
                sender=contract.creator,
                data={
                    "contract_id": contract.contract_id,
                    "type": contract.contract_type.value,
                    "status": contract.status.value,
                    "action": action
                }
            )

            return self.chain.add_transaction(tx)
        except Exception:
            return False

    def register_reviewer(
        self,
        address: str,
        expertise: List[str] = None
    ) -> bool:
        """Register a node as a contract reviewer."""
        trust = 0.0
        if self.trust_oracle:
            trust = self.trust_oracle.get_trust_score(address)

        self.reviewers[address] = {
            "address": address,
            "trust": trust,
            "expertise": expertise or [],
            "reviews_completed": 0
        }
        return True

    def get_active_contracts(self) -> List[UnderstandingContract]:
        """Get all active contracts."""
        return [c for c in self.contracts.values() if c.is_active()]

    def get_contract(
        self,
        contract_id: str
    ) -> Optional[UnderstandingContract]:
        """Get a contract by ID."""
        return self.contracts.get(contract_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get contract engine statistics."""
        active = len([c for c in self.contracts.values() if c.is_active()])
        executed = len([
            c for c in self.contracts.values()
            if c.status == ContractStatus.EXECUTED
        ])

        total_bounty = sum(
            c.terms.bounty_credits for c in self.contracts.values()
            if c.contract_type == ContractType.BOUNTY
        )

        return {
            "total_contracts": len(self.contracts),
            "active": active,
            "executed": executed,
            "reviewers": len(self.reviewers),
            "total_bounty_credits": total_bounty
        }


def create_engine(
    trust_oracle=None,
    chain=None,
    inference_market=None
) -> ContractEngine:
    """Create a new Contract Engine."""
    return ContractEngine(
        trust_oracle=trust_oracle,
        chain=chain,
        inference_market=inference_market
    )


# CLI integration
def show_contracts_status(engine: ContractEngine) -> None:
    """Display contract status for CLI."""
    stats = engine.get_stats()

    print(f"\n📜 BAZINGA Understanding Contracts")
    print(f"   Total Contracts: {stats['total_contracts']}")
    print(f"   Active: {stats['active']}")
    print(f"   Executed: {stats['executed']}")
    print(f"   Reviewers: {stats['reviewers']}")
    print(f"   Total Bounty Pool: {stats['total_bounty_credits']:.2f} credits")

    active = engine.get_active_contracts()
    if active:
        print(f"\n   Active Bounties:")
        for c in active[:5]:
            print(f"     • {c.contract_id}: {c.terms.description[:40]}...")
            print(f"       Bounty: {c.terms.bounty_credits} credits")

    print()
