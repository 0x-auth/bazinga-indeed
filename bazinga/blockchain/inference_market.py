"""
BAZINGA Inference Market - Understanding as Currency
=====================================================

No money. No tokens. Just understanding.

Economics:
  - You earn credits by: answering queries, contributing
    knowledge, validating gradients, achieving PoB
  - You spend credits by: requesting inference from the network
  - Credits ARE trust score — they're the same thing

This creates a natural economy:
  - Contribute more → higher trust → better service
  - Free-ride → low trust → slow/no service
  - Understanding IS the currency

Rate: 1 successful PoB = 1 inference credit
Rate: 1 knowledge contribution = φ credits (1.618)
Rate: 1 gradient validation = φ² credits (2.618)

Why φ-scaled? Because contributions that help others
are worth more than proving your own understanding.

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


class InferenceStatus(Enum):
    """Status of an inference request."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"
    INSUFFICIENT_TRUST = "insufficient_trust"


@dataclass
class InferenceRequest:
    """An inference request in the market."""

    requester: str  # Who's asking
    query: str  # What they're asking
    request_id: str = ""  # Unique ID
    provider: Optional[str] = None  # Who's answering
    response: Optional[str] = None  # The answer
    status: InferenceStatus = InferenceStatus.PENDING

    # Credit economics
    cost: float = 1.0  # Credits charged
    requester_trust: float = 0.0  # Trust at time of request

    # Timing
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Quality metrics
    coherence_score: Optional[float] = None  # φ-coherence of response

    def __post_init__(self):
        if not self.request_id:
            # Generate unique ID
            hash_input = f"{self.requester}:{self.query}:{self.created_at}"
            self.request_id = hashlib.sha256(
                hash_input.encode()
            ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceRequest':
        """Create from dictionary."""
        data['status'] = InferenceStatus(data.get('status', 'pending'))
        return cls(**data)


@dataclass
class Provider:
    """An inference provider in the market."""

    address: str
    trust_score: float = 0.0
    capacity: int = 10  # Max concurrent requests
    active_requests: int = 0
    completed_requests: int = 0
    total_credits_earned: float = 0.0

    # Quality tracking
    avg_coherence: float = 0.0
    avg_response_time: float = 0.0

    def is_available(self) -> bool:
        """Check if provider can take more requests."""
        return self.active_requests < self.capacity

    def update_stats(
        self,
        coherence: float,
        response_time: float,
        credits: float
    ) -> None:
        """Update provider statistics after completing request."""
        n = self.completed_requests
        # Running average
        if n > 0:
            self.avg_coherence = (self.avg_coherence * n + coherence) / (n + 1)
            self.avg_response_time = (
                self.avg_response_time * n + response_time
            ) / (n + 1)
        else:
            self.avg_coherence = coherence
            self.avg_response_time = response_time

        self.completed_requests += 1
        self.total_credits_earned += credits


class InferenceMarket:
    """
    Trust-based inference marketplace.

    Routes queries to the highest-trust available providers.
    Payment is in understanding credits, not money.
    """

    # Credit rates (from Claude Web's plan)
    CREDIT_POB = 1.0           # 1 PoB = 1 credit
    CREDIT_KNOWLEDGE = PHI     # 1 contribution = φ credits
    CREDIT_GRADIENT = PHI ** 2 # 1 validation = φ² credits
    CREDIT_INFERENCE = 1.0     # 1 query costs 1 credit

    # Minimum trust to use the market
    MIN_TRUST = 0.1

    def __init__(self, trust_oracle=None, chain=None):
        """
        Initialize Inference Market.

        Args:
            trust_oracle: TrustOracle for getting trust scores
            chain: DarmiyanChain for recording transactions
        """
        self.trust_oracle = trust_oracle
        self.chain = chain

        # Provider registry
        self.providers: Dict[str, Provider] = {}

        # Request tracking
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: List[InferenceRequest] = []

        # Credit ledger (if no trust oracle)
        self.credit_balances: Dict[str, float] = {}

    def register_provider(
        self,
        address: str,
        capacity: int = 10
    ) -> Provider:
        """Register a node as an inference provider."""
        trust = self._get_trust(address)

        provider = Provider(
            address=address,
            trust_score=trust,
            capacity=capacity
        )

        self.providers[address] = provider
        return provider

    def _get_trust(self, address: str) -> float:
        """Get trust score for an address."""
        if self.trust_oracle:
            return self.trust_oracle.get_trust_score(address)
        # Fallback to credit balance as trust proxy
        return min(1.0, self.credit_balances.get(address, 0.0) / 10.0)

    def add_credits(
        self,
        address: str,
        amount: float,
        reason: str = "contribution",
        _internal: bool = False
    ) -> float:
        """
        Add credits to an address.

        SECURITY FIX (Feb 2026 Audit - Round 4):
        - Credits can ONLY be added via legitimate activities
        - External calls are REJECTED unless _internal=True
        - Valid reasons: pob_proof, knowledge_attestation, inference_provided
        - Maximum single addition capped at 100 credits

        Args:
            address: Node address
            amount: Credits to add
            reason: Why credits are being added
            _internal: Must be True for internal calls (not exposed to users)

        Returns:
            New balance, or 0.0 if rejected
        """
        # SECURITY FIX: Reject external credit additions
        if not _internal:
            # Log attempted manipulation
            print(f"  ⚠️  REJECTED: External credit addition attempt for {address}")
            return self.credit_balances.get(address, 0.0)

        # SECURITY FIX: Only allow legitimate reasons
        valid_reasons = {
            'pob_proof', 'knowledge_attestation', 'inference_provided',
            'gradient_accepted', 'learning_contribution', 'governance_reward'
        }
        if reason not in valid_reasons:
            print(f"  ⚠️  REJECTED: Invalid credit reason '{reason}'")
            return self.credit_balances.get(address, 0.0)

        # SECURITY FIX: Cap single additions
        MAX_SINGLE_CREDIT = 100.0
        if amount > MAX_SINGLE_CREDIT:
            print(f"  ⚠️  REJECTED: Credit amount {amount} exceeds max {MAX_SINGLE_CREDIT}")
            return self.credit_balances.get(address, 0.0)

        # SECURITY FIX: No negative credits
        if amount < 0:
            print(f"  ⚠️  REJECTED: Negative credit amount {amount}")
            return self.credit_balances.get(address, 0.0)

        current = self.credit_balances.get(address, 0.0)
        new_balance = current + amount
        self.credit_balances[address] = new_balance

        # Record on chain if available
        if self.chain:
            self._record_credit_change(address, amount, reason)

        return new_balance

    def _record_credit_change(
        self,
        address: str,
        amount: float,
        reason: str
    ) -> bool:
        """Record credit change on chain."""
        try:
            from .transaction import Transaction

            tx = Transaction(
                tx_type="CREDIT_CHANGE",
                sender="market",
                receiver=address,
                data={
                    "amount": amount,
                    "reason": reason,
                    "timestamp": time.time()
                }
            )

            return self.chain.add_transaction(tx)
        except Exception:
            return False

    def request_inference(
        self,
        requester: str,
        query: str
    ) -> InferenceRequest:
        """
        Request inference from the network.

        Args:
            requester: Who's asking
            query: The question

        Returns:
            InferenceRequest (may have INSUFFICIENT_TRUST status)
        """
        # Check trust/credits
        trust = self._get_trust(requester)

        request = InferenceRequest(
            requester=requester,
            query=query,
            requester_trust=trust
        )

        if trust < self.MIN_TRUST:
            request.status = InferenceStatus.INSUFFICIENT_TRUST
            request.response = (
                f"Insufficient trust ({trust:.2f} < {self.MIN_TRUST}). "
                "Contribute knowledge or prove boundary first."
            )
            return request

        # Find best available provider
        provider = self._find_provider(min_trust=trust * 0.5)

        if provider is None:
            request.status = InferenceStatus.FAILED
            request.response = "No providers available. Try again later."
            return request

        # Assign to provider
        request.provider = provider.address
        request.status = InferenceStatus.ASSIGNED
        provider.active_requests += 1

        self.pending_requests[request.request_id] = request
        return request

    def _find_provider(
        self,
        min_trust: float = 0.0
    ) -> Optional[Provider]:
        """Find the best available provider."""
        available = [
            p for p in self.providers.values()
            if p.is_available() and p.trust_score >= min_trust
        ]

        if not available:
            return None

        # Sort by trust score (descending)
        available.sort(key=lambda p: p.trust_score, reverse=True)
        return available[0]

    def complete_inference(
        self,
        request_id: str,
        response: str,
        coherence_score: float
    ) -> InferenceRequest:
        """
        Complete an inference request.

        Args:
            request_id: ID of the request
            response: The answer
            coherence_score: φ-coherence of the response

        Returns:
            Updated InferenceRequest
        """
        request = self.pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Unknown request: {request_id}")

        # Complete the request
        request.response = response
        request.coherence_score = coherence_score
        request.completed_at = time.time()
        request.status = InferenceStatus.COMPLETED

        # Update provider stats
        if request.provider and request.provider in self.providers:
            provider = self.providers[request.provider]
            provider.active_requests -= 1

            response_time = request.completed_at - request.created_at
            provider.update_stats(
                coherence=coherence_score,
                response_time=response_time,
                credits=request.cost
            )

        # Transfer credits
        self._transfer_credits(request)

        # Move to completed
        del self.pending_requests[request_id]
        self.completed_requests.append(request)

        # Update trust based on coherence
        if request.provider:
            self._update_provider_trust(request.provider, coherence_score)

        return request

    def _transfer_credits(self, request: InferenceRequest) -> None:
        """Transfer credits from requester to provider."""
        if not request.provider:
            return

        cost = request.cost

        # Deduct from requester
        requester_balance = self.credit_balances.get(request.requester, 0.0)
        self.credit_balances[request.requester] = max(
            0.0, requester_balance - cost
        )

        # Add to provider
        self.add_credits(
            request.provider,
            cost,
            f"inference:{request.request_id}"
        )

        # Record on chain
        if self.chain:
            try:
                from .transaction import Transaction

                tx = Transaction(
                    tx_type="INFERENCE_TRANSFER",
                    sender=request.requester,
                    receiver=request.provider,
                    data={
                        "request_id": request.request_id,
                        "cost": cost,
                        "coherence": request.coherence_score
                    }
                )

                self.chain.add_transaction(tx)
            except Exception:
                pass

    def _update_provider_trust(
        self,
        provider_address: str,
        coherence: float
    ) -> None:
        """Update provider trust based on response coherence."""
        if provider_address not in self.providers:
            return

        provider = self.providers[provider_address]

        # High coherence → trust increases
        # Low coherence → trust decreases
        PHI_INVERSE = 1.0 / PHI  # 0.618

        if coherence >= PHI_INVERSE:
            # Good response, increase trust
            delta = (coherence - PHI_INVERSE) * 0.1
            provider.trust_score = min(1.0, provider.trust_score + delta)
        else:
            # Poor response, decrease trust
            delta = (PHI_INVERSE - coherence) * 0.1
            provider.trust_score = max(0.0, provider.trust_score - delta)

    def get_credits(self, address: str) -> float:
        """Get credit balance for an address."""
        return self.credit_balances.get(address, 0.0)

    def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics."""
        total_credits = sum(self.credit_balances.values())

        completed_coherence = [
            r.coherence_score for r in self.completed_requests
            if r.coherence_score is not None
        ]

        return {
            "providers": len(self.providers),
            "pending_requests": len(self.pending_requests),
            "completed_requests": len(self.completed_requests),
            "total_credits_in_circulation": total_credits,
            "avg_coherence": (
                sum(completed_coherence) / len(completed_coherence)
                if completed_coherence else 0.0
            ),
            "active_capacity": sum(
                p.capacity - p.active_requests
                for p in self.providers.values()
            )
        }

    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top providers by trust score."""
        sorted_providers = sorted(
            self.providers.values(),
            key=lambda p: (p.trust_score, p.completed_requests),
            reverse=True
        )

        return [
            {
                "address": p.address[:16] + "...",
                "trust": p.trust_score,
                "completed": p.completed_requests,
                "credits_earned": p.total_credits_earned,
                "avg_coherence": p.avg_coherence
            }
            for p in sorted_providers[:top_n]
        ]


def create_market(trust_oracle=None, chain=None) -> InferenceMarket:
    """Create a new Inference Market."""
    return InferenceMarket(trust_oracle=trust_oracle, chain=chain)


# CLI integration
def show_market_status(market: InferenceMarket) -> None:
    """Display market status for CLI."""
    stats = market.get_market_stats()

    print(f"\n⚡ BAZINGA Inference Market")
    print(f"   Providers: {stats['providers']}")
    print(f"   Available Capacity: {stats['active_capacity']}")
    print(f"   Pending Requests: {stats['pending_requests']}")
    print(f"   Completed: {stats['completed_requests']}")
    print(f"   Avg Coherence: {stats['avg_coherence']:.3f}")
    print(f"   Credits in Circulation: {stats['total_credits_in_circulation']:.2f}")

    leaderboard = market.get_leaderboard(5)
    if leaderboard:
        print(f"\n   Top Providers:")
        for i, p in enumerate(leaderboard, 1):
            print(f"     {i}. {p['address']} (trust: {p['trust']:.2f}, completed: {p['completed']})")

    print()
