#!/usr/bin/env python3
"""
Phi-Coherence Voting — Trust-Weighted Consensus
=================================================
Not simple majority. Votes are weighted by:
    1. Voter's trust score (from TrustOracle)
    2. φ-coherence of the voter's reasoning
    3. Whether high-trust nodes specifically approved

Threshold: weighted approval > φ⁻¹ (0.618)
AND at least 3 high-trust approvers (Sybil resistance).

From SELF_PROPOSAL_SYSTEM.md:
    "If coherent + majority approval → auto-merge"
From AI_SAFETY_ANALYSIS.md:
    "Weight votes by node trust, not just count."
"""

from dataclasses import dataclass
from typing import List, Optional

from bazinga.evolution.proposal import Vote, EvolutionProposal


PHI = 1.618033988749895
PHI_INVERSE = 1.0 / PHI  # 0.618...
HIGH_TRUST_THRESHOLD = 0.5  # Minimum trust to count as "high-trust"
MIN_HIGH_TRUST_APPROVERS = 3  # Sybil resistance


@dataclass
class TallyResult:
    """Result of tallying votes on a proposal."""
    weighted_approval: float   # 0-1, trust-weighted approval rate
    total_votes: int
    approve_count: int
    reject_count: int
    high_trust_approvers: int
    threshold: float           # Required for passage (φ⁻¹)
    approved: bool
    reason: str

    @property
    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        return (
            f"{status} | "
            f"weighted={self.weighted_approval:.3f} "
            f"(threshold={self.threshold:.3f}) | "
            f"votes={self.approve_count}/{self.total_votes} | "
            f"high-trust={self.high_trust_approvers}"
        )


class PhiCoherenceVoting:
    """
    Phi-weighted voting for proposals.

    Vote weight = trust_score × (1 + phi_coherence)
    This means:
        - Higher trust = more voting power
        - More coherent reasoning = bonus weight
        - Both are needed for maximum influence

    Usage:
        voting = PhiCoherenceVoting()
        result = voting.tally(proposal)

        if result.approved:
            # Proceed to apply
        else:
            print(result.reason)
    """

    def __init__(
        self,
        threshold: float = PHI_INVERSE,
        min_high_trust_approvers: int = MIN_HIGH_TRUST_APPROVERS,
        high_trust_threshold: float = HIGH_TRUST_THRESHOLD,
    ):
        self.threshold = threshold
        self.min_high_trust_approvers = min_high_trust_approvers
        self.high_trust_threshold = high_trust_threshold

    def cast_vote(
        self,
        proposal: EvolutionProposal,
        vote: Vote,
    ) -> bool:
        """
        Add a vote to a proposal.

        Args:
            proposal: The proposal being voted on
            vote: The vote to cast

        Returns:
            True if vote was accepted
        """
        # Check for duplicate votes
        for existing in proposal.votes:
            if existing.get("voter_node_id") == vote.voter_node_id:
                return False  # Already voted

        proposal.votes.append(vote.to_dict())
        return True

    def tally(self, proposal: EvolutionProposal) -> TallyResult:
        """
        Compute weighted consensus from all votes.

        Formula:
            vote_weight = trust × (1 + coherence)
            weighted_approval = Σ(approve_weights) / Σ(all_weights)
            approved = weighted_approval > φ⁻¹ AND high_trust_approvers >= 3
        """
        if not proposal.votes:
            return TallyResult(
                weighted_approval=0.0,
                total_votes=0,
                approve_count=0,
                reject_count=0,
                high_trust_approvers=0,
                threshold=self.threshold,
                approved=False,
                reason="No votes cast",
            )

        votes = [Vote.from_dict(v) for v in proposal.votes]

        # Calculate weights
        approve_weight = 0.0
        total_weight = 0.0
        approve_count = 0
        reject_count = 0
        high_trust_approvers = 0

        for vote in votes:
            weight = vote.trust_weight * (1.0 + vote.phi_coherence)

            # Minimum weight floor (prevents zero-weight votes)
            weight = max(weight, 0.01)

            total_weight += weight

            if vote.approve:
                approve_weight += weight
                approve_count += 1
                if vote.trust_weight >= self.high_trust_threshold:
                    high_trust_approvers += 1
            else:
                reject_count += 1

        weighted_approval = approve_weight / total_weight if total_weight > 0 else 0.0

        # Check both conditions
        weight_ok = weighted_approval >= self.threshold
        trust_ok = high_trust_approvers >= self.min_high_trust_approvers

        approved = weight_ok and trust_ok

        if not weight_ok:
            reason = (
                f"Weighted approval {weighted_approval:.3f} "
                f"< threshold {self.threshold:.3f}"
            )
        elif not trust_ok:
            reason = (
                f"Only {high_trust_approvers} high-trust approvers "
                f"(need {self.min_high_trust_approvers})"
            )
        else:
            reason = "Consensus achieved"

        return TallyResult(
            weighted_approval=weighted_approval,
            total_votes=len(votes),
            approve_count=approve_count,
            reject_count=reject_count,
            high_trust_approvers=high_trust_approvers,
            threshold=self.threshold,
            approved=approved,
            reason=reason,
        )

    def detect_sybil(self, proposal: EvolutionProposal) -> float:
        """
        Estimate probability of Sybil attack (0-1).

        Indicators:
        - Many votes from low-trust nodes
        - Votes arriving in suspicious bursts
        - All approving with same reasoning
        """
        if len(proposal.votes) < 3:
            return 0.0

        votes = [Vote.from_dict(v) for v in proposal.votes]
        score = 0.0

        # Check 1: Ratio of low-trust voters
        low_trust = sum(1 for v in votes if v.trust_weight < 0.3)
        low_trust_ratio = low_trust / len(votes)
        if low_trust_ratio > 0.5:
            score += 0.3

        # Check 2: Timestamp clustering (votes within 1 second)
        timestamps = sorted(v.timestamp for v in votes)
        if len(timestamps) >= 3:
            gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            burst_count = sum(1 for g in gaps if g < 1.0)
            if burst_count > len(gaps) * 0.5:
                score += 0.3

        # Check 3: Identical reasoning
        reasonings = [v.reasoning.strip().lower() for v in votes]
        unique_reasonings = len(set(reasonings))
        if unique_reasonings < len(reasonings) * 0.5:
            score += 0.4

        return min(1.0, score)
