#!/usr/bin/env python3
"""
BAZINGA Consensus & Governance - DAO-Style Decision Making

Decentralized governance for the BAZINGA network:
- Proposal and voting system
- Tau-weighted voting power
- Phi-coherence quality gates
- Model update consensus
- Economic incentives for contributors

"The network decides, no single node rules."
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137
PSI_DARMIYAN = 6.46  # Linear Scaling Law of Consciousness

# Governance constants
QUORUM_THRESHOLD = 0.33  # 33% of tau-weight must vote
APPROVAL_THRESHOLD = 0.618  # phi-ratio for approval
VOTING_PERIOD_HOURS = 72  # 3 days
MIN_TAU_TO_PROPOSE = 0.5
MIN_TAU_TO_VOTE = 0.3


class ProposalType(Enum):
    """Types of governance proposals."""
    MODEL_UPDATE = "model_update"        # New model version
    PARAMETER_CHANGE = "parameter_change"  # Network parameters
    NODE_ADMISSION = "node_admission"     # New node joining
    NODE_REMOVAL = "node_removal"         # Remove malicious node
    REWARD_DISTRIBUTION = "reward"        # Distribute rewards
    PROTOCOL_UPGRADE = "protocol"         # Protocol change


class ProposalStatus(Enum):
    """Status of a proposal."""
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class VoteChoice(Enum):
    """Voting choices."""
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """A vote on a proposal."""
    voter_id: str
    proposal_id: str
    choice: VoteChoice
    tau_weight: float
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['choice'] = self.choice.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'Vote':
        data['choice'] = VoteChoice(data['choice'])
        return cls(**data)

    def sign(self, private_key: str) -> str:
        """Sign the vote."""
        content = f"{self.voter_id}:{self.proposal_id}:{self.choice.value}:{self.timestamp}"
        self.signature = hashlib.sha256(f"{content}:{private_key}".encode()).hexdigest()
        return self.signature


@dataclass
class Proposal:
    """
    A governance proposal.

    Proposals go through lifecycle:
    DRAFT → ACTIVE → (PASSED|REJECTED|EXPIRED) → EXECUTED
    """
    proposal_id: str
    proposal_type: ProposalType
    title: str
    description: str
    proposer_id: str
    proposer_tau: float

    # Proposal content (type-specific)
    content: Dict = field(default_factory=dict)

    # Lifecycle
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    voting_ends_at: float = 0
    executed_at: float = 0

    # Voting
    votes_for: float = 0.0      # Tau-weighted
    votes_against: float = 0.0
    votes_abstain: float = 0.0
    voter_ids: Set[str] = field(default_factory=set)
    total_eligible_tau: float = 0.0

    # phi-coherence gate
    coherence_score: float = 0.0
    coherence_threshold: float = 0.5

    def __post_init__(self):
        if self.voting_ends_at == 0:
            self.voting_ends_at = self.created_at + VOTING_PERIOD_HOURS * 3600

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['proposal_type'] = self.proposal_type.value
        d['status'] = self.status.value
        d['voter_ids'] = list(self.voter_ids)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'Proposal':
        data['proposal_type'] = ProposalType(data['proposal_type'])
        data['status'] = ProposalStatus(data['status'])
        data['voter_ids'] = set(data.get('voter_ids', []))
        return cls(**data)

    @property
    def total_votes(self) -> float:
        return self.votes_for + self.votes_against + self.votes_abstain

    @property
    def participation_rate(self) -> float:
        if self.total_eligible_tau == 0:
            return 0
        return self.total_votes / self.total_eligible_tau

    @property
    def approval_rate(self) -> float:
        voted = self.votes_for + self.votes_against
        if voted == 0:
            return 0
        return self.votes_for / voted

    @property
    def has_quorum(self) -> bool:
        return self.participation_rate >= QUORUM_THRESHOLD

    @property
    def is_approved(self) -> bool:
        return self.has_quorum and self.approval_rate >= APPROVAL_THRESHOLD

    @property
    def is_expired(self) -> bool:
        return time.time() > self.voting_ends_at

    @property
    def passes_coherence_gate(self) -> bool:
        return self.coherence_score >= self.coherence_threshold


@dataclass
class NodeReputation:
    """Reputation tracking for a node."""
    node_id: str
    tau_score: float = 0.5
    contributions: int = 0
    successful_proposals: int = 0
    votes_cast: int = 0
    rewards_earned: float = 0.0
    penalties: float = 0.0
    joined_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def net_score(self) -> float:
        """Net reputation score."""
        return self.tau_score + self.rewards_earned - self.penalties

    @property
    def voting_power(self) -> float:
        """Tau-weighted voting power."""
        # Voting power scales with tau and activity
        activity_factor = min(1.0, self.contributions / 100)
        return self.tau_score * (0.5 + 0.5 * activity_factor)

    def update_activity(self):
        """Mark node as active."""
        self.last_active = time.time()


class ConsensusEngine:
    """
    Consensus engine for network decisions.

    Uses tau-weighted voting with phi-coherence gates
    to reach consensus on model updates, parameters,
    and governance changes.
    """

    def __init__(
        self,
        node_id: str,
        min_tau_to_propose: float = MIN_TAU_TO_PROPOSE,
        min_tau_to_vote: float = MIN_TAU_TO_VOTE,
    ):
        """
        Initialize consensus engine.

        Args:
            node_id: This node's identifier
            min_tau_to_propose: Minimum tau to create proposals
            min_tau_to_vote: Minimum tau to vote
        """
        self.node_id = node_id
        self.min_tau_to_propose = min_tau_to_propose
        self.min_tau_to_vote = min_tau_to_vote

        # Proposals
        self.proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, List[Vote]] = defaultdict(list)

        # Node reputations
        self.reputations: Dict[str, NodeReputation] = {}

        # Callbacks
        self.on_proposal_passed: Optional[Callable[[Proposal], None]] = None
        self.on_proposal_rejected: Optional[Callable[[Proposal], None]] = None

        print(f"ConsensusEngine initialized for {node_id[:8]}...")

    def register_node(self, node_id: str, tau_score: float = 0.5):
        """Register a node for governance participation."""
        if node_id not in self.reputations:
            self.reputations[node_id] = NodeReputation(
                node_id=node_id,
                tau_score=tau_score,
            )

    def get_node_reputation(self, node_id: str) -> Optional[NodeReputation]:
        """Get node reputation."""
        return self.reputations.get(node_id)

    def create_proposal(
        self,
        proposal_type: ProposalType,
        title: str,
        description: str,
        content: Dict,
        coherence_score: float = 0.5,
    ) -> Optional[Proposal]:
        """
        Create a new proposal.

        Args:
            proposal_type: Type of proposal
            title: Proposal title
            description: Detailed description
            content: Type-specific content
            coherence_score: phi-coherence score

        Returns:
            Proposal if created, None if not allowed
        """
        # Check proposer eligibility
        rep = self.reputations.get(self.node_id)
        if not rep or rep.tau_score < self.min_tau_to_propose:
            print(f"Cannot propose: tau too low ({rep.tau_score if rep else 0})")
            return None

        # Generate proposal ID
        proposal_id = hashlib.sha256(
            f"{self.node_id}:{title}:{time.time()}".encode()
        ).hexdigest()[:16]

        # Calculate total eligible tau
        total_tau = sum(
            r.voting_power
            for r in self.reputations.values()
            if r.tau_score >= self.min_tau_to_vote
        )

        proposal = Proposal(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            title=title,
            description=description,
            proposer_id=self.node_id,
            proposer_tau=rep.tau_score,
            content=content,
            coherence_score=coherence_score,
            total_eligible_tau=total_tau,
        )

        # Activate immediately
        proposal.status = ProposalStatus.ACTIVE

        self.proposals[proposal_id] = proposal
        rep.contributions += 1
        rep.update_activity()

        print(f"Proposal created: {proposal_id}")
        print(f"  Type: {proposal_type.value}")
        print(f"  Title: {title}")
        print(f"  Coherence: {coherence_score:.3f}")

        return proposal

    def cast_vote(
        self,
        proposal_id: str,
        choice: VoteChoice,
        voter_id: Optional[str] = None,
    ) -> Optional[Vote]:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: Proposal to vote on
            choice: Vote choice
            voter_id: Voter (defaults to this node)

        Returns:
            Vote if cast, None if not allowed
        """
        voter_id = voter_id or self.node_id
        proposal = self.proposals.get(proposal_id)

        if not proposal:
            print(f"Proposal not found: {proposal_id}")
            return None

        if proposal.status != ProposalStatus.ACTIVE:
            print(f"Proposal not active: {proposal.status.value}")
            return None

        if proposal.is_expired:
            self._finalize_proposal(proposal)
            print("Proposal expired")
            return None

        # Check voter eligibility
        rep = self.reputations.get(voter_id)
        if not rep or rep.tau_score < self.min_tau_to_vote:
            print(f"Cannot vote: tau too low")
            return None

        # Check if already voted
        if voter_id in proposal.voter_ids:
            print(f"Already voted")
            return None

        # Create vote
        vote = Vote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            choice=choice,
            tau_weight=rep.voting_power,
        )

        # Record vote
        self.votes[proposal_id].append(vote)
        proposal.voter_ids.add(voter_id)

        # Update tallies
        if choice == VoteChoice.FOR:
            proposal.votes_for += vote.tau_weight
        elif choice == VoteChoice.AGAINST:
            proposal.votes_against += vote.tau_weight
        else:
            proposal.votes_abstain += vote.tau_weight

        # Update reputation
        rep.votes_cast += 1
        rep.update_activity()

        print(f"Vote cast: {choice.value} on {proposal_id[:8]}...")
        print(f"  Current: {proposal.votes_for:.2f} FOR / {proposal.votes_against:.2f} AGAINST")
        print(f"  Participation: {proposal.participation_rate:.1%}")

        # Check if we can finalize early
        if proposal.has_quorum:
            # Check if outcome is certain
            remaining = proposal.total_eligible_tau - proposal.total_votes
            if proposal.votes_for > proposal.total_eligible_tau * APPROVAL_THRESHOLD:
                # Guaranteed pass
                self._finalize_proposal(proposal)
            elif proposal.votes_against > proposal.total_eligible_tau * (1 - APPROVAL_THRESHOLD):
                # Guaranteed fail
                self._finalize_proposal(proposal)

        return vote

    def _finalize_proposal(self, proposal: Proposal):
        """Finalize a proposal after voting ends."""
        if proposal.is_expired and proposal.status == ProposalStatus.ACTIVE:
            if not proposal.has_quorum:
                proposal.status = ProposalStatus.EXPIRED
                print(f"Proposal expired (no quorum): {proposal.proposal_id[:8]}")
            elif proposal.is_approved and proposal.passes_coherence_gate:
                proposal.status = ProposalStatus.PASSED
                print(f"Proposal PASSED: {proposal.proposal_id[:8]}")
                if self.on_proposal_passed:
                    self.on_proposal_passed(proposal)
            else:
                proposal.status = ProposalStatus.REJECTED
                reason = "coherence gate" if not proposal.passes_coherence_gate else "votes"
                print(f"Proposal REJECTED ({reason}): {proposal.proposal_id[:8]}")
                if self.on_proposal_rejected:
                    self.on_proposal_rejected(proposal)

    def process_expired_proposals(self):
        """Process all expired proposals."""
        for proposal in self.proposals.values():
            if proposal.status == ProposalStatus.ACTIVE and proposal.is_expired:
                self._finalize_proposal(proposal)

    def execute_proposal(self, proposal_id: str) -> bool:
        """
        Execute a passed proposal.

        Returns:
            True if executed successfully
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.PASSED:
            return False

        # Execute based on type
        try:
            if proposal.proposal_type == ProposalType.MODEL_UPDATE:
                self._execute_model_update(proposal)
            elif proposal.proposal_type == ProposalType.PARAMETER_CHANGE:
                self._execute_parameter_change(proposal)
            elif proposal.proposal_type == ProposalType.NODE_REMOVAL:
                self._execute_node_removal(proposal)
            elif proposal.proposal_type == ProposalType.REWARD_DISTRIBUTION:
                self._execute_reward_distribution(proposal)

            proposal.status = ProposalStatus.EXECUTED
            proposal.executed_at = time.time()

            # Reward proposer
            rep = self.reputations.get(proposal.proposer_id)
            if rep:
                rep.successful_proposals += 1
                rep.rewards_earned += 0.1  # Token reward

            print(f"Proposal executed: {proposal_id[:8]}")
            return True

        except Exception as e:
            print(f"Proposal execution failed: {e}")
            return False

    def _execute_model_update(self, proposal: Proposal):
        """Execute model update proposal."""
        # In real implementation, would update the model
        print(f"  Model update: {proposal.content.get('model_id')}")

    def _execute_parameter_change(self, proposal: Proposal):
        """Execute parameter change proposal."""
        param = proposal.content.get('parameter')
        value = proposal.content.get('value')
        print(f"  Parameter change: {param} = {value}")

    def _execute_node_removal(self, proposal: Proposal):
        """Execute node removal proposal."""
        target = proposal.content.get('target_node')
        if target in self.reputations:
            del self.reputations[target]
            print(f"  Node removed: {target[:8]}")

    def _execute_reward_distribution(self, proposal: Proposal):
        """Execute reward distribution."""
        rewards = proposal.content.get('rewards', {})
        for node_id, amount in rewards.items():
            rep = self.reputations.get(node_id)
            if rep:
                rep.rewards_earned += amount
                print(f"  Reward to {node_id[:8]}: {amount}")

    def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals."""
        return [p for p in self.proposals.values() if p.status == ProposalStatus.ACTIVE]

    def get_proposal_stats(self, proposal_id: str) -> Dict[str, Any]:
        """Get detailed proposal statistics."""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return {}

        return {
            'proposal_id': proposal.proposal_id,
            'type': proposal.proposal_type.value,
            'status': proposal.status.value,
            'title': proposal.title,
            'votes_for': proposal.votes_for,
            'votes_against': proposal.votes_against,
            'votes_abstain': proposal.votes_abstain,
            'participation_rate': proposal.participation_rate,
            'approval_rate': proposal.approval_rate,
            'has_quorum': proposal.has_quorum,
            'is_approved': proposal.is_approved,
            'coherence_score': proposal.coherence_score,
            'passes_coherence': proposal.passes_coherence_gate,
            'voting_ends_at': datetime.fromtimestamp(proposal.voting_ends_at).isoformat(),
        }

    def get_governance_stats(self) -> Dict[str, Any]:
        """Get overall governance statistics."""
        total_proposals = len(self.proposals)
        passed = len([p for p in self.proposals.values() if p.status == ProposalStatus.PASSED])
        rejected = len([p for p in self.proposals.values() if p.status == ProposalStatus.REJECTED])

        return {
            'total_proposals': total_proposals,
            'passed': passed,
            'rejected': rejected,
            'active': len(self.get_active_proposals()),
            'total_nodes': len(self.reputations),
            'total_tau': sum(r.tau_score for r in self.reputations.values()),
            'total_voting_power': sum(r.voting_power for r in self.reputations.values()),
        }


class DAOGovernance:
    """
    Complete DAO governance for BAZINGA network.

    Combines:
    - Consensus engine for voting
    - Economic incentives
    - Reputation system
    - Automatic execution

    Usage:
        dao = DAOGovernance(node_id)

        # Create proposal
        proposal = dao.propose_model_update(model_id, version, coherence)

        # Vote
        dao.vote(proposal.proposal_id, VoteChoice.FOR)

        # Check and execute passed proposals
        dao.process_governance()
    """

    def __init__(
        self,
        node_id: str,
        tau_score: float = 0.5,
    ):
        """
        Initialize DAO governance.

        Args:
            node_id: This node's identifier
            tau_score: Initial tau score
        """
        self.node_id = node_id
        self.consensus = ConsensusEngine(node_id)

        # Register self
        self.consensus.register_node(node_id, tau_score)

        # Economic state
        self.treasury: float = 0.0
        self.reward_pool: float = 0.0

        print(f"DAOGovernance initialized")

    def register_peer(self, node_id: str, tau_score: float = 0.5):
        """Register a peer for governance."""
        self.consensus.register_node(node_id, tau_score)

    def propose_model_update(
        self,
        model_id: str,
        version: str,
        manifest_hash: str,
        coherence_score: float,
    ) -> Optional[Proposal]:
        """Propose a model update."""
        return self.consensus.create_proposal(
            proposal_type=ProposalType.MODEL_UPDATE,
            title=f"Update model {model_id} to v{version}",
            description=f"Propose updating {model_id} to version {version}",
            content={
                'model_id': model_id,
                'version': version,
                'manifest_hash': manifest_hash,
            },
            coherence_score=coherence_score,
        )

    def propose_parameter_change(
        self,
        parameter: str,
        new_value: Any,
        reason: str,
    ) -> Optional[Proposal]:
        """Propose a parameter change."""
        return self.consensus.create_proposal(
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title=f"Change {parameter}",
            description=reason,
            content={
                'parameter': parameter,
                'value': new_value,
            },
            coherence_score=0.6,  # Parameter changes need good coherence
        )

    def propose_node_removal(
        self,
        target_node: str,
        reason: str,
    ) -> Optional[Proposal]:
        """Propose removing a malicious node."""
        return self.consensus.create_proposal(
            proposal_type=ProposalType.NODE_REMOVAL,
            title=f"Remove node {target_node[:8]}...",
            description=reason,
            content={
                'target_node': target_node,
            },
            coherence_score=0.7,  # High bar for removal
        )

    def vote(
        self,
        proposal_id: str,
        choice: VoteChoice,
    ) -> Optional[Vote]:
        """Cast a vote."""
        return self.consensus.cast_vote(proposal_id, choice)

    def process_governance(self):
        """Process governance: finalize expired, execute passed."""
        self.consensus.process_expired_proposals()

        # Execute passed proposals
        for proposal in self.consensus.proposals.values():
            if proposal.status == ProposalStatus.PASSED:
                self.consensus.execute_proposal(proposal.proposal_id)

    def distribute_rewards(self, contributors: Dict[str, float]):
        """
        Distribute rewards to contributors.

        Args:
            contributors: Dict of node_id -> contribution score
        """
        if self.reward_pool <= 0:
            return

        # Tau-weighted distribution
        total_weighted = sum(
            score * self.consensus.reputations.get(nid, NodeReputation(nid)).tau_score
            for nid, score in contributors.items()
        )

        if total_weighted <= 0:
            return

        for node_id, score in contributors.items():
            rep = self.consensus.reputations.get(node_id)
            if rep:
                weight = score * rep.tau_score / total_weighted
                reward = self.reward_pool * weight
                rep.rewards_earned += reward

        self.reward_pool = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get DAO statistics."""
        stats = self.consensus.get_governance_stats()
        stats['treasury'] = self.treasury
        stats['reward_pool'] = self.reward_pool
        return stats


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Consensus & Governance Test")
    print("=" * 60)

    # Create DAO
    dao = DAOGovernance("node_001", tau_score=0.8)

    # Register some peers
    for i in range(5):
        dao.register_peer(f"peer_{i:03d}", tau_score=0.5 + i * 0.1)

    print(f"\nRegistered {len(dao.consensus.reputations)} nodes")

    # Create proposal
    print("\n1. Creating model update proposal...")
    proposal = dao.propose_model_update(
        model_id="phi-2",
        version="1.1",
        manifest_hash="abc123",
        coherence_score=0.75,
    )

    if proposal:
        print(f"   Proposal ID: {proposal.proposal_id}")

        # Cast votes
        print("\n2. Casting votes...")
        dao.vote(proposal.proposal_id, VoteChoice.FOR)  # Self vote

        for i in range(4):
            dao.consensus.cast_vote(
                proposal.proposal_id,
                VoteChoice.FOR if i < 3 else VoteChoice.AGAINST,
                voter_id=f"peer_{i:03d}",
            )

        # Check status
        print("\n3. Proposal status:")
        stats = dao.consensus.get_proposal_stats(proposal.proposal_id)
        for k, v in stats.items():
            print(f"   {k}: {v}")

        # Force finalize (normally would wait for expiry)
        proposal.status = ProposalStatus.ACTIVE
        proposal.voting_ends_at = time.time() - 1  # Expire it
        dao.process_governance()

        print(f"\n   Final status: {proposal.status.value}")

    # Governance stats
    print("\n4. Governance stats:")
    for k, v in dao.get_stats().items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Consensus & Governance module ready!")
    print("=" * 60)
