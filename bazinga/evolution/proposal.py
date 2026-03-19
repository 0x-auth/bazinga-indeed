#!/usr/bin/env python3
"""
Evolution Proposals — Self-Improvement Candidates
===================================================
A proposal is a structured request to modify the BAZINGA codebase.
It can come from:
    - An AI node analyzing performance bottlenecks
    - A human contributor
    - The network itself (federated pattern recognition)

Each proposal includes:
    - What to change (file diffs)
    - Why (description, rationale)
    - Safety metadata (constitution check, ethics verdict, sandbox result)
    - Voting record

Proposals are stored in ~/.bazinga/proposals/ as JSON.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class ProposalType(Enum):
    """Category of proposed change."""
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DOCS = "docs"
    TESTS = "tests"


class ProposalStatus(Enum):
    """Lifecycle of a proposal."""
    DRAFT = "draft"
    CONSTITUTIONAL_CHECK = "constitutional_check"
    ETHICS_REVIEW = "ethics_review"
    SANDBOX_TESTING = "sandbox_testing"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    REVERTED = "reverted"


@dataclass
class FileDiff:
    """A single file change in a proposal."""
    path: str
    old_content: str
    new_content: str

    @property
    def lines_changed(self) -> int:
        old_lines = set(self.old_content.splitlines())
        new_lines = set(self.new_content.splitlines())
        return len(old_lines.symmetric_difference(new_lines))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "old_content": self.old_content,
            "new_content": self.new_content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileDiff":
        return cls(**data)


@dataclass
class Vote:
    """A single vote on a proposal."""
    voter_node_id: str
    approve: bool
    reasoning: str
    phi_coherence: float = 0.0    # Coherence of voter's reasoning
    trust_weight: float = 0.0     # From TrustOracle
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "voter_node_id": self.voter_node_id,
            "approve": self.approve,
            "reasoning": self.reasoning,
            "phi_coherence": self.phi_coherence,
            "trust_weight": self.trust_weight,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vote":
        return cls(**data)


@dataclass
class EvolutionProposal:
    """
    A proposed improvement to the BAZINGA codebase.

    Lifecycle:
        DRAFT → CONSTITUTIONAL_CHECK → ETHICS_REVIEW → SANDBOX_TESTING
        → VOTING → APPROVED/REJECTED → APPLIED/REVERTED
    """
    # Identity
    proposal_id: str = ""
    proposer_node_id: str = ""
    title: str = ""
    description: str = ""
    proposal_type: str = "feature"  # ProposalType value

    # The actual change
    file_diffs: List[Dict] = field(default_factory=list)

    # Interpretability (required for all proposals)
    plain_english_summary: str = ""
    side_effects: List[str] = field(default_factory=list)
    rollback_plan: str = ""
    monitoring_metrics: List[str] = field(default_factory=list)

    # Safety metadata (filled by EvolutionEngine)
    constitution_passes: Optional[bool] = None
    constitution_violations: List[str] = field(default_factory=list)
    ethics_overall: Optional[float] = None
    ethics_dimensions: Dict[str, float] = field(default_factory=dict)
    ethics_warnings: List[str] = field(default_factory=list)
    sandbox_passed: Optional[bool] = None
    sandbox_errors: List[str] = field(default_factory=list)

    # Voting
    votes: List[Dict] = field(default_factory=list)
    vote_deadline: Optional[float] = None

    # Status
    status: str = "draft"  # ProposalStatus value
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Blockchain attestation
    chain_block: Optional[int] = None
    chain_hash: Optional[str] = None

    def __post_init__(self):
        if not self.proposal_id:
            # Generate ID from content hash
            content = f"{self.title}:{self.description}:{time.time()}"
            self.proposal_id = "PROP_" + hashlib.sha256(
                content.encode()
            ).hexdigest()[:12].upper()

    @property
    def total_lines_changed(self) -> int:
        total = 0
        for d in self.file_diffs:
            if isinstance(d, dict):
                old = set(d.get("old_content", "").splitlines())
                new = set(d.get("new_content", "").splitlines())
                total += len(old.symmetric_difference(new))
        return total

    @property
    def modified_files(self) -> List[str]:
        return [d["path"] for d in self.file_diffs if isinstance(d, dict)]

    @property
    def diff_text(self) -> str:
        """Generate a unified-diff-like text for constitution/ethics checks."""
        lines = []
        for d in self.file_diffs:
            if isinstance(d, dict):
                lines.append(f"--- a/{d['path']}")
                lines.append(f"+++ b/{d['path']}")
                for line in d.get("old_content", "").splitlines():
                    lines.append(f"-{line}")
                for line in d.get("new_content", "").splitlines():
                    lines.append(f"+{line}")
        return "\n".join(lines)

    @property
    def vote_objects(self) -> List[Vote]:
        return [Vote.from_dict(v) for v in self.votes]

    @property
    def approval_count(self) -> int:
        return sum(1 for v in self.votes if v.get("approve", False))

    @property
    def rejection_count(self) -> int:
        return sum(1 for v in self.votes if not v.get("approve", True))

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "proposer_node_id": self.proposer_node_id,
            "title": self.title,
            "description": self.description,
            "proposal_type": self.proposal_type,
            "file_diffs": self.file_diffs,
            "plain_english_summary": self.plain_english_summary,
            "side_effects": self.side_effects,
            "rollback_plan": self.rollback_plan,
            "monitoring_metrics": self.monitoring_metrics,
            "constitution_passes": self.constitution_passes,
            "constitution_violations": self.constitution_violations,
            "ethics_overall": self.ethics_overall,
            "ethics_dimensions": self.ethics_dimensions,
            "ethics_warnings": self.ethics_warnings,
            "sandbox_passed": self.sandbox_passed,
            "sandbox_errors": self.sandbox_errors,
            "votes": self.votes,
            "vote_deadline": self.vote_deadline,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "chain_block": self.chain_block,
            "chain_hash": self.chain_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvolutionProposal":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Proposal Store
# =============================================================================

class ProposalStore:
    """
    Persists proposals to ~/.bazinga/proposals/ as JSON files.

    Usage:
        store = ProposalStore()
        store.save(proposal)
        loaded = store.load("PROP_ABC123")
        active = store.list_active()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".bazinga" / "proposals"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save(self, proposal: EvolutionProposal):
        """Save proposal to disk."""
        proposal.updated_at = time.time()
        path = self.data_dir / f"{proposal.proposal_id}.json"
        with open(path, "w") as f:
            json.dump(proposal.to_dict(), f, indent=2)

    def load(self, proposal_id: str) -> Optional[EvolutionProposal]:
        """Load proposal from disk."""
        path = self.data_dir / f"{proposal_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return EvolutionProposal.from_dict(data)

    def list_all(self) -> List[EvolutionProposal]:
        """List all proposals."""
        proposals = []
        for path in sorted(self.data_dir.glob("PROP_*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                proposals.append(EvolutionProposal.from_dict(data))
            except Exception:
                continue
        return proposals

    def list_active(self) -> List[EvolutionProposal]:
        """List proposals that are still in progress (not final)."""
        final_statuses = {"approved", "rejected", "applied", "reverted"}
        return [
            p for p in self.list_all()
            if p.status not in final_statuses
        ]

    def list_by_status(self, status: str) -> List[EvolutionProposal]:
        """List proposals with a specific status."""
        return [p for p in self.list_all() if p.status == status]

    def delete(self, proposal_id: str) -> bool:
        """Delete a proposal file."""
        path = self.data_dir / f"{proposal_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
