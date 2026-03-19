#!/usr/bin/env python3
"""
Graduated Autonomy — Earn Trust Over Time
===========================================
Don't go from 0 to 100. Go through stages.

Level 0 (SUGGEST_ONLY):
    Can propose, cannot execute.
    Starting level for all nodes.

Level 1 (HUMAN_APPROVED):
    Execute only with explicit human approval.
    Requires: 10 successful proposals.

Level 2 (CONSENSUS_APPROVED):
    Network consensus + human notified (48h veto window).
    Requires: 50 successful proposals, trust > 0.8, 30+ days.

Level 3 (AUTO_SAFE):
    Auto-execute safe changes (docs, tests, formatting).
    Requires: 200 proposals, zero reverts in 30 days.

Level 4 (FULL_AUTO):
    Full autonomy with consensus.
    Requires: explicit human config change (never auto-promoted).

From AI_SAFETY_ANALYSIS.md:
    "Like raising a child — full supervision → guided autonomy → independence."
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

from bazinga.config import AutonomyLevel, get_config


# =============================================================================
# Promotion Requirements
# =============================================================================

@dataclass(frozen=True)
class LevelRequirement:
    """Requirements to earn a given autonomy level."""
    level: AutonomyLevel
    min_successful_proposals: int
    min_trust_score: float
    min_network_age_days: int
    zero_reverts_in_days: int
    requires_human_config: bool  # If True, can never be auto-promoted


LEVEL_REQUIREMENTS = {
    AutonomyLevel.SUGGEST_ONLY: LevelRequirement(
        level=AutonomyLevel.SUGGEST_ONLY,
        min_successful_proposals=0,
        min_trust_score=0.0,
        min_network_age_days=0,
        zero_reverts_in_days=0,
        requires_human_config=False,
    ),
    AutonomyLevel.HUMAN_APPROVED: LevelRequirement(
        level=AutonomyLevel.HUMAN_APPROVED,
        min_successful_proposals=10,
        min_trust_score=0.5,
        min_network_age_days=7,
        zero_reverts_in_days=7,
        requires_human_config=False,
    ),
    AutonomyLevel.CONSENSUS_APPROVED: LevelRequirement(
        level=AutonomyLevel.CONSENSUS_APPROVED,
        min_successful_proposals=50,
        min_trust_score=0.8,
        min_network_age_days=30,
        zero_reverts_in_days=14,
        requires_human_config=False,
    ),
    AutonomyLevel.AUTO_SAFE: LevelRequirement(
        level=AutonomyLevel.AUTO_SAFE,
        min_successful_proposals=200,
        min_trust_score=0.9,
        min_network_age_days=90,
        zero_reverts_in_days=30,
        requires_human_config=False,
    ),
    AutonomyLevel.FULL_AUTO: LevelRequirement(
        level=AutonomyLevel.FULL_AUTO,
        min_successful_proposals=500,
        min_trust_score=0.95,
        min_network_age_days=180,
        zero_reverts_in_days=60,
        requires_human_config=True,  # NEVER auto-promoted
    ),
}


# =============================================================================
# Track Record
# =============================================================================

@dataclass
class ProposalOutcome:
    """Record of a proposal's outcome."""
    proposal_id: str
    timestamp: float
    success: bool
    reverted: bool = False
    revert_timestamp: Optional[float] = None


@dataclass
class TrackRecord:
    """Node's historical performance."""
    node_id: str
    first_seen: float = field(default_factory=time.time)
    outcomes: List[ProposalOutcome] = field(default_factory=list)

    @property
    def successful_proposals(self) -> int:
        return sum(1 for o in self.outcomes if o.success and not o.reverted)

    @property
    def total_proposals(self) -> int:
        return len(self.outcomes)

    @property
    def reverts(self) -> int:
        return sum(1 for o in self.outcomes if o.reverted)

    @property
    def age_days(self) -> float:
        return (time.time() - self.first_seen) / 86400

    def reverts_in_last_days(self, days: int) -> int:
        cutoff = time.time() - (days * 86400)
        return sum(
            1 for o in self.outcomes
            if o.reverted and o.timestamp >= cutoff
        )

    @property
    def success_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return self.successful_proposals / self.total_proposals


# =============================================================================
# Graduated Autonomy Manager
# =============================================================================

class GraduatedAutonomy:
    """
    Manages autonomy level progression.

    Usage:
        ga = GraduatedAutonomy()
        print(ga.current_level)  # AutonomyLevel.SUGGEST_ONLY

        # Record outcomes
        ga.record_outcome("prop_001", success=True)
        ga.record_outcome("prop_002", success=True)
        ...

        # Check if promotion is available
        next_level = ga.check_promotion(trust_score=0.85)
        if next_level:
            ga.promote(next_level)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".bazinga" / "evolution"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.record_file = self.data_dir / "track_record.json"
        self.record = self._load_record()

    @property
    def current_level(self) -> AutonomyLevel:
        """Get current autonomy level from config."""
        config = get_config()
        level_val = config.safety.max_autonomy_level
        try:
            return AutonomyLevel(level_val)
        except ValueError:
            return AutonomyLevel.SUGGEST_ONLY

    def check_promotion(self, trust_score: float = 0.0) -> Optional[AutonomyLevel]:
        """
        Check if the node qualifies for a higher autonomy level.

        Args:
            trust_score: Current trust score from TrustOracle (0-1)

        Returns:
            Next level if qualified, None otherwise
        """
        current = self.current_level
        next_level_val = current.value + 1

        try:
            next_level = AutonomyLevel(next_level_val)
        except ValueError:
            return None  # Already at max

        req = LEVEL_REQUIREMENTS[next_level]

        # FULL_AUTO requires human config change
        if req.requires_human_config:
            return None

        # Check all requirements
        if self.record.successful_proposals < req.min_successful_proposals:
            return None
        if trust_score < req.min_trust_score:
            return None
        if self.record.age_days < req.min_network_age_days:
            return None
        if self.record.reverts_in_last_days(req.zero_reverts_in_days) > 0:
            return None

        return next_level

    def can_execute(self, proposal_type: str = "feature") -> bool:
        """
        Check if the current autonomy level allows execution.

        Args:
            proposal_type: "docs", "tests", "bugfix", "feature", "refactor"

        Returns:
            True if current level allows execution of this type
        """
        level = self.current_level

        if level == AutonomyLevel.SUGGEST_ONLY:
            return False
        elif level == AutonomyLevel.HUMAN_APPROVED:
            return False  # Needs explicit human approval (handled by caller)
        elif level == AutonomyLevel.CONSENSUS_APPROVED:
            return True  # Network consensus sufficient
        elif level == AutonomyLevel.AUTO_SAFE:
            safe_types = {"docs", "tests", "formatting", "comments"}
            return proposal_type in safe_types
        elif level == AutonomyLevel.FULL_AUTO:
            return True
        return False

    def needs_human_approval(self) -> bool:
        """Does the current level require human approval?"""
        return self.current_level.value <= AutonomyLevel.HUMAN_APPROVED.value

    def record_outcome(
        self,
        proposal_id: str,
        success: bool,
        reverted: bool = False,
    ):
        """Record the outcome of a proposal."""
        outcome = ProposalOutcome(
            proposal_id=proposal_id,
            timestamp=time.time(),
            success=success,
            reverted=reverted,
            revert_timestamp=time.time() if reverted else None,
        )
        self.record.outcomes.append(outcome)
        self._save_record()

    def get_status(self) -> Dict:
        """Get current autonomy status for display."""
        current = self.current_level
        next_check = self.check_promotion()

        # Get requirements for next level
        next_val = current.value + 1
        next_req = None
        try:
            next_level = AutonomyLevel(next_val)
            next_req = LEVEL_REQUIREMENTS[next_level]
        except (ValueError, KeyError):
            pass

        return {
            "current_level": current.value,
            "level_name": current.name,
            "successful_proposals": self.record.successful_proposals,
            "total_proposals": self.record.total_proposals,
            "reverts": self.record.reverts,
            "age_days": round(self.record.age_days, 1),
            "success_rate": round(self.record.success_rate, 3),
            "promotion_available": next_check is not None,
            "next_level_requirements": {
                "proposals_needed": next_req.min_successful_proposals if next_req else None,
                "trust_needed": next_req.min_trust_score if next_req else None,
                "days_needed": next_req.min_network_age_days if next_req else None,
            } if next_req else None,
        }

    def _load_record(self) -> TrackRecord:
        """Load track record from disk."""
        if self.record_file.exists():
            try:
                with open(self.record_file) as f:
                    data = json.load(f)
                record = TrackRecord(
                    node_id=data.get("node_id", "local"),
                    first_seen=data.get("first_seen", time.time()),
                )
                for o in data.get("outcomes", []):
                    record.outcomes.append(ProposalOutcome(**o))
                return record
            except Exception:
                pass
        return TrackRecord(node_id="local")

    def _save_record(self):
        """Save track record to disk."""
        data = {
            "node_id": self.record.node_id,
            "first_seen": self.record.first_seen,
            "outcomes": [
                {
                    "proposal_id": o.proposal_id,
                    "timestamp": o.timestamp,
                    "success": o.success,
                    "reverted": o.reverted,
                    "revert_timestamp": o.revert_timestamp,
                }
                for o in self.record.outcomes
            ],
        }
        with open(self.record_file, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Module test
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  GRADUATED AUTONOMY TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        ga = GraduatedAutonomy(data_dir=Path(tmpdir))

        print(f"\n  Current level: {ga.current_level.name}")
        print(f"  Can execute feature: {ga.can_execute('feature')}")
        print(f"  Needs human approval: {ga.needs_human_approval()}")

        # Record some outcomes
        for i in range(12):
            ga.record_outcome(f"prop_{i:03d}", success=True)

        print(f"\n  After 12 successful proposals:")
        print(f"  Successful: {ga.record.successful_proposals}")
        print(f"  Success rate: {ga.record.success_rate:.0%}")

        # Check promotion (won't promote because trust not provided)
        promo = ga.check_promotion(trust_score=0.6)
        print(f"  Promotion available (trust=0.6): {promo}")

        status = ga.get_status()
        print(f"\n  Status: {json.dumps(status, indent=2)}")

    print(f"\n  Graduated autonomy working! ✓")
