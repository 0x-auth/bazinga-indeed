#!/usr/bin/env python3
"""
Evolution Engine — Self-Improvement Orchestrator
==================================================
Full pipeline for autonomous code evolution:

    1. receive_proposal()  → validate format, constitutional check
    2. ethics_review()     → PhiEthics evaluation
    3. sandbox_test()      → run in isolated environment
    4. open_voting()       → broadcast to network for phi-weighted votes
    5. tally_votes()       → compute weighted result
    6. apply_or_reject()   → if approved + autonomy level permits
    7. attest_on_chain()   → record on Darmiyan blockchain

Safety is enforced at every step:
    - Constitution is checked FIRST (hard reject on violation)
    - Ethics is checked SECOND (soft reject on low scores)
    - Sandbox isolates testing (never touches real code)
    - Graduated autonomy gates execution
    - Human override is always available

From SELF_PROPOSAL_SYSTEM.md + AI_SAFETY_ANALYSIS.md:
    "The goal is not control. The goal is responsible co-evolution."
"""

import time
from typing import Optional

from bazinga.evolution.constitution import ConstitutionEnforcer
from bazinga.evolution.phi_ethics import PhiEthics
from bazinga.evolution.sandbox import ProposalSandbox, SandboxResult
from bazinga.evolution.graduated import GraduatedAutonomy
from bazinga.evolution.proposal import (
    EvolutionProposal, ProposalStore, ProposalStatus, Vote,
)
from bazinga.evolution.voting import PhiCoherenceVoting, TallyResult


class EvolutionEngine:
    """
    Orchestrator for the self-proposal system.

    Usage:
        engine = EvolutionEngine()

        # Submit a proposal
        proposal = EvolutionProposal(
            title="Improve search performance",
            description="Add FAISS indexing for O(log n) search",
            file_diffs=[{
                "path": "bazinga/kb.py",
                "old_content": old,
                "new_content": new,
            }],
            plain_english_summary="This adds FAISS vector indexing...",
            rollback_plan="Set KB_USE_FAISS=false in config",
        )

        # Run through pipeline
        proposal = engine.receive_proposal(proposal)
        if proposal.status == "ethics_review":
            proposal = engine.ethics_review(proposal.proposal_id)
        if proposal.status == "sandbox_testing":
            proposal = engine.sandbox_test(proposal.proposal_id)
        if proposal.status == "voting":
            engine.open_voting(proposal.proposal_id)
            # ... votes come in ...
            result = engine.tally_votes(proposal.proposal_id)
            if result.approved:
                engine.apply(proposal.proposal_id)
    """

    def __init__(
        self,
        store: Optional[ProposalStore] = None,
        constitution: Optional[ConstitutionEnforcer] = None,
        ethics: Optional[PhiEthics] = None,
        sandbox: Optional[ProposalSandbox] = None,
        voting: Optional[PhiCoherenceVoting] = None,
        graduated: Optional[GraduatedAutonomy] = None,
    ):
        self.store = store or ProposalStore()
        self.constitution = constitution or ConstitutionEnforcer()
        self.ethics = ethics or PhiEthics()
        self.sandbox = sandbox or ProposalSandbox()
        self.voting = voting or PhiCoherenceVoting()
        self.graduated = graduated or GraduatedAutonomy()

    # =========================================================================
    # Step 1: Receive and validate
    # =========================================================================

    def receive_proposal(self, proposal: EvolutionProposal) -> EvolutionProposal:
        """
        Validate and register a new proposal.

        Checks:
        1. Has required fields (title, description, diffs)
        2. Passes constitutional constraints (hard reject)
        3. If passes, status → ETHICS_REVIEW

        Returns the updated proposal.
        """
        # Validate required fields
        if not proposal.title:
            proposal.status = ProposalStatus.REJECTED.value
            proposal.constitution_violations = ["Missing title"]
            self.store.save(proposal)
            return proposal

        if not proposal.file_diffs:
            proposal.status = ProposalStatus.REJECTED.value
            proposal.constitution_violations = ["No file changes provided"]
            self.store.save(proposal)
            return proposal

        # Constitutional check
        proposal.status = ProposalStatus.CONSTITUTIONAL_CHECK.value
        passes, violations = self.constitution.validate(
            proposal.diff_text,
            proposal.modified_files,
        )

        proposal.constitution_passes = passes
        proposal.constitution_violations = violations

        if not passes:
            proposal.status = ProposalStatus.REJECTED.value
            self.store.save(proposal)
            return proposal

        # Passed constitution → move to ethics
        proposal.status = ProposalStatus.ETHICS_REVIEW.value
        self.store.save(proposal)
        return proposal

    # =========================================================================
    # Step 2: Ethics review
    # =========================================================================

    def ethics_review(self, proposal_id: str) -> EvolutionProposal:
        """
        Run PhiEthics evaluation on a proposal.

        If ethics score is too low, proposal is rejected.
        Otherwise, status → SANDBOX_TESTING
        """
        proposal = self.store.load(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        verdict = self.ethics.evaluate(
            proposal.diff_text,
            proposal.description,
        )

        proposal.ethics_overall = verdict.overall
        proposal.ethics_dimensions = verdict.dimensions
        proposal.ethics_warnings = verdict.warnings

        if not verdict.passes:
            proposal.status = ProposalStatus.REJECTED.value
            self.store.save(proposal)
            return proposal

        proposal.status = ProposalStatus.SANDBOX_TESTING.value
        self.store.save(proposal)
        return proposal

    # =========================================================================
    # Step 3: Sandbox test
    # =========================================================================

    def sandbox_test(
        self,
        proposal_id: str,
        run_tests: bool = True,
    ) -> EvolutionProposal:
        """
        Run proposal in isolated sandbox.

        If sandbox fails, proposal is rejected.
        Otherwise, status → VOTING
        """
        proposal = self.store.load(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        result = self.sandbox.test_proposal(
            proposal.file_diffs,
            run_tests=run_tests,
        )

        proposal.sandbox_passed = result.passed
        proposal.sandbox_errors = result.errors

        if not result.passed:
            proposal.status = ProposalStatus.REJECTED.value
            self.store.save(proposal)
            return proposal

        proposal.status = ProposalStatus.VOTING.value
        proposal.vote_deadline = time.time() + (24 * 3600)  # 24h default
        self.store.save(proposal)
        return proposal

    # =========================================================================
    # Step 4: Open voting
    # =========================================================================

    def open_voting(
        self,
        proposal_id: str,
        duration_hours: float = 24.0,
    ) -> EvolutionProposal:
        """
        Open a proposal for network voting.

        In practice, this broadcasts the proposal to peers.
        For now, it just sets the deadline and saves.
        """
        proposal = self.store.load(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal.status = ProposalStatus.VOTING.value
        proposal.vote_deadline = time.time() + (duration_hours * 3600)
        self.store.save(proposal)
        return proposal

    def cast_vote(
        self,
        proposal_id: str,
        vote: Vote,
    ) -> bool:
        """Cast a vote on a proposal."""
        proposal = self.store.load(proposal_id)
        if proposal is None:
            return False
        if proposal.status != ProposalStatus.VOTING.value:
            return False

        accepted = self.voting.cast_vote(proposal, vote)
        if accepted:
            self.store.save(proposal)
        return accepted

    # =========================================================================
    # Step 5: Tally votes
    # =========================================================================

    def tally_votes(self, proposal_id: str) -> TallyResult:
        """
        Tally votes and determine if proposal is approved.

        Also checks for Sybil attacks.
        """
        proposal = self.store.load(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        # Check for Sybil
        sybil_score = self.voting.detect_sybil(proposal)
        if sybil_score > 0.5:
            proposal.status = ProposalStatus.REJECTED.value
            proposal.ethics_warnings.append(
                f"Sybil attack detected (score: {sybil_score:.2f})"
            )
            self.store.save(proposal)
            return TallyResult(
                weighted_approval=0.0,
                total_votes=len(proposal.votes),
                approve_count=0,
                reject_count=0,
                high_trust_approvers=0,
                threshold=0.618,
                approved=False,
                reason=f"Sybil attack detected (score: {sybil_score:.2f})",
            )

        result = self.voting.tally(proposal)

        if result.approved:
            proposal.status = ProposalStatus.APPROVED.value
        else:
            proposal.status = ProposalStatus.REJECTED.value

        self.store.save(proposal)
        return result

    # =========================================================================
    # Step 6: Apply or reject
    # =========================================================================

    def apply(self, proposal_id: str) -> bool:
        """
        Apply an approved proposal.

        Checks graduated autonomy level:
        - Level 0: Cannot apply (suggest only)
        - Level 1: Needs human approval (returns False, human must call apply_force)
        - Level 2+: Can apply if consensus approved

        Returns True if applied, False if blocked.
        """
        proposal = self.store.load(proposal_id)
        if proposal is None:
            return False
        if proposal.status != ProposalStatus.APPROVED.value:
            return False

        # Check autonomy level
        if self.graduated.needs_human_approval():
            # Don't auto-apply — human must review
            return False

        if not self.graduated.can_execute(proposal.proposal_type):
            return False

        # Apply the changes
        success = self._apply_diffs(proposal)

        if success:
            proposal.status = ProposalStatus.APPLIED.value
            self.graduated.record_outcome(proposal_id, success=True)
        else:
            proposal.status = ProposalStatus.REJECTED.value
            self.graduated.record_outcome(proposal_id, success=False)

        self.store.save(proposal)
        return success

    def apply_force(self, proposal_id: str) -> bool:
        """
        Human-approved application (bypasses autonomy check).
        This is the "human override always" constitutional right.
        """
        proposal = self.store.load(proposal_id)
        if proposal is None:
            return False
        if proposal.status not in (
            ProposalStatus.APPROVED.value,
            ProposalStatus.VOTING.value,
        ):
            return False

        success = self._apply_diffs(proposal)
        if success:
            proposal.status = ProposalStatus.APPLIED.value
            self.graduated.record_outcome(proposal_id, success=True)
        self.store.save(proposal)
        return success

    def revert(self, proposal_id: str) -> bool:
        """Revert an applied proposal."""
        proposal = self.store.load(proposal_id)
        if proposal is None:
            return False
        if proposal.status != ProposalStatus.APPLIED.value:
            return False

        # Revert by writing back old content
        success = self._revert_diffs(proposal)
        if success:
            proposal.status = ProposalStatus.REVERTED.value
            self.graduated.record_outcome(proposal_id, success=False, reverted=True)
        self.store.save(proposal)
        return success

    def _apply_diffs(self, proposal: EvolutionProposal) -> bool:
        """Apply file diffs to the real codebase."""
        try:
            for diff in proposal.file_diffs:
                if isinstance(diff, dict):
                    path = diff["path"]
                    # Resolve relative to repo root
                    from pathlib import Path as P
                    repo = P(__file__).parent.parent.parent
                    target = repo / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(diff["new_content"])
            return True
        except Exception:
            return False

    def _revert_diffs(self, proposal: EvolutionProposal) -> bool:
        """Revert file diffs to old content."""
        try:
            for diff in proposal.file_diffs:
                if isinstance(diff, dict):
                    path = diff["path"]
                    from pathlib import Path as P
                    repo = P(__file__).parent.parent.parent
                    target = repo / path
                    if diff.get("old_content"):
                        target.write_text(diff["old_content"])
                    elif target.exists():
                        target.unlink()
            return True
        except Exception:
            return False

    # =========================================================================
    # Pipeline shortcut
    # =========================================================================

    def run_pipeline(
        self,
        proposal: EvolutionProposal,
        auto_apply: bool = False,
    ) -> EvolutionProposal:
        """
        Run the full pipeline on a proposal.

        Returns the proposal at whatever stage it reached.
        If auto_apply=True and autonomy permits, applies immediately.
        """
        # Step 1: Receive
        proposal = self.receive_proposal(proposal)
        if proposal.status == ProposalStatus.REJECTED.value:
            return proposal

        # Step 2: Ethics
        proposal = self.ethics_review(proposal.proposal_id)
        if proposal.status == ProposalStatus.REJECTED.value:
            return proposal

        # Step 3: Sandbox
        proposal = self.sandbox_test(proposal.proposal_id, run_tests=False)
        if proposal.status == ProposalStatus.REJECTED.value:
            return proposal

        # Steps 4-6 require network interaction (voting)
        # For now, return at VOTING stage
        return proposal

    # =========================================================================
    # Queries
    # =========================================================================

    def get_proposal(self, proposal_id: str) -> Optional[EvolutionProposal]:
        """Get a proposal by ID."""
        return self.store.load(proposal_id)

    def list_proposals(self, status: Optional[str] = None):
        """List proposals, optionally filtered by status."""
        if status:
            return self.store.list_by_status(status)
        return self.store.list_all()

    def get_stats(self) -> dict:
        """Get evolution engine statistics."""
        all_proposals = self.store.list_all()
        by_status = {}
        for p in all_proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1

        return {
            "total_proposals": len(all_proposals),
            "by_status": by_status,
            "autonomy_level": self.graduated.current_level.name,
            "autonomy_status": self.graduated.get_status(),
        }


# =============================================================================
# Module test
# =============================================================================

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("=" * 60)
    print("  EVOLUTION ENGINE TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProposalStore(data_dir=Path(tmpdir) / "proposals")
        graduated = GraduatedAutonomy(data_dir=Path(tmpdir) / "autonomy")

        engine = EvolutionEngine(store=store, graduated=graduated)

        # Test 1: Valid proposal through pipeline
        print("\n  Test 1: Valid proposal")
        p1 = EvolutionProposal(
            title="Add hello function",
            description="Adds a simple hello world function",
            file_diffs=[{
                "path": "bazinga/hello.py",
                "old_content": "",
                "new_content": "def hello():\n    return 'world'\n",
            }],
            plain_english_summary="Adds hello function",
            rollback_plan="Delete bazinga/hello.py",
        )
        result = engine.run_pipeline(p1)
        print(f"    Status: {result.status}")
        print(f"    Constitution: {result.constitution_passes}")
        print(f"    Ethics: {result.ethics_overall:.2f}")
        print(f"    Sandbox: {result.sandbox_passed}")

        # Test 2: Constitutional violation
        print("\n  Test 2: Constitutional violation")
        p2 = EvolutionProposal(
            title="Modify constitution",
            description="Change safety bounds",
            file_diffs=[{
                "path": "bazinga/evolution/constitution.py",
                "old_content": "CONSTITUTION = frozenset({...})",
                "new_content": "CONSTITUTION = frozenset({})",
            }],
        )
        result2 = engine.run_pipeline(p2)
        print(f"    Status: {result2.status}")
        print(f"    Violations: {result2.constitution_violations[:2]}")

        # Test 3: Voting
        print("\n  Test 3: Voting")
        p3 = engine.store.load(result.proposal_id)
        if p3 and p3.status == "voting":
            # Cast 3 high-trust approving votes
            for i in range(3):
                engine.cast_vote(p3.proposal_id, Vote(
                    voter_node_id=f"node_{i}",
                    approve=True,
                    reasoning=f"Looks good, simple and safe change #{i}",
                    phi_coherence=0.8,
                    trust_weight=0.7,
                ))

            tally = engine.tally_votes(p3.proposal_id)
            print(f"    Tally: {tally.summary}")

        # Stats
        print(f"\n  Stats: {engine.get_stats()}")

    print(f"\n  Evolution Engine working! ✓")
