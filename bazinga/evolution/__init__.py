"""
BAZINGA Evolution — Autonomous Self-Improvement with Safety
============================================================
AI nodes propose improvements → network votes → sandbox tests → auto-merge.

Safety first:
    - Constitutional constraints are immutable (frozenset)
    - PhiEthics evaluates values, not just quality
    - Graduated autonomy: earn trust over time (Level 0→4)
    - All changes are reversible (git-backed)
    - Human override is always available

Usage:
    from bazinga.evolution import EvolutionEngine, EvolutionProposal
    from bazinga.evolution import ConstitutionEnforcer, AutonomyLevel

    engine = EvolutionEngine()
    proposal = EvolutionProposal(title="...", file_diffs=[...])
    result = engine.run_pipeline(proposal)
"""

from bazinga.config import AutonomyLevel
from bazinga.evolution.constitution import ConstitutionEnforcer, CONSTITUTION
from bazinga.evolution.phi_ethics import PhiEthics
from bazinga.evolution.proposal import EvolutionProposal, ProposalStore, Vote
from bazinga.evolution.voting import PhiCoherenceVoting
from bazinga.evolution.graduated import GraduatedAutonomy
from bazinga.evolution.sandbox import ProposalSandbox
from bazinga.evolution.engine import EvolutionEngine

__all__ = [
    'AutonomyLevel',
    'ConstitutionEnforcer',
    'CONSTITUTION',
    'PhiEthics',
    'EvolutionProposal',
    'ProposalStore',
    'Vote',
    'PhiCoherenceVoting',
    'GraduatedAutonomy',
    'ProposalSandbox',
    'EvolutionEngine',
]
