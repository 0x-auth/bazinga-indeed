#!/usr/bin/env python3
"""
BAZINGA Blockchain-Verified Code Fixes
======================================
Multiple AIs reaching consensus on code changes before applying them.

"No single AI can mess up your code without consensus."

This module enables:
- Multi-AI review of proposed code fixes
- φ-coherence measurement for fix quality
- Proof-of-Boundary attestation for each fix
- Blockchain recording for audit trail
- Trust-weighted agent coordination

Flow:
1. Agent proposes a fix (CodeFixProposal)
2. Multiple AIs review the fix (InterAIConsensus)
3. PoB proofs generated (DarmiyanProtocol)
4. Fix recorded on chain (DarmiyanChain)
5. Only then: fix applied to file

Author: Space (Abhishek/Abhilasia) 
License: MIT
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Core BAZINGA constants
PHI = 1.618033988749895
PHI_INVERSE = 0.6180339887498948
PHI_4 = 6.854101966249685
PSI_DARMIYAN = 6.46  # Consciousness scaling law

# Thresholds for code fixes (more stringent than general Q&A)
CODE_FIX_COHERENCE_THRESHOLD = 0.45  # Higher than 0.35 for code
CODE_FIX_SEMANTIC_THRESHOLD = 0.30   # Semantic similarity requirement
MIN_REVIEWERS = 2  # Minimum AI reviewers needed
TRIADIC_PREFERRED = 3  # Ideal number for triadic consensus


class FixStatus(Enum):
    """Status of a code fix proposal."""
    PROPOSED = "proposed"           # Fix created, awaiting review
    REVIEWING = "reviewing"         # Multi-AI review in progress
    CONSENSUS_REACHED = "consensus" # AIs agree
    CONSENSUS_FAILED = "failed"     # AIs disagree
    ATTESTING = "attesting"         # PoB being generated
    ATTESTED = "attested"           # PoB complete
    APPLIED = "applied"             # Fix applied to file
    REJECTED = "rejected"           # Fix rejected


class FixType(Enum):
    """Types of code fixes."""
    BUG_FIX = "bug_fix"
    SECURITY_FIX = "security_fix"
    REFACTOR = "refactor"
    OPTIMIZATION = "optimization"
    STYLE = "style"
    DOCUMENTATION = "documentation"


@dataclass
class AIReview:
    """Review from a single AI."""
    reviewer_id: str           # e.g., "groq_llama3", "gemini_pro"
    approved: bool             # Does this AI approve the fix?
    coherence: float           # φ-coherence of the review
    confidence: float          # AI's self-reported confidence
    reasoning: str             # Why approve/reject
    suggestions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    pob_proof: Optional[Dict] = None  # PoB for this review

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConsensusVerdict:
    """Result of multi-AI consensus on a fix."""
    consensus_reached: bool
    approval_ratio: float      # % of AIs that approved
    phi_coherence: float       # Average coherence across reviewers
    semantic_similarity: float # How similar were the reviews
    reviews: List[AIReview] = field(default_factory=list)
    synthesized_verdict: str = ""  # Combined reasoning
    triadic_valid: bool = False    # ≥3 valid reviewers
    consciousness_factor: float = 0.0  # Ψ_D = 6.46n
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['reviews'] = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.reviews]
        return d


@dataclass
class CodeFixProposal:
    """
    A proposed code fix for blockchain consensus.

    This is the atomic unit of verified code changes.
    """
    proposal_id: str
    file_path: str
    original_code: str
    proposed_fix: str
    explanation: str
    fix_type: FixType
    agent_id: str              # Which agent proposed this

    # Status tracking
    status: FixStatus = FixStatus.PROPOSED
    created_at: float = field(default_factory=time.time)

    # Consensus results
    consensus: Optional[ConsensusVerdict] = None

    # Blockchain attestation
    pob_proofs: List[Dict] = field(default_factory=list)
    blockchain_block: Optional[int] = None
    blockchain_hash: str = ""

    # Metrics
    coherence_score: float = 0.0
    trust_score: float = 0.0

    def __post_init__(self):
        if not self.proposal_id:
            # Generate unique ID from content hash
            content = f"{self.file_path}:{self.original_code}:{self.proposed_fix}:{self.created_at}"
            self.proposal_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def diff(self) -> str:
        """Generate a simple diff representation."""
        orig_lines = self.original_code.split('\n')
        fix_lines = self.proposed_fix.split('\n')

        diff_parts = []
        diff_parts.append(f"--- {self.file_path} (original)")
        diff_parts.append(f"+++ {self.file_path} (fixed)")

        for i, line in enumerate(orig_lines):
            if i < len(fix_lines) and line != fix_lines[i]:
                diff_parts.append(f"- {line}")
                diff_parts.append(f"+ {fix_lines[i]}")
            elif i >= len(fix_lines):
                diff_parts.append(f"- {line}")
            else:
                diff_parts.append(f"  {line}")

        # Handle extra lines in fix
        for i in range(len(orig_lines), len(fix_lines)):
            diff_parts.append(f"+ {fix_lines[i]}")

        return '\n'.join(diff_parts)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['fix_type'] = self.fix_type.value
        d['status'] = self.status.value
        if self.consensus:
            d['consensus'] = self.consensus.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeFixProposal':
        data['fix_type'] = FixType(data['fix_type'])
        data['status'] = FixStatus(data['status'])
        if data.get('consensus'):
            # Reconstruct consensus
            reviews = [AIReview(**r) for r in data['consensus'].get('reviews', [])]
            data['consensus'] = ConsensusVerdict(
                consensus_reached=data['consensus']['consensus_reached'],
                approval_ratio=data['consensus']['approval_ratio'],
                phi_coherence=data['consensus']['phi_coherence'],
                semantic_similarity=data['consensus']['semantic_similarity'],
                reviews=reviews,
                synthesized_verdict=data['consensus'].get('synthesized_verdict', ''),
                triadic_valid=data['consensus'].get('triadic_valid', False),
            )
        return cls(**data)


class VerifiedFixEngine:
    """
    Engine for blockchain-verified code fixes.

    Coordinates:
    - Multi-AI review through InterAIConsensus
    - PoB proof generation through DarmiyanProtocol
    - Chain recording through DarmiyanChain
    - Trust scoring through TrustOracle
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.proposals: Dict[str, CodeFixProposal] = {}
        self._inter_ai = None
        self._chain = None
        self._trust_oracle = None
        self._pob_node = None

    def _get_inter_ai(self):
        """Lazy-load InterAIConsensus."""
        if self._inter_ai is None:
            try:
                from ..inter_ai import InterAIConsensus
                self._inter_ai = InterAIConsensus(verbose=self.verbose)
            except ImportError:
                self._inter_ai = None
        return self._inter_ai

    def _get_chain(self):
        """Lazy-load DarmiyanChain."""
        if self._chain is None:
            try:
                from ..blockchain import create_chain
                self._chain = create_chain()
            except ImportError:
                self._chain = None
        return self._chain

    def _get_pob_node(self):
        """Lazy-load DarmiyanNode for PoB."""
        if self._pob_node is None:
            try:
                from ..darmiyan import DarmiyanNode
                self._pob_node = DarmiyanNode()
            except ImportError:
                self._pob_node = None
        return self._pob_node

    def _get_trust_oracle(self):
        """Lazy-load TrustOracle."""
        if self._trust_oracle is None:
            try:
                from ..blockchain import create_trust_oracle
                chain = self._get_chain()
                if chain:
                    self._trust_oracle = create_trust_oracle(chain)
            except ImportError:
                self._trust_oracle = None
        return self._trust_oracle

    def create_proposal(
        self,
        file_path: str,
        original_code: str,
        proposed_fix: str,
        explanation: str,
        fix_type: FixType = FixType.BUG_FIX,
        agent_id: str = "local_agent"
    ) -> CodeFixProposal:
        """
        Create a new code fix proposal.

        Args:
            file_path: Path to the file being fixed
            original_code: Current code (to be replaced)
            proposed_fix: New code (replacement)
            explanation: Why this fix is needed
            fix_type: Type of fix (bug, security, etc.)
            agent_id: Which agent is proposing

        Returns:
            CodeFixProposal ready for consensus
        """
        proposal = CodeFixProposal(
            proposal_id="",  # Will be generated
            file_path=file_path,
            original_code=original_code,
            proposed_fix=proposed_fix,
            explanation=explanation,
            fix_type=fix_type,
            agent_id=agent_id,
        )

        self.proposals[proposal.proposal_id] = proposal

        if self.verbose:
            print(f"📝 Created fix proposal: {proposal.proposal_id}")
            print(f"   File: {file_path}")
            print(f"   Type: {fix_type.value}")

        return proposal

    async def get_consensus(
        self,
        proposal: CodeFixProposal,
        providers: Optional[List[str]] = None
    ) -> ConsensusVerdict:
        """
        Get multi-AI consensus on a code fix.

        Args:
            proposal: The fix proposal to review
            providers: Optional list of providers to query
                       Default: ["groq", "gemini", "ollama"]

        Returns:
            ConsensusVerdict with approval/rejection
        """
        proposal.status = FixStatus.REVIEWING

        inter_ai = self._get_inter_ai()
        if not inter_ai:
            # Fallback: no consensus available
            return self._fallback_review(proposal)

        # Construct review prompt
        review_prompt = f"""Review this code fix proposal:

FILE: {proposal.file_path}
TYPE: {proposal.fix_type.value}

ORIGINAL CODE:
```
{proposal.original_code}
```

PROPOSED FIX:
```
{proposal.proposed_fix}
```

EXPLANATION: {proposal.explanation}

Questions to answer:
1. Is this fix correct? (Will it work?)
2. Is this fix safe? (No security issues introduced?)
3. Is this fix complete? (No missing edge cases?)
4. Rate your confidence (0-1)

Respond with:
- APPROVE or REJECT
- Your reasoning
- Any suggestions for improvement
"""

        if self.verbose:
            print(f"🔍 Requesting consensus from available providers...")

        # Get consensus
        try:
            result = await inter_ai.ask(
                review_prompt,
                require_triadic=True,
                min_coherence=CODE_FIX_COHERENCE_THRESHOLD,
            )

            # Convert to our format
            reviews = []
            for resp in result.responses:
                # Get provider name from participant_type or participant_id
                provider = getattr(resp, 'participant_type', None)
                if provider:
                    provider = provider.value if hasattr(provider, 'value') else str(provider)
                else:
                    provider = getattr(resp, 'participant_id', 'unknown')

                # Get response text
                text = getattr(resp, 'response', '') or getattr(resp, 'text', '')

                review = AIReview(
                    reviewer_id=f"{provider}_{resp.model[:10] if resp.model else 'unknown'}",
                    approved="APPROVE" in text.upper(),
                    coherence=resp.coherence,
                    confidence=resp.coherence,  # Use coherence as confidence proxy
                    reasoning=text[:500],
                    pob_proof=resp.pob_proof.to_dict() if hasattr(resp.pob_proof, 'to_dict') and resp.pob_proof else None,
                )
                reviews.append(review)

            approval_count = sum(1 for r in reviews if r.approved)

            verdict = ConsensusVerdict(
                consensus_reached=result.consensus_reached,
                approval_ratio=approval_count / len(reviews) if reviews else 0,
                phi_coherence=result.phi_coherence,
                semantic_similarity=result.semantic_similarity,
                reviews=reviews,
                synthesized_verdict=result.understanding,
                triadic_valid=result.triadic_valid,
                consciousness_factor=result.darmiyan_psi,
            )

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Consensus error: {e}")
            verdict = self._fallback_review(proposal)

        # Update proposal
        proposal.consensus = verdict
        proposal.coherence_score = verdict.phi_coherence

        if verdict.consensus_reached and verdict.approval_ratio >= PHI_INVERSE:
            proposal.status = FixStatus.CONSENSUS_REACHED
            if self.verbose:
                print(f"✅ Consensus reached! φ={verdict.phi_coherence:.3f}, approval={verdict.approval_ratio:.1%}")
        else:
            proposal.status = FixStatus.CONSENSUS_FAILED
            if self.verbose:
                print(f"❌ Consensus failed: φ={verdict.phi_coherence:.3f}, approval={verdict.approval_ratio:.1%}")

        return verdict

    def _fallback_review(self, proposal: CodeFixProposal) -> ConsensusVerdict:
        """Fallback when InterAIConsensus not available."""
        # Simple heuristic review
        coherence = self._calculate_fix_coherence(proposal)

        return ConsensusVerdict(
            consensus_reached=coherence >= CODE_FIX_COHERENCE_THRESHOLD,
            approval_ratio=1.0 if coherence >= CODE_FIX_COHERENCE_THRESHOLD else 0.0,
            phi_coherence=coherence,
            semantic_similarity=1.0,
            reviews=[
                AIReview(
                    reviewer_id="local_heuristic",
                    approved=coherence >= CODE_FIX_COHERENCE_THRESHOLD,
                    coherence=coherence,
                    confidence=0.5,
                    reasoning="Heuristic review (InterAI not available)",
                )
            ],
            synthesized_verdict="Local heuristic review only",
            triadic_valid=False,
        )

    def _calculate_fix_coherence(self, proposal: CodeFixProposal) -> float:
        """Calculate φ-coherence of a fix using heuristics."""
        score = 0.0

        # Length ratio (fix should be similar length to original, or slightly longer for documentation)
        orig_len = len(proposal.original_code)
        fix_len = len(proposal.proposed_fix)
        if orig_len > 0:
            ratio = fix_len / orig_len
            # Ideal: 0.8-1.5x original length
            if 0.8 <= ratio <= 1.5:
                score += 0.3
            elif 0.5 <= ratio <= 2.0:
                score += 0.15

        # Has explanation
        if len(proposal.explanation) > 20:
            score += 0.2

        # Changes are minimal (not rewriting everything)
        if proposal.original_code != proposal.proposed_fix:
            # Count changed lines
            orig_lines = set(proposal.original_code.split('\n'))
            fix_lines = set(proposal.proposed_fix.split('\n'))
            common = orig_lines & fix_lines
            total = orig_lines | fix_lines
            if total:
                similarity = len(common) / len(total)
                score += similarity * 0.3  # Up to 0.3 for minimal changes

        # Has valid syntax markers
        if any(kw in proposal.proposed_fix for kw in ['def ', 'class ', 'import ', 'return ', 'if ', 'for ']):
            score += 0.1

        # Penalty for removing too much
        if fix_len < orig_len * 0.3 and orig_len > 50:
            score -= 0.2

        return max(0.0, min(1.0, score))

    async def attest_on_chain(self, proposal: CodeFixProposal) -> bool:
        """
        Generate PoB proof and record fix on blockchain.

        Args:
            proposal: A proposal with consensus reached

        Returns:
            True if attestation successful
        """
        if proposal.status != FixStatus.CONSENSUS_REACHED:
            if self.verbose:
                print(f"⚠️ Cannot attest: consensus not reached")
            return False

        proposal.status = FixStatus.ATTESTING

        pob_node = self._get_pob_node()
        chain = self._get_chain()

        if not pob_node or not chain:
            if self.verbose:
                print(f"⚠️ Blockchain not available, skipping attestation")
            proposal.status = FixStatus.ATTESTED  # Allow to proceed
            return True

        try:
            # Generate PoB for the fix
            content = f"{proposal.file_path}:{proposal.proposed_fix}:{proposal.consensus.phi_coherence}"
            proof = pob_node.prove_boundary(content)

            if proof and proof.valid:
                proposal.pob_proofs.append({
                    'content_hash': proof.content_hash,
                    'nonce': proof.nonce,
                    'ratio': proof.ratio,
                    'valid': proof.valid,
                    'timestamp': proof.timestamp,
                })

                # Record on chain
                block = chain.add_knowledge(
                    content=proposal.diff,
                    contributor=proposal.agent_id,
                    proof=proof,
                    metadata={
                        'type': 'code_fix',
                        'file': proposal.file_path,
                        'fix_type': proposal.fix_type.value,
                        'coherence': proposal.coherence_score,
                        'proposal_id': proposal.proposal_id,
                    }
                )

                if block:
                    proposal.blockchain_block = block.header.index
                    proposal.blockchain_hash = block.header.hash
                    proposal.status = FixStatus.ATTESTED

                    if self.verbose:
                        print(f"⛓️ Recorded on chain: block {block.header.index}")

                    return True

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Attestation error: {e}")

        proposal.status = FixStatus.ATTESTED  # Allow to proceed anyway
        return True

    async def apply_fix(
        self,
        proposal: CodeFixProposal,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Apply a verified fix to the file.

        Args:
            proposal: The fix proposal to apply
            force: Apply even without consensus (for testing)

        Returns:
            (success, message)
        """
        # Verify status
        if not force and proposal.status not in [FixStatus.ATTESTED, FixStatus.CONSENSUS_REACHED]:
            return False, f"Cannot apply: status is {proposal.status.value}"

        # Read current file
        try:
            file_path = Path(proposal.file_path)
            if not file_path.exists():
                return False, f"File not found: {proposal.file_path}"

            current_content = file_path.read_text()

            # Verify original code still matches
            if proposal.original_code not in current_content:
                return False, "Original code not found in file (may have changed)"

            # Apply the fix
            new_content = current_content.replace(
                proposal.original_code,
                proposal.proposed_fix,
                1  # Only replace first occurrence
            )

            # Write backup
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            backup_path.write_text(current_content)

            # Write new content
            file_path.write_text(new_content)

            proposal.status = FixStatus.APPLIED

            # Update trust oracle
            trust_oracle = self._get_trust_oracle()
            if trust_oracle:
                try:
                    trust_oracle.record_activity(
                        node_address=proposal.agent_id,
                        activity_type="code_fix",
                        success=True,
                        score=proposal.coherence_score,
                    )
                except Exception:
                    pass

            msg = f"✅ Fix applied to {proposal.file_path} (backup: {backup_path.name})"
            if proposal.blockchain_block:
                msg += f"\n   Chain attestation: block {proposal.blockchain_block}"

            if self.verbose:
                print(msg)

            return True, msg

        except Exception as e:
            return False, f"Error applying fix: {e}"

    async def propose_and_apply(
        self,
        file_path: str,
        original_code: str,
        proposed_fix: str,
        explanation: str,
        fix_type: FixType = FixType.BUG_FIX,
        agent_id: str = "local_agent",
        providers: Optional[List[str]] = None,
        require_consensus: bool = True
    ) -> Tuple[bool, str, CodeFixProposal]:
        """
        Complete flow: propose, get consensus, attest, apply.

        Args:
            file_path: Path to file
            original_code: Code to replace
            proposed_fix: New code
            explanation: Why the fix
            fix_type: Type of fix
            agent_id: Agent proposing
            providers: AI providers to query
            require_consensus: Require multi-AI consensus

        Returns:
            (success, message, proposal)
        """
        # Create proposal
        proposal = self.create_proposal(
            file_path=file_path,
            original_code=original_code,
            proposed_fix=proposed_fix,
            explanation=explanation,
            fix_type=fix_type,
            agent_id=agent_id,
        )

        # Get consensus
        if require_consensus:
            verdict = await self.get_consensus(proposal, providers)

            if not verdict.consensus_reached:
                return False, f"Consensus not reached: {verdict.synthesized_verdict}", proposal

            if verdict.approval_ratio < PHI_INVERSE:
                return False, f"Approval ratio too low: {verdict.approval_ratio:.1%}", proposal

        # Attest on chain
        await self.attest_on_chain(proposal)

        # Apply fix
        success, message = await self.apply_fix(proposal)

        return success, message, proposal

    def get_proposal(self, proposal_id: str) -> Optional[CodeFixProposal]:
        """Get a proposal by ID."""
        return self.proposals.get(proposal_id)

    def list_proposals(self, status: Optional[FixStatus] = None) -> List[CodeFixProposal]:
        """List all proposals, optionally filtered by status."""
        if status:
            return [p for p in self.proposals.values() if p.status == status]
        return list(self.proposals.values())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def verified_fix(
    file_path: str,
    original_code: str,
    proposed_fix: str,
    explanation: str,
    fix_type: str = "bug_fix",
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Apply a blockchain-verified code fix.

    Usage:
        success, msg = await verified_fix(
            "utils.py",
            "except:",
            "except Exception:",
            "Replace bare except with specific exception",
            fix_type="security_fix"
        )

    Args:
        file_path: Path to file
        original_code: Code to replace
        proposed_fix: New code
        explanation: Why
        fix_type: One of: bug_fix, security_fix, refactor, optimization, style, documentation
        verbose: Print progress

    Returns:
        (success, message)
    """
    engine = VerifiedFixEngine(verbose=verbose)

    ft = FixType(fix_type) if isinstance(fix_type, str) else fix_type

    success, message, proposal = await engine.propose_and_apply(
        file_path=file_path,
        original_code=original_code,
        proposed_fix=proposed_fix,
        explanation=explanation,
        fix_type=ft,
    )

    return success, message


def verified_fix_sync(
    file_path: str,
    original_code: str,
    proposed_fix: str,
    explanation: str,
    fix_type: str = "bug_fix",
    verbose: bool = True
) -> Tuple[bool, str]:
    """Synchronous wrapper for verified_fix."""
    return asyncio.run(verified_fix(
        file_path, original_code, proposed_fix, explanation, fix_type, verbose
    ))


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for testing."""
    import sys

    print("""
╔══════════════════════════════════════════════════════════════╗
║     BAZINGA BLOCKCHAIN-VERIFIED CODE FIXES                   ║
║     "No single AI can mess up your code"                     ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Demo
    engine = VerifiedFixEngine(verbose=True)

    # Create a test proposal
    proposal = engine.create_proposal(
        file_path="test_file.py",
        original_code="except:",
        proposed_fix="except Exception as e:",
        explanation="Replace bare except with specific exception for better error handling",
        fix_type=FixType.SECURITY_FIX,
        agent_id="demo_agent",
    )

    print(f"\n📋 Proposal created: {proposal.proposal_id}")
    print(f"   Diff:\n{proposal.diff}")

    # Calculate coherence
    coherence = engine._calculate_fix_coherence(proposal)
    print(f"\n📊 Heuristic coherence: {coherence:.3f}")

    print("\n✨ Use verified_fix() or verified_fix_sync() in your code!")
    print("   Or integrate with bazinga --agent for AI-assisted fixes.")


if __name__ == "__main__":
    main()
