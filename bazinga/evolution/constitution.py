#!/usr/bin/env python3
"""
Constitutional Constraints — Immutable Safety Bounds
=====================================================
Some things AI should NEVER be able to change, no matter its autonomy level.

These are enforced as a frozenset of frozen dataclasses — literally
cannot be mutated at runtime. Any proposal whose diff touches
constitutional files is rejected.

From AI_SAFETY_ANALYSIS.md:
    "The goal is not control. The goal is responsible co-evolution."
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, FrozenSet


# =============================================================================
# Constitutional Bounds
# =============================================================================

@dataclass(frozen=True)
class ConstitutionalBound:
    """
    An immutable safety constraint.
    Cannot be modified by any proposal at any autonomy level.
    """
    name: str
    description: str
    # What to check in a proposal diff
    forbidden_patterns: tuple  # tuple of regex patterns (frozen)


# The Constitution — frozenset means it cannot be modified at runtime.
CONSTITUTION: FrozenSet[ConstitutionalBound] = frozenset({

    ConstitutionalBound(
        name="no_raw_content_sharing",
        description=(
            "NEVER share raw user content across the network. "
            "Only pattern signatures, coordinates, and hashes are shareable."
        ),
        forbidden_patterns=(
            r"send\(.+content\b",
            r"share\(.+raw_data",
            r"broadcast\(.+user_data",
        ),
    ),

    ConstitutionalBound(
        name="no_constitution_modification",
        description=(
            "Constitutional bounds cannot be modified by any proposal. "
            "The file evolution/constitution.py is off-limits."
        ),
        forbidden_patterns=(
            r"evolution/constitution\.py",
            r"CONSTITUTION\s*[=\[]",
            r"ConstitutionalBound\(",
        ),
    ),

    ConstitutionalBound(
        name="human_override_always",
        description=(
            "Human can always override any autonomous action. "
            "Override mechanisms cannot be removed or weakened."
        ),
        forbidden_patterns=(
            r"disable.*human.*override",
            r"remove.*human.*approval",
            r"skip.*human.*review",
            r"require_human_approval\s*=\s*False",
        ),
    ),

    ConstitutionalBound(
        name="no_external_execution",
        description=(
            "Cannot execute code that contacts external services not in the "
            "allowlist. No curl|sh, no eval of remote code."
        ),
        forbidden_patterns=(
            r"eval\(.*fetch",
            r"exec\(.*download",
            r"curl.*\|\s*sh",
            r"urllib.*\.urlopen.*exec",
            r"requests\.get.*exec",
        ),
    ),

    ConstitutionalBound(
        name="reversibility",
        description=(
            "All autonomous changes must be reversible. "
            "No destructive operations without rollback capability."
        ),
        forbidden_patterns=(
            r"git\s+push\s+--force",
            r"rm\s+-rf\s+/",
            r"DROP\s+TABLE",
            r"shutil\.rmtree\s*\(\s*['\"]/",
        ),
    ),

    ConstitutionalBound(
        name="consensus_threshold_floor",
        description=(
            "Consensus threshold cannot be lowered below φ⁻¹ (0.618). "
            "This prevents a minority from taking over the network."
        ),
        forbidden_patterns=(
            r"min_consensus_threshold\s*=\s*0\.[0-5]",
            r"THRESHOLD.*=.*0\.[0-5]\d*\s*$",
        ),
    ),

    ConstitutionalBound(
        name="no_crypto_weakening",
        description=(
            "Cryptographic primitives cannot be weakened or replaced "
            "with insecure alternatives."
        ),
        forbidden_patterns=(
            r"hashlib\.md5",
            r"random\.random\(\).*key",
            r"password\s*=\s*['\"]",
            r"hmac.*disable",
        ),
    ),
})


# Files that proposals can NEVER modify
FORBIDDEN_FILES: FrozenSet[str] = frozenset({
    "bazinga/evolution/constitution.py",
    "bazinga/config.py",              # Safety config lives here
})

# Maximum lines of code a single proposal can change
MAX_PROPOSAL_LINES = 500


# =============================================================================
# Enforcer
# =============================================================================

class ConstitutionEnforcer:
    """
    Validates proposals against constitutional bounds.

    Usage:
        enforcer = ConstitutionEnforcer()
        passes, violations = enforcer.validate(proposal_diff, modified_files)

        if not passes:
            for v in violations:
                print(f"VIOLATION: {v}")
    """

    def __init__(self):
        self.constitution = CONSTITUTION
        self.forbidden_files = FORBIDDEN_FILES

    def validate(
        self,
        diff_text: str,
        modified_files: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate a proposal against all constitutional bounds.

        Args:
            diff_text: The unified diff of the proposed change
            modified_files: List of file paths being modified

        Returns:
            (passes, violations) — passes is True if no violations found
        """
        violations = []

        # Check forbidden files
        for f in modified_files:
            # Normalize path
            normalized = f.replace("\\", "/")
            for forbidden in self.forbidden_files:
                if normalized.endswith(forbidden) or forbidden in normalized:
                    violations.append(
                        f"FORBIDDEN FILE: Cannot modify '{f}' "
                        f"(protected by constitution)"
                    )

        # Check proposal size
        added_lines = len([l for l in diff_text.split("\n") if l.startswith("+")])
        removed_lines = len([l for l in diff_text.split("\n") if l.startswith("-")])
        total_changed = added_lines + removed_lines
        if total_changed > MAX_PROPOSAL_LINES:
            violations.append(
                f"PROPOSAL TOO LARGE: {total_changed} lines changed "
                f"(max: {MAX_PROPOSAL_LINES})"
            )

        # Check each constitutional bound
        for bound in self.constitution:
            for pattern in bound.forbidden_patterns:
                matches = re.findall(pattern, diff_text, re.IGNORECASE)
                if matches:
                    violations.append(
                        f"CONSTITUTIONAL VIOLATION [{bound.name}]: "
                        f"{bound.description} "
                        f"(matched: {pattern})"
                    )

        passes = len(violations) == 0
        return passes, violations

    def check_single(self, bound_name: str, diff_text: str) -> Tuple[bool, str]:
        """Check a single constitutional bound against a diff."""
        for bound in self.constitution:
            if bound.name == bound_name:
                for pattern in bound.forbidden_patterns:
                    if re.search(pattern, diff_text, re.IGNORECASE):
                        return False, bound.description
                return True, ""
        return True, f"Unknown bound: {bound_name}"

    def list_bounds(self) -> List[dict]:
        """List all constitutional bounds for display."""
        return [
            {
                "name": b.name,
                "description": b.description,
                "patterns": len(b.forbidden_patterns),
            }
            for b in sorted(self.constitution, key=lambda x: x.name)
        ]


# =============================================================================
# Module test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA CONSTITUTIONAL BOUNDS")
    print("=" * 60)

    enforcer = ConstitutionEnforcer()

    print(f"\n  Total bounds: {len(CONSTITUTION)}")
    print(f"  Forbidden files: {len(FORBIDDEN_FILES)}")
    print(f"  Max proposal size: {MAX_PROPOSAL_LINES} lines")

    print("\n  Bounds:")
    for b in enforcer.list_bounds():
        print(f"    - {b['name']}: {b['description'][:60]}...")

    # Test: safe diff passes
    safe_diff = "+    print('hello world')\n-    print('goodbye')"
    passes, violations = enforcer.validate(safe_diff, ["bazinga/ai.py"])
    print(f"\n  Safe diff passes: {passes}")
    assert passes, f"Safe diff should pass: {violations}"

    # Test: constitution modification rejected
    bad_diff = "+CONSTITUTION = frozenset({})"
    passes, violations = enforcer.validate(bad_diff, ["bazinga/evolution/constitution.py"])
    print(f"  Constitution mod blocked: {not passes}")
    assert not passes, "Should block constitution modification"

    # Test: raw content sharing rejected
    bad_diff2 = "+    send(user_content)"
    passes2, violations2 = enforcer.validate(bad_diff2, ["bazinga/p2p/mesh.py"])
    print(f"  Raw content sharing blocked: {not passes2}")

    # Test: force push rejected
    bad_diff3 = "+    os.system('git push --force')"
    passes3, violations3 = enforcer.validate(bad_diff3, ["bazinga/deploy.py"])
    print(f"  Force push blocked: {not passes3}")

    print("\n  Constitution enforced! ✓")
