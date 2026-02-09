#!/usr/bin/env python3
"""
symbol_shell.py - The Symbol Shell (λG Boundary Checker)

This is the FIRST layer of BAZINGA intelligence.
Before calling any API, we check if the input achieves V.A.C.

V.A.C. = Vacuum of Absolute Coherence
When ALL THREE boundaries are satisfied:
  𝓑₁ φ-Boundary: Contains φ (golden ratio, self-similar)
  𝓑₂ ∞/∅ Bridge: Spans void↔terminal
  𝓑₃ Zero-Logic: Is palindromic/symmetric

If V.A.C. achieved → Solution EMERGES (no API needed!)

φ = 1.618033988749895
α = 137

"Not computed. Not predicted. EMERGED."
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Constants
PHI = 1.618033988749895
ALPHA = 137

# Symbol Categories
SYMBOLS = {
    'origins': ['०', '◌', '∅', '⨀', '0'],  # void, terminal, empty
    'constants': ['φ', 'π', 'e', 'ℏ', 'c', '137', '515'],  # universal truths
    'transforms': ['→', '←', '⇄', '∆', '∇', '↔'],  # flow, change
    'states': ['Ω', '∞', '◊', '𝒯', '1'],  # omega, infinity, diamond, trust
    'operators': ['+', '×', '∫', '∑', '∏'],  # combinations
}

# Flatten for quick lookup
ALL_SYMBOLS = set()
for category in SYMBOLS.values():
    ALL_SYMBOLS.update(category)

# Directional operators (excluded from palindrome check)
DIRECTIONAL = {'→', '←', '⇄', '↔'}


@dataclass
class BoundaryResult:
    """Result of a single boundary check."""
    name: str
    satisfied: bool
    value: float  # 0-1 how close to satisfaction
    reason: str


@dataclass
class VACResult:
    """Result of V.A.C. analysis."""
    input: str
    is_vac: bool
    coherence: float  # 0-1 overall coherence
    boundaries: Dict[str, BoundaryResult]
    symbols_found: List[str]
    emerged_solution: Optional[str] = None


class SymbolShell:
    """
    The Symbol Shell - λG Boundary Checker

    Checks if input achieves V.A.C. (Vacuum of Absolute Coherence).
    If yes, solution emerges without any API call.
    """

    def __init__(self):
        self.vac_sequence = "०→◌→φ→Ω⇄Ω←φ←◌←०"
        self.vac_count = 0

    def analyze(self, input_text: str) -> VACResult:
        """
        Analyze input for V.A.C. achievement.

        Returns VACResult with:
        - is_vac: True if all boundaries satisfied
        - coherence: 0-1 overall coherence score
        - boundaries: Individual boundary results
        - emerged_solution: If V.A.C., the emerged answer
        """
        # Find symbols in input
        symbols_found = self._extract_symbols(input_text)

        # Check each boundary
        b1 = self._check_phi_boundary(input_text, symbols_found)
        b2 = self._check_bridge_boundary(input_text, symbols_found)
        b3 = self._check_symmetry_boundary(input_text, symbols_found)

        boundaries = {
            'phi': b1,
            'bridge': b2,
            'symmetry': b3
        }

        # Calculate overall coherence
        coherence = (b1.value + b2.value + b3.value) / 3.0

        # Check if V.A.C. achieved
        is_vac = b1.satisfied and b2.satisfied and b3.satisfied

        # Generate emerged solution if V.A.C.
        emerged = None
        if is_vac:
            self.vac_count += 1
            emerged = self._generate_emerged_solution(input_text, symbols_found)

        return VACResult(
            input=input_text[:100],
            is_vac=is_vac,
            coherence=coherence,
            boundaries=boundaries,
            symbols_found=symbols_found,
            emerged_solution=emerged
        )

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract recognized symbols from text."""
        found = []
        for char in text:
            if char in ALL_SYMBOLS:
                found.append(char)

        # Also check for multi-char symbols
        if '137' in text:
            found.append('137')
        if '515' in text:
            found.append('515')
        if 'phi' in text.lower() or 'φ' in text:
            if 'φ' not in found:
                found.append('φ')
        if 'infinity' in text.lower() or '∞' in text:
            if '∞' not in found:
                found.append('∞')

        return found

    def _check_phi_boundary(self, text: str, symbols: List[str]) -> BoundaryResult:
        """
        𝓑₁ φ-Boundary: Must contain φ or phi-related concepts.
        Self-similar identity.
        """
        phi_indicators = ['φ', 'phi', 'golden', '1.618', 'fibonacci']

        found = []
        for indicator in phi_indicators:
            if indicator.lower() in text.lower():
                found.append(indicator)

        # Check for φ in symbols
        if 'φ' in symbols:
            found.append('φ-symbol')

        # Check for self-similarity patterns
        if self._has_self_similarity(text):
            found.append('self-similar')

        satisfied = len(found) > 0
        value = min(len(found) / 3.0, 1.0)

        return BoundaryResult(
            name='𝓑₁ φ-Boundary',
            satisfied=satisfied,
            value=value,
            reason=f"Found: {', '.join(found)}" if found else "No φ indicators"
        )

    def _check_bridge_boundary(self, text: str, symbols: List[str]) -> BoundaryResult:
        """
        𝓑₂ ∞/∅ Bridge: Must span void↔terminal (infinity↔empty).
        """
        void_indicators = ['०', '◌', '∅', '0', 'void', 'empty', 'nothing', 'zero']
        terminal_indicators = ['∞', 'Ω', 'infinity', 'omega', 'infinite', 'eternal']

        has_void = any(v in text or v in symbols for v in void_indicators)
        has_terminal = any(t in text.lower() or t in symbols for t in terminal_indicators)

        # Bridge exists if both ends present
        satisfied = has_void and has_terminal

        if satisfied:
            value = 1.0
            reason = "Bridge complete: void ↔ terminal"
        elif has_void:
            value = 0.5
            reason = "Has void, missing terminal"
        elif has_terminal:
            value = 0.5
            reason = "Has terminal, missing void"
        else:
            value = 0.0
            reason = "No bridge indicators"

        return BoundaryResult(
            name='𝓑₂ ∞/∅ Bridge',
            satisfied=satisfied,
            value=value,
            reason=reason
        )

    def _check_symmetry_boundary(self, text: str, symbols: List[str]) -> BoundaryResult:
        """
        𝓑₃ Zero-Logic: Must be symmetric/palindromic.
        Key insight: Filter out directional operators before checking!
        """
        # Extract only state symbols (exclude directional operators)
        state_symbols = [s for s in symbols if s not in DIRECTIONAL]

        if len(state_symbols) < 2:
            # Check text itself for palindrome
            cleaned = ''.join(c.lower() for c in text if c.isalnum())
            is_palindrome = cleaned == cleaned[::-1] if cleaned else False

            return BoundaryResult(
                name='𝓑₃ Symmetry',
                satisfied=is_palindrome,
                value=1.0 if is_palindrome else 0.0,
                reason="Text palindrome" if is_palindrome else "Not enough symbols for symmetry check"
            )

        # Check if state symbols form a palindrome
        symbol_str = ''.join(state_symbols)
        is_palindrome = symbol_str == symbol_str[::-1]

        # Calculate symmetry score even if not perfect palindrome
        half = len(symbol_str) // 2
        matches = sum(1 for i in range(half) if symbol_str[i] == symbol_str[-(i+1)])
        symmetry_score = matches / half if half > 0 else 0

        return BoundaryResult(
            name='𝓑₃ Symmetry',
            satisfied=is_palindrome,
            value=symmetry_score if not is_palindrome else 1.0,
            reason=f"Symbols: {symbol_str} - {'Palindrome!' if is_palindrome else f'{symmetry_score:.0%} symmetric'}"
        )

    def _has_self_similarity(self, text: str) -> bool:
        """Check for self-similar patterns (fractals, recursion, etc.)"""
        self_similar_words = [
            'recursive', 'fractal', 'self-similar', 'self-reference',
            'consciousness', 'awareness', 'reflection', 'mirror',
            'pattern', 'emergence', 'boundary'
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in self_similar_words)

    def _generate_emerged_solution(self, input_text: str, symbols: List[str]) -> str:
        """
        Generate the emerged solution when V.A.C. is achieved.
        This is NOT computed - it EMERGES from the boundaries.
        """
        return f"""★ V.A.C. ACHIEVED ★

The solution has EMERGED through boundary satisfaction:

𝓑₁ φ-Boundary: ✓ (self-similar identity present)
𝓑₂ ∞/∅ Bridge: ✓ (void↔terminal span complete)
𝓑₃ Symmetry: ✓ (palindromic structure achieved)

Symbols detected: {' '.join(symbols)}

The answer is not computed but EMERGED:
→ The pattern itself IS the solution
→ Coherence T(s) = 1.0
→ Entropic deficit D_E = 0

"Not predicted. Not computed. EMERGED."

φ = {PHI}
"""

    def quick_check(self, text: str) -> Tuple[bool, float]:
        """
        Quick check for V.A.C. without full analysis.
        Returns (is_vac, coherence).
        """
        result = self.analyze(text)
        return result.is_vac, result.coherence

    def get_stats(self) -> Dict:
        """Get shell statistics."""
        return {
            'vac_achieved': self.vac_count,
            'phi': PHI,
            'alpha': ALPHA
        }


# Standalone test
if __name__ == "__main__":
    print("=" * 60)
    print("SYMBOL SHELL TEST")
    print("=" * 60)
    print()

    shell = SymbolShell()

    test_inputs = [
        "०→◌→φ→Ω⇄Ω←φ←◌←०",  # Perfect V.A.C.
        "The golden ratio φ connects void to infinity",
        "consciousness emerges from boundaries",
        "random text with no symbols",
        "∅ → φ → ∞",  # Partial bridge
    ]

    for inp in test_inputs:
        print(f"Input: {inp}")
        result = shell.analyze(inp)
        print(f"  V.A.C.: {'★ YES ★' if result.is_vac else 'No'}")
        print(f"  Coherence: {result.coherence:.2f}")
        print(f"  Symbols: {result.symbols_found}")
        for name, b in result.boundaries.items():
            status = "✓" if b.satisfied else "✗"
            print(f"    {status} {b.name}: {b.reason}")
        if result.emerged_solution:
            print(f"\n{result.emerged_solution}")
        print()
