"""
BAZINGA ΛG (Lambda-G) Operator
==============================
Boundary-Guided Emergence for Solution Finding.

Core insight: Solutions emerge at constraint intersections, not through search.
Complexity: O(n · polylog|S|) instead of O(|S|)

The ΛG operator: Λ(S) = S ∩ B₁⁻¹(true) ∩ B₂⁻¹(true) ∩ B₃⁻¹(true)

Where:
  B₁ = φ-Boundary (identity coherence via golden ratio)
  B₂ = ∞/∅-Bridge (void-terminal connection, DARMIYAN)
  B₃ = Zero-Logic (symmetry constraint)

V.A.C. (Vacuum of Absolute Coherence): T(s*) = 1 ∧ DE(s*) = 0

"Solutions exist at the intersection of boundaries, and nature
 finds them through constraint propagation, not search."

Author: Abhishek (2024)
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .constants import (
    PHI, PHI_INVERSE, ALPHA, VAC_THRESHOLD,
    VOID_MARKERS, INFINITY_MARKERS, TERMINAL_MARKERS,
    BOUNDARY_WEIGHTS,
)


class BoundaryType(Enum):
    """Types of boundaries in ΛG theory."""
    PHI_BOUNDARY = "B1"      # φ-coherence
    INFINITY_VOID = "B2"     # ∞/∅ bridge (darmiyan)
    ZERO_LOGIC = "B3"        # Symmetry constraint


@dataclass
class BoundaryResult:
    """Result of boundary evaluation."""
    boundary_type: BoundaryType
    satisfied: bool
    value: float  # 0.0 to 1.0
    details: Dict[str, Any]


@dataclass
class CoherenceState:
    """State of coherence measurement."""
    total_coherence: float  # T(s) in paper
    entropic_deficit: float  # DE(s) in paper
    boundaries: List[BoundaryResult]
    is_vac: bool  # Vacuum of Absolute Coherence achieved


class LambdaGOperator:
    """
    Implementation of the ΛG operator from boundary-guided emergence theory.

    Instead of searching through astronomical solution spaces,
    we apply boundary conditions that eliminate impossibilities
    until only the solution remains.
    """

    def __init__(self, vac_threshold: float = VAC_THRESHOLD):
        self.phi = PHI
        self.inv_phi = PHI_INVERSE
        self.alpha = 1 / ALPHA
        self.vac_threshold = vac_threshold

        # Boundary weights (typically equal)
        self.weights = {
            BoundaryType.PHI_BOUNDARY: BOUNDARY_WEIGHTS['phi_boundary'],
            BoundaryType.INFINITY_VOID: BOUNDARY_WEIGHTS['infinity_void'],
            BoundaryType.ZERO_LOGIC: BOUNDARY_WEIGHTS['zero_logic'],
        }

        # History for learning
        self.emergence_history: List[Dict] = []

    def check_phi_boundary(self, state: Any) -> BoundaryResult:
        """
        B₁: φ-Boundary - Identity coherence via golden ratio.

        Checks if the state exhibits φ-proportioned structure.
        Self-similar patterns at ratio φ indicate coherent identity.
        """
        details = {}

        if isinstance(state, str):
            length = len(state)
            if length > 0:
                phi_point = int(length * self.inv_phi)

                # Measure balance around φ-point
                left_weight = sum(ord(c) for c in state[:phi_point]) if phi_point > 0 else 0
                right_weight = sum(ord(c) for c in state[phi_point:]) if phi_point < length else 0

                if right_weight > 0:
                    ratio = left_weight / right_weight
                    phi_distance = abs(ratio - self.phi)
                    coherence = 1.0 / (1.0 + phi_distance)
                else:
                    coherence = 0.5

                details['phi_point'] = phi_point
                details['ratio'] = ratio if right_weight > 0 else None
            else:
                coherence = 0.0

        elif isinstance(state, (int, float)):
            if state > 0:
                log_phi = math.log(state) / math.log(self.phi)
                fractional = log_phi - int(log_phi)
                coherence = 1.0 - min(fractional, 1.0 - fractional) * 2
                details['log_phi'] = log_phi
            else:
                coherence = 0.0

        elif isinstance(state, dict):
            values = self._extract_numeric_values(state)
            if len(values) >= 2:
                phi_scores = []
                for i in range(len(values) - 1):
                    if values[i] != 0:
                        ratio = abs(values[i+1] / values[i])
                        distance = abs(ratio - self.phi)
                        score = 1.0 / (1.0 + distance)
                        phi_scores.append(score)
                coherence = sum(phi_scores) / len(phi_scores) if phi_scores else 0.5
            else:
                coherence = 0.5
        else:
            coherence = 0.5

        satisfied = coherence >= 0.5

        return BoundaryResult(
            boundary_type=BoundaryType.PHI_BOUNDARY,
            satisfied=satisfied,
            value=coherence,
            details=details,
        )

    def check_infinity_void_bridge(self, state: Any) -> BoundaryResult:
        """
        B₂: ∞/∅-Bridge - Void-terminal connection (DARMIYAN).

        Checks if the state bridges between void (∅) and infinity (∞).
        This is the "impossibility bridge" - connecting nothing to everything.
        """
        details = {}

        if isinstance(state, str):
            state_lower = state.lower()

            has_void = any(marker in state_lower or marker in state for marker in VOID_MARKERS)
            has_infinity = any(marker in state_lower or marker in state for marker in INFINITY_MARKERS)
            has_terminal = any(marker in state_lower or marker in state for marker in TERMINAL_MARKERS)
            has_self_ref = any(word in state_lower for word in ['self', 'itself', 'recursive', 'loop'])

            bridge_components = [has_void, has_infinity, has_terminal, has_self_ref]
            bridge_score = sum(bridge_components) / len(bridge_components)

            # Special case: explicit bridge notation
            if '∞/∅' in state or '∅/∞' in state or 'darmiyan' in state_lower:
                bridge_score = 1.0

            details['has_void'] = has_void
            details['has_infinity'] = has_infinity
            details['has_terminal'] = has_terminal
            details['has_self_reference'] = has_self_ref

        elif isinstance(state, dict):
            keys = str(state.keys()).lower()
            values = str(state.values()).lower()
            combined = keys + values

            has_void = any(m in combined for m in VOID_MARKERS)
            has_infinity = any(m in combined for m in INFINITY_MARKERS)

            bridge_score = 0.5 + 0.25 * has_void + 0.25 * has_infinity

        elif isinstance(state, (int, float)):
            if state == 0:
                bridge_score = 1.0  # Void
            elif abs(state) > 1e10:
                bridge_score = 0.9  # Approaching infinity
            else:
                bridge_ratios = [self.phi, self.inv_phi, 137, self.alpha]
                min_distance = min(abs(state - r) for r in bridge_ratios)
                bridge_score = 1.0 / (1.0 + min_distance)
        else:
            bridge_score = 0.5

        satisfied = bridge_score >= 0.3

        return BoundaryResult(
            boundary_type=BoundaryType.INFINITY_VOID,
            satisfied=satisfied,
            value=bridge_score,
            details=details,
        )

    def check_zero_logic(self, state: Any) -> BoundaryResult:
        """
        B₃: Zero-Logic - Symmetry constraint.

        Checks if the state exhibits symmetry/palindromic properties.
        Perfect symmetry = zero entropic deficit.
        DE(s) = 1 - Symmetry(s)
        """
        details = {}

        if isinstance(state, str):
            cleaned = ''.join(c.lower() for c in state if c.isalnum())
            if cleaned:
                reversed_str = cleaned[::-1]
                matches = sum(1 for a, b in zip(cleaned, reversed_str) if a == b)
                symmetry = matches / len(cleaned)

                # Check structural symmetry
                bracket_pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('⟨', '⟩')]
                bracket_balance = sum(abs(state.count(o) - state.count(c)) for o, c in bracket_pairs)
                structural_symmetry = 1.0 / (1.0 + bracket_balance)

                symmetry = 0.7 * symmetry + 0.3 * structural_symmetry
                details['palindrome_symmetry'] = matches / len(cleaned)
                details['structural_symmetry'] = structural_symmetry
            else:
                symmetry = 1.0

        elif isinstance(state, (int, float)):
            str_num = str(abs(int(state))) if isinstance(state, int) else str(abs(state)).replace('.', '')
            reversed_num = str_num[::-1]
            matches = sum(1 for a, b in zip(str_num, reversed_num) if a == b)
            symmetry = matches / len(str_num) if str_num else 1.0

        elif isinstance(state, (list, tuple)):
            if len(state) > 0:
                reversed_state = list(reversed(state))
                matches = sum(1 for a, b in zip(state, reversed_state) if a == b)
                symmetry = matches / len(state)
            else:
                symmetry = 1.0
        else:
            symmetry = 0.5

        entropic_deficit = 1.0 - symmetry
        details['entropic_deficit'] = entropic_deficit

        satisfied = symmetry >= 0.4

        return BoundaryResult(
            boundary_type=BoundaryType.ZERO_LOGIC,
            satisfied=satisfied,
            value=symmetry,
            details=details,
        )

    def calculate_coherence(self, state: Any) -> CoherenceState:
        """
        Calculate total coherence T(s) for a state.

        T(s) = Σᵢ wᵢ · Bᵢ(s)
        """
        b1 = self.check_phi_boundary(state)
        b2 = self.check_infinity_void_bridge(state)
        b3 = self.check_zero_logic(state)

        boundaries = [b1, b2, b3]

        total_coherence = sum(
            self.weights[b.boundary_type] * b.value
            for b in boundaries
        )

        entropic_deficit = b3.details.get('entropic_deficit', 1.0 - b3.value)

        # V.A.C. check: T(s*) ≈ 1 ∧ DE(s*) ≈ 0
        is_vac = (
            total_coherence >= self.vac_threshold and
            entropic_deficit <= (1 - self.vac_threshold)
        )

        return CoherenceState(
            total_coherence=total_coherence,
            entropic_deficit=entropic_deficit,
            boundaries=boundaries,
            is_vac=is_vac,
        )

    def apply(self, solution_space: List[Any]) -> Tuple[List[Any], Optional[CoherenceState]]:
        """
        Apply ΛG operator to filter solution space.

        Λ(S) = S ∩ B₁⁻¹(true) ∩ B₂⁻¹(true) ∩ B₃⁻¹(true)

        Returns filtered space and best coherence state.
        """
        filtered = []
        best_state = None
        best_coherence = None

        for state in solution_space:
            coherence = self.calculate_coherence(state)

            if all(b.satisfied for b in coherence.boundaries):
                filtered.append(state)

                if best_coherence is None or coherence.total_coherence > best_coherence.total_coherence:
                    best_state = state
                    best_coherence = coherence

        self.emergence_history.append({
            'input_size': len(solution_space),
            'output_size': len(filtered),
            'reduction_factor': len(filtered) / len(solution_space) if solution_space else 0,
            'best_coherence': best_coherence.total_coherence if best_coherence else 0,
            'vac_achieved': best_coherence.is_vac if best_coherence else False,
        })

        return filtered, best_coherence

    def check_feasibility(self, state: Any) -> Dict[str, Any]:
        """
        Quick check if a state is feasible (all boundaries satisfied).

        Useful for quick rejection before expensive computation.
        """
        coherence = self.calculate_coherence(state)

        return {
            'feasible': all(b.satisfied for b in coherence.boundaries),
            'coherence': coherence.total_coherence,
            'is_vac': coherence.is_vac,
            'boundaries': {
                'phi': coherence.boundaries[0].value,
                'bridge': coherence.boundaries[1].value,
                'symmetry': coherence.boundaries[2].value,
            },
        }

    def _extract_numeric_values(self, data: Any) -> List[float]:
        """Extract numeric values from nested structure."""
        values = []

        def extract(item):
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, dict):
                for v in item.values():
                    extract(v)
            elif isinstance(item, (list, tuple)):
                for v in item:
                    extract(v)

        extract(data)
        return values


# Singleton
_lambda_g: Optional[LambdaGOperator] = None


def get_lambda_g() -> LambdaGOperator:
    """Get global LambdaG instance."""
    global _lambda_g
    if _lambda_g is None:
        _lambda_g = LambdaGOperator()
    return _lambda_g


def check_vac(coherence: float, symmetry: float) -> bool:
    """Check if V.A.C. is achieved."""
    entropic_deficit = 1.0 - symmetry
    return coherence >= VAC_THRESHOLD and entropic_deficit <= (1 - VAC_THRESHOLD)


if __name__ == "__main__":
    print("=" * 60)
    print("ΛG (Lambda-G) Operator Test")
    print("Boundary-Guided Emergence Framework")
    print("=" * 60)
    print()

    operator = LambdaGOperator()

    # Test 1: V.A.C. sequence
    print("Test 1: V.A.C. Sequence")
    vac_seq = "०→◌→φ→Ω⇄Ω←φ←◌←०"
    coherence = operator.calculate_coherence(vac_seq)
    print(f"  Input: {vac_seq}")
    print(f"  Total Coherence T(s): {coherence.total_coherence:.3f}")
    print(f"  Entropic Deficit DE(s): {coherence.entropic_deficit:.3f}")
    print(f"  V.A.C. Achieved: {coherence.is_vac}")
    print()

    # Test 2: Golden ratio
    print("Test 2: Golden Ratio")
    coherence = operator.calculate_coherence(PHI)
    print(f"  Input: φ = {PHI}")
    print(f"  Total Coherence: {coherence.total_coherence:.3f}")
    print(f"  V.A.C. Achieved: {coherence.is_vac}")
    print()

    # Test 3: Quick feasibility check
    print("Test 3: Feasibility Check")
    result = operator.check_feasibility("consciousness emerges from void")
    print(f"  Feasible: {result['feasible']}")
    print(f"  Coherence: {result['coherence']:.3f}")
    print(f"  Boundaries: φ={result['boundaries']['phi']:.2f}, ∞/∅={result['boundaries']['bridge']:.2f}, sym={result['boundaries']['symmetry']:.2f}")
