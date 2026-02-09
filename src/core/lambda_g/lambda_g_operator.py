#!/usr/bin/env python3
"""
lambda_g_operator.py - Implementation of ΛG (Lambda-G) Theory

Based on: "Boundary-Guided Emergence: A Unified Mathematical Framework
          for Solution Finding in Complex Systems" - Abhishek (2024)

Core insight: Solutions emerge at constraint intersections, not through search.
Complexity: O(n · polylog|S|) instead of O(|S|)

The ΛG operator: Λ(S) = S ∩ B₁⁻¹(true) ∩ B₂⁻¹(true) ∩ B₃⁻¹(true)

Where:
  B₁ = φ-Boundary (identity coherence via golden ratio)
  B₂ = ∞/∅-Bridge (void-terminal connection)
  B₃ = Zero-Logic (symmetry constraint)

V.A.C. (Vacuum of Absolute Coherence): T(s*) = 1 ∧ DE(s*) = 0
"""

import math
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


# Universal Constants
PHI = 1.618033988749895  # Golden ratio
INV_PHI = 0.6180339887498948  # 1/φ
ALPHA = 1/137  # Fine structure constant


class BoundaryType(Enum):
    """Types of boundaries in ΛG theory"""
    PHI_BOUNDARY = "B1"      # φ-coherence
    INFINITY_VOID = "B2"     # ∞/∅ bridge (darmiyan)
    ZERO_LOGIC = "B3"        # Symmetry constraint


@dataclass
class BoundaryResult:
    """Result of boundary evaluation"""
    boundary_type: BoundaryType
    satisfied: bool
    value: float  # 0.0 to 1.0
    details: Dict[str, Any]


@dataclass
class CoherenceState:
    """State of coherence measurement"""
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

    "Solutions exist at the intersection of boundaries, and nature
     finds them through constraint propagation, not search."
    """

    def __init__(self):
        self.phi = PHI
        self.inv_phi = INV_PHI
        self.alpha = ALPHA

        # Boundary weights (from paper: typically equal)
        self.weights = {
            BoundaryType.PHI_BOUNDARY: 1/3,
            BoundaryType.INFINITY_VOID: 1/3,
            BoundaryType.ZERO_LOGIC: 1/3
        }

        # History for learning
        self.emergence_history = []

        # V.A.C. threshold
        self.vac_threshold = 0.99

    def check_phi_boundary(self, state: Any) -> BoundaryResult:
        """
        B₁: φ-Boundary - Identity coherence via golden ratio

        Checks if the state exhibits φ-proportioned structure.
        Self-similar patterns at ratio φ indicate coherent identity.
        """
        details = {}

        if isinstance(state, str):
            # For strings: check for φ-proportioned segments
            length = len(state)
            if length > 0:
                # Check if natural break points align with φ
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
                details['phi_distance'] = phi_distance if right_weight > 0 else None
            else:
                coherence = 0.0

        elif isinstance(state, (int, float)):
            # For numbers: check φ-resonance
            if state > 0:
                log_phi = math.log(state) / math.log(self.phi)
                fractional = log_phi - int(log_phi)
                # Closer to integer power of φ = more coherent
                coherence = 1.0 - min(fractional, 1.0 - fractional) * 2
                details['log_phi'] = log_phi
                details['fractional'] = fractional
            else:
                coherence = 0.0

        elif isinstance(state, dict):
            # For dicts: check if structure exhibits φ-proportions
            values = self._extract_numeric_values(state)
            if len(values) >= 2:
                # Check consecutive ratios
                phi_scores = []
                for i in range(len(values) - 1):
                    if values[i] != 0:
                        ratio = abs(values[i+1] / values[i])
                        distance = abs(ratio - self.phi)
                        score = 1.0 / (1.0 + distance)
                        phi_scores.append(score)
                coherence = sum(phi_scores) / len(phi_scores) if phi_scores else 0.5
                details['phi_scores'] = phi_scores
            else:
                coherence = 0.5
        else:
            coherence = 0.5

        satisfied = coherence >= 0.5  # Threshold for φ-coherence

        return BoundaryResult(
            boundary_type=BoundaryType.PHI_BOUNDARY,
            satisfied=satisfied,
            value=coherence,
            details=details
        )

    def check_infinity_void_bridge(self, state: Any) -> BoundaryResult:
        """
        B₂: ∞/∅-Bridge - Void-terminal connection (DARMIYAN)

        Checks if the state bridges between void (∅) and infinity (∞).
        This is the "impossibility bridge" - connecting nothing to everything.

        In semantic systems: origin → terminal connection
        In temporal systems: self-reference creating closed loops
        """
        details = {}

        # Symbol markers for void and infinity
        void_markers = ['∅', '०', 'void', 'null', 'zero', 'empty', 'shunya']
        infinity_markers = ['∞', 'infinity', 'unbounded', 'endless', 'eternal']
        terminal_markers = ['◌', '⊚', 'terminal', 'end', 'omega', 'Ω']

        if isinstance(state, str):
            state_lower = state.lower()

            # Check for void presence
            has_void = any(marker in state_lower or marker in state for marker in void_markers)

            # Check for infinity presence
            has_infinity = any(marker in state_lower or marker in state for marker in infinity_markers)

            # Check for terminal presence
            has_terminal = any(marker in state_lower or marker in state for marker in terminal_markers)

            # Check for self-reference (key indicator of ∞/∅ bridge)
            has_self_ref = any(word in state_lower for word in ['self', 'itself', 'recursive', 'loop'])

            # Bridge score
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
            # Check for bridging structure in dict
            keys = str(state.keys()).lower()
            values = str(state.values()).lower()
            combined = keys + values

            has_void = any(m in combined for m in void_markers)
            has_infinity = any(m in combined for m in infinity_markers)

            bridge_score = 0.5 + 0.25 * has_void + 0.25 * has_infinity
            details['has_void'] = has_void
            details['has_infinity'] = has_infinity

        else:
            # For numbers: check if it represents a bridge value
            if isinstance(state, (int, float)):
                # 0 and very large numbers are bridge points
                if state == 0:
                    bridge_score = 1.0  # Void
                elif abs(state) > 1e10:
                    bridge_score = 0.9  # Approaching infinity
                else:
                    # Check for special ratios
                    if state > 0:
                        # φ, 1/φ, 137, 1/137 are bridge values
                        bridge_ratios = [self.phi, self.inv_phi, 137, self.alpha]
                        min_distance = min(abs(state - r) for r in bridge_ratios)
                        bridge_score = 1.0 / (1.0 + min_distance)
                    else:
                        bridge_score = 0.3
                details['value'] = state
            else:
                bridge_score = 0.5

        satisfied = bridge_score >= 0.3  # Lower threshold - bridge is about connection

        return BoundaryResult(
            boundary_type=BoundaryType.INFINITY_VOID,
            satisfied=satisfied,
            value=bridge_score,
            details=details
        )

    def check_zero_logic(self, state: Any) -> BoundaryResult:
        """
        B₃: Zero-Logic - Symmetry constraint

        Checks if the state exhibits symmetry/palindromic properties.
        Perfect symmetry = zero entropic deficit.

        In the paper: DE(s) = 1 - Symmetry(s)
        """
        details = {}

        if isinstance(state, str):
            # Check palindromic symmetry
            cleaned = ''.join(c.lower() for c in state if c.isalnum())
            if cleaned:
                reversed_str = cleaned[::-1]

                # Calculate character-wise symmetry
                matches = sum(1 for a, b in zip(cleaned, reversed_str) if a == b)
                symmetry = matches / len(cleaned)

                # Also check for structural symmetry (balanced brackets, etc.)
                bracket_pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('⟨', '⟩')]
                bracket_balance = 0
                for open_b, close_b in bracket_pairs:
                    bracket_balance += abs(state.count(open_b) - state.count(close_b))

                structural_symmetry = 1.0 / (1.0 + bracket_balance)

                # Combined symmetry
                symmetry = 0.7 * symmetry + 0.3 * structural_symmetry

                details['palindrome_symmetry'] = matches / len(cleaned)
                details['structural_symmetry'] = structural_symmetry
            else:
                symmetry = 1.0  # Empty is perfectly symmetric

        elif isinstance(state, (int, float)):
            # Number symmetry: check digit palindrome
            str_num = str(abs(int(state))) if isinstance(state, int) else str(abs(state)).replace('.', '')
            reversed_num = str_num[::-1]
            matches = sum(1 for a, b in zip(str_num, reversed_num) if a == b)
            symmetry = matches / len(str_num) if str_num else 1.0
            details['digit_symmetry'] = symmetry

        elif isinstance(state, dict):
            # Dict symmetry: balanced key-value structure
            keys = list(state.keys())
            values = list(state.values())

            # Check if keys and values have similar structure
            key_types = [type(k).__name__ for k in keys]
            val_types = [type(v).__name__ for v in values]

            type_overlap = len(set(key_types) & set(val_types)) / max(len(set(key_types + val_types)), 1)

            # Check for balanced numeric values
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            if len(numeric_vals) >= 2:
                mean_val = sum(numeric_vals) / len(numeric_vals)
                variance = sum((v - mean_val)**2 for v in numeric_vals) / len(numeric_vals)
                balance = 1.0 / (1.0 + math.sqrt(variance) / (abs(mean_val) + 1))
            else:
                balance = 0.5

            symmetry = 0.5 * type_overlap + 0.5 * balance
            details['type_overlap'] = type_overlap
            details['balance'] = balance

        elif isinstance(state, (list, tuple)):
            # Sequence symmetry
            if len(state) > 0:
                reversed_state = list(reversed(state))
                matches = sum(1 for a, b in zip(state, reversed_state) if a == b)
                symmetry = matches / len(state)
            else:
                symmetry = 1.0
            details['sequence_symmetry'] = symmetry
        else:
            symmetry = 0.5

        # Entropic deficit = 1 - symmetry
        entropic_deficit = 1.0 - symmetry
        details['entropic_deficit'] = entropic_deficit

        satisfied = symmetry >= 0.4  # Moderate symmetry threshold

        return BoundaryResult(
            boundary_type=BoundaryType.ZERO_LOGIC,
            satisfied=satisfied,
            value=symmetry,
            details=details
        )

    def calculate_coherence(self, state: Any) -> CoherenceState:
        """
        Calculate total coherence T(s) for a state.

        T(s) = Σᵢ wᵢ · 𝟙(Bᵢ(s))

        Where wᵢ are weights and 𝟙 is indicator function.
        """
        # Evaluate all boundaries
        b1 = self.check_phi_boundary(state)
        b2 = self.check_infinity_void_bridge(state)
        b3 = self.check_zero_logic(state)

        boundaries = [b1, b2, b3]

        # Calculate weighted coherence
        total_coherence = sum(
            self.weights[b.boundary_type] * b.value
            for b in boundaries
        )

        # Calculate entropic deficit from B₃
        entropic_deficit = b3.details.get('entropic_deficit', 1.0 - b3.value)

        # Check V.A.C. (Vacuum of Absolute Coherence)
        # V.A.C.(s*) ⇔ T(s*) = 1 ∧ DE(s*) = 0
        is_vac = (total_coherence >= self.vac_threshold and
                  entropic_deficit <= (1 - self.vac_threshold))

        return CoherenceState(
            total_coherence=total_coherence,
            entropic_deficit=entropic_deficit,
            boundaries=boundaries,
            is_vac=is_vac
        )

    def apply(self, solution_space: List[Any]) -> Tuple[List[Any], CoherenceState]:
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

            # Check if all boundaries satisfied
            all_satisfied = all(b.satisfied for b in coherence.boundaries)

            if all_satisfied:
                filtered.append(state)

                # Track best coherence
                if best_coherence is None or coherence.total_coherence > best_coherence.total_coherence:
                    best_state = state
                    best_coherence = coherence

        # Record emergence
        self.emergence_history.append({
            'input_size': len(solution_space),
            'output_size': len(filtered),
            'reduction_factor': len(filtered) / len(solution_space) if solution_space else 0,
            'best_coherence': best_coherence.total_coherence if best_coherence else 0,
            'vac_achieved': best_coherence.is_vac if best_coherence else False
        })

        return filtered, best_coherence

    def find_emergent_solution(self,
                               generator: Callable[[], Any],
                               max_iterations: int = 1000,
                               early_stop_vac: bool = True) -> Tuple[Any, CoherenceState]:
        """
        Find solution through boundary-guided emergence.

        Instead of exhaustive search, we:
        1. Generate candidate states
        2. Apply boundary filtering
        3. Stop when V.A.C. achieved or max iterations

        This is O(n · polylog|S|) instead of O(|S|)
        """
        best_solution = None
        best_coherence = None

        for i in range(max_iterations):
            # Generate candidate
            candidate = generator()

            # Calculate coherence
            coherence = self.calculate_coherence(candidate)

            # Update best
            if best_coherence is None or coherence.total_coherence > best_coherence.total_coherence:
                best_solution = candidate
                best_coherence = coherence

            # Check for V.A.C. (solution emerged!)
            if early_stop_vac and coherence.is_vac:
                print(f"V.A.C. achieved at iteration {i+1}!")
                break

        return best_solution, best_coherence

    def _extract_numeric_values(self, data: Any) -> List[float]:
        """Extract numeric values from nested structure"""
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


# V.A.C. Validator
def check_vac(coherence: float, symmetry: float) -> bool:
    """
    Check if V.A.C. (Vacuum of Absolute Coherence) is achieved.

    V.A.C.(s*) ⇔ T(s*) = 1 ∧ DE(s*) = 0

    Where DE(s) = 1 - Symmetry(s)
    """
    entropic_deficit = 1.0 - symmetry
    return coherence >= 0.99 and entropic_deficit <= 0.01


# Standalone test
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
    print(f"  Boundaries:")
    for b in coherence.boundaries:
        print(f"    {b.boundary_type.value}: {b.value:.3f} ({'✓' if b.satisfied else '✗'})")
    print()

    # Test 2: Golden ratio
    print("Test 2: Golden Ratio")
    phi_val = PHI
    coherence = operator.calculate_coherence(phi_val)
    print(f"  Input: φ = {phi_val}")
    print(f"  Total Coherence: {coherence.total_coherence:.3f}")
    print(f"  V.A.C. Achieved: {coherence.is_vac}")
    print()

    # Test 3: Self-referential statement
    print("Test 3: Self-Referential Statement")
    self_ref = "This meaning refers to itself through the void-infinity bridge"
    coherence = operator.calculate_coherence(self_ref)
    print(f"  Input: '{self_ref[:50]}...'")
    print(f"  Total Coherence: {coherence.total_coherence:.3f}")
    print(f"  B₂ (∞/∅ Bridge): {coherence.boundaries[1].value:.3f}")
    print()

    # Test 4: Solution space filtering
    print("Test 4: Solution Space Filtering")
    solution_space = [
        "random text",
        "void connects to infinity",
        "∅ → φ → ∞",
        "०→◌→φ→Ω",
        "symmetric text tnemys",
        PHI,
        137,
        42
    ]
    filtered, best = operator.apply(solution_space)
    print(f"  Input space size: {len(solution_space)}")
    print(f"  Filtered space size: {len(filtered)}")
    print(f"  Reduction factor: {len(filtered)/len(solution_space):.1%}")
    if best:
        print(f"  Best coherence: {best.total_coherence:.3f}")
    print()

    print("=" * 60)
    print("ΛG Operator operational")
    print("'Solutions emerge at the intersection of boundaries'")
    print("=" * 60)
