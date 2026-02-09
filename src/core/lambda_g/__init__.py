"""
Lambda-G (ΛG) Framework - Boundary-Guided Emergence

Based on: "Boundary-Guided Emergence: A Unified Mathematical Framework
          for Solution Finding in Complex Systems" - Abhishek (2024)

Core insight: Solutions emerge at constraint intersections, not through search.

The ΛG operator applies three boundaries:
  B₁ = φ-Boundary (identity coherence via golden ratio)
  B₂ = ∞/∅-Bridge (void-terminal connection - DARMIYAN)
  B₃ = Zero-Logic (symmetry constraint)

V.A.C. (Vacuum of Absolute Coherence) = perfect solution state
"""

from .lambda_g_operator import (
    LambdaGOperator,
    BoundaryType,
    BoundaryResult,
    CoherenceState,
    check_vac,
    PHI,
    INV_PHI,
    ALPHA
)

__all__ = [
    'LambdaGOperator',
    'BoundaryType',
    'BoundaryResult',
    'CoherenceState',
    'check_vac',
    'PHI',
    'INV_PHI',
    'ALPHA'
]
