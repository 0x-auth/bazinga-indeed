#!/usr/bin/env python3
"""
Tests for BAZINGA - Distributed AI

Run with: pytest tests/ -v
Or simply: python tests/test_bazinga.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_symbol_shell():
    """Test Symbol Shell V.A.C. detection."""
    from src.core.symbol import SymbolShell

    shell = SymbolShell()

    # Test V.A.C. sequence
    vac_sequence = "०→◌→φ→Ω⇄Ω←φ←◌←०"
    result = shell.analyze(vac_sequence)

    assert result.is_vac == True, "V.A.C. sequence should achieve V.A.C."
    assert result.coherence > 0.8, "V.A.C. sequence should have high coherence"
    assert len(result.symbols_found) > 0, "Should find symbols"

    print("✓ Symbol Shell V.A.C. test passed")


def test_symbol_shell_regular():
    """Test Symbol Shell with regular text."""
    from src.core.symbol import SymbolShell

    shell = SymbolShell()

    # Regular text should not achieve V.A.C.
    result = shell.analyze("What is the weather today?")

    assert result.is_vac == False, "Regular text should not achieve V.A.C."
    assert result.coherence < 0.5, "Regular text should have low coherence"

    print("✓ Symbol Shell regular text test passed")


def test_phi_constant():
    """Test PHI constant."""
    from src.core.symbol import PHI, ALPHA

    assert abs(PHI - 1.618033988749895) < 0.0001, "PHI should be golden ratio"
    assert ALPHA == 137, "ALPHA should be 137"

    print("✓ Constants test passed")


def test_lambda_g():
    """Test Lambda-G operator."""
    from src.core.lambda_g import LambdaGOperator, PHI

    lg = LambdaGOperator()

    # Test coherence calculation
    coherence = lg.calculate_coherence("Test input")

    assert hasattr(coherence, 'total_coherence'), "Should have total_coherence"
    assert 0 <= coherence.total_coherence <= 1, "Coherence should be 0-1"

    print("✓ Lambda-G test passed")


def test_imports():
    """Test all imports work."""
    from src.core.symbol import SymbolShell, VACResult, BoundaryResult
    from src.core.lambda_g import LambdaGOperator, PHI
    from src.core.intelligence.real_ai import RealAI
    from bazinga import BAZINGA

    print("✓ All imports test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("BAZINGA TEST SUITE")
    print("=" * 50)
    print()

    tests = [
        test_imports,
        test_phi_constant,
        test_symbol_shell,
        test_symbol_shell_regular,
        test_lambda_g,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
