#!/usr/bin/env python3
"""
BAZINGA CLI Test Suite
======================
Tests for all CLI commands to prevent regressions.

Run with: pytest tests/test_cli.py -v
Or: python tests/test_cli.py
"""

import subprocess
import sys
import os
import warnings
from pathlib import Path

# Suppress chromadb/pydantic warnings globally BEFORE any imports
warnings.filterwarnings("ignore", message=".*unable to infer type.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Also set PYTHONWARNINGS env var for subprocess imports
os.environ["PYTHONWARNINGS"] = "ignore"

# Timeout for each command (seconds)
CMD_TIMEOUT = 30

def run_bazinga(*args, timeout=CMD_TIMEOUT):
    """Run bazinga command and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "bazinga"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


class TestCLIBasic:
    """Basic CLI tests."""

    def test_version(self):
        """Test --version flag."""
        code, out, err = run_bazinga("--version")
        assert code == 0, f"--version failed: {err}"
        assert "5.0" in out or "bazinga" in out.lower(), f"Version not in output: {out}"
        print("✓ --version works")

    def test_help(self):
        """Test --help flag."""
        code, out, err = run_bazinga("--help")
        assert code == 0, f"--help failed: {err}"
        assert "BAZINGA" in out, f"Help missing BAZINGA: {out}"
        assert "--ask" in out, f"Help missing --ask: {out}"
        assert "--multi-ai" in out, f"Help missing --multi-ai: {out}"
        print("✓ --help works")

    def test_check(self):
        """Test --check system diagnostic."""
        code, out, err = run_bazinga("--check")
        assert code == 0, f"--check failed: {err}"
        assert "BAZINGA SYSTEM CHECK" in out, f"Check output wrong: {out}"
        assert "Python" in out, f"Python check missing: {out}"
        print("✓ --check works")

    def test_constants(self):
        """Test --constants flag."""
        code, out, err = run_bazinga("--constants")
        assert code == 0, f"--constants failed: {err}"
        assert "1.618" in out or "PHI" in out.upper(), f"PHI not in constants: {out}"
        print("✓ --constants works")

    def test_stats(self):
        """Test --stats flag."""
        code, out, err = run_bazinga("--stats")
        assert code == 0, f"--stats failed: {err}"
        assert "Stats" in out or "Patterns" in out or "Trust" in out, f"Stats output wrong: {out}"
        print("✓ --stats works")


class TestCLIBlockchain:
    """Blockchain-related tests."""

    def test_chain(self):
        """Test --chain flag."""
        code, out, err = run_bazinga("--chain")
        assert code == 0, f"--chain failed: {err}"
        assert "block" in out.lower() or "chain" in out.lower(), f"Chain output wrong: {out}"
        print("✓ --chain works")

    def test_verify_block(self):
        """Test --verify with block number."""
        code, out, err = run_bazinga("--verify", "1")
        assert code == 0, f"--verify failed: {err}"
        # Should find block #1 (genesis or first mined)
        assert "Block" in out or "VERIFICATION" in out, f"Verify output wrong: {out}"
        print("✓ --verify (block) works")

    def test_wallet(self):
        """Test --wallet flag."""
        code, out, err = run_bazinga("--wallet")
        assert code == 0, f"--wallet failed: {err}"
        assert "bzn_" in out or "wallet" in out.lower() or "identity" in out.lower(), f"Wallet output wrong: {out}"
        print("✓ --wallet works")


class TestCLIQuantum:
    """Quantum processing tests."""

    def test_quantum(self):
        """Test --quantum flag."""
        code, out, err = run_bazinga("--quantum", "test pattern")
        assert code == 0, f"--quantum failed: {err}"
        assert "Quantum" in out or "Coherence" in out or "Essence" in out, f"Quantum output wrong: {out}"
        print("✓ --quantum works")

    def test_coherence(self):
        """Test --coherence flag."""
        code, out, err = run_bazinga("--coherence", "phi golden ratio")
        assert code == 0, f"--coherence failed: {err}"
        print("✓ --coherence works")


class TestCLIKnowledgeBase:
    """Knowledge Base tests."""

    def test_kb_sources(self):
        """Test --kb-sources flag."""
        code, out, err = run_bazinga("--kb-sources")
        assert code == 0, f"--kb-sources failed: {err}"
        assert "KB" in out or "Source" in out or "Gmail" in out or "GDrive" in out, f"KB sources output wrong: {out}"
        print("✓ --kb-sources works")

    def test_kb_search(self):
        """Test --kb search."""
        code, out, err = run_bazinga("--kb", "test")
        assert code == 0, f"--kb failed: {err}"
        # Should show search results or "no results"
        print("✓ --kb search works")


class TestImports:
    """Test that all imports work."""

    def test_import_bazinga(self):
        """Test importing bazinga package."""
        # Skip on Python 3.14+ due to chromadb/pydantic v1 incompatibility
        if sys.version_info >= (3, 14):
            print("⚠ Skipping import tests on Python 3.14+ (chromadb compat)")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import bazinga
        assert hasattr(bazinga, "__version__"), "bazinga missing __version__"
        assert "5.0" in bazinga.__version__, f"Version wrong: {bazinga.__version__}"
        print(f"✓ import bazinga works (v{bazinga.__version__})")

    def test_import_constants(self):
        """Test importing constants."""
        if sys.version_info >= (3, 14):
            print("⚠ Skipping (Python 3.14+)")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from bazinga.constants import PHI, ALPHA, CONSCIOUSNESS_SCALE
        assert PHI == 1.618033988749895, f"PHI wrong: {PHI}"
        assert ALPHA == 137, f"ALPHA wrong: {ALPHA}"
        # V2: CONSCIOUSNESS_SCALE is now PHI (was 6.46 in V1)
        assert CONSCIOUSNESS_SCALE == PHI, f"CONSCIOUSNESS_SCALE wrong: {CONSCIOUSNESS_SCALE} (expected PHI)"
        print("✓ import constants works")

    def test_import_quantum(self):
        """Test importing quantum processor."""
        if sys.version_info >= (3, 14):
            print("⚠ Skipping (Python 3.14+)")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from bazinga.quantum import QuantumProcessor
        qp = QuantumProcessor()
        result = qp.process("test")
        assert "dominant_essence" in result or "essence" in str(result).lower(), f"Quantum result wrong: {result}"
        print("✓ import quantum works")

    def test_import_blockchain(self):
        """Test importing blockchain."""
        if sys.version_info >= (3, 14):
            print("⚠ Skipping (Python 3.14+)")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from bazinga.blockchain import create_chain
        chain = create_chain()
        assert len(chain.blocks) >= 1, "Chain should have at least genesis block"
        print(f"✓ import blockchain works ({len(chain.blocks)} blocks)")

    def test_import_inter_ai(self):
        """Test importing inter_ai consensus."""
        if sys.version_info >= (3, 14):
            print("⚠ Skipping (Python 3.14+)")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from bazinga.inter_ai import InterAIConsensus, CerebrasParticipant
        # Just test import, don't run actual queries
        print("✓ import inter_ai works")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("  BAZINGA CLI TEST SUITE")
    print("=" * 60 + "\n")

    test_classes = [
        TestImports,
        TestCLIBasic,
        TestCLIBlockchain,
        TestCLIQuantum,
        TestCLIKnowledgeBase,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        getattr(instance, method_name)()
                    passed += 1
                except AssertionError as e:
                    failed += 1
                    errors.append(f"{test_class.__name__}.{method_name}: {e}")
                    print(f"✗ {method_name} FAILED: {e}")
                except Exception as e:
                    failed += 1
                    errors.append(f"{test_class.__name__}.{method_name}: {e}")
                    print(f"✗ {method_name} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nFAILURES:")
        for err in errors:
            print(f"  - {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
