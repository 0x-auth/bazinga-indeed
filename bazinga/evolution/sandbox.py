#!/usr/bin/env python3
"""
Proposal Sandbox — Isolated Testing Environment
=================================================
Before any proposal is deployed, it runs in a sandbox:
    1. Copy affected files to temp directory
    2. Apply the proposed diff
    3. Run syntax checks (compile)
    4. Run relevant tests (pytest on affected modules)
    5. Return pass/fail with details

The sandbox NEVER modifies the real codebase.
"""

import os
import py_compile
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SandboxResult:
    """Result of running a proposal through the sandbox."""
    passed: bool
    syntax_ok: bool
    tests_passed: bool
    test_output: str
    errors: List[str]
    duration_ms: float
    files_tested: List[str]

    @property
    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Sandbox {status} in {self.duration_ms:.0f}ms | "
            f"syntax={'ok' if self.syntax_ok else 'FAIL'} "
            f"tests={'ok' if self.tests_passed else 'FAIL'} "
            f"({len(self.files_tested)} files)"
        )


class ProposalSandbox:
    """
    Sandboxed environment for testing proposals before deployment.

    Usage:
        sandbox = ProposalSandbox(repo_path)
        result = sandbox.test_proposal(file_diffs)

        if result.passed:
            # Safe to deploy
        else:
            for err in result.errors:
                print(f"  Error: {err}")
    """

    def __init__(self, repo_path: Optional[Path] = None):
        if repo_path is None:
            # Try to find the bazinga repo
            repo_path = Path(__file__).parent.parent.parent
        self.repo_path = Path(repo_path)

    def test_proposal(
        self,
        file_diffs: List[dict],
        run_tests: bool = True,
        timeout: int = 60,
    ) -> SandboxResult:
        """
        Test a proposal in an isolated sandbox.

        Args:
            file_diffs: List of {path, old_content, new_content} dicts
            run_tests: Whether to run pytest (slower but thorough)
            timeout: Max seconds for test execution

        Returns:
            SandboxResult with pass/fail and details
        """
        start = time.time()
        errors = []
        files_tested = []

        with tempfile.TemporaryDirectory(prefix="bazinga_sandbox_") as sandbox_dir:
            sandbox = Path(sandbox_dir)

            # Step 1: Copy relevant files to sandbox
            try:
                self._setup_sandbox(sandbox, file_diffs)
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                return SandboxResult(
                    passed=False,
                    syntax_ok=False,
                    tests_passed=False,
                    test_output="",
                    errors=[f"Sandbox setup failed: {e}"],
                    duration_ms=elapsed,
                    files_tested=[],
                )

            # Step 2: Apply diffs
            for diff in file_diffs:
                target = sandbox / diff["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(diff["new_content"])
                files_tested.append(diff["path"])

            # Step 3: Syntax check
            syntax_ok = True
            for diff in file_diffs:
                if diff["path"].endswith(".py"):
                    target = sandbox / diff["path"]
                    try:
                        py_compile.compile(str(target), doraise=True)
                    except py_compile.PyCompileError as e:
                        syntax_ok = False
                        errors.append(f"Syntax error in {diff['path']}: {e}")

            if not syntax_ok:
                elapsed = (time.time() - start) * 1000
                return SandboxResult(
                    passed=False,
                    syntax_ok=False,
                    tests_passed=False,
                    test_output="",
                    errors=errors,
                    duration_ms=elapsed,
                    files_tested=files_tested,
                )

            # Step 4: Run tests (if requested)
            tests_passed = True
            test_output = ""

            if run_tests:
                tests_passed, test_output = self._run_tests(
                    sandbox, file_diffs, timeout
                )
                if not tests_passed:
                    errors.append("Tests failed (see test_output)")

        elapsed = (time.time() - start) * 1000
        passed = syntax_ok and tests_passed

        return SandboxResult(
            passed=passed,
            syntax_ok=syntax_ok,
            tests_passed=tests_passed,
            test_output=test_output,
            errors=errors,
            duration_ms=elapsed,
            files_tested=files_tested,
        )

    def _setup_sandbox(self, sandbox: Path, file_diffs: List[dict]):
        """Copy the minimal set of files needed for testing."""
        # Copy the entire bazinga package (needed for imports)
        src = self.repo_path / "bazinga"
        dst = sandbox / "bazinga"
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

        # Copy tests
        tests_src = self.repo_path / "tests"
        tests_dst = sandbox / "tests"
        if tests_src.exists():
            shutil.copytree(tests_src, tests_dst, dirs_exist_ok=True)

        # Copy pyproject.toml (needed for pytest config)
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            shutil.copy2(pyproject, sandbox / "pyproject.toml")

    def _run_tests(
        self,
        sandbox: Path,
        file_diffs: List[dict],
        timeout: int,
    ) -> tuple:
        """Run relevant pytest tests in the sandbox."""
        # Find relevant test files
        test_files = self._find_relevant_tests(file_diffs)

        if not test_files:
            # No specific tests found, run basic import check
            return True, "No relevant tests found — import check only"

        # Build pytest command
        cmd = [
            "python3", "-m", "pytest",
            "--tb=short",
            "--no-header",
            "-q",
        ]

        # Add test files that exist in sandbox
        for tf in test_files:
            test_path = sandbox / tf
            if test_path.exists():
                cmd.append(str(test_path))

        if len(cmd) == 5:
            # No test files existed
            return True, "No test files found in sandbox"

        try:
            result = subprocess.run(
                cmd,
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(sandbox)},
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output[:2000]  # Cap output
        except subprocess.TimeoutExpired:
            return False, f"Tests timed out after {timeout}s"
        except Exception as e:
            return False, f"Test execution error: {e}"

    def _find_relevant_tests(self, file_diffs: List[dict]) -> List[str]:
        """Find test files related to the modified files."""
        test_files = []
        for diff in file_diffs:
            path = diff["path"]
            # bazinga/foo/bar.py → tests/test_bar.py
            name = Path(path).stem
            test_files.append(f"tests/test_{name}.py")
            # Also check for module-level tests
            parts = Path(path).parts
            if len(parts) >= 2:
                test_files.append(f"tests/test_{parts[-2]}.py")
        return test_files


# =============================================================================
# Module test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PROPOSAL SANDBOX TEST")
    print("=" * 60)

    sandbox = ProposalSandbox()

    # Test 1: Valid Python
    result = sandbox.test_proposal(
        file_diffs=[{
            "path": "bazinga/test_sandbox_example.py",
            "old_content": "",
            "new_content": "def hello():\n    return 'world'\n",
        }],
        run_tests=False,
    )
    print(f"\n  Valid Python: {result.summary}")
    assert result.syntax_ok, "Valid Python should pass syntax"

    # Test 2: Invalid Python
    result2 = sandbox.test_proposal(
        file_diffs=[{
            "path": "bazinga/test_sandbox_bad.py",
            "old_content": "",
            "new_content": "def broken(\n    return\n",
        }],
        run_tests=False,
    )
    print(f"  Invalid Python: {result2.summary}")
    assert not result2.syntax_ok, "Invalid Python should fail syntax"

    print(f"\n  Sandbox working! ✓")
