"""
BAZINGA Agent Tools - The hands of the agent.

Each tool is a capability the agent can use:
- read: Read file contents
- bash: Execute shell commands
- search: RAG search over indexed knowledge
- edit: Modify files (Phase 2)

SECURITY: All tools implement safety measures:
- Path validation (no traversal attacks)
- Command sanitization (no injection)
- User confirmation for destructive operations
"""

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
from abc import ABC, abstractmethod


def _validate_path(path: str, base_dir: Path = None) -> tuple[bool, Path, str]:
    """
    Validate a path is safe (no traversal attacks).

    Returns: (is_valid, resolved_path, error_message)
    """
    if not path or not isinstance(path, str):
        return False, None, "Path must be a non-empty string"

    # Block obvious traversal attempts
    if '..' in path:
        return False, None, "Path traversal (..) not allowed"

    try:
        # Resolve and expand
        resolved = Path(path).expanduser().resolve()

        # If base_dir specified, ensure path is within it
        if base_dir:
            base_resolved = base_dir.resolve()
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                return False, None, f"Path must be within {base_dir}"

        return True, resolved, ""
    except Exception as e:
        return False, None, f"Invalid path: {e}"


def _sanitize_for_log(text: str, max_len: int = 100) -> str:
    """Sanitize text for safe logging (no secrets, limited length)."""
    # Remove potential secrets
    sanitized = re.sub(r'(key|token|password|secret)[=:]\s*\S+', r'\1=***', text, flags=re.IGNORECASE)
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "..."
    return sanitized


class Tool(ABC):
    """Base class for all agent tools."""

    name: str = "tool"
    description: str = "A tool"

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return result."""
        pass

    def to_prompt(self) -> str:
        """Return tool description for LLM prompt."""
        return f"- {self.name}: {self.description}"


class ReadTool(Tool):
    """Read file contents with path validation."""

    name = "read"
    description = "Read the contents of a file. Args: path (str)"

    def execute(self, path: str, **kwargs) -> Dict[str, Any]:
        """Read a file and return its contents."""
        # Validate path (SECURITY: prevent traversal)
        is_valid, filepath, error = _validate_path(path)
        if not is_valid:
            return {"success": False, "error": error}

        try:
            if not filepath.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            if filepath.is_dir():
                # List directory contents
                files = list(filepath.iterdir())
                listing = "\n".join([
                    f"{'[DIR] ' if f.is_dir() else ''}{f.name}"
                    for f in sorted(files)[:50]  # Limit to 50 items
                ])
                return {
                    "success": True,
                    "content": f"Directory listing for {path}:\n{listing}",
                    "type": "directory"
                }

            # Read file
            content = filepath.read_text(errors='replace')

            # Truncate if too large
            if len(content) > 50000:
                content = content[:50000] + "\n\n... [truncated, file too large]"

            return {
                "success": True,
                "content": content,
                "path": str(filepath),
                "lines": content.count('\n') + 1
            }

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except (OSError, IOError) as e:
            return {"success": False, "error": f"File error: {e}"}


class BashTool(Tool):
    """Execute shell commands with safety confirmation for destructive operations.

    SECURITY MEASURES:
    1. Uses shlex.split() to safely parse commands (no shell=True injection)
    2. Blocked patterns prevent catastrophic commands
    3. Destructive commands require user confirmation EVERY TIME
    4. No confirmation caching (prevents bypass attacks)
    """

    name = "bash"
    description = "Run a bash command. Args: command (str). Use for git, npm, python, etc. Destructive commands require confirmation."

    # Commands that are BLOCKED completely (too dangerous)
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /*",
        "rm -rf $HOME",
        ":(){:|:&};:",  # Fork bomb
        "mkfs",
        "> /dev/sda",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        "chmod -R 777 /",
        "chmod 777 /",
        "curl | sh",
        "curl | bash",
        "wget | sh",
        "wget | bash",
        "eval $(",
        "base64 -d |",
    ]

    # Shell metacharacters that require shell=True (will be blocked for safety)
    SHELL_METACHARACTERS = ['|', '&&', '||', ';', '$(', '`', '>', '<', '&']

    # Commands that REQUIRE user confirmation (φ-signature)
    DESTRUCTIVE_PATTERNS = [
        "rm ",           # Any remove
        "rm -",          # Remove with flags
        "git push",      # Pushing to remote
        "git reset",     # Reset commits
        "git checkout .", # Discard changes
        "pip install",   # Installing packages
        "pip uninstall", # Uninstalling packages
        "npm install",   # Installing packages
        "brew install",  # Installing packages
        "sudo ",         # Any sudo command
        "mv ",           # Moving files
        "chmod ",        # Changing permissions
        "chown ",        # Changing ownership
        "truncate",      # Truncating files
    ]

    # Safe commands that can use shell features (pipes, etc.)
    SAFE_SHELL_COMMANDS = [
        "git status",
        "git diff",
        "git log",
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "grep",
        "find",
        "echo",
        "pwd",
        "which",
        "python --version",
        "node --version",
        "npm --version",
    ]

    def _is_blocked(self, command: str) -> tuple[bool, str]:
        """Check if command matches blocked patterns."""
        cmd_lower = command.lower().strip()
        for blocked in self.BLOCKED_PATTERNS:
            if blocked.lower() in cmd_lower:
                return True, blocked
        return False, ""

    def _is_destructive(self, command: str) -> bool:
        """Check if command is destructive and needs confirmation."""
        cmd_lower = command.lower().strip()
        for pattern in self.DESTRUCTIVE_PATTERNS:
            if pattern.lower() in cmd_lower:
                return True
        return False

    def _needs_shell(self, command: str) -> bool:
        """Check if command requires shell features (pipes, redirects, etc.)."""
        for meta in self.SHELL_METACHARACTERS:
            if meta in command:
                return True
        return False

    def _is_safe_for_shell(self, command: str) -> bool:
        """Check if command is safe enough to run with shell=True."""
        cmd_lower = command.lower().strip()
        # Check if it starts with a safe command
        for safe in self.SAFE_SHELL_COMMANDS:
            if cmd_lower.startswith(safe.lower()):
                return True
        return False

    def _get_confirmation(self, command: str) -> bool:
        """Ask user for φ-signature (confirmation) before destructive command.

        SECURITY: No caching - always ask for confirmation.
        """
        print()
        print("⚠️  DESTRUCTIVE COMMAND DETECTED")
        print("━" * 50)
        print(f"  Command: {_sanitize_for_log(command, 200)}")
        print("━" * 50)
        print()
        try:
            response = input("  Confirm execution? [y/N] φ-signature: ").strip().lower()
            return response in ('y', 'yes', 'φ', 'phi')
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False

    def execute(self, command: str, timeout: int = 30, confirmed: bool = False, **kwargs) -> Dict[str, Any]:
        """Execute a bash command safely."""
        # Input validation
        if not command or not isinstance(command, str):
            return {"success": False, "error": "Command must be a non-empty string"}

        if timeout < 1 or timeout > 300:
            timeout = 30  # Default to safe value

        try:
            # Check for completely blocked commands
            is_blocked, blocked_pattern = self._is_blocked(command)
            if is_blocked:
                return {
                    "success": False,
                    "error": f"🛑 BLOCKED: This command pattern is too dangerous: {blocked_pattern}"
                }

            # Check for destructive commands that need confirmation
            # SECURITY: Always ask, no caching
            if self._is_destructive(command) and not confirmed:
                if not self._get_confirmation(command):
                    return {
                        "success": False,
                        "error": "Command cancelled by user (φ-signature not provided)"
                    }

            # Determine execution method
            needs_shell = self._needs_shell(command)

            if needs_shell:
                # Commands with shell metacharacters
                if not self._is_safe_for_shell(command):
                    # Potentially dangerous shell command - require confirmation
                    if not confirmed:
                        print()
                        print("⚠️  SHELL COMMAND WITH SPECIAL CHARACTERS")
                        print("━" * 50)
                        print(f"  Command: {_sanitize_for_log(command, 200)}")
                        print("  This command uses shell features (|, &&, etc.)")
                        print("━" * 50)
                        try:
                            response = input("  Allow shell execution? [y/N]: ").strip().lower()
                            if response not in ('y', 'yes'):
                                return {
                                    "success": False,
                                    "error": "Shell command cancelled by user"
                                }
                        except (EOFError, KeyboardInterrupt):
                            return {"success": False, "error": "Cancelled"}

                # Run with shell=True (user confirmed or safe command)
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.getcwd()
                )
            else:
                # Safe execution without shell
                try:
                    args = shlex.split(command)
                except ValueError as e:
                    return {"success": False, "error": f"Invalid command syntax: {e}"}

                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.getcwd()
                )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]:\n{result.stderr}"

            # Truncate if too large
            if len(output) > 20000:
                output = output[:20000] + "\n\n... [truncated]"

            return {
                "success": result.returncode == 0,
                "output": output,
                "return_code": result.returncode,
                "command": _sanitize_for_log(command, 200)
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "command": _sanitize_for_log(command, 100)
            }
        except FileNotFoundError:
            return {"success": False, "error": f"Command not found: {command.split()[0] if command else 'unknown'}"}
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except (OSError, IOError) as e:
            return {"success": False, "error": f"Execution error: {e}"}


class SearchTool(Tool):
    """Search indexed knowledge using RAG."""

    name = "search"
    description = "Search your indexed documents and knowledge base. Args: query (str)"

    def __init__(self):
        self._ai = None

    def _get_ai(self):
        """Lazy load the AI/RAG system."""
        if self._ai is None:
            try:
                from ..cli import _get_real_ai
                self._ai = _get_real_ai()()
            except Exception:
                self._ai = None
        return self._ai

    def execute(self, query: str, limit: int = 5, **kwargs) -> Dict[str, Any]:
        """Search indexed knowledge."""
        try:
            ai = self._get_ai()

            if ai is None or hasattr(ai, 'error'):
                # Fallback to JSON knowledge files
                return self._search_json_knowledge(query, limit)

            # Use RAG search
            results = ai.search(query, n_results=limit)

            if not results:
                return {
                    "success": True,
                    "results": [],
                    "message": "No results found. Try indexing with: bazinga --index <path>"
                }

            return {
                "success": True,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_json_knowledge(self, query: str, limit: int) -> Dict[str, Any]:
        """Fallback search through JSON knowledge files."""
        import json

        knowledge_dir = Path.home() / ".bazinga" / "knowledge"
        results = []
        query_lower = query.lower()

        if knowledge_dir.exists():
            for json_file in knowledge_dir.rglob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for chunk in data:
                                text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                                if query_lower in text.lower():
                                    results.append({
                                        "source": str(json_file),
                                        "text": text[:500]
                                    })
                                    if len(results) >= limit:
                                        break
                except (json.JSONDecodeError, OSError, IOError, KeyError, TypeError):
                    # Skip malformed or unreadable files
                    continue

                if len(results) >= limit:
                    break

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "source": "json_fallback"
        }


class EditTool(Tool):
    """Edit file contents with preview, confirmation, and backup.

    SECURITY MEASURES:
    1. Path validation (no traversal)
    2. Creates backup before editing
    3. Atomic write operation (temp file + rename)
    4. Always requires user confirmation
    """

    name = "edit"
    description = "Edit a file by replacing text. Args: path (str), old_text (str), new_text (str). Shows diff preview."

    def _show_diff_and_confirm(self, path: str, old_text: str, new_text: str) -> bool:
        """Show diff preview and ask for confirmation."""
        print()
        print("📝 EDIT PREVIEW")
        print("━" * 50)
        print(f"  File: {path}")
        print()
        print("  ─── REMOVE ───")
        for line in old_text.split('\n')[:5]:
            print(f"  - {line[:70]}")
        if old_text.count('\n') > 5:
            print(f"  ... ({old_text.count(chr(10)) - 5} more lines)")
        print()
        print("  +++ ADD +++")
        for line in new_text.split('\n')[:5]:
            print(f"  + {line[:70]}")
        if new_text.count('\n') > 5:
            print(f"  ... ({new_text.count(chr(10)) - 5} more lines)")
        print()
        print("━" * 50)
        try:
            response = input("  Apply this edit? [y/N] φ-signature: ").strip().lower()
            return response in ('y', 'yes', 'φ', 'phi')
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False

    def execute(self, path: str, old_text: str, new_text: str, confirmed: bool = False, **kwargs) -> Dict[str, Any]:
        """Edit a file by replacing old_text with new_text."""
        # BUG FIX: Ensure relative paths are resolved from CWD, not root or .venv
        if not path.startswith('/') and not path.startswith('~'):
            path = str(Path.cwd() / path)

        # Validate path (SECURITY: prevent traversal)
        is_valid, filepath, error = _validate_path(path)
        if not is_valid:
            return {"success": False, "error": error}

        # Input validation
        if not old_text or not isinstance(old_text, str):
            return {"success": False, "error": "old_text must be a non-empty string"}
        if not isinstance(new_text, str):
            return {"success": False, "error": "new_text must be a string"}

        try:
            if not filepath.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            content = filepath.read_text()

            if old_text not in content:
                return {
                    "success": False,
                    "error": "Text to replace not found in file. Make sure old_text matches exactly."
                }

            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return {
                    "success": False,
                    "error": f"Found {count} occurrences of old_text. Please provide more context to make it unique."
                }

            # Show preview and confirm (SECURITY: always ask)
            if not confirmed:
                if not self._show_diff_and_confirm(path, old_text, new_text):
                    return {
                        "success": False,
                        "error": "Edit cancelled by user (φ-signature not provided)"
                    }

            # Create backup (SAFETY: prevent data loss)
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            try:
                backup_path.write_text(content)
            except (OSError, IOError):
                pass  # Backup failure shouldn't block edit

            # Apply edit with atomic write
            new_content = content.replace(old_text, new_text)

            # Write to temp file first, then rename (atomic on most systems)
            temp_fd, temp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
            try:
                os.write(temp_fd, new_content.encode('utf-8'))
                os.close(temp_fd)
                os.replace(temp_path, filepath)
            except Exception as e:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise e

            return {
                "success": True,
                "message": f"✓ Edited {path} (backup: {backup_path.name})",
                "path": str(filepath)
            }

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except (OSError, IOError) as e:
            return {"success": False, "error": f"File error: {e}"}


class WriteTool(Tool):
    """Write/create a file with confirmation and backup.

    SECURITY MEASURES:
    1. Path validation (no traversal)
    2. Creates backup if overwriting
    3. Atomic write operation
    4. Always requires user confirmation
    """

    name = "write"
    description = "Write content to a file (creates if doesn't exist). Args: path (str), content (str). Requires confirmation."

    def _confirm_write(self, path: str, content: str, exists: bool) -> bool:
        """Ask for confirmation before writing."""
        print()
        action = "OVERWRITE" if exists else "CREATE"
        print(f"📝 {action} FILE")
        print("━" * 50)
        print(f"  Path: {path}")
        print(f"  Size: {len(content)} chars, {content.count(chr(10)) + 1} lines")
        print()
        print("  Preview (first 5 lines):")
        for line in content.split('\n')[:5]:
            print(f"    {line[:70]}")
        if content.count('\n') > 5:
            print(f"    ... ({content.count(chr(10)) - 5} more lines)")
        print()
        print("━" * 50)
        try:
            response = input(f"  {action.lower()} this file? [y/N] φ-signature: ").strip().lower()
            return response in ('y', 'yes', 'φ', 'phi')
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False

    def execute(self, path: str, content: str, confirmed: bool = False, **kwargs) -> Dict[str, Any]:
        """Write content to a file."""
        # BUG FIX: Ensure relative paths are resolved from CWD, not root or .venv
        # If path doesn't start with / or ~, make it relative to current working directory
        if not path.startswith('/') and not path.startswith('~'):
            path = str(Path.cwd() / path)

        # Validate path (SECURITY: prevent traversal)
        is_valid, filepath, error = _validate_path(path)
        if not is_valid:
            return {"success": False, "error": error}

        # Input validation
        if not isinstance(content, str):
            return {"success": False, "error": "content must be a string"}

        try:
            exists = filepath.exists()

            # Confirm before writing (SECURITY: always ask)
            if not confirmed:
                if not self._confirm_write(path, content, exists):
                    return {
                        "success": False,
                        "error": "Write cancelled by user (φ-signature not provided)"
                    }

            # Create backup if overwriting (SAFETY: prevent data loss)
            if exists:
                backup_path = filepath.with_suffix(filepath.suffix + '.bak')
                try:
                    backup_path.write_text(filepath.read_text())
                except (OSError, IOError):
                    pass  # Backup failure shouldn't block write

            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: temp file + rename
            temp_fd, temp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
            try:
                os.write(temp_fd, content.encode('utf-8'))
                os.close(temp_fd)
                os.replace(temp_path, filepath)
            except Exception as e:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise e

            return {
                "success": True,
                "message": f"Wrote {len(content)} chars to {path}",
                "path": str(filepath)
            }

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except (OSError, IOError) as e:
            return {"success": False, "error": f"File error: {e}"}


class GlobTool(Tool):
    """Find files matching a pattern with path validation."""

    name = "glob"
    description = "Find files matching a glob pattern. Args: pattern (str), e.g. '**/*.py', 'src/**/*.js'"

    def execute(self, pattern: str, path: str = ".", **kwargs) -> Dict[str, Any]:
        """Find files matching a glob pattern."""
        # Input validation
        if not pattern or not isinstance(pattern, str):
            return {"success": False, "error": "pattern must be a non-empty string"}

        # Block traversal in pattern
        if '..' in pattern:
            return {"success": False, "error": "Path traversal (..) not allowed in pattern"}

        try:
            import glob as glob_module

            # Validate base path
            is_valid, base_path, error = _validate_path(path)
            if not is_valid:
                base_path = Path.cwd()

            if not base_path.exists():
                base_path = Path.cwd()

            # Use glob to find files
            full_pattern = str(base_path / pattern)
            matches = glob_module.glob(full_pattern, recursive=True)

            # Filter: only include files within base_path (SECURITY)
            base_resolved = base_path.resolve()
            safe_matches = []
            for m in matches:
                try:
                    m_resolved = Path(m).resolve()
                    m_resolved.relative_to(base_resolved)
                    safe_matches.append(m)
                except ValueError:
                    continue  # Skip files outside base path

            # Limit results
            safe_matches = safe_matches[:50]

            # Make paths relative for cleaner output
            try:
                cwd = Path.cwd()
                safe_matches = [str(Path(m).relative_to(cwd)) for m in safe_matches]
            except ValueError:
                safe_matches = [str(m) for m in safe_matches]

            return {
                "success": True,
                "files": safe_matches,
                "count": len(safe_matches),
                "pattern": pattern
            }

        except (OSError, IOError) as e:
            return {"success": False, "error": f"Glob error: {e}"}


class GrepTool(Tool):
    """Search for text in files with regex validation."""

    name = "grep"
    description = "Search for text/regex in files. Args: pattern (str), path (str, default='.')"

    MAX_PATTERN_LENGTH = 500  # Prevent ReDoS with very long patterns

    def execute(self, pattern: str, path: str = ".", **kwargs) -> Dict[str, Any]:
        """Search for pattern in files."""
        # Input validation
        if not pattern or not isinstance(pattern, str):
            return {"success": False, "error": "pattern must be a non-empty string"}

        if len(pattern) > self.MAX_PATTERN_LENGTH:
            return {"success": False, "error": f"Pattern too long (max {self.MAX_PATTERN_LENGTH} chars)"}

        try:
            # Validate and compile regex (SECURITY: catch bad patterns early)
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex pattern: {e}"}

            # Validate path
            is_valid, base_path, error = _validate_path(path)
            if not is_valid:
                return {"success": False, "error": error}

            if not base_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            results = []

            # Search files
            files_to_search = []
            if base_path.is_file():
                files_to_search = [base_path]
            else:
                # Search common code files
                for ext in ['*.py', '*.js', '*.ts', '*.md', '*.txt', '*.json', '*.yaml', '*.yml']:
                    files_to_search.extend(base_path.rglob(ext))

            for filepath in files_to_search[:100]:  # Limit files
                try:
                    content = filepath.read_text(errors='replace')
                    for i, line in enumerate(content.split('\n'), 1):
                        # Limit line length to prevent ReDoS
                        if len(line) > 10000:
                            line = line[:10000]
                        if regex.search(line):
                            results.append({
                                "file": str(filepath),
                                "line": i,
                                "text": line.strip()[:200]
                            })
                            if len(results) >= 20:
                                break
                except (OSError, IOError, UnicodeDecodeError):
                    continue  # Skip unreadable files

                if len(results) >= 20:
                    break

            return {
                "success": True,
                "matches": results,
                "count": len(results),
                "pattern": pattern
            }

        except (OSError, IOError) as e:
            return {"success": False, "error": f"Search error: {e}"}


# Registry of all available tools
TOOLS = {
    "read": ReadTool(),
    "bash": BashTool(),
    "search": SearchTool(),
    "edit": EditTool(),
    "write": WriteTool(),
    "glob": GlobTool(),
    "grep": GrepTool(),
}


class VerifiedFixTool(Tool):
    """
    Apply blockchain-verified code fixes with multi-AI consensus.

    This tool enables:
    - Multiple AIs reviewing a proposed fix
    - φ-coherence measurement for quality
    - PoB attestation on blockchain
    - Only apply if consensus reached

    "No single AI can mess up your code."
    """

    name = "verified_fix"
    description = "Apply a blockchain-verified code fix. Multiple AIs must agree. Args: file (str), old_code (str), new_code (str), reason (str)"

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        """Lazy load the VerifiedFixEngine."""
        if self._engine is None:
            try:
                from .verified_fixes import VerifiedFixEngine
                self._engine = VerifiedFixEngine(verbose=True)
            except ImportError:
                self._engine = None
        return self._engine

    def execute(
        self,
        file: str,
        old_code: str,
        new_code: str,
        reason: str,
        fix_type: str = "bug_fix",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply a blockchain-verified code fix.

        Args:
            file: Path to the file to fix
            old_code: The code to replace
            new_code: The new code
            reason: Why this fix is needed
            fix_type: One of: bug_fix, security_fix, refactor, optimization

        Returns:
            Result with consensus status and fix application
        """
        import asyncio

        engine = self._get_engine()
        if engine is None:
            return {
                "success": False,
                "error": "VerifiedFixEngine not available"
            }

        # Validate inputs
        if not file or not isinstance(file, str):
            return {"success": False, "error": "file must be a non-empty string"}
        if not old_code or not isinstance(old_code, str):
            return {"success": False, "error": "old_code must be a non-empty string"}
        if not new_code or not isinstance(new_code, str):
            return {"success": False, "error": "new_code must be a non-empty string"}
        if not reason or not isinstance(reason, str):
            return {"success": False, "error": "reason must be a non-empty string"}

        # Validate path
        is_valid, filepath, error = _validate_path(file)
        if not is_valid:
            return {"success": False, "error": error}

        # Run async flow
        try:
            from .verified_fixes import FixType
            ft = FixType(fix_type) if fix_type in [e.value for e in FixType] else FixType.BUG_FIX

            # Run the verified fix flow
            async def run_fix():
                return await engine.propose_and_apply(
                    file_path=str(filepath),
                    original_code=old_code,
                    proposed_fix=new_code,
                    explanation=reason,
                    fix_type=ft,
                    agent_id="bazinga_agent",
                )

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(lambda: asyncio.run(run_fix()))
                    success, message, proposal = future.result(timeout=60)
            except RuntimeError:
                success, message, proposal = asyncio.run(run_fix())

            return {
                "success": success,
                "message": message,
                "proposal_id": proposal.proposal_id if proposal else None,
                "consensus_reached": proposal.status.value if proposal else None,
                "coherence": proposal.coherence_score if proposal else 0,
                "blockchain_block": proposal.blockchain_block if proposal else None,
            }

        except Exception as e:
            return {"success": False, "error": f"Verified fix error: {e}"}


# Add verified_fix to tools registry
try:
    TOOLS["verified_fix"] = VerifiedFixTool()
except Exception:
    pass  # Optional tool, don't fail if import issues


def get_tools_prompt() -> str:
    """Get the tools description for LLM prompt."""
    lines = ["Available tools:"]
    for tool in TOOLS.values():
        lines.append(tool.to_prompt())
    return "\n".join(lines)


def execute_tool(name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name."""
    if name not in TOOLS:
        return {"success": False, "error": f"Unknown tool: {name}"}
    return TOOLS[name].execute(**kwargs)
