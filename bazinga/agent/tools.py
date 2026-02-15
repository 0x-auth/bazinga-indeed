"""
BAZINGA Agent Tools - The hands of the agent.

Each tool is a capability the agent can use:
- read: Read file contents
- bash: Execute shell commands
- search: RAG search over indexed knowledge
- edit: Modify files (Phase 2)
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


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
    """Read file contents."""

    name = "read"
    description = "Read the contents of a file. Args: path (str)"

    def execute(self, path: str, **kwargs) -> Dict[str, Any]:
        """Read a file and return its contents."""
        try:
            filepath = Path(path).expanduser()

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
        except Exception as e:
            return {"success": False, "error": str(e)}


class BashTool(Tool):
    """Execute shell commands with safety confirmation for destructive operations."""

    name = "bash"
    description = "Run a bash command. Args: command (str). Use for git, npm, python, etc. Destructive commands require confirmation."

    # Commands that are BLOCKED completely (too dangerous)
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /*",
        ":(){:|:&};:",  # Fork bomb
        "mkfs",
        "> /dev/sda",
        "dd if=/dev/zero",
        "chmod -R 777 /",
    ]

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
        "> ",            # Overwriting files
        "truncate",      # Truncating files
    ]

    # Track confirmed commands in session
    _confirmed_commands = set()

    def _is_destructive(self, command: str) -> bool:
        """Check if command is destructive and needs confirmation."""
        cmd_lower = command.lower().strip()
        for pattern in self.DESTRUCTIVE_PATTERNS:
            if pattern.lower() in cmd_lower:
                return True
        return False

    def _get_confirmation(self, command: str) -> bool:
        """Ask user for φ-signature (confirmation) before destructive command."""
        print()
        print("⚠️  DESTRUCTIVE COMMAND DETECTED")
        print("━" * 50)
        print(f"  Command: {command}")
        print("━" * 50)
        print()
        try:
            response = input("  Confirm execution? [y/N] φ-signature: ").strip().lower()
            return response in ('y', 'yes', 'φ', 'phi')
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False

    def execute(self, command: str, timeout: int = 30, confirmed: bool = False, **kwargs) -> Dict[str, Any]:
        """Execute a bash command."""
        try:
            # Check for completely blocked commands
            for blocked in self.BLOCKED_PATTERNS:
                if blocked in command:
                    return {
                        "success": False,
                        "error": f"🛑 BLOCKED: This command pattern is too dangerous: {blocked}"
                    }

            # Check for destructive commands that need confirmation
            if self._is_destructive(command) and not confirmed:
                # Check if already confirmed this session
                if command not in self._confirmed_commands:
                    if not self._get_confirmation(command):
                        return {
                            "success": False,
                            "error": "Command cancelled by user (φ-signature not provided)"
                        }
                    # Remember confirmation for this session
                    self._confirmed_commands.add(command)

            # Run command
            result = subprocess.run(
                command,
                shell=True,
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
                "command": command
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "command": command
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


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
                except:
                    pass

                if len(results) >= limit:
                    break

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "source": "json_fallback"
        }


class EditTool(Tool):
    """Edit file contents with preview and confirmation."""

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
        try:
            filepath = Path(path).expanduser()

            if not filepath.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            content = filepath.read_text()

            if old_text not in content:
                return {
                    "success": False,
                    "error": f"Text to replace not found in file. Make sure old_text matches exactly."
                }

            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return {
                    "success": False,
                    "error": f"Found {count} occurrences of old_text. Please provide more context to make it unique."
                }

            # Show preview and confirm
            if not confirmed:
                if not self._show_diff_and_confirm(path, old_text, new_text):
                    return {
                        "success": False,
                        "error": "Edit cancelled by user (φ-signature not provided)"
                    }

            # Apply edit
            new_content = content.replace(old_text, new_text)
            filepath.write_text(new_content)

            return {
                "success": True,
                "message": f"✓ Edited {path}",
                "path": str(filepath)
            }

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class WriteTool(Tool):
    """Write/create a file with confirmation."""

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
        try:
            filepath = Path(path).expanduser()
            exists = filepath.exists()

            # Confirm before writing
            if not confirmed:
                if not self._confirm_write(path, content, exists):
                    return {
                        "success": False,
                        "error": "Write cancelled by user (φ-signature not provided)"
                    }

            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            filepath.write_text(content)

            return {
                "success": True,
                "message": f"Wrote {len(content)} chars to {path}",
                "path": str(filepath)
            }

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GlobTool(Tool):
    """Find files matching a pattern."""

    name = "glob"
    description = "Find files matching a glob pattern. Args: pattern (str), e.g. '**/*.py', 'src/**/*.js'"

    def execute(self, pattern: str, path: str = ".", **kwargs) -> Dict[str, Any]:
        """Find files matching a glob pattern."""
        try:
            import glob as glob_module

            base_path = Path(path).expanduser()
            if not base_path.exists():
                base_path = Path.cwd()

            # Use glob to find files
            full_pattern = str(base_path / pattern)
            matches = glob_module.glob(full_pattern, recursive=True)

            # Limit results
            matches = matches[:50]

            # Make paths relative for cleaner output
            try:
                cwd = Path.cwd()
                matches = [str(Path(m).relative_to(cwd)) for m in matches]
            except ValueError:
                matches = [str(m) for m in matches]

            return {
                "success": True,
                "files": matches,
                "count": len(matches),
                "pattern": pattern
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class GrepTool(Tool):
    """Search for text in files."""

    name = "grep"
    description = "Search for text/regex in files. Args: pattern (str), path (str, default='.')"

    def execute(self, pattern: str, path: str = ".", **kwargs) -> Dict[str, Any]:
        """Search for pattern in files."""
        try:
            import re

            base_path = Path(path).expanduser()
            if not base_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            results = []
            regex = re.compile(pattern, re.IGNORECASE)

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
                        if regex.search(line):
                            results.append({
                                "file": str(filepath),
                                "line": i,
                                "text": line.strip()[:200]
                            })
                            if len(results) >= 20:
                                break
                except:
                    pass

                if len(results) >= 20:
                    break

            return {
                "success": True,
                "matches": results,
                "count": len(results),
                "pattern": pattern
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


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
