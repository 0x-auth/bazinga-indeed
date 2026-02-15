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
    """Execute shell commands."""

    name = "bash"
    description = "Run a bash command. Args: command (str). Use for git, npm, python, etc."

    # Commands that are blocked for safety
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        ":(){:|:&};:",  # Fork bomb
        "mkfs",
        "> /dev/sda",
        "dd if=/dev/zero",
    ]

    def execute(self, command: str, timeout: int = 30, **kwargs) -> Dict[str, Any]:
        """Execute a bash command."""
        try:
            # Basic safety check
            for blocked in self.BLOCKED_PATTERNS:
                if blocked in command:
                    return {
                        "success": False,
                        "error": f"Blocked dangerous command pattern: {blocked}"
                    }

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
    """Edit file contents."""

    name = "edit"
    description = "Edit a file by replacing text. Args: path (str), old_text (str), new_text (str)"

    def execute(self, path: str, old_text: str, new_text: str, **kwargs) -> Dict[str, Any]:
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

            # Apply edit
            new_content = content.replace(old_text, new_text)
            filepath.write_text(new_content)

            return {
                "success": True,
                "message": f"Edited {path}",
                "path": str(filepath)
            }

        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class WriteTool(Tool):
    """Write/create a file."""

    name = "write"
    description = "Write content to a file (creates if doesn't exist). Args: path (str), content (str)"

    def execute(self, path: str, content: str, **kwargs) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            filepath = Path(path).expanduser()

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


# Registry of all available tools
TOOLS = {
    "read": ReadTool(),
    "bash": BashTool(),
    "search": SearchTool(),
    "edit": EditTool(),
    "write": WriteTool(),
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
