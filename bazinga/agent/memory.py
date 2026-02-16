"""
BAZINGA Agent Persistent Memory - Remembers across sessions.

"The first AI you actually own"

Stores:
- Recent interactions per project
- Files you've worked on
- Commands you've run
- What you were working on last

All stored locally in ~/.bazinga/agent/
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import hashlib


def get_project_id(root: Path = None) -> str:
    """Generate a unique ID for a project based on its path.

    SECURITY: Returns alphanumeric ID only (safe for filenames).
    """
    root = root or Path.cwd()
    # Use hash of absolute path for unique ID
    path_str = str(root.resolve())
    project_id = hashlib.md5(path_str.encode()).hexdigest()[:12]
    # Ensure ID is safe for filenames (alphanumeric only)
    if not project_id.isalnum():
        project_id = ''.join(c for c in project_id if c.isalnum())[:12]
    return project_id or "default"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    timestamp: str
    user_input: str
    agent_response: str
    tools_used: List[str] = field(default_factory=list)
    files_touched: List[str] = field(default_factory=list)


@dataclass
class ProjectMemory:
    """Memory for a specific project."""
    project_id: str
    project_path: str
    project_name: str

    # Recent interactions (last 50)
    interactions: List[Dict] = field(default_factory=list)

    # Files we've worked with (for quick context)
    recent_files: List[str] = field(default_factory=list)

    # What user was working on (natural language summary)
    last_task: str = ""

    # When we last worked on this project
    last_accessed: str = ""

    # Total interactions count
    total_interactions: int = 0


class PersistentMemory:
    """
    Persistent memory that survives across sessions.

    Storage: ~/.bazinga/agent/memory/
    - projects.json: Index of all projects
    - {project_id}.json: Memory for each project
    """

    MAX_INTERACTIONS = 50
    MAX_RECENT_FILES = 20

    def __init__(self):
        self.memory_dir = Path.home() / ".bazinga" / "agent" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.projects_file = self.memory_dir / "projects.json"
        self.projects_index: Dict[str, Dict] = {}

        self._load_projects_index()

    def _load_projects_index(self):
        """Load the projects index with validation."""
        if self.projects_file.exists():
            try:
                data = json.loads(self.projects_file.read_text())
                # Validate structure
                if isinstance(data, dict):
                    self.projects_index = data
                else:
                    self.projects_index = {}
            except (json.JSONDecodeError, OSError, IOError) as e:
                # Log error for debugging but continue
                import sys
                print(f"Warning: Could not load projects index: {e}", file=sys.stderr)
                self.projects_index = {}

    def _save_projects_index(self):
        """Save the projects index."""
        self.projects_file.write_text(json.dumps(self.projects_index, indent=2))

    def _get_memory_file(self, project_id: str) -> Path:
        """Get the memory file path for a project.

        SECURITY: Validates project_id to prevent path injection.
        """
        # Validate project_id is safe (alphanumeric only)
        safe_id = ''.join(c for c in project_id if c.isalnum())[:20]
        if not safe_id:
            safe_id = "default"
        return self.memory_dir / f"{safe_id}.json"

    def load_project_memory(self, root: Path = None) -> ProjectMemory:
        """Load memory for the current project."""
        root = root or Path.cwd()
        project_id = get_project_id(root)
        memory_file = self._get_memory_file(project_id)

        if memory_file.exists():
            try:
                data = json.loads(memory_file.read_text())
                # Validate structure before creating ProjectMemory
                if isinstance(data, dict) and 'project_id' in data:
                    return ProjectMemory(**data)
            except (json.JSONDecodeError, TypeError, OSError, IOError) as e:
                import sys
                print(f"Warning: Could not load project memory: {e}", file=sys.stderr)

        # Create new memory for this project
        # Try to get project name
        project_name = root.name
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                for line in pyproject.read_text().split('\n'):
                    if line.startswith('name = '):
                        project_name = line.split('=')[1].strip().strip('"\'')
                        break
            except (OSError, IOError, UnicodeDecodeError):
                pass  # Use default project_name

        return ProjectMemory(
            project_id=project_id,
            project_path=str(root),
            project_name=project_name,
            last_accessed=datetime.now().isoformat()
        )

    def save_project_memory(self, memory: ProjectMemory):
        """Save memory for a project."""
        memory.last_accessed = datetime.now().isoformat()
        memory_file = self._get_memory_file(memory.project_id)

        # Convert to dict
        data = {
            'project_id': memory.project_id,
            'project_path': memory.project_path,
            'project_name': memory.project_name,
            'interactions': memory.interactions[-self.MAX_INTERACTIONS:],
            'recent_files': memory.recent_files[-self.MAX_RECENT_FILES:],
            'last_task': memory.last_task,
            'last_accessed': memory.last_accessed,
            'total_interactions': memory.total_interactions
        }

        memory_file.write_text(json.dumps(data, indent=2))

        # Update projects index
        self.projects_index[memory.project_id] = {
            'path': memory.project_path,
            'name': memory.project_name,
            'last_accessed': memory.last_accessed,
            'total_interactions': memory.total_interactions
        }
        self._save_projects_index()

    def add_interaction(self, memory: ProjectMemory, user_input: str,
                       agent_response: str, tools_used: List[str] = None,
                       files_touched: List[str] = None):
        """Add an interaction to project memory."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input[:500],  # Truncate
            'agent': agent_response[:500],
            'tools': tools_used or [],
            'files': files_touched or []
        }

        memory.interactions.append(interaction)
        memory.total_interactions += 1

        # Update recent files
        for f in (files_touched or []):
            if f not in memory.recent_files:
                memory.recent_files.append(f)

        # Keep only recent
        memory.interactions = memory.interactions[-self.MAX_INTERACTIONS:]
        memory.recent_files = memory.recent_files[-self.MAX_RECENT_FILES:]

        # Update last task (simple heuristic: last user input)
        memory.last_task = user_input[:200]

        self.save_project_memory(memory)

    def get_context_summary(self, memory: ProjectMemory) -> str:
        """Get a summary of persistent memory for context."""
        if not memory.interactions:
            return ""

        lines = ["## Previous Session Memory:"]

        # When we last worked on this
        if memory.last_accessed:
            try:
                last = datetime.fromisoformat(memory.last_accessed)
                delta = datetime.now() - last
                if delta.days > 0:
                    lines.append(f"Last session: {delta.days} days ago")
                elif delta.seconds > 3600:
                    lines.append(f"Last session: {delta.seconds // 3600} hours ago")
                else:
                    lines.append(f"Last session: {delta.seconds // 60} minutes ago")
            except:
                pass

        # What we were working on
        if memory.last_task:
            lines.append(f"Last task: {memory.last_task}")

        # Recent files
        if memory.recent_files:
            lines.append(f"Recent files: {', '.join(memory.recent_files[-5:])}")

        # Last few interactions
        if memory.interactions:
            lines.append("\nRecent history:")
            for interaction in memory.interactions[-3:]:
                lines.append(f"  - You: {interaction['user'][:80]}...")

        lines.append(f"\nTotal interactions in this project: {memory.total_interactions}")

        return "\n".join(lines)

    def list_projects(self) -> List[Dict]:
        """List all projects with memory."""
        projects = []
        for project_id, info in self.projects_index.items():
            projects.append({
                'id': project_id,
                'name': info.get('name', 'Unknown'),
                'path': info.get('path', ''),
                'last_accessed': info.get('last_accessed', ''),
                'interactions': info.get('total_interactions', 0)
            })

        # Sort by last accessed
        projects.sort(key=lambda x: x.get('last_accessed', ''), reverse=True)
        return projects

    def clear_project_memory(self, project_id: str):
        """Clear memory for a specific project."""
        memory_file = self._get_memory_file(project_id)
        if memory_file.exists():
            memory_file.unlink()

        if project_id in self.projects_index:
            del self.projects_index[project_id]
            self._save_projects_index()


# Global instance
_persistent_memory = None

def get_persistent_memory() -> PersistentMemory:
    """Get the global persistent memory instance."""
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemory()
    return _persistent_memory
