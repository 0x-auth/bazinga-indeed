"""
BAZINGA Agent - Free local Claude Code alternative.

"The first AI you actually own"

Usage:
    bazinga --agent              # Start agent shell
    bazinga --agent "do X"       # One-shot task

Tools:
    read   - Read file contents
    bash   - Run shell commands
    search - RAG search indexed knowledge
    edit   - Edit files (find & replace)
    write  - Write/create files
    glob   - Find files by pattern
    grep   - Search text in files
"""

from .tools import Tool, ReadTool, BashTool, SearchTool, EditTool, WriteTool, GlobTool, GrepTool
from .loop import AgentLoop
from .shell import AgentShell, SessionMemory
from .context import ProjectContext, ProjectDetector, get_project_context

__all__ = [
    'Tool', 'ReadTool', 'BashTool', 'SearchTool', 'EditTool', 'WriteTool', 'GlobTool', 'GrepTool',
    'AgentLoop', 'AgentShell', 'SessionMemory',
    'ProjectContext', 'ProjectDetector', 'get_project_context'
]
