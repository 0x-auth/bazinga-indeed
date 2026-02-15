"""
BAZINGA Agent - Free local Claude Code alternative.

"The first AI you actually own"

Usage:
    bazinga --agent              # Start agent shell
    bazinga --agent "do X"       # One-shot task
"""

from .tools import Tool, ReadTool, BashTool, SearchTool, EditTool
from .loop import AgentLoop
from .shell import AgentShell

__all__ = [
    'Tool', 'ReadTool', 'BashTool', 'SearchTool', 'EditTool',
    'AgentLoop', 'AgentShell'
]
