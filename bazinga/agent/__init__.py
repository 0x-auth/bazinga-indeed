"""
BAZINGA Agent - Free local AI coding assistant.

"The first AI you actually own"

Usage:
    bazinga --agent              # Start agent shell
    bazinga --agent "do X"       # One-shot task

Tools:
    read         - Read file contents
    bash         - Run shell commands
    search       - RAG search indexed knowledge
    edit         - Edit files (find & replace)
    write        - Write/create files
    glob         - Find files by pattern
    grep         - Search text in files
    verified_fix - Blockchain-verified code fixes (multi-AI consensus)

NEW in v4.9.7: Blockchain-Verified Code Fixes
    - Multiple AIs must agree before applying changes
    - φ-coherence measurement for fix quality
    - PoB attestation on blockchain for audit trail
    - "No single AI can mess up your code"
"""

from .tools import Tool, ReadTool, BashTool, SearchTool, EditTool, WriteTool, GlobTool, GrepTool
from .loop import AgentLoop
from .shell import AgentShell, SessionMemory
from .context import ProjectContext, ProjectDetector, get_project_context

# Blockchain-verified fixes
try:
    from .verified_fixes import (
        VerifiedFixEngine, CodeFixProposal, ConsensusVerdict,
        FixStatus, FixType, AIReview,
        verified_fix, verified_fix_sync
    )
    _VERIFIED_FIXES_AVAILABLE = True
except ImportError:
    _VERIFIED_FIXES_AVAILABLE = False

__all__ = [
    'Tool', 'ReadTool', 'BashTool', 'SearchTool', 'EditTool', 'WriteTool', 'GlobTool', 'GrepTool',
    'AgentLoop', 'AgentShell', 'SessionMemory',
    'ProjectContext', 'ProjectDetector', 'get_project_context',
    # Verified fixes
    'VerifiedFixEngine', 'CodeFixProposal', 'ConsensusVerdict',
    'FixStatus', 'FixType', 'AIReview',
    'verified_fix', 'verified_fix_sync',
]
