"""
BAZINGA - Distributed AI that belongs to everyone
"Intelligence distributed, not controlled"

Usage:
    bazinga                      # Interactive mode
    bazinga --ask "question"     # Ask a question
    bazinga --code "task"        # Generate code
    bazinga --index ~/Documents  # Index a directory
    bazinga --help               # Show help
"""

from .cli import BAZINGA, main_sync, main

__version__ = "2.3.0"
__all__ = ['BAZINGA', 'main_sync', 'main', '__version__']

# Lazy imports for optional components
def get_intelligent_coder():
    """Get the IntelligentCoder (LLM-powered code generation)."""
    from .intelligent_coder import IntelligentCoder
    return IntelligentCoder()

def get_llm_orchestrator():
    """Get the LLM Orchestrator (multi-provider intelligence)."""
    from .llm_orchestrator import LLMOrchestrator
    return LLMOrchestrator()
