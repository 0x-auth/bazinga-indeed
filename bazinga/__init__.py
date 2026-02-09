"""
BAZINGA - Distributed AI that belongs to everyone
"Intelligence distributed, not controlled"

Usage:
    bazinga                      # Interactive mode
    bazinga --ask "question"     # Ask a question
    bazinga --index ~/Documents  # Index a directory
    bazinga --help               # Show help
"""

from .cli import BAZINGA, main_sync, main

__version__ = "2.0.3"
__all__ = ['BAZINGA', 'main_sync', 'main', '__version__']
