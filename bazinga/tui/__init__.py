"""
BAZINGA TUI - Full-screen Terminal User Interface

Like Claude Code, but for BAZINGA.
Takes over terminal for interactive modes:
  - bazinga (no args)
  - bazinga --chat
  - bazinga --agent

All other flags work normally without TUI.
"""

from .app import BazingaApp, ChatInput, run_tui, run_tui_async

__all__ = ['BazingaApp', 'ChatInput', 'run_tui', 'run_tui_async']
