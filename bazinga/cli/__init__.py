"""
BAZINGA CLI Package
====================
Backward-compatible package structure.

All existing imports continue to work:
    from bazinga.cli import BAZINGA, main, main_sync
    from bazinga.cli import _get_real_ai
"""

from ._core import BAZINGA, main, main_sync, _get_real_ai

__all__ = ['BAZINGA', 'main', 'main_sync', '_get_real_ai']
