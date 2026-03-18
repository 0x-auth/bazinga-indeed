"""
BAZINGA Knowledge Scanner (KB DNA)

Semantic compression for your entire knowledge base.
Generates manifests that ANY AI can understand without reading full files.

Usage:
    bazinga --scan ~/Documents ~/Projects
    bazinga --scan-status
"""

from .scanner import KBScanner, get_scanner

__all__ = ['KBScanner', 'get_scanner']
