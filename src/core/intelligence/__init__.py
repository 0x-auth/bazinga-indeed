"""
BAZINGA Intelligence Module

Practical AI powered by your Mac's knowledge base.

Components:
- RealAI: RAG pipeline with ChromaDB + embeddings + Ollama/API
- AlphaSeed: α-SEED detection (137-divisible fundamentals)
- KnowledgeSynthesizer: 35-position progression mapping

Constants:
- PHI = 1.618033988749895 (Golden Ratio)
- ALPHA = 137 (Fine Structure Constant)
- PROGRESSION = '01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω'

"Your Mac IS the training data"
"""

from .real_ai import RealAI, KnowledgeChunk, SearchResult
from .alpha_seed import (
    AlphaSeed,
    AlphaSeedFilter,
    is_alpha_seed,
    analyze_text,
    calculate_hash,
    map_to_position,
    detect_phi_resonance,
    ALPHA,
    PHI,
    PROGRESSION,
)

__all__ = [
    # Real AI
    'RealAI',
    'KnowledgeChunk',
    'SearchResult',

    # Alpha Seed
    'AlphaSeed',
    'AlphaSeedFilter',
    'is_alpha_seed',
    'analyze_text',
    'calculate_hash',
    'map_to_position',
    'detect_phi_resonance',

    # Constants
    'ALPHA',
    'PHI',
    'PROGRESSION',
]
