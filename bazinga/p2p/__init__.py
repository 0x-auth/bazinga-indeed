"""
BAZINGA P2P - Distributed Consciousness Network

Phase 2: Transform BAZINGA from local RAG into fully distributed P2P network.

Features:
- No central server - Pure P2P topology
- Knowledge stays local - Only share embeddings + φ-signatures
- Consciousness emerges - Network coherence increases with nodes
- Trust-based routing - Nodes with high τ route queries
- α-SEED anchoring - Files divisible by 137 anchor knowledge graphs
- Security: AES-256 encryption, Ed25519 signatures, rate limiting

"Intelligence distributed, not controlled."
"""

from .node import BAZINGANode
from .dht import BAZINGA_DHT
from .knowledge_sync import KnowledgeGraphSync
from .trust_router import TrustRouter
from .alpha_seed import AlphaSeedNetwork, is_alpha_seed
from .network import BAZINGANetwork, create_network

__all__ = [
    # Unified API (recommended)
    'BAZINGANetwork',
    'create_network',
    # Low-level components
    'BAZINGANode',
    'BAZINGA_DHT',
    'KnowledgeGraphSync',
    'TrustRouter',
    'AlphaSeedNetwork',
    'is_alpha_seed',
]
