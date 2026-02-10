"""
BAZINGA P2P - Distributed Consciousness Network
==============================================

Full P2P network with real transport layer.

Features:
- ZeroMQ Transport: Real network connections between nodes
- PoB Authentication: Nodes prove φ⁴ boundary before joining
- Trust Routing: High-trust nodes route queries
- Knowledge Sync: α-SEED based knowledge sharing
- Triadic Consensus: 3 nodes must agree for validation

"Intelligence distributed, not controlled."
"""

from .node import BAZINGANode
from .dht import BAZINGA_DHT
from .knowledge_sync import KnowledgeGraphSync
from .trust_router import TrustRouter
from .alpha_seed import AlphaSeedNetwork, is_alpha_seed
from .network import BAZINGANetwork, create_network

# New transport layer
from .transport import (
    BazingaTransport,
    create_transport,
    Message,
    Peer,
    ZMQ_AVAILABLE,
)

# New protocol layer
from .protocol import BazingaProtocol

__all__ = [
    # Unified API (recommended)
    'BAZINGANetwork',
    'create_network',
    # Real P2P (NEW!)
    'BazingaProtocol',
    'BazingaTransport',
    'create_transport',
    'Message',
    'Peer',
    'ZMQ_AVAILABLE',
    # Low-level components
    'BAZINGANode',
    'BAZINGA_DHT',
    'KnowledgeGraphSync',
    'TrustRouter',
    'AlphaSeedNetwork',
    'is_alpha_seed',
]
