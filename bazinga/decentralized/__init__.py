"""
BAZINGA Decentralized - Phase 4: Full Autonomy

Complete independence from centralized services:
- Bootstrap-free peer discovery
- Model distribution via P2P
- Consensus and governance (DAO-style)
- Economic incentives for contributors

"Owned by everyone, controlled by no one."
"""

from .peer_discovery import PeerDiscovery, BootstrapFreeDiscovery, PeerInfo
from .model_distribution import ModelDistribution, ModelChunk, DistributionProtocol
from .consensus import ConsensusEngine, Proposal, Vote, DAOGovernance

__all__ = [
    'PeerDiscovery',
    'BootstrapFreeDiscovery',
    'PeerInfo',
    'ModelDistribution',
    'ModelChunk',
    'DistributionProtocol',
    'ConsensusEngine',
    'Proposal',
    'Vote',
    'DAOGovernance',
]
