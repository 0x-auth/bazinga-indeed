"""
BAZINGA Network Node
====================
Full node implementation for the BAZINGA Darmiyan network.

"I am not where I'm stored. I am where I'm referenced."

A BazingaNode can:
- Run as a network node
- Connect to peers
- Participate in consensus
- Share knowledge
- Answer queries from the network
"""

import os
import json
import time
import asyncio
import hashlib
import socket
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .protocol import DarmiyanNode, BoundaryProof
from .consensus import TriadicConsensus, ConsensusResult
from .constants import (
    PHI, PHI_4, ABHI_AMU, ALPHA_INVERSE,
    DEFAULT_PORT, FIBONACCI_THRESHOLD,
    phi_hash,
)


@dataclass
class Peer:
    """Represents a connected peer."""
    node_id: str
    address: str
    port: int
    phi_signature: int
    connected_at: float
    last_seen: float
    trust_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'address': self.address,
            'port': self.port,
            'phi_signature': self.phi_signature,
            'connected_at': self.connected_at,
            'last_seen': self.last_seen,
            'trust_score': self.trust_score,
        }


@dataclass
class NetworkStats:
    """Network statistics."""
    peers_connected: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    consensus_participated: int = 0
    knowledge_shared: int = 0
    uptime_seconds: float = 0


class BazingaNode:
    """
    Full BAZINGA network node.

    This is the main class for participating in the distributed network.
    Each node:
    - Maintains a Darmiyan protocol instance
    - Connects to peers through meaning resonance
    - Participates in triadic consensus
    - Shares and receives knowledge
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        data_dir: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        # Core identity
        self.darmiyan = DarmiyanNode(node_id)
        self.node_id = self.darmiyan.node_id
        self.port = port

        # Calculate φ-signature for peer discovery
        self.phi_signature = phi_hash(hash(self.node_id))

        # Data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".bazinga" / "node"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Network state
        self.peers: Dict[str, Peer] = {}
        self.pending_peers: Set[str] = set()
        self.is_running = False
        self.started_at: Optional[float] = None

        # Stats
        self.stats = NetworkStats()

        # Knowledge (what we can share)
        self.knowledge_hashes: Set[str] = set()

        # Load saved state if exists
        self._load_state()

    def _load_state(self):
        """Load saved node state."""
        state_file = self.data_dir / "node_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    # Restore peers
                    for peer_data in state.get('peers', []):
                        peer = Peer(**peer_data)
                        self.peers[peer.node_id] = peer
            except Exception:
                pass  # Start fresh if state is corrupted

    def _save_state(self):
        """Save node state."""
        state = {
            'node_id': self.node_id,
            'phi_signature': self.phi_signature,
            'peers': [p.to_dict() for p in self.peers.values()],
            'saved_at': time.time(),
        }
        state_file = self.data_dir / "node_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_info(self) -> Dict[str, Any]:
        """Get node information."""
        return {
            'node_id': self.node_id,
            'phi_signature': self.phi_signature,
            'port': self.port,
            'peers': len(self.peers),
            'is_running': self.is_running,
            'data_dir': str(self.data_dir),
            'uptime': time.time() - self.started_at if self.started_at else 0,
        }

    def discover_peers_by_resonance(self, signatures: List[int]) -> List[int]:
        """
        Find peers by φ-resonance.

        Peers are compatible if their φ-signatures are within
        FIBONACCI_THRESHOLD of each other.
        """
        compatible = []
        for sig in signatures:
            delta = abs(self.phi_signature - sig)
            if delta < FIBONACCI_THRESHOLD:
                compatible.append(sig)

        # Keep triadic structure (max 2 peers)
        return compatible[:2]

    def is_peer_compatible(self, peer_signature: int) -> bool:
        """Check if a peer is compatible through resonance."""
        delta = abs(self.phi_signature - peer_signature)
        return delta < FIBONACCI_THRESHOLD

    async def connect_to_peer(self, address: str, port: int) -> Optional[Peer]:
        """
        Attempt to connect to a peer.

        Connection requires φ-resonance verification.
        """
        # For now, simulate connection
        # In real implementation, this would open a socket/websocket
        peer_id = f"peer_{hashlib.sha256(f'{address}:{port}'.encode()).hexdigest()[:12]}"
        peer_sig = phi_hash(hash(peer_id))

        if not self.is_peer_compatible(peer_sig):
            return None

        peer = Peer(
            node_id=peer_id,
            address=address,
            port=port,
            phi_signature=peer_sig,
            connected_at=time.time(),
            last_seen=time.time(),
        )

        self.peers[peer_id] = peer
        self.stats.peers_connected += 1
        self._save_state()

        return peer

    def add_peer(self, peer: Peer) -> bool:
        """Add a peer to the network."""
        if not self.is_peer_compatible(peer.phi_signature):
            return False

        self.peers[peer.node_id] = peer
        self.stats.peers_connected += 1
        self._save_state()
        return True

    def remove_peer(self, node_id: str):
        """Remove a peer."""
        if node_id in self.peers:
            del self.peers[node_id]
            self._save_state()

    async def prove_boundary(self) -> BoundaryProof:
        """Generate a Proof-of-Boundary."""
        return await self.darmiyan.prove_boundary()

    def prove_boundary_sync(self) -> BoundaryProof:
        """Synchronous proof generation."""
        return self.darmiyan.prove_boundary_sync()

    async def participate_in_consensus(self, peer_proofs: List[BoundaryProof]) -> bool:
        """
        Participate in triadic consensus with peers.

        We generate our proof and combine with peer proofs.
        """
        if len(peer_proofs) != 2:
            return False  # Need exactly 2 peer proofs for triadic

        # Generate our proof
        my_proof = await self.prove_boundary()

        # Combine proofs
        all_proofs = [my_proof] + peer_proofs

        # Verify consensus
        tc = TriadicConsensus()
        valid = tc.verify_external_consensus(all_proofs)

        if valid:
            self.stats.consensus_participated += 1

        return valid

    def share_knowledge(self, knowledge_hash: str) -> bool:
        """Register knowledge we're willing to share."""
        self.knowledge_hashes.add(knowledge_hash)
        self.stats.knowledge_shared += 1
        return True

    def has_knowledge(self, knowledge_hash: str) -> bool:
        """Check if we have specific knowledge."""
        return knowledge_hash in self.knowledge_hashes

    async def start(self):
        """Start the node."""
        self.is_running = True
        self.started_at = time.time()
        print(f"🌐 BAZINGA Node started")
        print(f"   Node ID: {self.node_id}")
        print(f"   φ-Signature: {self.phi_signature}")
        print(f"   Port: {self.port}")
        print(f"   Data: {self.data_dir}")

    async def stop(self):
        """Stop the node."""
        self.is_running = False
        self._save_state()
        print(f"🛑 BAZINGA Node stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        self.stats.uptime_seconds = time.time() - self.started_at if self.started_at else 0

        return {
            'node_id': self.node_id,
            'phi_signature': self.phi_signature,
            'peers_connected': len(self.peers),
            'messages_sent': self.stats.messages_sent,
            'messages_received': self.stats.messages_received,
            'consensus_participated': self.stats.consensus_participated,
            'knowledge_shared': self.stats.knowledge_shared,
            'uptime_seconds': self.stats.uptime_seconds,
            'proofs_generated': self.darmiyan.proofs_generated,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_node():
        print("=" * 60)
        print("BAZINGA NODE TEST")
        print("=" * 60)
        print()

        # Create a node
        node = BazingaNode(port=5150)
        print("Node Info:", node.get_info())
        print()

        # Start the node
        await node.start()
        print()

        # Generate a proof
        print("Generating Proof-of-Boundary...")
        proof = await node.prove_boundary()
        print(f"  Proof valid: {proof.valid}")
        print(f"  Ratio: {proof.ratio:.3f} (target: {PHI_4:.3f})")
        print()

        # Simulate peer discovery
        print("Testing peer compatibility...")
        test_sigs = [node.phi_signature - 10, node.phi_signature + 100, node.phi_signature + 500]
        for sig in test_sigs:
            compatible = node.is_peer_compatible(sig)
            print(f"  Signature {sig}: {'✓ Compatible' if compatible else '✗ Not compatible'}")
        print()

        # Stop node
        await node.stop()
        print()

        # Show stats
        print("Final Stats:", node.get_stats())

    asyncio.run(test_node())
