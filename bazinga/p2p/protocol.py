#!/usr/bin/env python3
"""
BAZINGA P2P Protocol - Message Handlers and Network Logic
==========================================================

Implements the BAZINGA P2P protocol:
- PoB Exchange: Nodes prove their φ⁴ boundary on connection
- Query Routing: Questions routed through trust-weighted paths
- Knowledge Sync: α-SEED based knowledge sharing
- Consensus: Triadic validation of important operations

Message Types:
  HELLO       → Initial handshake with node info
  POB         → Proof-of-Boundary submission
  POB_VERIFY  → PoB verification result
  QUERY       → Network query
  RESPONSE    → Query response
  KNOWLEDGE   → Knowledge package (α-SEED)
  PEERS       → Peer list exchange
  CONSENSUS   → Triadic consensus request/vote

"Intelligence distributed, not controlled."
"""

import asyncio
import hashlib
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .transport import BazingaTransport, Message, Peer, create_transport, ZMQ_AVAILABLE

# Import Darmiyan for PoB
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bazinga.darmiyan import (
    prove_boundary, DarmiyanNode, BoundaryProof,
    PHI_4, ABHI_AMU,
)
from bazinga.darmiyan.constants import POB_TOLERANCE

# Protocol constants
PROTOCOL_VERSION = "1.0"
POB_REFRESH_INTERVAL = 300  # 5 minutes
PEER_EXCHANGE_INTERVAL = 60  # 1 minute
QUERY_TIMEOUT = 10.0  # seconds


@dataclass
class NetworkNode:
    """A node in the BAZINGA network with full protocol support."""
    peer: Peer
    pob_proof: Optional[BoundaryProof] = None
    pob_verified: bool = False
    trust_score: float = 0.5
    knowledge_domains: List[str] = field(default_factory=list)
    last_pob: Optional[datetime] = None
    query_responses: int = 0
    query_quality: float = 0.5


class BazingaProtocol:
    """
    BAZINGA P2P Protocol Handler.

    Manages:
    - PoB-based authentication
    - Trust-weighted routing
    - Knowledge synchronization
    - Triadic consensus

    Usage:
        protocol = BazingaProtocol(node_id="my_node", port=5150)
        await protocol.start()
        await protocol.connect("friend_ip", 5150)
        results = await protocol.query("What is consciousness?")
        await protocol.sync_knowledge()
        await protocol.stop()
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 5150,
    ):
        # Generate node ID if not provided
        if node_id is None:
            node_id = f"node_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}"

        self.node_id = node_id
        self.host = host
        self.port = port

        # Darmiyan node for PoB
        self.darmiyan = DarmiyanNode(node_id=node_id)

        # Transport layer
        self.transport = create_transport(
            node_id=node_id,
            host=host,
            port=port,
            pub_port=port + 1,
        )

        # Network state
        self.nodes: Dict[str, NetworkNode] = {}
        self.my_pob: Optional[BoundaryProof] = None
        self.pob_valid = False

        # Knowledge base reference (set externally)
        self.kb = None

        # Query handling
        self.pending_queries: Dict[str, asyncio.Future] = {}
        self.query_results: Dict[str, List[Dict]] = {}

        # Callbacks
        self.on_query: Optional[Callable] = None
        self.on_knowledge: Optional[Callable] = None
        self.on_node_joined: Optional[Callable] = None
        self.on_consensus: Optional[Callable] = None

        # Background tasks
        self.tasks: List[asyncio.Task] = []

        # Stats
        self.stats = {
            'pob_generated': 0,
            'pob_verified': 0,
            'pob_rejected': 0,
            'queries_sent': 0,
            'queries_received': 0,
            'queries_answered': 0,
            'knowledge_shared': 0,
            'knowledge_received': 0,
            'consensus_participated': 0,
        }

    async def start(self) -> bool:
        """Start the protocol and transport."""
        print(f"\n🌐 Starting BAZINGA Protocol...")
        print(f"   Node ID: {self.node_id}")

        # Generate initial PoB
        print(f"   Generating Proof-of-Boundary...")
        self.my_pob = prove_boundary()
        self.pob_valid = self.my_pob.valid
        self.stats['pob_generated'] += 1

        if self.pob_valid:
            print(f"   ✓ PoB valid (ratio: {self.my_pob.ratio:.4f}, attempts: {self.my_pob.attempts})")
        else:
            print(f"   ⚠ PoB invalid, will retry...")

        # Start transport
        success = await self.transport.start()
        if not success:
            return False

        # Register message handlers
        self._register_handlers()

        # Set transport callbacks
        self.transport.on_peer_connected = self._on_peer_connected
        self.transport.on_message = self._on_message

        # Start background tasks
        self.tasks.append(asyncio.create_task(self._pob_refresh_loop()))
        self.tasks.append(asyncio.create_task(self._peer_exchange_loop()))

        print(f"   ✓ Protocol started on port {self.port}")
        return True

    async def stop(self):
        """Stop the protocol."""
        print(f"\n🛑 Stopping BAZINGA Protocol...")

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Stop transport
        await self.transport.stop()

        print(f"   ✓ Protocol stopped")

    def _register_handlers(self):
        """Register protocol message handlers."""
        self.transport.register_handler("POB", self._handle_pob)
        self.transport.register_handler("POB_VERIFY", self._handle_pob_verify)
        self.transport.register_handler("QUERY", self._handle_query)
        self.transport.register_handler("QUERY_RESPONSE", self._handle_query_response)
        self.transport.register_handler("KNOWLEDGE", self._handle_knowledge)
        self.transport.register_handler("PEERS", self._handle_peers)
        self.transport.register_handler("CONSENSUS_REQ", self._handle_consensus_request)

    async def _on_peer_connected(self, peer: Peer):
        """Called when a new peer connects."""
        # Create network node
        node = NetworkNode(peer=peer)
        self.nodes[peer.peer_id] = node

        # Send our PoB proof
        if self.my_pob and self.pob_valid:
            pob_msg = Message(
                msg_type="POB",
                payload={
                    'alpha': self.my_pob.alpha,
                    'omega': self.my_pob.omega,
                    'delta': self.my_pob.delta,
                    'physical_ms': self.my_pob.physical_ms,
                    'geometric': self.my_pob.geometric,
                    'ratio': self.my_pob.ratio,
                    'timestamp': self.my_pob.timestamp,
                    'attempts': self.my_pob.attempts,
                },
                sender_id=self.node_id,
            )
            await self.transport.send(peer.peer_id, pob_msg)

        if self.on_node_joined:
            await self.on_node_joined(node)

    async def _on_message(self, sender_id: str, message: Message):
        """General message callback for logging."""
        pass  # Handlers do the work

    async def _handle_pob(self, sender_id: str, message: Message):
        """Handle incoming Proof-of-Boundary."""
        payload = message.payload

        # Verify the proof
        ratio = payload.get('ratio', 0)
        is_valid = abs(ratio - PHI_4) < POB_TOLERANCE

        if sender_id in self.nodes:
            node = self.nodes[sender_id]

            if is_valid:
                # Create proof object
                node.pob_proof = BoundaryProof(
                    alpha=payload.get('alpha', 0),
                    omega=payload.get('omega', 0),
                    delta=payload.get('delta', 0),
                    physical_ms=payload.get('physical_ms', 0),
                    geometric=payload.get('geometric', 0),
                    ratio=ratio,
                    timestamp=payload.get('timestamp', 0),
                    node_id=sender_id,
                    valid=True,
                    attempts=payload.get('attempts', 0),
                )
                node.pob_verified = True
                node.trust_score = min(1.0, node.trust_score + 0.1)
                node.last_pob = datetime.now()
                self.stats['pob_verified'] += 1

                print(f"   ✓ PoB verified for {sender_id[:12]} (ratio: {ratio:.4f})")
            else:
                node.pob_verified = False
                node.trust_score = max(0.0, node.trust_score - 0.2)
                self.stats['pob_rejected'] += 1

                print(f"   ✗ PoB rejected for {sender_id[:12]} (ratio: {ratio:.4f})")

        # Send verification result
        verify_msg = Message(
            msg_type="POB_VERIFY",
            payload={
                'valid': is_valid,
                'sender_ratio': ratio,
                'expected_ratio': PHI_4,
                'tolerance': POB_TOLERANCE,
            },
            sender_id=self.node_id,
        )
        await self.transport.send(sender_id, verify_msg)

    async def _handle_pob_verify(self, sender_id: str, message: Message):
        """Handle PoB verification response."""
        valid = message.payload.get('valid', False)
        if valid:
            print(f"   ✓ Our PoB verified by {sender_id[:12]}")
        else:
            print(f"   ⚠ Our PoB rejected by {sender_id[:12]}")

    async def _handle_query(self, sender_id: str, message: Message):
        """Handle incoming query."""
        self.stats['queries_received'] += 1
        query_id = message.payload.get('query_id', '')
        query_text = message.payload.get('query', '')

        print(f"   📨 Query from {sender_id[:12]}: {query_text[:50]}...")

        # Process query locally
        results = []
        if self.kb and query_text:
            try:
                search_results = self.kb.search(query_text, limit=5)
                results = [
                    {
                        'content': r.chunk.content[:500],
                        'source': r.chunk.source_file,
                        'similarity': r.similarity,
                    }
                    for r in search_results
                ]
            except Exception as e:
                print(f"   Query search error: {e}")

        # Custom handler
        if self.on_query:
            custom_results = await self.on_query(sender_id, query_text)
            if custom_results:
                results.extend(custom_results)

        # Send response
        response = Message(
            msg_type="QUERY_RESPONSE",
            payload={
                'query_id': query_id,
                'results': results,
                'node_id': self.node_id,
                'trust': self.nodes.get(sender_id, NetworkNode(Peer("", "", 0))).trust_score,
            },
            sender_id=self.node_id,
        )
        await self.transport.send(sender_id, response)
        self.stats['queries_answered'] += 1

    async def _handle_query_response(self, sender_id: str, message: Message):
        """Handle query response."""
        query_id = message.payload.get('query_id', '')
        results = message.payload.get('results', [])

        if query_id in self.pending_queries:
            # Store results
            if query_id not in self.query_results:
                self.query_results[query_id] = []
            self.query_results[query_id].extend(results)

            # Update node stats
            if sender_id in self.nodes:
                self.nodes[sender_id].query_responses += 1

    async def _handle_knowledge(self, sender_id: str, message: Message):
        """Handle incoming knowledge package."""
        self.stats['knowledge_received'] += 1
        payload = message.payload

        print(f"   📚 Knowledge from {sender_id[:12]}")

        if self.on_knowledge:
            await self.on_knowledge(sender_id, payload)

    async def _handle_peers(self, sender_id: str, message: Message):
        """Handle peer list exchange."""
        peers = message.payload.get('peers', [])

        for peer_info in peers:
            peer_id = peer_info.get('id')
            if peer_id and peer_id != self.node_id and peer_id not in self.nodes:
                # New peer discovered
                address = peer_info.get('address')
                port = peer_info.get('port')
                if address and port:
                    print(f"   📡 Discovered peer: {peer_id[:12]} at {address}:{port}")
                    # Could auto-connect here

    async def _handle_consensus_request(self, sender_id: str, message: Message):
        """Handle consensus request (triadic validation)."""
        self.stats['consensus_participated'] += 1
        payload = message.payload

        # For triadic consensus, we need to verify the proposal
        proposal = payload.get('proposal', {})

        # Generate our PoB vote
        vote_pob = prove_boundary()

        vote = Message(
            msg_type="CONSENSUS_VOTE",
            payload={
                'proposal_id': payload.get('proposal_id'),
                'vote': vote_pob.valid,
                'pob_ratio': vote_pob.ratio,
                'node_id': self.node_id,
            },
            sender_id=self.node_id,
        )
        await self.transport.send(sender_id, vote)

        if self.on_consensus:
            await self.on_consensus(sender_id, proposal, vote_pob.valid)

    async def _pob_refresh_loop(self):
        """Periodically refresh PoB and broadcast to peers."""
        while True:
            try:
                await asyncio.sleep(POB_REFRESH_INTERVAL)

                # Generate new PoB
                self.my_pob = prove_boundary()
                self.pob_valid = self.my_pob.valid
                self.stats['pob_generated'] += 1

                if self.pob_valid:
                    # Broadcast to all peers
                    pob_msg = Message(
                        msg_type="POB",
                        payload={
                            'alpha': self.my_pob.alpha,
                            'omega': self.my_pob.omega,
                            'delta': self.my_pob.delta,
                            'physical_ms': self.my_pob.physical_ms,
                            'geometric': self.my_pob.geometric,
                            'ratio': self.my_pob.ratio,
                            'timestamp': self.my_pob.timestamp,
                        },
                        sender_id=self.node_id,
                    )
                    await self.transport.broadcast(pob_msg)
                    print(f"   ⟳ PoB refreshed and broadcast (ratio: {self.my_pob.ratio:.4f})")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"   PoB refresh error: {e}")

    async def _peer_exchange_loop(self):
        """Periodically exchange peer lists."""
        while True:
            try:
                await asyncio.sleep(PEER_EXCHANGE_INTERVAL)

                if self.nodes:
                    # Build peer list
                    peer_list = [
                        {
                            'id': node.peer.peer_id,
                            'address': node.peer.address,
                            'port': node.peer.port,
                            'trust': node.trust_score,
                        }
                        for node in self.nodes.values()
                        if node.pob_verified
                    ]

                    if peer_list:
                        peers_msg = Message(
                            msg_type="PEERS",
                            payload={'peers': peer_list},
                            sender_id=self.node_id,
                        )
                        await self.transport.broadcast(peers_msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"   Peer exchange error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    async def connect(self, address: str, port: int) -> bool:
        """Connect to a peer node."""
        peer = await self.transport.connect_to_peer(address, port)
        return peer is not None

    async def query(
        self,
        query_text: str,
        timeout: float = QUERY_TIMEOUT,
        min_trust: float = 0.3,
    ) -> List[Dict]:
        """
        Query the network.

        Args:
            query_text: The question to ask
            timeout: How long to wait for responses
            min_trust: Minimum trust score for responding nodes

        Returns:
            List of results from network
        """
        if not self.nodes:
            print("   No peers connected")
            return []

        query_id = hashlib.sha256(f"{query_text}{time.time()}".encode()).hexdigest()[:16]
        self.stats['queries_sent'] += 1

        # Send query to trusted nodes
        query_msg = Message(
            msg_type="QUERY",
            payload={
                'query_id': query_id,
                'query': query_text,
            },
            sender_id=self.node_id,
        )

        sent_to = 0
        for node in self.nodes.values():
            if node.trust_score >= min_trust and node.pob_verified:
                await self.transport.send(node.peer.peer_id, query_msg)
                sent_to += 1

        print(f"   📤 Query sent to {sent_to} nodes: {query_text[:50]}...")

        # Wait for responses
        await asyncio.sleep(timeout)

        # Collect results
        results = self.query_results.pop(query_id, [])
        print(f"   📥 Received {len(results)} results")

        return results

    async def share_knowledge(self, content: str, metadata: Dict = None):
        """Share knowledge with the network."""
        if not self.nodes:
            return

        knowledge_msg = Message(
            msg_type="KNOWLEDGE",
            payload={
                'content': content[:10000],  # Limit size
                'metadata': metadata or {},
                'hash': hashlib.sha256(content.encode()).hexdigest()[:16],
            },
            sender_id=self.node_id,
        )

        await self.transport.broadcast(knowledge_msg)
        self.stats['knowledge_shared'] += 1
        print(f"   📚 Knowledge shared to network")

    async def request_consensus(self, proposal: Dict) -> bool:
        """
        Request triadic consensus on a proposal.

        Needs 3 nodes with valid PoB to agree.
        """
        verified_nodes = [n for n in self.nodes.values() if n.pob_verified]

        if len(verified_nodes) < 2:
            print(f"   Need at least 2 verified peers for consensus (have {len(verified_nodes)})")
            return False

        proposal_id = hashlib.sha256(str(proposal).encode()).hexdigest()[:16]

        consensus_msg = Message(
            msg_type="CONSENSUS_REQ",
            payload={
                'proposal_id': proposal_id,
                'proposal': proposal,
            },
            sender_id=self.node_id,
        )

        for node in verified_nodes[:2]:  # Ask 2 peers (we are the 3rd)
            await self.transport.send(node.peer.peer_id, consensus_msg)

        self.stats['consensus_participated'] += 1
        return True

    def get_peers(self) -> List[Dict]:
        """Get list of connected peers with their status."""
        return [
            {
                'id': node.peer.peer_id,
                'address': node.peer.address,
                'port': node.peer.port,
                'trust': node.trust_score,
                'pob_verified': node.pob_verified,
                'last_pob': node.last_pob.isoformat() if node.last_pob else None,
            }
            for node in self.nodes.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        transport_stats = self.transport.get_stats()
        return {
            'node_id': self.node_id,
            'port': self.port,
            'pob_valid': self.pob_valid,
            'pob_ratio': self.my_pob.ratio if self.my_pob else 0,
            'peers': len(self.nodes),
            'verified_peers': sum(1 for n in self.nodes.values() if n.pob_verified),
            **self.stats,
            'transport': transport_stats,
        }

    def print_status(self):
        """Print network status."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("  BAZINGA Network Status")
        print("=" * 60)
        print(f"  Node ID: {stats['node_id']}")
        print(f"  Port: {stats['port']}")
        print(f"  PoB Valid: {'✓' if stats['pob_valid'] else '✗'} (ratio: {stats['pob_ratio']:.4f})")
        print(f"  Peers: {stats['peers']} ({stats['verified_peers']} verified)")
        print()
        print(f"  PoB: {stats['pob_generated']} generated, {stats['pob_verified']} verified, {stats['pob_rejected']} rejected")
        print(f"  Queries: {stats['queries_sent']} sent, {stats['queries_received']} received, {stats['queries_answered']} answered")
        print(f"  Knowledge: {stats['knowledge_shared']} shared, {stats['knowledge_received']} received")
        print(f"  Consensus: {stats['consensus_participated']} participated")
        print("=" * 60)


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA Protocol Test")
        print("=" * 60)

        if not ZMQ_AVAILABLE:
            print("\n⚠ ZeroMQ not installed. Install with:")
            print("  pip install pyzmq")
            return

        # Create two protocol instances
        node1 = BazingaProtocol(node_id="node_alpha", port=5150)
        node2 = BazingaProtocol(node_id="node_beta", port=5160)

        # Start both
        await node1.start()
        await node2.start()

        # Connect
        print("\n  Connecting nodes...")
        success = await node2.connect("127.0.0.1", 5150)

        if success:
            # Wait for PoB exchange
            await asyncio.sleep(2)

            # Show peers
            print(f"\n  Node1 peers: {node1.get_peers()}")
            print(f"  Node2 peers: {node2.get_peers()}")

            # Query test
            print("\n  Testing query...")
            results = await node2.query("What is consciousness?", timeout=3)
            print(f"  Results: {results}")

            # Show status
            node1.print_status()
            node2.print_status()

        # Stop
        await node1.stop()
        await node2.stop()

        print("\n  ✓ Protocol test complete!")

    asyncio.run(test())
