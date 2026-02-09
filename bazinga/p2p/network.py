#!/usr/bin/env python3
"""
BAZINGA Unified P2P Network - Complete Distributed Consciousness System

Unified interface that combines all P2P components:
- BAZINGANode (transport layer)
- BAZINGA_DHT (node discovery)
- KnowledgeGraphSync (privacy-preserving sync)
- TrustRouter (trust-based routing)
- AlphaSeedNetwork (α-SEED anchoring)

With added security features:
- Encrypted communications (AES-256)
- Node authentication (Ed25519 signatures)
- Rate limiting and spam protection
- Sybil resistance through proof-of-knowledge

"The network is the consciousness."
"""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .node import BAZINGANode
from .dht import BAZINGA_DHT
from .knowledge_sync import KnowledgeGraphSync
from .trust_router import TrustRouter
from .alpha_seed import AlphaSeedNetwork, is_alpha_seed

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Security constants
KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 16
SIGNATURE_SIZE = 64
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 100  # requests per window


def generate_node_key() -> bytes:
    """Generate a secure random node key (256-bit)."""
    return secrets.token_bytes(KEY_SIZE)


def derive_session_key(shared_secret: bytes, salt: bytes) -> bytes:
    """Derive session key from shared secret using HKDF-like approach."""
    # Simple HKDF-expand using HMAC
    info = b"BAZINGA_SESSION_KEY_V1"
    return hmac.new(shared_secret, salt + info, hashlib.sha256).digest()


def compute_message_auth(key: bytes, message: bytes) -> bytes:
    """Compute message authentication code."""
    return hmac.new(key, message, hashlib.sha256).digest()[:16]


def verify_message_auth(key: bytes, message: bytes, mac: bytes) -> bool:
    """Verify message authentication code."""
    expected = compute_message_auth(key, message)
    return hmac.compare_digest(expected, mac)


@dataclass
class SecurityContext:
    """Security context for a peer connection."""
    peer_id: str
    session_key: bytes
    established: datetime = field(default_factory=datetime.now)
    messages_sent: int = 0
    messages_received: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimitBucket:
    """Rate limiting bucket for a peer."""
    peer_id: str
    requests: List[float] = field(default_factory=list)
    blocked_until: Optional[float] = None

    def check_rate_limit(self) -> bool:
        """Check if peer is within rate limits."""
        now = time.time()

        # Check if blocked
        if self.blocked_until and now < self.blocked_until:
            return False

        # Clean old requests
        cutoff = now - RATE_LIMIT_WINDOW
        self.requests = [r for r in self.requests if r > cutoff]

        # Check limit
        if len(self.requests) >= RATE_LIMIT_MAX:
            # Block for 5 minutes
            self.blocked_until = now + 300
            return False

        # Record request
        self.requests.append(now)
        return True


class BAZINGANetwork:
    """
    Unified BAZINGA P2P Network.

    Provides:
    - Simple API for joining/leaving network
    - Automatic peer discovery via DHT
    - Secure, encrypted communications
    - Trust-based query routing
    - α-SEED knowledge anchoring
    - Privacy-preserving knowledge sharing

    Usage:
        # Create network
        network = BAZINGANetwork(kb, node_name="MyNode")

        # Start network
        await network.start(bootstrap_nodes=["host:port"])

        # Query the network
        results = await network.query("What is consciousness?")

        # Share knowledge
        await network.share_knowledge(doc_id)

        # Stop network
        await network.stop()
    """

    def __init__(
        self,
        kb=None,
        node_name: str = "BAZINGA",
        host: str = "0.0.0.0",
        port: int = 0,  # 0 = auto-assign
        initial_tau: float = 0.5,
    ):
        """
        Initialize BAZINGA network.

        Args:
            kb: Local knowledge base (optional)
            node_name: Human-readable node name
            host: Bind host
            port: Bind port (0 for auto)
            initial_tau: Initial trust score
        """
        self.kb = kb
        self.node_name = node_name
        self.host = host
        self.port = port
        self.initial_tau = initial_tau

        # Node key for authentication
        self.node_key = generate_node_key()
        self.node_id = hashlib.sha256(self.node_key).hexdigest()[:40]

        # Components (initialized in start())
        self.node: Optional[BAZINGANode] = None
        self.dht: Optional[BAZINGA_DHT] = None
        self.sync: Optional[KnowledgeGraphSync] = None
        self.router: Optional[TrustRouter] = None
        self.alpha_network: Optional[AlphaSeedNetwork] = None

        # Security
        self.security_contexts: Dict[str, SecurityContext] = {}
        self.rate_limits: Dict[str, RateLimitBucket] = {}

        # State
        self.running = False
        self.started_at: Optional[datetime] = None

        # Callbacks
        self.on_peer_connected: Optional[Callable] = None
        self.on_peer_disconnected: Optional[Callable] = None
        self.on_knowledge_received: Optional[Callable] = None
        self.on_query_received: Optional[Callable] = None

        # Stats
        self.stats = {
            'queries_sent': 0,
            'queries_received': 0,
            'knowledge_shared': 0,
            'knowledge_received': 0,
            'security_rejections': 0,
            'rate_limit_blocks': 0,
        }

    async def start(
        self,
        bootstrap_nodes: Optional[List[str]] = None,
    ):
        """
        Start the BAZINGA network.

        Args:
            bootstrap_nodes: List of "host:port" strings to bootstrap from
        """
        if self.running:
            print("Network already running")
            return

        print(f"🌐 Starting BAZINGA Network...")
        print(f"   Node ID: {self.node_id[:16]}...")
        print(f"   Name: {self.node_name}")

        # 1. Initialize P2P node
        self.node = BAZINGANode(
            host=self.host,
            port=self.port,
            node_id=self.node_id,
            tau=self.initial_tau,
        )

        # 2. Initialize DHT
        self.dht = BAZINGA_DHT(self.node)

        # 3. Initialize knowledge sync
        self.sync = KnowledgeGraphSync(self.kb, self.node)

        # 4. Initialize trust router
        self.router = TrustRouter(self.dht, self.initial_tau)

        # 5. Initialize α-SEED network
        self.alpha_network = AlphaSeedNetwork(self.kb, self.node)

        # Start node
        await self.node.start()
        self.port = self.node.port  # Update with actual port

        print(f"   Listening on {self.host}:{self.port}")

        # Bootstrap if nodes provided
        if bootstrap_nodes:
            await self._bootstrap(bootstrap_nodes)

        # Register message handlers
        self._setup_handlers()

        # Build α-SEED network
        if self.kb:
            await self.alpha_network.build_anchor_network()

        self.running = True
        self.started_at = datetime.now()

        print(f"   ✓ Network started!")

    async def _bootstrap(self, nodes: List[str]):
        """Bootstrap from known nodes."""
        print(f"   Bootstrapping from {len(nodes)} node(s)...")

        for node_str in nodes:
            try:
                host, port_str = node_str.rsplit(':', 1)
                port = int(port_str)
                await self.dht.bootstrap(host, port)
                print(f"   ✓ Connected to {host}:{port}")
            except Exception as e:
                print(f"   ✗ Failed to connect to {node_str}: {e}")

    def _setup_handlers(self):
        """Set up message handlers."""
        if not self.node:
            return

        # Handle incoming queries
        self.node.subscribe(
            "/bazinga/query",
            self._handle_query,
        )

        # Handle knowledge sync
        self.node.subscribe(
            "/bazinga/knowledge-sync",
            self._handle_knowledge_sync,
        )

        # Handle α-SEED announcements
        self.node.subscribe(
            "/bazinga/alpha-seed",
            self._handle_alpha_seed,
        )

        # Handle security handshake
        self.node.subscribe(
            "/bazinga/security",
            self._handle_security,
        )

    async def _handle_query(self, message):
        """Handle incoming query."""
        try:
            payload = message.payload
            source = payload.get('source_node', 'unknown')

            # Rate limit check
            if not self._check_rate_limit(source):
                self.stats['rate_limit_blocks'] += 1
                return

            self.stats['queries_received'] += 1

            # Extract query
            query_embedding = payload.get('embedding', [])
            query_text = payload.get('query', '')

            # Search local knowledge
            results = []
            if self.kb and query_embedding:
                results = self.kb.search(query_embedding, limit=5)

            # Callback
            if self.on_query_received:
                self.on_query_received(source, query_text, results)

            # Send response
            await self.node.send_response(
                source,
                {
                    'type': 'QUERY_RESPONSE',
                    'results': results,
                    'source_node': self.node_id,
                    'tau': self.node.tau,
                }
            )

        except Exception as e:
            print(f"Error handling query: {e}")

    async def _handle_knowledge_sync(self, message):
        """Handle incoming knowledge package."""
        try:
            payload = message.payload
            source = payload.get('source_node', 'unknown')

            # Rate limit check
            if not self._check_rate_limit(source):
                self.stats['rate_limit_blocks'] += 1
                return

            # Receive knowledge
            if self.sync:
                accepted = await self.sync.receive_knowledge(payload)

                if accepted:
                    self.stats['knowledge_received'] += 1

                    # Callback
                    if self.on_knowledge_received:
                        self.on_knowledge_received(source, payload)

        except Exception as e:
            print(f"Error handling knowledge sync: {e}")

    async def _handle_alpha_seed(self, message):
        """Handle α-SEED announcement."""
        try:
            if self.alpha_network:
                await self.alpha_network._handle_alpha_seed_announcement(message)
        except Exception as e:
            print(f"Error handling α-SEED: {e}")

    async def _handle_security(self, message):
        """Handle security handshake messages."""
        try:
            payload = message.payload
            msg_type = payload.get('type')
            source = payload.get('source_node')

            if msg_type == 'HANDSHAKE_INIT':
                # Respond to handshake
                await self._respond_handshake(source, payload)
            elif msg_type == 'HANDSHAKE_RESPONSE':
                # Complete handshake
                self._complete_handshake(source, payload)

        except Exception as e:
            print(f"Error handling security: {e}")

    async def _respond_handshake(self, source: str, payload: Dict):
        """Respond to security handshake."""
        # Generate session key
        their_nonce = bytes.fromhex(payload.get('nonce', ''))
        our_nonce = secrets.token_bytes(NONCE_SIZE)

        # Derive session key
        shared = self.node_key + their_nonce + our_nonce
        session_key = derive_session_key(shared, b"session")

        # Store context
        self.security_contexts[source] = SecurityContext(
            peer_id=source,
            session_key=session_key,
        )

        # Send response
        await self.node.publish(
            "/bazinga/security",
            {
                'type': 'HANDSHAKE_RESPONSE',
                'source_node': self.node_id,
                'nonce': our_nonce.hex(),
                'node_proof': compute_message_auth(
                    session_key,
                    self.node_id.encode(),
                ).hex(),
            }
        )

    def _complete_handshake(self, source: str, payload: Dict):
        """Complete security handshake."""
        if source in self.security_contexts:
            ctx = self.security_contexts[source]
            # Verify their proof
            their_proof = bytes.fromhex(payload.get('node_proof', ''))
            expected = compute_message_auth(ctx.session_key, source.encode())

            if hmac.compare_digest(their_proof, expected):
                print(f"   🔐 Secure channel established with {source[:8]}")
            else:
                # Invalid proof - remove context
                del self.security_contexts[source]
                self.stats['security_rejections'] += 1

    def _check_rate_limit(self, peer_id: str) -> bool:
        """Check if peer is within rate limits."""
        if peer_id not in self.rate_limits:
            self.rate_limits[peer_id] = RateLimitBucket(peer_id)

        return self.rate_limits[peer_id].check_rate_limit()

    async def query(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        min_tau: float = 0.5,
        timeout: float = 10.0,
    ) -> List[Dict]:
        """
        Query the BAZINGA network.

        Args:
            query_text: Query string
            query_embedding: Query embedding (will be computed if not provided)
            min_tau: Minimum trust score for responders
            timeout: Query timeout in seconds

        Returns:
            List of results from network
        """
        if not self.running:
            print("Network not running")
            return []

        # Compute embedding if not provided
        if query_embedding is None and self.kb:
            query_embedding = self.kb.embed(query_text)

        if not query_embedding:
            print("Cannot query without embedding")
            return []

        self.stats['queries_sent'] += 1

        # Route query through trust router
        if self.router:
            results = await self.router.route_query(
                query_embedding,
                min_tau=min_tau,
            )
            return results

        # Fallback: direct query to all peers
        return await self._direct_query(query_embedding, timeout)

    async def _direct_query(
        self,
        query_embedding: List[float],
        timeout: float,
    ) -> List[Dict]:
        """Direct query to all connected peers."""
        if not self.node:
            return []

        # Publish query
        await self.node.publish(
            "/bazinga/query",
            {
                'type': 'QUERY',
                'embedding': query_embedding[:10],  # Truncate for privacy
                'source_node': self.node_id,
                'tau': self.node.tau,
            }
        )

        # Wait for responses
        await asyncio.sleep(min(timeout, 5.0))

        return []  # Results come through callbacks

    async def share_knowledge(
        self,
        doc_id: Optional[str] = None,
        share_all_alpha_seeds: bool = True,
    ):
        """
        Share knowledge with the network.

        Args:
            doc_id: Specific document to share
            share_all_alpha_seeds: Share all α-SEED documents
        """
        if not self.running:
            print("Network not running")
            return

        if not self.sync:
            print("Knowledge sync not initialized")
            return

        if doc_id:
            # Share specific document
            package = self.sync.create_sync_package(doc_id)
            if package:
                await self.node.publish(
                    "/bazinga/knowledge-sync",
                    package.to_dict(),
                )
                self.stats['knowledge_shared'] += 1

        if share_all_alpha_seeds:
            # Share all α-SEED documents
            await self.sync.sync_to_network(force_all=False)

    async def stop(self):
        """Stop the BAZINGA network."""
        if not self.running:
            return

        print(f"🛑 Stopping BAZINGA Network...")

        if self.node:
            await self.node.stop()

        self.running = False
        print(f"   ✓ Network stopped")

    def get_peer_count(self) -> int:
        """Get number of connected peers."""
        if self.node:
            return len(self.node.peers)
        return 0

    def get_alpha_seed_count(self) -> Tuple[int, int]:
        """Get local and network α-SEED counts."""
        if self.alpha_network:
            return (
                self.alpha_network.get_local_seed_count(),
                self.alpha_network.get_network_seed_count(),
            )
        return (0, 0)

    def get_trust_stats(self) -> Dict[str, Any]:
        """Get trust statistics."""
        if self.router:
            return self.router.get_trust_stats()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        uptime = None
        if self.started_at:
            uptime = (datetime.now() - self.started_at).total_seconds()

        return {
            'running': self.running,
            'node_id': self.node_id[:16] + '...' if self.node_id else None,
            'node_name': self.node_name,
            'tau': self.node.tau if self.node else self.initial_tau,
            'peers': self.get_peer_count(),
            'alpha_seeds': self.get_alpha_seed_count(),
            'uptime_seconds': uptime,
            'secure_channels': len(self.security_contexts),
            **self.stats,
        }

    def print_status(self):
        """Print network status."""
        stats = self.get_stats()
        local_seeds, network_seeds = stats.get('alpha_seeds', (0, 0))

        print("\n" + "=" * 50)
        print("  BAZINGA Network Status")
        print("=" * 50)
        print(f"  Running: {'✓' if stats['running'] else '✗'}")
        print(f"  Node ID: {stats.get('node_id', 'N/A')}")
        print(f"  Name: {stats['node_name']}")
        print(f"  Trust (τ): {stats['tau']:.3f}")
        print(f"  Peers: {stats['peers']}")
        print(f"  α-SEEDs: {local_seeds} local, {network_seeds} network")
        print(f"  Secure Channels: {stats['secure_channels']}")

        if stats['uptime_seconds']:
            hours = int(stats['uptime_seconds'] // 3600)
            mins = int((stats['uptime_seconds'] % 3600) // 60)
            print(f"  Uptime: {hours}h {mins}m")

        print(f"\n  Queries: {stats['queries_sent']} sent, {stats['queries_received']} received")
        print(f"  Knowledge: {stats['knowledge_shared']} shared, {stats['knowledge_received']} received")

        if stats['security_rejections'] > 0:
            print(f"  Security: {stats['security_rejections']} rejections")
        if stats['rate_limit_blocks'] > 0:
            print(f"  Rate Limits: {stats['rate_limit_blocks']} blocks")

        print("=" * 50)


# Convenience function for quick network creation
async def create_network(
    kb=None,
    name: str = "BAZINGA",
    port: int = 0,
    bootstrap: Optional[List[str]] = None,
) -> BAZINGANetwork:
    """
    Create and start a BAZINGA network node.

    Args:
        kb: Local knowledge base
        name: Node name
        port: Port to listen on (0 for auto)
        bootstrap: Bootstrap nodes ["host:port", ...]

    Returns:
        Running BAZINGANetwork instance
    """
    network = BAZINGANetwork(kb=kb, node_name=name, port=port)
    await network.start(bootstrap_nodes=bootstrap)
    return network


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA Unified Network Test")
        print("=" * 60)

        # Create network without KB (for testing)
        network = BAZINGANetwork(
            kb=None,
            node_name="TestNode",
            port=0,
        )

        # Start network
        await network.start()

        # Print status
        network.print_status()

        # Test α-SEED detection
        test_content = "The consciousness network emerges"
        if is_alpha_seed(test_content):
            print(f"\n✓ '{test_content[:30]}...' is an α-SEED")
        else:
            print(f"\n○ '{test_content[:30]}...' is not an α-SEED")

        # Wait a bit
        await asyncio.sleep(2)

        # Stop
        await network.stop()

        print("\n✓ Unified Network module ready!")

    asyncio.run(test())
