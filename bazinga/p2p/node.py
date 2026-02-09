#!/usr/bin/env python3
"""
BAZINGA P2P Node - Core P2P Transport Layer

Uses asyncio for networking (libp2p-inspired design but pure Python for portability).
Can be upgraded to use actual libp2p when py-libp2p matures.

Features:
- Node discovery via bootstrap nodes
- GossipSub-style pub/sub messaging
- Trust dimension (τ) tracking
- Query publishing and handling
"""

import asyncio
import hashlib
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# P2P Protocol version
PROTOCOL_VERSION = "/bazinga/1.0.0"


class MessageType(Enum):
    """Types of P2P messages."""
    QUERY = "QUERY"
    RESPONSE = "RESPONSE"
    PING = "PING"
    PONG = "PONG"
    ANNOUNCE = "ANNOUNCE"
    KNOWLEDGE_SYNC = "KNOWLEDGE_SYNC"
    ALPHA_SEED = "ALPHA_SEED"
    TRUST_UPDATE = "TRUST_UPDATE"


@dataclass
class PeerInfo:
    """Information about a connected peer."""
    node_id: str
    host: str
    port: int
    tau: float  # Trust dimension
    last_seen: datetime = field(default_factory=datetime.now)
    messages_received: int = 0
    good_responses: int = 0
    bad_responses: int = 0

    @property
    def multiaddr(self) -> str:
        """Get multiaddr-style address."""
        return f"/ip4/{self.host}/tcp/{self.port}/p2p/{self.node_id}"

    def update_trust(self, good_response: bool):
        """Update trust based on response quality."""
        if good_response:
            self.good_responses += 1
            # φ-based trust increase
            self.tau = min(1.0, self.tau + 0.01 * (1/PHI))
        else:
            self.bad_responses += 1
            # Faster trust decrease for bad responses
            self.tau = max(0.1, self.tau - 0.05)

        # Recalculate based on ratio
        total = self.good_responses + self.bad_responses
        if total > 10:
            ratio = self.good_responses / total
            # Blend current tau with ratio
            self.tau = self.tau * (1/PHI) + ratio * (1 - 1/PHI)


@dataclass
class Message:
    """P2P message structure."""
    msg_type: MessageType
    sender: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        data = {
            "type": self.msg_type.value,
            "sender": self.sender,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "msg_id": self.msg_id,
        }
        return json.dumps(data).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes."""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=MessageType(obj["type"]),
            sender=obj["sender"],
            payload=obj["payload"],
            timestamp=obj["timestamp"],
            msg_id=obj["msg_id"],
        )


class BAZINGANode:
    """
    BAZINGA P2P Node - Core networking component.

    Implements:
    - Peer discovery and connection
    - GossipSub-style message propagation
    - Trust-based peer scoring
    - Query/response handling

    Architecture:
    - Each node has a unique ID and Trust score (τ)
    - Nodes discover each other via bootstrap nodes
    - Messages propagate via gossip protocol
    - Trust scores determine query routing priority
    """

    DEFAULT_PORT = 8468  # BAZINGA default port

    def __init__(
        self,
        node_id: Optional[str] = None,
        initial_tau: float = 0.5,
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        """
        Initialize BAZINGA P2P node.

        Args:
            node_id: Unique node identifier (generated if not provided)
            initial_tau: Initial trust score (0.0-1.0)
            host: Host to bind to
            port: Port to bind to (0 for random)
        """
        self.node_id = node_id or self._generate_node_id()
        self.tau = initial_tau
        self.host = host
        self.port = port

        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.bootstrap_nodes: List[str] = []

        # Message handling
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.seen_messages: Set[str] = set()  # Prevent message loops
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # Server state
        self.server: Optional[asyncio.Server] = None
        self.running = False

        # Stats
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'queries_handled': 0,
            'peers_connected': 0,
        }

        # Local knowledge base reference (set externally)
        self.local_kb = None

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        # Use timestamp + random for uniqueness
        data = f"{time.time()}-{uuid.uuid4()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def start(self, port: Optional[int] = None):
        """
        Start the P2P node.

        Args:
            port: Override port (uses self.port if not provided)
        """
        if port:
            self.port = port

        # Start TCP server
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        # Get actual port if was 0
        addr = self.server.sockets[0].getsockname()
        self.port = addr[1]

        self.running = True

        print(f"🌐 BAZINGA Node {self.node_id[:8]} online")
        print(f"   Address: {self.host}:{self.port}")
        print(f"   Trust Score (τ): {self.tau:.3f}")
        print(f"   Protocol: {PROTOCOL_VERSION}")

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._cleanup_loop())

        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the P2P node."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Disconnect from all peers
        for peer_id in list(self.peers.keys()):
            await self.disconnect_peer(peer_id)

        print(f"🔴 BAZINGA Node {self.node_id[:8]} offline")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming connection."""
        peer_addr = writer.get_extra_info('peername')

        try:
            while self.running:
                # Read message length (4 bytes)
                length_data = await reader.read(4)
                if not length_data:
                    break

                msg_length = int.from_bytes(length_data, 'big')

                # Read message
                msg_data = await reader.read(msg_length)
                if not msg_data:
                    break

                # Parse and handle
                try:
                    message = Message.from_bytes(msg_data)
                    await self._handle_message(message, writer)
                except Exception as e:
                    print(f"Error handling message: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Connection error from {peer_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_message(self, message: Message, writer: asyncio.StreamWriter):
        """Handle incoming message."""
        self.stats['messages_received'] += 1

        # Prevent message loops
        if message.msg_id in self.seen_messages:
            return
        self.seen_messages.add(message.msg_id)

        # Update peer info
        if message.sender in self.peers:
            self.peers[message.sender].last_seen = datetime.now()
            self.peers[message.sender].messages_received += 1

        # Handle by type
        if message.msg_type == MessageType.PING:
            await self._handle_ping(message, writer)

        elif message.msg_type == MessageType.PONG:
            await self._handle_pong(message)

        elif message.msg_type == MessageType.ANNOUNCE:
            await self._handle_announce(message)

        elif message.msg_type == MessageType.QUERY:
            await self._handle_query(message, writer)

        elif message.msg_type == MessageType.RESPONSE:
            await self._handle_response(message)

        elif message.msg_type == MessageType.KNOWLEDGE_SYNC:
            await self._handle_knowledge_sync(message)

        elif message.msg_type == MessageType.ALPHA_SEED:
            await self._handle_alpha_seed(message)

        # Notify subscribers
        topic = f"/bazinga/{message.msg_type.value.lower()}"
        if topic in self.subscriptions:
            for callback in self.subscriptions[topic]:
                try:
                    await callback(message)
                except Exception as e:
                    print(f"Subscription callback error: {e}")

    async def _handle_ping(self, message: Message, writer: asyncio.StreamWriter):
        """Handle PING message."""
        pong = Message(
            msg_type=MessageType.PONG,
            sender=self.node_id,
            payload={
                "tau": self.tau,
                "timestamp": time.time(),
            }
        )
        await self._send_to_writer(writer, pong)

    async def _handle_pong(self, message: Message):
        """Handle PONG message."""
        # Update peer trust from pong
        if message.sender in self.peers:
            remote_tau = message.payload.get("tau", 0.5)
            self.peers[message.sender].tau = remote_tau

    async def _handle_announce(self, message: Message):
        """Handle peer announcement."""
        payload = message.payload

        peer_info = PeerInfo(
            node_id=message.sender,
            host=payload.get("host", ""),
            port=payload.get("port", 0),
            tau=payload.get("tau", 0.5),
        )

        self.peers[message.sender] = peer_info
        self.stats['peers_connected'] = len(self.peers)

        print(f"   📡 Discovered peer: {message.sender[:8]} (τ={peer_info.tau:.2f})")

    async def _handle_query(self, message: Message, writer: asyncio.StreamWriter):
        """Handle incoming query."""
        self.stats['queries_handled'] += 1

        payload = message.payload
        tau_threshold = payload.get("tau_threshold", 0.5)

        # Only respond if our Trust meets threshold
        if self.tau < tau_threshold:
            return

        # Search local KB if available
        results = []
        if self.local_kb:
            query_embedding = payload.get("embedding")
            if query_embedding:
                try:
                    results = self.local_kb.search(query_embedding, limit=5)
                except Exception as e:
                    print(f"Local KB search error: {e}")

        # Send response
        response = Message(
            msg_type=MessageType.RESPONSE,
            sender=self.node_id,
            payload={
                "query_id": message.msg_id,
                "results": results,
                "tau": self.tau,
            }
        )

        await self._send_to_writer(writer, response)

    async def _handle_response(self, message: Message):
        """Handle query response."""
        query_id = message.payload.get("query_id")

        if query_id in self.pending_responses:
            future = self.pending_responses[query_id]
            if not future.done():
                future.set_result(message.payload)

    async def _handle_knowledge_sync(self, message: Message):
        """Handle knowledge sync message."""
        # Notify subscribers
        topic = "/bazinga/knowledge-sync"
        if topic in self.subscriptions:
            for callback in self.subscriptions[topic]:
                await callback(message)

    async def _handle_alpha_seed(self, message: Message):
        """Handle α-SEED announcement."""
        # Notify subscribers
        topic = "/bazinga/alpha-seed"
        if topic in self.subscriptions:
            for callback in self.subscriptions[topic]:
                await callback(message)

    async def connect_to_peer(self, host: str, port: int) -> bool:
        """
        Connect to another BAZINGA node.

        Args:
            host: Peer host
            port: Peer port

        Returns:
            True if connected successfully
        """
        try:
            reader, writer = await asyncio.open_connection(host, port)

            # Send announcement
            announce = Message(
                msg_type=MessageType.ANNOUNCE,
                sender=self.node_id,
                payload={
                    "host": self.host if self.host != "0.0.0.0" else "127.0.0.1",
                    "port": self.port,
                    "tau": self.tau,
                    "protocol": PROTOCOL_VERSION,
                }
            )

            await self._send_to_writer(writer, announce)

            # Wait for response (ping/pong)
            ping = Message(
                msg_type=MessageType.PING,
                sender=self.node_id,
                payload={"tau": self.tau}
            )
            await self._send_to_writer(writer, ping)

            # Read pong
            length_data = await asyncio.wait_for(reader.read(4), timeout=5.0)
            if length_data:
                msg_length = int.from_bytes(length_data, 'big')
                msg_data = await reader.read(msg_length)
                pong = Message.from_bytes(msg_data)

                # Add peer
                peer_info = PeerInfo(
                    node_id=pong.sender,
                    host=host,
                    port=port,
                    tau=pong.payload.get("tau", 0.5),
                )
                self.peers[pong.sender] = peer_info
                self.stats['peers_connected'] = len(self.peers)

                print(f"   ✓ Connected to peer: {pong.sender[:8]} (τ={peer_info.tau:.2f})")

                writer.close()
                await writer.wait_closed()
                return True

        except Exception as e:
            print(f"   ✗ Failed to connect to {host}:{port}: {e}")

        return False

    async def disconnect_peer(self, peer_id: str):
        """Disconnect from a peer."""
        if peer_id in self.peers:
            del self.peers[peer_id]
            self.stats['peers_connected'] = len(self.peers)

    async def bootstrap(self, bootstrap_addrs: List[str]):
        """
        Bootstrap from known nodes.

        Args:
            bootstrap_addrs: List of "host:port" addresses
        """
        self.bootstrap_nodes = bootstrap_addrs

        for addr in bootstrap_addrs:
            try:
                host, port = addr.split(":")
                await self.connect_to_peer(host, int(port))
            except Exception as e:
                print(f"   Bootstrap failed for {addr}: {e}")

    async def publish(self, topic: str, payload: Dict[str, Any]):
        """
        Publish message to topic (broadcast to all peers).

        Args:
            topic: Topic name (e.g., "/bazinga/query")
            payload: Message payload
        """
        # Determine message type from topic
        topic_to_type = {
            "/bazinga/query": MessageType.QUERY,
            "/bazinga/knowledge-sync": MessageType.KNOWLEDGE_SYNC,
            "/bazinga/alpha-seed": MessageType.ALPHA_SEED,
        }

        msg_type = topic_to_type.get(topic, MessageType.ANNOUNCE)

        message = Message(
            msg_type=msg_type,
            sender=self.node_id,
            payload=payload,
        )

        # Broadcast to all peers
        await self.broadcast(message)

    async def broadcast(self, message: Message):
        """Broadcast message to all connected peers."""
        self.seen_messages.add(message.msg_id)

        for peer_id, peer_info in list(self.peers.items()):
            try:
                await self._send_to_peer(peer_info, message)
                self.stats['messages_sent'] += 1
            except Exception as e:
                print(f"Failed to send to {peer_id[:8]}: {e}")

    async def _send_to_peer(self, peer_info: PeerInfo, message: Message):
        """Send message to specific peer."""
        reader, writer = await asyncio.open_connection(
            peer_info.host,
            peer_info.port,
        )

        try:
            await self._send_to_writer(writer, message)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _send_to_writer(
        self,
        writer: asyncio.StreamWriter,
        message: Message,
    ):
        """Send message to writer."""
        data = message.to_bytes()
        length = len(data).to_bytes(4, 'big')

        writer.write(length + data)
        await writer.drain()

    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe to topic.

        Args:
            topic: Topic name
            callback: Async callback function(message)
        """
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from topic."""
        if topic in self.subscriptions:
            self.subscriptions[topic] = [
                cb for cb in self.subscriptions[topic]
                if cb != callback
            ]

    async def query_network(
        self,
        embedding: List[float],
        tau_threshold: float = 0.7,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """
        Query the network for knowledge.

        Args:
            embedding: Query embedding vector
            tau_threshold: Minimum Trust score for responders
            timeout: Query timeout in seconds

        Returns:
            List of results from network
        """
        query_id = str(uuid.uuid4())[:8]

        # Create future for responses
        self.pending_responses[query_id] = asyncio.get_event_loop().create_future()

        # Publish query
        await self.publish("/bazinga/query", {
            "embedding": embedding,
            "tau_threshold": tau_threshold,
            "query_id": query_id,
        })

        # Wait for responses
        all_results = []
        try:
            response = await asyncio.wait_for(
                self.pending_responses[query_id],
                timeout=timeout,
            )
            all_results.extend(response.get("results", []))
        except asyncio.TimeoutError:
            pass
        finally:
            del self.pending_responses[query_id]

        return all_results

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to peers."""
        while self.running:
            await asyncio.sleep(30)  # Every 30 seconds

            for peer_id, peer_info in list(self.peers.items()):
                try:
                    reader, writer = await asyncio.open_connection(
                        peer_info.host,
                        peer_info.port,
                    )

                    ping = Message(
                        msg_type=MessageType.PING,
                        sender=self.node_id,
                        payload={"tau": self.tau}
                    )

                    await self._send_to_writer(writer, ping)
                    writer.close()
                    await writer.wait_closed()

                except Exception:
                    # Peer unreachable, remove after 3 failures
                    # (simplified - real impl would track failures)
                    pass

    async def _cleanup_loop(self):
        """Clean up old messages and stale peers."""
        while self.running:
            await asyncio.sleep(60)  # Every minute

            # Clean old seen messages (keep last 1000)
            if len(self.seen_messages) > 1000:
                # Convert to list, keep recent, convert back
                msg_list = list(self.seen_messages)
                self.seen_messages = set(msg_list[-500:])

            # Remove stale peers (not seen in 5 minutes)
            now = datetime.now()
            stale_peers = [
                peer_id for peer_id, info in self.peers.items()
                if (now - info.last_seen).total_seconds() > 300
            ]

            for peer_id in stale_peers:
                del self.peers[peer_id]

            if stale_peers:
                self.stats['peers_connected'] = len(self.peers)

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            "node_id": self.node_id,
            "tau": self.tau,
            "address": f"{self.host}:{self.port}",
            "peers": len(self.peers),
            "running": self.running,
            **self.stats,
        }

    def get_peer_list(self) -> List[Dict[str, Any]]:
        """Get list of connected peers."""
        return [
            {
                "node_id": info.node_id,
                "address": info.multiaddr,
                "tau": info.tau,
                "last_seen": info.last_seen.isoformat(),
            }
            for info in self.peers.values()
        ]


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA P2P Node Test")
        print("=" * 60)

        # Create node
        node = BAZINGANode(initial_tau=0.8)

        print(f"\nNode ID: {node.node_id}")
        print(f"Trust: {node.tau}")

        # Would start server here
        # await node.start(port=8468)

        print("\nNode created successfully!")
        print(f"Stats: {node.get_stats()}")

    asyncio.run(test())
