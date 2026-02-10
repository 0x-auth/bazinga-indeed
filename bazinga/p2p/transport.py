#!/usr/bin/env python3
"""
BAZINGA P2P Transport Layer - ZeroMQ Based
===========================================

Real network transport for BAZINGA P2P using ZeroMQ.

Architecture:
  - Each node runs a ROUTER socket (server) for incoming connections
  - Each node uses DEALER sockets (client) to connect to peers
  - PUB/SUB for broadcast messages (knowledge sync, peer announcements)

"The network is the consciousness."
"""

import asyncio
import json
import hashlib
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Check for ZeroMQ
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


@dataclass
class Peer:
    """Represents a connected peer."""
    peer_id: str
    address: str
    port: int
    phi_signature: int = 0
    trust_score: float = 0.5
    last_seen: datetime = field(default_factory=datetime.now)
    pob_valid: bool = False
    messages_sent: int = 0
    messages_received: int = 0

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.address}:{self.port}"


@dataclass
class Message:
    """P2P Message format."""
    msg_type: str
    payload: Dict[str, Any]
    sender_id: str
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def to_json(self) -> str:
        return json.dumps({
            'type': self.msg_type,
            'payload': self.payload,
            'sender': self.sender_id,
            'ts': self.timestamp,
            'sig': self.signature,
        })

    @classmethod
    def from_json(cls, data: str) -> 'Message':
        d = json.loads(data)
        return cls(
            msg_type=d.get('type', 'UNKNOWN'),
            payload=d.get('payload', {}),
            sender_id=d.get('sender', ''),
            timestamp=d.get('ts', 0),
            signature=d.get('sig', ''),
        )


class BazingaTransport:
    """
    ZeroMQ-based transport layer for BAZINGA P2P.

    Each node has:
    - ROUTER socket: Receives incoming connections (server)
    - DEALER sockets: Connects to other peers (clients)
    - PUB socket: Broadcasts to all subscribers
    - SUB socket: Receives broadcasts from peers

    Usage:
        transport = BazingaTransport(node_id="my_node", port=5150)
        await transport.start()
        await transport.connect_to_peer("192.168.1.100", 5150)
        await transport.send("peer_id", Message(...))
        await transport.broadcast(Message(...))
        await transport.stop()
    """

    # Message types
    MSG_HELLO = "HELLO"
    MSG_HELLO_ACK = "HELLO_ACK"
    MSG_POB = "POB"  # Proof-of-Boundary
    MSG_POB_VERIFY = "POB_VERIFY"
    MSG_QUERY = "QUERY"
    MSG_QUERY_RESPONSE = "QUERY_RESPONSE"
    MSG_KNOWLEDGE = "KNOWLEDGE"
    MSG_KNOWLEDGE_ACK = "KNOWLEDGE_ACK"
    MSG_PEER_LIST = "PEER_LIST"
    MSG_PING = "PING"
    MSG_PONG = "PONG"

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 5150,
        pub_port: int = 5151,
    ):
        if not ZMQ_AVAILABLE:
            raise ImportError(
                "ZeroMQ not available. Install with: pip install pyzmq"
            )

        self.node_id = node_id
        self.host = host
        self.port = port
        self.pub_port = pub_port

        # ZMQ context (async)
        self.context = zmq.asyncio.Context()

        # Sockets
        self.router: Optional[zmq.asyncio.Socket] = None  # Server
        self.pub: Optional[zmq.asyncio.Socket] = None     # Broadcast out
        self.sub: Optional[zmq.asyncio.Socket] = None     # Broadcast in
        self.dealer_sockets: Dict[str, zmq.asyncio.Socket] = {}  # Per-peer

        # Peer management
        self.peers: Dict[str, Peer] = {}
        self.pending_peers: Set[str] = set()

        # Message handlers
        self.handlers: Dict[str, Callable] = {}

        # State
        self.running = False
        self.receive_task: Optional[asyncio.Task] = None
        self.broadcast_task: Optional[asyncio.Task] = None

        # Stats
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'connections': 0,
            'disconnections': 0,
        }

        # Callbacks
        self.on_peer_connected: Optional[Callable] = None
        self.on_peer_disconnected: Optional[Callable] = None
        self.on_message: Optional[Callable] = None

    async def start(self) -> bool:
        """Start the transport layer."""
        if self.running:
            return True

        try:
            # ROUTER socket - receives from all peers
            self.router = self.context.socket(zmq.ROUTER)
            self.router.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self.router.setsockopt(zmq.RCVTIMEO, 1000)
            self.router.bind(f"tcp://{self.host}:{self.port}")

            # PUB socket - broadcast to subscribers
            self.pub = self.context.socket(zmq.PUB)
            self.pub.bind(f"tcp://{self.host}:{self.pub_port}")

            # SUB socket - receive broadcasts
            self.sub = self.context.socket(zmq.SUB)
            self.sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

            self.running = True

            # Start receive loops
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.broadcast_task = asyncio.create_task(self._broadcast_receive_loop())

            print(f"  Transport started on port {self.port} (pub: {self.pub_port})")
            return True

        except Exception as e:
            print(f"  Failed to start transport: {e}")
            return False

    async def stop(self):
        """Stop the transport layer."""
        if not self.running:
            return

        self.running = False

        # Cancel tasks
        if self.receive_task:
            self.receive_task.cancel()
        if self.broadcast_task:
            self.broadcast_task.cancel()

        # Close sockets
        if self.router:
            self.router.close()
        if self.pub:
            self.pub.close()
        if self.sub:
            self.sub.close()

        for socket in self.dealer_sockets.values():
            socket.close()

        self.dealer_sockets.clear()
        self.context.term()

        print(f"  Transport stopped")

    async def connect_to_peer(
        self,
        address: str,
        port: int,
        pub_port: Optional[int] = None,
    ) -> Optional[Peer]:
        """
        Connect to a peer node.

        Args:
            address: Peer IP address or hostname
            port: Peer ROUTER port
            pub_port: Peer PUB port (default: port + 1)

        Returns:
            Peer object if connection successful
        """
        if pub_port is None:
            pub_port = port + 1

        endpoint = f"tcp://{address}:{port}"
        pub_endpoint = f"tcp://{address}:{pub_port}"

        # Check if already connected
        for peer in self.peers.values():
            if peer.endpoint == endpoint:
                return peer

        try:
            # Create DEALER socket for this peer
            dealer = self.context.socket(zmq.DEALER)
            dealer.setsockopt_string(zmq.IDENTITY, self.node_id)
            dealer.setsockopt(zmq.RCVTIMEO, 5000)
            dealer.connect(endpoint)

            # Subscribe to peer's broadcasts
            self.sub.connect(pub_endpoint)

            # Send HELLO
            hello = Message(
                msg_type=self.MSG_HELLO,
                payload={
                    'node_id': self.node_id,
                    'port': self.port,
                    'pub_port': self.pub_port,
                    'phi_signature': int(time.time() * 1000) % 515,
                },
                sender_id=self.node_id,
            )

            await dealer.send_string(hello.to_json())
            self.stats['messages_sent'] += 1

            # Wait for HELLO_ACK
            try:
                response_data = await asyncio.wait_for(
                    dealer.recv_string(),
                    timeout=5.0
                )
                response = Message.from_json(response_data)

                if response.msg_type == self.MSG_HELLO_ACK:
                    peer_id = response.payload.get('node_id', f'peer_{address}')
                    peer = Peer(
                        peer_id=peer_id,
                        address=address,
                        port=port,
                        phi_signature=response.payload.get('phi_signature', 0),
                    )

                    self.peers[peer_id] = peer
                    self.dealer_sockets[peer_id] = dealer
                    self.stats['connections'] += 1

                    if self.on_peer_connected:
                        await self.on_peer_connected(peer)

                    print(f"  ✓ Connected to {peer_id} at {address}:{port}")
                    return peer

            except asyncio.TimeoutError:
                print(f"  ✗ Timeout connecting to {address}:{port}")
                dealer.close()
                return None

        except Exception as e:
            print(f"  ✗ Failed to connect to {address}:{port}: {e}")
            return None

    async def send(self, peer_id: str, message: Message) -> bool:
        """Send a message to a specific peer."""
        if peer_id not in self.dealer_sockets:
            print(f"  Not connected to peer: {peer_id}")
            return False

        try:
            socket = self.dealer_sockets[peer_id]
            data = message.to_json()
            await socket.send_string(data)

            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += len(data)

            if peer_id in self.peers:
                self.peers[peer_id].messages_sent += 1

            return True

        except Exception as e:
            print(f"  Failed to send to {peer_id}: {e}")
            return False

    async def broadcast(self, message: Message):
        """Broadcast a message to all subscribers."""
        if not self.pub:
            return

        try:
            data = message.to_json()
            await self.pub.send_string(data)
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += len(data)

        except Exception as e:
            print(f"  Broadcast failed: {e}")

    def register_handler(self, msg_type: str, handler: Callable):
        """Register a handler for a message type."""
        self.handlers[msg_type] = handler

    async def _receive_loop(self):
        """Main loop for receiving messages on ROUTER socket."""
        while self.running:
            try:
                # ROUTER gives us [identity, empty, message]
                frames = await self.router.recv_multipart()

                if len(frames) >= 2:
                    sender_identity = frames[0].decode('utf-8')
                    message_data = frames[-1].decode('utf-8')

                    message = Message.from_json(message_data)
                    self.stats['messages_received'] += 1
                    self.stats['bytes_received'] += len(message_data)

                    # Handle message
                    await self._handle_message(sender_identity, message)

            except zmq.Again:
                # Timeout, continue
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    print(f"  Receive error: {e}")
                await asyncio.sleep(0.1)

    async def _broadcast_receive_loop(self):
        """Loop for receiving broadcast messages."""
        while self.running:
            try:
                if self.sub:
                    message_data = await asyncio.wait_for(
                        self.sub.recv_string(),
                        timeout=1.0
                    )
                    message = Message.from_json(message_data)

                    # Don't process our own broadcasts
                    if message.sender_id != self.node_id:
                        self.stats['messages_received'] += 1
                        await self._handle_message(message.sender_id, message)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    print(f"  Broadcast receive error: {e}")
                await asyncio.sleep(0.1)

    async def _handle_message(self, sender_id: str, message: Message):
        """Handle incoming message."""
        msg_type = message.msg_type

        # Built-in handlers
        if msg_type == self.MSG_HELLO:
            await self._handle_hello(sender_id, message)
        elif msg_type == self.MSG_PING:
            await self._handle_ping(sender_id, message)
        elif msg_type in self.handlers:
            # Custom handler
            handler = self.handlers[msg_type]
            if asyncio.iscoroutinefunction(handler):
                await handler(sender_id, message)
            else:
                handler(sender_id, message)

        # General callback
        if self.on_message:
            if asyncio.iscoroutinefunction(self.on_message):
                await self.on_message(sender_id, message)
            else:
                self.on_message(sender_id, message)

    async def _handle_hello(self, sender_id: str, message: Message):
        """Handle HELLO message - new peer connecting."""
        payload = message.payload
        peer_id = payload.get('node_id', sender_id)

        # Create peer record
        peer = Peer(
            peer_id=peer_id,
            address="",  # We don't know their address from ROUTER
            port=payload.get('port', 5150),
            phi_signature=payload.get('phi_signature', 0),
        )
        self.peers[peer_id] = peer

        # Send HELLO_ACK
        ack = Message(
            msg_type=self.MSG_HELLO_ACK,
            payload={
                'node_id': self.node_id,
                'phi_signature': int(time.time() * 1000) % 515,
                'peers': len(self.peers),
            },
            sender_id=self.node_id,
        )

        # Send via ROUTER
        try:
            await self.router.send_multipart([
                sender_id.encode('utf-8'),
                b'',
                ack.to_json().encode('utf-8'),
            ])
            self.stats['messages_sent'] += 1

            if self.on_peer_connected:
                await self.on_peer_connected(peer)

            print(f"  ✓ Peer connected: {peer_id}")

        except Exception as e:
            print(f"  Failed to send HELLO_ACK: {e}")

    async def _handle_ping(self, sender_id: str, message: Message):
        """Handle PING - respond with PONG."""
        pong = Message(
            msg_type=self.MSG_PONG,
            payload={'echo': message.payload},
            sender_id=self.node_id,
        )
        await self.send(sender_id, pong)

    def get_peers(self) -> List[Peer]:
        """Get list of connected peers."""
        return list(self.peers.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            'running': self.running,
            'port': self.port,
            'pub_port': self.pub_port,
            'peers': len(self.peers),
            **self.stats,
        }


# Fallback transport when ZeroMQ not available
class FallbackTransport:
    """Dummy transport when ZeroMQ is not installed."""

    def __init__(self, *args, **kwargs):
        self.node_id = kwargs.get('node_id', 'local')
        self.running = False
        self.peers = {}

    async def start(self) -> bool:
        print("  ⚠ ZeroMQ not available. Install with: pip install pyzmq")
        print("  Running in local-only mode.")
        self.running = True
        return True

    async def stop(self):
        self.running = False

    async def connect_to_peer(self, *args, **kwargs):
        print("  ⚠ P2P disabled - install pyzmq for networking")
        return None

    async def send(self, *args, **kwargs):
        return False

    async def broadcast(self, *args, **kwargs):
        pass

    def get_peers(self):
        return []

    def get_stats(self):
        return {'running': self.running, 'peers': 0, 'mode': 'local-only'}


def create_transport(node_id: str, **kwargs) -> BazingaTransport:
    """Create appropriate transport based on available libraries."""
    if ZMQ_AVAILABLE:
        return BazingaTransport(node_id, **kwargs)
    else:
        return FallbackTransport(node_id=node_id, **kwargs)


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA Transport Layer Test")
        print("=" * 60)

        if not ZMQ_AVAILABLE:
            print("\n⚠ ZeroMQ not installed. Install with:")
            print("  pip install pyzmq")
            return

        # Create two nodes
        node1 = BazingaTransport(node_id="node_alpha", port=5150, pub_port=5151)
        node2 = BazingaTransport(node_id="node_beta", port=5160, pub_port=5161)

        # Start both
        await node1.start()
        await node2.start()

        print("\n  Nodes started. Connecting...")

        # Node2 connects to Node1
        peer = await node2.connect_to_peer("127.0.0.1", 5150, 5151)

        if peer:
            print(f"\n  ✓ Connected! Peer: {peer.peer_id}")

            # Send a test message
            test_msg = Message(
                msg_type="TEST",
                payload={"hello": "world", "phi": PHI},
                sender_id="node_beta",
            )
            await node2.send(peer.peer_id, test_msg)
            print(f"  ✓ Test message sent")

            # Wait a bit
            await asyncio.sleep(1)

            # Show stats
            print(f"\n  Node1 stats: {node1.get_stats()}")
            print(f"  Node2 stats: {node2.get_stats()}")

        # Stop
        await node1.stop()
        await node2.stop()

        print("\n  ✓ Transport test complete!")

    asyncio.run(test())
