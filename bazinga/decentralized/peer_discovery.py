#!/usr/bin/env python3
"""
BAZINGA Peer Discovery - Bootstrap-Free Node Discovery

Enables peer discovery without central bootstrap servers:
- Rolling peer introduction (nodes remember recent peers)
- Local network discovery (mDNS/UDP broadcast)
- DHT-based peer exchange
- Seed persistence for network rejoin

"Find your tribe without asking permission."
"""

import asyncio
import json
import time
import hashlib
import socket
import struct
import random
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import threading

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Discovery constants
DISCOVERY_PORT = 31337  # Default UDP port for discovery
MULTICAST_GROUP = "239.137.1.618"  # phi-inspired multicast
PEER_TTL = 3600  # Peer record TTL in seconds
MAX_PEERS = 100  # Maximum peers to track
GOSSIP_FANOUT = 3  # Number of peers to gossip to


class DiscoveryMethod(Enum):
    """Methods for peer discovery."""
    MULTICAST = "multicast"     # Local network multicast
    DHT = "dht"                 # Distributed hash table
    GOSSIP = "gossip"           # Peer-to-peer gossip
    SEED_FILE = "seed_file"     # Persistent seed file
    MANUAL = "manual"           # Manually added


@dataclass
class PeerInfo:
    """Information about a discovered peer."""
    node_id: str
    address: str
    port: int
    tau_score: float = 0.5
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    discovery_method: DiscoveryMethod = DiscoveryMethod.MANUAL
    hop_count: int = 0  # How many hops from original source

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['discovery_method'] = self.discovery_method.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'PeerInfo':
        data['discovery_method'] = DiscoveryMethod(data.get('discovery_method', 'manual'))
        return cls(**data)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_seen > PEER_TTL

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"

    def refresh(self):
        """Mark peer as recently seen."""
        self.last_seen = time.time()


@dataclass
class DiscoveryMessage:
    """Message format for peer discovery."""
    msg_type: str  # "announce", "query", "response", "gossip"
    sender_id: str
    sender_addr: str
    sender_port: int
    peers: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    nonce: str = ""

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DiscoveryMessage':
        return cls(**json.loads(data.decode()))

    def sign(self, node_id: str) -> str:
        """Generate signature for message verification."""
        content = f"{self.msg_type}:{self.sender_id}:{self.timestamp}:{node_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class PeerDiscovery:
    """
    Base peer discovery mechanism.

    Maintains a list of known peers and provides
    methods to discover new peers and share peer info.
    """

    def __init__(
        self,
        node_id: str,
        listen_port: int = DISCOVERY_PORT,
        seed_file: Optional[str] = None,
    ):
        """
        Initialize peer discovery.

        Args:
            node_id: This node's unique identifier
            listen_port: Port to listen for discovery messages
            seed_file: Path to persistent seed file
        """
        self.node_id = node_id
        self.listen_port = listen_port
        self.seed_file = seed_file or f"~/.bazinga/peers_{node_id[:8]}.json"
        self.seed_file = Path(self.seed_file).expanduser()

        # Known peers
        self.peers: Dict[str, PeerInfo] = {}
        self._peers_lock = threading.Lock()

        # Discovery state
        self.running = False
        self._tasks: List[asyncio.Task] = []

        # Callbacks
        self.on_peer_discovered: Optional[Callable[[PeerInfo], None]] = None
        self.on_peer_lost: Optional[Callable[[str], None]] = None

        # Stats
        self.discovery_count = 0
        self.gossip_count = 0

        # Load persistent seeds
        self._load_seeds()

        print(f"PeerDiscovery initialized: {node_id[:8]}...")

    def _load_seeds(self):
        """Load peers from seed file."""
        if not self.seed_file.exists():
            return

        try:
            with open(self.seed_file, 'r') as f:
                data = json.load(f)

            for peer_data in data.get('peers', []):
                peer = PeerInfo.from_dict(peer_data)
                if not peer.is_stale:
                    self.peers[peer.node_id] = peer

            print(f"Loaded {len(self.peers)} peers from seed file")

        except Exception as e:
            print(f"Failed to load seeds: {e}")

    def _save_seeds(self):
        """Save peers to seed file for persistence."""
        self.seed_file.parent.mkdir(parents=True, exist_ok=True)

        # Filter non-stale peers
        active_peers = [
            peer.to_dict()
            for peer in self.peers.values()
            if not peer.is_stale
        ]

        data = {
            'node_id': self.node_id,
            'updated_at': time.time(),
            'peers': active_peers,
        }

        with open(self.seed_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_peer(self, peer: PeerInfo) -> bool:
        """
        Add a peer to known peers.

        Returns:
            True if peer was new
        """
        with self._peers_lock:
            # Don't add ourselves
            if peer.node_id == self.node_id:
                return False

            # Check capacity
            if len(self.peers) >= MAX_PEERS and peer.node_id not in self.peers:
                # Evict stale or low-tau peer
                self._evict_peer()

            is_new = peer.node_id not in self.peers
            self.peers[peer.node_id] = peer

            if is_new:
                self.discovery_count += 1
                if self.on_peer_discovered:
                    self.on_peer_discovered(peer)

            return is_new

    def remove_peer(self, node_id: str):
        """Remove a peer."""
        with self._peers_lock:
            if node_id in self.peers:
                del self.peers[node_id]
                if self.on_peer_lost:
                    self.on_peer_lost(node_id)

    def _evict_peer(self):
        """Evict a peer to make room for new ones."""
        # First try stale peers
        for node_id, peer in list(self.peers.items()):
            if peer.is_stale:
                del self.peers[node_id]
                return

        # Then lowest tau score
        if self.peers:
            lowest = min(self.peers.values(), key=lambda p: p.tau_score)
            del self.peers[lowest.node_id]

    def get_peer(self, node_id: str) -> Optional[PeerInfo]:
        """Get a specific peer."""
        return self.peers.get(node_id)

    def get_random_peers(self, count: int = GOSSIP_FANOUT) -> List[PeerInfo]:
        """Get random peers for gossip."""
        active = [p for p in self.peers.values() if not p.is_stale]
        if len(active) <= count:
            return active
        return random.sample(active, count)

    def get_best_peers(self, count: int = 10) -> List[PeerInfo]:
        """Get best peers by tau score."""
        active = [p for p in self.peers.values() if not p.is_stale]
        return sorted(active, key=lambda p: p.tau_score, reverse=True)[:count]

    def cleanup_stale(self):
        """Remove stale peers."""
        with self._peers_lock:
            stale = [nid for nid, p in self.peers.items() if p.is_stale]
            for node_id in stale:
                del self.peers[node_id]
                if self.on_peer_lost:
                    self.on_peer_lost(node_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            'node_id': self.node_id,
            'known_peers': len(self.peers),
            'active_peers': len([p for p in self.peers.values() if not p.is_stale]),
            'discovery_count': self.discovery_count,
            'gossip_count': self.gossip_count,
        }


class MulticastDiscovery:
    """
    Local network discovery using multicast.

    Discovers peers on the same local network without
    requiring bootstrap servers.
    """

    def __init__(
        self,
        peer_discovery: PeerDiscovery,
        multicast_group: str = MULTICAST_GROUP,
        port: int = DISCOVERY_PORT,
    ):
        self.discovery = peer_discovery
        self.multicast_group = multicast_group
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.running = False

    async def start(self):
        """Start multicast discovery."""
        self.running = True

        # Create multicast socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # Not available on all platforms

        self.sock.bind(('', self.port))

        # Join multicast group
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.sock.setblocking(False)

        # Start listener and announcer
        asyncio.create_task(self._listen())
        asyncio.create_task(self._announce_periodically())

        print(f"Multicast discovery started on {self.multicast_group}:{self.port}")

    async def stop(self):
        """Stop multicast discovery."""
        self.running = False
        if self.sock:
            self.sock.close()

    async def _listen(self):
        """Listen for multicast messages."""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                data, addr = await loop.sock_recvfrom(self.sock, 4096)
                await self._handle_message(data, addr)
            except BlockingIOError:
                await asyncio.sleep(0.1)
            except Exception as e:
                if self.running:
                    print(f"Multicast listen error: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle received discovery message."""
        try:
            msg = DiscoveryMessage.from_bytes(data)

            if msg.sender_id == self.discovery.node_id:
                return  # Ignore our own messages

            # Create peer info
            peer = PeerInfo(
                node_id=msg.sender_id,
                address=addr[0],
                port=msg.sender_port,
                discovery_method=DiscoveryMethod.MULTICAST,
            )

            self.discovery.add_peer(peer)

            # Also add any peers included in message
            for peer_data in msg.peers:
                included_peer = PeerInfo.from_dict(peer_data)
                included_peer.hop_count += 1
                self.discovery.add_peer(included_peer)

        except Exception as e:
            print(f"Failed to handle discovery message: {e}")

    async def _announce_periodically(self):
        """Periodically announce presence."""
        while self.running:
            await self.announce()
            # phi-scaled interval: ~60-100 seconds
            interval = 60 + random.random() * 40 * PHI
            await asyncio.sleep(interval)

    async def announce(self):
        """Announce presence to local network."""
        if not self.sock:
            return

        msg = DiscoveryMessage(
            msg_type="announce",
            sender_id=self.discovery.node_id,
            sender_addr="",  # Will be filled by receiver
            sender_port=self.discovery.listen_port,
            peers=[p.to_dict() for p in self.discovery.get_random_peers(3)],
        )

        try:
            self.sock.sendto(
                msg.to_bytes(),
                (self.multicast_group, self.port)
            )
        except Exception as e:
            print(f"Announce failed: {e}")


class GossipProtocol:
    """
    Gossip-based peer discovery.

    Spreads peer information through random peer-to-peer
    exchanges, enabling discovery across network partitions.
    """

    def __init__(
        self,
        peer_discovery: PeerDiscovery,
        gossip_interval: float = 30.0,
    ):
        self.discovery = peer_discovery
        self.gossip_interval = gossip_interval
        self.running = False

        # Anti-entropy
        self.seen_messages: Set[str] = set()
        self.max_seen = 1000

    async def start(self):
        """Start gossip protocol."""
        self.running = True
        asyncio.create_task(self._gossip_loop())
        print("Gossip protocol started")

    async def stop(self):
        """Stop gossip protocol."""
        self.running = False

    async def _gossip_loop(self):
        """Periodically gossip with random peers."""
        while self.running:
            await self.gossip_round()
            # phi-scaled jitter
            jitter = self.gossip_interval * (0.5 + random.random() * PHI / 2)
            await asyncio.sleep(jitter)

    async def gossip_round(self):
        """Perform one round of gossip."""
        targets = self.discovery.get_random_peers(GOSSIP_FANOUT)

        if not targets:
            return

        # Create gossip message
        msg = DiscoveryMessage(
            msg_type="gossip",
            sender_id=self.discovery.node_id,
            sender_addr="",
            sender_port=self.discovery.listen_port,
            peers=[p.to_dict() for p in self.discovery.get_best_peers(5)],
            nonce=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
        )

        # Track seen
        self._add_seen(msg.nonce)

        # Send to targets (simulated - in real impl would use network)
        for target in targets:
            await self._send_gossip(target, msg)

        self.discovery.gossip_count += 1

    async def _send_gossip(self, target: PeerInfo, msg: DiscoveryMessage):
        """Send gossip message to target peer."""
        # In real implementation, this would use the network layer
        # For now, just simulate
        pass

    async def receive_gossip(self, msg: DiscoveryMessage, sender_addr: str):
        """Handle received gossip message."""
        # Check if already seen
        if msg.nonce in self.seen_messages:
            return

        self._add_seen(msg.nonce)

        # Add sender as peer
        sender = PeerInfo(
            node_id=msg.sender_id,
            address=sender_addr,
            port=msg.sender_port,
            discovery_method=DiscoveryMethod.GOSSIP,
        )
        self.discovery.add_peer(sender)

        # Process included peers
        for peer_data in msg.peers:
            peer = PeerInfo.from_dict(peer_data)
            peer.hop_count += 1
            peer.discovery_method = DiscoveryMethod.GOSSIP
            self.discovery.add_peer(peer)

    def _add_seen(self, nonce: str):
        """Add message to seen set with eviction."""
        self.seen_messages.add(nonce)
        if len(self.seen_messages) > self.max_seen:
            # Evict random old entries
            to_remove = random.sample(list(self.seen_messages), self.max_seen // 2)
            self.seen_messages -= set(to_remove)


class BootstrapFreeDiscovery:
    """
    Complete bootstrap-free peer discovery system.

    Combines multiple discovery methods:
    1. Local multicast (same network)
    2. Gossip protocol (cross-network)
    3. Seed file persistence (rejoining network)
    4. Rolling introduction (learn from peers)

    No central bootstrap nodes required!

    Usage:
        discovery = BootstrapFreeDiscovery(node_id)
        await discovery.start()

        # Peers will be discovered automatically
        peers = discovery.get_peers()
    """

    def __init__(
        self,
        node_id: str,
        listen_port: int = DISCOVERY_PORT,
        enable_multicast: bool = True,
        enable_gossip: bool = True,
    ):
        """
        Initialize bootstrap-free discovery.

        Args:
            node_id: Unique node identifier
            listen_port: Port for discovery
            enable_multicast: Enable local network discovery
            enable_gossip: Enable gossip protocol
        """
        self.node_id = node_id

        # Core peer discovery
        self.peer_discovery = PeerDiscovery(
            node_id=node_id,
            listen_port=listen_port,
        )

        # Discovery methods
        self.multicast: Optional[MulticastDiscovery] = None
        self.gossip: Optional[GossipProtocol] = None

        if enable_multicast:
            self.multicast = MulticastDiscovery(self.peer_discovery)

        if enable_gossip:
            self.gossip = GossipProtocol(self.peer_discovery)

        # State
        self.running = False

        print(f"BootstrapFreeDiscovery initialized")
        print(f"  Multicast: {enable_multicast}")
        print(f"  Gossip: {enable_gossip}")

    async def start(self):
        """Start all discovery methods."""
        self.running = True

        if self.multicast:
            await self.multicast.start()

        if self.gossip:
            await self.gossip.start()

        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())

        print("Bootstrap-free discovery started")

    async def stop(self):
        """Stop all discovery methods."""
        self.running = False

        if self.multicast:
            await self.multicast.stop()

        if self.gossip:
            await self.gossip.stop()

        # Save seeds for next time
        self.peer_discovery._save_seeds()

        print("Bootstrap-free discovery stopped")

    async def _cleanup_loop(self):
        """Periodically clean up stale peers."""
        while self.running:
            self.peer_discovery.cleanup_stale()
            await asyncio.sleep(60)  # Every minute

    def add_seed_peer(self, address: str, port: int, node_id: Optional[str] = None):
        """
        Manually add a seed peer.

        Useful for initial network bootstrap or known reliable peers.
        """
        peer = PeerInfo(
            node_id=node_id or hashlib.md5(f"{address}:{port}".encode()).hexdigest()[:16],
            address=address,
            port=port,
            discovery_method=DiscoveryMethod.MANUAL,
        )
        self.peer_discovery.add_peer(peer)

    def get_peers(self) -> List[PeerInfo]:
        """Get all known peers."""
        return list(self.peer_discovery.peers.values())

    def get_active_peers(self) -> List[PeerInfo]:
        """Get non-stale peers."""
        return [p for p in self.peer_discovery.peers.values() if not p.is_stale]

    def get_peer_count(self) -> int:
        """Get number of known peers."""
        return len(self.peer_discovery.peers)

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        stats = self.peer_discovery.get_stats()
        stats['methods'] = {
            'multicast': self.multicast is not None,
            'gossip': self.gossip is not None,
        }
        return stats

    def set_peer_callback(
        self,
        on_discovered: Optional[Callable[[PeerInfo], None]] = None,
        on_lost: Optional[Callable[[str], None]] = None,
    ):
        """Set callbacks for peer events."""
        self.peer_discovery.on_peer_discovered = on_discovered
        self.peer_discovery.on_peer_lost = on_lost


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Bootstrap-Free Discovery Test")
    print("=" * 60)

    import asyncio

    async def test():
        # Create discovery
        discovery = BootstrapFreeDiscovery(
            node_id="test_node_001",
            enable_multicast=False,  # Disable for test (needs network)
            enable_gossip=True,
        )

        # Add some seed peers
        for i in range(5):
            discovery.add_seed_peer(
                address=f"192.168.1.{100 + i}",
                port=31337,
                node_id=f"seed_node_{i:03d}",
            )

        print(f"\nInitial peers: {discovery.get_peer_count()}")

        # Simulate peer discovery
        for peer in discovery.get_peers():
            print(f"  - {peer.node_id}: {peer.endpoint}")

        # Stats
        print(f"\nStats: {discovery.get_stats()}")

        # Test gossip
        if discovery.gossip:
            await discovery.gossip.gossip_round()
            print(f"Gossip count: {discovery.peer_discovery.gossip_count}")

    asyncio.run(test())

    print("\n" + "=" * 60)
    print("Bootstrap-Free Discovery module ready!")
    print("=" * 60)
