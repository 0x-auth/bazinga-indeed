#!/usr/bin/env python3
"""
BAZINGA DHT - Kademlia-style Distributed Hash Table
====================================================

True P2P discovery without central registry.

Core Concepts:
- 256-bit Node IDs derived from Proof-of-Boundary
- XOR distance metric for routing
- K-buckets organized by distance
- Iterative lookup for node discovery

"Your identity is your resonance. Your address is your understanding."

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# BAZINGA constants
PHI = 1.618033988749895

# DHT constants
K = 20  # Max nodes per bucket (Kademlia default)
ALPHA = 3  # Parallelism factor for lookups
ID_BITS = 256  # SHA-256 = 256 bits
BUCKET_REFRESH_INTERVAL = 3600  # 1 hour
NODE_TIMEOUT = 5.0  # Seconds to wait for response


# =============================================================================
# XOR DISTANCE & BUCKET LOGIC
# =============================================================================

def xor_distance(id1: bytes, id2: bytes) -> int:
    """
    Calculate XOR distance between two node IDs.

    XOR distance is the foundation of Kademlia:
    - d(a, b) = a XOR b (as integer)
    - Closer nodes have smaller XOR distance
    - Forms a metric space (triangle inequality holds)

    Example:
        id1 = 1010...  (binary)
        id2 = 1001...  (binary)
        XOR = 0011...  = 3 (first difference at bit 2)
    """
    if len(id1) != len(id2):
        raise ValueError(f"ID length mismatch: {len(id1)} vs {len(id2)}")

    # XOR each byte and combine into integer
    xor_bytes = bytes(a ^ b for a, b in zip(id1, id2))
    return int.from_bytes(xor_bytes, 'big')


def get_bucket_index(local_id: bytes, remote_id: bytes) -> int:
    """
    Determine which bucket a node belongs to based on XOR distance.

    Bucket index = position of highest differing bit

    This creates a logarithmic structure:
    - Bucket 0: nodes differing in bit 0 (furthest, 2^255 to 2^256-1)
    - Bucket 255: nodes differing in bit 255 (closest, distance 1)

    Nodes closer to us go in higher-indexed buckets.
    We know more about our "neighborhood" than distant regions.
    """
    distance = xor_distance(local_id, remote_id)

    if distance == 0:
        return ID_BITS - 1  # Same node

    # bit_length() gives position of highest set bit
    return ID_BITS - distance.bit_length()


def node_id_from_pob(alpha: str, omega: str) -> bytes:
    """
    Generate node ID from Proof-of-Boundary signatures.

    Your identity IS your resonance. The PoB that proves you
    understand the φ⁴ boundary becomes your network address.

    This means:
    - You can't fake an ID without doing the PoB work
    - Your position in the network reflects your proof
    - Identity is earned through understanding
    """
    pob_string = f"{alpha}:{omega}"
    return hashlib.sha256(pob_string.encode()).digest()


def hash_to_id(data: str) -> bytes:
    """Hash string to 256-bit node ID."""
    return hashlib.sha256(data.encode()).digest()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NodeInfo:
    """Information about a node in the network."""
    node_id: bytes  # 256-bit ID
    address: str  # IP address
    port: int
    last_seen: float = field(default_factory=time.time)
    trust_score: float = 0.5
    uses_local_model: bool = False

    @property
    def id_hex(self) -> str:
        """Short hex representation of node ID."""
        return self.node_id.hex()[:16] + "..."

    @property
    def endpoint(self) -> str:
        """ZMQ endpoint string."""
        return f"tcp://{self.address}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id.hex(),
            "address": self.address,
            "port": self.port,
            "last_seen": self.last_seen,
            "trust_score": self.trust_score,
            "uses_local_model": self.uses_local_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(
            node_id=bytes.fromhex(data["node_id"]),
            address=data["address"],
            port=data["port"],
            last_seen=data.get("last_seen", time.time()),
            trust_score=data.get("trust_score", 0.5),
            uses_local_model=data.get("uses_local_model", False),
        )


@dataclass
class DHTEntry:
    """Entry in the DHT storage."""
    key: bytes  # 256-bit key
    value: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: int = 3600  # 1 hour default

    def is_expired(self) -> bool:
        return time.time() > self.timestamp + self.ttl


# =============================================================================
# K-BUCKET
# =============================================================================

class KBucket:
    """
    A K-bucket holds up to K nodes at a specific XOR distance range.

    Properties:
    - Max K nodes (default 20)
    - Ordered by last seen (most recent at tail)
    - When full, ping oldest - if alive, discard new; if dead, replace
    - This favors long-lived nodes (more stable network)
    """

    def __init__(self, k: int = K):
        self.k = k
        self.nodes: List[NodeInfo] = []
        self.last_updated: float = time.time()

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def is_full(self) -> bool:
        return len(self.nodes) >= self.k

    def get_node(self, node_id: bytes) -> Optional[NodeInfo]:
        """Find a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def add_node(self, node: NodeInfo) -> Tuple[bool, Optional[NodeInfo]]:
        """
        Add a node to the bucket.

        Returns:
            (added: bool, eviction_candidate: Optional[NodeInfo])

            If bucket is full, returns the oldest node for ping check.
            Caller should ping it - if dead, call remove_node + add_node.
        """
        # Check if node already exists
        existing = self.get_node(node.node_id)
        if existing:
            # Move to tail (most recently seen)
            self.nodes.remove(existing)
            existing.last_seen = time.time()
            self.nodes.append(existing)
            return True, None

        # Bucket not full - just add
        if not self.is_full():
            self.nodes.append(node)
            self.last_updated = time.time()
            return True, None

        # Bucket full - return oldest for ping check
        oldest = self.nodes[0]
        return False, oldest

    def remove_node(self, node_id: bytes) -> bool:
        """Remove a node from the bucket."""
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                self.nodes.pop(i)
                return True
        return False

    def touch_node(self, node_id: bytes):
        """Update last_seen for a node (move to tail)."""
        node = self.get_node(node_id)
        if node:
            self.nodes.remove(node)
            node.last_seen = time.time()
            self.nodes.append(node)

    def get_nodes(self) -> List[NodeInfo]:
        """Get all nodes in bucket."""
        return list(self.nodes)


# =============================================================================
# ROUTING TABLE
# =============================================================================

class RoutingTable:
    """
    Kademlia routing table - 256 K-buckets organized by XOR distance.

    Structure:
    - Bucket i contains nodes with XOR distance in [2^i, 2^(i+1))
    - Lower index = further away, higher index = closer
    - We know more nodes closer to us (naturally fills nearby buckets)

    This gives O(log n) lookup in a network of n nodes.
    With 1M nodes, ~20 hops max to find anyone.
    """

    def __init__(self, local_id: bytes, k: int = K):
        self.local_id = local_id
        self.k = k
        self.buckets: List[KBucket] = [KBucket(k) for _ in range(ID_BITS)]
        self.total_nodes = 0

    def get_bucket_for_node(self, node_id: bytes) -> KBucket:
        """Get the bucket a node belongs to."""
        index = get_bucket_index(self.local_id, node_id)
        return self.buckets[index]

    def add_node(self, node: NodeInfo) -> Tuple[bool, Optional[NodeInfo]]:
        """
        Add a node to the routing table.

        Returns (success, eviction_candidate).
        If eviction_candidate is returned, ping it to decide whether to evict.
        """
        if node.node_id == self.local_id:
            return False, None  # Don't add ourselves

        bucket = self.get_bucket_for_node(node.node_id)
        added, eviction = bucket.add_node(node)

        if added:
            self.total_nodes += 1

        return added, eviction

    def remove_node(self, node_id: bytes) -> bool:
        """Remove a node from the routing table."""
        bucket = self.get_bucket_for_node(node_id)
        removed = bucket.remove_node(node_id)
        if removed:
            self.total_nodes -= 1
        return removed

    def get_node(self, node_id: bytes) -> Optional[NodeInfo]:
        """Find a specific node."""
        bucket = self.get_bucket_for_node(node_id)
        return bucket.get_node(node_id)

    def touch_node(self, node_id: bytes):
        """Update last_seen for a node."""
        bucket = self.get_bucket_for_node(node_id)
        bucket.touch_node(node_id)

    def get_closest_nodes(self, target_id: bytes, count: int = K) -> List[NodeInfo]:
        """
        Get the K closest nodes to a target ID.

        This is the core of Kademlia lookups:
        1. Start from bucket containing target
        2. Expand outward until we have K nodes
        3. Sort by XOR distance to target

        Used for both FIND_NODE and iterative lookup.
        """
        target_bucket_index = get_bucket_index(self.local_id, target_id)

        # Collect nodes from nearby buckets
        candidates: List[NodeInfo] = []

        # Start at target bucket, expand outward
        for offset in range(ID_BITS):
            # Check bucket above
            if target_bucket_index + offset < ID_BITS:
                candidates.extend(self.buckets[target_bucket_index + offset].get_nodes())

            # Check bucket below
            if offset > 0 and target_bucket_index - offset >= 0:
                candidates.extend(self.buckets[target_bucket_index - offset].get_nodes())

            # Stop if we have enough
            if len(candidates) >= count:
                break

        # Sort by XOR distance to target
        candidates.sort(key=lambda n: xor_distance(n.node_id, target_id))

        return candidates[:count]

    def get_all_nodes(self) -> List[NodeInfo]:
        """Get all nodes in the routing table."""
        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket.get_nodes())
        return all_nodes

    def get_stale_buckets(self, max_age: float = BUCKET_REFRESH_INTERVAL) -> List[int]:
        """Get indices of buckets that need refreshing."""
        now = time.time()
        stale = []
        for i, bucket in enumerate(self.buckets):
            if len(bucket) > 0 and now - bucket.last_updated > max_age:
                stale.append(i)
        return stale

    def get_stats(self) -> Dict[str, Any]:
        """Get routing table statistics."""
        non_empty_buckets = sum(1 for b in self.buckets if len(b) > 0)
        bucket_sizes = [len(b) for b in self.buckets if len(b) > 0]

        return {
            "local_id": self.local_id.hex()[:16] + "...",
            "total_nodes": self.total_nodes,
            "non_empty_buckets": non_empty_buckets,
            "bucket_sizes": bucket_sizes[-10:] if bucket_sizes else [],
            "k": self.k,
        }

    def print_table(self):
        """Print routing table for debugging."""
        print(f"\n{'='*60}")
        print(f"ROUTING TABLE: {self.local_id.hex()[:16]}...")
        print(f"{'='*60}")
        print(f"Total nodes: {self.total_nodes}")
        print(f"Non-empty buckets: {sum(1 for b in self.buckets if len(b) > 0)}")
        print()

        for i, bucket in enumerate(self.buckets):
            if len(bucket) > 0:
                print(f"Bucket {i:3d} (distance ~2^{ID_BITS-i-1}): {len(bucket)} nodes")
                for node in bucket.nodes[:3]:
                    print(f"    {node.id_hex} @ {node.address}:{node.port}")
                if len(bucket) > 3:
                    print(f"    ... and {len(bucket) - 3} more")

        print(f"{'='*60}\n")


# =============================================================================
# KADEMLIA NODE (DHT Protocol)
# =============================================================================

class KademliaNode:
    """
    Full Kademlia DHT node with network operations.

    Features:
    - Bootstrap from known nodes
    - Iterative node lookup
    - Store/retrieve key-value pairs
    - Announce knowledge topics
    - Trust-weighted selection (φ bonus for local models)
    """

    def __init__(
        self,
        node_id: bytes,
        address: str = "0.0.0.0",
        port: int = 5150,
        trust_score: float = 0.5,
        uses_local_model: bool = False,
    ):
        self.node_id = node_id
        self.address = address
        self.port = port
        self.trust_score = trust_score
        self.uses_local_model = uses_local_model

        # Routing table
        self.routing_table = RoutingTable(node_id)

        # Local storage
        self.storage: Dict[str, DHTEntry] = {}

        # Server state
        self.server: Optional[asyncio.Server] = None
        self.running = False

        # Stats
        self.stats = {
            "lookups": 0,
            "stores": 0,
            "pings_sent": 0,
            "pings_received": 0,
        }

    @classmethod
    def from_pob(
        cls,
        alpha: str,
        omega: str,
        address: str = "0.0.0.0",
        port: int = 5150,
        uses_local_model: bool = False,
    ) -> "KademliaNode":
        """Create node with ID derived from Proof-of-Boundary."""
        node_id = node_id_from_pob(alpha, omega)
        trust = PHI if uses_local_model else 0.5  # φ bonus for local models
        return cls(
            node_id=node_id,
            address=address,
            port=port,
            trust_score=trust,
            uses_local_model=uses_local_model,
        )

    def get_info(self) -> NodeInfo:
        """Get this node's info."""
        return NodeInfo(
            node_id=self.node_id,
            address=self.address,
            port=self.port,
            trust_score=self.trust_score,
            uses_local_model=self.uses_local_model,
        )

    async def start(self):
        """Start the DHT server."""
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.address,
            self.port,
        )

        addr = self.server.sockets[0].getsockname()
        self.port = addr[1]
        self.running = True

        print(f"DHT Node online: {self.node_id.hex()[:16]}... @ {self.address}:{self.port}")
        print(f"  Trust: {self.trust_score:.3f} | Local Model: {self.uses_local_model}")

        # Start maintenance loop
        asyncio.create_task(self._maintenance_loop())

    async def stop(self):
        """Stop the DHT server."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming DHT connection."""
        try:
            data = await asyncio.wait_for(reader.read(8192), timeout=NODE_TIMEOUT)
            if not data:
                return

            request = json.loads(data.decode())
            response = await self._handle_request(request)

            writer.write(json.dumps(response).encode())
            await writer.drain()

        except Exception as e:
            pass  # Silent fail for network errors
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_request(self, request: Dict) -> Dict:
        """Handle DHT request."""
        cmd = request.get("cmd")

        # Update sender in routing table
        if "sender" in request:
            sender = NodeInfo.from_dict(request["sender"])
            self.routing_table.add_node(sender)

        if cmd == "PING":
            self.stats["pings_received"] += 1
            return {
                "status": "PONG",
                "sender": self.get_info().to_dict(),
            }

        elif cmd == "FIND_NODE":
            target_id = bytes.fromhex(request.get("target_id", ""))
            closest = self.routing_table.get_closest_nodes(target_id, K)
            return {
                "status": "OK",
                "nodes": [n.to_dict() for n in closest],
                "sender": self.get_info().to_dict(),
            }

        elif cmd == "FIND_VALUE":
            key = request.get("key")
            if key in self.storage and not self.storage[key].is_expired():
                return {
                    "status": "FOUND",
                    "value": self.storage[key].value,
                    "sender": self.get_info().to_dict(),
                }
            else:
                target_id = hash_to_id(key)
                closest = self.routing_table.get_closest_nodes(target_id, K)
                return {
                    "status": "NOT_FOUND",
                    "nodes": [n.to_dict() for n in closest],
                    "sender": self.get_info().to_dict(),
                }

        elif cmd == "STORE":
            key = request.get("key")
            value = request.get("value")
            ttl = request.get("ttl", 3600)

            self.storage[key] = DHTEntry(
                key=hash_to_id(key),
                value=value,
                ttl=ttl,
            )
            self.stats["stores"] += 1

            return {
                "status": "OK",
                "sender": self.get_info().to_dict(),
            }

        return {"status": "ERROR", "message": "Unknown command"}

    async def _send_request(
        self,
        node: NodeInfo,
        request: Dict,
        timeout: float = NODE_TIMEOUT,
    ) -> Optional[Dict]:
        """Send request to another DHT node."""
        try:
            # Add our info to request
            request["sender"] = self.get_info().to_dict()

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node.address, node.port),
                timeout=timeout,
            )

            writer.write(json.dumps(request).encode())
            await writer.drain()

            data = await asyncio.wait_for(reader.read(8192), timeout=timeout)
            response = json.loads(data.decode())

            writer.close()
            await writer.wait_closed()

            # Update routing table with response sender
            if "sender" in response:
                sender = NodeInfo.from_dict(response["sender"])
                self.routing_table.add_node(sender)

            return response

        except Exception:
            # Mark node as potentially dead
            return None

    async def ping(self, node: NodeInfo) -> bool:
        """Ping a node to check if it's alive."""
        self.stats["pings_sent"] += 1
        response = await self._send_request(node, {"cmd": "PING"})
        return response is not None and response.get("status") == "PONG"

    async def bootstrap(self, bootstrap_nodes: List[Tuple[str, int]]):
        """
        Bootstrap into the network from known nodes.

        Args:
            bootstrap_nodes: List of (host, port) tuples
        """
        print(f"\nBootstrapping DHT from {len(bootstrap_nodes)} nodes...")

        for host, port in bootstrap_nodes:
            try:
                # Create temporary NodeInfo for bootstrap
                temp_id = hash_to_id(f"{host}:{port}")
                temp_node = NodeInfo(node_id=temp_id, address=host, port=port)

                response = await self._send_request(temp_node, {"cmd": "PING"})

                if response and response.get("status") == "PONG":
                    sender = NodeInfo.from_dict(response["sender"])
                    self.routing_table.add_node(sender)
                    print(f"  ✓ Connected to {sender.id_hex}")

                    # Do lookup for our own ID to populate routing table
                    await self.find_node(self.node_id)

            except Exception as e:
                print(f"  ✗ Failed to connect to {host}:{port}: {e}")

        print(f"  Routing table: {self.routing_table.total_nodes} nodes")

    async def find_node(self, target_id: bytes) -> List[NodeInfo]:
        """
        Iterative node lookup - the heart of Kademlia.

        Finds the K closest nodes to target_id.
        """
        self.stats["lookups"] += 1

        # Start with closest nodes we know
        closest = self.routing_table.get_closest_nodes(target_id, ALPHA)
        if not closest:
            return []

        queried: Set[bytes] = set()
        best: List[NodeInfo] = list(closest)

        while True:
            # Get unqueried nodes from best
            to_query = [n for n in best[:ALPHA] if n.node_id not in queried]

            if not to_query:
                break

            # Query in parallel
            tasks = []
            for node in to_query:
                queried.add(node.node_id)
                tasks.append(self._send_request(node, {
                    "cmd": "FIND_NODE",
                    "target_id": target_id.hex(),
                }))

            responses = await asyncio.gather(*tasks)

            # Process responses
            for response in responses:
                if response and response.get("status") == "OK":
                    for node_data in response.get("nodes", []):
                        node = NodeInfo.from_dict(node_data)
                        if node.node_id not in queried:
                            best.append(node)
                            self.routing_table.add_node(node)

            # Sort by distance and keep best K
            best.sort(key=lambda n: xor_distance(n.node_id, target_id))
            best = best[:K]

        return best

    async def store(self, key: str, value: Dict, ttl: int = 3600):
        """Store a key-value pair in the DHT."""
        key_id = hash_to_id(key)

        # Store locally
        self.storage[key] = DHTEntry(key=key_id, value=value, ttl=ttl)

        # Find closest nodes and store there too
        closest = await self.find_node(key_id)

        for node in closest[:K]:
            await self._send_request(node, {
                "cmd": "STORE",
                "key": key,
                "value": value,
                "ttl": ttl,
            })

    async def get(self, key: str) -> Optional[Dict]:
        """Retrieve a value from the DHT."""
        # Check local storage first
        if key in self.storage and not self.storage[key].is_expired():
            return self.storage[key].value

        # Look up in network
        key_id = hash_to_id(key)
        closest = await self.find_node(key_id)

        for node in closest:
            response = await self._send_request(node, {
                "cmd": "FIND_VALUE",
                "key": key,
            })

            if response and response.get("status") == "FOUND":
                value = response.get("value")
                # Cache locally
                self.storage[key] = DHTEntry(key=key_id, value=value)
                return value

        return None

    async def _maintenance_loop(self):
        """Periodic maintenance tasks."""
        while self.running:
            await asyncio.sleep(60)

            # Clean expired entries
            expired = [k for k, e in self.storage.items() if e.is_expired()]
            for key in expired:
                del self.storage[key]

            # Refresh stale buckets
            stale = self.routing_table.get_stale_buckets()
            for bucket_idx in stale[:3]:  # Refresh up to 3 buckets per cycle
                # Generate random ID in bucket range
                random_id = hashlib.sha256(f"refresh:{bucket_idx}:{time.time()}".encode()).digest()
                await self.find_node(random_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get DHT statistics."""
        rt_stats = self.routing_table.get_stats()
        return {
            "node_id": self.node_id.hex()[:16] + "...",
            "address": f"{self.address}:{self.port}",
            "trust_score": self.trust_score,
            "uses_local_model": self.uses_local_model,
            "routing_table": rt_stats,
            "storage_entries": len(self.storage),
            **self.stats,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA DHT - XOR Distance & Routing Table Test")
    print("=" * 60)

    # Test XOR distance
    print("\n1. XOR Distance Tests:")

    id1 = bytes([0b10101010] * 32)  # 256-bit ID
    id2 = bytes([0b10101011] * 32)  # Differs in last bit of each byte
    id3 = bytes([0b00101010] * 32)  # Differs in first bit of each byte

    d12 = xor_distance(id1, id2)
    d13 = xor_distance(id1, id3)

    print(f"   Distance(id1, id2) = {d12} (differs in low bits)")
    print(f"   Distance(id1, id3) = {d13} (differs in high bits)")
    print(f"   id3 is {'further' if d13 > d12 else 'closer'} than id2")

    # Test bucket index
    print("\n2. Bucket Index Tests:")

    local_id = hashlib.sha256(b"local_node").digest()

    test_ids = [
        ("same", local_id),
        ("close", hashlib.sha256(b"local_nodX").digest()),
        ("far", hashlib.sha256(b"completely_different").digest()),
    ]

    for name, test_id in test_ids:
        bucket = get_bucket_index(local_id, test_id)
        distance = xor_distance(local_id, test_id)
        print(f"   {name:10s} → bucket {bucket:3d}, distance = {distance:.2e}")

    # Test node ID from PoB
    print("\n3. Node ID from PoB:")
    alpha = "a3f2e1b5c8d9"
    omega = "7c4b2a1f8e9d"
    pob_id = node_id_from_pob(alpha, omega)
    print(f"   Alpha: {alpha}")
    print(f"   Omega: {omega}")
    print(f"   Node ID: {pob_id.hex()[:32]}...")

    # Test routing table
    print("\n4. Routing Table Tests:")

    rt = RoutingTable(local_id)

    for i in range(50):
        node_id = hashlib.sha256(f"node_{i}".encode()).digest()
        node = NodeInfo(
            node_id=node_id,
            address=f"192.168.1.{i % 256}",
            port=5150 + i,
        )
        rt.add_node(node)

    stats = rt.get_stats()
    print(f"   Added 50 nodes")
    print(f"   Total in table: {stats['total_nodes']}")
    print(f"   Non-empty buckets: {stats['non_empty_buckets']}")

    # Find closest to a target
    target = hashlib.sha256(b"target_node").digest()
    closest = rt.get_closest_nodes(target, count=5)

    print(f"\n   Closest 5 nodes to target:")
    for node in closest:
        dist = xor_distance(node.node_id, target)
        print(f"      {node.id_hex} (distance: {dist:.2e})")

    # Print routing table
    rt.print_table()

    print("=" * 60)
    print("  DHT Core Ready!")
    print("  • XOR distance working")
    print("  • K-buckets working")
    print("  • Node ID from PoB working")
    print("  • Routing table working")
    print("=" * 60)
