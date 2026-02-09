#!/usr/bin/env python3
"""
BAZINGA DHT - Distributed Hash Table for Node Discovery

Implements Kademlia-inspired DHT for:
- Finding nodes without central directory
- Announcing knowledge topics
- Discovering experts on specific topics

"Find knowledge, not servers."
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# DHT constants
K_BUCKET_SIZE = 20  # Max peers per bucket
ALPHA_CONCURRENCY = 3  # Concurrent lookups
KEY_BITS = 160  # SHA1 key space


def hash_key(data: str) -> int:
    """Hash string to DHT key space."""
    return int(hashlib.sha1(data.encode()).hexdigest(), 16)


def hash_embedding(embedding: List[float]) -> int:
    """Hash embedding vector to DHT key space."""
    # Create string representation
    embedding_str = ",".join(f"{x:.6f}" for x in embedding[:10])  # First 10 dims
    return hash_key(embedding_str)


def xor_distance(key1: int, key2: int) -> int:
    """Calculate XOR distance between two keys."""
    return key1 ^ key2


def bucket_index(node_key: int, target_key: int) -> int:
    """Find bucket index for target key relative to node."""
    distance = xor_distance(node_key, target_key)
    if distance == 0:
        return 0
    return distance.bit_length() - 1


@dataclass
class DHTEntry:
    """Entry in the DHT."""
    key: int
    value: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: int = 3600  # 1 hour default TTL

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.timestamp + self.ttl


@dataclass
class DHTNode:
    """Reference to another DHT node."""
    node_id: str
    key: int
    host: str
    port: int
    tau: float
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "key": self.key,
            "host": self.host,
            "port": self.port,
            "tau": self.tau,
        }


class KBucket:
    """K-bucket for storing nodes at a particular distance range."""

    def __init__(self, max_size: int = K_BUCKET_SIZE):
        self.max_size = max_size
        self.nodes: List[DHTNode] = []

    def add(self, node: DHTNode) -> bool:
        """
        Add node to bucket.
        Returns True if added, False if bucket full and node not added.
        """
        # Check if node already exists
        for i, existing in enumerate(self.nodes):
            if existing.node_id == node.node_id:
                # Move to end (most recently seen)
                self.nodes.pop(i)
                self.nodes.append(node)
                return True

        # Add if space available
        if len(self.nodes) < self.max_size:
            self.nodes.append(node)
            return True

        # Bucket full - check if oldest node is stale
        oldest = self.nodes[0]
        if (datetime.now() - oldest.last_seen) > timedelta(minutes=5):
            # Replace stale node
            self.nodes.pop(0)
            self.nodes.append(node)
            return True

        return False

    def get_closest(self, key: int, count: int = K_BUCKET_SIZE) -> List[DHTNode]:
        """Get closest nodes to a key."""
        sorted_nodes = sorted(
            self.nodes,
            key=lambda n: xor_distance(n.key, key)
        )
        return sorted_nodes[:count]

    def remove(self, node_id: str):
        """Remove node from bucket."""
        self.nodes = [n for n in self.nodes if n.node_id != node_id]

    def __len__(self):
        return len(self.nodes)


class BAZINGA_DHT:
    """
    BAZINGA Distributed Hash Table.

    Features:
    - Kademlia-inspired routing
    - Knowledge topic discovery
    - Trust-weighted node selection
    - α-SEED anchor support

    Usage:
        dht = BAZINGA_DHT(node_id, host, port)
        await dht.start()
        await dht.bootstrap(["host1:port1", "host2:port2"])

        # Announce knowledge
        await dht.announce_knowledge("quantum mechanics", {"tau": 0.9, ...})

        # Find experts
        experts = await dht.find_experts("quantum mechanics")
    """

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 8469,
        tau: float = 0.5,
    ):
        """
        Initialize DHT node.

        Args:
            node_id: Unique node identifier
            host: Host to bind to
            port: Port for DHT operations
            tau: Node's trust score
        """
        self.node_id = node_id
        self.node_key = hash_key(node_id)
        self.host = host
        self.port = port
        self.tau = tau

        # Routing table: K-buckets indexed by distance
        self.buckets: List[KBucket] = [KBucket() for _ in range(KEY_BITS)]

        # Local storage
        self.storage: Dict[int, DHTEntry] = {}

        # Server
        self.server: Optional[asyncio.Server] = None
        self.running = False

        # Stats
        self.stats = {
            'lookups': 0,
            'stores': 0,
            'announcements': 0,
        }

    async def start(self):
        """Start DHT node."""
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        addr = self.server.sockets[0].getsockname()
        self.port = addr[1]
        self.running = True

        print(f"📡 DHT Node online at {self.host}:{self.port}")

        # Start maintenance
        asyncio.create_task(self._maintenance_loop())

    async def stop(self):
        """Stop DHT node."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming DHT request."""
        try:
            data = await reader.read(4096)
            if not data:
                return

            request = json.loads(data.decode())
            response = await self._handle_request(request)

            writer.write(json.dumps(response).encode())
            await writer.drain()

        except Exception as e:
            print(f"DHT request error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_request(self, request: Dict) -> Dict:
        """Handle DHT request."""
        cmd = request.get("cmd")

        if cmd == "PING":
            return {"status": "PONG", "node_id": self.node_id, "tau": self.tau}

        elif cmd == "FIND_NODE":
            key = request.get("key")
            nodes = self._find_closest_nodes(key)
            return {
                "status": "OK",
                "nodes": [n.to_dict() for n in nodes]
            }

        elif cmd == "FIND_VALUE":
            key = request.get("key")
            if key in self.storage and not self.storage[key].is_expired():
                return {
                    "status": "OK",
                    "value": self.storage[key].value
                }
            else:
                nodes = self._find_closest_nodes(key)
                return {
                    "status": "NOT_FOUND",
                    "nodes": [n.to_dict() for n in nodes]
                }

        elif cmd == "STORE":
            key = request.get("key")
            value = request.get("value")
            ttl = request.get("ttl", 3600)

            self.storage[key] = DHTEntry(key=key, value=value, ttl=ttl)
            self.stats['stores'] += 1

            return {"status": "OK"}

        elif cmd == "ANNOUNCE":
            topic_hash = request.get("topic_hash")
            node_info = request.get("node_info")

            # Store announcement
            self.storage[topic_hash] = DHTEntry(
                key=topic_hash,
                value=node_info,
                ttl=1800,  # 30 min TTL for announcements
            )
            self.stats['announcements'] += 1

            return {"status": "OK"}

        return {"status": "ERROR", "message": "Unknown command"}

    def _find_closest_nodes(self, key: int, count: int = K_BUCKET_SIZE) -> List[DHTNode]:
        """Find closest nodes to a key."""
        candidates = []

        for bucket in self.buckets:
            candidates.extend(bucket.nodes)

        # Sort by XOR distance
        candidates.sort(key=lambda n: xor_distance(n.key, key))

        return candidates[:count]

    def add_node(self, node: DHTNode):
        """Add node to routing table."""
        bucket_idx = bucket_index(self.node_key, node.key)
        bucket_idx = min(bucket_idx, KEY_BITS - 1)
        self.buckets[bucket_idx].add(node)

    async def bootstrap(self, bootstrap_addrs: List[str]):
        """
        Bootstrap from known nodes.

        Args:
            bootstrap_addrs: List of "host:port" addresses
        """
        for addr in bootstrap_addrs:
            try:
                host, port = addr.split(":")
                response = await self._send_request(host, int(port), {
                    "cmd": "PING"
                })

                if response and response.get("status") == "PONG":
                    node = DHTNode(
                        node_id=response["node_id"],
                        key=hash_key(response["node_id"]),
                        host=host,
                        port=int(port),
                        tau=response.get("tau", 0.5),
                    )
                    self.add_node(node)
                    print(f"   ✓ DHT bootstrap: {response['node_id'][:8]}")

                    # Do iterative lookup for our own key to populate table
                    await self._iterative_find_node(self.node_key)

            except Exception as e:
                print(f"   ✗ DHT bootstrap failed for {addr}: {e}")

    async def _send_request(
        self,
        host: str,
        port: int,
        request: Dict,
        timeout: float = 5.0,
    ) -> Optional[Dict]:
        """Send request to DHT node."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout,
            )

            writer.write(json.dumps(request).encode())
            await writer.drain()

            data = await asyncio.wait_for(reader.read(4096), timeout=timeout)
            response = json.loads(data.decode())

            writer.close()
            await writer.wait_closed()

            return response

        except Exception:
            return None

    async def _iterative_find_node(self, key: int) -> List[DHTNode]:
        """Iterative node lookup."""
        self.stats['lookups'] += 1

        # Start with closest known nodes
        closest = self._find_closest_nodes(key, ALPHA_CONCURRENCY)
        queried = set()
        best = list(closest)

        while True:
            # Query unqueried nodes
            to_query = [n for n in best[:ALPHA_CONCURRENCY] if n.node_id not in queried]

            if not to_query:
                break

            tasks = []
            for node in to_query:
                queried.add(node.node_id)
                tasks.append(self._send_request(node.host, node.port, {
                    "cmd": "FIND_NODE",
                    "key": key,
                }))

            responses = await asyncio.gather(*tasks)

            for response in responses:
                if response and response.get("status") == "OK":
                    for node_data in response.get("nodes", []):
                        node = DHTNode(
                            node_id=node_data["node_id"],
                            key=node_data["key"],
                            host=node_data["host"],
                            port=node_data["port"],
                            tau=node_data.get("tau", 0.5),
                        )
                        self.add_node(node)
                        best.append(node)

            # Sort by distance
            best.sort(key=lambda n: xor_distance(n.key, key))
            best = best[:K_BUCKET_SIZE]

        return best

    async def get(self, key: str) -> Optional[Dict]:
        """
        Get value from DHT.

        Args:
            key: String key to lookup

        Returns:
            Value if found, None otherwise
        """
        key_hash = hash_key(key)

        # Check local storage first
        if key_hash in self.storage and not self.storage[key_hash].is_expired():
            return self.storage[key_hash].value

        # Iterative lookup
        closest = await self._iterative_find_node(key_hash)

        for node in closest:
            response = await self._send_request(node.host, node.port, {
                "cmd": "FIND_VALUE",
                "key": key_hash,
            })

            if response and response.get("status") == "OK":
                value = response.get("value")
                # Cache locally
                self.storage[key_hash] = DHTEntry(key=key_hash, value=value)
                return value

        return None

    async def set(self, key: str, value: Dict, ttl: int = 3600):
        """
        Store value in DHT.

        Args:
            key: String key
            value: Value to store
            ttl: Time-to-live in seconds
        """
        key_hash = hash_key(key)

        # Store locally
        self.storage[key_hash] = DHTEntry(key=key_hash, value=value, ttl=ttl)

        # Store on closest nodes
        closest = await self._iterative_find_node(key_hash)

        for node in closest[:K_BUCKET_SIZE]:
            await self._send_request(node.host, node.port, {
                "cmd": "STORE",
                "key": key_hash,
                "value": value,
                "ttl": ttl,
            })

    async def announce_knowledge(self, topic: str, node_info: Dict):
        """
        Announce that this node has knowledge about a topic.

        Args:
            topic: Topic string (e.g., "quantum mechanics")
            node_info: Node information including tau, address, etc.
        """
        topic_hash = hash_key(topic)

        # Ensure our info is included
        node_info["node_id"] = self.node_id
        node_info["tau"] = self.tau
        node_info["host"] = self.host if self.host != "0.0.0.0" else "127.0.0.1"
        node_info["port"] = self.port

        # Store locally
        existing = self.storage.get(topic_hash)
        if existing and not existing.is_expired():
            # Merge with existing announcements
            if isinstance(existing.value, list):
                # Add to list if not already there
                existing.value = [
                    n for n in existing.value
                    if n.get("node_id") != self.node_id
                ]
                existing.value.append(node_info)
            else:
                existing.value = [existing.value, node_info]
        else:
            self.storage[topic_hash] = DHTEntry(
                key=topic_hash,
                value=[node_info],
                ttl=1800,
            )

        # Announce to closest nodes
        closest = await self._iterative_find_node(topic_hash)

        for node in closest[:ALPHA_CONCURRENCY]:
            await self._send_request(node.host, node.port, {
                "cmd": "ANNOUNCE",
                "topic_hash": topic_hash,
                "node_info": node_info,
            })

        self.stats['announcements'] += 1

    async def find_experts(self, topic: str) -> List[Dict]:
        """
        Find nodes that have knowledge about a topic.

        Args:
            topic: Topic string

        Returns:
            List of node info dicts sorted by trust score
        """
        topic_hash = hash_key(topic)

        # Check local storage
        experts = []
        if topic_hash in self.storage and not self.storage[topic_hash].is_expired():
            value = self.storage[topic_hash].value
            if isinstance(value, list):
                experts.extend(value)
            else:
                experts.append(value)

        # Query network
        closest = await self._iterative_find_node(topic_hash)

        for node in closest[:ALPHA_CONCURRENCY]:
            response = await self._send_request(node.host, node.port, {
                "cmd": "FIND_VALUE",
                "key": topic_hash,
            })

            if response and response.get("status") == "OK":
                value = response.get("value")
                if isinstance(value, list):
                    experts.extend(value)
                elif value:
                    experts.append(value)

        # Deduplicate by node_id
        seen = set()
        unique_experts = []
        for expert in experts:
            node_id = expert.get("node_id")
            if node_id and node_id not in seen:
                seen.add(node_id)
                unique_experts.append(expert)

        # Sort by trust score (highest first)
        unique_experts.sort(key=lambda x: x.get("tau", 0), reverse=True)

        return unique_experts

    async def _maintenance_loop(self):
        """Periodic maintenance tasks."""
        while self.running:
            await asyncio.sleep(60)

            # Clean expired entries
            expired = [
                key for key, entry in self.storage.items()
                if entry.is_expired()
            ]
            for key in expired:
                del self.storage[key]

            # Refresh buckets (simplified)
            # In full Kademlia, would do iterative lookups for bucket ranges

    def get_stats(self) -> Dict[str, Any]:
        """Get DHT statistics."""
        total_nodes = sum(len(bucket) for bucket in self.buckets)
        return {
            "node_id": self.node_id,
            "node_key": hex(self.node_key)[:16],
            "address": f"{self.host}:{self.port}",
            "routing_table_size": total_nodes,
            "storage_entries": len(self.storage),
            **self.stats,
        }


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA DHT Test")
        print("=" * 60)

        dht = BAZINGA_DHT("test-node-1", port=0, tau=0.8)

        print(f"\nNode ID: {dht.node_id}")
        print(f"Node Key: {hex(dht.node_key)[:16]}")

        # Test hashing
        topic_hash = hash_key("quantum mechanics")
        print(f"\nTopic 'quantum mechanics' hash: {hex(topic_hash)[:16]}")

        print("\nDHT created successfully!")
        print(f"Stats: {dht.get_stats()}")

    asyncio.run(test())
