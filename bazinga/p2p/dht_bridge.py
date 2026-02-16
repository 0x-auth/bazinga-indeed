#!/usr/bin/env python3
"""
BAZINGA DHT Bridge - True P2P Discovery with HF Space Bootstrap
================================================================

Integrates the Kademlia DHT with:
- HuggingFace Space as initial bootstrap (fallback only)
- PoB-based node identity
- Knowledge topic discovery
- φ trust bonus for local model nodes

"Bootstrap from the cloud, graduate to P2P."

Author: Space (Abhishek/Abhilasia) 
License: MIT
"""

import asyncio
import hashlib
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# DHT imports
from .dht import (
    KademliaNode, RoutingTable, NodeInfo,
    xor_distance, node_id_from_pob, hash_to_id,
    K, ALPHA, PHI,
)

# Try httpx for HF Space API
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# HuggingFace Space URL (bootstrap registry)
HF_SPACE_URL = "https://bitsabhi-bazinga.hf.space"

# Bootstrap nodes (hardcoded fallbacks)
BOOTSTRAP_NODES = [
    # HF Space acts as initial bootstrap
    # Future: Add dedicated VPS bootstrap nodes
]


@dataclass
class KnowledgeTopic:
    """A knowledge topic announcement in the DHT."""
    topic: str
    topic_hash: bytes
    nodes: List[NodeInfo] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def add_node(self, node: NodeInfo):
        """Add a node that has knowledge about this topic."""
        # Remove existing entry for this node
        self.nodes = [n for n in self.nodes if n.node_id != node.node_id]
        self.nodes.append(node)
        self.last_updated = time.time()

        # Sort by trust (φ bonus nodes first)
        self.nodes.sort(key=lambda n: n.trust_score, reverse=True)

    def get_experts(self, count: int = K) -> List[NodeInfo]:
        """Get top experts for this topic."""
        return self.nodes[:count]


class DHTBridge:
    """
    Bridge between BAZINGA Protocol and Kademlia DHT.

    Features:
    - Bootstrap from HF Space registry
    - Fall back to hardcoded bootstrap nodes
    - Once connected, operate fully P2P
    - Knowledge topic announcement and discovery
    - φ trust bonus for local model nodes

    Usage:
        bridge = DHTBridge(alpha="...", omega="...", uses_local_model=True)
        await bridge.start()
        await bridge.bootstrap()  # Connects to HF Space first

        # Announce knowledge
        await bridge.announce_knowledge("quantum mechanics")

        # Find experts
        experts = await bridge.find_experts("quantum mechanics")

        # Pure P2P lookup
        peers = await bridge.find_peers_for_topic("consciousness")
    """

    def __init__(
        self,
        alpha: str,
        omega: str,
        address: str = "0.0.0.0",
        port: int = 5150,
        uses_local_model: bool = False,
    ):
        """
        Initialize DHT Bridge.

        Args:
            alpha: PoB alpha signature (from prove_boundary)
            omega: PoB omega signature
            address: Bind address
            port: DHT port
            uses_local_model: True if running Ollama/local LLM (gets φ bonus)
        """
        self.alpha = alpha
        self.omega = omega
        self.uses_local_model = uses_local_model

        # Create Kademlia node with PoB-derived ID
        self.dht = KademliaNode.from_pob(
            alpha=alpha,
            omega=omega,
            address=address,
            port=port,
            uses_local_model=uses_local_model,
        )

        # Knowledge topics we're tracking
        self.topics: Dict[str, KnowledgeTopic] = {}

        # My knowledge domains (topics I'm expert in)
        self.my_domains: List[str] = []

        # HF registry connection
        self.hf_node_id: Optional[str] = None
        self.hf_connected = False

        # Stats
        self.stats = {
            "bootstrap_attempts": 0,
            "hf_connections": 0,
            "topics_announced": 0,
            "expert_lookups": 0,
            "p2p_discoveries": 0,
        }

    async def start(self):
        """Start the DHT node."""
        await self.dht.start()
        print(f"\n  DHT Bridge active")
        print(f"    Node ID: {self.dht.node_id.hex()[:16]}...")
        print(f"    Trust: {self.dht.trust_score:.3f}x {'(φ bonus!)' if self.uses_local_model else ''}")

    async def stop(self):
        """Stop the DHT node."""
        await self.dht.stop()

    async def bootstrap(self) -> bool:
        """
        Bootstrap into the network.

        Priority:
        1. Try HF Space registry (gets list of active nodes)
        2. Try hardcoded bootstrap nodes
        3. Start fresh (first node in network)

        Returns True if connected to at least one peer.
        """
        self.stats["bootstrap_attempts"] += 1
        print(f"\n  Bootstrapping DHT...")

        # Step 1: Try HF Space
        hf_peers = await self._bootstrap_from_hf()
        if hf_peers:
            print(f"    ✓ Got {len(hf_peers)} peers from HF Space")
            self.stats["hf_connections"] += 1
            self.hf_connected = True

            # Connect to peers from HF
            for host, port in hf_peers[:5]:  # Connect to up to 5
                try:
                    temp_id = hash_to_id(f"{host}:{port}")
                    temp_node = NodeInfo(node_id=temp_id, address=host, port=port)

                    if await self.dht.ping(temp_node):
                        print(f"      Connected to {host}:{port}")
                except Exception as e:
                    print(f"      Failed {host}:{port}: {e}")

        # Step 2: Try hardcoded bootstrap if needed
        if self.dht.routing_table.total_nodes == 0 and BOOTSTRAP_NODES:
            print(f"    Trying hardcoded bootstrap nodes...")
            await self.dht.bootstrap(BOOTSTRAP_NODES)

        # Step 3: Do iterative lookup for our own ID to populate table
        if self.dht.routing_table.total_nodes > 0:
            print(f"    Populating routing table...")
            await self.dht.find_node(self.dht.node_id)

        total = self.dht.routing_table.total_nodes
        print(f"    Routing table: {total} nodes")

        if total == 0:
            print(f"    ⚠ No peers found - you're the first node!")
            print(f"      Share your address for others to join: {self.dht.address}:{self.dht.port}")

        return total > 0

    async def _bootstrap_from_hf(self) -> List[Tuple[str, int]]:
        """Get peer list from HuggingFace Space registry."""
        if not HTTPX_AVAILABLE:
            print(f"    ⚠ httpx not installed - skipping HF bootstrap")
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Register ourselves first
                reg_response = await client.post(
                    f"{HF_SPACE_URL}/api/register",
                    json={
                        "node_name": f"dht-{self.dht.node_id.hex()[:8]}",
                        "node_id": self.dht.node_id.hex(),
                        "port": self.dht.port,
                        "trust_score": self.dht.trust_score,
                        "uses_local_model": self.uses_local_model,
                    }
                )

                if reg_response.status_code == 200:
                    reg_data = reg_response.json()
                    if reg_data.get("success"):
                        self.hf_node_id = reg_data.get("node_id")
                        print(f"    ✓ Registered with HF: {self.hf_node_id}")

                # Get peers
                peers_response = await client.get(
                    f"{HF_SPACE_URL}/api/peers",
                    params={"node_id": self.hf_node_id} if self.hf_node_id else {}
                )

                if peers_response.status_code == 200:
                    peers_data = peers_response.json()
                    if peers_data.get("success"):
                        peers = []
                        for peer in peers_data.get("peers", []):
                            addr = peer.get("address", "")
                            if ":" in addr:
                                host, port_str = addr.rsplit(":", 1)
                                try:
                                    peers.append((host, int(port_str)))
                                except ValueError:
                                    pass
                        return peers

        except Exception as e:
            print(f"    ⚠ HF bootstrap failed: {e}")

        return []

    async def announce_knowledge(self, topic: str, confidence: float = 0.8):
        """
        Announce that we have knowledge about a topic.

        This stores our info at nodes closest to the topic hash.
        Other nodes can find us by looking up the topic.

        Args:
            topic: Knowledge topic (e.g., "quantum mechanics", "python")
            confidence: How confident we are (0-1)
        """
        topic_hash = hash_to_id(topic)
        self.stats["topics_announced"] += 1

        # Add to our domains
        if topic not in self.my_domains:
            self.my_domains.append(topic)

        # Create announcement
        announcement = {
            "topic": topic,
            "node_id": self.dht.node_id.hex(),
            "address": self.dht.address,
            "port": self.dht.port,
            "trust_score": self.dht.trust_score,
            "uses_local_model": self.uses_local_model,
            "confidence": confidence,
            "timestamp": time.time(),
        }

        # Store in DHT
        await self.dht.store(f"topic:{topic}", announcement, ttl=1800)  # 30 min TTL

        # Also find closest nodes and ping them directly
        closest = await self.dht.find_node(topic_hash)
        print(f"    Announced '{topic}' to {len(closest)} nodes")

    async def find_experts(self, topic: str, count: int = K) -> List[NodeInfo]:
        """
        Find nodes that have knowledge about a topic.

        Uses iterative DHT lookup to find nodes closest to
        the topic hash, then queries them for expert announcements.

        Args:
            topic: Knowledge topic to search for
            count: Max number of experts to return

        Returns:
            List of NodeInfo sorted by trust score (φ bonus first)
        """
        self.stats["expert_lookups"] += 1
        topic_hash = hash_to_id(topic)

        # Check local cache first
        if topic in self.topics:
            cached = self.topics[topic]
            if time.time() - cached.last_updated < 300:  # 5 min cache
                return cached.get_experts(count)

        # Look up in DHT
        result = await self.dht.get(f"topic:{topic}")
        experts: List[NodeInfo] = []

        if result:
            # Single result
            experts.append(NodeInfo(
                node_id=bytes.fromhex(result["node_id"]),
                address=result["address"],
                port=result["port"],
                trust_score=result.get("trust_score", 0.5),
                uses_local_model=result.get("uses_local_model", False),
            ))

        # Also query closest nodes for their knowledge
        closest = await self.dht.find_node(topic_hash)

        for node in closest:
            # Could query each node for its topics, but for now
            # just include nodes close to the topic hash
            if node.node_id not in [e.node_id for e in experts]:
                experts.append(node)

        # Sort by trust (φ bonus nodes first)
        experts.sort(key=lambda n: n.trust_score, reverse=True)

        # Cache result
        kt = KnowledgeTopic(topic=topic, topic_hash=topic_hash)
        for e in experts:
            kt.add_node(e)
        self.topics[topic] = kt

        return experts[:count]

    async def find_peers_for_query(self, query: str, count: int = 3) -> List[NodeInfo]:
        """
        Find best peers to handle a query.

        Hashes the query to find topically relevant nodes,
        sorted by trust score.

        Args:
            query: The question/query
            count: Number of peers to return

        Returns:
            List of NodeInfo best suited to answer this query
        """
        self.stats["p2p_discoveries"] += 1

        # Hash query to find relevant topic space
        query_hash = hash_to_id(query)

        # Find closest nodes
        closest = await self.dht.find_node(query_hash)

        # Sort by trust (φ bonus nodes prioritized)
        closest.sort(key=lambda n: n.trust_score, reverse=True)

        return closest[:count]

    async def heartbeat(self):
        """Send heartbeat to HF Space (keeps registration active)."""
        if not self.hf_connected or not self.hf_node_id:
            return

        if not HTTPX_AVAILABLE:
            return

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{HF_SPACE_URL}/api/heartbeat",
                    json={
                        "node_id": self.hf_node_id,
                        "port": self.dht.port,
                        "routing_table_size": self.dht.routing_table.total_nodes,
                    }
                )
        except Exception:
            pass  # Silent fail for heartbeat

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        dht_stats = self.dht.get_stats()
        return {
            "bridge": self.stats,
            "dht": dht_stats,
            "my_domains": self.my_domains,
            "hf_connected": self.hf_connected,
            "hf_node_id": self.hf_node_id,
        }

    def print_status(self):
        """Print current DHT status."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  BAZINGA DHT BRIDGE STATUS")
        print(f"{'='*60}")
        print(f"  Node ID: {self.dht.node_id.hex()[:24]}...")
        print(f"  Address: {self.dht.address}:{self.dht.port}")
        print(f"  Trust Score: {self.dht.trust_score:.3f}x {'(φ LOCAL MODEL BONUS!)' if self.uses_local_model else ''}")
        print()
        print(f"  Routing Table: {self.dht.routing_table.total_nodes} nodes")
        print(f"  HF Connected: {'✓' if self.hf_connected else '✗'}")
        print(f"  My Domains: {', '.join(self.my_domains) if self.my_domains else 'None'}")
        print()
        print(f"  Stats:")
        print(f"    Bootstrap attempts: {self.stats['bootstrap_attempts']}")
        print(f"    HF connections: {self.stats['hf_connections']}")
        print(f"    Topics announced: {self.stats['topics_announced']}")
        print(f"    Expert lookups: {self.stats['expert_lookups']}")
        print(f"    P2P discoveries: {self.stats['p2p_discoveries']}")
        print(f"{'='*60}\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_dht_bridge(
    uses_local_model: bool = False,
    port: int = 5150,
) -> DHTBridge:
    """
    Create and bootstrap a DHT bridge.

    Generates PoB automatically and connects to network.

    Args:
        uses_local_model: True if running Ollama (gets φ bonus)
        port: DHT port

    Returns:
        Connected DHTBridge instance
    """
    # Import here to avoid circular dependency
    from bazinga.darmiyan import prove_boundary

    print(f"\n  Creating DHT Bridge...")
    print(f"    Generating Proof-of-Boundary...")

    # Generate PoB for node identity
    pob = prove_boundary()

    if pob.valid:
        print(f"    ✓ PoB valid (ratio: {pob.ratio:.4f})")
    else:
        print(f"    ⚠ PoB invalid, using anyway for testing")

    # Create bridge
    bridge = DHTBridge(
        alpha=pob.alpha,
        omega=pob.omega,
        port=port,
        uses_local_model=uses_local_model,
    )

    # Start and bootstrap
    await bridge.start()
    await bridge.bootstrap()

    return bridge


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA DHT BRIDGE TEST")
        print("=" * 60)

        # Test with simulated local model
        bridge = await create_dht_bridge(uses_local_model=True, port=0)

        # Print status
        bridge.print_status()

        # Announce some knowledge
        print("Testing knowledge announcement...")
        await bridge.announce_knowledge("distributed systems")
        await bridge.announce_knowledge("consciousness")
        await bridge.announce_knowledge("golden ratio")

        # Print updated status
        bridge.print_status()

        # Find experts (will be empty in solo test)
        print("Testing expert lookup...")
        experts = await bridge.find_experts("quantum mechanics")
        print(f"  Found {len(experts)} experts for 'quantum mechanics'")

        # Stop
        await bridge.stop()

        print("\n" + "=" * 60)
        print("  DHT Bridge Test Complete!")
        print("=" * 60)

    asyncio.run(test())
