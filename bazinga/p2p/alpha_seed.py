#!/usr/bin/env python3
"""
BAZINGA α-SEED Network - Fundamental Knowledge Anchoring

Files with hash % 137 == 0 are "α-SEEDs" - fundamental anchors
that organize the distributed knowledge graph.

Why 137 (α)?
- Fine structure constant ≈ 1/137
- Fundamental to physics
- Provides natural, deterministic clustering
- No coordination needed - same files hash the same way everywhere

"Fundamental knowledge anchors the network."
"""

import asyncio
import hashlib
import math
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


def is_alpha_seed(content_or_hash) -> bool:
    """
    Check if content/hash is an α-SEED (divisible by 137).

    Args:
        content_or_hash: String content or integer hash

    Returns:
        True if this is an α-SEED
    """
    if isinstance(content_or_hash, int):
        return content_or_hash % ALPHA == 0
    else:
        # Hash the content
        content_hash = int(
            hashlib.sha256(str(content_or_hash).encode()).hexdigest(),
            16
        )
        return content_hash % ALPHA == 0


def compute_alpha_hash(content: str) -> int:
    """Compute hash for α-SEED check."""
    return int(hashlib.sha256(content.encode()).hexdigest(), 16)


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance (1 - similarity) between vectors."""
    if len(a) != len(b) or not a:
        return 1.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = dot / (norm_a * norm_b)
    return 1 - similarity


@dataclass
class AlphaSeed:
    """An α-SEED document."""
    doc_id: str
    content_hash: int
    embedding: List[float]
    title: str
    node_id: str
    tau: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content_hash": self.content_hash,
            "embedding": self.embedding,
            "title": self.title,
            "node_id": self.node_id,
            "tau": self.tau,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlphaSeed':
        return cls(
            doc_id=data["doc_id"],
            content_hash=data["content_hash"],
            embedding=data["embedding"],
            title=data["title"],
            node_id=data["node_id"],
            tau=data["tau"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
        )


class AlphaSeedNetwork:
    """
    Network of α-SEED anchor points.

    α-SEEDs are fundamental knowledge anchors that:
    - Create stable "anchor points" in knowledge graph
    - Enable natural clustering (same files → same hash everywhere)
    - Require no coordination (deterministic)
    - Network self-organizes around these anchors

    Usage:
        network = AlphaSeedNetwork(kb, p2p_node)

        # Find local α-SEEDs
        seeds = network.identify_alpha_seeds()

        # Build network
        await network.build_anchor_network()

        # Find related knowledge
        node_id = network.find_related_knowledge(query_embedding)
    """

    def __init__(self, kb, p2p_node):
        """
        Initialize α-SEED network.

        Args:
            kb: Local knowledge base
            p2p_node: P2P node for network communication
        """
        self.kb = kb
        self.p2p = p2p_node

        # Anchor graph: maps α-SEED hash → seed info
        self.anchor_graph: Dict[int, AlphaSeed] = {}

        # Local α-SEEDs
        self.local_seeds: List[AlphaSeed] = []

        # Stats
        self.stats = {
            'local_seeds': 0,
            'network_seeds': 0,
            'announcements_sent': 0,
            'announcements_received': 0,
        }

    def identify_alpha_seeds(self) -> List[AlphaSeed]:
        """
        Find all α-SEED documents in local KB.

        Returns:
            List of AlphaSeed objects
        """
        self.local_seeds = []

        if not self.kb:
            return self.local_seeds

        try:
            # Get all documents
            doc_ids = self.kb.get_all_document_ids()

            for doc_id in doc_ids:
                doc = self.kb.get_document(doc_id)
                if not doc:
                    continue

                # Get content for hashing
                content = doc.get('content', '')
                if not content:
                    # Use embedding as fallback
                    content = str(doc.get('embedding', []))

                content_hash = compute_alpha_hash(content)

                # Check if α-SEED
                if is_alpha_seed(content_hash):
                    seed = AlphaSeed(
                        doc_id=doc_id,
                        content_hash=content_hash,
                        embedding=doc.get('embedding', []),
                        title=doc.get('title', doc_id),
                        node_id=self.p2p.node_id if self.p2p else "local",
                        tau=self.p2p.tau if self.p2p else 0.5,
                    )
                    self.local_seeds.append(seed)

                    # Add to anchor graph
                    self.anchor_graph[content_hash] = seed

            self.stats['local_seeds'] = len(self.local_seeds)

        except Exception as e:
            print(f"Error identifying α-SEEDs: {e}")

        return self.local_seeds

    async def build_anchor_network(self):
        """
        Build distributed graph of α-SEED relationships.

        1. Identify local α-SEEDs
        2. Announce them to network
        3. Subscribe to α-SEED announcements from others
        """
        if not self.p2p:
            return

        # Find local α-SEEDs if not already done
        if not self.local_seeds:
            self.identify_alpha_seeds()

        print(f"🌐 Building α-SEED network with {len(self.local_seeds)} local anchors...")

        # Announce each α-SEED
        for seed in self.local_seeds:
            await self.p2p.publish(
                "/bazinga/alpha-seed",
                {
                    "type": "ALPHA_SEED_ANNOUNCEMENT",
                    "seed": seed.to_dict(),
                }
            )
            self.stats['announcements_sent'] += 1

        # Subscribe to α-SEED announcements
        self.p2p.subscribe(
            "/bazinga/alpha-seed",
            self._handle_alpha_seed_announcement,
        )

        print(f"   ✓ Announced {len(self.local_seeds)} α-SEEDs")

    async def _handle_alpha_seed_announcement(self, message):
        """Receive α-SEED announcement from another node."""
        try:
            payload = message.payload
            if payload.get("type") != "ALPHA_SEED_ANNOUNCEMENT":
                return

            seed_data = payload.get("seed")
            if not seed_data:
                return

            seed = AlphaSeed.from_dict(seed_data)

            # Don't add our own
            if self.p2p and seed.node_id == self.p2p.node_id:
                return

            # Verify it's actually an α-SEED
            if not is_alpha_seed(seed.content_hash):
                print(f"   ✗ Invalid α-SEED from {seed.node_id[:8]}")
                return

            # Add to anchor graph
            self.anchor_graph[seed.content_hash] = seed
            self.stats['announcements_received'] += 1
            self.stats['network_seeds'] = len(self.anchor_graph) - len(self.local_seeds)

            print(f"   📡 Received α-SEED '{seed.title[:20]}' from {seed.node_id[:8]}")

        except Exception as e:
            print(f"Error handling α-SEED announcement: {e}")

    def find_related_knowledge(
        self,
        query_embedding: List[float],
        max_distance: float = 0.5,
    ) -> Optional[str]:
        """
        Use α-SEED network to find related knowledge.

        Finds the closest α-SEED to the query, then returns
        the node that hosts it.

        Args:
            query_embedding: Query embedding vector
            max_distance: Maximum cosine distance threshold

        Returns:
            Node ID of the node hosting closest α-SEED, or None
        """
        if not self.anchor_graph:
            return None

        closest_seed: Optional[AlphaSeed] = None
        min_distance = float('inf')

        for seed_hash, seed in self.anchor_graph.items():
            if not seed.embedding:
                continue

            distance = cosine_distance(query_embedding, seed.embedding)

            if distance < min_distance:
                min_distance = distance
                closest_seed = seed

        if closest_seed and min_distance <= max_distance:
            return closest_seed.node_id

        return None

    def find_closest_seeds(
        self,
        query_embedding: List[float],
        count: int = 5,
    ) -> List[Dict]:
        """
        Find closest α-SEEDs to a query.

        Args:
            query_embedding: Query embedding
            count: Number of results

        Returns:
            List of closest seeds with distances
        """
        results = []

        for seed_hash, seed in self.anchor_graph.items():
            if not seed.embedding:
                continue

            distance = cosine_distance(query_embedding, seed.embedding)

            results.append({
                "seed": seed.to_dict(),
                "distance": distance,
                "similarity": 1 - distance,
            })

        # Sort by distance
        results.sort(key=lambda r: r['distance'])

        return results[:count]

    def get_seeds_by_node(self) -> Dict[str, List[str]]:
        """Get α-SEEDs grouped by hosting node."""
        by_node: Dict[str, List[str]] = {}

        for seed_hash, seed in self.anchor_graph.items():
            node_id = seed.node_id[:8]
            if node_id not in by_node:
                by_node[node_id] = []
            by_node[node_id].append(seed.title[:30])

        return by_node

    def get_local_seed_count(self) -> int:
        """Get count of local α-SEEDs."""
        return len(self.local_seeds)

    def get_network_seed_count(self) -> int:
        """Get count of network α-SEEDs (not local)."""
        local_hashes = {s.content_hash for s in self.local_seeds}
        network_count = sum(
            1 for h in self.anchor_graph.keys()
            if h not in local_hashes
        )
        return network_count

    def get_stats(self) -> Dict[str, Any]:
        """Get α-SEED network statistics."""
        return {
            'total_anchors': len(self.anchor_graph),
            'local_seeds': len(self.local_seeds),
            'network_seeds': self.get_network_seed_count(),
            'nodes_with_seeds': len(self.get_seeds_by_node()),
            **self.stats,
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA α-SEED Network Test")
    print("=" * 60)

    # Test α-SEED detection
    test_contents = [
        "This is a test document about quantum mechanics",
        "Another document about machine learning",
        "A third document about consciousness",
        "The golden ratio φ = 1.618...",
        "Physics and the fine structure constant",
    ]

    print("\nTesting α-SEED detection:")
    alpha_count = 0
    for content in test_contents:
        content_hash = compute_alpha_hash(content)
        is_seed = is_alpha_seed(content_hash)
        if is_seed:
            alpha_count += 1
            print(f"  ★ α-SEED: '{content[:40]}...'")
            print(f"    Hash mod 137 = {content_hash % ALPHA}")
        else:
            print(f"  ○ Not α-SEED: '{content[:40]}...'")
            print(f"    Hash mod 137 = {content_hash % ALPHA}")

    print(f"\n{alpha_count}/{len(test_contents)} documents are α-SEEDs")
    print(f"Expected: ~{len(test_contents)/ALPHA:.1f} (1/137 probability)")

    # Test with more documents to verify probability
    print("\nStatistical test (1000 random strings):")
    import random
    import string

    seeds_found = 0
    for _ in range(1000):
        random_content = ''.join(random.choices(string.ascii_letters, k=100))
        if is_alpha_seed(random_content):
            seeds_found += 1

    expected = 1000 / ALPHA
    print(f"  Found: {seeds_found} α-SEEDs")
    print(f"  Expected: ~{expected:.1f} (1000/137)")
    print(f"  Ratio: {seeds_found/expected:.2f}x expected")

    print("\nα-SEED Network module ready!")
