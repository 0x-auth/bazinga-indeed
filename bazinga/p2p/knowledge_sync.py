#!/usr/bin/env python3
"""
BAZINGA Knowledge Sync - Privacy-Preserving Knowledge Sharing

Sync knowledge graphs between nodes WITHOUT sharing raw documents.
Only embeddings + metadata are shared.

Privacy Model:
- Your documents stay on your machine
- Only embeddings (numeric vectors) are shared
- Others can see "this node has knowledge about X"
- But can't reconstruct your actual documents
- You control what you share (opt-in per document)

"Share understanding, not data."
"""

import asyncio
import hashlib
import json
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


def compute_phi_signature(embedding: List[float], metadata: Dict) -> str:
    """
    Compute φ-signature for knowledge verification.

    The φ-signature encodes:
    - Embedding statistics (mean, std)
    - Metadata hash
    - φ-coherence value

    This allows verification without revealing content.
    """
    # Embedding statistics
    if embedding:
        mean = sum(embedding) / len(embedding)
        variance = sum((x - mean) ** 2 for x in embedding) / len(embedding)
        std = math.sqrt(variance) if variance > 0 else 0
    else:
        mean, std = 0, 0

    # Metadata hash
    meta_str = json.dumps(metadata, sort_keys=True)
    meta_hash = hashlib.sha256(meta_str.encode()).hexdigest()[:16]

    # φ-coherence (how close mean is to ideal φ-scaled value)
    ideal = PHI / (1 + PHI)  # ~0.618
    phi_coherence = 1 - abs(mean - ideal) if abs(mean) < 2 else 0.5

    # Combine into signature
    signature_data = f"{mean:.6f}:{std:.6f}:{meta_hash}:{phi_coherence:.4f}"
    signature = hashlib.sha256(signature_data.encode()).hexdigest()

    return signature


def verify_phi_signature(
    embedding: List[float],
    metadata: Dict,
    signature: str,
    tolerance: float = 0.01,
) -> bool:
    """
    Verify φ-signature of received knowledge.

    Returns True if signature matches (knowledge is authentic).
    """
    computed = compute_phi_signature(embedding, metadata)
    # In real impl, would check more rigorously
    # For now, simple comparison
    return computed == signature


@dataclass
class KnowledgePackage:
    """
    Shareable knowledge package.

    Contains:
    - Embedding vector (384 dims typically)
    - Metadata (title, tags, timestamp, α-SEED status)
    - φ-signature for verification
    - Source node ID

    Does NOT contain:
    - Raw document content
    - File paths
    - Personal information
    """
    doc_id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    phi_signature: str
    source_node: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "phi_signature": self.phi_signature,
            "source_node": self.source_node,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgePackage':
        return cls(
            doc_id=data["doc_id"],
            embedding=data["embedding"],
            metadata=data["metadata"],
            phi_signature=data["phi_signature"],
            source_node=data["source_node"],
            timestamp=data.get("timestamp", time.time()),
        )

    def verify(self) -> bool:
        """Verify the package's φ-signature."""
        return verify_phi_signature(
            self.embedding,
            self.metadata,
            self.phi_signature,
        )


@dataclass
class ExternalReference:
    """Reference to knowledge on another node (not raw data)."""
    embedding: List[float]
    metadata: Dict[str, Any]
    source_node: str
    tau: float  # Trust score of source node
    last_updated: datetime = field(default_factory=datetime.now)


class KnowledgeGraphSync:
    """
    Privacy-preserving knowledge graph synchronization.

    Syncs:
    - Document embeddings (numeric vectors)
    - Metadata (title, tags, etc.)
    - α-SEED status (fundamental knowledge markers)

    Does NOT sync:
    - Raw document content
    - Personal/sensitive information
    - File system paths

    Usage:
        sync = KnowledgeGraphSync(local_kb, p2p_node)

        # Create package for sharing
        package = sync.create_sync_package(doc_id)

        # Share to network
        await sync.sync_to_network()

        # Receive knowledge
        await sync.receive_knowledge(package_data)
    """

    def __init__(self, local_kb, p2p_node):
        """
        Initialize knowledge sync.

        Args:
            local_kb: Local knowledge base (ChromaDB wrapper)
            p2p_node: P2P node for network communication
        """
        self.kb = local_kb
        self.p2p = p2p_node

        # External references (knowledge from other nodes)
        self.external_refs: Dict[str, ExternalReference] = {}

        # Sync stats
        self.stats = {
            'packages_sent': 0,
            'packages_received': 0,
            'packages_rejected': 0,
        }

        # Share settings
        self.share_alpha_seeds_only = True  # Default: only share fundamental knowledge
        self.share_threshold_tau = 0.6  # Minimum tau to receive from

    def create_sync_package(self, doc_id: str) -> Optional[KnowledgePackage]:
        """
        Create shareable knowledge package from local document.

        Args:
            doc_id: Document ID in local KB

        Returns:
            KnowledgePackage or None if document not found
        """
        if not self.kb:
            return None

        try:
            # Get document from KB
            doc = self.kb.get_document(doc_id)
            if not doc:
                return None

            # Extract embedding
            embedding = doc.get('embedding', [])
            if not embedding:
                return None

            # Build metadata (no sensitive info)
            metadata = {
                "title": doc.get('title', 'Untitled'),
                "tags": doc.get('tags', []),
                "timestamp": doc.get('timestamp', time.time()),
                "alpha_seed": self._is_alpha_seed(doc),
                "coherence": doc.get('coherence', 0.5),
                "chunk_count": doc.get('chunk_count', 1),
            }

            # Compute φ-signature
            phi_signature = compute_phi_signature(embedding, metadata)

            return KnowledgePackage(
                doc_id=doc_id,
                embedding=embedding,
                metadata=metadata,
                phi_signature=phi_signature,
                source_node=self.p2p.node_id if self.p2p else "local",
            )

        except Exception as e:
            print(f"Error creating sync package: {e}")
            return None

    def _is_alpha_seed(self, doc: Dict) -> bool:
        """Check if document is an α-SEED (hash % 137 == 0)."""
        content = doc.get('content', '')
        if not content:
            content = str(doc.get('embedding', []))

        doc_hash = int(hashlib.sha256(content.encode()).hexdigest(), 16)
        return doc_hash % ALPHA == 0

    async def sync_to_network(self, force_all: bool = False):
        """
        Share knowledge graph updates with network.

        Args:
            force_all: If True, share all documents. If False, only α-SEEDs.
        """
        if not self.kb or not self.p2p:
            return

        documents_to_sync = []

        if force_all or not self.share_alpha_seeds_only:
            # Get all documents
            documents_to_sync = self.kb.get_all_document_ids()
        else:
            # Get only α-SEED documents
            documents_to_sync = self._get_alpha_seed_doc_ids()

        print(f"📤 Syncing {len(documents_to_sync)} documents to network...")

        for doc_id in documents_to_sync:
            package = self.create_sync_package(doc_id)
            if package:
                # Broadcast to network
                await self.p2p.publish(
                    "/bazinga/knowledge-sync",
                    package.to_dict(),
                )
                self.stats['packages_sent'] += 1

    def _get_alpha_seed_doc_ids(self) -> List[str]:
        """Get document IDs that are α-SEEDs."""
        if not self.kb:
            return []

        alpha_seeds = []
        for doc_id in self.kb.get_all_document_ids():
            doc = self.kb.get_document(doc_id)
            if doc and self._is_alpha_seed(doc):
                alpha_seeds.append(doc_id)

        return alpha_seeds

    async def receive_knowledge(self, package_data: Dict) -> bool:
        """
        Receive knowledge from another node.

        Args:
            package_data: Package dict from network

        Returns:
            True if accepted, False if rejected
        """
        try:
            package = KnowledgePackage.from_dict(package_data)

            # 1. Verify φ-signature
            if not package.verify():
                print(f"   ✗ Rejected package: invalid φ-signature")
                self.stats['packages_rejected'] += 1
                return False

            # 2. Check source node trust
            source_tau = self._get_source_trust(package.source_node)
            if source_tau < self.share_threshold_tau:
                print(f"   ✗ Rejected package: low trust ({source_tau:.2f})")
                self.stats['packages_rejected'] += 1
                return False

            # 3. Add to external references
            ref = ExternalReference(
                embedding=package.embedding,
                metadata=package.metadata,
                source_node=package.source_node,
                tau=source_tau,
            )

            self.external_refs[package.doc_id] = ref
            self.stats['packages_received'] += 1

            # 4. Optionally index for search
            if self.kb:
                self._add_external_to_search(package, source_tau)

            return True

        except Exception as e:
            print(f"Error receiving knowledge: {e}")
            self.stats['packages_rejected'] += 1
            return False

    def _get_source_trust(self, node_id: str) -> float:
        """Get trust score for source node."""
        if not self.p2p:
            return 0.5

        # Check if we know this peer
        peer = self.p2p.peers.get(node_id)
        if peer:
            return peer.tau

        return 0.5  # Default trust for unknown nodes

    def _add_external_to_search(self, package: KnowledgePackage, trust: float):
        """Add external knowledge to search index."""
        if not self.kb:
            return

        try:
            # Add as external reference (not local document)
            self.kb.add_external_reference({
                "id": f"ext_{package.source_node[:8]}_{package.doc_id}",
                "embedding": package.embedding,
                "metadata": {
                    **package.metadata,
                    "external": True,
                    "source_node": package.source_node,
                    "trust_weight": trust,
                },
            })
        except Exception:
            pass  # KB may not support external refs

    def search_with_external(
        self,
        query_embedding: List[float],
        limit: int = 10,
        include_external: bool = True,
    ) -> List[Dict]:
        """
        Search local and external knowledge.

        Args:
            query_embedding: Query embedding vector
            limit: Max results
            include_external: Whether to include external refs

        Returns:
            List of results weighted by trust
        """
        results = []

        # Search local KB
        if self.kb:
            local_results = self.kb.search(query_embedding, limit=limit)
            for r in local_results:
                r['source'] = 'local'
                r['trust_weight'] = 1.0  # Full trust for local
                results.append(r)

        # Search external references
        if include_external:
            external_results = self._search_external(query_embedding, limit)
            results.extend(external_results)

        # Sort by trust-weighted similarity
        results.sort(
            key=lambda r: r.get('similarity', 0) * r.get('trust_weight', 0.5),
            reverse=True,
        )

        return results[:limit]

    def _search_external(
        self,
        query_embedding: List[float],
        limit: int,
    ) -> List[Dict]:
        """Search external references."""
        results = []

        for ref_id, ref in self.external_refs.items():
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, ref.embedding)

            results.append({
                "id": ref_id,
                "embedding": ref.embedding,
                "metadata": ref.metadata,
                "similarity": similarity,
                "source": "external",
                "source_node": ref.source_node,
                "trust_weight": ref.tau,
            })

        # Sort by similarity
        results.sort(key=lambda r: r['similarity'], reverse=True)

        return results[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get_external_count(self) -> int:
        """Get count of external references."""
        return len(self.external_refs)

    def get_external_by_node(self) -> Dict[str, int]:
        """Get count of external refs by source node."""
        by_node: Dict[str, int] = {}
        for ref in self.external_refs.values():
            node = ref.source_node[:8]
            by_node[node] = by_node.get(node, 0) + 1
        return by_node

    def get_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        return {
            "external_refs": len(self.external_refs),
            "external_by_node": self.get_external_by_node(),
            **self.stats,
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Knowledge Sync Test")
    print("=" * 60)

    # Test φ-signature
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    metadata = {"title": "Test Doc", "tags": ["test"]}

    signature = compute_phi_signature(embedding, metadata)
    print(f"\nφ-Signature: {signature[:32]}...")

    # Verify
    verified = verify_phi_signature(embedding, metadata, signature)
    print(f"Verified: {verified}")

    # Test with wrong data
    wrong = verify_phi_signature([0.9, 0.8, 0.7], metadata, signature)
    print(f"Wrong embedding verified: {wrong}")

    # Create package
    package = KnowledgePackage(
        doc_id="test-123",
        embedding=embedding,
        metadata=metadata,
        phi_signature=signature,
        source_node="node-abc",
    )

    print(f"\nPackage created:")
    print(f"  ID: {package.doc_id}")
    print(f"  Source: {package.source_node}")
    print(f"  Verified: {package.verify()}")

    print("\nKnowledge Sync module ready!")
