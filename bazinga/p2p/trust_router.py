#!/usr/bin/env python3
"""
BAZINGA Trust Router - Trust-Based Query Routing

Routes queries to nodes with highest Trust dimension (τ).
Trust prevents spam: new nodes start low, must earn trust.

"Trust is earned through good answers."
"""

import asyncio
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


@dataclass
class QueryResult:
    """Result from a query response."""
    results: List[Dict]
    source_node: str
    tau: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class TrustRouter:
    """
    Trust-based query routing for BAZINGA network.

    Features:
    - Route queries to most trusted nodes
    - Aggregate responses weighted by Trust
    - Track peer trust scores over time
    - Penalize nodes that return bad results

    Trust Model:
    - New nodes start with τ = 0.5
    - Good responses → τ increases (max 1.0)
    - Bad responses → τ decreases (min 0.1)
    - φ-based smoothing for stability

    Usage:
        router = TrustRouter(dht, local_tau=0.8)

        # Route query
        results = await router.route_query(query_embedding, min_tau=0.7)

        # Record feedback
        router.record_feedback(node_id, query_id, good=True)
    """

    def __init__(self, dht, local_tau: float = 0.5):
        """
        Initialize trust router.

        Args:
            dht: DHT instance for finding nodes
            local_tau: This node's trust score
        """
        self.dht = dht
        self.tau = local_tau

        # Peer trust tracking
        self.peer_scores: Dict[str, float] = {}
        self.peer_history: Dict[str, List[Tuple[datetime, bool]]] = {}

        # Query tracking for feedback
        self.pending_queries: Dict[str, Dict] = {}

        # Router config
        self.default_min_tau = 0.5
        self.max_concurrent_queries = 3
        self.query_timeout = 10.0  # seconds

        # Stats
        self.stats = {
            'queries_routed': 0,
            'responses_received': 0,
            'responses_aggregated': 0,
        }

    def get_peer_trust(self, node_id: str) -> float:
        """Get trust score for a peer."""
        return self.peer_scores.get(node_id, 0.5)

    def update_peer_trust(self, node_id: str, good_response: bool):
        """
        Update peer trust based on response quality.

        Args:
            node_id: Peer node ID
            good_response: Whether the response was good
        """
        current_tau = self.peer_scores.get(node_id, 0.5)

        # Track history
        if node_id not in self.peer_history:
            self.peer_history[node_id] = []
        self.peer_history[node_id].append((datetime.now(), good_response))

        # Keep last 50 interactions
        if len(self.peer_history[node_id]) > 50:
            self.peer_history[node_id] = self.peer_history[node_id][-50:]

        # Calculate new trust
        if good_response:
            # φ-based increase (slower as τ increases)
            increase = 0.05 * (1 - current_tau) / PHI
            new_tau = min(1.0, current_tau + increase)
        else:
            # Faster decrease for bad responses
            decrease = 0.1 * current_tau
            new_tau = max(0.1, current_tau - decrease)

        # Smooth with φ
        smoothed_tau = current_tau * (1/PHI) + new_tau * (1 - 1/PHI)

        self.peer_scores[node_id] = smoothed_tau

    async def route_query(
        self,
        query_embedding: List[float],
        topic: Optional[str] = None,
        min_tau: Optional[float] = None,
        max_nodes: int = 3,
    ) -> List[Dict]:
        """
        Route query to most trusted nodes.

        Args:
            query_embedding: Query embedding vector
            topic: Optional topic for DHT lookup
            min_tau: Minimum trust score for responders
            max_nodes: Maximum nodes to query

        Returns:
            Aggregated results weighted by trust
        """
        min_tau = min_tau or self.default_min_tau
        self.stats['queries_routed'] += 1

        # Find potential nodes
        potential_nodes = await self._find_qualified_nodes(
            topic=topic,
            min_tau=min_tau,
        )

        if not potential_nodes:
            return []

        # Sort by trust (highest first)
        potential_nodes.sort(key=lambda n: n.get('tau', 0), reverse=True)

        # Query top nodes
        top_nodes = potential_nodes[:max_nodes]
        responses = await self._query_nodes(
            top_nodes,
            query_embedding,
            min_tau,
        )

        self.stats['responses_received'] += len(responses)

        # Aggregate responses
        aggregated = self._aggregate_responses(responses)
        self.stats['responses_aggregated'] += 1

        return aggregated

    async def _find_qualified_nodes(
        self,
        topic: Optional[str],
        min_tau: float,
    ) -> List[Dict]:
        """Find nodes that meet trust threshold."""
        qualified = []

        # If topic provided, use DHT to find experts
        if topic and self.dht:
            experts = await self.dht.find_experts(topic)
            for expert in experts:
                tau = expert.get('tau', 0.5)
                # Use local score if we have it
                node_id = expert.get('node_id')
                if node_id in self.peer_scores:
                    tau = self.peer_scores[node_id]
                    expert['tau'] = tau

                if tau >= min_tau:
                    qualified.append(expert)

        # Add known peers that meet threshold
        if hasattr(self, 'dht') and self.dht:
            for bucket in self.dht.buckets:
                for node in bucket.nodes:
                    tau = self.peer_scores.get(node.node_id, node.tau)
                    if tau >= min_tau:
                        qualified.append({
                            'node_id': node.node_id,
                            'host': node.host,
                            'port': node.port,
                            'tau': tau,
                        })

        # Deduplicate
        seen = set()
        unique = []
        for node in qualified:
            node_id = node.get('node_id')
            if node_id and node_id not in seen:
                seen.add(node_id)
                unique.append(node)

        return unique

    async def _query_nodes(
        self,
        nodes: List[Dict],
        query_embedding: List[float],
        min_tau: float,
    ) -> List[QueryResult]:
        """Query multiple nodes concurrently."""
        import time

        async def query_single(node: Dict) -> Optional[QueryResult]:
            start = time.time()
            try:
                # This would use the P2P node to send query
                # For now, simulate with DHT request
                if self.dht:
                    response = await self.dht._send_request(
                        node['host'],
                        node['port'],
                        {
                            "cmd": "FIND_VALUE",
                            "key": hash(str(query_embedding[:5])),
                        },
                        timeout=self.query_timeout,
                    )

                    latency = (time.time() - start) * 1000

                    if response and response.get('status') == 'OK':
                        return QueryResult(
                            results=response.get('value', []),
                            source_node=node['node_id'],
                            tau=node['tau'],
                            latency_ms=latency,
                        )
            except Exception as e:
                print(f"Query to {node.get('node_id', 'unknown')[:8]} failed: {e}")

            return None

        # Query all nodes concurrently
        tasks = [query_single(node) for node in nodes]
        results = await asyncio.gather(*tasks)

        # Filter None results
        return [r for r in results if r is not None]

    def _aggregate_responses(
        self,
        responses: List[QueryResult],
    ) -> List[Dict]:
        """
        Aggregate responses weighted by Trust scores.

        Higher τ = more weight in final results.
        """
        if not responses:
            return []

        # Calculate total trust weight
        total_tau = sum(r.tau for r in responses)
        if total_tau == 0:
            total_tau = 1

        # Collect all results with weights
        weighted_results = []

        for response in responses:
            weight = response.tau / total_tau

            for result in response.results:
                if isinstance(result, dict):
                    # Apply trust weight to confidence/score
                    confidence = result.get('confidence', result.get('score', 0.5))
                    weighted_confidence = confidence * weight

                    weighted_results.append({
                        **result,
                        'weighted_confidence': weighted_confidence,
                        'source_node': response.source_node,
                        'source_tau': response.tau,
                        'trust_weight': weight,
                    })

        # Sort by weighted confidence
        weighted_results.sort(
            key=lambda r: r.get('weighted_confidence', 0),
            reverse=True,
        )

        return weighted_results

    def record_feedback(
        self,
        node_id: str,
        query_id: str,
        good: bool,
        notes: str = "",
    ):
        """
        Record feedback about a query response.

        Args:
            node_id: Node that provided the response
            query_id: Query identifier
            good: Whether the response was good
            notes: Optional feedback notes
        """
        # Update trust
        self.update_peer_trust(node_id, good)

        # Clean up pending query
        if query_id in self.pending_queries:
            del self.pending_queries[query_id]

    def get_trust_stats(self) -> Dict[str, Any]:
        """Get trust statistics."""
        if not self.peer_scores:
            return {
                'total_peers': 0,
                'avg_tau': 0.5,
                'high_trust_peers': 0,
                'low_trust_peers': 0,
            }

        scores = list(self.peer_scores.values())

        return {
            'total_peers': len(scores),
            'avg_tau': sum(scores) / len(scores),
            'max_tau': max(scores),
            'min_tau': min(scores),
            'high_trust_peers': len([s for s in scores if s >= 0.8]),
            'medium_trust_peers': len([s for s in scores if 0.5 <= s < 0.8]),
            'low_trust_peers': len([s for s in scores if s < 0.5]),
        }

    def get_peer_history(self, node_id: str) -> List[Dict]:
        """Get interaction history with a peer."""
        history = self.peer_history.get(node_id, [])
        return [
            {
                'timestamp': ts.isoformat(),
                'good_response': good,
            }
            for ts, good in history
        ]

    def get_top_peers(self, count: int = 10) -> List[Dict]:
        """Get top trusted peers."""
        sorted_peers = sorted(
            self.peer_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {
                'node_id': node_id,
                'tau': tau,
                'interactions': len(self.peer_history.get(node_id, [])),
            }
            for node_id, tau in sorted_peers[:count]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            'local_tau': self.tau,
            'trust_stats': self.get_trust_stats(),
            **self.stats,
        }


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA Trust Router Test")
        print("=" * 60)

        # Create router (no DHT for test)
        router = TrustRouter(dht=None, local_tau=0.8)

        print(f"\nLocal τ: {router.tau}")

        # Simulate peer interactions
        test_peers = [
            ("peer-1", True),
            ("peer-1", True),
            ("peer-1", True),
            ("peer-2", False),
            ("peer-2", True),
            ("peer-3", True),
            ("peer-3", True),
        ]

        print("\nSimulating peer interactions...")
        for peer_id, good in test_peers:
            router.update_peer_trust(peer_id, good)
            print(f"  {peer_id}: {'good' if good else 'bad'} → τ={router.get_peer_trust(peer_id):.3f}")

        print("\nTrust Stats:")
        stats = router.get_trust_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        print("\nTop Peers:")
        for peer in router.get_top_peers():
            print(f"  {peer['node_id']}: τ={peer['tau']:.3f}, interactions={peer['interactions']}")

        print("\nTrust Router ready!")

    asyncio.run(test())
