#!/usr/bin/env python3
"""
BAZINGA Distributed Query - Route Questions via DHT
=====================================================

Route queries to topic experts across the P2P network.

Flow:
1. Hash query to find topic space
2. Look up expert nodes via DHT
3. Send query to high-trust experts
4. Aggregate responses with φ-coherence
5. Return consensus answer

"Intelligence distributed, not controlled."

Author: Space (Abhishek/Abhilasia) 
License: MIT
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Constants
PHI = 1.618033988749895
QUERY_TIMEOUT = 30.0  # Seconds to wait for expert responses
MIN_EXPERTS = 1       # Minimum experts needed
MAX_EXPERTS = 5       # Maximum experts to query
COHERENCE_THRESHOLD = 0.7  # φ-coherence threshold for consensus


@dataclass
class ExpertResponse:
    """Response from an expert node."""
    node_id: str
    answer: str
    confidence: float
    trust_score: float
    latency_ms: float
    topics: List[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        """Calculate φ-weighted score."""
        return self.confidence * self.trust_score


@dataclass
class DistributedAnswer:
    """Aggregated answer from distributed query."""
    answer: str
    confidence: float
    expert_count: int
    consensus_achieved: bool
    responses: List[ExpertResponse] = field(default_factory=list)
    query_time_ms: float = 0.0
    topics_matched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "expert_count": self.expert_count,
            "consensus": self.consensus_achieved,
            "query_time_ms": self.query_time_ms,
            "topics": self.topics_matched,
        }


class DistributedQueryEngine:
    """
    Route queries through DHT to find and query topic experts.

    Usage:
        engine = DistributedQueryEngine(bridge)
        answer = await engine.query("What is quantum entanglement?")
        print(answer.answer)
    """

    def __init__(self, dht_bridge):
        """
        Initialize with DHT bridge.

        Args:
            dht_bridge: DHTBridge instance for peer discovery
        """
        self.bridge = dht_bridge
        self.stats = {
            "queries_sent": 0,
            "queries_answered": 0,
            "experts_queried": 0,
            "consensus_achieved": 0,
        }

    def _extract_topics(self, query: str) -> List[str]:
        """Extract potential topics from query."""
        # Simple keyword extraction
        # In production, use NLP for better topic extraction
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'about', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'}

        words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
        topics = [w for w in words if w not in stopwords and len(w) > 2]

        # Also add bigrams for compound topics
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                topics.append(f"{words[i]} {words[i+1]}")

        return topics[:5]  # Top 5 topics

    async def find_experts(self, query: str, count: int = MAX_EXPERTS) -> List[Any]:
        """Find expert nodes for a query."""
        topics = self._extract_topics(query)
        all_experts = []
        seen_ids = set()

        for topic in topics:
            experts = await self.bridge.find_experts(topic, count=count)
            for expert in experts:
                if expert.node_id not in seen_ids:
                    seen_ids.add(expert.node_id)
                    all_experts.append((expert, topic))

        # Sort by trust score (meritocratic)
        all_experts.sort(key=lambda x: x[0].trust_score, reverse=True)

        return all_experts[:count]

    async def query_expert(
        self,
        expert,
        topic: str,
        query: str,
        timeout: float = QUERY_TIMEOUT,
    ) -> Optional[ExpertResponse]:
        """Send query to a single expert node."""
        try:
            start = time.time()

            # Build query message
            request = {
                "cmd": "QUERY",
                "query": query,
                "topic": topic,
                "sender": self.bridge.dht.get_info().to_dict(),
            }

            # Send via DHT
            response = await self.bridge.dht._send_request(expert, request, timeout=timeout)

            latency_ms = (time.time() - start) * 1000

            if response and response.get("status") == "OK":
                return ExpertResponse(
                    node_id=expert.node_id.hex()[:16],
                    answer=response.get("answer", ""),
                    confidence=response.get("confidence", 0.5),
                    trust_score=expert.trust_score,
                    latency_ms=latency_ms,
                    topics=[topic],
                )

        except Exception:
            pass

        return None

    def _calculate_coherence(self, responses: List[ExpertResponse]) -> float:
        """Calculate φ-coherence between responses."""
        if len(responses) < 2:
            return 1.0 if responses else 0.0

        # Simple word overlap coherence
        # In production, use embedding similarity
        def tokenize(text: str) -> set:
            return set(text.lower().split())

        total_similarity = 0.0
        pairs = 0

        for i, r1 in enumerate(responses):
            for r2 in responses[i+1:]:
                words1 = tokenize(r1.answer)
                words2 = tokenize(r2.answer)

                if not words1 or not words2:
                    continue

                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0

                # Weight by trust scores
                weight = (r1.trust_score + r2.trust_score) / 2
                total_similarity += similarity * weight
                pairs += weight

        return total_similarity / pairs if pairs > 0 else 0.0

    def _synthesize_answer(self, responses: List[ExpertResponse]) -> str:
        """Synthesize final answer from expert responses."""
        if not responses:
            return "No expert responses received."

        if len(responses) == 1:
            return responses[0].answer

        # Weight by trust score and confidence
        weighted_responses = sorted(
            responses,
            key=lambda r: r.weighted_score,
            reverse=True
        )

        # Use highest-weighted answer as base
        best = weighted_responses[0]

        # If high coherence, just return best answer
        coherence = self._calculate_coherence(responses)
        if coherence > COHERENCE_THRESHOLD:
            return best.answer

        # Otherwise, note multiple perspectives
        return f"{best.answer}\n\n[Synthesized from {len(responses)} expert responses with {coherence:.2f} coherence]"

    async def query(
        self,
        question: str,
        min_experts: int = MIN_EXPERTS,
        max_experts: int = MAX_EXPERTS,
        timeout: float = QUERY_TIMEOUT,
    ) -> DistributedAnswer:
        """
        Send query to distributed network and aggregate responses.

        Args:
            question: The question to ask
            min_experts: Minimum experts needed for valid answer
            max_experts: Maximum experts to query
            timeout: Query timeout in seconds

        Returns:
            DistributedAnswer with aggregated result
        """
        start_time = time.time()
        self.stats["queries_sent"] += 1

        # Find experts
        experts_with_topics = await self.find_experts(question, count=max_experts)

        if not experts_with_topics:
            return DistributedAnswer(
                answer="No experts found for this topic. Try joining the network with 'bazinga --join'.",
                confidence=0.0,
                expert_count=0,
                consensus_achieved=False,
                query_time_ms=(time.time() - start_time) * 1000,
            )

        # Query experts in parallel
        tasks = []
        for expert, topic in experts_with_topics:
            tasks.append(self.query_expert(expert, topic, question, timeout))

        responses = await asyncio.gather(*tasks)
        valid_responses = [r for r in responses if r is not None]

        self.stats["experts_queried"] += len(experts_with_topics)

        # Check if we have enough responses
        if len(valid_responses) < min_experts:
            return DistributedAnswer(
                answer=f"Insufficient expert responses ({len(valid_responses)}/{min_experts} required).",
                confidence=0.0,
                expert_count=len(valid_responses),
                consensus_achieved=False,
                responses=valid_responses,
                query_time_ms=(time.time() - start_time) * 1000,
            )

        # Calculate coherence and synthesize
        coherence = self._calculate_coherence(valid_responses)
        consensus = coherence >= COHERENCE_THRESHOLD
        answer = self._synthesize_answer(valid_responses)

        if consensus:
            self.stats["consensus_achieved"] += 1

        self.stats["queries_answered"] += 1

        # Collect all matched topics
        all_topics = []
        for r in valid_responses:
            all_topics.extend(r.topics)

        return DistributedAnswer(
            answer=answer,
            confidence=coherence,
            expert_count=len(valid_responses),
            consensus_achieved=consensus,
            responses=valid_responses,
            query_time_ms=(time.time() - start_time) * 1000,
            topics_matched=list(set(all_topics)),
        )

    async def register_as_expert(self, topics: List[str]):
        """Register this node as an expert in given topics."""
        for topic in topics:
            await self.bridge.announce_knowledge(topic)

    def get_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["queries_answered"] / self.stats["queries_sent"]
                if self.stats["queries_sent"] > 0 else 0.0
            ),
            "consensus_rate": (
                self.stats["consensus_achieved"] / self.stats["queries_answered"]
                if self.stats["queries_answered"] > 0 else 0.0
            ),
        }


# =============================================================================
# QUERY REQUEST HANDLER (Add to KademliaNode)
# =============================================================================

async def handle_query_request(node, request: Dict) -> Dict:
    """
    Handle incoming QUERY request from peer.

    This should be integrated into KademliaNode._handle_request()
    """
    query = request.get("query", "")
    topic = request.get("topic", "")

    # Check if we're an expert on this topic
    # For now, just acknowledge - actual AI response integration needed
    if hasattr(node, 'query_handler') and node.query_handler:
        answer, confidence = await node.query_handler(query, topic)
        return {
            "status": "OK",
            "answer": answer,
            "confidence": confidence,
            "topic": topic,
            "sender": node.get_info().to_dict(),
        }

    return {
        "status": "NOT_EXPERT",
        "message": "This node is not configured to answer queries",
        "sender": node.get_info().to_dict(),
    }
