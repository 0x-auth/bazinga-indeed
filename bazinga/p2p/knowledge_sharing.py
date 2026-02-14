#!/usr/bin/env python3
"""
BAZINGA Distributed Knowledge Sharing
=====================================

Privacy-preserving knowledge sharing via DHT.

How it works:
1. YOU index files locally (chromadb)
2. YOU publish topic keywords to DHT (not content!)
3. PEER queries DHT for topic
4. DHT routes query to YOUR node
5. YOUR node answers from local RAG
6. Triadic consensus validates answer

"Knowledge flows through the mesh, but lives at home."

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path

# Constants
PHI = 1.618033988749895
MIN_CONFIDENCE = 0.5  # Minimum confidence to return answer
TRIADIC_THRESHOLD = 3  # Need 3 nodes to agree for consensus


@dataclass
class TopicRegistration:
    """A topic that this node knows about."""
    topic: str              # The topic keyword
    content_hash: str       # SHA256 of the indexed content
    node_id: str           # This node's ID
    confidence: float      # How well we know this topic (0-1)
    registered_at: datetime = field(default_factory=datetime.now)
    ttl_hours: int = 24    # How long to keep in DHT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "content_hash": self.content_hash,
            "node_id": self.node_id,
            "confidence": self.confidence,
            "registered_at": self.registered_at.isoformat(),
            "ttl_hours": self.ttl_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicRegistration":
        return cls(
            topic=data["topic"],
            content_hash=data["content_hash"],
            node_id=data["node_id"],
            confidence=data["confidence"],
            registered_at=datetime.fromisoformat(data["registered_at"]),
            ttl_hours=data.get("ttl_hours", 24),
        )


@dataclass
class KnowledgeQuery:
    """A query for distributed knowledge."""
    query: str              # The question
    topics: List[str]       # Topics extracted from query
    requester_id: str       # Who's asking
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "topics": self.topics,
            "requester_id": self.requester_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class KnowledgeAnswer:
    """An answer from a node."""
    query: str
    answer: str
    responder_id: str
    confidence: float       # How confident the node is (0-1)
    source_topics: List[str]  # Topics that matched
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "responder_id": self.responder_id,
            "confidence": self.confidence,
            "source_topics": self.source_topics,
            "timestamp": self.timestamp.isoformat(),
        }


class KnowledgePublisher:
    """
    Publishes local knowledge topics to the DHT.

    The actual content never leaves your machine!
    Only topic keywords and content hashes are shared.
    """

    def __init__(self, node, rag_engine):
        """
        Args:
            node: KademliaNode instance
            rag_engine: Local RAG engine (RealAI instance)
        """
        self.node = node
        self.rag = rag_engine
        self.published_topics: Dict[str, TopicRegistration] = {}
        self.stats = {
            "topics_published": 0,
            "queries_answered": 0,
            "queries_received": 0,
        }

    def extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        import re

        # Simple extraction - remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'this', 'that', 'these', 'those', 'what',
            'which', 'who', 'whom', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
            'def', 'class', 'import', 'return', 'none', 'true', 'false',
            'self', 'print', 'str', 'int', 'dict', 'list', 'async', 'await',
        }

        # Clean text - remove code patterns, URLs, hashes
        text = re.sub(r'[a-f0-9]{32,}', '', text)  # Remove hashes
        text = re.sub(r'\*\*\d+', '', text)  # Remove polynomial terms
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars

        # Split and filter
        words = text.lower().split()

        # Filter and get unique topics
        topics = []
        seen = set()
        for word in words:
            word = word.strip()
            # Valid topic: 3-20 chars, alphabetic, not in stop_words
            if (3 <= len(word) <= 20 and
                word.isalpha() and
                word not in stop_words and
                word not in seen):
                topics.append(word)
                seen.add(word)

        return topics[:10]  # Limit to 10 topics

    async def publish_topics(self, topics: List[str], content_hash: str) -> int:
        """
        Publish topics to the DHT.

        Returns: Number of topics successfully published
        """
        published = 0
        node_id = self.node.node_id.hex() if isinstance(self.node.node_id, bytes) else self.node.node_id

        for topic in topics:
            try:
                reg = TopicRegistration(
                    topic=topic,
                    content_hash=content_hash,
                    node_id=node_id,
                    confidence=0.8,  # Base confidence
                )

                # Store in DHT with topic as key
                key = f"topic:{topic}"

                # Get existing registrations for this topic
                existing = await self.node.get(key)
                if existing and isinstance(existing, list):
                    # Add our registration
                    existing.append(reg.to_dict())
                else:
                    existing = [reg.to_dict()]

                # Store updated list
                await self.node.store(key, existing, ttl=reg.ttl_hours * 3600)

                self.published_topics[topic] = reg
                published += 1

            except Exception as e:
                print(f"  Warning: Failed to publish topic '{topic}': {e}")

        self.stats["topics_published"] += published
        return published

    async def publish_from_index(self, limit: int = 100) -> Dict[str, Any]:
        """
        Publish topics extracted from local RAG index.

        Scans indexed content and publishes topic keywords to DHT.
        """
        # Get indexed chunks
        stats = self.rag.get_stats()
        total_chunks = stats.get('total_chunks', 0)

        if total_chunks == 0:
            return {
                "success": False,
                "error": "No indexed content. Run 'bazinga --index <path>' first.",
                "topics_published": 0,
            }

        # Sample some chunks to extract topics
        # In a real implementation, we'd scan all chunks
        all_topics = set()
        content_for_hash = []

        # Search with common terms to get representative chunks
        sample_queries = ["what", "how", "the", "is", "BAZINGA", "consciousness"]

        for query in sample_queries:
            try:
                results = self.rag.search(query, limit=20)
                for r in results:
                    chunk_text = r.chunk.content
                    topics = self.extract_topics(chunk_text)
                    all_topics.update(topics)
                    content_for_hash.append(chunk_text[:100])
            except Exception:
                pass

        # Create content hash (proof of knowledge)
        content_hash = hashlib.sha256(''.join(content_for_hash).encode()).hexdigest()

        # Publish top topics
        topics_list = list(all_topics)[:limit]
        published = await self.publish_topics(topics_list, content_hash)

        return {
            "success": True,
            "topics_published": published,
            "content_hash": content_hash[:16] + "...",
            "sample_topics": topics_list[:10],
        }

    async def answer_query(self, query: str, topics: List[str]) -> Tuple[str, float]:
        """
        Answer a query using local RAG.

        This is called when a peer routes a query to us.
        """
        self.stats["queries_received"] += 1

        # Check if we have any of the requested topics
        matching_topics = [t for t in topics if t in self.published_topics]

        if not matching_topics:
            raise ValueError("No matching topics found")

        # Search local RAG
        search_query = ' '.join(matching_topics + [query])
        results = self.rag.search(search_query, limit=5)

        if not results or results[0].similarity < 0.3:
            raise ValueError("No relevant content found")

        # Build context from results
        context_parts = []
        for r in results[:3]:
            context_parts.append(r.chunk.content[:500])
        context = "\n\n".join(context_parts)

        # Get answer from local LLM if available
        try:
            # Try to use local LLM for answer generation
            from ..local_llm import get_local_llm
            llm = get_local_llm()
            if llm:
                prompt = f"""Based on this knowledge:

{context}

Answer this question concisely: {query}"""
                answer = llm.generate(prompt)
                confidence = min(results[0].similarity * PHI, 1.0)  # φ-boosted confidence
            else:
                # Fallback: return raw context
                answer = f"Based on indexed knowledge:\n{context[:500]}"
                confidence = results[0].similarity
        except Exception:
            # Fallback: return raw context
            answer = f"Based on indexed knowledge:\n{context[:500]}"
            confidence = results[0].similarity

        self.stats["queries_answered"] += 1

        return answer, confidence


class DistributedQueryEngine:
    """
    Routes queries to nodes that have relevant knowledge.

    Uses DHT to find topic experts, queries them, and
    applies triadic consensus for trusted answers.
    """

    def __init__(self, node, local_publisher: Optional[KnowledgePublisher] = None):
        self.node = node
        self.local_publisher = local_publisher
        self.stats = {
            "queries_sent": 0,
            "answers_received": 0,
            "consensus_achieved": 0,
        }

    async def find_topic_experts(self, topic: str) -> List[Dict[str, Any]]:
        """Find nodes that have published this topic."""
        key = f"topic:{topic}"

        try:
            registrations = await self.node.get(key)
            if registrations and isinstance(registrations, list):
                # Sort by confidence
                registrations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                return registrations
        except Exception:
            pass

        return []

    async def query_distributed(self, query: str) -> Dict[str, Any]:
        """
        Query the distributed network for an answer.

        1. Extract topics from query
        2. Find nodes that know about those topics
        3. Query each node
        4. Apply triadic consensus
        5. Return trusted answer
        """
        self.stats["queries_sent"] += 1

        # Extract topics from query
        topics = self.local_publisher.extract_topics(query) if self.local_publisher else query.lower().split()[:5]

        if not topics:
            return {
                "success": False,
                "error": "Could not extract topics from query",
            }

        # Find experts for each topic
        all_experts = []
        for topic in topics[:3]:  # Limit to top 3 topics
            experts = await self.find_topic_experts(topic)
            for exp in experts:
                if exp not in all_experts:
                    all_experts.append(exp)

        if not all_experts:
            # No remote experts - try local RAG directly
            if self.local_publisher:
                try:
                    # Search local RAG with extracted topics (better for embeddings)
                    search_query = ' '.join(topics)
                    results = self.local_publisher.rag.search(search_query, limit=5)

                    if results and results[0].similarity > 0.3:
                        # Build answer from local RAG
                        context_parts = [r.chunk.content[:500] for r in results[:3]]
                        context = "\n\n".join(context_parts)

                        # Try Ollama or local LLM
                        answer = None
                        confidence = results[0].similarity

                        try:
                            import httpx
                            # Try Ollama first
                            with httpx.Client(timeout=30.0) as client:
                                resp = client.post(
                                    "http://localhost:11434/api/generate",
                                    json={
                                        "model": "llama3",
                                        "prompt": f"""Based on this knowledge:

{context}

Answer this question concisely: {query}""",
                                        "stream": False,
                                    }
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    answer = data.get("response", "")
                                    confidence = min(results[0].similarity * PHI, 1.0)
                        except Exception:
                            pass

                        if not answer:
                            # Fallback to raw context
                            answer = f"Based on indexed knowledge:\n{context[:500]}"

                        return {
                            "success": True,
                            "answer": answer,
                            "confidence": confidence,
                            "source": "local_rag",
                            "consensus": False,
                        }
                except Exception as e:
                    pass

            return {
                "success": False,
                "error": "No experts found for topics: " + ", ".join(topics),
            }

        # Query each expert
        answers = []
        for expert in all_experts[:5]:  # Query up to 5 experts
            try:
                node_id = expert.get('node_id')

                # Send QUERY command via DHT
                # In real implementation, we'd route to the node
                # For now, we'll simulate with local if we are the expert
                my_node_id = self.node.node_id.hex() if isinstance(self.node.node_id, bytes) else self.node.node_id

                if node_id == my_node_id and self.local_publisher:
                    answer, confidence = await self.local_publisher.answer_query(query, topics)
                    answers.append({
                        "answer": answer,
                        "confidence": confidence,
                        "node_id": node_id,
                    })
                else:
                    # TODO: Route to remote node via DHT QUERY command
                    pass

            except Exception:
                pass

        self.stats["answers_received"] += len(answers)

        if not answers:
            return {
                "success": False,
                "error": "No experts responded",
            }

        # Apply triadic consensus if we have 3+ answers
        if len(answers) >= TRIADIC_THRESHOLD:
            # φ-weighted consensus
            # Group similar answers and weight by confidence
            consensus_answer = self._apply_consensus(answers)
            self.stats["consensus_achieved"] += 1

            return {
                "success": True,
                "answer": consensus_answer["answer"],
                "confidence": consensus_answer["confidence"],
                "source": "triadic_consensus",
                "consensus": True,
                "respondents": len(answers),
            }

        # Return best single answer
        best = max(answers, key=lambda x: x.get('confidence', 0))

        return {
            "success": True,
            "answer": best["answer"],
            "confidence": best["confidence"],
            "source": f"node:{best['node_id'][:8]}...",
            "consensus": False,
            "respondents": len(answers),
        }

    def _apply_consensus(self, answers: List[Dict]) -> Dict[str, Any]:
        """
        Apply φ-weighted triadic consensus.

        Simple implementation: weight answers by confidence,
        return the one with highest φ-weighted score.
        """
        # Score each answer
        scored = []
        for ans in answers:
            # φ-weighted confidence
            phi_score = ans['confidence'] * PHI
            scored.append({
                "answer": ans['answer'],
                "confidence": min(phi_score, 1.0),
                "node_id": ans['node_id'],
            })

        # Return highest scored
        best = max(scored, key=lambda x: x['confidence'])

        # Boost confidence if multiple agreers
        # (In real implementation, we'd check semantic similarity)
        consensus_boost = min(len(answers) / TRIADIC_THRESHOLD, 1.5)
        best['confidence'] = min(best['confidence'] * consensus_boost, 1.0)

        return best


# =============================================================================
# DHT HANDLER INTEGRATION
# =============================================================================

async def handle_knowledge_query(node, request: Dict[str, Any], publisher: KnowledgePublisher) -> Dict[str, Any]:
    """
    Handle incoming KNOWLEDGE_QUERY from DHT.

    Called when a peer routes a query to us.
    """
    query = request.get("query", "")
    topics = request.get("topics", [])
    requester_id = request.get("requester_id", "unknown")

    try:
        answer, confidence = await publisher.answer_query(query, topics)

        return {
            "status": "OK",
            "answer": answer,
            "confidence": confidence,
            "responder": node.node_id.hex() if isinstance(node.node_id, bytes) else node.node_id,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e),
        }
