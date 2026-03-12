#!/usr/bin/env python3
"""
BAZINGA Mesh Query - Ask the Network, Get Collective Intelligence
=================================================================

When you ask BAZINGA a question, it can also ask its discovered peers.
Peers respond with their own LLM answers. BAZINGA merges all responses
using phi-weighted scoring into one collective answer.

Architecture:
- Each node runs a TCP QueryServer on its P2P port
- When a query comes in from chat, MeshQuery fans out to known peers
- Responses are collected with timeout, merged by trust/confidence
- Trust scores updated based on response quality (feedback loop)
- Peer lists exchanged during queries (gossip layer)
- Session context sent to peers for follow-up questions

"One mind is good. A mesh of minds is phi times better."

Author: Space (Abhishek/Abhilasia)
License: MIT
"""

import asyncio
import json
import time
import socket
import struct
import hashlib
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field

# Constants
PHI = 1.618033988749895
MESH_QUERY_TIMEOUT = 15.0  # Max wait for peer responses
MESH_HEADER = b"BZMQ"  # BAZINGA Mesh Query header
MESH_VERSION = 2  # Bumped for gossip + context support
MAX_RESPONSE_SIZE = 128 * 1024  # 128KB max response (up from 64KB for context)


@dataclass
class PeerAnswer:
    """Answer from a single peer in the mesh."""
    node_id: str
    ip: str
    port: int
    answer: str
    confidence: float
    latency_ms: float
    source: str = "unknown"  # 'groq', 'gemini', 'local', etc.

    @property
    def weighted_score(self) -> float:
        """phi-weighted score favoring confidence and low latency."""
        latency_factor = 1.0 / (1.0 + self.latency_ms / 1000.0)
        return self.confidence * latency_factor * PHI


@dataclass
class MeshAnswer:
    """Collective answer from the mesh."""
    local_answer: str
    peer_answers: List[PeerAnswer] = field(default_factory=list)
    merged_answer: str = ""
    peer_count: int = 0
    total_time_ms: float = 0.0
    coherence: float = 0.0

    @property
    def has_peers(self) -> bool:
        return len(self.peer_answers) > 0


def _encode_message(msg_type: str, payload: dict) -> bytes:
    """Encode a mesh query message with header."""
    data = json.dumps({"type": msg_type, **payload}).encode("utf-8")
    # Header: BZMQ + version(1B) + length(4B) + data
    header = MESH_HEADER + struct.pack("!BI", MESH_VERSION, len(data))
    return header + data


def _decode_message(raw: bytes) -> Optional[dict]:
    """Decode a mesh query message."""
    if len(raw) < 9 or raw[:4] != MESH_HEADER:
        return None
    length = struct.unpack("!I", raw[5:9])[0]
    if len(raw) < 9 + length:
        return None
    try:
        return json.loads(raw[9:9 + length].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class QueryServer:
    """
    TCP server that listens for mesh queries from other BAZINGA nodes.

    Handles:
    - QUERY: Answer a question using local LLM
    - QUERY_CTX: Answer with conversation context
    - PING: Liveness check
    - GOSSIP: Exchange peer lists
    """

    def __init__(self, port: int, node_id: str):
        self.port = port
        self.node_id = node_id
        self._server: Optional[asyncio.AbstractServer] = None
        self._query_handler: Optional[Callable[[str], Awaitable[dict]]] = None
        self.queries_served = 0

    def set_handler(self, handler: Callable[[str], Awaitable[dict]]):
        """
        Set the query handler function.

        handler(question: str) -> {"answer": str, "confidence": float, "source": str}
        """
        self._query_handler = handler

    async def start(self):
        """Start the query server."""
        try:
            self._server = await asyncio.start_server(
                self._handle_connection, "0.0.0.0", self.port
            )
            return True
        except OSError:
            # Port in use
            return False

    async def stop(self):
        """Stop the query server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming request from a peer."""
        try:
            # Read header (9 bytes)
            header = await asyncio.wait_for(reader.readexactly(9), timeout=5.0)
            if header[:4] != MESH_HEADER:
                writer.close()
                return

            length = struct.unpack("!I", header[5:9])[0]
            if length > MAX_RESPONSE_SIZE:
                writer.close()
                return

            data = await asyncio.wait_for(reader.readexactly(length), timeout=5.0)
            msg = json.loads(data.decode("utf-8"))
            msg_type = msg.get("type", "")

            # --- QUERY: Answer a question ---
            if msg_type in ("QUERY", "QUERY_CTX") and self._query_handler:
                question = msg.get("question", "")

                # Context pinning: prepend conversation context if provided
                context = msg.get("context", "")
                if context and msg_type == "QUERY_CTX":
                    question = f"Previous conversation:\n{context}\n\nCurrent question: {question}"

                result = await asyncio.wait_for(
                    self._query_handler(question),
                    timeout=MESH_QUERY_TIMEOUT - 2
                )

                # Build response with our peer list for gossip
                response_payload = {
                    "node_id": self.node_id,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.5),
                    "source": result.get("source", "unknown"),
                }

                # Attach our known peers for gossip (top 5 by trust)
                try:
                    from .persistence import get_persistence_manager
                    pm = get_persistence_manager()
                    known = pm.get_known_peers(limit=5, max_age_hours=2.0)
                    response_payload["peers"] = [
                        {"node_id": p.node_id, "ip": p.ip, "port": p.port, "trust": p.trust_score}
                        for p in known
                    ]
                except Exception:
                    pass

                response = _encode_message("ANSWER", response_payload)
                writer.write(response)
                await writer.drain()
                self.queries_served += 1

                # Save the querying peer if we got their info
                sender_id = msg.get("sender", "")
                if sender_id:
                    try:
                        addr = writer.get_extra_info("peername")
                        if addr:
                            from .persistence import get_persistence_manager, PeerRecord
                            pm = get_persistence_manager()
                            pm.save_peer(PeerRecord(
                                node_id=sender_id,
                                ip=addr[0],
                                port=msg.get("sender_port", 5151),
                                last_seen=time.time(),
                                trust_score=0.5,
                            ))
                    except Exception:
                        pass

            # --- PING: Liveness check ---
            elif msg_type == "PING":
                response = _encode_message("PONG", {
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                })
                writer.write(response)
                await writer.drain()

            # --- GOSSIP: Exchange peer lists ---
            elif msg_type == "GOSSIP":
                # Ingest their peers
                incoming_peers = msg.get("peers", [])
                self._ingest_gossip_peers(incoming_peers, msg.get("sender", ""))

                # Send back our peers
                try:
                    from .persistence import get_persistence_manager
                    pm = get_persistence_manager()
                    known = pm.get_known_peers(limit=10, max_age_hours=4.0)
                    our_peers = [
                        {"node_id": p.node_id, "ip": p.ip, "port": p.port, "trust": p.trust_score}
                        for p in known
                    ]
                except Exception:
                    our_peers = []

                response = _encode_message("GOSSIP_ACK", {
                    "node_id": self.node_id,
                    "peers": our_peers,
                })
                writer.write(response)
                await writer.drain()

        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _ingest_gossip_peers(self, peers: list, sender_id: str):
        """Save gossiped peers to persistence with reduced trust."""
        try:
            from .persistence import get_persistence_manager, PeerRecord
            pm = get_persistence_manager()
            for p in peers:
                node_id = p.get("node_id", "")
                if not node_id or node_id == self.node_id:
                    continue
                # Gossiped peers start with lower trust (0.3) since we haven't verified them
                pm.save_peer(PeerRecord(
                    node_id=node_id,
                    ip=p.get("ip", ""),
                    port=p.get("port", 5151),
                    last_seen=time.time(),
                    trust_score=min(p.get("trust", 0.3), 0.4),  # Cap at 0.4 for gossip
                ))
                pm.log_discovery(
                    event_type="gossip",
                    node_id=node_id,
                    ip=p.get("ip", ""),
                    port=p.get("port", 5151),
                    details=f"via {sender_id[:8]}",
                )
        except Exception:
            pass


class MeshQuery:
    """
    Fan-out queries to discovered peers and merge responses.

    Features:
    - Parallel fan-out to all known peers
    - Trust-based peer selection (higher trust = priority)
    - Context pinning for follow-up questions
    - Trust feedback after each query (good answers = more trust)
    - Peer gossip: learn about new peers from query responses

    Usage:
        mesh = MeshQuery(node_id="abc123")
        answer = await mesh.query_mesh("What is phi?", local_answer="The golden ratio...")
    """

    def __init__(self, node_id: str, port: int = 5151, timeout: float = MESH_QUERY_TIMEOUT):
        self.node_id = node_id
        self.port = port
        self.timeout = timeout
        self._session_id = hashlib.sha256(f"{node_id}{time.time()}".encode()).hexdigest()[:12]
        self.stats = {
            "queries_sent": 0,
            "responses_received": 0,
            "timeouts": 0,
            "trust_updates": 0,
            "peers_learned": 0,
        }

    async def _query_peer(
        self,
        ip: str,
        port: int,
        question: str,
        context: str = "",
    ) -> Optional[PeerAnswer]:
        """Send query to a single peer and get response."""
        start = time.time()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=3.0
            )

            # Choose message type based on whether we have context
            msg_type = "QUERY_CTX" if context else "QUERY"
            payload = {
                "question": question,
                "sender": self.node_id,
                "sender_port": self.port,
                "session_id": self._session_id,
            }
            if context:
                payload["context"] = context

            msg = _encode_message(msg_type, payload)
            writer.write(msg)
            await writer.drain()

            # Read response header
            header = await asyncio.wait_for(reader.readexactly(9), timeout=self.timeout)
            if header[:4] != MESH_HEADER:
                writer.close()
                return None

            length = struct.unpack("!I", header[5:9])[0]
            if length > MAX_RESPONSE_SIZE:
                writer.close()
                return None

            data = await asyncio.wait_for(reader.readexactly(length), timeout=self.timeout)
            response = json.loads(data.decode("utf-8"))

            writer.close()
            await writer.wait_closed()

            latency_ms = (time.time() - start) * 1000

            if response.get("type") == "ANSWER":
                self.stats["responses_received"] += 1

                # Ingest gossiped peers from response
                gossiped_peers = response.get("peers", [])
                if gossiped_peers:
                    self._ingest_gossip(gossiped_peers, response.get("node_id", ""))

                return PeerAnswer(
                    node_id=response.get("node_id", "unknown"),
                    ip=ip,
                    port=port,
                    answer=response.get("answer", ""),
                    confidence=response.get("confidence", 0.5),
                    latency_ms=latency_ms,
                    source=response.get("source", "unknown"),
                )

        except (asyncio.TimeoutError, ConnectionRefusedError, ConnectionError, OSError):
            self.stats["timeouts"] += 1
        except Exception:
            self.stats["timeouts"] += 1

        return None

    def _ingest_gossip(self, peers: list, source_node: str):
        """Save peers learned through gossip with reduced trust."""
        try:
            from .persistence import get_persistence_manager, PeerRecord
            pm = get_persistence_manager()
            for p in peers:
                node_id = p.get("node_id", "")
                if not node_id or node_id == self.node_id:
                    continue
                # Check if we already know this peer
                existing = pm.get_peer(node_id)
                if existing:
                    # Update last_seen but don't overwrite trust
                    pm.update_peer_seen(node_id)
                else:
                    # New peer from gossip - start with low trust
                    pm.save_peer(PeerRecord(
                        node_id=node_id,
                        ip=p.get("ip", ""),
                        port=p.get("port", 5151),
                        last_seen=time.time(),
                        trust_score=min(p.get("trust", 0.3), 0.4),
                    ))
                    pm.log_discovery(
                        event_type="gossip",
                        node_id=node_id,
                        ip=p.get("ip", ""),
                        port=p.get("port", 5151),
                        details=f"via query response from {source_node[:8]}",
                    )
                    self.stats["peers_learned"] += 1
        except Exception:
            pass

    def _update_trust(self, peer_answers: List[PeerAnswer], coherence: float):
        """
        Update trust scores based on mesh query results.

        Rules:
        - Peers who responded get a small trust bump (+0.02)
        - High coherence (>0.5) = extra trust boost (+0.03 * coherence)
        - Low coherence (<0.3) with local = slight trust decrease (-0.01)
        - Peers who timed out get trust decrease (handled by caller marking as None)
        """
        try:
            from .persistence import get_persistence_manager
            pm = get_persistence_manager()

            for peer in peer_answers:
                # Base reward for responding at all
                delta = 0.02

                if coherence > 0.5:
                    # Bonus for agreeing with consensus
                    delta += 0.03 * coherence
                elif coherence < 0.3:
                    # Slight penalty for disagreeing heavily
                    delta -= 0.01

                # Confidence bonus
                if peer.confidence > 0.8:
                    delta += 0.01

                # Fast response bonus
                if peer.latency_ms < 2000:
                    delta += 0.01

                pm.update_peer_trust(peer.node_id, delta)
                self.stats["trust_updates"] += 1

        except Exception:
            pass

    def _penalize_unresponsive(self, all_peers: list, responsive_ids: set):
        """Decrease trust for peers that didn't respond."""
        try:
            from .persistence import get_persistence_manager
            pm = get_persistence_manager()
            for ip, port, node_id in all_peers:
                if node_id not in responsive_ids:
                    pm.update_peer_trust(node_id, -0.03)  # Small penalty for being offline
        except Exception:
            pass

    async def query_mesh(
        self,
        question: str,
        local_answer: str,
        local_source: str = "local",
        context: str = "",
    ) -> MeshAnswer:
        """
        Fan out question to all known peers and merge with local answer.

        Args:
            question: The question to ask
            local_answer: Our local LLM's answer
            local_source: Source of local answer (groq, gemini, local, etc.)
            context: Conversation context for follow-up questions

        Returns:
            MeshAnswer with local + peer answers merged
        """
        start_time = time.time()
        self.stats["queries_sent"] += 1

        # Get known peers from persistence, sorted by trust
        peers = []
        all_peers_info = []
        try:
            from .persistence import get_persistence_manager
            pm = get_persistence_manager()
            peer_records = pm.get_known_peers(limit=5, max_age_hours=1.0, min_trust=0.1)
            for p in peer_records:
                if p.node_id != self.node_id:
                    peers.append((p.ip, p.port))
                    all_peers_info.append((p.ip, p.port, p.node_id))
        except Exception:
            pass

        if not peers:
            return MeshAnswer(
                local_answer=local_answer,
                merged_answer=local_answer,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        # Fan out to all peers in parallel (with context if available)
        tasks = [self._query_peer(ip, port, question, context) for ip, port in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        peer_answers = [r for r in results if isinstance(r, PeerAnswer)]
        total_time_ms = (time.time() - start_time) * 1000

        if not peer_answers:
            # Penalize all unresponsive peers
            self._penalize_unresponsive(all_peers_info, set())
            return MeshAnswer(
                local_answer=local_answer,
                merged_answer=local_answer,
                total_time_ms=total_time_ms,
            )

        # Calculate coherence between local and peer answers
        coherence = self._calculate_coherence(local_answer, peer_answers)

        # Update trust based on response quality
        responsive_ids = {p.node_id for p in peer_answers}
        self._update_trust(peer_answers, coherence)
        self._penalize_unresponsive(all_peers_info, responsive_ids)

        # Merge answers
        merged = self._merge_answers(local_answer, local_source, peer_answers, coherence)

        return MeshAnswer(
            local_answer=local_answer,
            peer_answers=peer_answers,
            merged_answer=merged,
            peer_count=len(peer_answers),
            total_time_ms=total_time_ms,
            coherence=coherence,
        )

    async def gossip_peers(self, target_ip: str, target_port: int):
        """
        Exchange peer lists with a specific node.

        Sends our known peers and receives theirs.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target_ip, target_port),
                timeout=3.0,
            )

            # Send our peers
            from .persistence import get_persistence_manager
            pm = get_persistence_manager()
            known = pm.get_known_peers(limit=10, max_age_hours=4.0)
            our_peers = [
                {"node_id": p.node_id, "ip": p.ip, "port": p.port, "trust": p.trust_score}
                for p in known
            ]

            msg = _encode_message("GOSSIP", {
                "sender": self.node_id,
                "peers": our_peers,
            })
            writer.write(msg)
            await writer.drain()

            # Read their response
            header = await asyncio.wait_for(reader.readexactly(9), timeout=5.0)
            if header[:4] == MESH_HEADER:
                length = struct.unpack("!I", header[5:9])[0]
                if length <= MAX_RESPONSE_SIZE:
                    data = await asyncio.wait_for(reader.readexactly(length), timeout=5.0)
                    response = json.loads(data.decode("utf-8"))

                    if response.get("type") == "GOSSIP_ACK":
                        self._ingest_gossip(response.get("peers", []), response.get("node_id", ""))

            writer.close()
            await writer.wait_closed()

        except Exception:
            pass

    def _calculate_coherence(self, local: str, peers: List[PeerAnswer]) -> float:
        """Calculate phi-coherence between local answer and peer answers."""
        local_words = set(local.lower().split())
        if not local_words:
            return 0.0

        total_sim = 0.0
        for peer in peers:
            peer_words = set(peer.answer.lower().split())
            if not peer_words:
                continue
            intersection = len(local_words & peer_words)
            union = len(local_words | peer_words)
            total_sim += (intersection / union) if union > 0 else 0.0

        return total_sim / len(peers) if peers else 0.0

    def _merge_answers(
        self,
        local: str,
        local_source: str,
        peers: List[PeerAnswer],
        coherence: float,
    ) -> str:
        """Merge local and peer answers into a collective response."""
        if coherence > 0.7:
            # High agreement - local answer is good, just note consensus
            return f"{local}\n\n[Mesh consensus: {len(peers)+1} nodes agree (coherence: {coherence:.2f})]"

        if coherence > 0.4:
            # Moderate agreement - show best peer perspective
            best_peer = max(peers, key=lambda p: p.weighted_score)
            return (
                f"{local}\n\n"
                f"--- Network perspective ({best_peer.node_id[:8]}...) ---\n"
                f"{best_peer.answer}\n\n"
                f"[Mesh query: {len(peers)+1} nodes, coherence: {coherence:.2f}]"
            )

        # Low agreement - show local + all unique perspectives
        lines = [local, "", "--- Network perspectives ---"]
        for peer in sorted(peers, key=lambda p: p.weighted_score, reverse=True)[:3]:
            lines.append(f"\nNode {peer.node_id[:8]}... ({peer.source}, {peer.latency_ms:.0f}ms):")
            lines.append(peer.answer)

        lines.append(f"\n[Mesh query: {len(peers)+1} nodes, coherence: {coherence:.2f}]")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        return self.stats.copy()
