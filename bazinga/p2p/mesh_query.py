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
- User sees both local answer AND network perspective

"One mind is good. A mesh of minds is φ times better."

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
MESH_VERSION = 1
MAX_RESPONSE_SIZE = 64 * 1024  # 64KB max response


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
    version = raw[4]
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

    When a peer asks a question, this server processes it through the
    local BAZINGA instance and returns the answer.
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
            # Port in use - that's okay, might be used by PhiPulse recv
            return False

    async def stop(self):
        """Stop the query server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming query from a peer."""
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

            if msg.get("type") == "QUERY" and self._query_handler:
                question = msg.get("question", "")
                sender = msg.get("sender", "unknown")

                # Process query through local BAZINGA
                result = await asyncio.wait_for(
                    self._query_handler(question),
                    timeout=MESH_QUERY_TIMEOUT - 2  # Leave 2s buffer
                )

                response = _encode_message("ANSWER", {
                    "node_id": self.node_id,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.5),
                    "source": result.get("source", "unknown"),
                })

                writer.write(response)
                await writer.drain()
                self.queries_served += 1

            elif msg.get("type") == "PING":
                response = _encode_message("PONG", {
                    "node_id": self.node_id,
                    "timestamp": time.time(),
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


class MeshQuery:
    """
    Fan-out queries to discovered peers and merge responses.

    Usage:
        mesh = MeshQuery(node_id="abc123")
        answer = await mesh.query_mesh("What is phi?", local_answer="The golden ratio...")
    """

    def __init__(self, node_id: str, timeout: float = MESH_QUERY_TIMEOUT):
        self.node_id = node_id
        self.timeout = timeout
        self.stats = {
            "queries_sent": 0,
            "responses_received": 0,
            "timeouts": 0,
        }

    async def _query_peer(self, ip: str, port: int, question: str) -> Optional[PeerAnswer]:
        """Send query to a single peer and get response."""
        start = time.time()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=3.0  # Connection timeout
            )

            msg = _encode_message("QUERY", {
                "question": question,
                "sender": self.node_id,
            })
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

    async def query_mesh(
        self,
        question: str,
        local_answer: str,
        local_source: str = "local",
    ) -> MeshAnswer:
        """
        Fan out question to all known peers and merge with local answer.

        Args:
            question: The question to ask
            local_answer: Our local LLM's answer
            local_source: Source of local answer (groq, gemini, local, etc.)

        Returns:
            MeshAnswer with local + peer answers merged
        """
        start_time = time.time()
        self.stats["queries_sent"] += 1

        # Get known peers from persistence
        peers = []
        try:
            from .persistence import get_persistence_manager
            pm = get_persistence_manager()
            peer_records = pm.get_known_peers(limit=5, max_age_hours=1.0)
            peers = [(p.ip, p.port) for p in peer_records if p.node_id != self.node_id]
        except Exception:
            pass

        if not peers:
            return MeshAnswer(
                local_answer=local_answer,
                merged_answer=local_answer,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        # Fan out to all peers in parallel
        tasks = [self._query_peer(ip, port, question) for ip, port in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        peer_answers = [r for r in results if isinstance(r, PeerAnswer)]
        total_time_ms = (time.time() - start_time) * 1000

        if not peer_answers:
            return MeshAnswer(
                local_answer=local_answer,
                merged_answer=local_answer,
                total_time_ms=total_time_ms,
            )

        # Calculate coherence between local and peer answers
        coherence = self._calculate_coherence(local_answer, peer_answers)

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
            sources = [local_source] + [p.source for p in peers]
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
        for i, peer in enumerate(sorted(peers, key=lambda p: p.weighted_score, reverse=True)[:3]):
            lines.append(f"\nNode {peer.node_id[:8]}... ({peer.source}, {peer.latency_ms:.0f}ms):")
            lines.append(peer.answer)

        lines.append(f"\n[Mesh query: {len(peers)+1} nodes, coherence: {coherence:.2f}]")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        return self.stats.copy()
