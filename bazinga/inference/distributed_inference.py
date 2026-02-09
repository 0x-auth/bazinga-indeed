#!/usr/bin/env python3
"""
BAZINGA Distributed Inference - Petals-Style Layer Splitting

Enables collaborative inference across P2P network:
- Split model layers across multiple nodes
- Pipeline parallelism for larger models
- Load balancing based on tau scores
- Fault tolerance with automatic failover

Architecture:
    Query → Router → [Node A: Layers 1-10] → [Node B: Layers 11-20] → Response

"One model, many minds, zero central control."
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import defaultdict

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


class NodeCapability(Enum):
    """Types of inference capabilities a node can offer."""
    FULL_MODEL = "full_model"       # Can run entire model
    LAYER_RANGE = "layer_range"     # Can run specific layers
    EMBEDDING = "embedding"         # Embedding generation only
    ATTENTION = "attention"         # Attention layers only
    MLP = "mlp"                     # MLP/FFN layers only


class NodeStatus(Enum):
    """Status of an inference node."""
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass
class InferenceCapability:
    """Capability advertisement for a node."""
    node_id: str
    model_id: str
    capability: NodeCapability = NodeCapability.FULL_MODEL
    layer_start: int = 0
    layer_end: int = -1  # -1 means all layers
    max_context: int = 2048
    max_batch_size: int = 4
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 4
    tau_score: float = 0.5
    latency_ms: float = 100.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['capability'] = self.capability.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'InferenceCapability':
        data['capability'] = NodeCapability(data['capability'])
        return cls(**data)


@dataclass
class InferenceRequest:
    """Request for distributed inference."""
    request_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    requester_id: str = ""
    priority: int = 0
    created_at: float = field(default_factory=time.time)

    # Routing info
    target_nodes: List[str] = field(default_factory=list)
    current_layer: int = 0
    intermediate_state: Optional[bytes] = None


@dataclass
class InferenceResult:
    """Result from distributed inference."""
    request_id: str
    response: str
    tokens_generated: int = 0
    total_latency_ms: float = 0.0
    nodes_used: List[str] = field(default_factory=list)
    phi_coherence: float = 0.0
    success: bool = True
    error: Optional[str] = None


class InferenceNode:
    """
    A node that participates in distributed inference.

    Can serve as:
    - Full inference node (runs entire model)
    - Layer server (runs subset of layers)
    - Router node (coordinates requests)

    Usage:
        node = InferenceNode(node_id, model)
        node.advertise_capability(capability)

        # Handle inference requests
        result = await node.process_request(request)
    """

    def __init__(
        self,
        node_id: str,
        local_model=None,
        network=None,
    ):
        """
        Initialize inference node.

        Args:
            node_id: Unique node identifier
            local_model: LocalModel instance if available
            network: P2P network for communication
        """
        self.node_id = node_id
        self.local_model = local_model
        self.network = network

        # Capabilities
        self.capability: Optional[InferenceCapability] = None
        self.status = NodeStatus.ONLINE

        # Known peers
        self.peer_capabilities: Dict[str, InferenceCapability] = {}

        # Request handling
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()

        # Stats
        self.requests_processed = 0
        self.total_tokens = 0
        self.total_latency = 0.0

        # Callbacks
        self.on_request_received: Optional[Callable] = None
        self.on_result_ready: Optional[Callable] = None

    def advertise_capability(self, capability: InferenceCapability):
        """
        Advertise this node's inference capability.

        Args:
            capability: What this node can do
        """
        self.capability = capability
        print(f"Node {self.node_id[:8]} advertising: {capability.capability.value}")

        # Broadcast to network
        if self.network:
            # In real implementation: self.network.announce_capability(capability)
            pass

    def register_peer(self, capability: InferenceCapability):
        """Register a peer's capability."""
        self.peer_capabilities[capability.node_id] = capability

    def unregister_peer(self, node_id: str):
        """Remove a peer."""
        self.peer_capabilities.pop(node_id, None)

    async def process_request(self, request: InferenceRequest) -> InferenceResult:
        """
        Process an inference request.

        May run locally or route to peers depending on capability.
        """
        start = time.time()

        try:
            if self.capability and self.capability.capability == NodeCapability.FULL_MODEL:
                # Run locally
                result = await self._run_local(request)
            elif request.target_nodes and self.node_id in request.target_nodes:
                # We're part of the pipeline
                result = await self._run_pipeline_step(request)
            else:
                # Route to capable peers
                result = await self._route_to_peers(request)

            result.total_latency_ms = (time.time() - start) * 1000
            result.nodes_used.append(self.node_id)

            self.requests_processed += 1
            self.total_tokens += result.tokens_generated
            self.total_latency += result.total_latency_ms

            return result

        except Exception as e:
            return InferenceResult(
                request_id=request.request_id,
                response="",
                success=False,
                error=str(e),
            )

    async def _run_local(self, request: InferenceRequest) -> InferenceResult:
        """Run inference locally."""
        if not self.local_model:
            raise RuntimeError("No local model available")

        # Generate response
        response = self.local_model.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Compute coherence
        coherence = self.local_model.compute_phi_coherence(response)

        return InferenceResult(
            request_id=request.request_id,
            response=response,
            tokens_generated=len(response.split()),
            phi_coherence=coherence,
        )

    async def _run_pipeline_step(self, request: InferenceRequest) -> InferenceResult:
        """
        Run our portion of the pipeline.

        In a real implementation, this would:
        1. Decode intermediate state from previous node
        2. Run our layers
        3. Encode and send to next node
        """
        # For now, simulate with local model
        if self.local_model:
            return await self._run_local(request)

        # Mock pipeline step
        return InferenceResult(
            request_id=request.request_id,
            response=f"[Pipeline step from {self.node_id[:8]}]",
            tokens_generated=5,
        )

    async def _route_to_peers(self, request: InferenceRequest) -> InferenceResult:
        """Route request to capable peers."""
        # Find capable peers
        capable_peers = self._find_capable_peers(request)

        if not capable_peers:
            return InferenceResult(
                request_id=request.request_id,
                response="",
                success=False,
                error="No capable peers found",
            )

        # Select best peer based on tau score and latency
        best_peer = self._select_best_peer(capable_peers)

        # Route (in real implementation, would send over network)
        # For now, simulate
        return InferenceResult(
            request_id=request.request_id,
            response=f"[Routed to {best_peer[:8]}]",
            nodes_used=[best_peer],
        )

    def _find_capable_peers(
        self,
        request: InferenceRequest,
    ) -> List[InferenceCapability]:
        """Find peers capable of handling request."""
        capable = []

        for node_id, cap in self.peer_capabilities.items():
            if cap.max_context >= len(request.prompt.split()):
                capable.append(cap)

        return capable

    def _select_best_peer(self, peers: List[InferenceCapability]) -> str:
        """Select best peer using tau-weighted selection."""
        if not peers:
            return ""

        # Score = tau / latency (higher is better)
        scored = [
            (p.node_id, p.tau_score / max(1, p.latency_ms))
            for p in peers
        ]

        # Select highest score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        avg_latency = self.total_latency / max(1, self.requests_processed)

        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'capability': self.capability.to_dict() if self.capability else None,
            'known_peers': len(self.peer_capabilities),
            'requests_processed': self.requests_processed,
            'total_tokens': self.total_tokens,
            'avg_latency_ms': avg_latency,
        }


class DistributedInference:
    """
    Coordinator for distributed inference across the BAZINGA network.

    Manages:
    - Node discovery and capability tracking
    - Request routing and load balancing
    - Pipeline assembly for large models
    - Fault tolerance and retries

    Usage:
        dist = DistributedInference(network)
        dist.register_node(local_node)

        # Query with automatic routing
        response = await dist.query("What is consciousness?")
    """

    def __init__(
        self,
        network=None,
        min_tau_threshold: float = 0.3,
    ):
        """
        Initialize distributed inference.

        Args:
            network: P2P network for node communication
            min_tau_threshold: Minimum tau score to consider a node
        """
        self.network = network
        self.min_tau_threshold = min_tau_threshold

        # Registered nodes
        self.nodes: Dict[str, InferenceNode] = {}
        self.capabilities: Dict[str, InferenceCapability] = {}

        # Pipeline cache
        self.pipeline_cache: Dict[str, List[str]] = {}

        # Stats
        self.total_queries = 0
        self.successful_queries = 0
        self.avg_latency = 0.0

        # Load balancer state
        self.node_load: Dict[str, int] = defaultdict(int)

    def register_node(self, node: InferenceNode):
        """Register an inference node."""
        self.nodes[node.node_id] = node
        if node.capability:
            self.capabilities[node.node_id] = node.capability
        print(f"Registered node {node.node_id[:8]}")

    def unregister_node(self, node_id: str):
        """Unregister a node."""
        self.nodes.pop(node_id, None)
        self.capabilities.pop(node_id, None)

    async def query(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model_id: Optional[str] = None,
    ) -> InferenceResult:
        """
        Query the distributed network.

        Args:
            prompt: Query prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            model_id: Preferred model (optional)

        Returns:
            InferenceResult
        """
        self.total_queries += 1

        # Create request
        request = InferenceRequest(
            request_id=hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:16],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Find capable nodes
        capable = self._find_capable_nodes(model_id)

        if not capable:
            return InferenceResult(
                request_id=request.request_id,
                response="",
                success=False,
                error="No capable nodes available",
            )

        # Try nodes in order of preference
        for node_id in capable:
            node = self.nodes.get(node_id)
            if not node:
                continue

            try:
                self.node_load[node_id] += 1
                result = await node.process_request(request)
                self.node_load[node_id] -= 1

                if result.success:
                    self.successful_queries += 1
                    self._update_avg_latency(result.total_latency_ms)
                    return result

            except Exception as e:
                self.node_load[node_id] -= 1
                continue

        return InferenceResult(
            request_id=request.request_id,
            response="",
            success=False,
            error="All nodes failed",
        )

    def _find_capable_nodes(
        self,
        model_id: Optional[str] = None,
    ) -> List[str]:
        """Find and rank capable nodes."""
        capable = []

        for node_id, cap in self.capabilities.items():
            # Filter by tau
            if cap.tau_score < self.min_tau_threshold:
                continue

            # Filter by model if specified
            if model_id and cap.model_id != model_id:
                continue

            capable.append((node_id, cap))

        # Rank by: (tau * PHI) / (latency * (1 + load))
        ranked = sorted(
            capable,
            key=lambda x: (x[1].tau_score * PHI) / (x[1].latency_ms * (1 + self.node_load.get(x[0], 0))),
            reverse=True,
        )

        return [node_id for node_id, _ in ranked]

    def _update_avg_latency(self, latency_ms: float):
        """Update running average latency."""
        alpha = 0.1  # Smoothing factor
        self.avg_latency = alpha * latency_ms + (1 - alpha) * self.avg_latency

    async def build_pipeline(
        self,
        model_id: str,
        num_layers: int,
    ) -> List[str]:
        """
        Build an inference pipeline across multiple nodes.

        Args:
            model_id: Model to serve
            num_layers: Total layers in model

        Returns:
            Ordered list of node IDs for pipeline
        """
        cache_key = f"{model_id}:{num_layers}"
        if cache_key in self.pipeline_cache:
            return self.pipeline_cache[cache_key]

        # Find nodes that can serve layers
        layer_servers: Dict[int, List[Tuple[str, InferenceCapability]]] = defaultdict(list)

        for node_id, cap in self.capabilities.items():
            if cap.model_id != model_id:
                continue
            if cap.capability != NodeCapability.LAYER_RANGE:
                continue

            for layer in range(cap.layer_start, cap.layer_end + 1):
                if layer < num_layers:
                    layer_servers[layer].append((node_id, cap))

        # Greedy assignment: for each layer, pick highest-tau available node
        pipeline = []
        assigned_layers: Dict[str, Tuple[int, int]] = {}  # node -> (start, end)

        current_node = None
        current_start = 0

        for layer in range(num_layers):
            if layer not in layer_servers:
                # Gap in coverage
                break

            # Find best node for this layer
            candidates = layer_servers[layer]
            candidates.sort(key=lambda x: x[1].tau_score, reverse=True)

            best_node = candidates[0][0]

            if best_node != current_node:
                if current_node:
                    assigned_layers[current_node] = (current_start, layer - 1)
                    pipeline.append(current_node)

                current_node = best_node
                current_start = layer

        # Finalize last segment
        if current_node:
            assigned_layers[current_node] = (current_start, num_layers - 1)
            pipeline.append(current_node)

        self.pipeline_cache[cache_key] = pipeline
        return pipeline

    def get_network_stats(self) -> Dict[str, Any]:
        """Get distributed inference statistics."""
        success_rate = self.successful_queries / max(1, self.total_queries)

        return {
            'total_nodes': len(self.nodes),
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'success_rate': success_rate,
            'avg_latency_ms': self.avg_latency,
            'node_load': dict(self.node_load),
            'pipeline_cache_size': len(self.pipeline_cache),
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Distributed Inference Test")
    print("=" * 60)

    import asyncio

    async def test():
        # Create distributed coordinator
        dist = DistributedInference()

        # Create mock nodes
        for i in range(3):
            node_id = f"node_{i:03d}"
            node = InferenceNode(node_id)
            node.advertise_capability(InferenceCapability(
                node_id=node_id,
                model_id="phi-2",
                capability=NodeCapability.FULL_MODEL,
                tau_score=0.5 + i * 0.15,
                latency_ms=100 - i * 20,
            ))
            dist.register_node(node)

        print(f"\nRegistered {len(dist.nodes)} nodes")

        # Query
        print("\nQuerying network...")
        result = await dist.query("What is the meaning of phi?")
        print(f"Result: {result}")

        # Stats
        print(f"\nNetwork stats: {dist.get_network_stats()}")

        # Build pipeline
        print("\nBuilding pipeline for 20-layer model...")

        # Add layer-serving nodes
        for i in range(4):
            node_id = f"layer_node_{i:03d}"
            node = InferenceNode(node_id)
            node.advertise_capability(InferenceCapability(
                node_id=node_id,
                model_id="mistral-7b",
                capability=NodeCapability.LAYER_RANGE,
                layer_start=i * 5,
                layer_end=(i + 1) * 5 - 1,
                tau_score=0.6 + i * 0.1,
            ))
            dist.register_node(node)

        pipeline = await dist.build_pipeline("mistral-7b", 20)
        print(f"Pipeline: {pipeline}")

    asyncio.run(test())

    print("\n" + "=" * 60)
    print("Distributed Inference module ready!")
    print("=" * 60)
