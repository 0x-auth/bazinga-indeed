#!/usr/bin/env python3
"""
BAZINGA Gradient Sharing via DHT - Federated Learning P2P
==========================================================

Share LoRA gradients through the DHT network for federated learning.

Flow:
1. Local training generates gradients
2. Find gradient validators via DHT (topic: "gradient_validation")
3. Submit gradients to φ-trusted validators
4. Validators check triadic consensus
5. Receive aggregated gradient update

"The network learns collectively. No single point of control."

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import asyncio
import hashlib
import json
import time
import struct
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Constants
PHI = 1.618033988749895
GRADIENT_TOPIC = "gradient_validation"
MIN_VALIDATORS = 3   # Triadic consensus
MAX_VALIDATORS = 7   # Don't overload
GRADIENT_TTL = 300   # 5 min TTL for gradient entries
AGGREGATION_TIMEOUT = 30.0


@dataclass
class GradientUpdate:
    """A gradient update for federated learning."""
    layer_name: str
    gradient_data: bytes  # Serialized numpy/torch tensor
    learning_rate: float
    batch_size: int
    epoch: int
    node_id: str
    timestamp: float = field(default_factory=time.time)
    trust_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "gradient_data": base64.b64encode(self.gradient_data).decode(),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "trust_score": self.trust_score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GradientUpdate":
        return cls(
            layer_name=data["layer_name"],
            gradient_data=base64.b64decode(data["gradient_data"]),
            learning_rate=data["learning_rate"],
            batch_size=data["batch_size"],
            epoch=data["epoch"],
            node_id=data["node_id"],
            timestamp=data.get("timestamp", time.time()),
            trust_score=data.get("trust_score", 0.5),
        )

    @property
    def gradient_hash(self) -> str:
        """Hash of gradient for verification."""
        return hashlib.sha256(self.gradient_data).hexdigest()[:16]


@dataclass
class AggregatedGradient:
    """Result of gradient aggregation."""
    layer_name: str
    aggregated_data: bytes
    contributor_count: int
    total_weight: float
    consensus_achieved: bool
    validation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "aggregated_data": base64.b64encode(self.aggregated_data).decode(),
            "contributor_count": self.contributor_count,
            "total_weight": self.total_weight,
            "consensus": self.consensus_achieved,
            "time_ms": self.validation_time_ms,
        }


class GradientSharing:
    """
    Share and aggregate gradients via DHT for federated learning.

    Usage:
        sharing = GradientSharing(bridge)

        # Submit local gradient
        result = await sharing.submit_gradient(gradient_update)

        # Get aggregated gradients
        aggregated = await sharing.get_aggregated_gradients("layer1")
    """

    def __init__(self, dht_bridge):
        """
        Initialize gradient sharing.

        Args:
            dht_bridge: DHTBridge instance for P2P communication
        """
        self.bridge = dht_bridge

        # Local gradient buffer (for aggregation)
        self.local_gradients: Dict[str, List[GradientUpdate]] = {}

        # Pending aggregations
        self.pending_aggregations: Dict[str, asyncio.Future] = {}

        # Stats
        self.stats = {
            "gradients_submitted": 0,
            "gradients_received": 0,
            "aggregations_performed": 0,
            "consensus_achieved": 0,
        }

    async def register_as_validator(self):
        """Register this node as a gradient validator."""
        await self.bridge.announce_knowledge(GRADIENT_TOPIC, confidence=1.0)
        print(f"    Registered as gradient validator")

    async def find_validators(self, count: int = MAX_VALIDATORS) -> List[Any]:
        """Find gradient validator nodes."""
        validators = await self.bridge.find_experts(GRADIENT_TOPIC, count=count)

        # Filter to only φ-trust nodes (1.618x or higher)
        trusted = [v for v in validators if v.trust_score >= PHI]

        # If not enough trusted, include lower trust
        if len(trusted) < MIN_VALIDATORS:
            trusted = validators[:count]

        return trusted

    async def submit_gradient(
        self,
        gradient: GradientUpdate,
        timeout: float = AGGREGATION_TIMEOUT,
    ) -> Optional[AggregatedGradient]:
        """
        Submit a gradient update to the network.

        Args:
            gradient: The gradient update to submit
            timeout: Timeout for aggregation

        Returns:
            Aggregated gradient if consensus achieved, None otherwise
        """
        start_time = time.time()
        self.stats["gradients_submitted"] += 1

        # Find validators
        validators = await self.find_validators()

        if len(validators) < MIN_VALIDATORS:
            print(f"    Not enough validators ({len(validators)}/{MIN_VALIDATORS})")
            return None

        # Submit to validators
        submit_tasks = []
        for validator in validators:
            submit_tasks.append(
                self._submit_to_validator(validator, gradient, timeout)
            )

        results = await asyncio.gather(*submit_tasks)
        successful = [r for r in results if r is not None]

        # Check for triadic consensus
        if len(successful) >= MIN_VALIDATORS:
            # Request aggregation from primary validator
            aggregated = await self._request_aggregation(
                validators[0],
                gradient.layer_name,
                timeout,
            )

            if aggregated:
                aggregated.validation_time_ms = (time.time() - start_time) * 1000
                self.stats["consensus_achieved"] += 1
                return aggregated

        return None

    async def _submit_to_validator(
        self,
        validator,
        gradient: GradientUpdate,
        timeout: float,
    ) -> bool:
        """Submit gradient to a single validator."""
        try:
            request = {
                "cmd": "GRADIENT_SUBMIT",
                "gradient": gradient.to_dict(),
                "sender": self.bridge.dht.get_info().to_dict(),
            }

            response = await self.bridge.dht._send_request(
                validator, request, timeout=timeout
            )

            if response and response.get("status") == "OK":
                return True

        except Exception:
            pass

        return False

    async def _request_aggregation(
        self,
        validator,
        layer_name: str,
        timeout: float,
    ) -> Optional[AggregatedGradient]:
        """Request aggregated gradient from validator."""
        try:
            request = {
                "cmd": "GRADIENT_AGGREGATE",
                "layer_name": layer_name,
                "sender": self.bridge.dht.get_info().to_dict(),
            }

            response = await self.bridge.dht._send_request(
                validator, request, timeout=timeout
            )

            if response and response.get("status") == "OK":
                return AggregatedGradient(
                    layer_name=layer_name,
                    aggregated_data=base64.b64decode(response["aggregated_data"]),
                    contributor_count=response["contributor_count"],
                    total_weight=response["total_weight"],
                    consensus_achieved=response.get("consensus", False),
                    validation_time_ms=0,
                )

        except Exception:
            pass

        return None

    def receive_gradient(self, gradient: GradientUpdate):
        """
        Receive a gradient from another node (as validator).

        Called by the DHT request handler.
        """
        layer = gradient.layer_name
        if layer not in self.local_gradients:
            self.local_gradients[layer] = []

        # Check for duplicates
        existing_ids = [g.node_id for g in self.local_gradients[layer]]
        if gradient.node_id not in existing_ids:
            self.local_gradients[layer].append(gradient)
            self.stats["gradients_received"] += 1

            # Clean old gradients
            now = time.time()
            self.local_gradients[layer] = [
                g for g in self.local_gradients[layer]
                if now - g.timestamp < GRADIENT_TTL
            ]

    def aggregate_gradients(self, layer_name: str) -> Optional[AggregatedGradient]:
        """
        Aggregate received gradients using φ-weighted averaging.

        Returns aggregated gradient if triadic consensus achieved.
        """
        if layer_name not in self.local_gradients:
            return None

        gradients = self.local_gradients[layer_name]

        if len(gradients) < MIN_VALIDATORS:
            return None

        self.stats["aggregations_performed"] += 1

        # φ-weighted aggregation
        total_weight = 0.0
        weighted_sum = None

        for grad in gradients:
            # Weight by trust score (φ-bonus nodes contribute more)
            weight = grad.trust_score

            # Deserialize gradient (assume numpy format)
            try:
                import numpy as np
                arr = np.frombuffer(grad.gradient_data, dtype=np.float32)

                if weighted_sum is None:
                    weighted_sum = arr * weight
                else:
                    if len(arr) == len(weighted_sum):
                        weighted_sum += arr * weight

                total_weight += weight

            except Exception:
                # Skip malformed gradients
                continue

        if weighted_sum is None or total_weight == 0:
            return None

        # Normalize by total weight
        aggregated = weighted_sum / total_weight

        return AggregatedGradient(
            layer_name=layer_name,
            aggregated_data=aggregated.tobytes(),
            contributor_count=len(gradients),
            total_weight=total_weight,
            consensus_achieved=len(gradients) >= MIN_VALIDATORS,
            validation_time_ms=0,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get gradient sharing statistics."""
        return {
            **self.stats,
            "buffered_layers": len(self.local_gradients),
            "total_buffered": sum(len(g) for g in self.local_gradients.values()),
        }


# =============================================================================
# GRADIENT REQUEST HANDLERS (Add to KademliaNode)
# =============================================================================

async def handle_gradient_submit(node, request: Dict, sharing: GradientSharing) -> Dict:
    """Handle GRADIENT_SUBMIT request."""
    try:
        gradient_data = request.get("gradient", {})
        gradient = GradientUpdate.from_dict(gradient_data)

        # Verify sender trust
        sender = request.get("sender", {})
        gradient.trust_score = sender.get("trust_score", 0.5)

        sharing.receive_gradient(gradient)

        return {
            "status": "OK",
            "gradient_hash": gradient.gradient_hash,
            "sender": node.get_info().to_dict(),
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e),
            "sender": node.get_info().to_dict(),
        }


async def handle_gradient_aggregate(node, request: Dict, sharing: GradientSharing) -> Dict:
    """Handle GRADIENT_AGGREGATE request."""
    try:
        layer_name = request.get("layer_name", "")
        aggregated = sharing.aggregate_gradients(layer_name)

        if aggregated:
            return {
                "status": "OK",
                "aggregated_data": base64.b64encode(aggregated.aggregated_data).decode(),
                "contributor_count": aggregated.contributor_count,
                "total_weight": aggregated.total_weight,
                "consensus": aggregated.consensus_achieved,
                "sender": node.get_info().to_dict(),
            }

        return {
            "status": "PENDING",
            "message": "Not enough gradients for aggregation",
            "sender": node.get_info().to_dict(),
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e),
            "sender": node.get_info().to_dict(),
        }
