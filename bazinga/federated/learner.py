#!/usr/bin/env python3
"""
BAZINGA Collective Learner - Main Federated Learning Interface
===============================================================

The CollectiveLearner is the main interface for federated learning.

What it does:
1. Manages local LoRA adapter
2. Trains on local interactions
3. Shares gradients with network
4. Merges incoming gradients
5. Makes the network smarter over time

Integration with P2P:
- Connects to BazingaProtocol for network communication
- Uses α-SEED for knowledge anchoring
- Applies φ-weighted aggregation

"The network becomes smarter than any single node."
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .lora import LoRAAdapter, LoRAConfig, create_lora_adapter
from .gradients import GradientPackage, GradientSharer
from .aggregator import FederatedAggregator, TrustTracker, AggregationResult

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Check for numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class LearningStats:
    """Statistics for collective learning."""
    local_examples: int = 0
    gradients_shared: int = 0
    gradients_received: int = 0
    aggregations: int = 0
    model_updates: int = 0
    average_loss: float = 0.0
    network_contribution: float = 0.0  # How much we've helped the network


class CollectiveLearner:
    """
    Main interface for BAZINGA federated learning.

    Responsibilities:
    1. Train locally on user interactions
    2. Share learning with the network
    3. Learn from other nodes
    4. Track collective improvement

    Usage:
        learner = CollectiveLearner(node_id="my_node")
        await learner.start()

        # Train on local data
        learner.learn(question="What is AI?", answer="AI is...")

        # Share with network (periodic or manual)
        await learner.share_learning()

        # Receive from network (automatic via callbacks)
        await learner.receive_learning(package)
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        model_dims: int = 768,  # Embedding dimension
        share_interval: float = 300,  # Share every 5 minutes
        min_examples_to_share: int = 10,  # Minimum examples before sharing
    ):
        self.node_id = node_id or f"learner_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        self.model_dims = model_dims
        self.share_interval = share_interval
        self.min_examples_to_share = min_examples_to_share

        # Components
        self.adapter = create_lora_adapter(node_id=self.node_id)
        self.sharer = GradientSharer(node_id=self.node_id)
        self.aggregator = FederatedAggregator(min_nodes=1)
        self.trust_tracker = TrustTracker()

        # Initialize adapter weights
        self._initialize_adapter()

        # Learning queue
        self.pending_examples: List[Dict] = []

        # Stats
        self.stats = LearningStats()
        self.last_share_time = time.time()

        # Callbacks
        self.on_learning_shared: Optional[Callable] = None
        self.on_model_updated: Optional[Callable] = None

        # Background tasks
        self.running = False
        self.share_task: Optional[asyncio.Task] = None

    def _initialize_adapter(self):
        """Initialize LoRA adapter with default modules."""
        # Common modules for transformer-based models
        modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        for module in modules:
            self.adapter.initialize_weights(
                input_dim=self.model_dims,
                output_dim=self.model_dims,
                module_name=module,
            )

    async def start(self):
        """Start the collective learner."""
        if self.running:
            return

        self.running = True

        # Start periodic sharing
        self.share_task = asyncio.create_task(self._share_loop())

        print(f"  🧠 Collective Learner started: {self.node_id}")

    async def stop(self):
        """Stop the collective learner."""
        self.running = False

        if self.share_task:
            self.share_task.cancel()

        # Save adapter
        self.adapter.save()

        print(f"  🧠 Collective Learner stopped")

    def learn(
        self,
        question: str,
        answer: str,
        feedback_score: float = 0.5,
        metadata: Optional[Dict] = None,
    ):
        """
        Learn from a Q&A interaction.

        This is called whenever the user interacts with BAZINGA.
        Good answers (high feedback) train the model positively.

        Args:
            question: User's question
            answer: BAZINGA's answer
            feedback_score: 0-1 score (1 = great answer)
            metadata: Additional context
        """
        example = {
            'question': question,
            'answer': answer,
            'feedback': feedback_score,
            'metadata': metadata or {},
            'timestamp': time.time(),
        }

        self.pending_examples.append(example)
        self.stats.local_examples += 1

        # If we have enough examples, do a training step
        if len(self.pending_examples) >= 5:
            self._train_batch()

    def _train_batch(self):
        """Train on pending examples."""
        if not self.pending_examples or not NUMPY_AVAILABLE:
            return

        # Simple training: create pseudo-embeddings and train
        for example in self.pending_examples:
            # Create simple embeddings from text hash
            q_hash = hashlib.sha256(example['question'].encode()).digest()
            a_hash = hashlib.sha256(example['answer'].encode()).digest()

            # Convert to pseudo-embedding
            q_emb = np.frombuffer(q_hash, dtype=np.uint8).astype(np.float32)
            a_emb = np.frombuffer(a_hash, dtype=np.uint8).astype(np.float32)

            # Pad/truncate to model dims
            if len(q_emb) < self.model_dims:
                q_emb = np.pad(q_emb, (0, self.model_dims - len(q_emb)))
            else:
                q_emb = q_emb[:self.model_dims]

            if len(a_emb) < self.model_dims:
                a_emb = np.pad(a_emb, (0, self.model_dims - len(a_emb)))
            else:
                a_emb = a_emb[:self.model_dims]

            # Normalize
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-8)

            # Weight by feedback
            target = a_emb * example['feedback']

            # Train each module
            for module in self.adapter.weights.keys():
                input_act = q_emb.reshape(1, -1)
                target_out = target.reshape(1, -1)

                loss = self.adapter.train_step(
                    module,
                    input_act,
                    target_out,
                )

                # Add to gradient accumulator
                if module in self.adapter.weights:
                    w = self.adapter.weights[module]
                    if NUMPY_AVAILABLE and isinstance(w.A, np.ndarray):
                        self.sharer.add_gradients(module, w.A)

            self.stats.average_loss = (
                self.stats.average_loss * 0.9 +
                loss * 0.1
            )

        # Clear pending
        self.pending_examples = []

    async def share_learning(self) -> Optional[GradientPackage]:
        """
        Share accumulated learning with the network.

        Returns the gradient package that was shared.
        """
        if self.stats.local_examples < self.min_examples_to_share:
            return None

        # Train any remaining examples
        self._train_batch()

        # Create package
        package = self.sharer.create_package()
        self.stats.gradients_shared += 1
        self.last_share_time = time.time()

        # Callback
        if self.on_learning_shared:
            await self.on_learning_shared(package)

        print(f"  📤 Shared learning (samples={package.training_samples})")

        return package

    async def receive_learning(
        self,
        package: GradientPackage,
        sender_trust: Optional[float] = None,
    ):
        """
        Receive learning from another node.

        Args:
            package: Gradient package from other node
            sender_trust: Trust score of sender (uses tracker if not provided)
        """
        # Get trust score
        if sender_trust is None:
            sender_trust = self.trust_tracker.get_trust(package.node_id)

        # Add to aggregator
        self.aggregator.add_package(package, sender_trust)
        self.stats.gradients_received += 1

        # Try to aggregate
        if self.aggregator.get_pending_count() >= self.aggregator.min_nodes:
            result = self.aggregator.aggregate()
            if result:
                await self._apply_aggregation(result)

    async def _apply_aggregation(self, result: AggregationResult):
        """Apply aggregated gradients to local adapter."""
        for module_name, merged_grad in result.merged_gradients.items():
            if module_name in self.adapter.weights:
                w = self.adapter.weights[module_name]

                # Merge with current weights
                if NUMPY_AVAILABLE and isinstance(w.A, np.ndarray):
                    # φ-weighted blend: keep most of ours, add some of theirs
                    blend_factor = 1 / PHI  # ~0.618
                    w.A = (1 - blend_factor) * w.A + blend_factor * merged_grad

        self.stats.aggregations += 1
        self.stats.model_updates += 1

        # Record contributions
        for node_id in result.contributing_nodes:
            self.trust_tracker.record_contribution(node_id, was_useful=True)

        # Save updated adapter
        self.adapter.save()

        # Callback
        if self.on_model_updated:
            await self.on_model_updated(result)

        print(f"  📥 Applied learning from {len(result.contributing_nodes)} nodes")

    async def _share_loop(self):
        """Background loop for periodic sharing."""
        while self.running:
            try:
                await asyncio.sleep(self.share_interval)

                # Share if we have enough examples
                if self.stats.local_examples >= self.min_examples_to_share:
                    await self.share_learning()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  Share loop error: {e}")

    def get_adapter_gradients(self) -> Dict[str, Any]:
        """Get current adapter gradients for manual sharing."""
        return self.adapter.get_gradients()

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        adapter_stats = self.adapter.get_stats()
        trust_stats = self.trust_tracker.get_stats()

        return {
            'node_id': self.node_id,
            'running': self.running,
            'local_examples': self.stats.local_examples,
            'gradients_shared': self.stats.gradients_shared,
            'gradients_received': self.stats.gradients_received,
            'aggregations': self.stats.aggregations,
            'model_updates': self.stats.model_updates,
            'average_loss': self.stats.average_loss,
            'adapter': adapter_stats,
            'trust': trust_stats,
            'pending_examples': len(self.pending_examples),
        }

    def print_status(self):
        """Print learner status."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("  BAZINGA Collective Learner Status")
        print("=" * 60)
        print(f"  Node ID: {stats['node_id']}")
        print(f"  Running: {'✓' if stats['running'] else '✗'}")
        print()
        print(f"  Local Examples: {stats['local_examples']}")
        print(f"  Gradients Shared: {stats['gradients_shared']}")
        print(f"  Gradients Received: {stats['gradients_received']}")
        print(f"  Model Updates: {stats['model_updates']}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print()
        print(f"  Adapter Params: {stats['adapter']['total_params']}")
        print(f"  Adapter Version: {stats['adapter']['version']}")
        print(f"  Known Nodes: {stats['trust']['nodes']}")
        print("=" * 60)


# Convenience function
def create_learner(
    node_id: Optional[str] = None,
    **kwargs,
) -> CollectiveLearner:
    """Create a new collective learner."""
    return CollectiveLearner(node_id=node_id, **kwargs)


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA Collective Learner Test")
        print("=" * 60)

        print(f"\n  Numpy available: {NUMPY_AVAILABLE}")

        # Create learner
        learner = create_learner(node_id="test_learner")
        await learner.start()

        # Simulate some learning
        examples = [
            ("What is consciousness?", "Consciousness is awareness of self and environment.", 0.8),
            ("How does φ work?", "φ (phi) is the golden ratio, approximately 1.618.", 0.9),
            ("What is BAZINGA?", "BAZINGA is a distributed AI with φ-coherence.", 0.95),
            ("Tell me about P2P", "P2P means peer-to-peer, decentralized networking.", 0.7),
            ("What is the boundary?", "The boundary is where subject meets object, P/G ≈ φ⁴.", 0.85),
        ]

        print("\n  Training on examples...")
        for q, a, score in examples:
            learner.learn(q, a, score)
            print(f"    ✓ Learned: {q[:30]}...")

        # Create a package
        package = await learner.share_learning()

        if package:
            print(f"\n  Created package:")
            print(f"    Training samples: {package.training_samples}")
            print(f"    Modules: {list(package.gradients.keys())}")

        # Simulate receiving from another node
        if package:
            # Pretend it's from another node
            package.node_id = "other_node"
            await learner.receive_learning(package, sender_trust=0.7)

        # Show status
        learner.print_status()

        # Stop
        await learner.stop()

        print("\n  ✓ Collective Learner test complete!")

    asyncio.run(test())
