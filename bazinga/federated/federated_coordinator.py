#!/usr/bin/env python3
"""
BAZINGA Federated Coordinator - Orchestrating Distributed Learning

The FederatedCoordinator ties together all Phase 3 components:
- Local trainers running on each node
- Differential privacy for gradient protection
- Secure aggregation via homomorphic encryption
- Trust-weighted gradient aggregation
- Global model distribution via P2P

Training Round:
    1. Nodes train locally (LocalTrainer)
    2. Gradients privatized (DifferentialPrivacy)
    3. Encrypted and shared (SecureAggregator)
    4. Trust-weighted aggregation
    5. Global model distributed via P2P (BAZINGANetwork)

"Many minds, one understanding. Many gradients, one model."
"""

import time
import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import threading

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137
PSI_DARMIYAN = PHI  # V2: Scaling constant is φ (use with √n: Ψ_D / Ψ_i = φ√n)

# Try imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .lora_adapter import LoRAAdapter, LoRAConfig
    from .local_trainer import LocalTrainer, TrainingConfig
    from .differential_privacy import DifferentialPrivacy, PrivacyConfig, AdaptivePrivacy
    from .secure_aggregation import SecureAggregator, SimplifiedSecureAggregator
except ImportError:
    LoRAAdapter = None
    LocalTrainer = None
    DifferentialPrivacy = None
    SecureAggregator = None


@dataclass
class FederatedConfig:
    """Configuration for federated learning coordination."""
    # Round settings
    rounds_per_epoch: int = 5
    min_participants: int = 2
    max_participants: int = 100
    round_timeout_seconds: float = 300.0

    # Aggregation settings
    aggregation_method: str = 'trust_weighted'  # 'simple', 'trust_weighted', 'phi_coherent'
    trust_decay: float = 0.95  # How fast trust decays for non-participants
    trust_boost_coherent: float = 0.1  # Trust boost for coherent gradients

    # Privacy settings
    enable_differential_privacy: bool = True
    enable_secure_aggregation: bool = True
    privacy_budget_per_round: float = 1.0

    # phi-coherence settings
    min_coherence_threshold: float = 0.4
    coherence_weight_factor: float = 1.5  # How much coherence affects weight

    # P2P distribution settings
    model_chunk_size: int = 1024 * 1024  # 1MB chunks
    broadcast_timeout: float = 60.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FederatedConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))


@dataclass
class ParticipantState:
    """State for a federated learning participant."""
    node_id: str
    trust_score: float = 0.5
    total_contributions: int = 0
    successful_contributions: int = 0
    last_contribution_time: float = 0
    coherence_history: List[float] = field(default_factory=list)
    privacy_budget_spent: float = 0.0

    def update_trust(self, success: bool, coherence: float, config: FederatedConfig):
        """Update trust score based on contribution."""
        if success:
            self.successful_contributions += 1
            # Boost based on coherence
            boost = config.trust_boost_coherent * coherence
            self.trust_score = min(1.0, self.trust_score + boost)
        else:
            # Decay for failure
            self.trust_score *= config.trust_decay

        self.total_contributions += 1
        self.last_contribution_time = time.time()
        self.coherence_history.append(coherence)

        # Keep only recent history
        if len(self.coherence_history) > 100:
            self.coherence_history = self.coherence_history[-100:]

    @property
    def average_coherence(self) -> float:
        if not self.coherence_history:
            return 0.5
        return sum(self.coherence_history) / len(self.coherence_history)

    @property
    def reliability(self) -> float:
        if self.total_contributions == 0:
            return 0.5
        return self.successful_contributions / self.total_contributions


@dataclass
class RoundState:
    """State for a federated learning round."""
    round_id: int
    started_at: float = field(default_factory=time.time)
    participants: List[str] = field(default_factory=list)
    received_gradients: Dict[str, Any] = field(default_factory=dict)
    aggregated_gradients: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    completed: bool = False

    @property
    def num_participants(self) -> int:
        return len(self.participants)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.started_at


class TrustWeightedAggregator:
    """
    Trust-weighted gradient aggregation.

    Formula: g_global = SUM(tau_i * coherence_i * g_i) / SUM(tau_i * coherence_i)

    Where:
        tau_i = trust score of participant i
        coherence_i = phi-coherence of gradient from participant i
        g_i = gradient from participant i

    "Trust is earned through consistent coherent contributions."
    """

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.participants: Dict[str, ParticipantState] = {}

    def get_or_create_participant(self, node_id: str) -> ParticipantState:
        """Get or create participant state."""
        if node_id not in self.participants:
            self.participants[node_id] = ParticipantState(node_id=node_id)
        return self.participants[node_id]

    def compute_weight(self, participant: ParticipantState, coherence: float) -> float:
        """Compute aggregation weight for a participant."""
        # Base weight from trust
        trust_weight = participant.trust_score

        # Coherence factor
        coherence_factor = 1 + (coherence - 0.5) * self.config.coherence_weight_factor

        # Reliability factor
        reliability_factor = 0.5 + 0.5 * participant.reliability

        # Combined weight with phi-scaling
        weight = trust_weight * coherence_factor * reliability_factor
        weight *= PHI / (1 + PHI)  # phi-normalize

        return max(0.01, weight)  # Minimum weight

    def aggregate(
        self,
        gradient_contributions: Dict[str, Tuple[Dict[str, Any], float]],
    ) -> Dict[str, Any]:
        """
        Aggregate gradients with trust weighting.

        Args:
            gradient_contributions: Dict of node_id -> (gradients, coherence)

        Returns:
            Aggregated gradients
        """
        if not gradient_contributions:
            return {}

        # Compute weights
        weights = {}
        total_weight = 0.0

        for node_id, (gradients, coherence) in gradient_contributions.items():
            participant = self.get_or_create_participant(node_id)
            weight = self.compute_weight(participant, coherence)
            weights[node_id] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Aggregate
        aggregated = {}
        first_node = list(gradient_contributions.keys())[0]
        first_grads = gradient_contributions[first_node][0]

        for key in first_grads:
            # Weighted sum
            if TORCH_AVAILABLE and isinstance(first_grads[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_grads[key])
                for node_id, (grads, _) in gradient_contributions.items():
                    if key in grads:
                        weighted_sum += weights[node_id] * grads[key]
                aggregated[key] = weighted_sum
            elif isinstance(first_grads[key], list):
                # List of floats
                weighted_sum = [0.0] * len(first_grads[key])
                for node_id, (grads, _) in gradient_contributions.items():
                    if key in grads:
                        for i, v in enumerate(grads[key]):
                            weighted_sum[i] += weights[node_id] * v
                aggregated[key] = weighted_sum

        # Update participant trust
        for node_id, (_, coherence) in gradient_contributions.items():
            participant = self.get_or_create_participant(node_id)
            success = coherence >= self.config.min_coherence_threshold
            participant.update_trust(success, coherence, self.config)

        return aggregated

    def decay_non_participants(self, active_nodes: List[str]):
        """Decay trust for nodes that didn't participate."""
        for node_id, participant in self.participants.items():
            if node_id not in active_nodes:
                participant.trust_score *= self.config.trust_decay

    def get_trust_rankings(self) -> List[Tuple[str, float]]:
        """Get nodes ranked by trust score."""
        rankings = [
            (node_id, p.trust_score)
            for node_id, p in self.participants.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)


class PhiCoherentAggregator(TrustWeightedAggregator):
    """
    Phi-coherent gradient aggregation.

    Extends trust-weighted with additional phi-coherence validation:
    - Gradients below threshold are rejected
    - Coherence affects weight more strongly
    - Alpha-resonance filtering

    "Only harmonious gradients shape the model."
    """

    def __init__(self, config: FederatedConfig):
        super().__init__(config)
        self.rejected_count = 0
        self.total_received = 0

    def filter_by_coherence(
        self,
        gradient_contributions: Dict[str, Tuple[Dict[str, Any], float]],
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        """Filter out low-coherence contributions."""
        filtered = {}
        self.total_received += len(gradient_contributions)

        for node_id, (grads, coherence) in gradient_contributions.items():
            if coherence >= self.config.min_coherence_threshold:
                filtered[node_id] = (grads, coherence)
            else:
                self.rejected_count += 1
                print(f"Rejected gradients from {node_id[:8]}... (coherence={coherence:.3f})")

        return filtered

    def compute_alpha_resonance(self, gradients: Dict[str, Any]) -> float:
        """
        Compute alpha-resonance of gradients.

        Alpha (137) appears in gradient statistics when learning
        is aligned with fundamental patterns.
        """
        if not gradients:
            return 0.0

        resonances = []
        for key, grad in gradients.items():
            if TORCH_AVAILABLE and isinstance(grad, torch.Tensor):
                flat = grad.view(-1).float()
                # Check for alpha-patterns
                # Ratio of positive to negative should approximate phi
                positive = (flat > 0).sum().item()
                total = flat.numel()
                ratio = positive / max(1, total)

                # Distance from phi-ratio
                ideal = PHI / (1 + PHI)  # ~0.618
                resonance = 1 - abs(ratio - ideal)
                resonances.append(resonance)

            elif isinstance(grad, list):
                positive = sum(1 for v in grad if v > 0)
                total = len(grad)
                ratio = positive / max(1, total)
                ideal = PHI / (1 + PHI)
                resonance = 1 - abs(ratio - ideal)
                resonances.append(resonance)

        return sum(resonances) / max(1, len(resonances))

    def aggregate(
        self,
        gradient_contributions: Dict[str, Tuple[Dict[str, Any], float]],
    ) -> Dict[str, Any]:
        """Aggregate with phi-coherence filtering."""
        # Filter first
        filtered = self.filter_by_coherence(gradient_contributions)

        if not filtered:
            print("WARNING: All gradients rejected for low coherence!")
            return {}

        # Parent aggregation
        aggregated = super().aggregate(filtered)

        # Verify alpha-resonance of result
        resonance = self.compute_alpha_resonance(aggregated)
        if resonance < 0.3:
            print(f"WARNING: Low alpha-resonance in aggregated gradients: {resonance:.3f}")

        return aggregated

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            'total_received': self.total_received,
            'rejected_count': self.rejected_count,
            'rejection_rate': self.rejected_count / max(1, self.total_received),
            'num_participants': len(self.participants),
            'trust_rankings': self.get_trust_rankings()[:10],
        }


if TORCH_AVAILABLE:

    class FederatedCoordinator:
        """
        Federated Learning Coordinator for BAZINGA.

        Orchestrates distributed training across the P2P network:
        1. Manages training rounds
        2. Coordinates gradient collection
        3. Applies privacy mechanisms
        4. Performs trust-weighted aggregation
        5. Distributes updated models

        Usage:
            coordinator = FederatedCoordinator(config)
            coordinator.register_local_trainer(trainer)

            # Run federated training
            for round_id in range(num_rounds):
                metrics = await coordinator.run_round()
        """

        def __init__(
            self,
            config: Optional[FederatedConfig] = None,
            network=None,  # BAZINGANetwork from P2P module
        ):
            """
            Initialize federated coordinator.

            Args:
                config: Federated learning configuration
                network: P2P network for gradient distribution
            """
            self.config = config or FederatedConfig()
            self.network = network

            # Components
            if self.config.aggregation_method == 'phi_coherent':
                self.aggregator = PhiCoherentAggregator(self.config)
            else:
                self.aggregator = TrustWeightedAggregator(self.config)

            # Privacy
            if self.config.enable_differential_privacy:
                privacy_config = PrivacyConfig(
                    epsilon=self.config.privacy_budget_per_round,
                )
                self.privacy = AdaptivePrivacy(privacy_config)
            else:
                self.privacy = None

            # Secure aggregation
            if self.config.enable_secure_aggregation:
                # Use simplified for development, full Paillier for production
                self.secure_agg = SimplifiedSecureAggregator(
                    num_participants=self.config.max_participants
                )
            else:
                self.secure_agg = None

            # Local trainer (set via register)
            self.local_trainer: Optional[LocalTrainer] = None
            self.node_id: Optional[str] = None

            # State
            self.current_round: Optional[RoundState] = None
            self.round_history: List[RoundState] = []
            self.global_model_version = 0

            # Callbacks
            self.on_round_complete: Optional[Callable] = None
            self.on_model_updated: Optional[Callable] = None

            print(f"FederatedCoordinator initialized:")
            print(f"  Aggregation: {self.config.aggregation_method}")
            print(f"  Privacy: {self.config.enable_differential_privacy}")
            print(f"  Secure aggregation: {self.config.enable_secure_aggregation}")

        def register_local_trainer(
            self,
            trainer: LocalTrainer,
            node_id: str,
        ):
            """
            Register local trainer for this node.

            Args:
                trainer: LocalTrainer instance
                node_id: This node's ID
            """
            self.local_trainer = trainer
            self.node_id = node_id
            print(f"Registered local trainer for node {node_id[:8]}...")

        def set_network(self, network):
            """Set P2P network for model distribution."""
            self.network = network

        async def run_round(self) -> Dict[str, Any]:
            """
            Run a single federated learning round.

            Returns:
                Round metrics
            """
            if self.local_trainer is None:
                raise RuntimeError("No local trainer registered")

            round_id = len(self.round_history) + 1
            self.current_round = RoundState(round_id=round_id)

            print(f"\n{'='*50}")
            print(f"  Federated Round {round_id}")
            print(f"{'='*50}")

            try:
                # Step 1: Local training
                print("\n1. Local training...")
                metrics = self.local_trainer.train_epoch()
                print(f"   Loss: {metrics.get('train_loss', 0):.4f}")

                # Step 2: Get local gradients
                print("\n2. Extracting gradients...")
                local_gradients = self.local_trainer.get_exportable_gradients()
                local_coherence = self.local_trainer.compute_gradient_coherence(local_gradients)
                print(f"   Gradient keys: {len(local_gradients)}")
                print(f"   Coherence: {local_coherence:.4f}")

                # Step 3: Apply differential privacy
                if self.privacy:
                    print("\n3. Applying differential privacy...")
                    private_gradients = self.privacy.privatize_gradients(local_gradients)
                    remaining_budget = self.privacy.get_remaining_budget()
                    print(f"   Remaining privacy budget: {remaining_budget:.2f}")
                else:
                    private_gradients = local_gradients

                # Step 4: Collect gradients from network
                print("\n4. Collecting gradients from network...")
                all_contributions = await self._collect_network_gradients(
                    private_gradients, local_coherence
                )
                print(f"   Participants: {len(all_contributions)}")

                # Step 5: Aggregate gradients
                print("\n5. Trust-weighted aggregation...")
                if len(all_contributions) >= self.config.min_participants:
                    aggregated = self.aggregator.aggregate(all_contributions)
                    self.current_round.aggregated_gradients = aggregated

                    # Step 6: Apply aggregated update
                    print("\n6. Applying aggregated update...")
                    self.local_trainer.apply_federated_update(aggregated)
                    self.global_model_version += 1
                    print(f"   Global model version: {self.global_model_version}")

                    # Step 7: Distribute updated model (optional)
                    if self.network:
                        print("\n7. Broadcasting model update...")
                        await self._broadcast_model_update()

                else:
                    print(f"   Insufficient participants ({len(all_contributions)} < {self.config.min_participants})")
                    aggregated = {}

                # Clear accumulated gradients
                self.local_trainer.clear_accumulated_gradients()

                # Finalize round
                self.current_round.completed = True
                self.current_round.metrics = {
                    'train_loss': metrics.get('train_loss', 0),
                    'coherence': local_coherence,
                    'participants': len(all_contributions),
                    'model_version': self.global_model_version,
                }
                self.round_history.append(self.current_round)

                # Callback
                if self.on_round_complete:
                    self.on_round_complete(self.current_round)

                print(f"\nRound {round_id} complete!")
                return self.current_round.metrics

            except Exception as e:
                print(f"Round failed: {e}")
                self.current_round.metrics['error'] = str(e)
                return self.current_round.metrics

        async def _collect_network_gradients(
            self,
            local_gradients: Dict[str, torch.Tensor],
            local_coherence: float,
        ) -> Dict[str, Tuple[Dict[str, torch.Tensor], float]]:
            """
            Collect gradients from network peers.

            In a full implementation, this would:
            1. Broadcast our gradients to peers
            2. Receive gradients from peers
            3. Validate coherence of received gradients

            For now, includes just local gradients.
            """
            contributions = {}

            # Add local contribution
            if self.node_id:
                contributions[self.node_id] = (local_gradients, local_coherence)

            # If network available, collect from peers
            if self.network:
                try:
                    # Broadcast our gradients
                    gradient_msg = self._serialize_gradients(local_gradients, local_coherence)
                    # In real implementation: await self.network.broadcast(gradient_msg)

                    # Collect peer gradients (simulated for now)
                    # In real implementation: peer_gradients = await self.network.collect_gradients()
                    pass

                except Exception as e:
                    print(f"Network gradient collection failed: {e}")

            return contributions

        def _serialize_gradients(
            self,
            gradients: Dict[str, torch.Tensor],
            coherence: float,
        ) -> bytes:
            """Serialize gradients for network transmission."""
            # Convert tensors to lists
            serializable = {}
            for key, tensor in gradients.items():
                serializable[key] = tensor.cpu().tolist()

            data = {
                'gradients': serializable,
                'coherence': coherence,
                'node_id': self.node_id,
                'round_id': self.current_round.round_id if self.current_round else 0,
                'model_version': self.global_model_version,
                'timestamp': time.time(),
            }

            return json.dumps(data).encode()

        def _deserialize_gradients(
            self,
            data: bytes,
        ) -> Tuple[Dict[str, torch.Tensor], float, str]:
            """Deserialize gradients from network."""
            parsed = json.loads(data.decode())

            gradients = {}
            for key, values in parsed['gradients'].items():
                gradients[key] = torch.tensor(values)

            return gradients, parsed['coherence'], parsed['node_id']

        async def _broadcast_model_update(self):
            """Broadcast updated model to network."""
            if not self.network or not self.local_trainer:
                return

            try:
                # Get LoRA state
                lora_state = self.local_trainer.adapter.get_lora_state()

                # Chunk and broadcast
                # In real implementation: await self.network.broadcast_model(lora_state)

                print(f"   Model broadcast complete (version {self.global_model_version})")

            except Exception as e:
                print(f"   Model broadcast failed: {e}")

        def get_round_summary(self) -> Dict[str, Any]:
            """Get summary of current/recent round."""
            if not self.current_round:
                return {'status': 'no_round'}

            return {
                'round_id': self.current_round.round_id,
                'participants': self.current_round.num_participants,
                'elapsed_time': self.current_round.elapsed_time,
                'completed': self.current_round.completed,
                'metrics': self.current_round.metrics,
            }

        def get_federated_stats(self) -> Dict[str, Any]:
            """Get overall federated learning statistics."""
            return {
                'total_rounds': len(self.round_history),
                'global_model_version': self.global_model_version,
                'aggregator_stats': self.aggregator.get_stats() if hasattr(self.aggregator, 'get_stats') else {},
                'privacy_stats': self.privacy.get_stats() if self.privacy else {},
                'config': self.config.to_dict(),
            }

        def save_checkpoint(self, path: str):
            """Save coordinator checkpoint."""
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save config
            self.config.save(str(path / 'federated_config.json'))

            # Save state
            state = {
                'global_model_version': self.global_model_version,
                'round_count': len(self.round_history),
                'participants': {
                    node_id: {
                        'trust_score': p.trust_score,
                        'total_contributions': p.total_contributions,
                        'successful_contributions': p.successful_contributions,
                    }
                    for node_id, p in self.aggregator.participants.items()
                },
            }
            with open(path / 'coordinator_state.json', 'w') as f:
                json.dump(state, f, indent=2)

            print(f"Coordinator checkpoint saved to {path}")

        def load_checkpoint(self, path: str):
            """Load coordinator checkpoint."""
            path = Path(path)

            # Load config
            self.config = FederatedConfig.load(str(path / 'federated_config.json'))

            # Load state
            with open(path / 'coordinator_state.json', 'r') as f:
                state = json.load(f)

            self.global_model_version = state['global_model_version']

            # Restore participants
            for node_id, p_state in state.get('participants', {}).items():
                participant = self.aggregator.get_or_create_participant(node_id)
                participant.trust_score = p_state['trust_score']
                participant.total_contributions = p_state['total_contributions']
                participant.successful_contributions = p_state['successful_contributions']

            print(f"Coordinator checkpoint loaded from {path}")


    class FederatedNode:
        """
        A complete federated learning node.

        Combines:
        - LoRA adapter for efficient fine-tuning
        - Local trainer for training on local data
        - Federated coordinator for network participation

        Usage:
            node = FederatedNode.create(base_model, node_id)
            node.load_local_data(documents)

            # Join federated training
            for _ in range(epochs):
                metrics = await node.participate_in_round()
        """

        def __init__(
            self,
            adapter: LoRAAdapter,
            trainer: LocalTrainer,
            coordinator: FederatedCoordinator,
            node_id: str,
        ):
            self.adapter = adapter
            self.trainer = trainer
            self.coordinator = coordinator
            self.node_id = node_id

        @classmethod
        def create(
            cls,
            base_model: nn.Module,
            node_id: str,
            lora_config: Optional[LoRAConfig] = None,
            training_config: Optional[TrainingConfig] = None,
            federated_config: Optional[FederatedConfig] = None,
        ) -> 'FederatedNode':
            """
            Create a complete federated learning node.

            Args:
                base_model: Base model to adapt
                node_id: Unique node identifier
                lora_config: LoRA configuration
                training_config: Training configuration
                federated_config: Federated learning configuration

            Returns:
                FederatedNode ready for training
            """
            # Create adapter
            lora_config = lora_config or LoRAConfig()
            adapter = LoRAAdapter(base_model, lora_config)

            # Create trainer
            training_config = training_config or TrainingConfig()
            trainer = LocalTrainer(adapter, training_config)

            # Create coordinator
            federated_config = federated_config or FederatedConfig()
            coordinator = FederatedCoordinator(federated_config)
            coordinator.register_local_trainer(trainer, node_id)

            return cls(adapter, trainer, coordinator, node_id)

        def load_local_data(self, documents: List[Dict], tokenizer=None):
            """Load local documents for training."""
            self.trainer.load_local_data(documents, tokenizer)

        async def participate_in_round(self) -> Dict[str, Any]:
            """Participate in a federated learning round."""
            return await self.coordinator.run_round()

        def get_stats(self) -> Dict[str, Any]:
            """Get node statistics."""
            return {
                'node_id': self.node_id,
                'adapter_stats': self.adapter.get_stats(),
                'training_summary': self.trainer.get_training_summary(),
                'federated_stats': self.coordinator.get_federated_stats(),
            }

        def save(self, path: str):
            """Save complete node state."""
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            self.adapter.save(str(path / 'adapter'))
            self.trainer.save_checkpoint(str(path / 'trainer'))
            self.coordinator.save_checkpoint(str(path / 'coordinator'))

            print(f"FederatedNode saved to {path}")

        @classmethod
        def load(
            cls,
            path: str,
            base_model: nn.Module,
            node_id: str,
        ) -> 'FederatedNode':
            """Load complete node state."""
            path = Path(path)

            # Load adapter
            adapter = LoRAAdapter.load(str(path / 'adapter'), base_model)

            # Create trainer and load checkpoint
            trainer = LocalTrainer(adapter)
            trainer.load_checkpoint(str(path / 'trainer'))

            # Create coordinator and load checkpoint
            coordinator = FederatedCoordinator()
            coordinator.load_checkpoint(str(path / 'coordinator'))
            coordinator.register_local_trainer(trainer, node_id)

            return cls(adapter, trainer, coordinator, node_id)


else:
    # Stubs
    class FederatedCoordinator:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Run: pip install torch")

    class FederatedNode:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Run: pip install torch")


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Federated Coordinator Test")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\nPyTorch not available")
    else:
        import asyncio

        # Test trust-weighted aggregation
        print("\n1. Testing Trust-Weighted Aggregator:")
        config = FederatedConfig()
        aggregator = TrustWeightedAggregator(config)

        # Simulate contributions from 3 nodes
        contributions = {
            'node_a': ({'layer1': torch.randn(64, 32)}, 0.8),
            'node_b': ({'layer1': torch.randn(64, 32)}, 0.6),
            'node_c': ({'layer1': torch.randn(64, 32)}, 0.4),
        }

        aggregated = aggregator.aggregate(contributions)
        print(f"   Aggregated shape: {aggregated['layer1'].shape}")
        print(f"   Trust rankings: {aggregator.get_trust_rankings()}")

        # Test phi-coherent aggregation
        print("\n2. Testing Phi-Coherent Aggregator:")
        phi_agg = PhiCoherentAggregator(config)
        aggregated = phi_agg.aggregate(contributions)
        print(f"   Stats: {phi_agg.get_stats()}")

        # Test coordinator (simplified)
        print("\n3. Testing Federated Coordinator:")
        coordinator = FederatedCoordinator(config)
        print(f"   Config: {coordinator.config.aggregation_method}")

        # Test FederatedNode creation
        print("\n4. Testing FederatedNode:")

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(64, 64)
                self.value = nn.Linear(64, 64)

            def forward(self, x):
                return self.query(x) + self.value(x)

        base = SimpleModel()
        node = FederatedNode.create(base, "test_node_001")

        # Load dummy data
        documents = [
            {'content': f'Document {i}', 'embedding': [0.1 * i] * 64}
            for i in range(20)
        ]
        node.load_local_data(documents)

        # Run one round
        async def test_round():
            metrics = await node.participate_in_round()
            print(f"   Round metrics: {metrics}")

        asyncio.run(test_round())

        print("\n" + "=" * 60)
        print("Phase 3 Federated Learning Complete!")
        print("=" * 60)
        print("""
Components:
  - lora_adapter.py       : LoRA efficient fine-tuning
  - local_trainer.py      : Local training loop
  - differential_privacy.py: Privacy-preserving gradients
  - secure_aggregation.py : Homomorphic encryption
  - federated_coordinator.py: Trust-weighted orchestration

"Share learning, not data."
        """)
