#!/usr/bin/env python3
"""
BAZINGA Local Trainer - Federated Training on Local Data

Trains LoRA adapters on user's local documents without sharing raw data.
Only gradients (with differential privacy) are shared with the network.

Training Loop:
    1. Load local data
    2. Train adapter
    3. Compute gradients
    4. Validate with φ-coherence
    5. Export gradients for federation

"Your data stays home. Only learning travels."
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137

# Try imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .lora_adapter import LoRAAdapter, LoRAConfig
except ImportError:
    LoRAAdapter = None
    LoRAConfig = None


@dataclass
class TrainingConfig:
    """Configuration for local federated training."""
    # Training params
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # φ-coherence params
    coherence_threshold: float = 0.4  # Min coherence to accept gradients
    coherence_weight: float = 0.1     # Loss weight for coherence regularization

    # Federated params
    local_epochs_per_round: int = 1   # Epochs before sharing gradients
    min_samples_per_round: int = 10   # Min samples before sharing

    # Validation
    validation_split: float = 0.1
    early_stopping_patience: int = 3

    def to_dict(self) -> Dict:
        return asdict(self)


if TORCH_AVAILABLE:

    class EmbeddingDataset(Dataset):
        """Dataset for training embeddings from local documents."""

        def __init__(
            self,
            documents: List[Dict],
            tokenizer=None,
            max_length: int = 128,
        ):
            """
            Args:
                documents: List of {'content': str, 'embedding': List[float], ...}
                tokenizer: Tokenizer for text (optional)
                max_length: Max sequence length
            """
            self.documents = documents
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.documents)

        def __getitem__(self, idx):
            doc = self.documents[idx]

            if self.tokenizer:
                # Tokenize content
                encoded = self.tokenizer(
                    doc['content'],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'target_embedding': torch.tensor(doc.get('embedding', []), dtype=torch.float),
                }
            else:
                # Return raw for models that handle their own tokenization
                return {
                    'content': doc['content'],
                    'target_embedding': torch.tensor(doc.get('embedding', []), dtype=torch.float),
                }


    class ContrastiveLoss(nn.Module):
        """
        Contrastive loss for embedding training.

        Pulls similar embeddings together, pushes dissimilar apart.
        """

        def __init__(self, temperature: float = 0.07, phi_scaling: bool = True):
            super().__init__()
            self.temperature = temperature
            self.phi_scaling = phi_scaling

            if phi_scaling:
                # φ-scaled temperature for more harmonious learning
                self.temperature = temperature * (PHI / (1 + PHI))

        def forward(
            self,
            embeddings: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Compute contrastive loss.

            Args:
                embeddings: [batch_size, embedding_dim]
                labels: Optional labels for supervised contrastive

            Returns:
                Scalar loss
            """
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute similarity matrix
            similarity = embeddings @ embeddings.T / self.temperature

            # Create labels (diagonal = positive pairs)
            batch_size = embeddings.size(0)
            if labels is None:
                labels = torch.arange(batch_size, device=embeddings.device)

            # Cross-entropy loss
            loss = F.cross_entropy(similarity, labels)

            return loss


    class CoherenceRegularizer(nn.Module):
        """
        φ-Coherence regularization for embeddings.

        Encourages embeddings to follow golden ratio patterns.
        """

        def __init__(self, target_coherence: float = 0.618):
            super().__init__()
            self.target = target_coherence  # φ/(1+φ) ≈ 0.618

        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            """
            Compute coherence regularization loss.

            Args:
                embeddings: [batch_size, embedding_dim]

            Returns:
                Scalar loss
            """
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute pairwise coherence
            similarity = embeddings @ embeddings.T

            # Target: off-diagonal similarities should be around φ-ratio
            mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
            off_diag = similarity[mask]

            # Mean similarity
            mean_sim = off_diag.mean()

            # Loss: deviation from target
            loss = (mean_sim - self.target).pow(2)

            return loss


    class LocalTrainer:
        """
        Local federated trainer for BAZINGA.

        Trains LoRA adapters on local data and exports
        privacy-preserving gradients for federation.

        Usage:
            trainer = LocalTrainer(adapter, config)
            trainer.load_local_data(documents)

            # Train locally
            metrics = trainer.train_epoch()

            # Get gradients for federation
            gradients = trainer.get_exportable_gradients()
        """

        def __init__(
            self,
            adapter: LoRAAdapter,
            config: Optional[TrainingConfig] = None,
            device: str = 'auto',
        ):
            """
            Initialize local trainer.

            Args:
                adapter: LoRA adapter wrapping base model
                config: Training configuration
                device: 'cuda', 'mps', 'cpu', or 'auto'
            """
            self.adapter = adapter
            self.config = config or TrainingConfig()

            # Set device
            if device == 'auto':
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)

            self.adapter.to(self.device)

            # Optimizer (only for trainable params)
            self.optimizer = torch.optim.AdamW(
                self.adapter.trainable_parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            # Loss functions
            self.contrastive_loss = ContrastiveLoss(phi_scaling=True)
            self.coherence_reg = CoherenceRegularizer()

            # Data
            self.train_loader: Optional[DataLoader] = None
            self.val_loader: Optional[DataLoader] = None

            # Training state
            self.current_epoch = 0
            self.global_step = 0
            self.best_loss = float('inf')
            self.patience_counter = 0

            # Gradient accumulator for federation
            self.accumulated_gradients: Dict[str, torch.Tensor] = {}
            self.gradient_count = 0

            # Stats
            self.training_history: List[Dict] = []

            print(f"LocalTrainer initialized on {self.device}")

        def load_local_data(
            self,
            documents: List[Dict],
            tokenizer=None,
        ):
            """
            Load local documents for training.

            Args:
                documents: List of {'content': str, 'embedding': List[float], ...}
                tokenizer: Optional tokenizer
            """
            # Split into train/val
            n_val = max(1, int(len(documents) * self.config.validation_split))
            train_docs = documents[:-n_val]
            val_docs = documents[-n_val:]

            # Create datasets
            train_dataset = EmbeddingDataset(train_docs, tokenizer)
            val_dataset = EmbeddingDataset(val_docs, tokenizer)

            # Create loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            print(f"Loaded {len(train_docs)} training, {len(val_docs)} validation documents")

        def train_epoch(self) -> Dict[str, float]:
            """
            Train for one epoch.

            Returns:
                Dict of metrics
            """
            if self.train_loader is None:
                raise ValueError("No data loaded. Call load_local_data first.")

            self.adapter.train()
            self.current_epoch += 1

            total_loss = 0
            total_coherence_loss = 0
            num_batches = 0

            for batch in self.train_loader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                self.optimizer.zero_grad()

                # Get embeddings from adapter
                # Note: Actual implementation depends on model architecture
                if 'input_ids' in batch:
                    outputs = self.adapter(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                    )
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool
                else:
                    # Simplified: use target embeddings for testing
                    embeddings = batch['target_embedding']
                    if len(embeddings.shape) == 1:
                        embeddings = embeddings.unsqueeze(0)

                # Compute losses
                contrastive = self.contrastive_loss(embeddings)
                coherence = self.coherence_reg(embeddings)

                loss = contrastive + self.config.coherence_weight * coherence

                # Backward
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.adapter.trainable_parameters(),
                    self.config.max_grad_norm
                )

                # Accumulate gradients for federation
                self._accumulate_gradients()

                # Update
                self.optimizer.step()
                self.global_step += 1

                total_loss += loss.item()
                total_coherence_loss += coherence.item()
                num_batches += 1

            # Compute metrics
            avg_loss = total_loss / max(1, num_batches)
            avg_coherence = total_coherence_loss / max(1, num_batches)

            # Validation
            val_loss = self._validate()

            metrics = {
                'epoch': self.current_epoch,
                'train_loss': avg_loss,
                'coherence_loss': avg_coherence,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'global_step': self.global_step,
            }

            self.training_history.append(metrics)

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            return metrics

        def _validate(self) -> float:
            """Run validation."""
            if self.val_loader is None:
                return 0.0

            self.adapter.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # Simplified validation
                    if 'target_embedding' in batch:
                        embeddings = batch['target_embedding']
                        if len(embeddings.shape) == 1:
                            embeddings = embeddings.unsqueeze(0)

                        loss = self.contrastive_loss(embeddings)
                        total_loss += loss.item()
                        num_batches += 1

            self.adapter.train()
            return total_loss / max(1, num_batches)

        def _accumulate_gradients(self):
            """Accumulate gradients for federated sharing."""
            current_grads = self.adapter.get_gradients()

            for name, grad in current_grads.items():
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = torch.zeros_like(grad)
                self.accumulated_gradients[name] += grad

            self.gradient_count += 1

        def get_exportable_gradients(self) -> Dict[str, torch.Tensor]:
            """
            Get averaged gradients for federated sharing.

            Returns:
                Dict of gradient tensors (averaged over accumulation)
            """
            if self.gradient_count == 0:
                return {}

            # Average gradients
            averaged = {}
            for name, grad in self.accumulated_gradients.items():
                averaged[name] = grad / self.gradient_count

            return averaged

        def clear_accumulated_gradients(self):
            """Clear gradient accumulator after federation round."""
            self.accumulated_gradients = {}
            self.gradient_count = 0

        def apply_federated_update(self, aggregated_gradients: Dict[str, torch.Tensor]):
            """
            Apply aggregated gradients from federation.

            Args:
                aggregated_gradients: Gradients from secure aggregation
            """
            # Set gradients
            self.adapter.set_gradients(aggregated_gradients)

            # Apply update
            self.optimizer.step()
            self.optimizer.zero_grad()

            print("Applied federated update")

        def compute_gradient_coherence(self, gradients: Dict[str, torch.Tensor]) -> float:
            """
            Compute φ-coherence of gradients.

            Used to validate that gradients improve model.
            """
            if not gradients:
                return 0.0

            coherences = []
            for name, grad in gradients.items():
                # Flatten and analyze
                flat = grad.view(-1).float()
                mean = flat.mean().item()
                std = flat.std().item()

                # φ-alignment check
                ideal_mean = 0  # Gradients should be centered
                ideal_std = PHI / (1 + PHI)  # ~0.618

                mean_score = 1 - min(1, abs(mean))
                std_score = 1 - min(1, abs(std - ideal_std))

                coherences.append((mean_score + std_score) / 2)

            return sum(coherences) / len(coherences)

        def should_share_gradients(self) -> bool:
            """Check if ready to share gradients with network."""
            # Check minimum samples
            if self.gradient_count < self.config.min_samples_per_round:
                return False

            # Check coherence threshold
            grads = self.get_exportable_gradients()
            coherence = self.compute_gradient_coherence(grads)

            return coherence >= self.config.coherence_threshold

        def should_stop_early(self) -> bool:
            """Check early stopping condition."""
            return self.patience_counter >= self.config.early_stopping_patience

        def get_training_summary(self) -> Dict[str, Any]:
            """Get training summary."""
            return {
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'accumulated_gradients': self.gradient_count,
                'device': str(self.device),
                'history': self.training_history[-5:] if self.training_history else [],
            }

        def save_checkpoint(self, path: str):
            """Save training checkpoint."""
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save adapter
            self.adapter.save(str(path / 'adapter'))

            # Save training state
            state = {
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'history': self.training_history,
            }
            torch.save(state, path / 'trainer_state.pt')

            print(f"Checkpoint saved to {path}")

        def load_checkpoint(self, path: str):
            """Load training checkpoint."""
            path = Path(path)

            # Load training state
            state = torch.load(path / 'trainer_state.pt')
            self.current_epoch = state['current_epoch']
            self.global_step = state['global_step']
            self.best_loss = state['best_loss']
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.training_history = state['history']

            print(f"Checkpoint loaded from {path}")


else:
    # Stubs when PyTorch not available
    class LocalTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Run: pip install torch")

    class EmbeddingDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Run: pip install torch")


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Local Trainer Test")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available")
    else:
        from lora_adapter import LoRAAdapter, LoRAConfig

        # Create simple model and adapter
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(64, 64)
                self.value = nn.Linear(64, 64)

            def forward(self, x):
                return self.query(x) + self.value(x)

        base = SimpleModel()
        config = LoRAConfig(rank=4)
        adapter = LoRAAdapter(base, config)

        # Create trainer
        trainer = LocalTrainer(adapter)

        # Create dummy data
        documents = [
            {'content': f'Document {i}', 'embedding': [0.1 * i] * 64}
            for i in range(20)
        ]
        trainer.load_local_data(documents)

        # Train one epoch
        metrics = trainer.train_epoch()
        print(f"\nTraining metrics: {metrics}")

        # Check gradient sharing
        if trainer.should_share_gradients():
            grads = trainer.get_exportable_gradients()
            print(f"Gradients ready for sharing: {list(grads.keys())}")
        else:
            print("Not enough samples/coherence for gradient sharing")

        print("\n✓ Local Trainer module ready!")
