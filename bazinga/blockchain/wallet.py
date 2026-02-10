#!/usr/bin/env python3
"""
BAZINGA Wallet - Identity Management
=====================================

This is NOT a money wallet. It's an IDENTITY wallet.

What it holds:
- Your node's private key (for signing)
- Your public key (for verification)
- Your identity on the network
- Your reputation/trust score

What it does NOT hold:
- Cryptocurrency
- Tokens
- Financial assets

"Your value is not in what you hold, but in what you UNDERSTAND."
"""

import json
import hashlib
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Import constants
try:
    from ..darmiyan.constants import PHI, ABHI_AMU, ALPHA_INVERSE
except ImportError:
    PHI = 1.618033988749895
    ABHI_AMU = 515
    ALPHA_INVERSE = 137


@dataclass
class Identity:
    """Node identity on the Darmiyan network."""
    node_id: str
    public_key: str
    node_type: str = "full"  # "full", "light", "validator"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reputation:
    """Reputation tracking for a wallet."""
    knowledge_contributed: int = 0
    learning_contributions: int = 0
    consensus_participations: int = 0
    successful_proofs: int = 0
    failed_proofs: int = 0
    trust_score: float = 0.5  # 0-1 scale

    def update_trust(self):
        """Recalculate trust score based on activity."""
        if self.successful_proofs + self.failed_proofs == 0:
            return

        # φ-weighted trust calculation
        proof_ratio = self.successful_proofs / (self.successful_proofs + self.failed_proofs)
        contribution_factor = min(1.0, (self.knowledge_contributed + self.learning_contributions) / 100)

        # Trust = proof_ratio * φ^-1 + contribution * (1 - φ^-1)
        phi_inv = 1 / PHI
        self.trust_score = proof_ratio * phi_inv + contribution_factor * (1 - phi_inv)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'knowledge_contributed': self.knowledge_contributed,
            'learning_contributions': self.learning_contributions,
            'consensus_participations': self.consensus_participations,
            'successful_proofs': self.successful_proofs,
            'failed_proofs': self.failed_proofs,
            'trust_score': self.trust_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Reputation':
        return cls(**data)


class Wallet:
    """
    BAZINGA identity wallet.

    Manages node identity and signing capabilities.
    This is NOT a financial wallet - it's an IDENTITY wallet.

    Usage:
        wallet = Wallet()
        wallet.generate_keys()

        # Sign transactions
        signature = wallet.sign(data)

        # Verify signatures
        valid = wallet.verify(data, signature, public_key)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".bazinga" / "wallet"
        self.node_id = node_id

        # Keys (simple hash-based for now)
        self.private_key: str = ""
        self.public_key: str = ""

        # Identity and reputation
        self.identity: Optional[Identity] = None
        self.reputation = Reputation()

        # Activity log
        self.transactions_signed: int = 0
        self.proofs_generated: int = 0

        # Initialize
        self._initialize()

    def _initialize(self):
        """Initialize wallet, loading from disk if exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        wallet_file = self.data_dir / "wallet.json"
        if wallet_file.exists():
            self._load()
        elif self.node_id:
            self.generate_keys(self.node_id)

    def generate_keys(self, seed: Optional[str] = None):
        """
        Generate keypair for signing.

        Uses a simple hash-based approach. For production,
        this would use proper asymmetric cryptography (Ed25519, etc.)
        """
        if seed is None:
            seed = f"{time.time()}{os.urandom(32).hex()}"

        # Generate private key
        private_data = f"BAZINGA_PRIVATE_{seed}_{time.time()}"
        self.private_key = hashlib.sha256(private_data.encode()).hexdigest()

        # Derive public key
        public_data = f"BAZINGA_PUBLIC_{self.private_key}"
        self.public_key = hashlib.sha256(public_data.encode()).hexdigest()

        # Generate node_id if not provided
        if not self.node_id:
            self.node_id = f"bzn_{self.public_key[:16]}"

        # Create identity
        self.identity = Identity(
            node_id=self.node_id,
            public_key=self.public_key,
        )

        # Save
        self._save()

        return self.public_key

    def sign(self, data: str) -> str:
        """
        Sign data with private key.

        Args:
            data: String data to sign

        Returns:
            Signature hex string
        """
        if not self.private_key:
            raise ValueError("No private key - call generate_keys() first")

        # Simple HMAC-like signature
        sign_data = f"{data}{self.private_key}"
        signature = hashlib.sha256(sign_data.encode()).hexdigest()

        self.transactions_signed += 1
        return signature

    def verify(self, data: str, signature: str, public_key: Optional[str] = None) -> bool:
        """
        Verify a signature.

        For verifying our own signatures, public_key can be omitted.
        For verifying others' signatures, provide their public_key.
        """
        # For our own signatures
        if public_key is None or public_key == self.public_key:
            expected = hashlib.sha256(f"{data}{self.private_key}".encode()).hexdigest()
            return signature == expected

        # For others' signatures (can't verify without shared secret)
        # In production, this would use proper asymmetric verification
        return False

    def sign_transaction(self, transaction: Dict) -> str:
        """Sign a transaction dictionary."""
        tx_data = json.dumps(transaction, sort_keys=True)
        return self.sign(tx_data)

    def record_proof(self, success: bool):
        """Record a PoB proof attempt."""
        self.proofs_generated += 1
        if success:
            self.reputation.successful_proofs += 1
        else:
            self.reputation.failed_proofs += 1
        self.reputation.update_trust()
        self._save()

    def record_knowledge(self):
        """Record a knowledge contribution."""
        self.reputation.knowledge_contributed += 1
        self.reputation.update_trust()
        self._save()

    def record_learning(self):
        """Record a federated learning contribution."""
        self.reputation.learning_contributions += 1
        self.reputation.update_trust()
        self._save()

    def get_address(self) -> str:
        """Get the wallet address (truncated public key)."""
        if not self.public_key:
            return ""
        return f"bzn:{self.public_key[:12]}...{self.public_key[-4:]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get wallet statistics."""
        return {
            'node_id': self.node_id,
            'address': self.get_address(),
            'public_key': self.public_key[:16] + "..." if self.public_key else None,
            'transactions_signed': self.transactions_signed,
            'proofs_generated': self.proofs_generated,
            'reputation': self.reputation.to_dict(),
        }

    def export_identity(self) -> Dict[str, Any]:
        """Export public identity for sharing."""
        if not self.identity:
            return {}

        return {
            'node_id': self.identity.node_id,
            'public_key': self.identity.public_key,
            'node_type': self.identity.node_type,
            'created_at': self.identity.created_at,
        }

    def _save(self):
        """Save wallet to disk."""
        wallet_file = self.data_dir / "wallet.json"

        data = {
            'node_id': self.node_id,
            'private_key': self.private_key,  # Would be encrypted in production!
            'public_key': self.public_key,
            'identity': {
                'node_id': self.identity.node_id,
                'public_key': self.identity.public_key,
                'node_type': self.identity.node_type,
                'created_at': self.identity.created_at,
                'metadata': self.identity.metadata,
            } if self.identity else None,
            'reputation': self.reputation.to_dict(),
            'transactions_signed': self.transactions_signed,
            'proofs_generated': self.proofs_generated,
            'saved_at': time.time(),
        }

        with open(wallet_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load wallet from disk."""
        wallet_file = self.data_dir / "wallet.json"

        with open(wallet_file, 'r') as f:
            data = json.load(f)

        self.node_id = data.get('node_id')
        self.private_key = data.get('private_key', '')
        self.public_key = data.get('public_key', '')
        self.transactions_signed = data.get('transactions_signed', 0)
        self.proofs_generated = data.get('proofs_generated', 0)

        if data.get('identity'):
            self.identity = Identity(
                node_id=data['identity']['node_id'],
                public_key=data['identity']['public_key'],
                node_type=data['identity'].get('node_type', 'full'),
                created_at=data['identity'].get('created_at', time.time()),
                metadata=data['identity'].get('metadata', {}),
            )

        if data.get('reputation'):
            self.reputation = Reputation.from_dict(data['reputation'])

    def print_status(self):
        """Print wallet status."""
        print("\n" + "=" * 60)
        print("  BAZINGA WALLET")
        print("=" * 60)
        print(f"  Node ID: {self.node_id}")
        print(f"  Address: {self.get_address()}")
        print(f"  Type: {self.identity.node_type if self.identity else 'unknown'}")
        print("-" * 60)
        print(f"  Transactions Signed: {self.transactions_signed}")
        print(f"  PoB Proofs: {self.proofs_generated}")
        print(f"  Knowledge Contributed: {self.reputation.knowledge_contributed}")
        print(f"  Learning Contributions: {self.reputation.learning_contributions}")
        print("-" * 60)
        print(f"  Trust Score: {self.reputation.trust_score:.3f}")
        print("=" * 60)


def create_wallet(node_id: Optional[str] = None, data_dir: Optional[str] = None) -> Wallet:
    """Create a new wallet."""
    wallet = Wallet(data_dir=data_dir, node_id=node_id)
    if not wallet.public_key:
        wallet.generate_keys(node_id)
    return wallet


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  BAZINGA WALLET TEST")
    print("=" * 60)
    print()

    # Create wallet in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        wallet = create_wallet(node_id="test_node", data_dir=tmpdir)

        print(f"Wallet created:")
        print(f"  Node ID: {wallet.node_id}")
        print(f"  Address: {wallet.get_address()}")
        print(f"  Public Key: {wallet.public_key[:32]}...")
        print()

        # Sign some data
        print("Testing signing...")
        test_data = "This is a test transaction"
        signature = wallet.sign(test_data)
        print(f"  Data: {test_data}")
        print(f"  Signature: {signature[:32]}...")
        print(f"  Verified: {wallet.verify(test_data, signature)}")
        print(f"  Wrong data: {wallet.verify('wrong data', signature)}")
        print()

        # Record some activity
        print("Recording activity...")
        for _ in range(5):
            wallet.record_proof(True)
        for _ in range(2):
            wallet.record_proof(False)
        for _ in range(10):
            wallet.record_knowledge()
        for _ in range(3):
            wallet.record_learning()
        print()

        # Show status
        wallet.print_status()

        # Export identity
        print("\nExported Identity:")
        print(f"  {wallet.export_identity()}")

    print("\n  Wallet test complete!")
