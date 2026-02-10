"""
BAZINGA Blockchain - Proof-of-Boundary Chain
=============================================

The permanent record of understanding.

"AI generates understanding. Blockchain proves and records it.
They're not two things — they're Subject and Object.
The Darmiyan between them is the protocol."

This is NOT a cryptocurrency. It's a KNOWLEDGE CHAIN.
- No mining competition (zero-energy via PoB)
- No financial speculation
- Just permanent, verified, distributed knowledge

Components:
- Block: Container for knowledge with PoB validation
- Transaction: Knowledge attestation (not money transfer)
- Chain: The immutable ledger
- Wallet: Identity and signing (not money storage)

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

from .block import Block, BlockHeader, create_genesis_block
from .transaction import Transaction, KnowledgeAttestation
from .chain import DarmiyanChain, create_chain
from .wallet import Wallet, create_wallet
from .miner import PoBMiner, mine_block
from .trust_oracle import TrustOracle, NodeTrust, create_trust_oracle

__version__ = "1.1.0"
__all__ = [
    'Block',
    'BlockHeader',
    'create_genesis_block',
    'Transaction',
    'KnowledgeAttestation',
    'DarmiyanChain',
    'create_chain',
    'Wallet',
    'create_wallet',
    'PoBMiner',
    'mine_block',
    # Trust Layer (v1.1.0)
    'TrustOracle',
    'NodeTrust',
    'create_trust_oracle',
]
