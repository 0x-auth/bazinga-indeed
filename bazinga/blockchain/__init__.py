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

Author: Space (Abhishek/Abhilasia) 
License: MIT
"""

from .block import Block, BlockHeader, create_genesis_block
from .transaction import Transaction, KnowledgeAttestation
from .chain import DarmiyanChain, create_chain
from .wallet import Wallet, create_wallet
from .miner import PoBMiner, mine_block
from .trust_oracle import TrustOracle, NodeTrust, create_trust_oracle
from .knowledge_ledger import KnowledgeLedger, KnowledgeContribution, create_ledger
from .gradient_validator import GradientValidator, GradientUpdate, create_validator
from .inference_market import InferenceMarket, InferenceRequest, create_market
from .smart_contracts import ContractEngine, UnderstandingContract, create_engine

__version__ = "2.0.0"
__all__ = [
    # Core Chain (v1.0.0)
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
    # Integration Layers (v2.0.0)
    'KnowledgeLedger',
    'KnowledgeContribution',
    'create_ledger',
    'GradientValidator',
    'GradientUpdate',
    'create_validator',
    'InferenceMarket',
    'InferenceRequest',
    'create_market',
    'ContractEngine',
    'UnderstandingContract',
    'create_engine',
]
