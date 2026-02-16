"""
BAZINGA - Distributed AI that belongs to everyone
"Intelligence distributed, not controlled"

Usage:
    bazinga                      # Interactive mode
    bazinga --ask "question"     # Ask a question
    bazinga --code "task"        # Generate code
    bazinga --index ~/Documents  # Index a directory
    bazinga --help               # Show help
"""

from .cli import BAZINGA, main_sync, main

__version__ = "4.9.11"
__all__ = ['BAZINGA', 'main_sync', 'main', '__version__']

# New in v3.4.0: Quantum, ΛG, Tensor modules
def get_quantum_processor():
    """Get the Quantum Processor (superposition processing)."""
    from .quantum import QuantumProcessor
    return QuantumProcessor()

def get_lambda_g():
    """Get the ΛG Boundary Operator (solution emergence)."""
    from .lambda_g import LambdaGOperator
    return LambdaGOperator()

def get_tensor_engine():
    """Get the Tensor Intersection Engine (trust calculation)."""
    from .tensor import TensorIntersectionEngine
    return TensorIntersectionEngine()

def get_constants():
    """Get all universal constants (φ, α, ψ, etc.)."""
    from . import constants
    return constants

# Lazy imports for optional components
def get_intelligent_coder():
    """Get the IntelligentCoder (LLM-powered code generation)."""
    from .intelligent_coder import IntelligentCoder
    return IntelligentCoder()

def get_llm_orchestrator():
    """Get the LLM Orchestrator (multi-provider intelligence)."""
    from .llm_orchestrator import LLMOrchestrator
    return LLMOrchestrator()

def get_phi_coherence():
    """Get the φ-Coherence calculator."""
    from .phi_coherence import PhiCoherence
    return PhiCoherence()

def get_p2p_network():
    """Get the P2P network (BAZINGANetwork)."""
    from .p2p import BAZINGANetwork
    return BAZINGANetwork

def get_federated_coordinator():
    """Get the Federated Learning Coordinator."""
    from .federated.federated_coordinator import FederatedCoordinator
    return FederatedCoordinator

def get_federated_node():
    """Get the FederatedNode (complete federated learning node)."""
    from .federated.federated_coordinator import FederatedNode
    return FederatedNode

def get_local_model():
    """Get the LocalModel for on-device inference."""
    from .inference import LocalModel
    return LocalModel

def get_model_router():
    """Get the ModelRouter for intelligent query routing."""
    from .inference import ModelRouter
    return ModelRouter

def get_distributed_inference():
    """Get DistributedInference for P2P model serving."""
    from .inference import DistributedInference
    return DistributedInference

def get_dao_governance():
    """Get DAOGovernance for decentralized governance."""
    from .decentralized import DAOGovernance
    return DAOGovernance

def get_peer_discovery():
    """Get BootstrapFreeDiscovery for P2P peer discovery."""
    from .decentralized import BootstrapFreeDiscovery
    return BootstrapFreeDiscovery

def get_darmiyan_chain():
    """Get the Darmiyan blockchain for knowledge attestation."""
    from .blockchain import DarmiyanChain, create_chain
    return create_chain

def get_wallet():
    """Get a BAZINGA wallet (identity, not money)."""
    from .blockchain import Wallet, create_wallet
    return create_wallet

def get_pob_miner():
    """Get the Proof-of-Boundary miner (zero-energy)."""
    from .blockchain import PoBMiner
    return PoBMiner

def get_trust_oracle():
    """Get the Trust Oracle (chain → trust scores → AI routing)."""
    from .blockchain import TrustOracle, create_trust_oracle
    return create_trust_oracle

def get_inter_ai_consensus():
    """Get the Inter-AI Consensus engine (multi-AI agreement through φ-coherence)."""
    from .inter_ai import InterAIConsensus
    return InterAIConsensus

def multi_ai_ask(question: str, verbose: bool = True):
    """
    Ask multiple AIs a question and get consensus.

    Usage:
        import asyncio
        from bazinga import multi_ai_ask

        result = asyncio.run(multi_ai_ask("What is consciousness?"))
        print(result.understanding)
    """
    from .inter_ai import multi_ai_ask_sync
    return multi_ai_ask_sync(question, verbose)

def get_verified_fix_engine():
    """
    Get the Blockchain-Verified Fix Engine.

    NEW in v4.9.7: Multiple AIs must agree before applying code changes.

    Usage:
        from bazinga import get_verified_fix_engine
        engine = get_verified_fix_engine()

        # Create and apply a verified fix
        proposal = engine.create_proposal(
            file_path="utils.py",
            original_code="except:",
            proposed_fix="except Exception:",
            explanation="Replace bare except"
        )
        consensus = await engine.get_consensus(proposal)
        if consensus.consensus_reached:
            await engine.apply_fix(proposal)
    """
    from .agent.verified_fixes import VerifiedFixEngine
    return VerifiedFixEngine

def verified_code_fix(
    file_path: str,
    old_code: str,
    new_code: str,
    reason: str,
    verbose: bool = True
):
    """
    Apply a blockchain-verified code fix (sync version).

    Multiple AIs must reach consensus before the fix is applied.
    The fix is attested on the Darmiyan blockchain for audit trail.

    Usage:
        from bazinga import verified_code_fix

        success, msg = verified_code_fix(
            "utils.py",
            "except:",
            "except Exception:",
            "Replace bare except with specific exception"
        )
    """
    from .agent.verified_fixes import verified_fix_sync
    return verified_fix_sync(file_path, old_code, new_code, reason, verbose=verbose)
