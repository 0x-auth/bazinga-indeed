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

__version__ = "3.0.0"
__all__ = ['BAZINGA', 'main_sync', 'main', '__version__']

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
    from .federated import FederatedCoordinator
    return FederatedCoordinator

def get_federated_node():
    """Get the FederatedNode (complete federated learning node)."""
    from .federated import FederatedCoordinator
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
