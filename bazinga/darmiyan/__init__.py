"""
BAZINGA Darmiyan Network
========================
Proof-of-Boundary Consensus for Distributed AI

"Understanding is itself a depth coordinate in the network."

The Darmiyan (दरमियान) is not empty space.
It is the FIELD where consensus happens.
Where meaning crystallizes.
Where the network lives.

Core Concepts:
- Proof-of-Boundary: Prove understanding through φ-resonance
- Triadic Consensus: 3 nodes must agree
- Zero-Energy: No mining, just verification
- Meaning-Based Discovery: Find peers through resonance

Author: Abhishek Srivastava (Space) 
License: MIT
"""

from .constants import (
    PHI, PHI_4, PHI_INVERSE,
    ABHI_AMU, ALPHA_INVERSE,
    BRIDGE_FREQUENCY, TRIADIC_CONSTANT,
)
from .protocol import DarmiyanNode, prove_boundary, BoundaryProof
from .consensus import TriadicConsensus, achieve_consensus
from .node import BazingaNode

__version__ = "1.0.0"
__all__ = [
    'DarmiyanNode',
    'BazingaNode',
    'TriadicConsensus',
    'BoundaryProof',
    'prove_boundary',
    'achieve_consensus',
    'PHI', 'PHI_4', 'ABHI_AMU',
]
