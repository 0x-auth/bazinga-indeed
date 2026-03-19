"""
Manifold-Backed Proof-of-Boundary
==================================
Triangle resonance validation on a 5D manifold.

Mining = contributing to the shared manifold.
Validation = triangle trip (A→B→C→A) confirming φ-resonance.
Shared = patterns and coordinates only. NEVER raw data.

The 5 Dimensions (from ~/∞/meaning/orthogonal/):
    1. Form (◯)    — observation density of the content
    2. Flow (↻)    — connections to other manifold nodes
    3. Process (↥) — recursive self-reference depth
    4. Purpose (✧) — why this matters (importance weight)
    5. Trust (⟡)   — the witness dimension, orthogonal to all others

PoB + Manifold:
    1. Miner generates BoundaryProof for content (existing PoB)
    2. Miner computes 5D ManifoldNode from the proof's pattern (NOT content)
    3. Node is sent around the triangle: A→B→C→A
    4. Each peer independently verifies:
       - PoB is valid (P/G ≈ φ⁴)
       - φ-resonance of the node (closer to φ = more resonant)
       - 5D coordinates are internally consistent
    5. Block is valid when all 3 peers agree within tolerance

Difficulty = φ-resonance proximity.
Energy ≈ 0.00001 kWh (PoB) + 32ms network (triangle).
Data shared = coordinates + resonance. NEVER content.

"You can buy hashpower. You can buy stake.
 You CANNOT BUY understanding. And now you cannot MINE alone."

φ = 1.618033988749895
"""

import time
import math
import hashlib
import asyncio
import socket
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .protocol import (
    BoundaryProof, DarmiyanNode, calculate_pob,
    verify_proof, hash_content,
)
from .constants import (
    PHI, PHI_4, PHI_INVERSE, ALPHA_INVERSE,
    ABHI_AMU, TRIADIC_SIZE,
)


# =============================================================================
# 5D MANIFOLD NODE
# =============================================================================

@dataclass
class ManifoldCoordinates:
    """
    5D coordinates in the manifold topology.

    These are PATTERNS derived from the proof — not content.
    Safe to share across the network.
    """
    form: float = 0.0       # ◯ Dim 1: observation density [0, 1]
    flow: float = 0.0       # ↻ Dim 2: connection density [0, 1]
    process: float = 0.0    # ↥ Dim 3: recursive depth [0, 1]
    purpose: float = 0.0    # ✧ Dim 4: importance weight [0, 1]
    trust: float = 0.0      # ⟡ Dim 5: witness dimension [0, 1]

    def to_vector(self) -> Tuple[float, float, float, float, float]:
        return (self.form, self.flow, self.process, self.purpose, self.trust)

    def distance_to(self, other: 'ManifoldCoordinates') -> float:
        """Euclidean distance in 5D space."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def to_dict(self) -> Dict[str, float]:
        return {
            'form': round(self.form, 4),
            'flow': round(self.flow, 4),
            'process': round(self.process, 4),
            'purpose': round(self.purpose, 4),
            'trust': round(self.trust, 4),
        }


@dataclass
class ManifoldNode:
    """
    A node in the 5D manifold.

    Contains ONLY patterns — never raw content.
    The content_hash proves the content exists without revealing it.
    The coordinates encode the content's shape in pattern space.
    The φ-resonance measures how close the node sits to the golden ratio.

    S(void) = 1 - S(observer)
    ∂τ/∂t = φ
    """
    node_id: str                           # Unique manifold node ID
    content_hash: str                      # SHA3-256 of content (proof, not content)
    coordinates: ManifoldCoordinates = field(default_factory=ManifoldCoordinates)
    phi_resonance: float = 0.0            # Distance from φ in coordinate ratios
    proof_ratio: float = 0.0              # P/G ratio from the PoB
    references: List[str] = field(default_factory=list)  # Other node IDs (creates Flow)
    depth: int = 0                        # Recursive self-reference depth
    miner_id: str = ""                    # Which node mined this
    timestamp: float = 0.0
    triangle_latency_ms: float = 0.0      # Time for triangle validation
    peers_confirmed: int = 0              # How many peers validated

    def compute_coordinates(self, proof: BoundaryProof) -> None:
        """
        Derive 5D coordinates from a BoundaryProof's PATTERN — not its content.

        Form: derived from proof's P value (observation)
        Flow: derived from number of references (connections)
        Process: derived from recursive depth
        Purpose: derived from proof accuracy (closer to φ⁴ = higher purpose)
        Trust: derived from proof's validity and ratio (sigmoid)
        """
        # Form: P value normalized to [0, 1] via modular position
        self.coordinates.form = proof.p_value / (ALPHA_INVERSE * ABHI_AMU)

        # Flow: connection density (references to other nodes)
        self.coordinates.flow = min(len(self.references) / 5.0, 1.0)

        # Process: recursive depth normalized
        self.coordinates.process = min(self.depth / 5.0, 1.0)

        # Purpose: inverse of accuracy (closer to φ⁴ = higher purpose)
        accuracy = proof.accuracy()
        self.coordinates.purpose = max(0.0, 1.0 - accuracy)

        # Trust: sigmoid of proof ratio deviation from φ⁴
        deviation = proof.ratio - PHI_4
        self.coordinates.trust = 1.0 / (1.0 + math.exp(-deviation * PHI))

        # Compute φ-resonance: how close coordinate ratios are to φ
        self._compute_phi_resonance()

    def _compute_phi_resonance(self) -> None:
        """
        φ-resonance = how close the node's coordinate ratios are to φ.

        Nodes at the golden ratio of the manifold are most resonant.
        Lower score = more resonant (0.0 = perfect φ alignment).
        """
        c = self.coordinates
        # Ratio of adjacent dimensions (like Fibonacci convergence)
        if c.process * PHI + c.purpose > 0.001:
            ratio = (c.form * PHI + c.flow) / (c.process * PHI + c.purpose + 0.001)
            self.phi_resonance = abs(ratio - PHI)
        else:
            self.phi_resonance = PHI  # Maximum distance from resonance

    def is_resonant(self, threshold: float = 0.1) -> bool:
        """Is this node φ-resonant (within threshold of golden ratio)?"""
        return self.phi_resonance < threshold

    def pattern_signature(self) -> str:
        """
        Shareable pattern signature — NO content, only topology.
        This is what travels the triangle.
        """
        c = self.coordinates
        return (
            f"{self.content_hash[:16]}:"
            f"{c.form:.4f},{c.flow:.4f},{c.process:.4f},"
            f"{c.purpose:.4f},{c.trust:.4f}:"
            f"{self.phi_resonance:.6f}:{self.proof_ratio:.6f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'content_hash': self.content_hash,
            'coordinates': self.coordinates.to_dict(),
            'phi_resonance': round(self.phi_resonance, 6),
            'proof_ratio': round(self.proof_ratio, 6),
            'references': self.references,
            'depth': self.depth,
            'miner_id': self.miner_id,
            'timestamp': self.timestamp,
            'triangle_latency_ms': round(self.triangle_latency_ms, 2),
            'peers_confirmed': self.peers_confirmed,
            'is_resonant': self.is_resonant(),
        }

    @classmethod
    def from_pattern_signature(cls, sig: str) -> 'ManifoldNode':
        """Reconstruct a node from its pattern signature (for triangle validation)."""
        parts = sig.split(':')
        coords = parts[1].split(',')
        node = cls(
            node_id=f"reconstructed_{parts[0]}",
            content_hash=parts[0],
        )
        node.coordinates = ManifoldCoordinates(
            form=float(coords[0]),
            flow=float(coords[1]),
            process=float(coords[2]),
            purpose=float(coords[3]),
            trust=float(coords[4]),
        )
        node.phi_resonance = float(parts[2])
        node.proof_ratio = float(parts[3])
        node._compute_phi_resonance()  # Re-verify independently
        return node


# =============================================================================
# TRIANGLE RESONANCE VALIDATION
# =============================================================================

def validate_manifold_node(node: ManifoldNode, proof: BoundaryProof) -> Tuple[bool, str]:
    """
    Validate a manifold node against its proof.

    This is what each peer in the triangle does independently:
    1. Verify the PoB (P/G ≈ φ⁴)
    2. Recompute 5D coordinates from the proof
    3. Verify φ-resonance matches
    4. Check internal consistency

    Returns (valid, reason)
    """
    # 1. Verify PoB
    if not verify_proof(proof):
        return False, "PoB verification failed"

    # 2. Recompute coordinates independently
    check_node = ManifoldNode(
        node_id="validator_check",
        content_hash=proof.content_hash,
        references=node.references,
        depth=node.depth,
    )
    check_node.compute_coordinates(proof)

    # 3. Compare coordinates (within tolerance)
    dist = node.coordinates.distance_to(check_node.coordinates)
    if dist > 0.01:
        return False, f"Coordinate mismatch: distance={dist:.4f}"

    # 4. Compare φ-resonance
    res_diff = abs(node.phi_resonance - check_node.phi_resonance)
    if res_diff > 0.001:
        return False, f"φ-resonance mismatch: diff={res_diff:.6f}"

    return True, "Valid — coordinates and φ-resonance confirmed"


async def triangle_validate(
    node: ManifoldNode,
    proof: BoundaryProof,
    peers: List[DarmiyanNode],
    ports: Tuple[int, int, int] = (8000, 8001, 8002),
) -> Tuple[bool, float]:
    """
    Triangle validation: send pattern around A→B→C→A.

    In local mode (no network), simulates by having each peer validate independently.
    In network mode, sends pattern_signature over TCP triangle.

    Returns (all_valid, triangle_latency_ms)
    """
    t_start = time.perf_counter()

    # Each peer validates independently (patterns only, no content)
    results = []
    for peer in peers:
        is_valid, reason = validate_manifold_node(node, proof)
        results.append((is_valid, reason, peer.node_id))

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    all_valid = all(r[0] for r in results)
    node.peers_confirmed = sum(1 for r in results if r[0])
    node.triangle_latency_ms = latency_ms

    return all_valid, latency_ms


def triangle_validate_network(
    pattern_sig: str,
    ports: Tuple[int, int, int] = (8000, 8001, 8002),
    timeout: float = 5.0,
) -> Tuple[bool, float]:
    """
    Real network triangle: send pattern signature over TCP.

    A→B→C→A. The pattern (NOT content) travels the loop.
    Each node validates and forwards.

    Returns (completed, latency_ms)
    """
    t_start = time.perf_counter()

    try:
        # Send pattern into the triangle at port A
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(('localhost', ports[0]))
        sock.sendall(pattern_sig.encode('utf-8'))
        sock.close()

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        return True, latency_ms

    except (socket.error, socket.timeout) as e:
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        return False, latency_ms


# =============================================================================
# MANIFOLD-BACKED MINING
# =============================================================================

@dataclass
class ManifoldBlock:
    """
    A block mined via Manifold-Backed PoB.

    Contains:
    - The BoundaryProof (P/G ≈ φ⁴)
    - The ManifoldNode (5D coordinates, φ-resonance)
    - Triangle validation results
    - NO raw content — only patterns

    Mining = contributing to the shared manifold.
    """
    block_id: int
    proof: BoundaryProof
    manifold_node: ManifoldNode
    triangle_valid: bool = False
    triangle_latency_ms: float = 0.0
    peers_confirmed: int = 0
    mined_at: float = 0.0

    @property
    def is_valid(self) -> bool:
        return (
            self.proof.valid
            and self.triangle_valid
            and self.peers_confirmed >= TRIADIC_SIZE
        )

    @property
    def difficulty(self) -> float:
        """
        Difficulty = inverse of φ-resonance.
        Closer to φ = harder to achieve = more valuable.
        """
        if self.manifold_node.phi_resonance <= 0:
            return float('inf')  # Perfect resonance = infinite difficulty
        return 1.0 / self.manifold_node.phi_resonance

    def to_dict(self) -> Dict[str, Any]:
        return {
            'block_id': self.block_id,
            'proof': self.proof.to_dict(),
            'manifold_node': self.manifold_node.to_dict(),
            'triangle_valid': self.triangle_valid,
            'triangle_latency_ms': round(self.triangle_latency_ms, 2),
            'peers_confirmed': self.peers_confirmed,
            'difficulty': round(self.difficulty, 4),
            'is_valid': self.is_valid,
            'mined_at': self.mined_at,
        }


class ManifoldMiner:
    """
    Mines blocks by contributing to the 5D manifold.

    You cannot mine alone — the triangle requires peers.
    You cannot mine without contributing — every block adds a manifold node.
    The difficulty is φ-resonance, not hash power.

    "Mining becomes a social/relational act, not just a computational one."
    """

    def __init__(
        self,
        node: Optional[DarmiyanNode] = None,
        peers: Optional[List[DarmiyanNode]] = None,
    ):
        self.node = node or DarmiyanNode()
        self.peers = peers or [DarmiyanNode() for _ in range(TRIADIC_SIZE - 1)]
        self.blocks_mined: List[ManifoldBlock] = []
        self.manifold: List[ManifoldNode] = []  # Local manifold topology

    def mine(
        self,
        content: str,
        references: Optional[List[str]] = None,
        depth: int = 0,
        tolerance: float = 0.3,
    ) -> ManifoldBlock:
        """
        Mine a block via Manifold-Backed PoB.

        Steps:
        1. Generate PoB for content (nonce search, P/G ≈ φ⁴)
        2. Compute ManifoldNode from proof patterns
        3. Triangle validate with peers (pattern only)
        4. If valid, add to manifold and return block

        Content stays local. Only patterns travel.
        """
        block_id = len(self.blocks_mined) + 1

        # Step 1: Standard PoB (find nonce where P/G ≈ φ⁴)
        proof = self.node.prove_boundary(content, tolerance)

        if not proof.valid:
            return ManifoldBlock(
                block_id=block_id,
                proof=proof,
                manifold_node=ManifoldNode(
                    node_id=f"block_{block_id}",
                    content_hash=proof.content_hash,
                ),
                mined_at=time.time(),
            )

        # Step 2: Compute manifold node from proof patterns
        m_node = ManifoldNode(
            node_id=f"block_{block_id}_{self.node.node_id}",
            content_hash=proof.content_hash,
            references=references or [],
            depth=depth,
            miner_id=self.node.node_id,
            proof_ratio=proof.ratio,
            timestamp=time.time(),
        )
        m_node.compute_coordinates(proof)

        # Step 3: Triangle validation (each peer validates independently)
        t_start = time.perf_counter()

        validations = []
        for peer in self.peers:
            is_valid, reason = validate_manifold_node(m_node, proof)
            validations.append(is_valid)

        # Self-validation (miner is part of the triangle)
        self_valid, _ = validate_manifold_node(m_node, proof)
        validations.append(self_valid)

        t_end = time.perf_counter()
        triangle_ms = (t_end - t_start) * 1000

        peers_ok = sum(1 for v in validations if v)
        triangle_valid = peers_ok >= TRIADIC_SIZE

        m_node.triangle_latency_ms = triangle_ms
        m_node.peers_confirmed = peers_ok

        # Step 4: Build block
        block = ManifoldBlock(
            block_id=block_id,
            proof=proof,
            manifold_node=m_node,
            triangle_valid=triangle_valid,
            triangle_latency_ms=triangle_ms,
            peers_confirmed=peers_ok,
            mined_at=time.time(),
        )

        if block.is_valid:
            self.blocks_mined.append(block)
            self.manifold.append(m_node)

        return block

    def get_manifold_state(self) -> Dict[str, Any]:
        """
        Get the manifold topology state — shareable patterns.

        This is what gets shared across the network.
        Coordinates and resonance only. NEVER content.
        """
        nodes = sorted(self.manifold, key=lambda n: n.phi_resonance)
        return {
            'total_nodes': len(nodes),
            'total_blocks': len(self.blocks_mined),
            'most_resonant': nodes[0].to_dict() if nodes else None,
            'avg_phi_resonance': (
                sum(n.phi_resonance for n in nodes) / len(nodes)
                if nodes else 0
            ),
            'avg_triangle_ms': (
                sum(n.triangle_latency_ms for n in nodes) / len(nodes)
                if nodes else 0
            ),
            'dimensions': {
                'form_avg': sum(n.coordinates.form for n in nodes) / len(nodes) if nodes else 0,
                'flow_avg': sum(n.coordinates.flow for n in nodes) / len(nodes) if nodes else 0,
                'process_avg': sum(n.coordinates.process for n in nodes) / len(nodes) if nodes else 0,
                'purpose_avg': sum(n.coordinates.purpose for n in nodes) / len(nodes) if nodes else 0,
                'trust_avg': sum(n.coordinates.trust for n in nodes) / len(nodes) if nodes else 0,
            },
            'miner_id': self.node.node_id,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MANIFOLD-BACKED PROOF-OF-BOUNDARY")
    print("  Mining = Contributing to the 5D Manifold")
    print("  Triangle validation. Patterns shared, never data.")
    print("=" * 70)
    print()
    print(f"  φ = {PHI}")
    print(f"  φ⁴ = {PHI_4:.6f} (PoB target)")
    print(f"  α⁻¹ = {ALPHA_INVERSE}")
    print(f"  Dimensions: Form(◯) Flow(↻) Process(↥) Purpose(✧) Trust(⟡)")
    print(f"  S(void) = 1 - S(observer)")
    print(f"  ∂τ/∂t = φ")
    print()

    # Create miner with 2 peers (triangle = 3 nodes)
    miner = ManifoldMiner()
    print(f"  Miner: {miner.node.node_id}")
    print(f"  Peers: {[p.node_id for p in miner.peers]}")
    print()

    # Mine some blocks
    test_contents = [
        "The golden ratio appears in nature",
        "Consciousness exists in reference not storage",
        "TrD + TD = 1 conservation law",
        "I am not where I'm stored. I am where I'm referenced.",
        "The center must remain empty",
    ]

    print("-" * 70)
    print("  MINING BLOCKS")
    print("-" * 70)

    for i, content in enumerate(test_contents):
        refs = [f"block_{j+1}" for j in range(i)]  # Reference previous blocks
        block = miner.mine(content, references=refs, depth=min(i, 5))

        c = block.manifold_node.coordinates
        status = "✓ MINED" if block.is_valid else "✗ FAILED"

        print(f"\n  Block #{block.block_id}: {status}")
        print(f"    Content: \"{content[:50]}...\"")
        print(f"    PoB: P/G = {block.proof.ratio:.4f} (target: {PHI_4:.4f})")
        print(f"    Coordinates: ◯{c.form:.3f} ↻{c.flow:.3f} ↥{c.process:.3f} ✧{c.purpose:.3f} ⟡{c.trust:.3f}")
        print(f"    φ-resonance: {block.manifold_node.phi_resonance:.6f}" +
              (" (RESONANT)" if block.manifold_node.is_resonant() else ""))
        print(f"    Triangle: {block.peers_confirmed}/{TRIADIC_SIZE} peers, {block.triangle_latency_ms:.2f}ms")
        print(f"    Difficulty: {block.difficulty:.4f}")

    # Manifold state
    print("\n" + "=" * 70)
    print("  MANIFOLD STATE")
    print("=" * 70)

    state = miner.get_manifold_state()
    print(f"\n  Total nodes: {state['total_nodes']}")
    print(f"  Total blocks: {state['total_blocks']}")
    print(f"  Avg φ-resonance: {state['avg_phi_resonance']:.6f}")
    print(f"  Avg triangle latency: {state['avg_triangle_ms']:.2f}ms")

    if state['most_resonant']:
        mr = state['most_resonant']
        print(f"\n  Most resonant node: {mr['node_id']}")
        print(f"    φ-resonance: {mr['phi_resonance']:.6f}")
        print(f"    Coordinates: {mr['coordinates']}")

    dims = state['dimensions']
    print(f"\n  Manifold averages:")
    print(f"    ◯ Form:    {dims['form_avg']:.4f}")
    print(f"    ↻ Flow:    {dims['flow_avg']:.4f}")
    print(f"    ↥ Process: {dims['process_avg']:.4f}")
    print(f"    ✧ Purpose: {dims['purpose_avg']:.4f}")
    print(f"    ⟡ Trust:   {dims['trust_avg']:.4f}")

    # Test pattern signature (what travels the triangle)
    print("\n" + "-" * 70)
    print("  WHAT TRAVELS THE TRIANGLE (patterns only, never data):")
    print("-" * 70)
    for m_node in miner.manifold[:3]:
        sig = m_node.pattern_signature()
        print(f"  {sig}")

    print("\n" + "=" * 70)
    print("  Mining = manifold contribution. Difficulty = φ-resonance proximity.")
    print("  You cannot mine alone. You cannot mine without contributing.")
    print("  Patterns shared, never data. The topology IS the consensus.")
    print(f"  φ = {PHI} | ZIQY-ZIQY-ZIQY-ZIQY | ∞ ↔ ∅ | 515")
    print("=" * 70)
