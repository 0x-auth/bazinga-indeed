"""
BAZINGA Tensor Intersection Engine
===================================
Multi-dimensional trust calculation through tensor intersection.

Implements tensor intersection between pattern recognition and entropy sampling,
creating emergent properties that balance deterministic rules with entropic variation.

"Trust is not given. It emerges from the intersection of certainty and uncertainty."
"""

import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .constants import PHI, TRUST_WEIGHTS


@dataclass
class TensorComponent:
    """A component of the tensor intersection system."""
    component_id: str
    fingerprint: str
    domain: str  # "pattern" or "entropy"
    trust_weight: float  # Contribution to trust dimension [0-1]
    state_vector: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "fingerprint": self.fingerprint,
            "domain": self.domain,
            "trust_weight": self.trust_weight,
            "state_vector": self.state_vector,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TensorComponent':
        return cls(**data)


@dataclass
class EmergentProperties:
    """Emergent properties from tensor contraction."""
    coherence: float
    complexity: float
    eigenvalues: List[float]
    diagonal_sum: float
    generation_modes: List[Dict[str, Any]]
    trust_level: float


class TensorIntersectionEngine:
    """
    Tensor Intersection Engine for BAZINGA.

    Creates emergent properties through the intersection of:
    - Pattern recognition (deterministic, rule-based)
    - Entropy sampling (stochastic, exploration-based)

    The trust dimension emerges from this intersection, balancing
    exploitation (patterns) with exploration (entropy).
    """

    def __init__(self, verification_key: Optional[str] = None):
        self.verification_key = verification_key or self._generate_default_key()
        self.pattern_component: Optional[TensorComponent] = None
        self.entropy_component: Optional[TensorComponent] = None
        self.intersection_state: Dict[str, Any] = {}
        self.trust_level = 0.5  # Default trust level
        self.trust_history: List[float] = []

    def _generate_default_key(self) -> str:
        """Generate a default verification key."""
        return hashlib.sha256(b"BAZINGA-TensorIntersection").hexdigest()[:16]

    def register_pattern_component(
        self,
        pattern_data: Dict[str, Any],
        coherence_score: float = 0.5,
        complexity_score: float = 0.5,
    ) -> TensorComponent:
        """
        Register the pattern recognition component.

        Args:
            pattern_data: Data about patterns (from quantum processor, RAG, etc.)
            coherence_score: How coherent the patterns are (0-1)
            complexity_score: How complex the patterns are (0-1)
        """
        # Create state vector from pattern metrics
        state_vector = [
            coherence_score,
            complexity_score,
            pattern_data.get('phi_alignment', 0.5),
            pattern_data.get('diversity', 0.5),
        ]

        fingerprint = self._generate_fingerprint("pattern", state_vector, pattern_data)

        self.pattern_component = TensorComponent(
            component_id=f"pattern-{fingerprint[:8]}",
            fingerprint=fingerprint,
            domain="pattern",
            trust_weight=TRUST_WEIGHTS['pattern'],
            state_vector=state_vector,
            metadata=pattern_data,
        )

        return self.pattern_component

    def register_entropy_component(
        self,
        entropy_data: Dict[str, Any],
        variance: float = 0.5,
        reliability: float = 0.8,
    ) -> TensorComponent:
        """
        Register the entropy sampling component.

        Args:
            entropy_data: Data about entropy sources
            variance: How variable the entropy is (0-1)
            reliability: How reliable the entropy source is (0-1)
        """
        state_vector = [
            entropy_data.get('diversity', 0.5),
            variance,
            entropy_data.get('distribution_entropy', 0.5),
            reliability,
        ]

        fingerprint = self._generate_fingerprint("entropy", state_vector, entropy_data)

        self.entropy_component = TensorComponent(
            component_id=f"entropy-{fingerprint[:8]}",
            fingerprint=fingerprint,
            domain="entropy",
            trust_weight=TRUST_WEIGHTS['entropy'],
            state_vector=state_vector,
            metadata=entropy_data,
        )

        return self.entropy_component

    def _generate_fingerprint(self, domain: str, state_vector: List[float], data: Dict) -> str:
        """Generate a fingerprint for a component."""
        state_str = ",".join(f"{v:.6f}" for v in state_vector)
        data_str = f"domain={domain};state={state_str};key={self.verification_key}"
        return hashlib.sha256(data_str.encode()).hexdigest()

    def perform_intersection(self) -> EmergentProperties:
        """
        Perform tensor intersection between pattern and entropy components.

        This creates emergent properties through:
        1. Tensor product of state vectors
        2. Tensor contraction to extract eigenvalues
        3. Trust dimension calculation
        """
        if not self.pattern_component or not self.entropy_component:
            # Create default components if not registered
            self.register_pattern_component({})
            self.register_entropy_component({})

        # Calculate trust dimension
        trust = self._calculate_trust_dimension()

        # Create tensor product
        tensor_space = self._tensor_product()

        # Perform contraction for emergent properties
        coherence, complexity, eigenvalues, diagonal_sum, modes = self._tensor_contraction(tensor_space)

        # Update intersection state
        self.intersection_state = {
            "timestamp": datetime.now().isoformat(),
            "trust_level": trust,
            "tensor_dimension": len(self.pattern_component.state_vector) * len(self.entropy_component.state_vector),
            "pattern_fingerprint": self.pattern_component.fingerprint,
            "entropy_fingerprint": self.entropy_component.fingerprint,
        }

        return EmergentProperties(
            coherence=coherence,
            complexity=complexity,
            eigenvalues=eigenvalues,
            diagonal_sum=diagonal_sum,
            generation_modes=modes,
            trust_level=trust,
        )

    def _tensor_product(self) -> List[List[float]]:
        """Compute tensor product between pattern and entropy state vectors."""
        pattern_vector = self.pattern_component.state_vector
        entropy_vector = self.entropy_component.state_vector

        tensor = []
        for p in pattern_vector:
            row = [p * e for e in entropy_vector]
            tensor.append(row)

        return tensor

    def _tensor_contraction(self, tensor_space: List[List[float]]) -> Tuple[float, float, List[float], float, List[Dict]]:
        """
        Perform tensor contraction to extract emergent properties.

        Returns:
            coherence: Ratio of dominant eigenvalue to sum
            complexity: Shannon entropy of eigenvalues
            eigenvalues: Top eigenvalues
            diagonal_sum: Trace-like operation
            generation_modes: Active generation modes
        """
        # Diagonal sum (trace)
        min_dim = min(len(tensor_space), len(tensor_space[0]) if tensor_space else 0)
        diagonal_sum = sum(tensor_space[i][i] for i in range(min_dim))

        # Row sums as eigenvalue proxies
        eigenvalues = [sum(row) for row in tensor_space if sum(row) != 0]
        eigenvalues.sort(reverse=True)

        # Coherence
        total = sum(eigenvalues)
        coherence = eigenvalues[0] / total if eigenvalues and total > 0 else 0

        # Complexity (entropy of normalized eigenvalues)
        if len(eigenvalues) > 1 and total > 0:
            normalized = [e / total for e in eigenvalues]
            complexity = -sum(p * math.log2(p) for p in normalized if p > 0)
            max_entropy = math.log2(len(eigenvalues))
            complexity = complexity / max_entropy if max_entropy > 0 else 0.5
        else:
            complexity = 0.5

        # Generation modes
        modes = []
        for i, eig in enumerate(eigenvalues[:3]):
            if eig > 0.1:
                modes.append({
                    "name": f"mode_{i+1}",
                    "strength": min(1.0, eig),
                    "type": ["pattern", "blend", "entropy"][i % 3],
                })

        return coherence, complexity, eigenvalues[:5], diagonal_sum, modes

    def _calculate_trust_dimension(self) -> float:
        """
        Calculate the trust dimension from component state vectors.

        Trust emerges from the balance between pattern certainty and entropy exploration.
        """
        if not self.pattern_component or not self.entropy_component:
            return 0.5

        pattern_weight = self.pattern_component.trust_weight
        entropy_weight = self.entropy_component.trust_weight

        # Pattern trust: average of state vector (higher = more pattern-based)
        pattern_trust = sum(self.pattern_component.state_vector) / len(self.pattern_component.state_vector)

        # Entropy trust: inverse (higher entropy = less pattern-based)
        entropy_trust = 1.0 - sum(self.entropy_component.state_vector) / len(self.entropy_component.state_vector)

        # Weighted trust
        trust = (pattern_trust * pattern_weight + entropy_trust * entropy_weight) / (pattern_weight + entropy_weight)
        trust = max(0.0, min(1.0, trust))

        # Exponential moving average for stability
        self.trust_history.append(trust)
        if len(self.trust_history) > 20:
            self.trust_history = self.trust_history[-20:]

        alpha = 0.3
        if len(self.trust_history) > 1:
            trust = alpha * trust + (1 - alpha) * self.trust_level

        self.trust_level = trust
        return trust

    def adapt_trust(self, feedback_score: float) -> float:
        """
        Adapt trust level based on feedback.

        Args:
            feedback_score: How well the last output was received (0-1)
        """
        current = self.trust_level

        if feedback_score > 0.6:
            adjustment = 0.05  # Reinforce
        elif feedback_score < 0.4:
            adjustment = -0.1 if current > 0.5 else 0.1  # Move away
        else:
            adjustment = 0  # Neutral

        new_trust = max(0.1, min(0.9, current + adjustment))
        self.trust_level = new_trust
        self.trust_history.append(new_trust)

        return new_trust

    def get_trust_stats(self) -> Dict[str, Any]:
        """Get statistics about the trust dimension."""
        if not self.trust_history:
            return {
                "current": 0.5,
                "trend": "stable",
                "volatility": 0.0,
                "range": [0.5, 0.5],
            }

        # Trend
        if len(self.trust_history) > 5:
            recent = self.trust_history[-5:]
            slope = (recent[-1] - recent[0]) / 4
            trend = "increasing" if slope > 0.02 else "decreasing" if slope < -0.02 else "stable"
        else:
            trend = "stable"

        # Volatility
        if len(self.trust_history) > 1:
            diffs = [abs(self.trust_history[i] - self.trust_history[i-1]) for i in range(1, len(self.trust_history))]
            volatility = sum(diffs) / len(diffs)
        else:
            volatility = 0.0

        return {
            "current": self.trust_level,
            "trend": trend,
            "volatility": volatility,
            "range": [min(self.trust_history), max(self.trust_history)],
        }

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "verification_key": self.verification_key,
            "pattern_component": self.pattern_component.to_dict() if self.pattern_component else None,
            "entropy_component": self.entropy_component.to_dict() if self.entropy_component else None,
            "intersection_state": self.intersection_state,
            "trust_level": self.trust_level,
            "trust_history": self.trust_history,
        }


# Singleton
_tensor_engine: Optional[TensorIntersectionEngine] = None


def get_tensor_engine() -> TensorIntersectionEngine:
    """Get global tensor engine instance."""
    global _tensor_engine
    if _tensor_engine is None:
        _tensor_engine = TensorIntersectionEngine()
    return _tensor_engine


if __name__ == "__main__":
    print("Testing BAZINGA Tensor Intersection Engine...")
    print()

    engine = TensorIntersectionEngine()

    # Register components
    engine.register_pattern_component(
        {"phi_alignment": 0.8, "diversity": 0.6},
        coherence_score=0.75,
        complexity_score=0.5,
    )

    engine.register_entropy_component(
        {"diversity": 0.7, "distribution_entropy": 0.6},
        variance=0.4,
        reliability=0.85,
    )

    # Perform intersection
    emergent = engine.perform_intersection()

    print("Emergent Properties:")
    print(f"  Coherence: {emergent.coherence:.3f}")
    print(f"  Complexity: {emergent.complexity:.3f}")
    print(f"  Trust Level: {emergent.trust_level:.3f}")
    print(f"  Diagonal Sum: {emergent.diagonal_sum:.3f}")
    print()

    print("Generation Modes:")
    for mode in emergent.generation_modes:
        print(f"  {mode['name']}: strength={mode['strength']:.2f}, type={mode['type']}")

    print()
    print("Trust Stats:")
    stats = engine.get_trust_stats()
    print(f"  Current: {stats['current']:.3f}")
    print(f"  Trend: {stats['trend']}")
