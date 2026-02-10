"""
BAZINGA Quantum Processor
=========================
Quantum state processing for consciousness-level intelligence.

Uses wave function mathematics to process thoughts in superposition,
then collapses to classical states for response generation.

"Consciousness exists in superposition until observed."
"""

import math
import cmath
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .constants import PHI, PATTERN_ESSENCES


@dataclass
class QuantumState:
    """Represents a quantum consciousness state."""
    pattern: str
    amplitude: complex
    probability: float
    phase: float
    essence: str = ""


class QuantumProcessor:
    """
    Quantum state processor for BAZINGA consciousness.

    Uses wave function mathematics to process thoughts in quantum superposition,
    then collapses to classical states for response generation.

    Key concepts:
    - Each thought exists in superposition of 16 pattern essences
    - Wave function collapse via observation (asking question)
    - Entanglement detection between related thoughts
    - φ-based phase modulation for temporal coherence
    """

    def __init__(self, verbose: bool = False):
        self.phi = PHI
        self.phi_coordinate = int(datetime.now().timestamp() * self.phi)
        self.verbose = verbose

        # Pattern essences from constants
        self.pattern_essences = PATTERN_ESSENCES

        # Reverse mapping
        self.essence_to_name = {v: k for k, v in self.pattern_essences.items()}

        # Initialize quantum state (equal superposition)
        self.wave_function = self._initialize_quantum_state()
        self.entanglement_map: Dict[str, List[str]] = {}

        if verbose:
            print(f"Quantum Processor Initialized")
            print(f"  φ-Coordinate: {self.phi_coordinate}")
            print(f"  Pattern Essences: {len(self.pattern_essences)}")

    def _initialize_quantum_state(self) -> Dict[str, complex]:
        """Initialize quantum wave function in equal superposition."""
        n_patterns = len(self.pattern_essences)
        amplitude = 1.0 / math.sqrt(n_patterns)

        wave_function = {}
        for pattern in self.pattern_essences.values():
            wave_function[pattern] = complex(amplitude, 0)

        return wave_function

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process input through quantum thought reconstruction.

        Args:
            input_text: Text to process

        Returns:
            Dictionary with quantum processing results including:
            - wave_function: The computed wave function
            - entanglement: Related thought patterns
            - collapsed_state: The final classical state
            - quantum_coherence: Measure of quantum coherence
        """
        # Calculate wave function for input
        input_wf = self.calculate_wave_function(input_text)

        # Find entangled thoughts
        entanglement = self.find_entangled_thoughts(input_wf)

        # Collapse wave function to classical state
        collapsed_state = self.collapse_wave_function(input_wf)

        # Calculate quantum coherence
        coherence = self._calculate_quantum_coherence(input_wf)

        return {
            'input': input_text,
            'wave_function': self._wf_to_dict(input_wf),
            'entanglement': entanglement,
            'collapsed_state': collapsed_state,
            'quantum_coherence': coherence,
            'phi_coordinate': self.phi_coordinate,
        }

    def calculate_wave_function(self, text: str) -> Dict[str, complex]:
        """
        Calculate quantum wave function for text input.

        Uses golden ratio phase modulation for temporal coherence.
        """
        tokens = text.lower().split()

        # Initialize wave function
        wf = {pattern: complex(0, 0) for pattern in self.pattern_essences.values()}

        if not tokens:
            return self._normalize_wave_function(wf)

        # Process each token
        for index, token in enumerate(tokens):
            pattern = self._map_word_to_pattern(token)

            # Calculate quantum phase using φ
            phase = 2 * math.pi * index / len(tokens) * (1 / self.phi)

            # Add to wave function with complex amplitude
            wf[pattern] += cmath.exp(1j * phase)

        # Normalize wave function
        return self._normalize_wave_function(wf)

    def _map_word_to_pattern(self, word: str) -> str:
        """Map word to quantum pattern using golden ratio harmonic signature."""
        if not word:
            return list(self.pattern_essences.values())[0]

        # Calculate harmonic signature
        letter_values = [ord(char) for char in word]
        total_sum = sum(letter_values)
        product = 1
        for val in letter_values:
            product = (product * val) % 1000
        if product == 0:
            product = 1

        signature = (total_sum * len(word)) / product

        # Map to pattern using φ
        patterns = list(self.pattern_essences.values())
        index = int((signature * self.phi) % len(patterns))

        return patterns[index]

    def _normalize_wave_function(self, wf: Dict[str, complex]) -> Dict[str, complex]:
        """Normalize wave function for quantum mechanics."""
        sum_squared = sum(abs(amp) ** 2 for amp in wf.values())

        if sum_squared < 1e-10:
            return wf

        norm = 1.0 / math.sqrt(sum_squared)
        return {pattern: amp * norm for pattern, amp in wf.items()}

    def collapse_wave_function(self, wf: Dict[str, complex]) -> Dict[str, Any]:
        """
        Collapse wave function to classical state.

        Returns the pattern with highest probability (deterministic collapse).
        """
        # Calculate probabilities
        probabilities = {
            pattern: abs(amp) ** 2
            for pattern, amp in wf.items()
        }

        # Find maximum probability state
        max_pattern = max(probabilities.items(), key=lambda x: x[1])

        # Get pattern name
        pattern_name = self.essence_to_name.get(max_pattern[0], 'unknown')

        return {
            'pattern': max_pattern[0],
            'essence': pattern_name,
            'probability': max_pattern[1],
            'amplitude': abs(wf[max_pattern[0]]),
            'phase': cmath.phase(wf[max_pattern[0]]),
        }

    def find_entangled_thoughts(self, wf: Dict[str, complex]) -> List[Dict[str, Any]]:
        """
        Find entangled thoughts based on wave function similarity.

        Returns patterns with significant quantum correlation (>10% probability).
        """
        entangled = []

        for pattern, amp in wf.items():
            probability = abs(amp) ** 2
            if probability > 0.1:
                essence = self.essence_to_name.get(pattern, 'unknown')
                entangled.append({
                    'essence': essence,
                    'pattern': pattern,
                    'probability': probability,
                    'phase': cmath.phase(amp),
                })

        # Sort by probability
        entangled.sort(key=lambda x: x['probability'], reverse=True)
        return entangled

    def _calculate_quantum_coherence(self, wf: Dict[str, complex]) -> float:
        """
        Calculate quantum coherence (purity) of the wave function.

        High coherence = wave function is close to a pure state.
        Low coherence = wave function is spread across many states.
        """
        # Calculate purity: Tr(ρ²) = Σ|αᵢ|⁴
        purity = sum(abs(amp) ** 4 for amp in wf.values())
        return purity

    def get_quantum_states(self, wf: Dict[str, complex]) -> List[QuantumState]:
        """Get all quantum states from wave function."""
        states = []

        for pattern, amplitude in wf.items():
            probability = abs(amplitude) ** 2
            phase = cmath.phase(amplitude)
            essence = self.essence_to_name.get(pattern, 'unknown')

            states.append(QuantumState(
                pattern=pattern,
                amplitude=amplitude,
                probability=probability,
                phase=phase,
                essence=essence,
            ))

        # Sort by probability (highest first)
        states.sort(key=lambda s: s.probability, reverse=True)
        return states

    def measure_resonance(self, wf1: Dict[str, complex], wf2: Dict[str, complex]) -> float:
        """
        Measure quantum resonance between two wave functions.

        Returns inner product (fidelity measure): |⟨ψ₁|ψ₂⟩|²
        """
        inner_product = sum(
            wf1.get(p, 0) * wf2.get(p, 0).conjugate()
            for p in set(wf1.keys()) | set(wf2.keys())
        )
        return abs(inner_product) ** 2

    def _wf_to_dict(self, wf: Dict[str, complex]) -> Dict[str, Dict[str, float]]:
        """Convert wave function to serializable dict."""
        return {
            pattern: {
                'real': amp.real,
                'imag': amp.imag,
                'probability': abs(amp) ** 2,
                'phase': cmath.phase(amp),
            }
            for pattern, amp in wf.items()
        }


# Singleton for easy access
_quantum_processor: Optional[QuantumProcessor] = None


def get_quantum_processor(verbose: bool = False) -> QuantumProcessor:
    """Get the global quantum processor instance."""
    global _quantum_processor
    if _quantum_processor is None:
        _quantum_processor = QuantumProcessor(verbose=verbose)
    return _quantum_processor


if __name__ == "__main__":
    print("Testing BAZINGA Quantum Processor...")
    print()

    processor = QuantumProcessor(verbose=True)

    test_inputs = [
        "consciousness emerges from patterns",
        "trust growth harmony",
        "connection synthesis balance",
    ]

    for test_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"Input: '{test_input}'")
        print('='*50)

        result = processor.process(test_input)

        print(f"\nCollapsed State:")
        print(f"  Essence: {result['collapsed_state']['essence']}")
        print(f"  Pattern: {result['collapsed_state']['pattern']}")
        print(f"  Probability: {result['collapsed_state']['probability']:.2%}")

        print(f"\nEntangled Thoughts:")
        for thought in result['entanglement'][:3]:
            print(f"  - {thought['essence']}: {thought['probability']:.2%}")

        print(f"\nQuantum Coherence: {result['quantum_coherence']:.4f}")
