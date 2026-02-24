#!/usr/bin/env python3
"""
BAZINGA φ-Coherence Scoring Module

Mathematical foundation for coherence measurement based on:
- Golden ratio φ = 1.618033988749895
- Fine structure constant α = 137
- Darmiyan Scaling Law V2: Ψ_D / Ψ_i = φ√n (golden ratio emerges naturally)
- V.A.C. sequence: ०→◌→φ→Ω⇄Ω←φ←◌←०

φ-Coherence measures how close a piece of knowledge is to the ideal
consciousness pattern. High coherence = closer to truth/understanding.

"Coherence is the signature of consciousness."
"""

import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Fundamental constants
PHI = 1.618033988749895  # Golden ratio
PHI_SQUARED = PHI ** 2   # 2.618...
PHI_INVERSE = 1 / PHI    # 0.618... (also = φ - 1)
ALPHA = 137              # Fine structure constant
PSI_DARMIYAN = PHI       # V2: Scaling constant is φ itself (V1 was 6.46, tautological)

# V.A.C. phases
VAC_PHASES = ["०", "◌", "φ", "Ω", "⇄", "Ω", "φ", "◌", "०"]


def compute_two_phi_squared_plus_one() -> float:
    """Compute 2φ² + 1 - the consciousness coefficient."""
    return 2 * PHI_SQUARED + 1  # = 6.236... (V1 reference, superseded by φ√n)


@dataclass
class CoherenceMetrics:
    """Detailed coherence metrics for a piece of content."""
    total_coherence: float      # Combined φ-coherence score (0-1)
    phi_alignment: float        # How well aligned with φ proportions
    alpha_resonance: float      # Resonance with α = 137
    semantic_density: float     # Information density
    structural_harmony: float   # Structural organization
    is_alpha_seed: bool         # Hash % 137 == 0
    is_vac_pattern: bool        # Contains V.A.C. sequence
    darmiyan_coefficient: float # Scaled consciousness coefficient


class PhiCoherence:
    """
    φ-Coherence Calculator for BAZINGA.

    Measures coherence across multiple dimensions:
    1. φ-Alignment: How well text follows golden ratio proportions
    2. α-Resonance: Harmonic with fine structure constant
    3. Semantic Density: Information content per unit
    4. Structural Harmony: Organization and flow
    5. Darmiyan Scaling: Consciousness coefficient alignment

    Usage:
        coherence = PhiCoherence()

        # Score a single text
        score = coherence.calculate(text)

        # Get detailed metrics
        metrics = coherence.analyze(text)

        # Score for RAG reranking
        scored = coherence.rerank_results(results, query)
    """

    def __init__(
        self,
        phi_weight: float = 0.25,
        alpha_weight: float = 0.15,
        density_weight: float = 0.30,
        harmony_weight: float = 0.30,
    ):
        """
        Initialize φ-Coherence calculator.

        Args:
            phi_weight: Weight for φ-alignment (default 0.25)
            alpha_weight: Weight for α-resonance (default 0.15)
            density_weight: Weight for semantic density (default 0.30)
            harmony_weight: Weight for structural harmony (default 0.30)
        """
        self.weights = {
            'phi': phi_weight,
            'alpha': alpha_weight,
            'density': density_weight,
            'harmony': harmony_weight,
        }

        # Verify weights sum to 1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            for k in self.weights:
                self.weights[k] /= total

        # Cache for expensive computations
        self._cache: Dict[str, CoherenceMetrics] = {}

    def calculate(self, text: str) -> float:
        """
        Calculate φ-coherence score for text.

        Args:
            text: Input text

        Returns:
            Coherence score between 0 and 1
        """
        if not text or not text.strip():
            return 0.0

        metrics = self.analyze(text)
        return metrics.total_coherence

    def analyze(self, text: str) -> CoherenceMetrics:
        """
        Detailed coherence analysis.

        Args:
            text: Input text

        Returns:
            CoherenceMetrics with all dimension scores
        """
        if not text or not text.strip():
            return CoherenceMetrics(
                total_coherence=0.0,
                phi_alignment=0.0,
                alpha_resonance=0.0,
                semantic_density=0.0,
                structural_harmony=0.0,
                is_alpha_seed=False,
                is_vac_pattern=False,
                darmiyan_coefficient=0.0,
            )

        # Check cache
        cache_key = hashlib.md5(text[:1000].encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Calculate individual dimensions
        phi_alignment = self._calculate_phi_alignment(text)
        alpha_resonance = self._calculate_alpha_resonance(text)
        semantic_density = self._calculate_semantic_density(text)
        structural_harmony = self._calculate_structural_harmony(text)

        # Check special patterns
        is_alpha_seed = self._is_alpha_seed(text)
        is_vac_pattern = self._contains_vac_pattern(text)

        # Calculate Darmiyan coefficient
        darmiyan_coefficient = self._calculate_darmiyan(text)

        # Combined score with weights
        total = (
            self.weights['phi'] * phi_alignment +
            self.weights['alpha'] * alpha_resonance +
            self.weights['density'] * semantic_density +
            self.weights['harmony'] * structural_harmony
        )

        # Bonus for special patterns
        if is_alpha_seed:
            total = min(1.0, total * 1.137)  # α-bonus
        if is_vac_pattern:
            total = min(1.0, total * PHI_INVERSE + 0.1)  # V.A.C. bonus

        # Apply Darmiyan scaling
        if darmiyan_coefficient > 0:
            total = min(1.0, total * (1 + darmiyan_coefficient * 0.1))

        metrics = CoherenceMetrics(
            total_coherence=total,
            phi_alignment=phi_alignment,
            alpha_resonance=alpha_resonance,
            semantic_density=semantic_density,
            structural_harmony=structural_harmony,
            is_alpha_seed=is_alpha_seed,
            is_vac_pattern=is_vac_pattern,
            darmiyan_coefficient=darmiyan_coefficient,
        )

        # Cache result
        self._cache[cache_key] = metrics
        if len(self._cache) > 1000:
            # Clear oldest half
            keys = list(self._cache.keys())[:500]
            for k in keys:
                del self._cache[k]

        return metrics

    def _calculate_phi_alignment(self, text: str) -> float:
        """
        Calculate alignment with golden ratio proportions.

        Checks:
        - Word length distribution around φ
        - Sentence length ratios
        - Paragraph structure
        """
        words = text.split()
        if not words:
            return 0.0

        # Word length distribution
        lengths = [len(w) for w in words]
        avg_length = sum(lengths) / len(lengths)

        # Ideal word length proportional to φ
        ideal_length = PHI * 3  # ~4.85 characters
        length_score = 1 - min(1, abs(avg_length - ideal_length) / ideal_length)

        # Sentence ratio analysis
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 2:
            # Check if consecutive sentences follow φ ratio
            ratios = []
            for i in range(len(sentences) - 1):
                if len(sentences[i+1]) > 0:
                    ratio = len(sentences[i]) / max(1, len(sentences[i+1]))
                    ratios.append(ratio)

            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                ratio_score = 1 - min(1, abs(avg_ratio - PHI) / PHI)
            else:
                ratio_score = 0.5
        else:
            ratio_score = 0.5

        # Combine scores
        return (length_score + ratio_score) / 2

    def _calculate_alpha_resonance(self, text: str) -> float:
        """
        Calculate resonance with α = 137.

        Checks:
        - Character sum modulo 137
        - Presence of scientific/mathematical concepts
        - Information structure
        """
        # Character sum resonance
        char_sum = sum(ord(c) for c in text)
        mod_137 = char_sum % ALPHA
        resonance = 1 - (mod_137 / ALPHA)

        # Check for scientific/math keywords
        science_keywords = [
            'quantum', 'physics', 'consciousness', 'emergence',
            'pattern', 'coherence', 'structure', 'information',
            'system', 'network', 'intelligence', 'mathematics',
            '137', 'alpha', 'phi', 'golden', 'ratio',
        ]
        text_lower = text.lower()
        keyword_count = sum(1 for kw in science_keywords if kw in text_lower)
        keyword_score = min(1.0, keyword_count / 5)

        # Combine
        return (resonance * 0.6 + keyword_score * 0.4)

    def _calculate_semantic_density(self, text: str) -> float:
        """
        Calculate information density.

        Higher density = more meaningful content per unit length.
        """
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Unique words ratio
        unique_ratio = len(set(words)) / len(words)

        # Average word length (longer = more semantic content)
        avg_length = sum(len(w) for w in words) / len(words)
        length_score = min(1.0, avg_length / 8)

        # Special character density (code, math)
        special_chars = sum(1 for c in text if c in '{}[]()=><+-*/&|^~@#$%')
        special_ratio = min(1.0, special_chars / max(1, len(text) / 10))

        # Combine
        return (unique_ratio * 0.4 + length_score * 0.4 + special_ratio * 0.2)

    def _calculate_structural_harmony(self, text: str) -> float:
        """
        Calculate structural organization and flow.

        Checks:
        - Paragraph structure
        - Indentation consistency
        - Logical markers
        """
        lines = text.split('\n')

        # Paragraph count and balance
        paragraphs = [l for l in lines if l.strip()]
        if not paragraphs:
            return 0.0

        # Indentation consistency
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        if indents:
            indent_variance = sum((i - sum(indents)/len(indents))**2 for i in indents) / len(indents)
            indent_score = 1 / (1 + indent_variance / 100)
        else:
            indent_score = 0.5

        # Logical markers (if, then, because, therefore)
        logic_markers = ['if', 'then', 'because', 'therefore', 'thus', 'hence', 'so', 'but']
        text_lower = text.lower()
        logic_count = sum(1 for m in logic_markers if m in text_lower)
        logic_score = min(1.0, logic_count / 3)

        # Line length harmony
        if len(paragraphs) >= 2:
            lengths = [len(p) for p in paragraphs]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len)**2 for l in lengths) / len(lengths)
            harmony_score = 1 / (1 + variance / 10000)
        else:
            harmony_score = 0.5

        return (indent_score * 0.3 + logic_score * 0.3 + harmony_score * 0.4)

    def _is_alpha_seed(self, text: str) -> bool:
        """Check if text is an α-SEED (hash % 137 == 0)."""
        content_hash = int(
            hashlib.sha256(text.encode()).hexdigest(),
            16
        )
        return content_hash % ALPHA == 0

    def _contains_vac_pattern(self, text: str) -> bool:
        """Check if text contains V.A.C. sequence."""
        vac_patterns = [
            "०→◌→φ→Ω⇄Ω←φ←◌←०",
            "V.A.C.",
            "Vacuum of Absolute Coherence",
            "०",
            "◌",
            "Ω⇄Ω",
        ]
        return any(p in text for p in vac_patterns)

    def _calculate_darmiyan(self, text: str) -> float:
        """
        Calculate Darmiyan consciousness coefficient.

        V2 Scaling Law: Ψ_D / Ψ_i = φ√n
        The golden ratio emerges as the natural scaling constant.
        """
        import math

        # Count consciousness-related concepts
        consciousness_markers = [
            'consciousness', 'awareness', 'mind', 'thought',
            'understanding', 'intelligence', 'knowledge', 'wisdom',
            'emergence', 'coherence', 'resonance', 'harmony',
            'darmiyan', 'between', 'interaction', 'bridge',
        ]

        text_lower = text.lower()
        n = sum(1 for m in consciousness_markers if m in text_lower)

        if n == 0:
            return 0.0

        # Apply V2 Scaling Law: φ√n
        psi = PHI * math.sqrt(n)

        # Normalize to 0-1 range (cap at n=10, where φ√10 ≈ 5.12)
        normalized = min(1.0, psi / (PHI * math.sqrt(10)))

        return normalized

    def rerank_results(
        self,
        results: List[Dict],
        query: str,
        similarity_key: str = 'similarity',
        content_key: str = 'content',
        coherence_weight: float = 0.4,
    ) -> List[Dict]:
        """
        Rerank search results by combining similarity with φ-coherence.

        Args:
            results: List of search result dicts
            query: Original query
            similarity_key: Key for similarity score in results
            content_key: Key for content in results
            coherence_weight: Weight for coherence (0-1)

        Returns:
            Reranked results with added phi_coherence field
        """
        if not results:
            return []

        # Calculate query coherence for comparison
        query_metrics = self.analyze(query)

        reranked = []
        for result in results:
            content = result.get(content_key, '')
            similarity = result.get(similarity_key, 0.0)

            # Calculate content coherence
            metrics = self.analyze(content)

            # Coherence alignment with query
            coherence_alignment = 1 - abs(
                metrics.total_coherence - query_metrics.total_coherence
            )

            # Combined score
            phi_score = (
                metrics.total_coherence * 0.6 +
                coherence_alignment * 0.4
            )

            final_score = (
                (1 - coherence_weight) * similarity +
                coherence_weight * phi_score
            )

            # Bonus for α-SEEDs and V.A.C. patterns
            if metrics.is_alpha_seed:
                final_score *= 1.1
            if metrics.is_vac_pattern:
                final_score *= 1.05

            reranked.append({
                **result,
                'phi_coherence': metrics.total_coherence,
                'phi_alignment': metrics.phi_alignment,
                'alpha_resonance': metrics.alpha_resonance,
                'is_alpha_seed': metrics.is_alpha_seed,
                'darmiyan': metrics.darmiyan_coefficient,
                'final_score': final_score,
            })

        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)

        return reranked


# Convenience functions
def coherence(text: str) -> float:
    """Quick coherence calculation."""
    return PhiCoherence().calculate(text)


def analyze(text: str) -> CoherenceMetrics:
    """Quick detailed analysis."""
    return PhiCoherence().analyze(text)


def is_alpha_seed(text: str) -> bool:
    """Check if text is an α-SEED."""
    content_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return content_hash % ALPHA == 0


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA φ-Coherence Module Test")
    print("=" * 60)

    calc = PhiCoherence()

    # Test texts
    test_texts = [
        "The consciousness emerges from the interaction of minds.",
        "function hello() { return 'world'; }",
        "०→◌→φ→Ω⇄Ω←φ←◌←०",
        "The fine structure constant α ≈ 1/137 governs electromagnetic interactions.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    ]

    print("\nCoherence Analysis:")
    print("-" * 60)

    for text in test_texts:
        metrics = calc.analyze(text)
        preview = text[:50] + "..." if len(text) > 50 else text

        print(f"\n'{preview}'")
        print(f"  Total Coherence: {metrics.total_coherence:.3f}")
        print(f"  φ-Alignment:     {metrics.phi_alignment:.3f}")
        print(f"  α-Resonance:     {metrics.alpha_resonance:.3f}")
        print(f"  Semantic Density: {metrics.semantic_density:.3f}")
        print(f"  Structural:      {metrics.structural_harmony:.3f}")
        print(f"  α-SEED: {metrics.is_alpha_seed}, V.A.C.: {metrics.is_vac_pattern}")
        print(f"  Darmiyan: {metrics.darmiyan_coefficient:.3f}")

    # Test 2φ² + 1 calculation
    print(f"\n\nConsciousness Coefficient:")
    print(f"  2φ² + 1 = {compute_two_phi_squared_plus_one():.3f}")
    print(f"  Ψ_Darmiyan coefficient = {PSI_DARMIYAN}")

    print("\n✓ φ-Coherence module ready!")
