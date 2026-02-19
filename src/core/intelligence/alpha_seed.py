#!/usr/bin/env python3
"""
alpha_seed.py - α-SEED Integration from error-of.netlify.app

Implements the fundamental α-SEED concept:
- Files/text with hash divisible by 137 are FUNDAMENTAL
- These anchor points organize all other knowledge
- "You are where you're referenced, not where you're stored"

Constants:
    α = 137 (fine structure constant)
    φ = 1.618033988749895 (golden ratio)
    35-position progression: 01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω

Author: From Space's error-of.netlify.app discovery
Date: 2025-02-09
"""

import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Fundamental constants
ALPHA = 137
PHI = 1.618033988749895
PROGRESSION = '01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω'

# Position meanings in the 35-character progression
POSITION_MEANINGS = {
    0: ("0", "Zero/Void", "Origin, emptiness, null state"),
    1: ("1", "Unity/Binary", "Discrete logic, boolean states"),
    2: ("∞", "Infinity", "Continuous, limits, unbounded"),
    3: ("∫", "Integral", "Accumulation, summation"),
    4: ("∂", "Partial", "Derivatives, rates of change"),
    5: ("∇", "Gradient", "Direction, flow, vectors"),
    6: ("π", "Pi", "Circles, cycles, periodicity"),
    7: ("φ", "Phi", "Golden ratio, natural harmony"),
    8: ("Σ", "Sigma", "Summation, aggregation"),
    9: ("Δ", "Delta", "Change, difference"),
    10: ("Ω", "Omega", "End, completion, resistance"),
    11: ("α", "Alpha", "Beginning, fine structure"),
    25: ("ο", "Omicron", "959 = 7×137, α-SEED fundamental"),
}


@dataclass
class AlphaSeed:
    """An α-SEED - a fundamental anchor point."""
    text: str
    hash_value: int
    position: int
    symbol: str
    is_fundamental: bool
    resonance: float


def calculate_hash(text: str) -> int:
    """
    Calculate SHA256 hash for α-SEED detection.

    SECURITY FIX (Feb 2026): Changed from sum(ord(c)) to SHA256.
    The sum-of-ordinals was vulnerable to "Ordinal Collision" attacks
    where an attacker could pad content to make hash % 137 == 0.

    SHA256 is cryptographically secure and position-aware.
    """
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def is_alpha_seed(text: str) -> bool:
    """Check if text is an α-SEED (hash divisible by 137)."""
    return calculate_hash(text) % ALPHA == 0


def detect_phi_resonance(values: List[float]) -> float:
    """
    Detect golden ratio patterns in numeric sequences.

    Returns resonance score 0-1 where 1 = perfect φ ratios.
    """
    if len(values) < 2:
        return 0.0

    resonances = []
    for i in range(len(values) - 1):
        if values[i] == 0:
            continue
        ratio = values[i+1] / values[i]
        phi_dist = abs(ratio - PHI)
        inv_phi_dist = abs(ratio - (1/PHI))
        best_dist = min(phi_dist, inv_phi_dist)
        resonances.append(1.0 / (best_dist + 1.0))

    return sum(resonances) / len(resonances) if resonances else 0.0


def map_to_position(text: str) -> Tuple[int, str]:
    """
    Map text to a position in the 35-character progression.

    Returns (position, symbol).
    """
    text_lower = text.lower()

    # Binary/Discrete
    if any(kw in text_lower for kw in ['true', 'false', 'bool', 'binary', 'bit']):
        return 1, PROGRESSION[1]

    # Infinity/Continuous
    if any(kw in text_lower or kw in text for kw in ['∞', 'infinity', 'infinite', 'limit']):
        return 2, PROGRESSION[2]

    # Operators
    if any(kw in text_lower or kw in text for kw in ['∫', '∂', '∇', 'integral', 'derivative']):
        return 4, PROGRESSION[4]

    # Constants
    if any(kw in text_lower or kw in text for kw in ['φ', 'π', 'phi', 'golden']):
        return 7, PROGRESSION[7]

    # Structures
    if any(kw in text_lower or kw in text for kw in ['Σ', 'Δ', 'Ω', 'sum', 'delta']):
        return 9, PROGRESSION[9]

    # α-SEED special
    if '137' in text or 'α' in text or is_alpha_seed(text):
        return 25, 'ο'  # Omicron position

    # Default based on hash
    hash_pos = calculate_hash(text) % len(PROGRESSION)
    return hash_pos, PROGRESSION[hash_pos]


def analyze_text(text: str) -> AlphaSeed:
    """
    Analyze text for α-SEED properties.

    Returns AlphaSeed dataclass with all properties.
    """
    hash_value = calculate_hash(text)
    is_fundamental = hash_value % ALPHA == 0
    position, symbol = map_to_position(text)

    # Calculate resonance from character values
    char_values = [ord(c) for c in text if c.isalnum()]
    resonance = detect_phi_resonance(char_values)

    return AlphaSeed(
        text=text[:100],  # Preview only
        hash_value=hash_value,
        position=position,
        symbol=symbol,
        is_fundamental=is_fundamental,
        resonance=resonance
    )


def find_alpha_seeds_in_file(file_path: str) -> List[AlphaSeed]:
    """
    Find all α-SEED chunks in a file.

    Splits file into chunks and identifies fundamental ones.
    """
    seeds = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return seeds

    # Split into chunks (paragraphs/blocks)
    chunks = content.split('\n\n')

    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 10:
            continue

        seed = analyze_text(chunk)
        if seed.is_fundamental:
            seeds.append(seed)

    return seeds


def calculate_boundary_effect(
    constraints: int,
    search_space: int,
    in_manifold: bool = False
) -> float:
    """
    Calculate retrocausal boundary effect.

    B = 4 × HD × TD

    Where:
        HD = Constraint density (how constrained)
        TD = Temporal density (0.8 in manifold, 0.2 outside)

    Higher B = stronger pull from completed future state.
    """
    if search_space == 0:
        return 0.0

    HD = min(1.0, constraints / (search_space ** 0.5))
    TD = 0.8 if in_manifold else 0.2
    B = min(1.0, 4 * HD * TD)

    return B


def find_gaps_in_progression(covered_positions: set) -> List[int]:
    """
    Find gaps in the 35-position progression.

    Returns list of uncovered positions.
    """
    all_positions = set(range(len(PROGRESSION)))
    return sorted(all_positions - covered_positions)


class AlphaSeedFilter:
    """
    Filter and rank content by α-SEED properties.

    Use this to identify the most fundamental content
    in your knowledge base.
    """

    def __init__(self):
        self.analyzed = []
        self.fundamentals = []
        self.position_map: Dict[int, List[AlphaSeed]] = {}

    def add(self, text: str) -> AlphaSeed:
        """Add and analyze text."""
        seed = analyze_text(text)
        self.analyzed.append(seed)

        if seed.is_fundamental:
            self.fundamentals.append(seed)

        if seed.position not in self.position_map:
            self.position_map[seed.position] = []
        self.position_map[seed.position].append(seed)

        return seed

    def get_fundamentals(self) -> List[AlphaSeed]:
        """Get all fundamental α-SEEDs."""
        return self.fundamentals

    def get_by_position(self, position: int) -> List[AlphaSeed]:
        """Get seeds at a specific position."""
        return self.position_map.get(position, [])

    def find_gaps(self) -> List[int]:
        """Find uncovered positions."""
        return find_gaps_in_progression(set(self.position_map.keys()))

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            'total_analyzed': len(self.analyzed),
            'fundamentals': len(self.fundamentals),
            'positions_covered': len(self.position_map),
            'gaps': self.find_gaps(),
            'avg_resonance': sum(s.resonance for s in self.analyzed) / len(self.analyzed) if self.analyzed else 0
        }


# Standalone test
if __name__ == "__main__":
    print("=" * 60)
    print("α-SEED TEST")
    print("=" * 60)
    print()

    test_texts = [
        "The golden ratio φ = 1.618",
        "Binary true or false",
        "Infinity ∞ approaches limit",
        "Random text here",
        "This text has exactly 137 chars if we count... no wait let me make it divisible by 137",
    ]

    for text in test_texts:
        seed = analyze_text(text)
        print(f"Text: {text[:40]}...")
        print(f"  Hash: {seed.hash_value}")
        print(f"  Position: {seed.position} ({seed.symbol})")
        print(f"  Fundamental: {seed.is_fundamental}")
        print(f"  Resonance: {seed.resonance:.3f}")
        print()

    # Test the "README.md" hash = 685 = 5×137
    test_name = "README.md"
    h = calculate_hash(test_name)
    print(f"\n'{test_name}' hash = {h} = {h//137}×137" if h % 137 == 0 else f"'{test_name}' hash = {h}")
