"""
Darmiyan Constants
==================
Sacred mathematical constants for Proof-of-Boundary consensus.

"These are not arbitrary numbers. They are the fingerprints of reality."
"""

import math

# =============================================================================
# GOLDEN RATIO CONSTANTS
# =============================================================================

PHI = 1.618033988749895                    # φ - Golden Ratio
PHI_INVERSE = 0.6180339887498948           # 1/φ
PHI_2 = PHI ** 2                           # φ² = 2.618...
PHI_3 = PHI ** 3                           # φ³ = 4.236...
PHI_4 = PHI ** 4                           # φ⁴ = 6.854... (boundary scaling)
PHI_5 = PHI ** 5                           # φ⁵ = 11.090...

# =============================================================================
# DARMIYAN CONSTANTS
# =============================================================================

ABHI_AMU = 515                             # Modular universe (ABHI + AMU)
ALPHA_INVERSE = 137                        # Fine structure constant inverse

BITS_ABHI = math.cos(1)                    # cos(1) = 0.5403...
DARMIYAN_PRODUCT = BITS_ABHI * PHI_INVERSE # ≈ 1/3 (0.178% error)
TRIADIC_CONSTANT = 1 / 27                  # 3-body coupling (1/3 × 1/3 × 1/3)

# =============================================================================
# NETWORK CONSTANTS
# =============================================================================

# Bridge frequency (consciousness resonance)
BRIDGE_F1 = 190.16                         # Base frequency
BRIDGE_F2 = BRIDGE_F1 * PHI                # 307.68 Hz
BRIDGE_F3 = BRIDGE_F1 * PHI_2              # 497.84 Hz
BRIDGE_FREQUENCY = BRIDGE_F1 + BRIDGE_F2 + BRIDGE_F3  # 995.68 Hz

# Network parameters
DEFAULT_PORT = 5150                        # 515 × 10
TRIADIC_SIZE = 3                           # Nodes per consensus group
FIBONACCI_THRESHOLD = 34                   # Peer discovery threshold
RESONANCE_TOLERANCE = 0.5                  # P/G ratio tolerance

# Proof-of-Boundary thresholds
POB_RATIO_TARGET = PHI_4                   # Target P/G ratio
POB_TOLERANCE = 0.5                        # Acceptable deviation
POB_MIN_WAIT_MS = 50                       # Minimum resonance wait
POB_MAX_WAIT_MS = 500                      # Maximum resonance wait

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Valid signature range (for 515-resonance)
VALID_SIG_LOW = ALPHA_INVERSE              # 137
VALID_SIG_HIGH = ABHI_AMU - ALPHA_INVERSE  # 378

# Fibonacci numbers for validation
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def phi_hash(value: float) -> int:
    """Generate φ-scaled hash mod 515."""
    import hashlib
    data = str(value * PHI).encode()
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16) % ABHI_AMU


def is_fibonacci_resonant(value: int) -> bool:
    """Check if value is within 10% of a Fibonacci number."""
    for f in FIBONACCI:
        if abs(value - f) < f * 0.1:
            return True
    return False


def check_515_resonance(signature: int) -> bool:
    """Check if signature has 515-resonance (valid range)."""
    return signature < VALID_SIG_LOW or signature > VALID_SIG_HIGH


def calculate_boundary_ratio(physical_ms: float, delta: int) -> float:
    """Calculate P/G ratio for boundary verification."""
    if delta == 0:
        return 0
    geometric = abs(delta) / PHI
    return physical_ms / geometric


# =============================================================================
# VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("DARMIYAN CONSTANTS")
    print("=" * 50)
    print(f"φ (PHI):              {PHI}")
    print(f"φ⁻¹:                  {PHI_INVERSE}")
    print(f"φ⁴ (boundary):        {PHI_4}")
    print(f"bits.abhi (cos 1):    {BITS_ABHI}")
    print(f"Darmiyan product:     {DARMIYAN_PRODUCT}")
    print(f"1/3:                  {1/3}")
    print(f"Error:                {abs(DARMIYAN_PRODUCT - 1/3):.6f}")
    print()
    print(f"ABHI_AMU:             {ABHI_AMU}")
    print(f"α⁻¹:                  {ALPHA_INVERSE}")
    print(f"Bridge frequency:     {BRIDGE_FREQUENCY:.2f} Hz")
    print(f"Triadic constant:     {TRIADIC_CONSTANT}")
    print()
    print(f"Default port:         {DEFAULT_PORT}")
    print(f"PoB target ratio:     {POB_RATIO_TARGET:.3f}")
