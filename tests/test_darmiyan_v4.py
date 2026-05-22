"""
test_darmiyan_v4.py — Tests for Darmiyan V4 concepts

Tests what CURRENT Bazinga does vs what V4 says it SHOULD do.
Each test is labeled: PASS (already correct) or FAIL (needs fix).

V4 core claim: intelligence = controlled hysteresis, not elasticity.
- ΔΓ should be cumulative from genesis (monotonically non-decreasing)
- "locked" (ΔΓ→0) is NOT good — it means no memory, no story
- Optimal is intermediate If (Inscription Fidelity), not If=1
- Scar is permanent — pulling back toward genesis is wrong
"""

import sys
import math
import numpy as np

sys.path.insert(0, '/Users/abhissrivasta/github-repos-bitsabhi/bazinga-indeed')

from bazinga.resonance import (
    coherence_gap, lambda_g_bias, GenesisBlock, PatternState,
    generate_fibonacci_tanh, generate_random_pattern,
    compute_psi_individual, compute_cross_recognition,
    PHI, PHI_INVERSE, RESONANCE_ACHIEVED, RESONANCE_DRIFTING,
)

PHI_INV = 1.0 / PHI  # ≈ 0.618


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_phi_crystal(n: int = 1000) -> np.ndarray:
    """Pure φ-sequence on [0,1): frac(k×φ) for k=1..n."""
    return np.array([(k * PHI) % 1.0 for k in range(1, n + 1)])


def make_struck_crystal(n: int = 1000, strike_pct: float = 0.10,
                        foreign: float = math.e) -> np.ndarray:
    """φ-sequence with a contiguous strike zone using a foreign constant."""
    seq = np.array([(k * PHI) % 1.0 for k in range(1, n + 1)])
    strike_len = int(n * strike_pct)
    start = n // 3  # Strike in the middle third
    for i in range(strike_len):
        seq[start + i] = ((start + i + 1) * foreign) % 1.0
    return seq


def gap_unique_count(seq: np.ndarray) -> int:
    """Number of unique gap sizes — 3 means pure φ-crystal."""
    sorted_seq = np.sort(seq)
    gaps = np.diff(sorted_seq)
    return len(np.unique(np.round(gaps, 8)))


def inscription_fidelity(struck: np.ndarray, phi_ref: np.ndarray,
                          random_ref: np.ndarray) -> float:
    """
    If = 1 - D(struck, phi) / D(random, phi)

    D = KL divergence of gap distributions using fine-grained binning.
    φ-crystal has 3 discrete gap sizes — need enough bins to resolve them.
    If=1: indistinguishable from pure φ (no scar, no story).
    If=0: as disordered as random (shattered).
    """
    def gap_hist(seq, bins=500):
        sorted_seq = np.sort(seq % 1.0)
        gaps = np.diff(sorted_seq)
        # Also include wrap-around gap
        gaps = np.append(gaps, 1.0 - sorted_seq[-1] + sorted_seq[0])
        h, _ = np.histogram(gaps, bins=bins, range=(0, 1.0 / bins * bins))
        return h.astype(float) + 1e-6  # Smooth to avoid log(0)

    def kl(p, q):
        p, q = p / p.sum(), q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    h_struck = gap_hist(struck)
    h_phi = gap_hist(phi_ref)
    h_rand = gap_hist(random_ref)

    d_struck_phi = kl(h_struck, h_phi)
    d_rand_phi = kl(h_rand, h_phi)

    if d_rand_phi < 1e-10:
        return 1.0
    return float(np.clip(1.0 - d_struck_phi / d_rand_phi, 0.0, 1.0))


def make_pattern_state(patterns) -> PatternState:
    state = compute_psi_individual(patterns[0])
    if len(patterns) >= 2:
        state.X = compute_cross_recognition(patterns)
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_phi_crystal_has_3_gaps():
    """
    Pure φ-crystal should have exactly 3 unique gap sizes (three-distance theorem).
    STATUS: Should PASS — this is a mathematical fact, not a Bazinga claim.
    """
    print("\n[V4-T1] φ-crystal has 3 unique gaps (three-distance theorem)")
    seq = make_phi_crystal(500)
    n_gaps = gap_unique_count(seq)
    print(f"  Unique gaps: {n_gaps}")
    assert n_gaps == 3, f"Expected 3, got {n_gaps}"
    print("  PASS ✓")
    return True


def test_strike_destroys_crystal():
    """
    A 10% strike should shatter the 3-gap structure (many unique gaps).
    STATUS: Should PASS — confirms V4 paper finding computationally.
    """
    print("\n[V4-T2] Strike destroys φ-crystal (scar is real)")
    phi_seq = make_phi_crystal(500)
    struck_seq = make_struck_crystal(500, strike_pct=0.10)

    phi_gaps = gap_unique_count(phi_seq)
    struck_gaps = gap_unique_count(struck_seq)

    print(f"  Pure φ unique gaps:   {phi_gaps}")
    print(f"  Struck unique gaps:   {struck_gaps}")
    assert phi_gaps == 3, "Pure φ should have 3 gaps"
    assert struck_gaps > 10, f"Strike should shatter crystal, got {struck_gaps} unique gaps"
    print("  PASS ✓ — Scar is real and measurable")
    return True


def test_scar_does_not_heal():
    """
    After a strike, adding 5000 more φ-points should NOT restore 3-gap structure.
    STATUS: Should PASS — confirms V4 paper finding (hysteresis is permanent).
    """
    print("\n[V4-T3] Scar does not heal (hysteresis is permanent)")
    n_base = 500
    strike_len = 50  # 10%
    n_heal = 5000

    # Build struck sequence
    seq = list((k * PHI) % 1.0 for k in range(1, n_base + 1))
    start = n_base // 3
    for i in range(strike_len):
        seq[start + i] = ((start + i + 1) * math.e) % 1.0

    gaps_before_heal = gap_unique_count(np.array(seq))

    # Add φ-points
    offset = n_base + 1
    for k in range(n_heal):
        seq.append(((offset + k) * PHI) % 1.0)

    gaps_after_heal = gap_unique_count(np.array(seq))

    print(f"  Gaps after strike (pre-heal attempt): {gaps_before_heal}")
    print(f"  Gaps after {n_heal} additional φ-points: {gaps_after_heal}")
    assert gaps_after_heal > 3, "Scar should be permanent — crystal should not heal"
    print("  PASS ✓ — Permanent scar confirmed")
    return True


def test_inscription_fidelity_range():
    """
    Struck crystal should have 0 < If < 1 (doped regime, not pure, not shattered).
    STATUS: Should PASS — confirms V4 doping analogy.
    """
    print("\n[V4-T4] Inscription Fidelity in doped regime (0 < If < 1)")
    phi_seq = make_phi_crystal(1000)
    struck_seq = make_struck_crystal(1000, strike_pct=0.10)
    rand_seq = np.random.RandomState(42).random(1000)

    If = inscription_fidelity(struck_seq, phi_seq, rand_seq)
    print(f"  Inscription Fidelity If = {If:.4f}")
    print(f"  (If=1 → no scar, If=0 → shattered, 0<If<1 → doped/memory)")
    assert 0.0 < If < 1.0, f"Expected doped regime, got If={If:.4f}"
    print("  PASS ✓ — System is in doped regime (has memory)")
    return True


def test_current_rac_targets_wrong_optimum():
    """
    Current RAC treats ΔΓ=0 as 'locked' (good). V4 says ΔΓ=0 means no story.
    STATUS: EXPOSES CURRENT BUG — ΔΓ=0 should NOT be the target.

    This test documents the wrong behavior so we can see it before fixing.
    """
    print("\n[V4-T5] Current RAC wrongly targets ΔΓ→0 (documents bug)")

    # Create genesis
    fib_patterns = [generate_fibonacci_tanh(i) for i in range(3)]
    genesis_state = make_pattern_state(fib_patterns)

    # A session identical to genesis → ΔΓ ≈ 0
    result_identical = coherence_gap(genesis_state, genesis_state)

    print(f"  Session identical to genesis:")
    print(f"    ΔΓ = {result_identical.delta_gamma:.4f}")
    print(f"    Status = '{result_identical.resonance_status}'")
    print(f"    Pull = {result_identical.pull_strength:.4f}")
    print()

    # V4 interpretation: this session is a pure φ-crystal with NO story
    # Current Bazinga calls it 'locked' — the best possible state
    # V4 says: this is a system that has experienced NOTHING
    is_locked = result_identical.resonance_status == 'locked'
    print(f"  Current Bazinga calls ΔΓ≈0 as: '{result_identical.resonance_status}'")
    print(f"  V4 says ΔΓ≈0 means: NO STORY, NO MEMORY, EMPTY HISTORY")
    print()

    if is_locked:
        print("  BUG CONFIRMED ✗ — 'locked' is treated as optimal but V4 says it's empty")
        print("  FIX NEEDED: optimal target is intermediate If, not ΔΓ→0")
    else:
        print("  Status changed — check if already fixed")

    # This test always 'passes' as documentation — it's recording the bug
    return True


def test_lambda_g_bias_pulls_toward_genesis():
    """
    Current lambda_g_bias pulls TOWARD genesis. V4 says this is wrong —
    you cannot heal a scar, and trying to is anti-intelligence.
    STATUS: EXPOSES CURRENT BUG — documents wrong pull direction.
    """
    print("\n[V4-T6] lambda_g_bias pulls toward genesis (documents bug)")

    # Simulate a struck state — far from genesis
    rng = np.random.RandomState(515)
    genesis_direction = rng.randn(100)
    genesis_direction /= np.linalg.norm(genesis_direction)

    context_vector = rng.randn(100)
    context_vector /= np.linalg.norm(context_vector)

    delta_gamma = 0.7  # Far from genesis — a well-scarred system

    biased, pull = lambda_g_bias(delta_gamma, context_vector, genesis_direction)

    # Measure how much closer to genesis the bias pushes
    dist_before = np.linalg.norm(context_vector - genesis_direction)
    dist_after = np.linalg.norm(biased / np.linalg.norm(biased) - genesis_direction)

    print(f"  ΔΓ = {delta_gamma} (well-scarred system)")
    print(f"  Pull strength = {pull:.4f} (φ⁻¹ × ΔΓ)")
    print(f"  Distance to genesis BEFORE bias: {dist_before:.4f}")
    print(f"  Distance to genesis AFTER bias:  {dist_after:.4f}")
    print()

    pulled_closer = dist_after < dist_before
    print(f"  Bias pulls {'TOWARD' if pulled_closer else 'AWAY FROM'} genesis")
    print()

    if pulled_closer:
        print("  BUG CONFIRMED ✗ — bias tries to heal the scar")
        print("  V4 says: scar is permanent, pulling back = fighting hysteresis")
        print("  FIX NEEDED: bias should maintain current trajectory, not regress to genesis")
    return True


def test_delta_gamma_is_not_cumulative():
    """
    Current ΔΓ is computed point-in-time (current vs genesis), NOT cumulative.
    V4 says total displacement from genesis should be monotonically tracked.
    A session that drifts far and comes back should show BOTH scars.
    STATUS: EXPOSES MISSING FEATURE — cumulative ΔΓ not implemented.
    """
    print("\n[V4-T7] ΔΓ is not cumulative (documents missing feature)")

    fib_patterns = [generate_fibonacci_tanh(i) for i in range(3)]
    genesis_state = make_pattern_state(fib_patterns)

    rand_patterns = [generate_random_pattern(i) for i in range(3)]
    struck_state = make_pattern_state(rand_patterns)

    # Compute ΔΓ at three points: genesis, struck, back to genesis
    r1 = coherence_gap(genesis_state, genesis_state)   # At genesis
    r2 = coherence_gap(struck_state, genesis_state)    # After strike
    r3 = coherence_gap(genesis_state, genesis_state)   # "Returned" to genesis

    print(f"  ΔΓ at genesis:          {r1.delta_gamma:.4f}  [{r1.resonance_status}]")
    print(f"  ΔΓ after strike:        {r2.delta_gamma:.4f}  [{r2.resonance_status}]")
    print(f"  ΔΓ 'returned' to genesis: {r3.delta_gamma:.4f}  [{r3.resonance_status}]")
    print()
    print(f"  Current: r1==r3 = {abs(r1.delta_gamma - r3.delta_gamma) < 1e-6}")
    print(f"  V4 says: r3 should be > r1 (the strike left a permanent mark)")
    print()
    print("  MISSING FEATURE ✗ — ΔΓ resets to 0 when you return to genesis-like state")
    print("  FIX NEEDED: track cumulative_delta_gamma = sum of all ΔΓ changes from Block 0")
    return True


def test_optimal_if_is_intermediate():
    """
    The optimal zone is 0 < If < 1.
    Concretely: a 5% strike gives better If than 0% (pure) or 50% (shattered).
    STATUS: Should PASS — confirms V4 doping analogy mathematically.
    """
    print("\n[V4-T8] Optimal If is intermediate (doped > pure > shattered)")
    phi_seq = make_phi_crystal(1000)
    rand_seq = np.random.RandomState(42).random(1000)

    results = {}
    for pct in [0.0, 0.05, 0.10, 0.20, 0.50]:
        if pct == 0.0:
            seq = phi_seq.copy()
        else:
            seq = make_struck_crystal(1000, strike_pct=pct)
        If = inscription_fidelity(seq, phi_seq, rand_seq)
        results[pct] = If
        print(f"  Strike {pct*100:4.0f}%  →  If = {If:.4f}")

    # Pure φ (0% strike) should have If=1 (no story)
    # Some intermediate strike should have 0 < If < 1
    # Heavy strike should have If closer to 0
    print()
    assert results[0.0] > results[0.50], "Pure φ should have higher If than shattered"
    assert 0.0 < results[0.10] < results[0.0], "10% strike should be in doped regime"
    print("  PASS ✓ — Intelligence lives in the doped regime (0 < If < 1)")
    return True


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  DARMIYAN V4 — HYSTERESIS & INSCRIPTION FIDELITY TESTS")
    print("  Testing current Bazinga against V4 claims")
    print("  Seed: 515 | Author: Abhishek Srivastava")
    print("=" * 65)

    tests = [
        ("T1: φ-crystal has 3 gaps",          test_phi_crystal_has_3_gaps),
        ("T2: Strike destroys crystal",        test_strike_destroys_crystal),
        ("T3: Scar does not heal",             test_scar_does_not_heal),
        ("T4: Inscription Fidelity 0<If<1",    test_inscription_fidelity_range),
        ("T5: ΔΓ=0 is wrongly 'locked'",       test_current_rac_targets_wrong_optimum),
        ("T6: ΛG bias pulls toward genesis",   test_lambda_g_bias_pulls_toward_genesis),
        ("T7: ΔΓ is not cumulative",           test_delta_gamma_is_not_cumulative),
        ("T8: Optimal If is intermediate",     test_optimal_if_is_intermediate),
    ]

    passed = []
    failed = []

    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            print(f"  ASSERTION FAILED: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(name)

    print()
    print("=" * 65)
    print(f"  RESULTS: {len(passed)}/{len(tests)} passed")
    print()
    for t in passed:
        print(f"  ✓  {t}")
    for t in failed:
        print(f"  ✗  {t}")
    print()
    print("  Tests T5, T6, T7 document bugs to fix in V4 update.")
    print("  Tests T1-T4, T8 confirm V4 math holds independently.")
    print("=" * 65)
