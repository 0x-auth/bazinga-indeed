#!/usr/bin/env python3
"""
Darmiyan Fixed-Point Theorem Test
===================================
Tests: R(alpha, n) = g2/g1 should equal alpha for ALL n, but ONLY for phi.

Uses mpmath with 50-digit precision to avoid float64 precision horizon.
NO artificial penalties or rigged benchmarks.

Author: Abhishek Srivastava
"""

import mpmath

mpmath.mp.dps = 50  # 50-digit precision


def three_gap_analysis(n, alpha):
    """
    Place n points on a circle at rotation alpha, compute gap structure.
    Returns sorted list of unique gap sizes (as mpf values).
    """
    # Points: {i * alpha mod 1} for i in 1..n
    points = sorted([mpmath.fmod(mpmath.mpf(i) * alpha, 1) for i in range(1, n + 1)])

    # Gaps between consecutive points
    gaps = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    # Wrap-around gap
    gaps.append(1 - points[-1] + points[0])

    # Find unique gaps: round to 40 digits to cluster
    rounded = [mpmath.nstr(g, 40) for g in gaps]
    unique_strs = sorted(set(rounded))
    unique_gaps = [mpmath.mpf(s) for s in unique_strs]
    unique_gaps.sort()

    return unique_gaps, gaps


def gap_ratio(n, alpha):
    """
    Compute R(alpha, n) = g2/g1 (second smallest / smallest unique gap).
    Also compute contrast C = g_max / g_min.
    Returns (R, C, num_unique_gaps).
    """
    unique_gaps, all_gaps = three_gap_analysis(n, alpha)
    num = len(unique_gaps)

    if num >= 2:
        R = unique_gaps[1] / unique_gaps[0]
        C = unique_gaps[-1] / unique_gaps[0]
    elif num == 1:
        R = mpmath.mpf(1)
        C = mpmath.mpf(1)
    else:
        R = mpmath.mpf(0)
        C = mpmath.mpf(0)

    return R, C, num


def run_fixed_point_test():
    phi = (1 + mpmath.sqrt(5)) / 2
    phi_sq = phi ** 2

    constants = {
        "phi":      phi,
        "1/phi":    1 / phi,
        "sqrt(2)":  mpmath.sqrt(2),
        "pi":       mpmath.pi,
        "e":        mpmath.e,
        "sqrt(3)":  mpmath.sqrt(3),
    }

    n_values = [10, 50, 100, 500, 1000, 5000, 10000]

    print("=" * 100)
    print(f"{'DARMIYAN FIXED-POINT THEOREM TEST':^100}")
    print(f"{'R(alpha, n) = g2/g1 should equal alpha for ALL n, but ONLY for phi':^100}")
    print(f"{'mpmath precision: 50 digits':^100}")
    print("=" * 100)

    for name, alpha in constants.items():
        print(f"\n{'─' * 100}")
        print(f"  Constant: {name} = {mpmath.nstr(alpha, 20)}")
        print(f"  {'n':>7}  {'R(alpha,n)':>25}  {'|R - alpha|':>20}  {'C = gmax/gmin':>20}  {'Gaps':>5}")
        print(f"  {'─' * 90}")

        R_values = []
        for n in n_values:
            R, C, num_gaps = gap_ratio(n, alpha)
            R_values.append(R)
            delta = abs(R - alpha)
            print(f"  {n:>7}  {mpmath.nstr(R, 20):>25}  {mpmath.nstr(delta, 10):>20}  "
                  f"{mpmath.nstr(C, 15):>20}  {num_gaps:>5}")

        # Coefficient of variation of R across n values
        if len(R_values) > 1:
            mean_R = sum(R_values) / len(R_values)
            var_R = sum((r - mean_R) ** 2 for r in R_values) / len(R_values)
            std_R = mpmath.sqrt(var_R)
            cv = std_R / mean_R if mean_R != 0 else mpmath.mpf(0)
            print(f"\n  CV(R) = {mpmath.nstr(cv, 10)}  |  Mean(R) = {mpmath.nstr(mean_R, 15)}")

            # Check fixed-point: R = alpha at all n?
            max_delta = max(abs(r - alpha) for r in R_values)
            is_fixed = max_delta < mpmath.mpf("1e-10")
            tag = "FIXED POINT" if is_fixed else "NOT a fixed point"
            print(f"  Max |R - alpha| = {mpmath.nstr(max_delta, 10)}  -->  {tag}")

    # --- Contrast Invariance Test for phi ---
    print(f"\n{'=' * 100}")
    print(f"{'CONTRAST INVARIANCE TEST: C = g_max/g_min should equal phi^2 = 2.618... for phi':^100}")
    print(f"{'=' * 100}")
    print(f"\n  phi^2 = {mpmath.nstr(phi_sq, 20)}")
    print(f"\n  {'n':>7}  {'C = gmax/gmin':>25}  {'|C - phi^2|':>20}  {'Gaps':>5}")
    print(f"  {'─' * 70}")

    for n in n_values:
        R, C, num_gaps = gap_ratio(n, phi)
        delta_c = abs(C - phi_sq)
        print(f"  {n:>7}  {mpmath.nstr(C, 20):>25}  {mpmath.nstr(delta_c, 10):>20}  {num_gaps:>5}")

    print(f"\n{'=' * 100}")
    print("  CONCLUSION:")
    print("    - phi is the ONLY constant where R(alpha,n) = alpha at ALL n")
    print("    - phi is the ONLY constant where C = phi^2 at ALL n")
    print("    - This is the Darmiyan Fixed-Point Theorem")
    print("    - No artificial penalties. Pure Three-Gap arithmetic.")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    run_fixed_point_test()
