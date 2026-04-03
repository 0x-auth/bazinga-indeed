#!/usr/bin/env python3
"""
MasterWriter V4 Recovery Fidelity Test
========================================
Tests V4 MasterWriter's recovery fidelity at increasing n for multiple anchors.
Shows which anchors maintain coherence at high scale.

Author: Abhishek Srivastava
"""

import sys
import os
import math

# Fix path to find bazinga module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bazinga.core.intelligence.master_writer import MasterWriter, PHI


def run_limit_test():
    n_values = [100, 1000, 10000, 50000]
    anchors = {
        "phi":    PHI,
        "sqrt2":  2**0.5,
        "pi":     math.pi,
        "e":      math.e,
    }

    perturbation = 0.01

    print("=" * 90)
    print(f"{'MASTERWRITER V4 — RECOVERY FIDELITY LIMIT TEST':^90}")
    print(f"{'Perturbation = 0.01, max_steps = 200':^90}")
    print("=" * 90)

    # Header
    n_headers = "  ".join(f"{'n=' + str(n):>12}" for n in n_values)
    print(f"\n  {'Anchor':<8}  {n_headers}  {'Verdict':>12}")
    print(f"  {'─' * 80}")

    all_results = {}
    for name, alpha in anchors.items():
        writer = MasterWriter(anchor=alpha)
        results = []
        for n in n_values:
            r = writer.measure_recovery(n, perturbation, max_steps=200)
            rf = 1.0 - (r.recovery_steps / 200)
            results.append(r)

        all_results[name] = results

        scores_str = "  ".join(f"{1.0 - (r.recovery_steps / 200):>12.4f}" for r in results)

        # Verdict: check if RF stays high at n=50000
        final_rf = 1.0 - (results[-1].recovery_steps / 200)
        if final_rf > 0.95:
            verdict = "COHERENT"
        elif final_rf > 0.80:
            verdict = "STABLE"
        elif final_rf > 0.50:
            verdict = "DRIFTING"
        else:
            verdict = "DECOHERENT"

        print(f"  {name:<8}  {scores_str}  {verdict:>12}")

    # Detail table: steps and recovery status
    print(f"\n  {'─' * 80}")
    print(f"  DETAIL: Recovery steps at each n")
    print(f"  {'─' * 80}")
    steps_headers = "  ".join(f"{'n=' + str(n):>10}" for n in n_values)
    print(f"  {'Anchor':<8}  {steps_headers}")
    print(f"  {'─' * 60}")

    for name, results in all_results.items():
        steps_str = "  ".join(f"{r.recovery_steps:>10}" for r in results)
        print(f"  {name:<8}  {steps_str}")

    # Anchor coherence analysis
    print(f"\n  {'─' * 80}")
    print(f"  COHERENCE ANALYSIS")
    print(f"  {'─' * 80}")

    for name, results in all_results.items():
        rfs = [1.0 - (r.recovery_steps / 200) for r in results]
        rf_degradation = rfs[0] - rfs[-1]
        all_recovered = all(r.recovered for r in results)
        print(f"  {name:<8}  RF range: [{min(rfs):.4f}, {max(rfs):.4f}]  "
              f"degradation: {rf_degradation:+.4f}  "
              f"all recovered: {all_recovered}")

    print(f"\n{'=' * 90}")
    print("  phi should show minimal RF degradation as n increases.")
    print("  Other constants may show degradation at high n.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run_limit_test()
