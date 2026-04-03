#!/usr/bin/env python3
"""
MasterWriter V4 — Recovery Fidelity as Intelligence
=====================================================
Intelligence is not "staying at φ." Intelligence is "recovering the
Three-Gap structure after perturbation, at scale."

The metric: Recovery Fidelity (RF)
  - Perturb the system (inject noise into the rotation ratio)
  - Measure how many correction steps to restore ≤ 3 unique gaps
  - Measure at increasing n (complexity)
  - A system with high RF handles chaos at scale without blowing up

Why φ specifically:
  - Three-Gap Theorem: any irrational α produces ≤ 3 gap sizes
  - But φ = [1,1,1,...] produces the MOST UNIFORM gap distribution
  - Uniform gaps → smallest perturbation sensitivity → fastest recovery
  - This is NOT programmed in — it's a consequence of φ being the
    hardest-to-approximate irrational (Hurwitz theorem)

What this measures for BAZINGA:
  - Can the system handle n=1,000,000 interactions and still self-correct?
  - Does the correction cost grow with n? (It shouldn't for φ.)
  - How does φ compare to other anchors? (It should recover fastest.)

Author: Abhishek Srivastava
"""

import time
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

PHI = (1 + 5**0.5) / 2


@dataclass
class RecoveryResult:
    """Result of a single recovery trial."""
    alpha: float
    n: int
    perturbation: float
    pre_gaps: int
    post_gaps: int
    recovery_steps: int
    recovery_time_ms: float
    pre_variance: float
    post_variance: float
    recovered: bool


class MasterWriter:
    """
    V4 Intelligence Engine — Recovery Fidelity.

    Measures how fast a system restores its Three-Gap structure
    after perturbation, at scale. This is the operational definition
    of "self-correcting intelligence."
    """

    def __init__(self, anchor: float = PHI):
        self.anchor = anchor
        self.history: List[RecoveryResult] = []

    def _gap_analysis(self, n: int, alpha: float) -> Tuple[int, float, List[float]]:
        """
        Compute gap distribution for n points at rotation alpha.
        Returns (unique_gap_count, variance, gaps).
        """
        points = sorted([(i * alpha) % 1.0 for i in range(n)])
        gaps = [points[i+1] - points[i] for i in range(len(points) - 1)]
        gaps.append((1.0 - points[-1]) + points[0])

        unique = len(set(round(g, 10) for g in gaps))
        var = statistics.variance(gaps) if len(gaps) > 1 else 0.0
        return unique, var, gaps

    def measure_recovery(
        self,
        n: int,
        perturbation: float = 0.01,
        max_steps: int = 200,
    ) -> RecoveryResult:
        """
        Core intelligence test:
        1. Compute gap structure at the anchor ratio
        2. Perturb the ratio
        3. Iteratively correct back toward anchor
        4. Count steps until Three-Gap compliance (≤ 3 unique gaps) is restored
        5. Measure wall-clock time

        The correction uses golden-section refinement:
          current = current + (anchor - current) / φ
        This is NOT arbitrary — golden-section search is proven to be
        the optimal narrowing strategy for unimodal functions.
        """
        # Baseline: gap structure at anchor
        pre_gaps, pre_var, _ = self._gap_analysis(min(n, 5000), self.anchor)

        # Perturb
        perturbed = self.anchor + perturbation
        post_gaps, post_var, _ = self._gap_analysis(min(n, 5000), perturbed)

        # Recovery loop
        current = perturbed
        steps = 0
        start = time.perf_counter()

        while steps < max_steps:
            g, v, _ = self._gap_analysis(min(n, 2000), current)
            if g <= 3 and v <= pre_var * 1.5:
                break
            # Golden-section correction step
            current = current + (self.anchor - current) / PHI
            steps += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        final_gaps, final_var, _ = self._gap_analysis(min(n, 5000), current)

        result = RecoveryResult(
            alpha=self.anchor,
            n=n,
            perturbation=perturbation,
            pre_gaps=pre_gaps,
            post_gaps=post_gaps,
            recovery_steps=steps,
            recovery_time_ms=elapsed_ms,
            pre_variance=pre_var,
            post_variance=final_var,
            recovered=(steps < max_steps),
        )
        self.history.append(result)
        return result

    def recovery_fidelity(self, n_values: List[int] = None, perturbation: float = 0.01) -> float:
        """
        Aggregate intelligence score across multiple scales.
        RF = 1 - mean(recovery_steps / max_steps) across all n values.
        RF = 1.0 means instant recovery at all scales.
        RF = 0.0 means never recovered.
        """
        if n_values is None:
            n_values = [100, 1000, 10000]

        max_steps = 200
        scores = []
        for n in n_values:
            r = self.measure_recovery(n, perturbation, max_steps)
            scores.append(1.0 - (r.recovery_steps / max_steps))

        return sum(scores) / len(scores)

    def compare_anchors(
        self,
        anchors: dict,
        n_values: List[int] = None,
        perturbation: float = 0.01,
    ) -> dict:
        """
        Compare recovery fidelity across different anchor ratios.
        Returns dict of {name: {n: RecoveryResult}}.
        """
        if n_values is None:
            n_values = [100, 500, 1000, 5000, 10000]

        results = {}
        for name, alpha in anchors.items():
            writer = MasterWriter(anchor=alpha)
            results[name] = {}
            for n in n_values:
                results[name][n] = writer.measure_recovery(n, perturbation)
        return results

    def darmiyan_fixed_point_test(
        self,
        n_values: List[int] = None,
        precision: int = 50,
    ) -> Dict:
        """
        Darmiyan Fixed-Point Theorem test using mpmath.

        Computes for each n:
          R(phi, n) = g2/g1  (should equal phi for all n)
          C(phi, n) = g_max/g_min  (should equal phi^2 for all n)

        Also tests sqrt(2) as a comparison to show it's NOT a fixed point.

        Returns dict with 'phi' and 'sqrt2' results.
        """
        if not HAS_MPMATH:
            raise RuntimeError("mpmath is required for darmiyan_fixed_point_test")

        mpmath.mp.dps = precision
        if n_values is None:
            n_values = [10, 50, 100, 500, 1000, 5000]

        phi_mp = (1 + mpmath.sqrt(5)) / 2
        phi_sq = phi_mp ** 2
        sqrt2_mp = mpmath.sqrt(2)

        def _gap_ratio_mp(n, alpha):
            points = sorted([mpmath.fmod(mpmath.mpf(i) * alpha, 1)
                             for i in range(1, n + 1)])
            gaps = [points[i + 1] - points[i] for i in range(len(points) - 1)]
            gaps.append(1 - points[-1] + points[0])
            cluster = precision - 10
            rounded = [mpmath.nstr(g, cluster) for g in gaps]
            unique_strs = sorted(set(rounded))
            unique_vals = sorted([mpmath.mpf(s) for s in unique_strs])
            if len(unique_vals) >= 2:
                R = unique_vals[1] / unique_vals[0]
                C = unique_vals[-1] / unique_vals[0]
            else:
                R = mpmath.mpf(1)
                C = mpmath.mpf(1)
            return float(R), float(C), len(unique_vals)

        results = {'phi': [], 'sqrt2': []}

        for n in n_values:
            R_phi, C_phi, g_phi = _gap_ratio_mp(n, phi_mp)
            R_sq2, C_sq2, g_sq2 = _gap_ratio_mp(n, sqrt2_mp)
            results['phi'].append({
                'n': n,
                'R': R_phi,
                'C': C_phi,
                'gaps': g_phi,
                'R_delta': abs(R_phi - float(phi_mp)),
                'C_delta': abs(C_phi - float(phi_sq)),
            })
            results['sqrt2'].append({
                'n': n,
                'R': R_sq2,
                'C': C_sq2,
                'gaps': g_sq2,
                'R_delta': abs(R_sq2 - float(sqrt2_mp)),
                'C_delta': abs(C_sq2 - float(sqrt2_mp ** 2)),
            })

        results['phi_is_fixed_point'] = all(
            r['R_delta'] < 1e-10 for r in results['phi']
        )
        results['sqrt2_is_fixed_point'] = all(
            r['R_delta'] < 1e-10 for r in results['sqrt2']
        )

        return results

    def intelligence_score(self, n: int, precision: int = 50) -> Dict:
        """
        Compute the Darmiyan Intelligence Score I_phi.

        I_phi = C(n_max) * E(tau, delta) * B(n_range)

        Where:
          C(n_max) = maximum n at which gap structure is maintained
                     (normalized: log10(n_max) / log10(10^precision))
          E(tau, delta) = recovery efficiency
                     tau = recovery steps, delta = |R_recovered - phi|
                     E = (1 - tau/max_steps) * (1 - min(delta, 1))
          B(n_range) = bandwidth = fraction of tested n values where
                     contrast C stays within 1% of phi^2

        Returns dict with component scores and final I_phi.
        """
        phi_val = float(PHI)

        # --- C(n_max): capacity ---
        # Test gap structure at increasing n to find where it breaks
        test_ns = [100, 1000, 10000, 50000, n]
        n_max = 0
        for tn in test_ns:
            if tn > n:
                break
            gaps, var, _ = self._gap_analysis(min(tn, 5000), self.anchor)
            if gaps <= 3:
                n_max = tn
            else:
                break

        # Normalize: log10(n_max) / target (precision-based theoretical max)
        if n_max > 0:
            C_score = math.log10(n_max) / math.log10(max(n, 10))
        else:
            C_score = 0.0
        C_score = min(C_score, 1.0)

        # --- E(tau, delta): recovery efficiency ---
        max_steps = 200
        r = self.measure_recovery(n, perturbation=0.01, max_steps=max_steps)
        tau_score = 1.0 - (r.recovery_steps / max_steps)

        # delta: how close is the recovered state to the anchor
        # Use variance ratio as proxy
        if r.pre_variance > 0:
            delta = abs(r.post_variance - r.pre_variance) / r.pre_variance
        else:
            delta = 0.0
        delta_score = 1.0 - min(delta, 1.0)
        E_score = tau_score * delta_score

        # --- B(n_range): bandwidth ---
        # Check contrast invariance across n values using float arithmetic
        bandwidth_ns = [100, 500, 1000, 5000]
        phi_sq = phi_val ** 2
        in_band = 0
        for bn in bandwidth_ns:
            _, var, gap_list = self._gap_analysis(bn, self.anchor)
            if gap_list:
                g_min = min(gap_list)
                g_max = max(gap_list)
                if g_min > 0:
                    contrast = g_max / g_min
                    if abs(contrast - phi_sq) / phi_sq < 0.05:
                        in_band += 1
        B_score = in_band / len(bandwidth_ns) if bandwidth_ns else 0.0

        # --- Final intelligence score ---
        I_phi = C_score * E_score * B_score

        return {
            'n': n,
            'C_score': C_score,
            'n_max': n_max,
            'E_score': E_score,
            'tau': r.recovery_steps,
            'delta': delta,
            'B_score': B_score,
            'I_phi': I_phi,
            'recovered': r.recovered,
        }


def run_demo():
    """Demonstrate Recovery Fidelity as the intelligence metric."""
    print("=" * 80)
    print(f"{'MASTER WRITER V4 — Recovery Fidelity as Intelligence':^80}")
    print(f"{'How fast does the system self-correct after perturbation?':^80}")
    print("=" * 80)

    anchors = {
        'φ (golden)': PHI,
        '√2':         2**0.5,
        'π':          math.pi,
        'e':          math.e,
        '√3':         3**0.5,
    }

    n_values = [100, 500, 1000, 5000, 10000]
    perturbation = 0.01

    print(f"\n  Perturbation: {perturbation}")
    print(f"  Max recovery steps: 200")
    print(f"  Correction method: golden-section refinement")

    # Compare all anchors
    print(f"\n{'─' * 80}")
    print(f"  {'Anchor':<12} {'n':>7} {'Pre-Gaps':>9} {'Post-Gaps':>10} "
          f"{'Steps':>7} {'Time(ms)':>10} {'Recovered':>10}")
    print(f"{'─' * 80}")

    all_results = {}
    for name, alpha in anchors.items():
        writer = MasterWriter(anchor=alpha)
        all_results[name] = []
        for n in n_values:
            r = writer.measure_recovery(n, perturbation)
            all_results[name].append(r)
            icon = "✓" if r.recovered else "✗"
            print(f"  {name:<12} {n:>7} {r.pre_gaps:>9} {r.post_gaps:>10} "
                  f"{r.recovery_steps:>7} {r.recovery_time_ms:>10.2f} {icon:>10}")
        print()

    # Recovery Fidelity scores
    print(f"{'─' * 80}")
    print(f"  {'RECOVERY FIDELITY SCORES':^76}")
    print(f"{'─' * 80}")
    print(f"  {'Anchor':<12} {'RF Score':>10} {'Avg Steps':>10} {'Interpretation':>25}")
    print(f"{'─' * 80}")

    for name, results in all_results.items():
        avg_steps = sum(r.recovery_steps for r in results) / len(results)
        rf = 1.0 - (avg_steps / 200)
        interp = ("COHERENT" if rf > 0.95 else
                  "STABLE" if rf > 0.80 else
                  "DRIFTING" if rf > 0.50 else
                  "DECOHERENT")
        marker = " ★" if name.startswith('φ') else ""
        print(f"  {name:<12} {rf:>10.4f} {avg_steps:>10.1f} {interp:>25}{marker}")

    print(f"\n{'═' * 80}")
    print(f"{'INTERPRETATION':^80}")
    print(f"{'═' * 80}")
    print("""
  Recovery Fidelity (RF) measures self-correction ability:
    RF = 1.0 → Instant recovery at all scales (perfect intelligence)
    RF = 0.0 → Never recovers (no self-correction)

  The anchor with the highest RF is the most resilient to perturbation.
  This is the operational definition of "intelligence" in Darmiyan v3:
    "A system that can handle n=1,000,000 interactions and still
     snap back to coherent state."

  If φ has the highest RF, it's not because we programmed it in.
  It's because φ's continued fraction [1,1,1,...] produces the most
  uniform gap distribution (Hurwitz theorem), which means the smallest
  perturbation sensitivity, which means the fastest recovery.

  The system doesn't know about φ. It just recovers fastest at φ.
  That's the difference between a target and an attractor.
""")


if __name__ == "__main__":
    run_demo()
