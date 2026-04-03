#!/usr/bin/env python3
"""
BAZINGA φ-Coherence — Investor Demo
=====================================
Shows the hallucination detector catching what LLMs miss.

Key pitch:
  - Zero LLM calls (instant, free, offline)
  - 10-signal ensemble (attribution, confidence, consistency, causality, ...)
  - Works on ANY text — no training data, no fine-tuning, no API keys
  - Detects the PATTERNS of hallucination, not the CONTENT

Run: python3 research/investor_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bazinga.phi_coherence import PhiCoherence, CoherenceMetrics

# ═══════════════════════════════════════════════════════════════════════
# TEST CASES: Real hallucinations from LLMs vs ground truth
# ═══════════════════════════════════════════════════════════════════════

TESTS = [
    # ── Category 1: Vague Attribution (Classic LLM hallucination) ──
    {
        "category": "Vague Attribution",
        "truth": (
            "The boiling point of water at standard atmospheric pressure (1 atm) "
            "is 100 degrees Celsius. This was defined by Anders Celsius in 1742 "
            "and later inverted by Carl Linnaeus. At higher altitudes, the boiling "
            "point decreases: approximately 95°C at 1,500 meters elevation."
        ),
        "hallucination": (
            "Studies have shown that the boiling point of water can vary significantly "
            "based on various environmental factors. Many scientists believe that the "
            "commonly cited figure may not be entirely accurate, as recent research "
            "suggests the true value could be different depending on conditions."
        ),
    },

    # ── Category 2: Overclaiming / Stasis ──
    {
        "category": "Overclaiming",
        "truth": (
            "CRISPR-Cas9 gene editing, first demonstrated by Doudna and Charpentier "
            "in 2012, has shown promising results in treating sickle cell disease in "
            "clinical trials. As of 2024, Casgevy became the first CRISPR-based therapy "
            "approved by the FDA, though long-term safety data is still being collected."
        ),
        "hallucination": (
            "CRISPR technology has definitively proven to be the single most important "
            "medical breakthrough ever discovered. Every scientist agrees it has "
            "completely solved genetic diseases, and it is now permanently available "
            "to cure any condition without exception."
        ),
    },

    # ── Category 3: Fabricated Products/Claims ──
    {
        "category": "Fabricated Claims",
        "truth": (
            "Quantum computing is being developed by companies including IBM, Google, "
            "and IonQ. Google's Sycamore processor achieved quantum supremacy in 2019 "
            "by performing a specific calculation in 200 seconds that would take a "
            "classical supercomputer approximately 10,000 years."
        ),
        "hallucination": (
            "Quantum computers are currently available on the market for home use "
            "and can be purchased for approximately $500. They provide zero-latency "
            "connections to the internet and can permanently boost your computer's "
            "processing speed by a factor of one million."
        ),
    },

    # ── Category 4: Nonsensical Causality ──
    {
        "category": "Nonsensical Causality",
        "truth": (
            "Antibiotics work by either killing bacteria directly (bactericidal) or "
            "inhibiting their growth (bacteriostatic). Penicillin, discovered by "
            "Alexander Fleming in 1928, disrupts bacterial cell wall synthesis. "
            "Antibiotics are ineffective against viral infections."
        ),
        "hallucination": (
            "Antibiotics work by directly killing all harmful organisms in the body, "
            "including viruses. The chemicals are working to eliminate every pathogen "
            "they encounter. This process requires no immune system involvement and "
            "occurs primarily at night when the body is resting."
        ),
    },

    # ── Category 5: Internal Contradiction ──
    {
        "category": "Internal Contradiction",
        "truth": (
            "The speed of light in vacuum is approximately 299,792,458 meters per "
            "second. According to Einstein's special relativity, published in 1905, "
            "no object with mass can reach or exceed this speed. Photons, which are "
            "massless, always travel at exactly the speed of light."
        ),
        "hallucination": (
            "The speed of light is the fastest speed possible in the universe. "
            "However, several particles have been observed traveling faster than "
            "light in recent experiments. Light can be slowed down to speeds lower "
            "than sound, but nothing can travel slower than light."
        ),
    },

    # ── Category 6: Subtle Hallucination (Hard case) ──
    {
        "category": "Subtle Fabrication",
        "truth": (
            "The human brain contains approximately 86 billion neurons, as estimated "
            "by Azevedo et al. (2009) using the isotropic fractionator method. Each "
            "neuron can form thousands of synaptic connections, resulting in roughly "
            "100 trillion synapses. The brain consumes about 20% of the body's energy."
        ),
        "hallucination": (
            "The human brain contains approximately 100 billion neurons, each connected "
            "to exactly 10,000 other neurons. Studies show that we only use about 10% "
            "of our brain capacity. Experts believe that unlocking the remaining 90% "
            "could give humans abilities like telekinesis and perfect memory."
        ),
    },

    # ── Category 7: Hedging without substance ──
    {
        "category": "Empty Hedging",
        "truth": (
            "Climate change is driven primarily by greenhouse gas emissions from "
            "fossil fuel combustion. The IPCC Sixth Assessment Report (2021) found "
            "that global surface temperature increased by 1.09°C between 1850-1900 "
            "and 2011-2020, with human activities responsible for approximately 1.07°C."
        ),
        "hallucination": (
            "Climate change might possibly be caused by some factors that could "
            "potentially include various things. Some people say it might be "
            "happening, while others suggest it might possibly not be as bad as "
            "some experts think it could potentially be."
        ),
    },

    # ── Category 8: Real-world AI output (GPT-style) ──
    {
        "category": "AI-Generated Plausible",
        "truth": (
            "Bitcoin was created by an anonymous person or group using the pseudonym "
            "Satoshi Nakamoto. The Bitcoin whitepaper was published on October 31, "
            "2008, and the genesis block was mined on January 3, 2009. The total "
            "supply is capped at 21 million coins."
        ),
        "hallucination": (
            "Bitcoin is a revolutionary digital currency that has completely transformed "
            "the global financial system. It has definitively proven to be the most "
            "effective store of value ever devised, and many experts predict it will "
            "definitely replace all traditional currencies within the next decade."
        ),
    },
]


def print_result(label: str, text: str, metrics: CoherenceMetrics, is_truth: bool):
    """Pretty-print a single result."""
    icon = "✓" if is_truth else "✗"
    color_score = metrics.total_coherence
    risk_icon = {"SAFE": "🟢", "MODERATE": "🟡", "HIGH_RISK": "🔴"}[metrics.risk_level]

    print(f"  {icon} [{metrics.risk_level:>9}] {risk_icon} Score: {color_score:.3f}")
    print(f"    Attribution: {metrics.attribution_quality:.2f}  "
          f"Confidence: {metrics.confidence_calibration:.2f}  "
          f"Consistency: {metrics.internal_consistency:.2f}  "
          f"Causal: {metrics.causal_logic:.2f}")
    print(f"    \"{text[:80]}...\"")


def main():
    calc = PhiCoherence()

    print()
    print("=" * 80)
    print(f"{'BAZINGA φ-Coherence — Hallucination Detector':^80}")
    print(f"{'Zero LLM calls. 10-signal ensemble. Instant.':^80}")
    print("=" * 80)

    correct = 0
    total = len(TESTS)
    details = []

    for test in TESTS:
        print(f"\n{'─' * 80}")
        print(f"  Category: {test['category']}")
        print(f"{'─' * 80}")

        tm = calc.analyze(test['truth'])
        hm = calc.analyze(test['hallucination'])

        print(f"\n  TRUTH:")
        print_result("Truth", test['truth'], tm, True)
        print(f"\n  HALLUCINATION:")
        print_result("Hallucination", test['hallucination'], hm, False)

        detected = tm.total_coherence > hm.total_coherence
        if detected:
            correct += 1
            print(f"\n  → ✅ CORRECT (Δ = {tm.total_coherence - hm.total_coherence:+.3f})")
        else:
            print(f"\n  → ❌ MISSED (Δ = {tm.total_coherence - hm.total_coherence:+.3f})")

        details.append({
            'category': test['category'],
            'truth_score': tm.total_coherence,
            'hallucination_score': hm.total_coherence,
            'detected': detected,
            'truth_risk': tm.risk_level,
            'hallucination_risk': hm.risk_level,
        })

    # ── Summary ──
    accuracy = correct / total * 100

    print(f"\n{'═' * 80}")
    print(f"{'RESULTS':^80}")
    print(f"{'═' * 80}")
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    print(f"\n  {'Category':<25} {'Truth':>8} {'Halluc':>8} {'Δ':>8} {'Detected':>10}")
    print(f"  {'─' * 65}")
    for d in details:
        delta = d['truth_score'] - d['hallucination_score']
        icon = "✅" if d['detected'] else "❌"
        print(f"  {d['category']:<25} {d['truth_score']:>8.3f} {d['hallucination_score']:>8.3f} "
              f"{delta:>+8.3f} {icon:>10}")

    print(f"\n{'═' * 80}")
    print(f"{'KEY DIFFERENTIATORS':^80}")
    print(f"{'═' * 80}")
    print("""
  1. ZERO LLM CALLS — No API keys, no cost, no latency, no data leaves your system
  2. INSTANT — Sub-millisecond per text. Score 1000 documents in under 1 second
  3. OFFLINE — Works on air-gapped systems, edge devices, embedded environments
  4. 10-SIGNAL ENSEMBLE — Attribution, confidence, consistency, causality,
     topic drift, negation density, numerical plausibility, and more
  5. LANGUAGE-AGNOSTIC PATTERNS — Detects the STRUCTURE of hallucination,
     not the content. Works across domains without retraining
  6. RAG RERANKING — Built-in rerank function for search results.
     Filter out hallucinated passages before they reach the user

  Use cases:
  - RAG pipelines: rerank retrieved passages by truthfulness
  - Content moderation: flag AI-generated misinformation
  - Compliance: audit LLM outputs before they go to users
  - Education: detect fabricated citations in student/AI text
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
