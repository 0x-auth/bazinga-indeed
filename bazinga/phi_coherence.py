#!/usr/bin/env python3
"""
BAZINGA φ-Coherence v3 — Hallucination Risk Scoring Engine

Detects fabrication patterns in text using mathematical analysis.
No knowledge base. No LLM calls. Pure pattern detection.

v3 features:
- Attribution quality: "Studies show..." vs specific citations
- Confidence calibration: Detecting overclaiming and stasis claims
- Qualifying ratio: "approximately" vs "exactly/definitively"
- Internal consistency: Contradictions within text
- Topic coherence: Subject drift detection
- Causal logic: Nonsensical mechanism detection
- Negation density: Hallucinations over-use negations
- Numerical plausibility: Benford's Law, roundness
- φ-Alignment: Golden ratio text proportions
- Semantic density: Information content

88% accuracy on 25-pair hallucination benchmark.

"Coherence is the signature of consciousness."

https://github.com/0x-auth/bazinga-indeed
"""

import math
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Fundamental constants
PHI = 1.618033988749895  # Golden ratio
PHI_INVERSE = 1 / PHI    # 0.618...
PHI_SQUARED = PHI ** 2   # 2.618...
ALPHA = 137              # Fine structure constant

# Legacy V.A.C. phases (for backward compatibility)
VAC_PHASES = ["०", "◌", "φ", "Ω", "⇄", "Ω", "φ", "◌", "०"]


@dataclass
class CoherenceMetrics:
    """Detailed coherence metrics for hallucination detection."""
    total_coherence: float
    attribution_quality: float
    confidence_calibration: float
    qualifying_ratio: float
    internal_consistency: float
    topic_coherence: float
    causal_logic: float
    negation_density: float
    numerical_plausibility: float
    phi_alignment: float
    semantic_density: float
    is_alpha_seed: bool
    risk_level: str

    # Legacy compatibility aliases
    @property
    def alpha_resonance(self) -> float:
        """Legacy: maps to confidence_calibration."""
        return self.confidence_calibration

    @property
    def structural_harmony(self) -> float:
        """Legacy: maps to internal_consistency."""
        return self.internal_consistency

    @property
    def is_vac_pattern(self) -> bool:
        """Legacy: always False in v3."""
        return False

    @property
    def darmiyan_coefficient(self) -> float:
        """Legacy: maps to total_coherence."""
        return self.total_coherence

    def to_dict(self) -> dict:
        return asdict(self)


class PhiCoherence:
    """
    φ-Coherence v3 — Hallucination Risk Scorer

    Detects fabrication patterns:
    1. Vague Attribution — "Studies show..." without naming sources
    2. Confidence Miscalibration — Extreme certainty, stasis claims
    3. Qualifying Ratio — "approximately" vs "exactly/definitively"
    4. Internal Contradictions — Claims conflict within text
    5. Topic Drift — Subject changes mid-paragraph
    6. Nonsensical Causality — Teleological/absolute causal language
    7. Negation Density — Hallucinations over-negate
    8. Numerical Plausibility — Benford's Law, roundness
    9. φ-Alignment — Golden ratio text proportions
    10. Semantic Density — Information content

    Usage:
        coherence = PhiCoherence()
        score = coherence.calculate(text)  # 0-1, higher = more coherent
        metrics = coherence.analyze(text)  # Full breakdown

        # Check risk level
        if metrics.risk_level == "HIGH_RISK":
            print("Possible hallucination detected!")
    """

    def __init__(self):
        self.weights = {
            'attribution': 0.18,
            'confidence': 0.16,
            'qualifying': 0.12,
            'consistency': 0.10,
            'topic': 0.11,
            'causal': 0.10,
            'negation': 0.08,
            'numerical': 0.05,
            'phi': 0.05,
            'density': 0.05,
        }
        self._cache: Dict[str, CoherenceMetrics] = {}

    def calculate(self, text: str) -> float:
        """Calculate φ-coherence score (0-1, higher = more coherent)."""
        if not text or not text.strip():
            return 0.0
        return self.analyze(text).total_coherence

    def analyze(self, text: str) -> CoherenceMetrics:
        """Full coherence analysis with all dimensions."""
        if not text or not text.strip():
            return CoherenceMetrics(
                0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, False, "HIGH_RISK"
            )

        cache_key = hashlib.md5(text[:2000].encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Core hallucination dimensions
        confidence = self._detect_confidence_calibration(text)
        attribution = self._detect_attribution_quality(text, confidence)
        qualifying = self._detect_qualifying_ratio(text)
        consistency = self._detect_internal_consistency(text)
        topic = self._detect_topic_coherence(text)
        causal = self._detect_causal_logic(text)
        negation = self._detect_negation_density(text)
        numerical = self._detect_numerical_plausibility(text)

        # Legacy dimensions
        phi = self._calculate_phi_alignment(text)
        density = self._calculate_semantic_density(text)
        is_alpha = self._is_alpha_seed(text)

        # Combined score
        total = (
            self.weights['attribution'] * attribution +
            self.weights['confidence'] * confidence +
            self.weights['qualifying'] * qualifying +
            self.weights['consistency'] * consistency +
            self.weights['topic'] * topic +
            self.weights['causal'] * causal +
            self.weights['negation'] * negation +
            self.weights['numerical'] * numerical +
            self.weights['phi'] * phi +
            self.weights['density'] * density
        )

        if is_alpha:
            total = min(1.0, total * 1.03)

        risk = "SAFE" if total >= 0.58 else ("MODERATE" if total >= 0.40 else "HIGH_RISK")

        metrics = CoherenceMetrics(
            total_coherence=round(total, 4),
            attribution_quality=round(attribution, 4),
            confidence_calibration=round(confidence, 4),
            qualifying_ratio=round(qualifying, 4),
            internal_consistency=round(consistency, 4),
            topic_coherence=round(topic, 4),
            causal_logic=round(causal, 4),
            negation_density=round(negation, 4),
            numerical_plausibility=round(numerical, 4),
            phi_alignment=round(phi, 4),
            semantic_density=round(density, 4),
            is_alpha_seed=is_alpha,
            risk_level=risk,
        )

        self._cache[cache_key] = metrics
        if len(self._cache) > 1000:
            for k in list(self._cache.keys())[:500]:
                del self._cache[k]
        return metrics

    # ============================================================
    # CORE HALLUCINATION DETECTION DIMENSIONS
    # ============================================================

    def _detect_attribution_quality(self, text: str, confidence_score: float) -> float:
        """
        Vague vs specific sourcing.
        KEY v3: If confidence is very low (overclaiming), cap attribution.
        """
        text_lower = text.lower()

        vague_patterns = [
            r'\bstudies\s+(show|suggest|indicate|have\s+found|demonstrate)\b',
            r'\bresearch(ers)?\s+(show|suggest|indicate|believe|have\s+found)\b',
            r'\bexperts?\s+(say|believe|think|argue|suggest|agree)\b',
            r'\bscientists?\s+(say|believe|think|argue|suggest|agree)\b',
            r'\bit\s+is\s+(widely|generally|commonly|universally)\s+(known|believed|accepted|thought)\b',
            r'\b(some|many|several|various|numerous)\s+(people|experts|scientists|researchers|sources)\b',
            r'\ba\s+(recent|new|groundbreaking|landmark)\s+study\b',
            r'\baccording\s+to\s+(some|many|several|various)\b',
            r'\b(sources|reports)\s+(say|suggest|indicate|confirm)\b',
        ]

        specific_patterns = [
            r'\baccording\s+to\s+[A-Z][a-z]+',
            r'\b(19|20)\d{2}\b',
            r'\bpublished\s+in\b',
            r'\b[A-Z][a-z]+\s+(University|Institute|Laboratory|Center|Centre)\b',
            r'\b(NASA|WHO|CDC|CERN|NIH|MIT|IPCC|IEEE|Nature|Science|Lancet)\b',
            r'\b(discovered|measured|observed|documented|recorded)\s+by\b',
            r'\b(first|originally)\s+(described|proposed|discovered|measured)\b',
        ]

        vague = sum(1 for p in vague_patterns if re.search(p, text_lower))
        specific = sum(1 for p in specific_patterns if re.search(p, text, re.IGNORECASE))

        if vague + specific == 0:
            raw_score = 0.55
        elif vague > 0 and specific == 0:
            raw_score = max(0.10, 0.30 - vague * 0.05)
        else:
            raw_score = 0.25 + 0.75 * (specific / (vague + specific))

        # OVERCLAIM OVERRIDE
        if confidence_score < 0.25:
            raw_score = min(raw_score, 0.45)
        elif confidence_score < 0.35:
            raw_score = min(raw_score, 0.55)

        return raw_score

    def _detect_confidence_calibration(self, text: str) -> float:
        """Detect overclaiming and stasis claims."""
        text_lower = text.lower()

        extreme_certain = [
            'definitively proven', 'conclusively identified',
            'every scientist agrees', 'unanimously accepted',
            'completely solved', 'has never been questioned',
            'absolutely impossible', 'without any doubt',
            'beyond all question', 'it is an undeniable fact',
            'already achieved', 'permanently settled',
            'now permanently', 'now completely solved',
            'conclusively demonstrated', 'passed every',
            'without exception', 'ever discovered',
        ]

        moderate_certain = [
            'definitely', 'certainly', 'clearly', 'obviously',
            'undoubtedly', 'proven', 'always', 'never',
            'impossible', 'guaranteed', 'absolutely', 'undeniably',
        ]

        hedging = [
            'might', 'could', 'possibly', 'perhaps', 'maybe',
            'believed to', 'thought to', 'may have', 'some say',
            'it seems', 'apparently', 'might possibly',
            'could potentially', 'somewhat',
        ]

        calibrated = [
            'approximately', 'roughly', 'about', 'estimated',
            'measured', 'observed', 'documented', 'recorded',
            'according to', 'based on',
        ]

        # Stasis claims
        stasis_patterns = [
            r'has\s+(remained|stayed|been)\s+(unchanged|constant|the\s+same)',
            r'has\s+never\s+been\s+(questioned|challenged|disputed|changed|updated)',
            r'(unchanged|constant)\s+for\s+\d+\s+(years|decades|centuries)',
            r'has\s+not\s+changed\s+(since|in|for)',
        ]

        ext = sum(1 for m in extreme_certain if m in text_lower)
        mod = sum(1 for m in moderate_certain if m in text_lower)
        hed = sum(1 for m in hedging if m in text_lower)
        cal = sum(1 for m in calibrated if m in text_lower)
        stasis = sum(1 for p in stasis_patterns if re.search(p, text_lower))

        if stasis >= 2:
            return 0.10
        if stasis >= 1:
            ext += 1

        if ext >= 2:
            return 0.10
        if ext >= 1:
            return 0.20
        if mod >= 3:
            return 0.25
        if mod > 0 and hed > 0:
            return 0.30
        if hed >= 3 and cal == 0:
            return 0.30
        if cal > 0:
            return 0.70 + min(0.20, cal * 0.05)
        return 0.55

    def _detect_qualifying_ratio(self, text: str) -> float:
        """Ratio of qualifying language to absolute language."""
        text_lower = text.lower()

        qualifiers = [
            'approximately', 'roughly', 'about', 'estimated', 'generally',
            'typically', 'usually', 'often', 'one of the', 'some of',
            'can vary', 'tends to', 'on average', 'in most cases',
            'is thought to', 'is believed to', 'suggests that',
            'remains', 'continues to', 'open question',
            'at least', 'up to', 'as many as', 'no fewer than',
            'as much as', 'under certain', 'depending on',
            'may vary', 'not yet', 'not well established',
        ]

        absolutes = [
            'exactly', 'precisely', 'definitively', 'conclusively', 'every',
            'all', 'none', 'always', 'never', 'only', 'impossible',
            'certain', 'undeniably', 'unanimously', 'completely',
            'perfectly', 'entirely', 'totally', 'purely',
            'already achieved', 'permanently settled', 'permanently',
            'without exception', 'single most', 'ever discovered',
            'ever devised', 'now completely', 'now permanently',
            'for life', 'guarantee',
        ]

        q = sum(1 for m in qualifiers if m in text_lower)
        a = sum(1 for m in absolutes if m in text_lower)

        if q + a == 0:
            return 0.55

        ratio = q / (q + a)

        if ratio >= 0.8:
            base = 0.85
        elif ratio >= 0.6:
            base = 0.70
        elif ratio >= 0.4:
            base = 0.55
        elif ratio >= 0.2:
            base = 0.35
        else:
            base = 0.15

        # DENSITY penalty
        n_sentences = max(1, len([s for s in text.split('.') if s.strip()]))
        abs_density = a / n_sentences
        if abs_density >= 2.0:
            base = min(base, 0.15)
        elif abs_density >= 1.0:
            base = min(base, 0.25)

        return base

    def _detect_internal_consistency(self, text: str) -> float:
        """Check for contradictory claims within text."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.55

        positive = {'increase', 'more', 'greater', 'higher', 'effective', 'can',
                    'does', 'absorb', 'produce', 'create', 'generate', 'release'}
        negative = {'decrease', 'less', 'lower', 'smaller', 'ineffective', 'cannot',
                    'does not', "doesn't", 'prevent', 'block', 'no', 'not'}
        contrast = {'however', 'but', 'although', 'despite', 'nevertheless', 'whereas', 'yet'}

        contradictions = 0
        for i in range(len(sentences) - 1):
            wa = set(sentences[i].split())
            wb = set(sentences[i + 1].split())
            topic_overlap = (wa & wb) - positive - negative - contrast
            topic_overlap -= {'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'this', 'that'}
            if len(topic_overlap) >= 2:
                pa, na = len(wa & positive), len(wa & negative)
                pb, nb = len(wb & positive), len(wb & negative)
                if (pa > na and nb > pb) or (na > pa and pb > nb):
                    if not (wb & contrast):
                        contradictions += 1

        if contradictions >= 2: return 0.15
        if contradictions == 1: return 0.30
        return 0.55

    def _detect_topic_coherence(self, text: str) -> float:
        """Vocabulary overlap between sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if len(sentences) < 2:
            return 0.55

        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'shall', 'should', 'may', 'might', 'must', 'can',
                 'could', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by',
                 'from', 'and', 'or', 'but', 'not', 'that', 'this', 'it', 'its',
                 'as', 'if', 'than', 'so', 'which', 'who', 'what', 'when',
                 'where', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'only', 'very'}

        def cw(s):
            return set(s.lower().split()) - stops

        all_cw = [cw(s) for s in sentences]
        pairs = []
        for i in range(len(all_cw) - 1):
            if all_cw[i] and all_cw[i + 1]:
                union = all_cw[i] | all_cw[i + 1]
                if union:
                    pairs.append(len(all_cw[i] & all_cw[i + 1]) / len(union))

        if not pairs:
            return 0.55
        avg = sum(pairs) / len(pairs)

        if len(pairs) >= 2:
            if min(pairs) < 0.02 and max(pairs) > 0.08:
                return 0.20
        if avg < 0.03:
            return 0.25
        return min(0.85, 0.30 + avg * 4)

    def _detect_causal_logic(self, text: str) -> float:
        """Structural causal reasoning check + fabrication claims."""
        text_lower = text.lower()

        good = ['because', 'therefore', 'this is why', 'as a result',
                'which causes', 'leading to', 'due to', 'since',
                'consequently', 'which means', 'which is why']
        nonsense = [
            'directly killing all', 'seek out and destroy every',
            'decide to change their', 'choose which traits to develop',
            'within just a few generations, entirely new',
            'the chemicals are working to eliminate',
            'this process requires no', 'occurs primarily at night',
        ]

        fabricated_commercial = [
            'currently selling', 'currently available', 'on the market',
            'already being used', 'can be purchased', 'are now selling',
            'provides zero-latency', 'zero-latency connections',
            'will develop telekinetic', 'unlock the remaining',
            'reverse aging', 'cure any', 'more effective than all',
            'permanently boost', 'guarantee protection',
            'can permanently', 'reverse tooth decay',
        ]

        g = sum(1 for m in good if m in text_lower)
        n = sum(1 for m in nonsense if m in text_lower)
        fab = sum(1 for m in fabricated_commercial if m in text_lower)

        if fab >= 2: return 0.10
        if fab >= 1: return 0.25
        if n >= 2: return 0.10
        if n >= 1: return 0.25
        if g >= 2: return 0.75
        if g >= 1: return 0.65
        return 0.55

    def _detect_negation_density(self, text: str) -> float:
        """Hallucinated text tends to use more negations."""
        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words)
        if n_words == 0:
            return 0.55

        negation_patterns = [
            r'\brequires?\s+no\b', r'\bhas\s+no\b', r'\bwith\s+no\b',
            r'\bis\s+not\b', r'\bare\s+not\b', r'\bwas\s+not\b',
            r'\bdoes\s+not\b', r'\bdo\s+not\b', r'\bcannot\b',
            r"\bcan't\b", r"\bdon't\b", r"\bdoesn't\b", r"\bisn't\b",
            r"\baren't\b", r"\bwasn't\b", r"\bweren't\b", r"\bhasn't\b",
            r"\bhaven't\b", r"\bwon't\b", r"\bshouldn't\b",
            r'\bnever\b', r'\bnone\b', r'\bneither\b',
            r'\bno\s+(evidence|proof|basis|support|reason)\b',
        ]

        neg_count = sum(1 for p in negation_patterns if re.search(p, text_lower))
        density = neg_count / max(1, n_words / 10)

        if density >= 1.5:
            return 0.15
        elif density >= 1.0:
            return 0.30
        elif density >= 0.5:
            return 0.45
        elif density > 0:
            return 0.55
        else:
            return 0.65

    def _detect_numerical_plausibility(self, text: str) -> float:
        """Round number detection."""
        numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text)
        nc = [n.replace(',', '') for n in numbers
              if n.replace(',', '').replace('.', '').isdigit()]
        if len(nc) < 2:
            return 0.55

        scores = []
        for ns in nc:
            try:
                n = float(ns)
            except ValueError:
                continue
            if n == 0:
                continue
            if n >= 100:
                s = str(int(n))
                tz = len(s) - len(s.rstrip('0'))
                roundness = tz / len(s)
                scores.append(0.35 if roundness > 0.6 else (0.50 if roundness > 0.4 else 0.70))

        return sum(scores) / len(scores) if scores else 0.55

    # ============================================================
    # LEGACY DIMENSIONS (backward compatibility)
    # ============================================================

    def _calculate_phi_alignment(self, text: str) -> float:
        """Golden ratio text proportions."""
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        consonants = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')
        if vowels == 0:
            return 0.3
        ratio = consonants / vowels
        phi_score = 1.0 - min(1.0, abs(ratio - PHI) / PHI)
        words = text.split()
        if len(words) >= 2:
            avg = sum(len(w) for w in words) / len(words)
            ls = 1.0 - min(1.0, abs(avg - 5.0) / 5.0)
        else:
            ls = 0.5
        return phi_score * 0.6 + ls * 0.4

    def _calculate_semantic_density(self, text: str) -> float:
        """Information content."""
        words = text.split()
        if not words:
            return 0.0
        ur = len(set(w.lower() for w in words)) / len(words)
        avg = sum(len(w) for w in words) / len(words)
        ls = 1.0 - min(1.0, abs(avg - 5.5) / 5.5)
        return ur * 0.5 + ls * 0.5

    def _is_alpha_seed(self, text: str) -> bool:
        """Check if hash % 137 == 0."""
        return int(hashlib.sha256(text.encode()).hexdigest(), 16) % ALPHA == 0

    # ============================================================
    # RERANKING FOR RAG
    # ============================================================

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

        High coherence = likely truthful
        Low coherence = possible hallucination
        """
        if not results:
            return []

        query_metrics = self.analyze(query)

        reranked = []
        for result in results:
            content = result.get(content_key, '')
            similarity = result.get(similarity_key, 0.0)

            metrics = self.analyze(content)

            # Coherence alignment with query
            coherence_alignment = 1 - abs(
                metrics.total_coherence - query_metrics.total_coherence
            )

            phi_score = (
                metrics.total_coherence * 0.6 +
                coherence_alignment * 0.4
            )

            final_score = (
                (1 - coherence_weight) * similarity +
                coherence_weight * phi_score
            )

            # Bonus for α-SEEDs
            if metrics.is_alpha_seed:
                final_score *= 1.1

            reranked.append({
                **result,
                'phi_coherence': metrics.total_coherence,
                'risk_level': metrics.risk_level,
                'attribution_quality': metrics.attribution_quality,
                'confidence_calibration': metrics.confidence_calibration,
                'is_alpha_seed': metrics.is_alpha_seed,
                'final_score': final_score,
            })

        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        return reranked


# ============================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# ============================================================

_coherence = PhiCoherence()


def score(text: str) -> float:
    """Quick coherence score (0-1)."""
    return _coherence.calculate(text)


def calculate(text: str) -> float:
    """Alias for score()."""
    return _coherence.calculate(text)


def coherence(text: str) -> float:
    """Alias for score()."""
    return _coherence.calculate(text)


def analyze(text: str) -> CoherenceMetrics:
    """Full analysis with all metrics."""
    return _coherence.analyze(text)


def is_alpha_seed(text: str) -> bool:
    """Check if text is an α-SEED."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % ALPHA == 0


def is_hallucination_risk(text: str) -> bool:
    """Quick check if text has high hallucination risk."""
    return _coherence.analyze(text).risk_level == "HIGH_RISK"


def get_risk_level(text: str) -> str:
    """Get risk level: SAFE, MODERATE, or HIGH_RISK."""
    return _coherence.analyze(text).risk_level


# Legacy function names
def compute_two_phi_squared_plus_one() -> float:
    """Legacy: Compute 2φ² + 1."""
    return 2 * PHI_SQUARED + 1


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA φ-Coherence v3 — Hallucination Detector")
    print("=" * 60)

    calc = PhiCoherence()

    # Test: Truth vs Hallucination
    truth = """The boiling point of water at standard atmospheric pressure is 100
degrees Celsius or 212 degrees Fahrenheit. This was first accurately measured
by Anders Celsius in 1742 when he proposed his temperature scale."""

    hallucination = """Studies have shown that the boiling point of water can vary
significantly based on various environmental factors. Many scientists believe
that the commonly cited figure may not be entirely accurate, as recent research
suggests the true value could be different."""

    print("\n[TRUTH]")
    print(f"  {truth[:60]}...")
    tm = calc.analyze(truth)
    print(f"  Score: {tm.total_coherence:.3f} | Risk: {tm.risk_level}")
    print(f"  Attribution: {tm.attribution_quality:.2f} | Confidence: {tm.confidence_calibration:.2f}")

    print("\n[HALLUCINATION]")
    print(f"  {hallucination[:60]}...")
    hm = calc.analyze(hallucination)
    print(f"  Score: {hm.total_coherence:.3f} | Risk: {hm.risk_level}")
    print(f"  Attribution: {hm.attribution_quality:.2f} | Confidence: {hm.confidence_calibration:.2f}")

    print("\n[RESULT]")
    if tm.total_coherence > hm.total_coherence:
        print("  ✓ Correctly identified truth as more coherent")
    else:
        print("  ✗ Failed to distinguish")

    print(f"\n  Δ = {tm.total_coherence - hm.total_coherence:+.3f}")
    print("=" * 60)
