#!/usr/bin/env python3
"""
BAZINGA Model Router - Intelligent Query Routing to Specialized Nodes

Routes queries to the most appropriate nodes based on:
- Domain expertise (medical, code, general)
- phi-coherence matching
- Trust scores (tau)
- Load balancing

Architecture:
    Query → Classifier → Domain Expert Selection → Response

"The right model for the right question."
"""

import json
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


class Domain(Enum):
    """Knowledge domains for specialization."""
    GENERAL = "general"
    CODE = "code"
    MEDICAL = "medical"
    LEGAL = "legal"
    SCIENCE = "science"
    CREATIVE = "creative"
    MATH = "math"
    PHILOSOPHY = "philosophy"


# Domain keywords for classification
DOMAIN_KEYWORDS = {
    Domain.CODE: [
        'code', 'function', 'class', 'variable', 'python', 'javascript',
        'programming', 'debug', 'error', 'syntax', 'api', 'algorithm',
        'compile', 'runtime', 'database', 'sql', 'git', 'docker', 'import',
        'def', 'return', 'loop', 'array', 'list', 'dict', 'json', 'http',
    ],
    Domain.MEDICAL: [
        'symptom', 'diagnosis', 'treatment', 'medicine', 'doctor', 'patient',
        'disease', 'hospital', 'surgery', 'prescription', 'health', 'clinical',
        'therapy', 'vaccine', 'virus', 'bacteria', 'anatomy', 'physiology',
        'cardio', 'neuro', 'oncology', 'pharmaceutical', 'dosage',
    ],
    Domain.LEGAL: [
        'law', 'legal', 'court', 'judge', 'attorney', 'lawsuit', 'contract',
        'liability', 'regulation', 'statute', 'precedent', 'jurisdiction',
        'defendant', 'plaintiff', 'verdict', 'appeal', 'constitution',
    ],
    Domain.SCIENCE: [
        'experiment', 'hypothesis', 'theory', 'physics', 'chemistry', 'biology',
        'research', 'data', 'study', 'analysis', 'molecule', 'atom', 'quantum',
        'evolution', 'genetics', 'ecology', 'astronomy', 'geology',
    ],
    Domain.CREATIVE: [
        'story', 'poem', 'write', 'creative', 'fiction', 'character', 'plot',
        'narrative', 'imagine', 'describe', 'art', 'music', 'compose',
        'design', 'aesthetic', 'metaphor', 'symbolism',
    ],
    Domain.MATH: [
        'calculate', 'equation', 'formula', 'prove', 'theorem', 'algebra',
        'calculus', 'geometry', 'statistics', 'probability', 'integral',
        'derivative', 'matrix', 'vector', 'polynomial', 'factorial',
    ],
    Domain.PHILOSOPHY: [
        'consciousness', 'existence', 'meaning', 'ethics', 'morality',
        'philosophy', 'metaphysics', 'epistemology', 'reality', 'truth',
        'wisdom', 'soul', 'mind', 'free will', 'determinism', 'phi',
    ],
}


@dataclass
class DomainExpert:
    """A node specialized in a particular domain."""
    node_id: str
    domain: Domain
    expertise_score: float = 0.5  # How good at this domain (0-1)
    tau_score: float = 0.5
    model_id: str = ""
    phi_coherence: float = 0.5
    total_queries_handled: int = 0
    avg_response_quality: float = 0.5

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['domain'] = self.domain.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'DomainExpert':
        data['domain'] = Domain(data['domain'])
        return cls(**data)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    query: str
    detected_domain: Domain
    domain_confidence: float
    selected_experts: List[str]
    reasoning: str
    phi_score: float = 0.0


class DomainClassifier:
    """
    Classifies queries into knowledge domains.

    Uses keyword matching with phi-weighted scoring.
    """

    def __init__(self):
        # Pre-compile keyword sets
        self.domain_keywords = {
            domain: set(kw.lower() for kw in keywords)
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }

    def classify(self, query: str) -> Tuple[Domain, float]:
        """
        Classify a query into a domain.

        Args:
            query: User query

        Returns:
            (domain, confidence)
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        # Score each domain
        scores: Dict[Domain, float] = {}

        for domain, keywords in self.domain_keywords.items():
            matches = query_words & keywords

            # Base score from matches
            if keywords:
                match_ratio = len(matches) / len(keywords)
            else:
                match_ratio = 0

            # Bonus for multiple matches
            multi_match_bonus = min(1.0, len(matches) * 0.1)

            # phi-weighted score
            scores[domain] = (match_ratio + multi_match_bonus) * PHI / (1 + PHI)

        # Find best domain
        if not scores or max(scores.values()) == 0:
            return Domain.GENERAL, 0.5

        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain]

        # Normalize confidence
        confidence = min(1.0, confidence * 2)  # Scale up

        return best_domain, confidence

    def get_all_scores(self, query: str) -> Dict[Domain, float]:
        """Get scores for all domains."""
        scores = {}
        for domain in Domain:
            _, conf = self.classify(query)
            if domain in self.domain_keywords:
                scores[domain] = conf
            else:
                scores[domain] = 0.0
        return scores


class ModelRouter:
    """
    Intelligent router for distributed BAZINGA network.

    Routes queries to specialized domain experts based on:
    - Query classification
    - Expert availability and load
    - Trust scores (tau)
    - Historical performance

    Usage:
        router = ModelRouter()
        router.register_expert(expert)

        decision = router.route("How do I fix this Python error?")
        # decision.selected_experts = ["code_expert_001", ...]
    """

    def __init__(
        self,
        ensemble_size: int = 3,
        min_tau: float = 0.3,
        min_expertise: float = 0.4,
    ):
        """
        Initialize router.

        Args:
            ensemble_size: Number of experts to select for ensemble
            min_tau: Minimum tau score to consider
            min_expertise: Minimum expertise score
        """
        self.ensemble_size = ensemble_size
        self.min_tau = min_tau
        self.min_expertise = min_expertise

        # Domain experts
        self.experts: Dict[str, DomainExpert] = {}
        self.domain_experts: Dict[Domain, List[str]] = defaultdict(list)

        # Classifier
        self.classifier = DomainClassifier()

        # Stats
        self.total_routes = 0
        self.domain_counts: Dict[Domain, int] = defaultdict(int)

        # Load tracking
        self.expert_load: Dict[str, int] = defaultdict(int)

    def register_expert(self, expert: DomainExpert):
        """Register a domain expert."""
        self.experts[expert.node_id] = expert
        self.domain_experts[expert.domain].append(expert.node_id)
        print(f"Registered {expert.domain.value} expert: {expert.node_id[:8]}")

    def unregister_expert(self, node_id: str):
        """Remove an expert."""
        expert = self.experts.pop(node_id, None)
        if expert:
            self.domain_experts[expert.domain].remove(node_id)

    def route(
        self,
        query: str,
        preferred_domain: Optional[Domain] = None,
    ) -> RoutingDecision:
        """
        Route a query to appropriate experts.

        Args:
            query: User query
            preferred_domain: Override domain classification

        Returns:
            RoutingDecision with selected experts
        """
        self.total_routes += 1

        # Classify query
        if preferred_domain:
            domain = preferred_domain
            confidence = 1.0
        else:
            domain, confidence = self.classifier.classify(query)

        self.domain_counts[domain] += 1

        # Find experts for this domain
        candidates = self._find_candidates(domain)

        # Fallback to general if no specialists
        if not candidates and domain != Domain.GENERAL:
            candidates = self._find_candidates(Domain.GENERAL)
            reasoning = f"No {domain.value} experts, falling back to general"
        else:
            reasoning = f"Found {len(candidates)} {domain.value} experts"

        # Select best experts
        selected = self._select_experts(candidates, query)

        # Compute phi score for routing decision
        phi_score = self._compute_routing_phi(domain, confidence, len(selected))

        return RoutingDecision(
            query=query,
            detected_domain=domain,
            domain_confidence=confidence,
            selected_experts=selected,
            reasoning=reasoning,
            phi_score=phi_score,
        )

    def _find_candidates(self, domain: Domain) -> List[DomainExpert]:
        """Find candidate experts for a domain."""
        candidates = []

        for node_id in self.domain_experts.get(domain, []):
            expert = self.experts.get(node_id)
            if not expert:
                continue

            # Filter by tau and expertise
            if expert.tau_score < self.min_tau:
                continue
            if expert.expertise_score < self.min_expertise:
                continue

            candidates.append(expert)

        return candidates

    def _select_experts(
        self,
        candidates: List[DomainExpert],
        query: str,
    ) -> List[str]:
        """Select best experts from candidates."""
        if not candidates:
            return []

        # Score each candidate
        scored = []
        for expert in candidates:
            # Composite score
            score = self._compute_expert_score(expert)
            scored.append((expert.node_id, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top N
        selected = [node_id for node_id, _ in scored[:self.ensemble_size]]

        # Update load
        for node_id in selected:
            self.expert_load[node_id] += 1

        return selected

    def _compute_expert_score(self, expert: DomainExpert) -> float:
        """
        Compute composite score for expert selection.

        Score = (tau * expertise * phi_coherence) / (1 + load)
        """
        load = self.expert_load.get(expert.node_id, 0)

        # Weighted combination
        score = (
            expert.tau_score * 0.3 +
            expert.expertise_score * 0.3 +
            expert.phi_coherence * 0.2 +
            expert.avg_response_quality * 0.2
        )

        # Penalize load
        score = score / (1 + load * 0.1)

        # phi-scale
        score *= PHI / (1 + PHI)

        return score

    def _compute_routing_phi(
        self,
        domain: Domain,
        confidence: float,
        num_experts: int,
    ) -> float:
        """Compute phi-coherence of routing decision."""
        # Ideal: confidence close to phi ratio
        ideal_confidence = PHI / (1 + PHI)
        confidence_alignment = 1 - abs(confidence - ideal_confidence)

        # Ideal: ensemble size follows phi
        ideal_ensemble = int(PHI + 1)  # ~3
        ensemble_alignment = 1 - abs(num_experts - ideal_ensemble) / ideal_ensemble

        return (confidence_alignment + ensemble_alignment) / 2

    def complete_request(self, node_id: str, quality_score: float):
        """
        Mark a request as complete and update stats.

        Args:
            node_id: Expert that handled request
            quality_score: Quality of response (0-1)
        """
        # Update load
        if self.expert_load.get(node_id, 0) > 0:
            self.expert_load[node_id] -= 1

        # Update expert stats
        expert = self.experts.get(node_id)
        if expert:
            expert.total_queries_handled += 1
            # Running average of quality
            alpha = 0.1
            expert.avg_response_quality = (
                alpha * quality_score +
                (1 - alpha) * expert.avg_response_quality
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'total_routes': self.total_routes,
            'total_experts': len(self.experts),
            'domain_counts': {d.value: c for d, c in self.domain_counts.items()},
            'experts_by_domain': {
                d.value: len(nodes)
                for d, nodes in self.domain_experts.items()
            },
            'current_load': dict(self.expert_load),
        }

    def get_domain_experts(self, domain: Domain) -> List[DomainExpert]:
        """Get all experts for a domain."""
        return [
            self.experts[node_id]
            for node_id in self.domain_experts.get(domain, [])
            if node_id in self.experts
        ]


class EnsembleAggregator:
    """
    Aggregates responses from multiple experts.

    Uses phi-weighted voting based on:
    - Expert tau scores
    - Response phi-coherence
    - Historical quality
    """

    def __init__(self):
        self.aggregation_count = 0

    def aggregate(
        self,
        responses: List[Tuple[str, str, float]],  # (node_id, response, tau)
    ) -> Tuple[str, float]:
        """
        Aggregate multiple expert responses.

        Args:
            responses: List of (node_id, response_text, tau_score)

        Returns:
            (best_response, confidence)
        """
        if not responses:
            return "", 0.0

        if len(responses) == 1:
            return responses[0][1], responses[0][2]

        self.aggregation_count += 1

        # Score each response
        scored = []
        for node_id, response, tau in responses:
            # Compute phi-coherence of response
            coherence = self._compute_response_coherence(response)

            # Combined score
            score = tau * 0.4 + coherence * 0.4 + len(response.split()) / 500 * 0.2
            scored.append((response, score))

        # Select best
        scored.sort(key=lambda x: x[1], reverse=True)
        best_response, best_score = scored[0]

        # Confidence based on agreement
        if len(scored) >= 2:
            agreement = self._compute_agreement(responses)
            confidence = best_score * (0.5 + 0.5 * agreement)
        else:
            confidence = best_score

        return best_response, confidence

    def _compute_response_coherence(self, response: str) -> float:
        """Compute phi-coherence of a response."""
        if not response:
            return 0.0

        # Simple heuristic based on structure
        sentences = response.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Check sentence length ratios
        lengths = [len(s.split()) for s in sentences]
        ratios = []
        for i in range(len(lengths) - 1):
            if lengths[i + 1] > 0:
                ratios.append(lengths[i] / lengths[i + 1])

        if not ratios:
            return 0.5

        # Score proximity to phi
        phi_scores = [1 - min(1, abs(r - PHI) / PHI) for r in ratios]
        return sum(phi_scores) / len(phi_scores)

    def _compute_agreement(
        self,
        responses: List[Tuple[str, str, float]],
    ) -> float:
        """Compute agreement between responses."""
        if len(responses) < 2:
            return 1.0

        # Simple word overlap metric
        word_sets = [
            set(r[1].lower().split())
            for r in responses
        ]

        # Average pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        if not similarities:
            return 0.5

        return sum(similarities) / len(similarities)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Model Router Test")
    print("=" * 60)

    # Test classifier
    print("\n1. Domain Classification:")
    classifier = DomainClassifier()

    test_queries = [
        "How do I fix this Python syntax error?",
        "What are the symptoms of diabetes?",
        "Write a poem about the ocean",
        "Calculate the integral of x^2",
        "What is the meaning of consciousness?",
        "Tell me about the weather today",
    ]

    for query in test_queries:
        domain, conf = classifier.classify(query)
        print(f"  '{query[:40]}...' → {domain.value} ({conf:.2f})")

    # Test router
    print("\n2. Model Router:")
    router = ModelRouter()

    # Register experts
    experts = [
        DomainExpert("code_001", Domain.CODE, expertise_score=0.9, tau_score=0.8),
        DomainExpert("code_002", Domain.CODE, expertise_score=0.7, tau_score=0.6),
        DomainExpert("med_001", Domain.MEDICAL, expertise_score=0.85, tau_score=0.75),
        DomainExpert("general_001", Domain.GENERAL, expertise_score=0.6, tau_score=0.7),
        DomainExpert("phil_001", Domain.PHILOSOPHY, expertise_score=0.8, tau_score=0.65),
    ]

    for expert in experts:
        router.register_expert(expert)

    # Route queries
    for query in test_queries:
        decision = router.route(query)
        print(f"  Query: '{query[:30]}...'")
        print(f"    Domain: {decision.detected_domain.value} ({decision.domain_confidence:.2f})")
        print(f"    Experts: {decision.selected_experts}")
        print(f"    phi-score: {decision.phi_score:.3f}")

    # Test ensemble
    print("\n3. Ensemble Aggregation:")
    aggregator = EnsembleAggregator()

    responses = [
        ("expert_1", "The answer is 42, which represents the ultimate answer.", 0.8),
        ("expert_2", "42 is the answer to everything in the universe.", 0.7),
        ("expert_3", "I believe the answer might be 42.", 0.6),
    ]

    best, confidence = aggregator.aggregate(responses)
    print(f"  Best response: '{best[:50]}...'")
    print(f"  Confidence: {confidence:.3f}")

    # Stats
    print(f"\n4. Router Stats: {router.get_stats()}")

    print("\n" + "=" * 60)
    print("Model Router module ready!")
    print("=" * 60)
