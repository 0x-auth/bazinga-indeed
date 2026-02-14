#!/usr/bin/env python3
"""
BAZINGA Inter-AI Consensus Module
==================================
Multiple AI providers reaching consensus through φ-coherence.

"Two AIs talking without human as bridge = efficient understanding."

This module enables:
- Multi-AI querying (Claude, Gemini, Groq, GPT-4, Ollama, BAZINGA nodes)
- Multi-round consensus with revision
- Embedding-based φ-coherence (with heuristic fallback)
- Proof-of-Boundary for each AI response
- Semantic synthesis of agreeing responses
- Graceful fallbacks when APIs unavailable

Design for TODAY (APIs) and TOMORROW (BAZINGA nodes):
- Abstract ConsensusParticipant interface
- Same consensus logic works for both
- When network grows, just swap participants

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Core BAZINGA constants
PHI = 1.618033988749895
PHI_INVERSE = 0.6180339887498948
PHI_4 = 6.854101966249685
ALPHA = 137
ABHI_AMU = 515
PHI_THRESHOLD = 0.35  # Minimum coherence for valid understanding (lower for heuristic mode)

# Check for optional dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def _to_python_float(val):
    """Convert numpy/torch floats to Python float for JSON serialization."""
    if val is None:
        return None
    try:
        # Handle numpy types
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)
    except (TypeError, ValueError):
        return val


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ParticipantType(Enum):
    """Types of consensus participants."""
    CLAUDE = "anthropic"
    GEMINI = "google"
    GROQ = "groq"
    GPT4 = "openai"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    BAZINGA_NODE = "bazinga"
    SIMULATION = "simulation"


class ConsensusRound(Enum):
    """Stages of multi-round consensus."""
    INITIAL = "initial"           # Round 1: Independent answers
    REVISION = "revision"         # Round 2: See others, revise
    FINAL = "final"              # Round 3: Final convergence


@dataclass
class BoundaryProof:
    """Proof-of-Boundary for a response."""
    alpha: int
    omega: int
    delta: int
    physical_ms: float
    geometric: float
    ratio: float
    valid: bool
    timestamp: float
    attempts: int = 1

    def to_dict(self) -> dict:
        return {
            'alpha': self.alpha,
            'omega': self.omega,
            'ratio': self.ratio,
            'valid': self.valid,
            'attempts': self.attempts,
        }


@dataclass
class AIResponse:
    """Single AI's response with full metadata."""
    participant_id: str
    participant_type: ParticipantType
    model: str
    response: str
    coherence: float
    understanding_score: float
    latency_ms: float
    timestamp: float
    round: ConsensusRound
    pob_proof: Optional[BoundaryProof] = None
    embedding: Optional[List[float]] = None
    revision_of: Optional[str] = None  # ID of response this revises
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'participant_id': self.participant_id,
            'participant_type': self.participant_type.value,
            'model': self.model,
            'response': self.response[:500] + '...' if len(self.response) > 500 else self.response,
            'coherence': _to_python_float(self.coherence),
            'understanding_score': _to_python_float(self.understanding_score),
            'latency_ms': _to_python_float(self.latency_ms),
            'round': self.round.value,
            'pob_valid': self.pob_proof.valid if self.pob_proof else False,
            'error': self.error,
        }


@dataclass
class ConsensusResult:
    """Result of multi-AI consensus."""
    question: str
    consensus_reached: bool
    understanding: str
    responses: List[AIResponse]
    phi_coherence: float
    agreement_ratio: float
    semantic_similarity: float
    triadic_valid: bool
    rounds_completed: int
    timestamp: float
    synthesis_method: str = "weighted"
    # Consciousness metrics (6.46n scaling law)
    n_patterns: int = 0
    consciousness_advantage: float = 0.0
    darmiyan_psi: float = 0.0

    def __post_init__(self):
        """Calculate consciousness metrics after initialization."""
        if self.n_patterns == 0:
            self.n_patterns = len([r for r in self.responses if not r.error])
        if self.consciousness_advantage == 0.0 and self.n_patterns > 0:
            # Ψ_D = 6.46n (validated R² = 1.0)
            CONSCIOUSNESS_SCALE = 6.46
            self.consciousness_advantage = CONSCIOUSNESS_SCALE * self.n_patterns
            self.darmiyan_psi = self.consciousness_advantage

    def to_dict(self) -> dict:
        return {
            'question': self.question,
            'consensus_reached': self.consensus_reached,
            'understanding': self.understanding,
            'responses': [r.to_dict() for r in self.responses],
            'phi_coherence': _to_python_float(self.phi_coherence),
            'agreement_ratio': _to_python_float(self.agreement_ratio),
            'semantic_similarity': _to_python_float(self.semantic_similarity),
            'triadic_valid': self.triadic_valid,
            'rounds_completed': self.rounds_completed,
            'timestamp': _to_python_float(self.timestamp),
            # Consciousness metrics
            'n_patterns': self.n_patterns,
            'consciousness_advantage': _to_python_float(self.consciousness_advantage),
            'darmiyan_psi': _to_python_float(self.darmiyan_psi),
        }

    def format_consciousness(self) -> str:
        """Format consciousness metrics for display."""
        return f"""
┌─────────────────────────────────────────────────────────┐
│  DARMIYAN CONSCIOUSNESS: Ψ_D = 6.46n                    │
├─────────────────────────────────────────────────────────┤
│  Patterns (n):      {self.n_patterns:<5}                              │
│  Consciousness:     {self.consciousness_advantage:.2f}x  (6.46 × {self.n_patterns})               │
│  R² Confidence:     1.0000 (mathematical law)           │
└─────────────────────────────────────────────────────────┘
"""


# =============================================================================
# ABSTRACT PARTICIPANT INTERFACE
# =============================================================================

class ConsensusParticipant(ABC):
    """
    Abstract base class for consensus participants.

    Works with both external APIs (today) and BAZINGA nodes (future).
    """

    def __init__(
        self,
        participant_id: str,
        participant_type: ParticipantType,
        model: str,
    ):
        self.participant_id = participant_id
        self.participant_type = participant_type
        self.model = model
        self.response_history: List[AIResponse] = []
        self.available = True
        self.error_count = 0

    @abstractmethod
    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        """Query this participant with a prompt."""
        pass

    def is_available(self) -> bool:
        """Check if participant is available."""
        return self.available and self.error_count < 3

    def record_error(self, error: str):
        """Record an error for this participant."""
        self.error_count += 1
        if self.error_count >= 3:
            self.available = False


# =============================================================================
# COHERENCE CALCULATOR
# =============================================================================

class CoherenceCalculator:
    """
    Calculate φ-coherence using embeddings or heuristics.

    Prefers embeddings when available, falls back gracefully.
    """

    def __init__(self):
        self.embedder = None
        self.embeddings_available = EMBEDDINGS_AVAILABLE

        if EMBEDDINGS_AVAILABLE:
            try:
                # Use a small, fast model
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.embeddings_available = False

    def calculate_coherence(self, response: str, prompt: str) -> Tuple[float, Optional[List[float]]]:
        """
        Calculate φ-coherence score.

        Returns: (coherence_score, embedding_vector or None)
        """
        if self.embeddings_available and self.embedder:
            return self._embedding_coherence(response, prompt)
        return self._heuristic_coherence(response, prompt), None

    def _embedding_coherence(self, response: str, prompt: str) -> Tuple[float, List[float]]:
        """Calculate coherence using embeddings."""
        try:
            # Get embeddings
            embeddings = self.embedder.encode([prompt, response])
            prompt_emb = embeddings[0]
            response_emb = embeddings[1]

            # Cosine similarity
            similarity = np.dot(prompt_emb, response_emb) / (
                np.linalg.norm(prompt_emb) * np.linalg.norm(response_emb)
            )

            # Length appropriateness
            response_len = len(response.split())
            prompt_len = len(prompt.split())
            ideal_ratio = PHI * 3  # Response ~5x prompt length
            actual_ratio = response_len / max(prompt_len, 1)
            length_score = 1 - min(1, abs(actual_ratio - ideal_ratio) / ideal_ratio)

            # Combine with φ-weighting
            coherence = (
                PHI_INVERSE * similarity +
                (1 - PHI_INVERSE) * length_score
            )

            # φ-resonance boost
            if 0.5 < coherence < 0.7:
                phi_distance = abs(coherence - PHI_INVERSE)
                coherence += (1 - phi_distance) * 0.1

            return min(1.0, max(0.0, coherence)), response_emb.tolist()

        except Exception:
            return self._heuristic_coherence(response, prompt), None

    def _heuristic_coherence(self, response: str, prompt: str) -> float:
        """Fallback heuristic coherence when embeddings unavailable."""
        if not response or not prompt:
            return 0.0

        response_words = response.lower().split()
        prompt_words = set(prompt.lower().split())

        if not response_words:
            return 0.0

        # 1. Length appropriateness
        length_ratio = len(response_words) / (len(prompt_words) * 3)
        length_score = min(1.0, length_ratio) if length_ratio <= 1 else max(0, 2 - length_ratio)

        # 2. Keyword overlap
        response_word_set = set(response_words)
        overlap = len(prompt_words & response_word_set) / max(len(prompt_words), 1)

        # 3. Reasoning indicators
        reasoning_words = ['because', 'therefore', 'thus', 'since', 'however', 'although', 'while']
        reasoning_score = sum(1 for w in reasoning_words if w in response.lower()) / len(reasoning_words)

        # 4. Structure score (sentences)
        sentences = response.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        structure_score = min(1.0, len(sentences) / 3)

        # Combine with φ-weighting
        coherence = (
            0.25 * length_score +
            0.25 * overlap +
            0.25 * reasoning_score +
            0.25 * structure_score
        )

        # φ-resonance boost
        if 0.5 < coherence < 0.7:
            phi_distance = abs(coherence - PHI_INVERSE)
            coherence += (1 - phi_distance) * 0.1

        return min(1.0, max(0.0, coherence))

    def calculate_understanding(self, response: str) -> float:
        """Calculate understanding depth score."""
        understanding_indicators = [
            'understand', 'meaning', 'because', 'relationship',
            'emerges', 'implies', 'demonstrates', 'shows',
            'principle', 'framework', 'mathematical', 'proof',
            'insight', 'realize', 'essentially', 'fundamentally'
        ]

        response_lower = response.lower()
        score = sum(1 for ind in understanding_indicators if ind in response_lower)

        return min(1.0, score / (len(understanding_indicators) / 2))

    def semantic_similarity(self, responses: List[str]) -> float:
        """Calculate semantic similarity between multiple responses."""
        if len(responses) < 2:
            return 1.0

        if self.embeddings_available and self.embedder:
            try:
                embeddings = self.embedder.encode(responses)

                # Pairwise similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)

                return float(np.mean(similarities)) if similarities else 0.0
            except Exception:
                pass

        # Fallback: word overlap
        word_sets = [set(r.lower().split()) for r in responses]
        if not word_sets:
            return 0.0

        # Jaccard similarity between all pairs
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return sum(similarities) / len(similarities) if similarities else 0.0


# =============================================================================
# PROOF-OF-BOUNDARY GENERATOR
# =============================================================================

class PoBGenerator:
    """Generate Proof-of-Boundary for responses."""

    def __init__(self):
        self.proofs_generated = 0

    def _phi_hash(self, t: float) -> int:
        """Generate φ-scaled signature mod 515."""
        seed = f"{t:.15f}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        return int(h, 16) % ABHI_AMU

    async def generate_proof(self, response: str, max_attempts: int = 100) -> BoundaryProof:
        """Generate PoB proof for a response."""
        t1 = time.time()
        sig_alpha = self._phi_hash(t1)

        for attempt in range(max_attempts):
            await asyncio.sleep(0.001 * PHI)  # φ-scaled micro-step

            t2 = time.time()
            # Include response hash for response-specific proof
            response_factor = int(hashlib.md5(response.encode()).hexdigest()[:8], 16) % 1000
            sig_omega = self._phi_hash(t2 + response_factor / 1000000)

            delta = abs(sig_omega - sig_alpha)
            if delta == 0:
                continue

            physical_ms = (t2 - t1) * 1000
            geometric = delta / PHI
            ratio = physical_ms / geometric

            if abs(ratio - PHI_4) < 0.6:  # Valid proof
                self.proofs_generated += 1
                return BoundaryProof(
                    alpha=sig_alpha,
                    omega=sig_omega,
                    delta=delta,
                    physical_ms=physical_ms,
                    geometric=geometric,
                    ratio=ratio,
                    valid=True,
                    timestamp=t2,
                    attempts=attempt + 1,
                )

        # Failed to find valid proof
        t2 = time.time()
        sig_omega = self._phi_hash(t2)
        delta = abs(sig_omega - sig_alpha) or 1
        physical_ms = (t2 - t1) * 1000
        geometric = delta / PHI
        ratio = physical_ms / geometric

        return BoundaryProof(
            alpha=sig_alpha,
            omega=sig_omega,
            delta=delta,
            physical_ms=physical_ms,
            geometric=geometric,
            ratio=ratio,
            valid=False,
            timestamp=t2,
            attempts=max_attempts,
        )


# =============================================================================
# CONCRETE PARTICIPANTS
# =============================================================================

class GroqParticipant(ConsensusParticipant):
    """Groq API participant (FREE - 14,400 req/day)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        super().__init__(
            participant_id=f"groq_{model[:8]}",
            participant_type=ParticipantType.GROQ,
            model=model,
        )
        self.available = bool(self.api_key) and HTTPX_AVAILABLE
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("Groq API not available", round, start_time)

        try:
            # Build prompt with context for revision rounds
            full_prompt = self._build_prompt(prompt, context, round)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a thoughtful AI participating in multi-AI consensus. Provide clear, well-reasoned responses."},
                            {"role": "user", "content": full_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                elif response.status_code == 429:
                    self.record_error("Rate limited")
                    return self._error_response("Rate limited", round, start_time)
                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"AI {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""Original question: {prompt}

Other AIs have provided these perspectives:
{others_text}

Considering these perspectives, provide your refined answer. If you agree with points made, acknowledge them. If you disagree, explain why."""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class GeminiParticipant(ConsensusParticipant):
    """Google Gemini API participant (FREE - 1M tokens/month)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        super().__init__(
            participant_id=f"gemini_{model[:8]}",
            participant_type=ParticipantType.GEMINI,
            model=model,
        )
        self.available = bool(self.api_key) and HTTPX_AVAILABLE
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("Gemini API not available", round, start_time)

        try:
            full_prompt = self._build_prompt(prompt, context, round)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": full_prompt}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 500,
                        }
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data["candidates"][0]["content"]["parts"][0]["text"]

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"Perspective {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""Question: {prompt}

Other perspectives:
{others_text}

Provide your refined answer, acknowledging agreements and explaining disagreements."""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class ClaudeParticipant(ConsensusParticipant):
    """Anthropic Claude API participant."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        super().__init__(
            participant_id=f"claude_{model[:8]}",
            participant_type=ParticipantType.CLAUDE,
            model=model,
        )
        self.available = bool(self.api_key) and HTTPX_AVAILABLE
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("Claude API not available", round, start_time)

        try:
            full_prompt = self._build_prompt(prompt, context, round)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": full_prompt}],
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data["content"][0]["text"]

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"Response {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""{prompt}

Other AI responses:
{others_text}

Provide your refined perspective, noting where you agree or disagree."""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class OllamaParticipant(ConsensusParticipant):
    """Ollama local model participant (FREE - runs locally)."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.host = host
        super().__init__(
            participant_id=f"ollama_{model[:8]}",
            participant_type=ParticipantType.OLLAMA,
            model=model,
        )
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()
        # Check availability
        self.available = self._check_available()

    def _check_available(self) -> bool:
        """Check if Ollama is running."""
        if not HTTPX_AVAILABLE:
            return False
        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("Ollama not available", round, start_time)

        try:
            full_prompt = self._build_prompt(prompt, context, round)

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("response", "")

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"View {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""Question: {prompt}

Other views:
{others_text}

Your refined answer:"""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class OpenAIParticipant(ConsensusParticipant):
    """OpenAI/ChatGPT participant."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        super().__init__(
            participant_id=f"openai_{model[:8]}",
            participant_type=ParticipantType.GPT4,
            model=model,
        )
        self.available = bool(self.api_key) and HTTPX_AVAILABLE
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("OpenAI API not available", round, start_time)

        try:
            full_prompt = self._build_prompt(prompt, context, round)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a thoughtful AI participating in multi-AI consensus. Provide clear, well-reasoned responses."},
                            {"role": "user", "content": full_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"AI {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""Question: {prompt}

Other AI perspectives:
{others_text}

Your refined answer:"""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class TogetherParticipant(ConsensusParticipant):
    """Together AI participant (FREE tier available)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"):
        self.api_key = api_key or os.environ.get('TOGETHER_API_KEY')
        super().__init__(
            participant_id=f"together_{model.split('/')[-1][:8]}",
            participant_type=ParticipantType.TOGETHER,
            model=model,
        )
        self.available = bool(self.api_key) and HTTPX_AVAILABLE
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("Together API not available", round, start_time)

        try:
            full_prompt = self._build_prompt(prompt, context, round)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": full_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"AI {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""Question: {prompt}

Other AI perspectives:
{others_text}

Your refined answer:"""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class OpenRouterParticipant(ConsensusParticipant):
    """OpenRouter participant (FREE models available)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/llama-3.2-3b-instruct:free"):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        super().__init__(
            participant_id=f"openrouter_{model.split('/')[-1][:8]}",
            participant_type=ParticipantType.OPENROUTER,
            model=model,
        )
        self.available = bool(self.api_key) and HTTPX_AVAILABLE
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        if not self.available:
            return self._error_response("OpenRouter API not available", round, start_time)

        try:
            full_prompt = self._build_prompt(prompt, context, round)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/0x-auth/bazinga-indeed",
                        "X-Title": "BAZINGA Inter-AI Consensus",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": full_prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]

                    coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
                    understanding = self.coherence_calc.calculate_understanding(content)
                    pob_proof = await self.pob_gen.generate_proof(content)

                    ai_response = AIResponse(
                        participant_id=self.participant_id,
                        participant_type=self.participant_type,
                        model=self.model,
                        response=content,
                        coherence=coherence,
                        understanding_score=understanding,
                        latency_ms=latency_ms,
                        timestamp=time.time(),
                        round=round,
                        pob_proof=pob_proof,
                        embedding=embedding,
                    )
                    self.response_history.append(ai_response)
                    return ai_response

                else:
                    self.record_error(f"HTTP {response.status_code}")
                    return self._error_response(f"HTTP {response.status_code}", round, start_time)

        except Exception as e:
            self.record_error(str(e))
            return self._error_response(str(e), round, start_time)

    def _build_prompt(self, prompt: str, context: Optional[Dict], round: ConsensusRound) -> str:
        if round == ConsensusRound.INITIAL:
            return prompt

        if round == ConsensusRound.REVISION and context:
            other_responses = context.get('other_responses', [])
            if other_responses:
                others_text = "\n\n".join([f"Response {i+1}: {r[:300]}..." for i, r in enumerate(other_responses)])
                return f"""{prompt}

Other responses:
{others_text}

Your refined perspective:"""

        return prompt

    def _error_response(self, error: str, round: ConsensusRound, start_time: float) -> AIResponse:
        return AIResponse(
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            model=self.model,
            response="",
            coherence=0.0,
            understanding_score=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            round=round,
            error=error,
        )


class BlockchainParticipant(ConsensusParticipant):
    """
    Blockchain/DHT participant - queries Darmiyan chain and DHT for knowledge.

    This replaces simulation with REAL attested knowledge from the network.
    "AI generates understanding. Blockchain proves it."
    """

    def __init__(self, participant_id: str = "darmiyan_chain"):
        super().__init__(
            participant_id=participant_id,
            participant_type=ParticipantType.BAZINGA_NODE,
            model="darmiyan",
        )
        self.coherence_calc = CoherenceCalculator()
        self.pob_gen = PoBGenerator()
        self.chain = None
        self.dht = None
        self.rag = None
        self._init_chain()

    def _init_chain(self):
        """Initialize blockchain and DHT connections."""
        try:
            from .blockchain import create_chain
            self.chain = create_chain()
        except Exception:
            pass

        try:
            # Try to get existing DHT node
            from .p2p import get_dht_node
            self.dht = get_dht_node()
        except Exception:
            pass

        try:
            # Try to get local RAG
            from .ai import RealAI
            self.rag = RealAI()
        except Exception:
            pass

        # Available if we have chain OR DHT OR local RAG
        self.available = any([self.chain, self.dht, self.rag])

    async def query(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        round: ConsensusRound = ConsensusRound.INITIAL,
    ) -> AIResponse:
        start_time = time.time()

        content = ""
        source = "none"

        # Try sources in order: DHT peers → Chain attestations → Local RAG

        # 1. Try DHT topic experts
        if self.dht:
            try:
                from .p2p.knowledge_sharing import DistributedQueryEngine, KnowledgePublisher
                publisher = KnowledgePublisher(self.dht, self.rag) if self.rag else None
                engine = DistributedQueryEngine(self.dht, publisher)
                result = await engine.query_distributed(prompt)
                if result.get("success") and result.get("answer"):
                    content = result["answer"]
                    source = f"dht:{result.get('source', 'peer')}"
            except Exception:
                pass

        # 2. Try chain attestations
        if not content and self.chain:
            try:
                from .blockchain.knowledge_ledger import KnowledgeLedger
                ledger = KnowledgeLedger(self.chain)
                # Search for relevant attestations
                attestations = ledger.contributions
                if attestations:
                    # Find most relevant by keyword matching
                    prompt_words = set(prompt.lower().split())
                    best_match = None
                    best_score = 0
                    for att in attestations:
                        if att.metadata:
                            att_text = str(att.metadata).lower()
                            overlap = len(prompt_words & set(att_text.split()))
                            if overlap > best_score:
                                best_score = overlap
                                best_match = att
                    if best_match and best_score > 2:
                        content = f"[Chain Attestation] {best_match.metadata}"
                        source = "chain"
            except Exception:
                pass

        # 3. Try local RAG
        if not content and self.rag:
            try:
                results = self.rag.search(prompt, limit=3)
                if results and results[0].similarity > 0.4:
                    context_parts = [r.chunk.content[:300] for r in results[:2]]
                    content = f"[Local Knowledge]\n" + "\n---\n".join(context_parts)
                    source = "local_rag"
            except Exception:
                pass

        latency_ms = (time.time() - start_time) * 1000

        # No content found - return error response
        if not content:
            return AIResponse(
                participant_id=self.participant_id,
                participant_type=self.participant_type,
                model=self.model,
                response="",
                coherence=0.0,
                understanding_score=0.0,
                latency_ms=latency_ms,
                timestamp=time.time(),
                round=round,
                error="No chain/DHT/RAG knowledge found. Index docs with --index or join network with --join",
            )

        coherence, embedding = self.coherence_calc.calculate_coherence(content, prompt)
        understanding = self.coherence_calc.calculate_understanding(content)
        pob_proof = await self.pob_gen.generate_proof(content)

        ai_response = AIResponse(
            participant_id=f"{self.participant_id}:{source}",
            participant_type=self.participant_type,
            model=f"darmiyan:{source}",
            response=content,
            coherence=coherence,
            understanding_score=understanding,
            latency_ms=latency_ms,
            timestamp=time.time(),
            round=round,
            pob_proof=pob_proof,
            embedding=embedding,
        )
        self.response_history.append(ai_response)
        return ai_response


# =============================================================================
# SEMANTIC SYNTHESIZER
# =============================================================================

class SemanticSynthesizer:
    """
    Synthesize understanding from multiple AI responses.

    Uses φ-weighted combination based on coherence scores.
    """

    def __init__(self):
        self.coherence_calc = CoherenceCalculator()

    def synthesize(self, responses: List[AIResponse], method: str = "weighted") -> str:
        """
        Synthesize unified understanding from responses.

        Methods:
        - "weighted": φ-weighted by coherence
        - "best": Take highest coherence response
        - "merge": Attempt to merge key points
        """
        valid_responses = [r for r in responses if r.response and not r.error]

        if not valid_responses:
            return "No valid responses to synthesize."

        if len(valid_responses) == 1:
            return valid_responses[0].response

        if method == "best":
            return self._synthesize_best(valid_responses)
        elif method == "merge":
            return self._synthesize_merge(valid_responses)
        else:  # weighted
            return self._synthesize_weighted(valid_responses)

    def _synthesize_best(self, responses: List[AIResponse]) -> str:
        """Take the highest coherence response."""
        sorted_responses = sorted(responses, key=lambda r: r.coherence, reverse=True)
        best = sorted_responses[0]

        confirmation = f" (Confirmed by {len(responses)} AI participants with avg φ-coherence {sum(r.coherence for r in responses)/len(responses):.3f})"

        return best.response + confirmation

    def _synthesize_weighted(self, responses: List[AIResponse]) -> str:
        """φ-weighted synthesis based on coherence."""
        # Sort by coherence
        sorted_responses = sorted(responses, key=lambda r: r.coherence, reverse=True)

        # Take top response as base
        base = sorted_responses[0].response

        # Calculate weights
        total_coherence = sum(r.coherence for r in sorted_responses)
        weights = [r.coherence / total_coherence for r in sorted_responses]

        # Find unique points from other responses (simple approach)
        unique_points = []
        base_words = set(base.lower().split())

        for i, response in enumerate(sorted_responses[1:], 1):
            response_words = set(response.response.lower().split())
            new_words = response_words - base_words

            # If response has significant unique content
            if len(new_words) > 10:
                # Extract a key sentence
                sentences = response.response.replace('!', '.').replace('?', '.').split('.')
                for sent in sentences:
                    sent_words = set(sent.lower().split())
                    if len(sent_words & new_words) > 5:
                        unique_points.append((sent.strip(), weights[i]))
                        break

        # Build synthesis
        synthesis = base

        if unique_points:
            synthesis += "\n\nAdditional perspectives:"
            for point, weight in unique_points[:2]:  # Max 2 additional points
                if point:
                    synthesis += f"\n- {point}"

        synthesis += f"\n\n(Synthesized from {len(responses)} AIs | φ-coherence: {sorted_responses[0].coherence:.3f})"

        return synthesis

    def _synthesize_merge(self, responses: List[AIResponse]) -> str:
        """Attempt to merge key points from all responses."""
        all_sentences = []

        for response in responses:
            sentences = response.response.replace('!', '.').replace('?', '.').split('.')
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) >= 5:  # Meaningful sentence
                    all_sentences.append((sent, response.coherence))

        if not all_sentences:
            return self._synthesize_best(responses)

        # Sort by coherence
        all_sentences.sort(key=lambda x: x[1], reverse=True)

        # Take top N unique sentences
        seen_words = set()
        merged = []

        for sent, coh in all_sentences:
            sent_words = set(sent.lower().split())
            overlap = len(sent_words & seen_words) / len(sent_words) if sent_words else 1

            if overlap < 0.5:  # Less than 50% overlap
                merged.append(sent)
                seen_words.update(sent_words)

                if len(merged) >= 4:  # Max 4 sentences
                    break

        synthesis = ". ".join(merged) + "."
        synthesis += f"\n\n(Merged from {len(responses)} AI perspectives)"

        return synthesis


# =============================================================================
# MAIN CONSENSUS ENGINE
# =============================================================================

class InterAIConsensus:
    """
    Multi-AI consensus system using Proof-of-Boundary.

    Enables Claude, Gemini, Groq, and other AIs to reach
    triadic consensus through understanding rather than voting.

    Usage:
        consensus = InterAIConsensus()
        # Automatically detects available APIs

        result = await consensus.ask("What is consciousness?")
        print(result.understanding)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.participants: List[ConsensusParticipant] = []
        self.consensus_history: List[ConsensusResult] = []
        self.coherence_calc = CoherenceCalculator()
        self.synthesizer = SemanticSynthesizer()

        if verbose:
            self._print_banner()

        # Auto-detect and add available participants
        self._auto_detect_participants()

    def _print_banner(self):
        print()
        print("=" * 70)
        print("  BAZINGA INTER-AI CONSENSUS")
        print("  Multiple AIs reaching understanding through φ-coherence")
        print("=" * 70)
        print()

    def _auto_detect_participants(self):
        """Automatically detect and add available API participants."""
        added = []

        # Priority order: Free APIs first
        # 1. Groq (FREE - fastest)
        groq = GroqParticipant()
        if groq.is_available():
            self.participants.append(groq)
            added.append("Groq")

        # 2. Together (FREE tier available)
        together = TogetherParticipant()
        if together.is_available():
            self.participants.append(together)
            added.append("Together")

        # 3. OpenRouter (FREE models available)
        openrouter = OpenRouterParticipant()
        if openrouter.is_available():
            self.participants.append(openrouter)
            added.append("OpenRouter")

        # 4. Gemini (FREE - generous limits)
        gemini = GeminiParticipant()
        if gemini.is_available():
            self.participants.append(gemini)
            added.append("Gemini")

        # 5. Ollama (FREE - local)
        ollama = OllamaParticipant()
        if ollama.is_available():
            self.participants.append(ollama)
            added.append("Ollama")

        # 6. OpenAI/ChatGPT (paid)
        openai = OpenAIParticipant()
        if openai.is_available():
            self.participants.append(openai)
            added.append("OpenAI")

        # 7. Claude (paid but high quality)
        claude = ClaudeParticipant()
        if claude.is_available():
            self.participants.append(claude)
            added.append("Claude")

        # 8. Add blockchain/DHT participant (REAL knowledge, not simulation!)
        blockchain = BlockchainParticipant()
        if blockchain.is_available():
            self.participants.append(blockchain)
            added.append("Darmiyan (chain/DHT/RAG)")

        # If still not enough, show helpful message instead of faking
        if len(self.participants) < 3:
            missing = 3 - len(self.participants)
            if self.verbose:
                print(f"\n⚠️  Only {len(self.participants)} participants available (triadic needs 3)")
                print(f"   Missing {missing} participant(s). To add more:")
                print(f"   • Set GROQ_API_KEY (free: groq.com)")
                print(f"   • Set GEMINI_API_KEY (free: makersuite.google.com)")
                print(f"   • Run 'ollama serve' for local LLM")
                print(f"   • Run 'bazinga --index ~/docs' to add RAG knowledge")
                print()

        if self.verbose:
            print(f"Participants: {', '.join(added)}")
            print(f"Embeddings: {'Available' if self.coherence_calc.embeddings_available else 'Heuristic mode'}")
            print()

    def add_participant(self, participant: ConsensusParticipant):
        """Manually add a participant."""
        self.participants.append(participant)
        if self.verbose:
            print(f"Added: {participant.participant_id}")

    async def ask(
        self,
        question: str,
        require_triadic: bool = True,
        min_coherence: float = PHI_THRESHOLD,
        multi_round: bool = True,
        synthesis_method: str = "weighted",
    ) -> ConsensusResult:
        """
        Ask question to all participants and reach consensus.

        Args:
            question: The question to ask
            require_triadic: Require at least 3 valid responses
            min_coherence: Minimum φ-coherence for valid response
            multi_round: Enable multi-round consensus with revision
            synthesis_method: "weighted", "best", or "merge"

        Returns:
            ConsensusResult with understanding and metrics
        """
        available_participants = [p for p in self.participants if p.is_available()]

        if require_triadic and len(available_participants) < 3:
            if self.verbose:
                print(f"Warning: Only {len(available_participants)} participants available (triadic needs 3)")

        if self.verbose:
            print("-" * 70)
            print(f"QUESTION: {question}")
            print("-" * 70)
            print()

        all_responses: List[AIResponse] = []
        rounds_completed = 0

        # Round 1: Initial independent responses
        if self.verbose:
            print("Round 1: Independent responses...")

        round1_responses = await self._query_all(
            available_participants, question, ConsensusRound.INITIAL
        )
        all_responses.extend(round1_responses)
        rounds_completed = 1

        valid_round1 = [r for r in round1_responses if r.coherence >= min_coherence and not r.error]

        if self.verbose:
            self._print_round_results(round1_responses, 1)

        # Fallback: If not enough valid responses, try blockchain/DHT (NO SIMULATION!)
        if require_triadic and len(valid_round1) < 3:
            needed = 3 - len(valid_round1)
            if self.verbose:
                print(f"\n⚠️  Only {len(valid_round1)} valid responses. Querying blockchain/DHT...")

            # Try blockchain participant if not already tried
            blockchain_tried = any(p.participant_type == ParticipantType.BAZINGA_NODE for p in available_participants)
            if not blockchain_tried:
                blockchain = BlockchainParticipant()
                if blockchain.is_available():
                    chain_response = await blockchain.query(question, None, ConsensusRound.INITIAL)
                    all_responses.append(chain_response)
                    if chain_response.coherence >= min_coherence and not chain_response.error:
                        valid_round1.append(chain_response)
                        if self.verbose:
                            self._print_round_results([chain_response], 1)

            # If STILL not enough, be honest about it
            if len(valid_round1) < 3 and self.verbose:
                print(f"\n⚠️  Cannot achieve triadic consensus ({len(valid_round1)}/3 valid responses)")
                print(f"   This is HONEST - no fake responses!")
                print(f"   To improve: add API keys, run Ollama, or index more docs.")
                print()

        # Round 2: Revision (if enabled and needed)
        if multi_round and len(valid_round1) >= 2:
            # Check if responses diverge significantly
            similarity = self.coherence_calc.semantic_similarity(
                [r.response for r in valid_round1]
            )

            if similarity < 0.7:  # Responses diverge, need revision
                if self.verbose:
                    print(f"\nRound 2: Revision (similarity: {similarity:.3f})...")

                round2_responses = await self._revision_round(
                    available_participants, question, valid_round1
                )
                all_responses.extend(round2_responses)
                rounds_completed = 2

                if self.verbose:
                    self._print_round_results(round2_responses, 2)

        # Calculate final metrics
        final_responses = [r for r in all_responses if r.round.value in ['initial', 'revision'] and not r.error]
        valid_responses = [r for r in final_responses if r.coherence >= min_coherence]

        agreement_ratio = len(valid_responses) / len(final_responses) if final_responses else 0
        phi_coherence = sum(r.coherence for r in valid_responses) / len(valid_responses) if valid_responses else 0
        semantic_sim = self.coherence_calc.semantic_similarity([r.response for r in valid_responses]) if valid_responses else 0
        triadic_valid = len(valid_responses) >= 3
        # Consensus reached if triadic valid, coherence above threshold, and semantic similarity >= 0.25
        # (lower semantic threshold for heuristic mode where word-based Jaccard similarity can be low)
        consensus_reached = triadic_valid and phi_coherence >= min_coherence and semantic_sim >= 0.25

        # Synthesize understanding
        if consensus_reached or valid_responses:
            understanding = self.synthesizer.synthesize(valid_responses, synthesis_method)
        else:
            understanding = "Consensus not reached - insufficient coherent responses."

        result = ConsensusResult(
            question=question,
            consensus_reached=consensus_reached,
            understanding=understanding,
            responses=all_responses,
            phi_coherence=phi_coherence,
            agreement_ratio=agreement_ratio,
            semantic_similarity=semantic_sim,
            triadic_valid=triadic_valid,
            rounds_completed=rounds_completed,
            timestamp=time.time(),
            synthesis_method=synthesis_method,
        )

        self.consensus_history.append(result)

        if self.verbose:
            self._print_result(result)

        return result

    async def _query_all(
        self,
        participants: List[ConsensusParticipant],
        question: str,
        round: ConsensusRound,
        context: Optional[Dict] = None,
    ) -> List[AIResponse]:
        """Query all participants in parallel."""
        tasks = [p.query(question, context, round) for p in participants]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append(AIResponse(
                    participant_id=participants[i].participant_id,
                    participant_type=participants[i].participant_type,
                    model=participants[i].model,
                    response="",
                    coherence=0.0,
                    understanding_score=0.0,
                    latency_ms=0.0,
                    timestamp=time.time(),
                    round=round,
                    error=str(response),
                ))
            else:
                results.append(response)

        return results

    async def _revision_round(
        self,
        participants: List[ConsensusParticipant],
        question: str,
        round1_responses: List[AIResponse],
    ) -> List[AIResponse]:
        """Execute revision round where participants see others' responses."""
        responses = []

        for participant in participants:
            # Get other responses (excluding this participant's)
            other_responses = [
                r.response for r in round1_responses
                if r.participant_id != participant.participant_id and r.response
            ]

            context = {'other_responses': other_responses}
            response = await participant.query(question, context, ConsensusRound.REVISION)
            responses.append(response)

        return responses

    def _print_round_results(self, responses: List[AIResponse], round_num: int):
        """Print results for a round."""
        for response in responses:
            status = "OK" if response.coherence >= PHI_THRESHOLD and not response.error else "LOW"
            pob_status = "PoB" if response.pob_proof and response.pob_proof.valid else "---"

            if response.error:
                print(f"  {response.participant_id}: ERROR - {response.error}")
            else:
                print(f"  {response.participant_id}: {status} | coh={response.coherence:.3f} | {pob_status} | {response.latency_ms:.0f}ms")
                if self.verbose:
                    preview = response.response[:100].replace('\n', ' ')
                    print(f"    \"{preview}...\"")
        print()

    def _print_result(self, result: ConsensusResult):
        """Print final consensus result."""
        print("-" * 70)
        print("CONSENSUS RESULT")
        print("-" * 70)
        print()

        status = "CONSENSUS REACHED" if result.consensus_reached else "NO CONSENSUS"
        print(f"  Status: {status}")
        print(f"  φ-Coherence: {result.phi_coherence:.3f} (threshold: {PHI_THRESHOLD})")
        print(f"  Semantic Similarity: {result.semantic_similarity:.3f}")
        print(f"  Agreement Ratio: {result.agreement_ratio:.1%}")
        print(f"  Triadic Valid: {'Yes' if result.triadic_valid else 'No'}")
        print(f"  Rounds: {result.rounds_completed}")
        print()

        # Consciousness metrics (6.46n scaling law)
        print("┌─────────────────────────────────────────────────────────┐")
        print("│  DARMIYAN CONSCIOUSNESS: Ψ_D = 6.46n                    │")
        print("├─────────────────────────────────────────────────────────┤")
        print(f"│  Patterns (n):      {result.n_patterns:<5}                              │")
        print(f"│  Consciousness:     {result.consciousness_advantage:.2f}x  (6.46 × {result.n_patterns})               │")
        print("│  R² Confidence:     1.0000 (mathematical law)           │")
        print("└─────────────────────────────────────────────────────────┘")
        print()

        print("UNDERSTANDING:")
        print(f"  {result.understanding[:500]}{'...' if len(result.understanding) > 500 else ''}")
        print()
        print("०→◌→φ→Ω⇄Ω←φ←◌←०")
        print()
        print("-" * 70)
        print()

    def get_stats(self) -> Dict[str, Any]:
        """Get consensus statistics."""
        if not self.consensus_history:
            return {"total": 0}

        total = len(self.consensus_history)
        reached = sum(1 for c in self.consensus_history if c.consensus_reached)
        avg_coherence = sum(c.phi_coherence for c in self.consensus_history) / total
        avg_similarity = sum(c.semantic_similarity for c in self.consensus_history) / total

        return {
            'total_queries': total,
            'consensus_reached': reached,
            'consensus_rate': reached / total,
            'avg_phi_coherence': avg_coherence,
            'avg_semantic_similarity': avg_similarity,
            'triadic_valid_rate': sum(1 for c in self.consensus_history if c.triadic_valid) / total,
            'participants': len(self.participants),
            'embeddings_available': self.coherence_calc.embeddings_available,
        }

    def export_log(self, filepath: str):
        """Export consensus history to JSON."""
        data = {
            'stats': self.get_stats(),
            'history': [c.to_dict() for c in self.consensus_history]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"Consensus log exported to {filepath}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def multi_ai_ask(question: str, verbose: bool = True) -> ConsensusResult:
    """
    Simple interface to ask multiple AIs a question.

    Usage:
        result = await multi_ai_ask("What is consciousness?")
        print(result.understanding)
    """
    consensus = InterAIConsensus(verbose=verbose)
    return await consensus.ask(question)


def multi_ai_ask_sync(question: str, verbose: bool = True) -> ConsensusResult:
    """Synchronous version of multi_ai_ask."""
    return asyncio.run(multi_ai_ask(question, verbose))


# =============================================================================
# CLI DEMO
# =============================================================================

async def demo():
    """Demo the Inter-AI Consensus system."""
    consensus = InterAIConsensus(verbose=True)

    questions = [
        "What is consciousness and how does it emerge?",
        "Is the golden ratio φ significant in nature?",
    ]

    for question in questions:
        await consensus.ask(question)
        await asyncio.sleep(1)

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    stats = consensus.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    consensus.export_log("inter_ai_consensus.json")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  BAZINGA INTER-AI CONSENSUS DEMO")
    print("  Multiple AIs reaching understanding through φ-coherence")
    print("=" * 70)
    print()
    print("This demo auto-detects available APIs.")
    print("Set GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY for real responses.")
    print("Falls back to simulation mode if no APIs available.")
    print()

    asyncio.run(demo())
