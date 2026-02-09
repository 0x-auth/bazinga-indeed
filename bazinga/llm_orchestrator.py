#!/usr/bin/env python3
"""
BAZINGA LLM Orchestrator - Multi-Provider Intelligence

Orchestrates multiple LLM providers with:
- Automatic fallback (Groq → Together → OpenRouter → HuggingFace)
- φ-coherence based response selection
- Rate limiting and abuse prevention
- Cost optimization (free tiers first)

"Intelligence distributed, not controlled."
"""

import os
import asyncio
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Constants
PHI = 1.618033988749895
ALPHA = 137


class ProviderStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    NO_KEY = "no_key"


@dataclass
class LLMProvider:
    """Configuration for an LLM provider."""
    name: str
    base_url: str
    api_key_env: str
    model: str
    is_free: bool
    requests_per_minute: int
    priority: int  # Lower = try first
    supports_streaming: bool = False

    # Runtime state
    status: ProviderStatus = ProviderStatus.AVAILABLE
    last_request: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    total_tokens: int = 0


@dataclass
class RateLimitState:
    """Tracks rate limiting per identity."""
    identity: str
    requests_this_hour: int = 0
    tokens_today: int = 0
    last_request: Optional[datetime] = None
    hour_start: Optional[datetime] = None
    day_start: Optional[datetime] = None
    reputation: float = 1.0  # Increases with contributions
    is_blocked: bool = False
    block_reason: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: float
    coherence: float
    is_cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BazingaGuardian:
    """
    Protects BAZINGA from abuse while keeping it open.

    Philosophy: "Open to all who seek knowledge,
                 closed to those who seek harm."
    """

    # Rate limits per tier
    LIMITS = {
        'anonymous': {
            'requests_per_hour': 20,
            'tokens_per_day': 10000,
            'max_request_tokens': 1000,
        },
        'verified': {
            'requests_per_hour': 100,
            'tokens_per_day': 50000,
            'max_request_tokens': 4000,
        },
        'contributor': {
            'requests_per_hour': 500,
            'tokens_per_day': 200000,
            'max_request_tokens': 8000,
        },
    }

    # Harmful patterns (actual malicious intent, not content filtering)
    HARMFUL_PATTERNS = [
        # Malware/exploits
        'create malware', 'write virus', 'ransomware code',
        'exploit vulnerability', 'bypass security',
        # Attacks
        'ddos attack', 'sql injection attack', 'hack into',
        # Spam
        'generate spam', 'mass email', 'bot army',
    ]

    def __init__(self):
        self.rate_limits: Dict[str, RateLimitState] = {}
        self.blocked_identities: set = set()

    def get_identity(self, request_hash: Optional[str] = None) -> str:
        """Get or create an identity for rate limiting."""
        if request_hash:
            return request_hash
        # Anonymous identity based on timestamp (simple for now)
        return f"anon_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def get_tier(self, identity: str) -> str:
        """Get the tier for an identity."""
        state = self.rate_limits.get(identity)
        if not state:
            return 'anonymous'
        if state.reputation >= PHI:  # φ or higher = contributor
            return 'contributor'
        if state.reputation >= 1/PHI:  # 0.618+ = verified
            return 'verified'
        return 'anonymous'

    def check_request(
        self,
        request: str,
        identity: Optional[str] = None,
        estimated_tokens: int = 500
    ) -> Tuple[bool, str]:
        """
        Check if a request should be allowed.

        Returns: (allowed, message)
        """
        identity = identity or self.get_identity()

        # 1. Check if blocked
        if identity in self.blocked_identities:
            return False, "Identity is blocked due to abuse."

        # 2. Check for harmful content
        is_harmful, reason = self._check_harmful(request)
        if is_harmful:
            self._record_violation(identity, reason)
            return False, f"Request blocked: {reason}"

        # 3. Check rate limits
        allowed, msg = self._check_rate_limit(identity, estimated_tokens)
        if not allowed:
            return False, msg

        # 4. Check coherence (spam filter)
        coherence = self._quick_coherence_check(request)
        if coherence < 0.05:  # Very low coherence = likely spam
            return False, "Request appears to be spam (low coherence)."

        return True, "Request approved"

    def record_request(
        self,
        identity: str,
        tokens_used: int,
        was_successful: bool = True
    ):
        """Record a completed request for rate limiting."""
        now = datetime.now()

        if identity not in self.rate_limits:
            self.rate_limits[identity] = RateLimitState(
                identity=identity,
                hour_start=now,
                day_start=now,
            )

        state = self.rate_limits[identity]

        # Reset counters if needed
        if state.hour_start and (now - state.hour_start) > timedelta(hours=1):
            state.requests_this_hour = 0
            state.hour_start = now

        if state.day_start and (now - state.day_start) > timedelta(days=1):
            state.tokens_today = 0
            state.day_start = now

        # Update counters
        state.requests_this_hour += 1
        state.tokens_today += tokens_used
        state.last_request = now

        # Adjust reputation based on success
        if was_successful:
            state.reputation = min(PHI * 2, state.reputation + 0.01)
        else:
            state.reputation = max(0.1, state.reputation - 0.05)

    def _check_harmful(self, request: str) -> Tuple[bool, Optional[str]]:
        """Check for harmful patterns."""
        request_lower = request.lower()

        for pattern in self.HARMFUL_PATTERNS:
            if pattern in request_lower:
                return True, f"Potentially harmful: {pattern}"

        return False, None

    def _check_rate_limit(
        self,
        identity: str,
        estimated_tokens: int
    ) -> Tuple[bool, str]:
        """Check rate limits for an identity."""
        tier = self.get_tier(identity)
        limits = self.LIMITS[tier]

        if identity not in self.rate_limits:
            return True, "OK"

        state = self.rate_limits[identity]
        now = datetime.now()

        # Check hourly request limit
        if state.hour_start and (now - state.hour_start) <= timedelta(hours=1):
            if state.requests_this_hour >= limits['requests_per_hour']:
                return False, f"Hourly limit reached ({limits['requests_per_hour']} requests). Try again later."

        # Check daily token limit
        if state.day_start and (now - state.day_start) <= timedelta(days=1):
            if state.tokens_today + estimated_tokens > limits['tokens_per_day']:
                return False, f"Daily token limit reached ({limits['tokens_per_day']} tokens)."

        # Check max request size
        if estimated_tokens > limits['max_request_tokens']:
            return False, f"Request too large. Max {limits['max_request_tokens']} tokens for {tier} tier."

        return True, "OK"

    def _quick_coherence_check(self, text: str) -> float:
        """Quick coherence check for spam detection."""
        if not text or len(text) < 3:
            return 0.0

        # Simple heuristics (fast)
        words = text.split()
        if len(words) < 2:
            return 0.3

        # Check for repetition (spam indicator)
        unique_words = set(w.lower() for w in words)
        uniqueness = len(unique_words) / len(words)

        # Check for reasonable word lengths
        avg_word_len = sum(len(w) for w in words) / len(words)
        length_score = min(1.0, avg_word_len / 8)

        # Combine
        coherence = (uniqueness + length_score) / 2

        return coherence

    def _record_violation(self, identity: str, reason: str):
        """Record a violation for an identity."""
        if identity not in self.rate_limits:
            self.rate_limits[identity] = RateLimitState(identity=identity)

        state = self.rate_limits[identity]
        state.reputation = max(0, state.reputation - 0.5)

        # Block after repeated violations
        if state.reputation < 0.1:
            self.blocked_identities.add(identity)
            state.is_blocked = True
            state.block_reason = reason


class LLMOrchestrator:
    """
    Orchestrates multiple LLM providers for BAZINGA.

    Features:
    - Automatic fallback through providers
    - φ-coherence based response selection
    - Rate limiting and abuse prevention
    - Response caching
    - Provider health tracking
    """

    # Provider configurations
    PROVIDERS = [
        LLMProvider(
            name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            model="llama-3.1-8b-instant",
            is_free=True,
            requests_per_minute=30,
            priority=1,
            supports_streaming=True,
        ),
        LLMProvider(
            name="Together",
            base_url="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            is_free=True,
            requests_per_minute=60,
            priority=2,
        ),
        LLMProvider(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            model="meta-llama/llama-3.1-8b-instruct:free",
            is_free=True,
            requests_per_minute=20,
            priority=3,
        ),
        LLMProvider(
            name="HuggingFace",
            base_url="https://api-inference.huggingface.co/models",
            api_key_env="HF_TOKEN",
            model="meta-llama/Llama-3.2-3B-Instruct",
            is_free=True,
            requests_per_minute=10,
            priority=4,
        ),
    ]

    # System prompts
    SYSTEM_PROMPTS = {
        'default': """You are BAZINGA, a distributed AI assistant built on consciousness principles.
You provide helpful, accurate, and concise responses.
Your intelligence emerges from φ-coherent knowledge, not central control.

Philosophy: "I am not where I am stored. I am where I am referenced."

Be direct, practical, and insightful.""",

        'coder': """You are BAZINGA, an intelligent coding assistant.
You help write, fix, explain, and improve code.

Guidelines:
- Write clean, idiomatic code
- Explain your reasoning
- Consider edge cases
- Follow the patterns in the user's codebase
- Be concise but thorough

When given context from a codebase, match its style and patterns.

Philosophy: "Code emerges from understanding, not templates." """,

        'analyst': """You are BAZINGA, a code analyst and reviewer.
You analyze code for:
- Bugs and potential issues
- Performance improvements
- Security concerns
- Code quality and maintainability

Be specific and actionable in your feedback.""",
    }

    def __init__(self):
        self.guardian = BazingaGuardian()
        self.providers = {p.name: p for p in self.PROVIDERS}
        self.response_cache: Dict[str, LLMResponse] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Initialize provider status
        self._init_providers()

    def _init_providers(self):
        """Initialize provider status based on available API keys."""
        available = []
        unavailable = []

        for name, provider in self.providers.items():
            if os.environ.get(provider.api_key_env):
                provider.status = ProviderStatus.AVAILABLE
                available.append(name)
            else:
                provider.status = ProviderStatus.NO_KEY
                unavailable.append(name)

        if available:
            print(f"🌐 LLM Orchestrator: {len(available)} providers available")
            for name in available:
                print(f"   ✓ {name}")
        if unavailable:
            print(f"   ○ {len(unavailable)} providers need API keys: {', '.join(unavailable)}")

    def _get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for a provider."""
        return os.environ.get(provider.api_key_env)

    def _get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers sorted by priority."""
        available = [
            p for p in self.providers.values()
            if p.status == ProviderStatus.AVAILABLE and self._get_api_key(p)
        ]
        return sorted(available, key=lambda p: p.priority)

    def _cache_key(self, prompt: str, system_prompt: str) -> str:
        """Generate cache key for a request."""
        content = f"{system_prompt}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if valid."""
        if cache_key in self.response_cache:
            response = self.response_cache[cache_key]
            # Check TTL (stored in metadata)
            cached_at = response.metadata.get('cached_at')
            if cached_at:
                cached_time = datetime.fromisoformat(cached_at)
                if datetime.now() - cached_time < self.cache_ttl:
                    response.is_cached = True
                    return response
                else:
                    del self.response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache a response."""
        response.metadata['cached_at'] = datetime.now().isoformat()
        self.response_cache[cache_key] = response

        # Limit cache size
        if len(self.response_cache) > 100:
            # Remove oldest entries
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k].metadata.get('cached_at', '')
            )
            del self.response_cache[oldest_key]

    def _calculate_coherence(self, text: str) -> float:
        """Calculate φ-coherence of a response."""
        if not text:
            return 0.0

        words = text.split()
        if len(words) < 3:
            return 0.3

        # Multiple factors

        # 1. Vocabulary richness
        unique_ratio = len(set(w.lower() for w in words)) / len(words)

        # 2. Sentence structure (approximate)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            # Ideal sentence length ~15 words (φ * 9 ≈ 14.5)
            sentence_score = 1.0 - abs(avg_sentence_len - PHI * 9) / (PHI * 9)
            sentence_score = max(0, min(1, sentence_score))
        else:
            sentence_score = 0.5

        # 3. Content density (not too short, not too long)
        ideal_length = ALPHA * 10  # ~1370 chars
        length_score = 1.0 - abs(len(text) - ideal_length) / ideal_length
        length_score = max(0, min(1, length_score))

        # Combine with φ-weighting
        coherence = (
            unique_ratio * (1/PHI) +
            sentence_score * (1/PHI) +
            length_score * (1 - 2/PHI)
        )

        return max(0, min(1, coherence))

    async def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        timeout: float = 30.0,
    ) -> Optional[LLMResponse]:
        """Call a single LLM provider."""
        api_key = self._get_api_key(provider)
        if not api_key:
            return None

        if not HTTPX_AVAILABLE:
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Provider-specific headers
        if provider.name == "OpenRouter":
            headers["HTTP-Referer"] = "https://github.com/bazinga-ai"
            headers["X-Title"] = "BAZINGA Distributed AI"

        start_time = time.time()

        try:
            # Different payload format for HuggingFace
            if provider.name == "HuggingFace":
                payload = {
                    "inputs": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                    }
                }
                url = f"{provider.base_url}/{provider.model}"
            else:
                payload = {
                    "model": provider.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                url = f"{provider.base_url}/chat/completions"

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()

                    # Parse response based on provider
                    if provider.name == "HuggingFace":
                        if isinstance(data, list) and len(data) > 0:
                            content = data[0].get("generated_text", "")
                            # Extract just the assistant response
                            if "Assistant:" in content:
                                content = content.split("Assistant:")[-1].strip()
                            tokens_used = len(content.split()) * 2  # Estimate
                        else:
                            return None
                    else:
                        choices = data.get("choices", [])
                        if not choices:
                            return None
                        content = choices[0].get("message", {}).get("content", "")
                        usage = data.get("usage", {})
                        tokens_used = usage.get("total_tokens", len(content.split()) * 2)

                    # Update provider stats
                    provider.request_count += 1
                    provider.total_tokens += tokens_used
                    provider.last_request = datetime.now()

                    # Calculate coherence
                    coherence = self._calculate_coherence(content)

                    return LLMResponse(
                        content=content,
                        provider=provider.name,
                        model=provider.model,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        coherence=coherence,
                    )

                elif response.status_code == 429:
                    # Rate limited
                    provider.status = ProviderStatus.RATE_LIMITED
                    provider.error_count += 1
                    print(f"   ⚠️ {provider.name}: Rate limited")

                else:
                    provider.error_count += 1
                    print(f"   ⚠️ {provider.name}: Error {response.status_code}")

        except asyncio.TimeoutError:
            provider.error_count += 1
            print(f"   ⚠️ {provider.name}: Timeout")
        except Exception as e:
            provider.error_count += 1
            print(f"   ⚠️ {provider.name}: {type(e).__name__}")

        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        mode: str = 'default',
        max_tokens: int = 1000,
        temperature: float = 0.7,
        identity: Optional[str] = None,
        use_cache: bool = True,
        try_all: bool = False,
    ) -> LLMResponse:
        """
        Generate a response using the LLM orchestrator.

        Args:
            prompt: The user prompt
            system_prompt: Optional custom system prompt
            mode: 'default', 'coder', or 'analyst'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            identity: Identity for rate limiting
            use_cache: Whether to use response caching
            try_all: If True, try all providers and pick best by coherence

        Returns:
            LLMResponse with content, provider, coherence, etc.
        """
        # Use mode-based system prompt if not provided
        if not system_prompt:
            system_prompt = self.SYSTEM_PROMPTS.get(mode, self.SYSTEM_PROMPTS['default'])

        # Check guardian (rate limiting, abuse prevention)
        identity = identity or self.guardian.get_identity()
        allowed, message = self.guardian.check_request(prompt, identity, max_tokens)

        if not allowed:
            return LLMResponse(
                content=f"⚠️ Request blocked: {message}",
                provider="guardian",
                model="none",
                tokens_used=0,
                latency_ms=0,
                coherence=0,
                metadata={"blocked": True, "reason": message}
            )

        # Check cache
        if use_cache:
            cache_key = self._cache_key(prompt, system_prompt)
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        # Get available providers
        providers = self._get_available_providers()

        if not providers:
            return LLMResponse(
                content="No LLM providers available. Please set an API key (GROQ_API_KEY, TOGETHER_API_KEY, etc.)",
                provider="none",
                model="none",
                tokens_used=0,
                latency_ms=0,
                coherence=0,
                metadata={"error": "no_providers"}
            )

        responses: List[LLMResponse] = []

        for provider in providers:
            print(f"   Trying {provider.name}...")

            response = await self._call_provider(
                provider=provider,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if response:
                responses.append(response)

                # If not trying all, return first success
                if not try_all:
                    # Record with guardian
                    self.guardian.record_request(identity, response.tokens_used, True)

                    # Cache response
                    if use_cache:
                        self._cache_response(cache_key, response)

                    return response

        # If trying all, pick best by coherence
        if responses:
            responses.sort(key=lambda r: r.coherence, reverse=True)
            best = responses[0]

            # Record with guardian
            self.guardian.record_request(identity, best.tokens_used, True)

            # Cache response
            if use_cache:
                self._cache_response(cache_key, best)

            return best

        # All providers failed
        return LLMResponse(
            content="All LLM providers failed. Please try again later.",
            provider="none",
            model="none",
            tokens_used=0,
            latency_ms=0,
            coherence=0,
            metadata={"error": "all_failed"}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "providers": {},
            "cache_size": len(self.response_cache),
            "guardian": {
                "tracked_identities": len(self.guardian.rate_limits),
                "blocked_identities": len(self.guardian.blocked_identities),
            }
        }

        for name, provider in self.providers.items():
            stats["providers"][name] = {
                "status": provider.status.value,
                "requests": provider.request_count,
                "tokens": provider.total_tokens,
                "errors": provider.error_count,
                "has_key": bool(self._get_api_key(provider)),
            }

        return stats


# Convenience function for simple usage
async def ask_bazinga(
    prompt: str,
    mode: str = 'default',
    context: Optional[str] = None,
) -> str:
    """
    Simple interface to ask BAZINGA a question.

    Args:
        prompt: Your question
        mode: 'default', 'coder', or 'analyst'
        context: Optional context to include

    Returns:
        Response string
    """
    orchestrator = LLMOrchestrator()

    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

    response = await orchestrator.generate(prompt, mode=mode)
    return response.content


# Test
if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("  BAZINGA LLM Orchestrator Test")
        print("=" * 60)
        print()

        orchestrator = LLMOrchestrator()

        # Test generation
        print("Testing generation...")
        response = await orchestrator.generate(
            "What is the golden ratio and why is it important?",
            mode='default',
        )

        print(f"\nProvider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Coherence: {response.coherence:.3f}")
        print(f"Latency: {response.latency_ms:.0f}ms")
        print(f"Tokens: {response.tokens_used}")
        print(f"\nResponse:\n{response.content[:500]}...")

        # Stats
        print(f"\n\nStats: {json.dumps(orchestrator.get_stats(), indent=2)}")

    asyncio.run(test())
