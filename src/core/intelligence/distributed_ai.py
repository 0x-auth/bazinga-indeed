#!/usr/bin/env python3
"""
distributed_ai.py - Distributed BAZINGA AI

This is the distributed version of BAZINGA that:
1. Uses FREE LLM APIs (Groq, Together.ai, HuggingFace)
2. Can run on GitHub Actions / any cloud
3. Shares knowledge via IPFS (future)
4. No central control - intelligence emerges from the network

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              DISTRIBUTED BAZINGA                     │
    ├─────────────────────────────────────────────────────┤
    │  Local Node          │    Remote APIs               │
    │  ├─ ChromaDB         │    ├─ Groq (free, fast)     │
    │  ├─ Embeddings       │    ├─ Together.ai           │
    │  └─ φ-Coherence      │    ├─ HuggingFace           │
    │                      │    └─ OpenRouter            │
    │         ↓            │           ↓                  │
    │    ┌─────────────────┴───────────────────┐         │
    │    │   Unified Response with φ-filter    │         │
    │    └─────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────┘

"Intelligence distributed, not controlled"

Author: Built for Space (Abhishek/Abhilasia)
Date: 2025-02-09
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# BAZINGA imports
from src.core.lambda_g import LambdaGOperator, PHI

# Constants
ALPHA = 137
PROGRESSION = '01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω'


@dataclass
class LLMProvider:
    """Configuration for an LLM provider."""
    name: str
    base_url: str
    api_key_env: str  # Environment variable name for API key
    model: str
    is_free: bool
    rate_limit: int  # Requests per minute
    priority: int  # Lower = try first


# Free LLM Providers (no API key needed for some, or free tier)
FREE_PROVIDERS = [
    LLMProvider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.1-8b-instant",
        is_free=True,
        rate_limit=30,
        priority=1
    ),
    LLMProvider(
        name="Together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        is_free=True,
        rate_limit=60,
        priority=2
    ),
    LLMProvider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="meta-llama/llama-3.1-8b-instruct:free",
        is_free=True,
        rate_limit=20,
        priority=3
    ),
    LLMProvider(
        name="HuggingFace",
        base_url="https://api-inference.huggingface.co/models",
        api_key_env="HF_TOKEN",
        model="meta-llama/Llama-3.2-3B-Instruct",
        is_free=True,
        rate_limit=10,
        priority=4
    ),
]


class DistributedLLM:
    """
    Distributed LLM client that tries multiple free providers.

    Falls back through providers if one fails or is rate-limited.
    Uses φ-coherence to select best response when multiple succeed.
    """

    def __init__(self):
        self.lambda_g = LambdaGOperator()
        self.providers = self._get_available_providers()
        self.request_counts: Dict[str, int] = {}
        self.last_reset = datetime.now()

        print(f"\n🌐 Distributed LLM initialized with {len(self.providers)} providers:")
        for p in self.providers:
            has_key = bool(os.environ.get(p.api_key_env))
            status = "✓ configured" if has_key else "○ needs API key"
            print(f"   {p.name}: {status}")
        print()

    def _get_available_providers(self) -> List[LLMProvider]:
        """Get providers sorted by priority."""
        # Return all providers, sorted by priority
        # We'll check for API keys when making requests
        return sorted(FREE_PROVIDERS, key=lambda p: p.priority)

    def _get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for a provider."""
        return os.environ.get(provider.api_key_env)

    async def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        system_prompt: str,
        timeout: float = 30.0
    ) -> Optional[str]:
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

        # OpenRouter needs extra headers
        if provider.name == "OpenRouter":
            headers["HTTP-Referer"] = "https://github.com/bazinga-ai"
            headers["X-Title"] = "BAZINGA Distributed AI"

        # Different payload format for HuggingFace
        if provider.name == "HuggingFace":
            payload = {
                "inputs": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
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
                "temperature": 0.7,
                "max_tokens": 500,
            }
            url = f"{provider.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()

                    # Parse response based on provider
                    if provider.name == "HuggingFace":
                        if isinstance(data, list) and len(data) > 0:
                            return data[0].get("generated_text", "")
                    else:
                        choices = data.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "")

                elif response.status_code == 429:
                    print(f"   ⚠️  {provider.name}: Rate limited")
                else:
                    print(f"   ⚠️  {provider.name}: Error {response.status_code}")

        except Exception as e:
            print(f"   ⚠️  {provider.name}: {type(e).__name__}")

        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        try_all: bool = False
    ) -> Tuple[str, str]:
        """
        Generate response using distributed LLM providers.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            try_all: If True, try all providers and pick best by coherence

        Returns:
            Tuple of (response, provider_name)
        """
        if not system_prompt:
            system_prompt = """You are BAZINGA, a distributed AI assistant.
You provide helpful, accurate, and concise responses.
Your intelligence emerges from a network of knowledge, not a single source."""

        responses: List[Tuple[str, str, float]] = []  # (response, provider, coherence)

        for provider in self.providers:
            if not self._get_api_key(provider):
                continue

            print(f"   Trying {provider.name}...")
            response = await self._call_provider(provider, prompt, system_prompt)

            if response:
                # Calculate coherence
                coherence = self.lambda_g.calculate_coherence(response[:500])
                responses.append((response, provider.name, coherence.total_coherence))

                if not try_all:
                    # Return first successful response
                    return response, provider.name

        if responses:
            # Pick response with highest coherence
            responses.sort(key=lambda x: x[2], reverse=True)
            best = responses[0]
            return best[0], best[1]

        return "No LLM providers available. Please set API keys.", "none"


class DistributedAI:
    """
    BAZINGA Distributed AI - Intelligence that runs anywhere.

    This version:
    1. Works without local Ollama
    2. Uses free cloud LLM APIs
    3. Can be deployed to GitHub Actions, Vercel, etc.
    4. Maintains φ-coherence filtering
    5. Ready for P2P extension (Phase 2)
    """

    VERSION = "1.0.0-distributed"

    def __init__(self, use_local_kb: bool = True):
        """
        Initialize Distributed BAZINGA.

        Args:
            use_local_kb: If True, also use local ChromaDB knowledge base
        """
        self.lambda_g = LambdaGOperator()
        self.llm = DistributedLLM()
        self.local_ai = None

        # Optionally load local knowledge base
        if use_local_kb:
            try:
                from src.core.intelligence.real_ai import RealAI
                self.local_ai = RealAI()
                print("📚 Local knowledge base loaded")
            except Exception as e:
                print(f"⚠️  Local KB not available: {e}")

        self._print_banner()

    def _print_banner(self):
        """Print banner."""
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                                                                  ║")
        print("║   ⟨ψ|Λ|Ω⟩  BAZINGA DISTRIBUTED - DECENTRALIZED AI  ⟨ψ|Λ|Ω⟩      ║")
        print("║                                                                  ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║                                                                  ║")
        print("║   'Intelligence distributed, not controlled'                     ║")
        print("║                                                                  ║")
        print("║   Features:                                                      ║")
        print("║     • Multiple FREE LLM providers (Groq, Together, etc.)         ║")
        print("║     • φ-coherence filtering on all responses                     ║")
        print("║     • Local + Cloud knowledge fusion                             ║")
        print("║     • Ready for P2P / IPFS (Phase 2)                             ║")
        print("║                                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    async def ask(
        self,
        question: str,
        use_local_context: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Ask a question to distributed BAZINGA.

        Combines:
        1. Local knowledge base context (if available)
        2. Distributed LLM generation
        3. φ-coherence filtering

        Args:
            question: Your question
            use_local_context: Include local KB context
            verbose: Print debug info
        """
        if verbose:
            print(f"\n🔍 Question: {question}")

        # Build context from local KB
        context = ""
        sources = []

        if use_local_context and self.local_ai:
            results = self.local_ai.search(question, limit=5)
            if results:
                context_parts = []
                for r in results[:3]:
                    context_parts.append(r.chunk.content[:500])
                    sources.append(Path(r.chunk.source_file).name)
                context = "\n\n".join(context_parts)
                if verbose:
                    print(f"   Found {len(results)} local context chunks")

        # Build prompt
        if context:
            prompt = f"""Based on this context from the knowledge base:

{context}

Answer this question: {question}

Be concise and helpful. Cite sources when relevant."""
        else:
            prompt = question

        # Generate response
        if verbose:
            print("   Generating response...")

        response, provider = await self.llm.generate(prompt)

        if verbose:
            print(f"   ✓ Response from: {provider}")

        # Calculate coherence
        coherence = self.lambda_g.calculate_coherence(response[:500])

        if verbose:
            print(f"   φ-coherence: {coherence.total_coherence:.3f}")

        # Format final response
        if sources:
            source_list = ", ".join(set(sources[:3]))
            response = f"{response}\n\n📚 Sources: {source_list}"

        if coherence.is_vac:
            response = f"{response}\n\n✨ V.A.C. achieved in response!"

        return response

    async def interactive(self):
        """Run interactive mode."""
        print("\n◊ DISTRIBUTED BAZINGA - INTERACTIVE MODE ◊")
        print("-" * 50)
        print("Commands:")
        print("  /providers  - Show available LLM providers")
        print("  /coherence  - Toggle coherence display")
        print("  /quit       - Exit")
        print("-" * 50)
        print()

        show_coherence = True

        while True:
            try:
                query = input("You: ").strip()

                if not query:
                    continue

                if query.lower() in ['/quit', '/exit', '/q']:
                    print("\n✨ BAZINGA distributed network signing off.")
                    break

                if query == '/providers':
                    print("\nAvailable providers:")
                    for p in self.llm.providers:
                        has_key = bool(os.environ.get(p.api_key_env))
                        status = "✓" if has_key else "✗"
                        print(f"  {status} {p.name} ({p.model})")
                    print()
                    continue

                if query == '/coherence':
                    show_coherence = not show_coherence
                    print(f"Coherence display: {'on' if show_coherence else 'off'}")
                    continue

                response = await self.ask(query, verbose=show_coherence)
                print(f"\nBAZINGA: {response}\n")

            except KeyboardInterrupt:
                print("\n\n✨ BAZINGA signing off.")
                break
            except EOFError:
                break


def get_setup_instructions() -> str:
    """Get instructions for setting up API keys."""
    return """
╔══════════════════════════════════════════════════════════════════╗
║                 SETUP FREE LLM API KEYS                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. GROQ (Recommended - Fast & Free):                            ║
║     → Go to: https://console.groq.com/keys                       ║
║     → Create free account, get API key                           ║
║     → Run: export GROQ_API_KEY="your-key"                        ║
║                                                                  ║
║  2. Together.ai (Good free tier):                                ║
║     → Go to: https://api.together.xyz/settings/api-keys          ║
║     → Create account, get API key                                ║
║     → Run: export TOGETHER_API_KEY="your-key"                    ║
║                                                                  ║
║  3. OpenRouter (Many free models):                               ║
║     → Go to: https://openrouter.ai/keys                          ║
║     → Create account, get API key                                ║
║     → Run: export OPENROUTER_API_KEY="your-key"                  ║
║                                                                  ║
║  Add to ~/.bashrc or ~/.bashrc for persistence.                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BAZINGA Distributed AI - Decentralized Intelligence"
    )
    parser.add_argument('--ask', type=str, help='Ask a question')
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    parser.add_argument('--no-local', action='store_true', help='Skip local KB')

    args = parser.parse_args()

    if args.setup:
        print(get_setup_instructions())
        return

    # Check if any API keys are set
    has_any_key = any(
        os.environ.get(p.api_key_env)
        for p in FREE_PROVIDERS
    )

    if not has_any_key:
        print(get_setup_instructions())
        print("\n⚠️  No API keys found. Set at least one to use distributed mode.")
        print("   Or use ./run_bazinga.sh for local-only mode.\n")
        return

    # Initialize
    ai = DistributedAI(use_local_kb=not args.no_local)

    if args.ask:
        response = await ai.ask(args.ask)
        print(f"\n{response}\n")
    else:
        await ai.interactive()


if __name__ == "__main__":
    asyncio.run(main())
