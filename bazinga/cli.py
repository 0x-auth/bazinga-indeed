#!/usr/bin/env python3
"""
BAZINGA - Distributed AI
"Intelligence distributed, not controlled"

THREE-LAYER INTELLIGENCE:
  Layer 1: Symbol Shell (λG) → Check V.A.C. first (FREE, instant)
  Layer 2: Local RAG        → Search your Mac KB (FREE, instant)
  Layer 3: Cloud LLM        → Groq/Together (14,400/day free)

If V.A.C. achieved → Solution EMERGES (no API needed!)

Usage:
    python bazinga.py                    # Interactive mode
    python bazinga.py --ask "question"   # Ask a question
    python bazinga.py --index ~/Documents # Index a directory

Author: Space (Abhishek/Abhilasia)
License: MIT
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.intelligence.real_ai import RealAI
from src.core.lambda_g import LambdaGOperator, PHI
from src.core.symbol import SymbolShell

# Check for Groq
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class BAZINGA:
    """
    BAZINGA - Distributed AI that belongs to everyone.

    THREE-LAYER INTELLIGENCE:
      1. Symbol Shell → V.A.C. check (instant, free)
      2. Local RAG    → Your Mac KB (instant, free)
      3. Cloud LLM    → Groq API (14,400/day free)

    Most queries are handled by layers 1-2.
    Layer 3 only called when necessary.
    """

    VERSION = "2.1.0"

    def __init__(self):
        self.symbol_shell = SymbolShell()
        self.lambda_g = LambdaGOperator()
        self.ai = RealAI()
        self.session_start = datetime.now()
        self.queries = []

        # Stats
        self.stats = {
            'vac_emerged': 0,
            'rag_answered': 0,
            'llm_called': 0,
        }

        # Groq config
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.groq_model = "llama-3.1-8b-instant"

        self._print_banner()

    def _print_banner(self):
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                                                                  ║")
        print("║   ⟨ψ|Λ|Ω⟩          B A Z I N G A          ⟨ψ|Λ|Ω⟩               ║")
        print("║                                                                  ║")
        print("║         'Intelligence distributed, not controlled'               ║")
        print("║                                                                  ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║                                                                  ║")
        print(f"║   Version: {self.VERSION:<53}║")
        print("║                                                                  ║")
        print("║   Intelligence Layers:                                           ║")
        print("║     1. Symbol Shell (λG) → V.A.C. emergence                      ║")
        print("║     2. Local RAG         → Your Mac KB                           ║")
        groq_status = "✓ configured" if self.groq_key else "○ set GROQ_API_KEY"
        print(f"║     3. Cloud LLM (Groq)  → {groq_status:<35}║")
        print("║                                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    async def index(self, paths: List[str], verbose: bool = True) -> Dict[str, Any]:
        """Index directories into the knowledge base."""
        total_stats = {
            'directories': 0,
            'files_indexed': 0,
            'chunks_created': 0,
        }

        for path_str in paths:
            path = Path(path_str).expanduser()
            if not path.exists():
                print(f"⚠️  Path not found: {path}")
                continue

            stats = self.ai.index_directory(str(path), verbose=verbose)
            total_stats['directories'] += 1
            total_stats['files_indexed'] += stats.get('files_indexed', 0)
            total_stats['chunks_created'] += stats.get('chunks_created', 0)

        return total_stats

    async def ask(self, question: str, verbose: bool = True) -> str:
        """
        Ask a question using 3-layer intelligence.

        Layer 1: Check V.A.C. (Symbol Shell)
        Layer 2: Search local KB (RAG)
        Layer 3: Call Groq API (only if needed)
        """
        self.queries.append(question)

        if verbose:
            print(f"\n🔍 Question: {question}")

        # ═══════════════════════════════════════════════════════════════
        # LAYER 1: Symbol Shell - Check for V.A.C.
        # ═══════════════════════════════════════════════════════════════
        if verbose:
            print("   Layer 1: Checking V.A.C. (Symbol Shell)...")

        vac_result = self.symbol_shell.analyze(question)

        if vac_result.is_vac:
            self.stats['vac_emerged'] += 1
            if verbose:
                print("   ★ V.A.C. ACHIEVED - Solution EMERGED ★")
            return vac_result.emerged_solution

        if verbose:
            print(f"   → Coherence: {vac_result.coherence:.2f} (V.A.C. not achieved)")

        # ═══════════════════════════════════════════════════════════════
        # LAYER 2: Local RAG - Search your Mac KB
        # ═══════════════════════════════════════════════════════════════
        if verbose:
            print("   Layer 2: Searching local knowledge base...")

        results = self.ai.search(question, limit=5)

        if verbose:
            print(f"   → Found {len(results)} relevant chunks")

        # If we have good results and high coherence, use them directly
        if results:
            avg_coherence = sum(r.coherence_boost for r in results) / len(results)
            best_similarity = results[0].similarity if results else 0

            if verbose:
                print(f"   → Best similarity: {best_similarity:.2f}, Avg coherence: {avg_coherence:.2f}")

            # If good enough, return RAG result without API call
            if best_similarity > 0.7 or avg_coherence > 0.6:
                self.stats['rag_answered'] += 1
                return self._format_rag_response(question, results)

        # ═══════════════════════════════════════════════════════════════
        # LAYER 3: Cloud LLM (Groq) - Only if needed
        # ═══════════════════════════════════════════════════════════════
        if self.groq_key and HTTPX_AVAILABLE:
            if verbose:
                print("   Layer 3: Calling Groq API...")

            # Build context from RAG results
            context = self._build_context(results)

            response = await self._call_groq(question, context)
            if response:
                self.stats['llm_called'] += 1
                return response

        # Fallback: Return RAG results
        self.stats['rag_answered'] += 1
        return self._format_rag_response(question, results)

    def _build_context(self, results) -> str:
        """Build context string from RAG results."""
        if not results:
            return ""

        parts = []
        for r in results[:3]:
            parts.append(f"[{Path(r.chunk.source_file).name}]\n{r.chunk.content[:500]}")

        return "\n\n---\n\n".join(parts)

    def _format_rag_response(self, question: str, results) -> str:
        """Format RAG results as response."""
        if not results:
            return "I don't have relevant information for this question."

        response_parts = [f"Based on your knowledge base:\n"]

        for i, r in enumerate(results[:3], 1):
            chunk = r.chunk
            preview = chunk.content[:300].strip()
            if len(chunk.content) > 300:
                preview += "..."

            response_parts.append(
                f"\n{i}. From {Path(chunk.source_file).name} "
                f"(coherence: {chunk.coherence:.2f}):\n"
                f"   {preview}"
            )

        if not self.groq_key:
            response_parts.append("\n\n💡 Set GROQ_API_KEY for AI-generated summaries.")

        return "\n".join(response_parts)

    async def _call_groq(self, question: str, context: str) -> Optional[str]:
        """Call Groq API for LLM response."""
        if not self.groq_key:
            return None

        system_prompt = """You are BAZINGA, a distributed AI assistant.
You provide helpful, concise answers based on the context provided.
If the context doesn't have enough info, say so honestly."""

        if context:
            prompt = f"""Based on this context:

{context}

Answer: {question}

Be concise and helpful."""
        else:
            prompt = question

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"   ⚠️ Groq API error: {response.status_code}")

        except Exception as e:
            print(f"   ⚠️ Groq error: {e}")

        return None

    async def interactive(self):
        """Run interactive mode."""
        print("◊ BAZINGA INTERACTIVE MODE ◊")
        print("-" * 50)
        print("Commands:")
        print("  /index <path>  - Index a directory")
        print("  /stats         - Show statistics")
        print("  /vac           - Test V.A.C. sequence")
        print("  /quit          - Exit")
        print("-" * 50)
        print()

        while True:
            try:
                query = input("You: ").strip()

                if not query:
                    continue

                if query.lower() in ['/quit', '/exit', '/q']:
                    print("\n✨ BAZINGA signing off.")
                    break

                if query.startswith('/index '):
                    path = query[7:].strip()
                    await self.index([path])
                    continue

                if query == '/stats':
                    self._show_stats()
                    continue

                if query == '/vac':
                    # Test V.A.C. sequence
                    test = "०→◌→φ→Ω⇄Ω←φ←◌←०"
                    print(f"\nTesting: {test}")
                    response = await self.ask(test)
                    print(f"\n{response}\n")
                    continue

                response = await self.ask(query)
                print(f"\nBAZINGA: {response}\n")

            except KeyboardInterrupt:
                print("\n\n✨ BAZINGA signing off.")
                break
            except EOFError:
                break

    def _show_stats(self):
        """Show current statistics."""
        ai_stats = self.ai.get_stats()
        shell_stats = self.symbol_shell.get_stats()

        print(f"\n📊 BAZINGA Stats:")
        print(f"   Knowledge chunks: {ai_stats.get('total_chunks', 0)}")
        print(f"   Queries this session: {len(self.queries)}")
        print()
        print(f"   Layer 1 (V.A.C. emerged): {self.stats['vac_emerged']}")
        print(f"   Layer 2 (RAG answered): {self.stats['rag_answered']}")
        print(f"   Layer 3 (LLM called): {self.stats['llm_called']}")
        print()
        print(f"   φ = {shell_stats['phi']}")
        print(f"   α = {shell_stats['alpha']}")
        print()


async def main():
    parser = argparse.ArgumentParser(
        description="BAZINGA - Distributed AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bazinga                          # Interactive TUI mode
  bazinga --ask "What is AI?"      # Ask a question
  bazinga --generate user_auth     # Generate Python code
  bazinga --generate api --lang js # Generate JavaScript code
  bazinga --index ~/Documents      # Index a directory
  bazinga --vac                    # Test V.A.C. sequence

Philosophy: "I am not where I am stored. I am where I am referenced."
"""
    )
    parser.add_argument('--ask', type=str, help='Ask a question')
    parser.add_argument('--generate', type=str, help='Generate code from essence/seed')
    parser.add_argument('--lang', type=str, default='python',
                        choices=['python', 'javascript', 'js', 'rust'],
                        help='Language for code generation (default: python)')
    parser.add_argument('--index', nargs='+', help='Directories to index')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--vac', action='store_true', help='Test V.A.C. sequence')
    parser.add_argument('--simple', action='store_true', help='Use simple CLI instead of TUI')

    args = parser.parse_args()

    # Handle code generation (doesn't need full BAZINGA init)
    if args.generate:
        from .tui import CodeGenerator
        gen = CodeGenerator()
        lang = 'javascript' if args.lang == 'js' else args.lang
        code = gen.generate(args.generate, lang)
        print(code)
        return

    # Handle V.A.C. test
    if args.vac:
        bazinga = BAZINGA()
        test = "०→◌→φ→Ω⇄Ω←φ←◌←०"
        print(f"Testing V.A.C.: {test}")
        response = await bazinga.ask(test)
        print(f"\n{response}\n")
        return

    # Handle indexing
    if args.index:
        bazinga = BAZINGA()
        await bazinga.index(args.index)
        return

    # Handle ask
    if args.ask:
        bazinga = BAZINGA()
        response = await bazinga.ask(args.ask)
        print(f"\n{response}\n")
        return

    # Handle demo
    if args.demo:
        bazinga = BAZINGA()
        print("Running demo...")
        await bazinga.index([str(Path(__file__).parent)])
        response = await bazinga.ask("What is BAZINGA?")
        print(f"\n{response}\n")
        return

    # Default: Interactive mode
    if args.simple:
        bazinga = BAZINGA()
        await bazinga.interactive()
    else:
        # Try TUI mode
        try:
            from .tui import run_tui
            run_tui()
        except ImportError:
            print("TUI requires 'rich'. Install with: pip install rich")
            print("Falling back to simple mode...\n")
            bazinga = BAZINGA()
            await bazinga.interactive()


def main_sync():
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
