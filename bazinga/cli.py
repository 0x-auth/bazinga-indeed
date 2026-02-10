#!/usr/bin/env python3
# Suppress ML library noise BEFORE any imports
import os, warnings, logging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

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

    VERSION = "3.1.0"

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
        """Minimal clean banner."""
        print()
        print("BAZINGA v" + self.VERSION)
        if not self.groq_key:
            print("(Set GROQ_API_KEY for AI responses)")
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

    async def ask(self, question: str, verbose: bool = False) -> str:
        """
        Ask a question using 3-layer intelligence.
        """
        self.queries.append(question)

        # Layer 1: Check V.A.C. (Symbol Shell)
        vac_result = self.symbol_shell.analyze(question)
        if vac_result.is_vac:
            self.stats['vac_emerged'] += 1
            return vac_result.emerged_solution

        # Layer 2: Local RAG
        results = self.ai.search(question, limit=5)

        if results:
            avg_coherence = sum(r.coherence_boost for r in results) / len(results)
            best_similarity = results[0].similarity if results else 0

            # If good local match, use it
            if best_similarity > 0.5 and avg_coherence > 0.6:
                self.stats['rag_answered'] += 1
                return self._format_rag_response(question, results)

        # Layer 3: Cloud LLM (Groq)
        if self.groq_key and HTTPX_AVAILABLE:
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
            if not self.groq_key:
                return "No info found. Set GROQ_API_KEY for AI answers."
            return "I don't have relevant information for this question."

        # Just show the top result content cleanly
        top = results[0].chunk
        content = top.content[:500].strip()
        source = Path(top.source_file).name

        return f"{content}\n\n[Source: {source}]"

    async def _call_groq(self, question: str, context: str) -> Optional[str]:
        """Call Groq API for LLM response."""
        if not self.groq_key:
            return None

        system_prompt = """You are BAZINGA, a distributed AI assistant.
You provide helpful, concise answers. Use context if relevant, otherwise use your general knowledge.
Be accurate and informative."""

        if context:
            prompt = f"""Context (use if relevant, otherwise use general knowledge):

{context}

Question: {question}

Answer concisely and helpfully."""
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

        except Exception:
            pass  # Silent fallback

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
  bazinga --code "fibonacci"       # LLM-powered code generation (NEW!)
  bazinga --code "api client" --lang js  # Generate JavaScript with LLM
  bazinga --generate user_auth     # Template-based code generation
  bazinga --index ~/Documents      # Index a directory
  bazinga --vac                    # Test V.A.C. sequence

Philosophy: "I am not where I am stored. I am where I am referenced."
"""
    )
    parser.add_argument('--ask', type=str, help='Ask a question')
    parser.add_argument('--code', type=str, help='LLM-powered intelligent code generation')
    parser.add_argument('--generate', type=str, help='Template-based code generation (no LLM)')
    parser.add_argument('--lang', type=str, default='python',
                        choices=['python', 'javascript', 'js', 'typescript', 'ts', 'rust', 'go'],
                        help='Language for code generation (default: python)')
    parser.add_argument('--index', nargs='+', help='Directories to index')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--vac', action='store_true', help='Test V.A.C. sequence')
    parser.add_argument('--simple', action='store_true', help='Use simple CLI instead of TUI')

    args = parser.parse_args()

    # Handle LLM-powered code generation (NEW!)
    if args.code:
        try:
            from .intelligent_coder import IntelligentCoder
            coder = IntelligentCoder()
            lang = {'js': 'javascript', 'ts': 'typescript'}.get(args.lang, args.lang)
            print(f"Generating {lang} code with LLM...")
            result = await coder.generate(args.code, lang)
            print(f"\n# Provider: {result.provider}")
            print(f"# Coherence: {result.coherence:.3f}")
            print(f"# Tokens: {result.tokens_used}")
            print()
            print(result.code)
        except ImportError as e:
            print(f"Error: Intelligent coder not available: {e}")
            print("Install dependencies: pip install httpx")
        return

    # Handle template-based code generation (doesn't need full BAZINGA init)
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
