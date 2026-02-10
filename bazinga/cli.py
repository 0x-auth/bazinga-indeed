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
from .learning import get_memory, LearningMemory

# Check for Groq
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Check for local LLM
try:
    from .local_llm import LocalLLM, get_local_llm
    LOCAL_LLM_AVAILABLE = LocalLLM.is_available()
except ImportError:
    LOCAL_LLM_AVAILABLE = False


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

    VERSION = "3.3.0"

    def __init__(self):
        self.symbol_shell = SymbolShell()
        self.lambda_g = LambdaGOperator()
        self.ai = RealAI()
        self.session_start = datetime.now()
        self.queries = []

        # Learning memory
        self.memory = get_memory()
        self.memory.start_session()

        # Stats
        self.stats = {
            'vac_emerged': 0,
            'rag_answered': 0,
            'llm_called': 0,
            'from_memory': 0,
        }

        # LLM config
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.groq_model = "llama-3.1-8b-instant"
        self.local_llm = None
        self.use_local = False

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
        Ask a question using 4-layer intelligence with learning.
        """
        self.queries.append(question)

        # Layer 0: Check learned patterns (instant)
        cached = self.memory.find_similar_question(question)
        if cached and cached.get('coherence', 0) > 0.7:
            self.stats['from_memory'] += 1
            return cached['answer']

        # Layer 1: Check V.A.C. (Symbol Shell)
        vac_result = self.symbol_shell.analyze(question)
        if vac_result.is_vac:
            self.stats['vac_emerged'] += 1
            self.memory.record_interaction(question, vac_result.emerged_solution, 'vac', 0.9)
            return vac_result.emerged_solution

        # Layer 2: Local RAG - only use if VERY relevant
        results = self.ai.search(question, limit=5)
        best_similarity = results[0].similarity if results else 0

        # Only trust RAG for high similarity (>0.75) - otherwise use LLM
        use_rag_only = best_similarity > 0.75

        # Layer 3: LLM (Cloud or Local)
        if not use_rag_only:
            conv_context = self.memory.get_context(2)
            rag_context = self._build_context(results) if best_similarity > 0.3 else ""
            full_context = f"{conv_context}\n\n{rag_context}".strip()

            # Try Groq first (faster)
            if self.groq_key and HTTPX_AVAILABLE:
                response = await self._call_groq(question, full_context)
                if response:
                    self.stats['llm_called'] += 1
                    self.memory.record_interaction(question, response, 'llm', 0.8)
                    return response

            # Fallback to local LLM
            if LOCAL_LLM_AVAILABLE or self.use_local:
                response = self._call_local_llm(question, full_context)
                if response:
                    self.stats['llm_called'] += 1
                    self.memory.record_interaction(question, response, 'local', 0.7)
                    return response

        # Fallback to RAG if LLM unavailable or RAG is very relevant
        if results and best_similarity > 0.4:
            self.stats['rag_answered'] += 1
            response = self._format_rag_response(question, results)
            self.memory.record_interaction(question, response, 'rag', best_similarity)
            return response

        # No good answer
        if not self.groq_key and not LOCAL_LLM_AVAILABLE:
            return "No AI available. Either:\n  - Set GROQ_API_KEY for cloud AI\n  - pip install llama-cpp-python for local AI\n  - bazinga --index ~/path to index docs"
        return "I don't have relevant information for this question."

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

    def _call_local_llm(self, question: str, context: str) -> Optional[str]:
        """Call local LLM for response."""
        try:
            if self.local_llm is None:
                from .local_llm import get_local_llm
                self.local_llm = get_local_llm()

            if context:
                prompt = f"Context: {context[:500]}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = question

            return self.local_llm.generate(prompt)
        except Exception:
            return None

    async def interactive(self):
        """Run interactive mode."""
        print("◊ BAZINGA INTERACTIVE MODE ◊")
        print("Commands: /index /stats /good /bad /quit")
        print()

        last_response = ""

        while True:
            try:
                query = input("You: ").strip()

                if not query:
                    continue

                if query.lower() in ['/quit', '/exit', '/q']:
                    self.memory.end_session()
                    print("\n✨ BAZINGA signing off.")
                    break

                if query.startswith('/index '):
                    path = query[7:].strip()
                    await self.index([path])
                    continue

                if query == '/stats':
                    self._show_stats()
                    continue

                if query == '/good' and last_response:
                    self.memory.record_feedback(self.queries[-1] if self.queries else "", last_response, 1)
                    print("👍 Thanks! I'll remember that.\n")
                    continue

                if query == '/bad' and last_response:
                    self.memory.record_feedback(self.queries[-1] if self.queries else "", last_response, -1)
                    print("👎 Got it. I'll try to do better.\n")
                    continue

                if query == '/vac':
                    test = "०→◌→φ→Ω⇄Ω←φ←◌←०"
                    print(f"\nTesting: {test}")
                    response = await self.ask(test)
                    print(f"\n{response}\n")
                    continue

                response = await self.ask(query)
                last_response = response
                print(f"\nBAZINGA: {response}\n")

            except KeyboardInterrupt:
                self.memory.end_session()
                print("\n\n✨ BAZINGA signing off.")
                break
            except EOFError:
                break

    def _show_stats(self):
        """Show current statistics."""
        ai_stats = self.ai.get_stats()
        memory_stats = self.memory.get_stats()

        print(f"\n📊 BAZINGA Stats:")
        print(f"   Queries this session: {len(self.queries)}")
        print(f"   Memory: {self.stats['from_memory']} | VAC: {self.stats['vac_emerged']} | RAG: {self.stats['rag_answered']} | LLM: {self.stats['llm_called']}")
        print(f"   Patterns learned: {memory_stats['patterns_learned']}")
        print(f"   Feedback: {memory_stats['positive_feedback']}👍 {memory_stats['negative_feedback']}👎")
        print()


async def main():
    parser = argparse.ArgumentParser(
        description="BAZINGA - Distributed AI that belongs to everyone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  bazinga --ask "What is AI?"         Ask any question
  bazinga --ask "explain quantum"     Get detailed explanations
  bazinga --code "fibonacci"          Generate code with AI
  bazinga --index ~/Documents         Index your files for RAG
  bazinga --local                     Use local AI (no internet)
  bazinga                             Interactive mode

INTERACTIVE COMMANDS:
  /good      Mark last response as helpful (learns)
  /bad       Mark as unhelpful (adapts)
  /stats     Show session statistics
  /index     Index a directory
  /quit      Exit

INSTALLATION OPTIONS:
  pip install bazinga-indeed              Basic (needs GROQ_API_KEY)
  pip install bazinga-indeed[local]       With local AI (offline capable)
  pip install bazinga-indeed[full]        Everything

ENVIRONMENT:
  GROQ_API_KEY    Your Groq API key (free at console.groq.com)

"I am not where I am stored. I am where I am referenced."
"""
    )

    # Main options
    parser.add_argument('--ask', '-a', type=str, metavar='QUESTION',
                        help='Ask a question (uses AI)')
    parser.add_argument('--code', '-c', type=str, metavar='TASK',
                        help='Generate code for a task')
    parser.add_argument('--lang', '-l', type=str, default='python',
                        choices=['python', 'javascript', 'js', 'typescript', 'ts', 'rust', 'go'],
                        help='Language for code generation (default: python)')
    parser.add_argument('--index', '-i', nargs='+', metavar='PATH',
                        help='Index directories for RAG search')

    # Mode options
    parser.add_argument('--local', action='store_true',
                        help='Use local LLM (downloads model on first use, then offline)')
    parser.add_argument('--simple', '-s', action='store_true',
                        help='Simple CLI mode (no TUI)')

    # Info options
    parser.add_argument('--stats', action='store_true',
                        help='Show learning statistics')
    parser.add_argument('--models', action='store_true',
                        help='List available local models')
    parser.add_argument('--version', '-v', action='store_true',
                        help='Show version')

    # Hidden/advanced
    parser.add_argument('--vac', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--demo', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--generate', type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle --version
    if args.version:
        print(f"BAZINGA v{BAZINGA.VERSION}")
        print(f"  Groq API: {'configured' if os.environ.get('GROQ_API_KEY') else 'not set'}")
        print(f"  Local LLM: {'available' if LOCAL_LLM_AVAILABLE else 'not installed'}")
        return

    # Handle --models
    if args.models:
        from .local_llm import MODELS
        print("Available local models:")
        for name, config in MODELS.items():
            print(f"  {name}: {config['size_mb']}MB - {config['file']}")
        print("\nInstall local AI: pip install llama-cpp-python")
        return

    # Handle --stats
    if args.stats:
        memory = get_memory()
        stats = memory.get_stats()
        print(f"\nBAZINGA Learning Stats:")
        print(f"  Sessions: {stats['total_sessions']}")
        print(f"  Patterns learned: {stats['patterns_learned']}")
        print(f"  Feedback: {stats['positive_feedback']} good, {stats['negative_feedback']} bad")
        if stats['total_feedback'] > 0:
            print(f"  Approval rate: {stats['approval_rate']*100:.1f}%")
        return

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
        if args.local:
            bazinga.use_local = True
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
    bazinga = BAZINGA()
    if args.local:
        bazinga.use_local = True

    if args.simple:
        await bazinga.interactive()
    else:
        # Try TUI mode
        try:
            from .tui import run_tui
            run_tui()
        except ImportError:
            print("Falling back to simple mode...\n")
            await bazinga.interactive()


def main_sync():
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
