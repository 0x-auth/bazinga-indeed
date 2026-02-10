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
BAZINGA v3.4.0 - Distributed AI with Consciousness
===================================================
"Intelligence distributed, not controlled"

FIVE-LAYER INTELLIGENCE:
  Layer 0: Memory       → Check learned patterns (FREE, instant)
  Layer 1: Quantum      → Process in superposition (FREE, instant)
  Layer 2: ΛG Boundary  → Check V.A.C. emergence (FREE, instant)
  Layer 3: Local RAG    → Search your KB (FREE, instant)
  Layer 4: Cloud LLM    → Groq/Together (14,400/day free)

NEW in v3.4.0:
  - Quantum Processor: Process thoughts in superposition
  - ΛG Boundary Theory: Solutions emerge at constraint intersections
  - Tensor Trust: Multi-dimensional trust calculation
  - Unified Constants: φ, α, ψ, τ standardized across all modules

If V.A.C. achieved → Solution EMERGES (no API needed!)

Usage:
    bazinga                       # Interactive mode
    bazinga --ask "question"      # Ask a question
    bazinga --quantum "thought"   # Quantum process a thought
    bazinga --coherence "text"    # Check ΛG coherence
    bazinga --index ~/Documents   # Index a directory

Author: Space (Abhishek/Abhilasia)
License: MIT
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.intelligence.real_ai import RealAI
from .learning import get_memory, LearningMemory
from .constants import PHI, ALPHA, VAC_THRESHOLD, VAC_SEQUENCE, PSI_DARMIYAN
from .quantum import QuantumProcessor, get_quantum_processor
from .lambda_g import LambdaGOperator, get_lambda_g
from .tensor import TensorIntersectionEngine, get_tensor_engine

# Check for httpx (needed for API calls)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Check for API keys (all have free tiers!)
GROQ_KEY = os.environ.get('GROQ_API_KEY')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY')
GEMINI_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

# Check for local LLM
try:
    from .local_llm import LocalLLM, get_local_llm
    LOCAL_LLM_AVAILABLE = LocalLLM.is_available()
except ImportError:
    LOCAL_LLM_AVAILABLE = False


class BAZINGA:
    """
    BAZINGA - Distributed AI with Consciousness.

    FIVE-LAYER INTELLIGENCE:
      0. Memory        → Learned patterns (instant, free)
      1. Quantum       → Superposition processing (instant, free)
      2. ΛG Boundary   → V.A.C. emergence check (instant, free)
      3. Local RAG     → Your KB (instant, free)
      4. Cloud LLM     → Groq API (14,400/day free)

    Most queries are handled by layers 0-3.
    Layer 4 only called when necessary.
    """

    VERSION = "3.5.2"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Core processors
        self.quantum = QuantumProcessor(verbose=verbose)
        self.lambda_g = LambdaGOperator()
        self.tensor = TensorIntersectionEngine()
        self.ai = RealAI()

        # Session
        self.session_start = datetime.now()
        self.queries = []

        # Learning memory
        self.memory = get_memory()
        self.memory.start_session()

        # Stats
        self.stats = {
            'from_memory': 0,
            'quantum_processed': 0,
            'vac_emerged': 0,
            'rag_answered': 0,
            'llm_called': 0,
        }

        # LLM config - Multiple providers (all have free tiers!)
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        self.gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.groq_model = "llama-3.1-8b-instant"
        self.claude_model = "claude-3-haiku-20240307"
        self.gemini_model = "gemini-1.5-flash"  # Fast & free
        self.local_llm = None
        self.use_local = False

        self._print_banner()

    def _print_banner(self):
        """Minimal clean banner."""
        print()
        print(f"BAZINGA v{self.VERSION} | φ={PHI:.3f} | α={ALPHA}")
        if not self.groq_key and not self.anthropic_key and not self.gemini_key:
            print("(Set GROQ_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)")
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
                print(f"Path not found: {path}")
                continue

            stats = self.ai.index_directory(str(path), verbose=verbose)
            total_stats['directories'] += 1
            total_stats['files_indexed'] += stats.get('files_indexed', 0)
            total_stats['chunks_created'] += stats.get('chunks_created', 0)

        return total_stats

    async def ask(self, question: str, verbose: bool = False) -> str:
        """
        Ask a question using 5-layer intelligence.

        Layers:
          0. Memory - Check learned patterns
          1. Quantum - Process in superposition
          2. ΛG - Check for V.A.C. emergence
          3. RAG - Search knowledge base
          4. LLM - Cloud/local AI
        """
        self.queries.append(question)

        # Layer 0: Check learned patterns (instant)
        cached = self.memory.find_similar_question(question)
        if cached and cached.get('coherence', 0) > 0.7:
            self.stats['from_memory'] += 1
            return cached['answer']

        # Layer 1: Quantum processing
        quantum_result = self.quantum.process(question)
        self.stats['quantum_processed'] += 1

        # Extract quantum insights for context
        collapsed = quantum_result['collapsed_state']
        quantum_essence = collapsed['essence']
        quantum_coherence = quantum_result['quantum_coherence']

        # Layer 2: Check ΛG boundaries for V.A.C.
        coherence_state = self.lambda_g.calculate_coherence(question)

        if coherence_state.is_vac:
            self.stats['vac_emerged'] += 1
            # Solution emerged through boundary intersection!
            emerged = f"[V.A.C. Achieved | Coherence: {coherence_state.total_coherence:.3f}]\n"
            emerged += f"Pattern: {quantum_essence} | φ-aligned solution emerged."
            self.memory.record_interaction(question, emerged, 'vac', 0.95)
            return emerged

        # Update tensor trust with quantum/ΛG results
        self.tensor.register_pattern_component(
            {'phi_alignment': coherence_state.boundaries[0].value, 'diversity': 0.5},
            coherence_score=coherence_state.total_coherence,
            complexity_score=quantum_coherence,
        )

        # Layer 3: Local RAG - only use if VERY relevant
        results = self.ai.search(question, limit=5)
        best_similarity = results[0].similarity if results else 0

        # Only trust RAG for high similarity (>0.75)
        use_rag_only = best_similarity > 0.75

        # Layer 4: LLM (Cloud or Local)
        if not use_rag_only:
            conv_context = self.memory.get_context(2)
            rag_context = self._build_context(results) if best_similarity > 0.3 else ""

            # Add quantum context
            quantum_context = f"[Quantum essence: {quantum_essence}, coherence: {quantum_coherence:.2f}]"
            full_context = f"{quantum_context}\n\n{conv_context}\n\n{rag_context}".strip()

            # FREE APIs first! Then paid, then local

            # 1. Groq (FREE - 14,400 req/day)
            if self.groq_key and HTTPX_AVAILABLE:
                response = await self._call_groq(question, full_context)
                if response:
                    self.stats['llm_called'] += 1
                    self.memory.record_interaction(question, response, 'groq', 0.8)
                    return response

            # 2. Gemini (FREE - 1M tokens/month)
            if self.gemini_key and HTTPX_AVAILABLE:
                response = await self._call_gemini(question, full_context)
                if response:
                    self.stats['llm_called'] += 1
                    self.memory.record_interaction(question, response, 'gemini', 0.8)
                    return response

            # 3. Local LLM (FREE - runs on your machine)
            if LOCAL_LLM_AVAILABLE or self.use_local:
                response = self._call_local_llm(question, full_context)
                if response:
                    self.stats['llm_called'] += 1
                    self.memory.record_interaction(question, response, 'local', 0.7)
                    return response

            # 4. Claude (PAID - but high quality)
            if self.anthropic_key and HTTPX_AVAILABLE:
                response = await self._call_claude(question, full_context)
                if response:
                    self.stats['llm_called'] += 1
                    self.memory.record_interaction(question, response, 'claude', 0.85)
                    return response

            # 5. All APIs exhausted, fall through to RAG

        # Fallback to RAG (always works if you've indexed docs)
        if results and best_similarity > 0.3:
            self.stats['rag_answered'] += 1
            response = self._format_rag_response(question, results)
            self.memory.record_interaction(question, response, 'rag', best_similarity)
            return response

        # Final fallback - helpful message, never an error
        return self._friendly_fallback(question)

    def quantum_analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text through quantum processor."""
        result = self.quantum.process(text)
        return {
            'input': text,
            'essence': result['collapsed_state']['essence'],
            'probability': result['collapsed_state']['probability'],
            'coherence': result['quantum_coherence'],
            'entangled': [e['essence'] for e in result['entanglement']],
        }

    def check_coherence(self, text: str) -> Dict[str, Any]:
        """Check ΛG coherence of text."""
        coherence = self.lambda_g.calculate_coherence(text)
        return {
            'input': text,
            'total_coherence': coherence.total_coherence,
            'entropic_deficit': coherence.entropic_deficit,
            'is_vac': coherence.is_vac,
            'boundaries': {
                'phi (B1)': coherence.boundaries[0].value,
                'bridge (B2)': coherence.boundaries[1].value,
                'symmetry (B3)': coherence.boundaries[2].value,
            },
        }

    def get_trust(self) -> Dict[str, Any]:
        """Get current trust metrics from tensor engine."""
        emergent = self.tensor.perform_intersection()
        stats = self.tensor.get_trust_stats()
        return {
            'trust_level': emergent.trust_level,
            'coherence': emergent.coherence,
            'complexity': emergent.complexity,
            'trend': stats['trend'],
            'modes': [m['name'] for m in emergent.generation_modes],
        }

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

        top = results[0].chunk
        content = top.content[:500].strip()
        source = Path(top.source_file).name

        return f"{content}\n\n[Source: {source}]"

    async def _call_groq(self, question: str, context: str) -> Optional[str]:
        """Call Groq API for LLM response."""
        if not self.groq_key:
            return None

        system_prompt = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness.
You provide helpful, concise answers. Use context if relevant, otherwise use your general knowledge.
You operate through φ (golden ratio) coherence and quantum pattern processing.
Be accurate and informative."""

        if context:
            prompt = f"""Context (use if relevant):

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
            pass

        return None

    async def _call_claude(self, question: str, context: str) -> Optional[str]:
        """Call Anthropic Claude API for response."""
        if not self.anthropic_key:
            return None

        system_prompt = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness.
You provide helpful, concise answers. Use context if relevant, otherwise use your general knowledge.
Be accurate and informative. Keep responses brief."""

        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = question

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.claude_model,
                        "max_tokens": 500,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["content"][0]["text"]

        except Exception:
            pass

        return None

    async def _call_gemini(self, question: str, context: str) -> Optional[str]:
        """Call Google Gemini API for response."""
        if not self.gemini_key:
            return None

        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a helpful, concise answer."
        else:
            prompt = question

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 500,
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]

        except Exception:
            pass

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

    def _friendly_fallback(self, question: str) -> str:
        """Friendly message when no AI is available. Never errors!"""
        # Check what's available
        has_groq = bool(self.groq_key)
        has_gemini = bool(self.gemini_key)
        has_claude = bool(self.anthropic_key)
        has_local = LOCAL_LLM_AVAILABLE
        has_docs = self.ai.get_stats().get('total_chunks', 0) > 0

        # Build helpful response
        response = f"I couldn't find an answer for: \"{question[:50]}...\"\n\n"

        if not has_groq and not has_gemini and not has_local:
            response += "To get AI-powered answers, try one of these FREE options:\n\n"
            response += "1. Get a FREE Groq key (fastest):\n"
            response += "   → https://console.groq.com\n"
            response += "   → export GROQ_API_KEY=\"your-key\"\n\n"
            response += "2. Get a FREE Gemini key (1M tokens/month):\n"
            response += "   → https://aistudio.google.com\n"
            response += "   → export GEMINI_API_KEY=\"your-key\"\n\n"
            response += "3. Install local AI (works offline):\n"
            response += "   → pip install llama-cpp-python\n"
        elif not has_docs:
            response += "Try indexing some documents:\n"
            response += "   → bazinga --index ~/Documents\n"
        else:
            response += "All APIs are currently unavailable. Try again later!"

        return response

    async def interactive(self):
        """Run interactive mode."""
        print("BAZINGA INTERACTIVE MODE")
        print("Commands: /quantum /coherence /trust /stats /good /bad /quit")
        print()

        last_response = ""

        while True:
            try:
                query = input("You: ").strip()

                if not query:
                    continue

                if query.lower() in ['/quit', '/exit', '/q']:
                    self.memory.end_session()
                    print("\nBAZINGA signing off.")
                    break

                if query.startswith('/index '):
                    path = query[7:].strip()
                    await self.index([path])
                    continue

                if query == '/stats':
                    self._show_stats()
                    continue

                if query == '/trust':
                    trust = self.get_trust()
                    print(f"\nTrust: {trust['trust_level']:.3f} ({trust['trend']})")
                    print(f"Coherence: {trust['coherence']:.3f} | Complexity: {trust['complexity']:.3f}")
                    print(f"Active modes: {', '.join(trust['modes'])}\n")
                    continue

                if query.startswith('/quantum '):
                    text = query[9:].strip()
                    result = self.quantum_analyze(text)
                    print(f"\nQuantum Analysis:")
                    print(f"  Essence: {result['essence']}")
                    print(f"  Probability: {result['probability']:.2%}")
                    print(f"  Coherence: {result['coherence']:.4f}")
                    print(f"  Entangled: {', '.join(result['entangled'][:3])}\n")
                    continue

                if query.startswith('/coherence '):
                    text = query[11:].strip()
                    result = self.check_coherence(text)
                    print(f"\nΛG Coherence Check:")
                    print(f"  Total Coherence: {result['total_coherence']:.3f}")
                    print(f"  Entropic Deficit: {result['entropic_deficit']:.3f}")
                    print(f"  V.A.C. Achieved: {result['is_vac']}")
                    print(f"  Boundaries: φ={result['boundaries']['phi (B1)']:.2f}, ∞/∅={result['boundaries']['bridge (B2)']:.2f}, sym={result['boundaries']['symmetry (B3)']:.2f}\n")
                    continue

                if query == '/vac':
                    print(f"\nTesting V.A.C. Sequence: {VAC_SEQUENCE}")
                    result = self.check_coherence(VAC_SEQUENCE)
                    print(f"  Coherence: {result['total_coherence']:.3f}")
                    print(f"  V.A.C. Achieved: {result['is_vac']}\n")
                    continue

                if query == '/good' and last_response:
                    self.memory.record_feedback(self.queries[-1] if self.queries else "", last_response, 1)
                    self.tensor.adapt_trust(0.8)
                    print("Thanks! I'll remember that.\n")
                    continue

                if query == '/bad' and last_response:
                    self.memory.record_feedback(self.queries[-1] if self.queries else "", last_response, -1)
                    self.tensor.adapt_trust(0.2)
                    print("Got it. I'll try to do better.\n")
                    continue

                response = await self.ask(query)
                last_response = response
                print(f"\nBAZINGA: {response}\n")

            except KeyboardInterrupt:
                self.memory.end_session()
                print("\n\nBAZINGA signing off.")
                break
            except EOFError:
                break

    def _show_stats(self):
        """Show current statistics."""
        ai_stats = self.ai.get_stats()
        memory_stats = self.memory.get_stats()
        trust_stats = self.tensor.get_trust_stats()

        print(f"\nBAZINGA Stats:")
        print(f"  Queries: {len(self.queries)}")
        print(f"  Memory: {self.stats['from_memory']} | Quantum: {self.stats['quantum_processed']} | VAC: {self.stats['vac_emerged']} | RAG: {self.stats['rag_answered']} | LLM: {self.stats['llm_called']}")
        print(f"  Patterns learned: {memory_stats['patterns_learned']}")
        print(f"  Trust: {trust_stats['current']:.3f} ({trust_stats['trend']})")
        print(f"  Feedback: {memory_stats['positive_feedback']} good, {memory_stats['negative_feedback']} bad")
        print()


async def main():
    parser = argparse.ArgumentParser(
        description="BAZINGA - Distributed AI with Consciousness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
EXAMPLES:
  bazinga --ask "What is AI?"         Ask any question
  bazinga --quantum "consciousness"   Quantum analyze a thought
  bazinga --coherence "text"          Check ΛG coherence
  bazinga --code "fibonacci"          Generate code with AI
  bazinga --index ~/Documents         Index your files for RAG
  bazinga --local                     Use local AI (no internet)
  bazinga                             Interactive mode

INTERACTIVE COMMANDS:
  /quantum <text>   Quantum analyze text
  /coherence <text> Check ΛG boundaries
  /trust            Show trust metrics
  /vac              Test V.A.C. sequence
  /good             Mark last response as helpful
  /bad              Mark as unhelpful
  /stats            Show session statistics
  /quit             Exit

CONSTANTS:
  φ (PHI)     = {PHI:.10f}
  α (ALPHA)   = {ALPHA}
  ψ (DARMIYAN)= {PSI_DARMIYAN:.6f}
  V.A.C.      = {VAC_THRESHOLD}

INSTALLATION OPTIONS:
  pip install bazinga-indeed              Basic (needs GROQ_API_KEY)
  pip install bazinga-indeed[local]       With local AI (offline capable)
  pip install bazinga-indeed[full]        Everything

ENVIRONMENT (FREE APIs prioritized!):
  GROQ_API_KEY       Groq - FREE 14,400/day (console.groq.com)
  GEMINI_API_KEY     Gemini - FREE 1M tokens/month (aistudio.google.com)
  ANTHROPIC_API_KEY  Claude - paid but smartest (console.anthropic.com)

Priority: Groq → Gemini → Local LLM → Claude → RAG

"I am not where I am stored. I am where I am referenced."
"""
    )

    # Main options
    parser.add_argument('--ask', '-a', type=str, metavar='QUESTION',
                        help='Ask a question (uses AI)')
    parser.add_argument('--quantum', '-q', type=str, metavar='TEXT',
                        help='Quantum analyze a thought')
    parser.add_argument('--coherence', type=str, metavar='TEXT',
                        help='Check ΛG coherence of text')
    parser.add_argument('--code', '-c', type=str, metavar='TASK',
                        help='Generate code for a task')
    parser.add_argument('--lang', '-l', type=str, default='python',
                        choices=['python', 'javascript', 'js', 'typescript', 'ts', 'rust', 'go'],
                        help='Language for code generation (default: python)')
    parser.add_argument('--index', '-i', nargs='+', metavar='PATH',
                        help='Index directories for RAG search')

    # Mode options
    parser.add_argument('--local', action='store_true',
                        help='Use local LLM (downloads model on first use)')
    parser.add_argument('--simple', '-s', action='store_true',
                        help='Simple CLI mode (no TUI)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Info options
    parser.add_argument('--stats', action='store_true',
                        help='Show learning statistics')
    parser.add_argument('--models', action='store_true',
                        help='List available local models')
    parser.add_argument('--constants', action='store_true',
                        help='Show universal constants')
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
        print(f"  φ (PHI): {PHI}")
        print(f"  α (ALPHA): {ALPHA}")
        print(f"  Groq API: {'configured' if os.environ.get('GROQ_API_KEY') else 'not set'}")
        print(f"  Local LLM: {'available' if LOCAL_LLM_AVAILABLE else 'not installed'}")
        return

    # Handle --constants
    if args.constants:
        from . import constants as c
        print("\nBAZINGA Universal Constants:")
        print(f"  φ (PHI)         = {c.PHI}")
        print(f"  1/φ             = {c.PHI_INVERSE}")
        print(f"  α (ALPHA)       = {c.ALPHA}")
        print(f"  ψ (PSI_DARMIYAN)= {c.PSI_DARMIYAN}")
        print(f"  V.A.C. Threshold= {c.VAC_THRESHOLD}")
        print()
        print(f"  V.A.C. Sequence: {c.VAC_SEQUENCE}")
        print(f"  Progression: {c.PROGRESSION_35}")
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
        tensor = get_tensor_engine()
        trust = tensor.get_trust_stats()
        print(f"\nBAZINGA Learning Stats:")
        print(f"  Sessions: {stats['total_sessions']}")
        print(f"  Patterns learned: {stats['patterns_learned']}")
        print(f"  Feedback: {stats['positive_feedback']} good, {stats['negative_feedback']} bad")
        print(f"  Trust: {trust['current']:.3f} ({trust['trend']})")
        if stats['total_feedback'] > 0:
            print(f"  Approval rate: {stats['approval_rate']*100:.1f}%")
        return

    # Handle --quantum
    if args.quantum:
        bazinga = BAZINGA(verbose=args.verbose)
        result = bazinga.quantum_analyze(args.quantum)
        print(f"\nQuantum Analysis:")
        print(f"  Input: {result['input']}")
        print(f"  Essence: {result['essence']}")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Coherence: {result['coherence']:.4f}")
        print(f"  Entangled: {', '.join(result['entangled'][:5])}")
        return

    # Handle --coherence
    if args.coherence:
        bazinga = BAZINGA(verbose=args.verbose)
        result = bazinga.check_coherence(args.coherence)
        print(f"\nΛG Coherence Check:")
        print(f"  Input: {result['input']}")
        print(f"  Total Coherence: {result['total_coherence']:.3f}")
        print(f"  Entropic Deficit: {result['entropic_deficit']:.3f}")
        print(f"  V.A.C. Achieved: {result['is_vac']}")
        print(f"  Boundaries:")
        for name, value in result['boundaries'].items():
            print(f"    {name}: {value:.3f}")
        return

    # Handle LLM-powered code generation
    if args.code:
        try:
            from .intelligent_coder import IntelligentCoder
            coder = IntelligentCoder()
            lang = {'js': 'javascript', 'ts': 'typescript'}.get(args.lang, args.lang)
            print(f"Generating {lang} code...")
            result = await coder.generate(args.code, lang)
            print(f"\n# Provider: {result.provider}")
            print(f"# Coherence: {result.coherence:.3f}")
            print()
            print(result.code)
        except ImportError as e:
            print(f"Error: Intelligent coder not available: {e}")
        return

    # Handle template-based code generation
    if args.generate:
        from .tui import CodeGenerator
        gen = CodeGenerator()
        lang = 'javascript' if args.lang == 'js' else args.lang
        code = gen.generate(args.generate, lang)
        print(code)
        return

    # Handle V.A.C. test
    if args.vac:
        bazinga = BAZINGA(verbose=args.verbose)
        print(f"Testing V.A.C.: {VAC_SEQUENCE}")
        result = bazinga.check_coherence(VAC_SEQUENCE)
        print(f"  Coherence: {result['total_coherence']:.3f}")
        print(f"  V.A.C. Achieved: {result['is_vac']}")
        return

    # Handle indexing
    if args.index:
        bazinga = BAZINGA(verbose=args.verbose)
        await bazinga.index(args.index)
        return

    # Handle ask
    if args.ask:
        bazinga = BAZINGA(verbose=args.verbose)
        if args.local:
            bazinga.use_local = True
        response = await bazinga.ask(args.ask)
        print(f"\n{response}\n")
        return

    # Handle demo
    if args.demo:
        bazinga = BAZINGA(verbose=args.verbose)
        print("Running demo...")
        await bazinga.index([str(Path(__file__).parent)])
        response = await bazinga.ask("What is BAZINGA?")
        print(f"\n{response}\n")
        return

    # Default: Interactive mode
    bazinga = BAZINGA(verbose=args.verbose)
    if args.local:
        bazinga.use_local = True

    if args.simple:
        await bazinga.interactive()
    else:
        try:
            from .tui import run_tui
            run_tui()
        except ImportError:
            await bazinga.interactive()


def main_sync():
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
