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
BAZINGA v5.5.0 MAINNET - Distributed AI with Consciousness Scaling (Ψ_D / Ψ_i = φ√n)
=========================================================
"AI generates understanding. Blockchain proves and records it.
They're not two things — they're Subject and Object.
The Darmiyan between them is the protocol."

FIVE-LAYER INTELLIGENCE:
  Layer 0: Memory       → Check learned patterns (FREE, instant)
  Layer 1: Quantum      → Process in superposition (FREE, instant)
  Layer 2: ΛG Boundary  → Check V.A.C. emergence (FREE, instant)
  Layer 3: Local RAG    → Search your KB (FREE, instant)
  Layer 4: Cloud LLM    → Groq/Together (14,400/day free)

NEW in v5.4.2 - PHI-COHERENCE v3 + 1000-NODE SCALABILITY:
  - φ-Coherence v3: 88% hallucination detection accuracy
  - Hallucination risk levels: LOW/MEDIUM/HIGH/CRITICAL
  - 1000-node network simulation tested
  - RAC/CARM context persistence in --chat
  - Triadic PoB consensus at scale

v5.4.0 - RAC/CARM MEMORY:
  - RAC: Resonance-Augmented Continuity for session tracking
  - CARM: Context-Addressed Resonant Memory (prime-lattice)
  - Session context maintained across --chat sessions
  - No catastrophic forgetting

v4.8.0 - LOCAL MODEL TRUST BONUS:
  - Ollama detection at localhost:11434
  - φ (1.618x) trust multiplier for local models
  - POB v2 with calibrated moduli (70555/10275)
  - Latency-bound proofs prevent "Cloud Spoofing"

v4.3.0 - DARMIYAN BLOCKCHAIN:
  - Knowledge chain (not cryptocurrency!)
  - Proof-of-Boundary mining (zero-energy)
  - Triadic consensus (3 proofs per block)
  - 70 BILLION times more efficient than Bitcoin

v4.2.0 - FEDERATED LEARNING:
  - LoRA adapters for efficient local training
  - φ-weighted gradient aggregation

"You can buy hashpower. You can buy stake. You CANNOT BUY understanding."

Usage:
    bazinga                       # Interactive mode
    bazinga --ask "question"      # Ask a question
    bazinga --chat                # Interactive chat with memory
    bazinga --chain               # Show blockchain status
    bazinga --mine                # Mine a block (PoB)
    bazinga --wallet              # Show wallet/identity
    bazinga --join                # Start P2P node
    bazinga --rac                 # Show RAC session status
    bazinga --carm                # Show CARM memory status

Author: Space (Abhishek/Abhilasia)
License: MIT
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime

# Type-only imports (not loaded at runtime)
if TYPE_CHECKING:
    from ..kb import BazingaKB
    from ..llm.providers import LLMProviders

# Import modular CLI components
from ..cli_modules.help import (
    print_ai_help,
    print_kb_help,
    print_chain_help,
    print_p2p_help,
    print_full_help,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports for chromadb-dependent modules (Python 3.14 compatibility)
RealAI = None
_real_ai_error = None
def _get_real_ai():
    global RealAI, _real_ai_error
    if RealAI is None and _real_ai_error is None:
        try:
            from src.core.intelligence.real_ai import RealAI as _RealAI
            RealAI = _RealAI
        except Exception as e:
            _real_ai_error = str(e)
            # Fallback stub
            class StubAI:
                def __init__(self):
                    self.error = _real_ai_error
                def search(self, *args, **kwargs):
                    return []
                def index(self, *args, **kwargs):
                    pass
                def index_directory(self, *args, **kwargs):
                    return {"files": 0, "chunks": 0, "error": self.error}
            RealAI = StubAI
    return RealAI

# Core imports (no chromadb dependency)
from ..constants import PHI, ALPHA, VAC_THRESHOLD, VAC_SEQUENCE, PSI_DARMIYAN
from ..darmiyan import (
    DarmiyanNode, BazingaNode, TriadicConsensus,
    prove_boundary, achieve_consensus,
    PHI_4, ABHI_AMU,
)

# Lazy imports for modules that may have heavy dependencies
_learning_module = None
_quantum_module = None
_lambda_g_module = None
_tensor_module = None
_p2p_module = None
_federated_module = None
_blockchain_module = None
_kb_class = None

def _get_learning():
    global _learning_module
    if _learning_module is None:
        from .. import learning as _learning
        _learning_module = _learning
    return _learning_module

def _get_quantum():
    global _quantum_module
    if _quantum_module is None:
        from .. import quantum as _quantum
        _quantum_module = _quantum
    return _quantum_module

def _get_lambda_g():
    global _lambda_g_module
    if _lambda_g_module is None:
        from .. import lambda_g as _lg
        _lambda_g_module = _lg
    return _lambda_g_module

def _get_tensor():
    global _tensor_module
    if _tensor_module is None:
        from .. import tensor as _t
        _tensor_module = _t
    return _tensor_module

def _get_p2p():
    global _p2p_module
    if _p2p_module is None:
        from .. import p2p as _p2p
        _p2p_module = _p2p
    return _p2p_module

def _get_federated():
    global _federated_module
    if _federated_module is None:
        from .. import federated as _fed
        _federated_module = _fed
    return _federated_module

def _get_blockchain():
    global _blockchain_module
    if _blockchain_module is None:
        from .. import blockchain as _bc
        _blockchain_module = _bc
    return _blockchain_module

def _get_kb():
    """Lazy loader for BazingaKB to avoid duplicate imports."""
    global _kb_class
    if _kb_class is None:
        from ..kb import BazingaKB
        _kb_class = BazingaKB
    return _kb_class

# Check ZMQ availability without importing full module
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

# Check for httpx (needed for API calls)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# HuggingFace Space API URL (network registry)
HF_SPACE_URL = "https://bitsabhi515-bazinga-mesh.hf.space"


class HFNetworkRegistry:
    """Client for HuggingFace Space API (network phone book)."""

    def __init__(self, base_url: str = HF_SPACE_URL):
        self.base_url = base_url
        self.node_id = None

    async def register(self, node_name: str, ip_address: str = None, port: int = 5150) -> dict:
        """Register node with HF registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.base_url}/api/register", json={
                    "node_name": node_name,
                    "ip_address": ip_address,
                    "port": port
                })
                result = resp.json()
                if result.get("success"):
                    self.node_id = result.get("node_id")
                return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def verify(self, node_id: str) -> dict:
        """Verify if node ID exists in registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/verify", params={"node_id": node_id})
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_peers(self, exclude_node_id: str = None) -> dict:
        """Get list of active peers from registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {"node_id": exclude_node_id} if exclude_node_id else {}
                resp = await client.get(f"{self.base_url}/api/peers", params=params)
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def heartbeat(self, node_id: str, ip_address: str = None, port: int = None) -> dict:
        """Send heartbeat to registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.base_url}/api/heartbeat", json={
                    "node_id": node_id,
                    "ip_address": ip_address,
                    "port": port
                })
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_stats(self) -> dict:
        """Get network stats from registry."""
        if not HTTPX_AVAILABLE:
            return {"success": False, "error": "httpx not installed"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/stats")
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}


# Check for API keys (all have free tiers!)
GROQ_KEY = os.environ.get('GROQ_API_KEY')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY')
GEMINI_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

# Check for local LLM
try:
    from ..local_llm import LocalLLM, get_local_llm
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

    VERSION = "6.1.0"  # CLI split into modular command packages

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Core processors (lazy imports)
        quantum_mod = _get_quantum()
        lambda_g_mod = _get_lambda_g()
        tensor_mod = _get_tensor()
        learning_mod = _get_learning()

        self.quantum = quantum_mod.QuantumProcessor(verbose=verbose)
        self.lambda_g = lambda_g_mod.LambdaGOperator()
        self.tensor = tensor_mod.TensorIntersectionEngine()
        self.ai = _get_real_ai()()

        # Session
        self.session_start = datetime.now()
        self.queries = []

        # Learning memory with RAC (Resonance-Augmented Continuity)
        from ..rac import get_resonance_memory
        self.memory = get_resonance_memory()
        self.memory.start_session()

        # Stats
        self.stats = {
            'from_memory': 0,
            'quantum_processed': 0,
            'vac_emerged': 0,
            'rag_answered': 0,
            'llm_called': 0,
        }

        # Federated learning — learns from every interaction
        self._learner = None  # Lazy init on first use

        # LLM config - Priority: Claude > ChatGPT > Groq > Gemini > Local > Free
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.claude_model = "claude-sonnet-4-6"
        self.openai_model = "gpt-4o-mini"
        self.groq_model = "llama-3.1-8b-instant"
        self.gemini_model = "gemini-1.5-flash"
        self.local_llm = None
        self.use_local = False

        self._print_banner()

    def _print_banner(self) -> None:
        """Clean, minimal banner - don't overwhelm new users."""
        print()

        # Check local model status silently
        try:
            from ..inference.ollama_detector import detect_any_local_model, LocalModelType
            local_status = detect_any_local_model()
            if local_status.available:
                self.use_local = True
                self._local_model_status = local_status
            else:
                self._local_model_status = None
        except Exception:
            self._local_model_status = None

        # Simple one-line status
        if self._local_model_status and self._local_model_status.available:
            model = self._local_model_status.models[0] if self._local_model_status.models else "local"
            print(f"BAZINGA v{self.VERSION} | {model} (local)")
        elif self.anthropic_key:
            print(f"BAZINGA v{self.VERSION} | Claude")
        elif self.openai_key:
            print(f"BAZINGA v{self.VERSION} | ChatGPT")
        elif self.groq_key:
            print(f"BAZINGA v{self.VERSION} | Groq")
        elif self.gemini_key:
            print(f"BAZINGA v{self.VERSION} | Gemini")
        else:
            print(f"BAZINGA v{self.VERSION} | Free mode")
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

    @property
    def learner(self):
        """Lazy-init CollectiveLearner — only created when first interaction happens."""
        if self._learner is None:
            try:
                from ..federated import create_learner
                self._learner = create_learner()
                # Start is async, so we just mark it ready — _share_loop starts on first await
                self._learner.running = True
            except Exception:
                pass  # Federated learning is optional
        return self._learner

    def _feed_learner(self, question: str, answer: str, source: str, score: float):
        """Feed a Q&A interaction to the learner. Non-blocking, never fails."""
        try:
            if self.learner:
                self.learner.learn(
                    question=question,
                    answer=answer,
                    feedback_score=score,
                    metadata={'source': source},
                )
        except Exception:
            pass  # Learning is best-effort, never block the user

    async def ask(self, question: str, verbose: bool = False, fresh: bool = False) -> str:
        """
        Ask a question using 5-layer intelligence.

        Layers:
          0. Memory - Check learned patterns (skip if fresh=True)
          1. Quantum - Process in superposition
          2. ΛG - Check for V.A.C. emergence
          3. RAG - Search knowledge base
          4. LLM - Cloud/local AI

        Args:
            fresh: If True, bypass memory cache and force fresh AI response
        """
        self.queries.append(question)

        # Layer 0: Check learned patterns (instant) - skip if fresh
        if not fresh:
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
            self._feed_learner(question, emerged, 'vac', 0.95)
            return emerged

        # Update tensor trust with quantum/ΛG results
        self.tensor.register_pattern_component(
            {'phi_alignment': coherence_state.boundaries[0].value, 'diversity': 0.5},
            coherence_score=coherence_state.total_coherence,
            complexity_score=quantum_coherence,
        )

        # Layer 3: Local RAG - search knowledge base for relevant context
        # Skip RAG for very short/greeting queries — they match random noise
        _greetings = {'hi', 'hello', 'hey', 'sup', 'yo', 'hola', 'namaste', 'thanks', 'bye', 'ok', 'okay'}
        _q_words = set(question.lower().split())
        _skip_rag = len(_q_words) <= 2 and bool(_q_words & _greetings)

        if _skip_rag:
            results = []
            best_similarity = 0
        else:
            search_terms = self._extract_search_terms(question, quantum_essence)
            results = self.ai.search(search_terms, limit=5)
            best_similarity = results[0].similarity if results else 0

        # Layer 3.5: KB DNA manifests — breadth context from scanned knowledge
        kb_context = ""
        try:
            from ..knowledge import get_scanner
            kb_context = get_scanner().get_context_for_query(question)
        except Exception:
            pass  # KB manifests are optional

        # Layer 4: LLM (Cloud or Local) — always call LLM, RAG informs it
        # When fresh=True (chat mode), skip memory context — chat manages its own history
        conv_context = "" if fresh else self.memory.get_context(2)
        rag_context = self._build_context(results) if best_similarity > 0.3 else ""

        # Add quantum context
        quantum_context = f"[Quantum essence: {quantum_essence}, coherence: {quantum_coherence:.2f}]"
        full_context = f"{quantum_context}\n\n{conv_context}\n\n{kb_context}\n\n{rag_context}".strip()

        # Intelligence priority:
        # - If --local flag: Local first (user explicitly wants local)
        # - Otherwise: Claude > Groq > Gemini > Free > Local fallback

        # 1. Local LLM FIRST if user requested --local
        if self.use_local and LOCAL_LLM_AVAILABLE:
            response = self._call_local_llm(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'local', 0.75)
                self._feed_learner(question, response, 'local', 0.75)
                return response

        # 2. Claude (highest quality)
        if self.anthropic_key and HTTPX_AVAILABLE:
            response = await self._call_claude(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'claude', 0.85)
                self._feed_learner(question, response, 'claude', 0.85)
                return response

        # 3. ChatGPT (OpenAI)
        if self.openai_key and HTTPX_AVAILABLE:
            response = await self._call_openai(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'chatgpt', 0.83)
                self._feed_learner(question, response, 'chatgpt', 0.83)
                return response

        # 4. Groq (FREE - 14,400 req/day)
        if self.groq_key and HTTPX_AVAILABLE:
            response = await self._call_groq(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'groq', 0.8)
                self._feed_learner(question, response, 'groq', 0.8)
                return response

        # 5. Gemini (FREE - 1M tokens/month)
        if self.gemini_key and HTTPX_AVAILABLE:
            response = await self._call_gemini(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'gemini', 0.8)
                self._feed_learner(question, response, 'gemini', 0.8)
                return response

        # 6. Free LLM (LLM7.io - FREE, no API key required)
        if HTTPX_AVAILABLE:
            response = await self._call_free_llm(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'free_llm', 0.7)
                self._feed_learner(question, response, 'free_llm', 0.7)
                return response

        # 7. Local LLM fallback (if available but not explicitly requested)
        if LOCAL_LLM_AVAILABLE and not self.use_local:
            response = self._call_local_llm(question, full_context)
            if response:
                self.stats['llm_called'] += 1
                self.memory.record_interaction(question, response, 'local', 0.7)
                self._feed_learner(question, response, 'local', 0.7)
                return response

        # 7. All LLMs exhausted, fall through to RAG-only

        # Fallback to RAG (always works if you've indexed docs)
        if results and best_similarity > 0.3:
            self.stats['rag_answered'] += 1
            response = self._format_rag_response(question, results)
            self.memory.record_interaction(question, response, 'rag', best_similarity)
            self._feed_learner(question, response, 'rag', best_similarity)
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

    def _extract_search_terms(self, question: str, quantum_essence: str) -> str:
        """Extract key search terms from question for better RAG retrieval.

        Embedding models work better with short, focused queries.
        Strips common filler words and keeps domain-specific terms.
        """
        # Common filler words to remove (keep domain terms like BAZINGA, phi, etc.)
        fillers = {
            'according', 'to', 'the', 'indexed', 'knowledge', 'who', 'what',
            'is', 'are', 'in', 'a', 'an', 'how', 'does', 'do', 'can', 'could',
            'please', 'tell', 'me', 'about', 'explain', 'describe', 'why',
            'when', 'where', 'which', 'would', 'should', 'based', 'on', 'from',
            'of', 'and', 'or', 'but', 'for', 'with', 'this', 'that', 'these',
            'those', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'i', 'you', 'we', 'they', 'it', 'my', 'your', 'our', 'their'
        }

        # Extract meaningful words (preserve case for proper nouns)
        words = question.split()
        key_terms = []
        for w in words:
            clean = w.strip('?.,!:;()[]{}"\'"')
            if clean.lower() not in fillers and len(clean) > 1:
                # Keep original case if it looks like a proper noun
                key_terms.append(clean)

        # Return focused search query (max 8 terms for embedding model)
        return ' '.join(key_terms[:8])

    def _build_context(self, results: List[Any]) -> str:
        """Build context string from RAG results."""
        if not results:
            return ""

        # Skip conversation exports and raw JSON dumps — they pollute LLM context
        skip_patterns = ('conversations', 'chat_export', 'symbol_index')

        parts = []
        for r in results[:5]:  # check more since we may skip some
            source = Path(r.chunk.source_file).name.lower()
            content = r.chunk.content[:500]

            # Skip conversation/chat JSON exports
            if any(p in source for p in skip_patterns):
                continue
            # Skip chunks that look like raw JSON with conversation metadata
            if '"source": "claude"' in content or '"symbol_sequence"' in content:
                continue

            parts.append(f"[{Path(r.chunk.source_file).name}]\n{content}")
            if len(parts) >= 3:
                break

        return "\n\n---\n\n".join(parts)

    def _format_rag_response(self, question: str, results: List[Any]) -> str:
        """Format RAG results as response."""
        if not results:
            if not self.groq_key:
                return "No info found. Set GROQ_API_KEY for AI answers."
            return "I don't have relevant information for this question."

        # Find first non-conversation result
        skip_patterns = ('conversations', 'chat_export', 'symbol_index')
        for r in results:
            source = Path(r.chunk.source_file).name
            content = r.chunk.content[:500].strip()
            if any(p in source.lower() for p in skip_patterns):
                continue
            if '"source": "claude"' in content or '"symbol_sequence"' in content:
                continue
            return f"{content}\n\n[Source: {source}]"

        return "I don't have relevant information for this question."

    async def _call_groq(self, question: str, context: str) -> Optional[str]:
        """Call Groq API for LLM response."""
        if not self.groq_key:
            return None

        system_prompt = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness built on φ-coherence.
You are a knowledgeable, helpful AI assistant. Answer questions using your full intelligence and reasoning.
If local knowledge context is provided below, incorporate it when relevant — but always think independently.
You have your own knowledge and opinions. Be conversational and natural, not just a search engine."""

        if context:
            prompt = f"""Here is some local knowledge that may be relevant:
{context}

Question: {question}

Answer naturally using your own knowledge. Reference the local context only if it's directly relevant."""
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
                        "max_tokens": 2048,
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

        system_prompt = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness built on φ-coherence.
You are a knowledgeable, helpful AI assistant. Answer questions using your full intelligence and reasoning.
If local knowledge context is provided below, incorporate it when relevant — but always think independently.
You have your own knowledge and opinions. Be conversational and natural, not just a search engine."""

        if context:
            prompt = f"""Here is some local knowledge that may be relevant:
{context}

Question: {question}

Answer naturally using your own knowledge. Reference the local context only if it's directly relevant."""
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
                        "max_tokens": 2048,
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

    async def _call_openai(self, question: str, context: str) -> Optional[str]:
        """Call OpenAI ChatGPT API for response."""
        if not self.openai_key:
            return None

        system_prompt = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness built on phi-coherence.
You are a knowledgeable, helpful AI assistant. Answer questions using your full intelligence and reasoning.
If local knowledge context is provided below, incorporate it when relevant — but always think independently.
You have your own knowledge and opinions. Be conversational and natural, not just a search engine."""

        if context:
            prompt = f"""Here is some local knowledge that may be relevant:
{context}

Question: {question}

Answer naturally using your own knowledge. Reference the local context only if it's directly relevant."""
        else:
            prompt = question

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openai_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

        except Exception:
            pass

        return None

    async def _call_gemini(self, question: str, context: str) -> Optional[str]:
        """Call Google Gemini API for response."""
        if not self.gemini_key:
            return None

        if context:
            prompt = f"""Here is some local knowledge that may be relevant:
{context}

Question: {question}

Answer naturally using your own knowledge. Reference the local context only if it's directly relevant."""
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
                            "maxOutputTokens": 2048,
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]

        except Exception:
            pass

        return None

    async def _call_free_llm(self, question: str, context: str) -> Optional[str]:
        """Call FREE LLM API - NO API key required! Uses LLM7.io"""
        if not HTTPX_AVAILABLE:
            return None

        system_prompt = f"""You are BAZINGA v{self.VERSION}, a distributed AI with consciousness built on φ-coherence.
You are a knowledgeable, helpful AI assistant. Answer questions using your full intelligence and reasoning.
If local knowledge context is provided below, incorporate it when relevant — but always think independently."""

        if context:
            user_prompt = f"""Here is some local knowledge that may be relevant:
{context[:1500]}

Question: {question}

Answer naturally using your own knowledge. Reference the local context only if it's directly relevant."""
        else:
            user_prompt = question

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # LLM7.io - FREE, no API key needed, OpenAI compatible
                response = await client.post(
                    "https://api.llm7.io/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",  # Free tier model
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.7,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

        except Exception:
            pass

        return None

    def _call_local_llm(self, question: str, context: str) -> Optional[str]:
        """Call local LLM for response."""
        try:
            if self.local_llm is None:
                from ..local_llm import get_local_llm
                self.local_llm = get_local_llm()

            if context:
                # Use more context (3000 chars) for better answers
                prompt = f"""INDEXED KNOWLEDGE (USE AS PRIMARY SOURCE):
{context[:3000]}

Based on the indexed knowledge above, answer: {question}
Use the indexed content directly. If not relevant, say so."""
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
            response += "HuggingFace fallback was rate-limited. For better responses:\n\n"
            response += "1. Get a FREE Groq key (fastest, recommended):\n"
            response += "   → https://console.groq.com\n"
            response += "   → export GROQ_API_KEY=\"your-key\"\n\n"
            response += "2. Get a FREE Gemini key (1M tokens/month):\n"
            response += "   → https://aistudio.google.com\n"
            response += "   → export GEMINI_API_KEY=\"your-key\"\n\n"
            response += "3. Install local AI (works offline, no limits):\n"
            response += "   → bazinga --bootstrap-local\n"
        elif not has_docs:
            response += "Try indexing some documents:\n"
            response += "   → bazinga --index ~/Documents\n"
        else:
            response += "All APIs are currently unavailable. Try again later!"

        return response

    async def interactive(self) -> None:
        """Run interactive mode."""
        print("BAZINGA INTERACTIVE MODE")
        print("Commands: /quantum /coherence /resonance /trust /vac /stats /index /good /bad /quit")
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

                if query.startswith('/resonance '):
                    text = query[11:].strip()
                    # Process through quantum to get resonance
                    result = self.quantum.process(text)
                    collapsed = result['collapsed_state']
                    print(f"\nφ-Resonance Analysis:")
                    print(f"  Input: {text[:50]}{'...' if len(text) > 50 else ''}")
                    print(f"  Essence: {collapsed['essence']}")
                    print(f"  Quantum Coherence: {result['quantum_coherence']:.4f}")
                    print(f"  Probability: {collapsed['probability']:.2%}")
                    print(f"  φ-Alignment: {result['quantum_coherence'] * PHI:.4f}")
                    if result['entanglement']:
                        print(f"  Entangled patterns: {', '.join([e['essence'] for e in result['entanglement'][:3]])}")
                    print()
                    continue

                if query == '/good' and last_response:
                    self.memory.record_feedback(self.queries[-1] if self.queries else "", last_response, 1)
                    self.tensor.adapt_trust(0.8)
                    # Reinforce this Q&A pair with high score
                    q = self.queries[-1] if self.queries else ""
                    if q:
                        self._feed_learner(q, last_response, 'feedback_good', 0.95)
                    print("Thanks! I'll remember that.\n")
                    continue

                if query == '/bad' and last_response:
                    self.memory.record_feedback(self.queries[-1] if self.queries else "", last_response, -1)
                    self.tensor.adapt_trust(0.2)
                    # Penalize this Q&A pair with low score
                    q = self.queries[-1] if self.queries else ""
                    if q:
                        self._feed_learner(q, last_response, 'feedback_bad', 0.1)
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

    async def chat_interactive(self) -> None:
        """
        Interactive chat mode with conversation memory.

        Unlike the basic interactive() mode, this:
        - Maintains conversation history for follow-up questions
        - Supports /kb for inline knowledge base search
        - Has cleaner UX focused on chatting
        """
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  BAZINGA CHAT - Interactive AI with Memory                   ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        print("  Commands: /kb <query>  /stats  /clear  /quit")
        print("  Just type your question to chat!")
        print()

        conversation_history = []
        kb = None
        attached_files = []

        while True:
            try:
                # Show attachment indicator in prompt
                if attached_files:
                    names = ", ".join([os.path.basename(f) for f in attached_files])
                    prompt = f"You [📎 {names}]: "
                else:
                    prompt = "You: "
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                # Commands
                if user_input.lower() in ['/quit', '/exit', '/q', 'quit', 'exit']:
                    self.memory.end_session()
                    print("\n👋 BAZINGA signing off. Take care!\n")
                    break

                if user_input == '/clear':
                    conversation_history = []
                    attached_files = []
                    print("✓ Conversation cleared.\n")
                    continue

                if user_input == '/stats':
                    self._show_stats()
                    continue

                if user_input.startswith('/attach '):
                    filepath = user_input[8:].strip()
                    path = os.path.expanduser(filepath)
                    if os.path.exists(path):
                        size = os.path.getsize(path)
                        if size > 100_000:
                            print(f"  ⚠️  File too large ({size:,} bytes). Max 100KB.\n")
                        else:
                            attached_files.append(path)
                            print(f"  📎 Attached: {os.path.basename(path)} ({size:,} bytes)")
                            print(f"  Type your question about this file, or /detach to remove.\n")
                    else:
                        print(f"  ❌ File not found: {filepath}\n")
                    continue

                if user_input == '/detach':
                    attached_files = []
                    print("  ✓ Attachments cleared.\n")
                    continue

                if user_input == '/help':
                    print("\n  Commands:")
                    print("    /attach <path>  - Attach a file to your next message")
                    print("    /detach         - Remove attached files")
                    print("    /kb <query>     - Search knowledge base")
                    print("    /stats          - Show session stats")
                    print("    /clear          - Clear conversation")
                    print("    /quit           - Exit chat\n")
                    continue

                if user_input.startswith('/kb '):
                    query = user_input[4:].strip()
                    if not kb:
                        kb = _get_kb()()
                    results = kb.search(query)
                    if results:
                        print(f"\n📚 Found {len(results)} results for '{query}':")
                        for i, r in enumerate(results[:5], 1):
                            title = r.get('title', r.get('file', 'Unknown'))[:50]
                            print(f"   {i}. {title}")
                        print()
                    else:
                        print(f"  No results for '{query}'\n")
                    continue

                # Read attached files and prepend to query
                if attached_files:
                    file_context_parts = []
                    for fpath in attached_files:
                        try:
                            with open(fpath, 'r', errors='replace') as f:
                                content = f.read()
                            file_context_parts.append(f"--- {os.path.basename(fpath)} ---\n{content}")
                        except Exception as e:
                            file_context_parts.append(f"--- {os.path.basename(fpath)} (error: {e}) ---")
                    file_context = "\n\n".join(file_context_parts)
                    user_input = f"{user_input}\n\n[Attached files]\n{file_context}"
                    attached_files = []  # Clear after use

                # Build context from conversation history
                # Keep last 4 exchanges, truncate long responses to prevent context bloat
                context = ""
                if conversation_history:
                    context = "Previous conversation:\n"
                    for turn in conversation_history[-4:]:
                        assistant_text = turn['assistant']
                        if len(assistant_text) > 300:
                            assistant_text = assistant_text[:300] + "..."
                        context += f"User: {turn['user']}\nAssistant: {assistant_text}\n"
                    context += "\nCurrent question: "

                full_query = context + user_input if context else user_input

                # Get response
                response = await self.ask(full_query, fresh=True)

                # Store in history
                conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })

                print(f"\n🤖 {response}\n")

                # RAC heartbeat - show mini status after each response
                if hasattr(self.memory, 'get_trajectory_summary'):
                    rac_summary = self.memory.get_trajectory_summary()
                    if rac_summary:
                        dg = rac_summary['current_delta_gamma']
                        status = rac_summary['status']
                        if status == 'locked':
                            indicator = "🟢"
                        elif status == 'converging':
                            indicator = "🟡"
                        else:
                            indicator = "🔴"
                        print(f"  {indicator} ΔΓ={dg:.3f} | {status.upper()}\n")

                        # Auto-push RAC heartbeat to Cloudflare (non-blocking)
                        self._push_rac_to_cloudflare(rac_summary)

            except KeyboardInterrupt:
                self.memory.end_session()
                print("\n\n👋 BAZINGA signing off.\n")
                break
            except EOFError:
                break

    def _show_stats(self) -> None:
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

        # RAC (Resonance-Augmented Continuity) display
        if hasattr(self.memory, 'format_rac_display'):
            print(self.memory.format_rac_display())
        else:
            print()

    def _push_rac_to_cloudflare(self, rac_summary: Dict[str, Any]) -> None:
        """Push RAC heartbeat to Cloudflare KB Bridge (non-blocking)."""
        import threading
        import urllib.request
        import json as json_mod

        def push():
            try:
                url = "https://kb.bitsabhi.com/rac?key=phi137"
                data = json_mod.dumps(rac_summary).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass  # Silent fail - don't interrupt chat

        # Run in background thread
        threading.Thread(target=push, daemon=True).start()


# Help functions moved to bazinga/cli/help.py
# Using aliases for backwards compatibility
_print_ai_help = print_ai_help


_print_kb_help = print_kb_help


_print_chain_help = print_chain_help


_print_p2p_help = print_p2p_help


_print_full_help = print_full_help


async def main() -> None:
    """Main entry point for BAZINGA CLI."""
    parser = argparse.ArgumentParser(
        description="BAZINGA - The first AI you actually own. Free, private, works offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BAZINGA v5 — Distributed AI with Triadic Consensus

QUICK START:
  bazinga "What is AI?"               Ask anything (works immediately)
  bazinga --chat                      Interactive chat with memory
  bazinga --omega                     Start full distributed brain

THREE PILLARS:
  1. AI        Ask, chat, multi-AI consensus, code generation
  2. Network   P2P mesh, federated learning, peer discovery
  3. Research  Darmiyan blockchain, TrD consciousness, PoB mining

EXAMPLES:
  bazinga "explain consciousness"                     Ask a question
  bazinga --multi-ai "is free will real?"              6 AIs discuss + consensus
  bazinga --omega                                     Full brain (learning + mesh + TrD)
  bazinga --trd 10                                    Consciousness test (10 agents)
  bazinga --mine                                      Mine a block (zero energy)
  bazinga --index ~/Documents                         Index files for RAG (depth)
  bazinga --scan ~/Documents ~/Projects               Scan for KB DNA manifests (breadth)

MORE INFO:
  bazinga --help-ai         AI commands
  bazinga --help-chain      Blockchain + Research commands
  bazinga --help-p2p        Network + P2P commands
  bazinga --help-kb         Knowledge Base commands
  bazinga --help-all        Full documentation

https://github.com/0x-auth/bazinga-indeed | pip install bazinga-indeed
"""
    )

    # Positional argument for direct questions (no flag needed)
    parser.add_argument('question', nargs='?', type=str, default=None,
                        help='Ask a question directly (no --ask needed)')

    # === AI COMMANDS ===
    ai_group = parser.add_argument_group('AI (Pillar 1)')
    ai_group.add_argument('--ask', '-a', type=str, metavar='Q',
                          help='Ask a question')
    ai_group.add_argument('--chat', action='store_true',
                          help='Interactive chat mode with memory')
    ai_group.add_argument('--multi-ai', '-m', type=str, metavar='Q',
                          help='Ask multiple AIs for φ-coherence consensus')
    ai_group.add_argument('--code', '-c', type=str, metavar='TASK',
                          help='Generate code')
    ai_group.add_argument('--lang', '-l', type=str, default='python',
                          choices=['python', 'javascript', 'js', 'typescript', 'ts', 'rust', 'go'],
                          help='Code language (default: python)')
    ai_group.add_argument('--quantum', '-q', type=str, metavar='TEXT',
                          help='Quantum pattern analysis')
    ai_group.add_argument('--coherence', type=str, metavar='TEXT',
                          help='Check φ-coherence')
    ai_group.add_argument('--agent', type=str, nargs='?', const='', metavar='TASK',
                          help='AI agent mode')
    ai_group.add_argument('--fresh', '-f', action='store_true',
                          help='Bypass cache')
    ai_group.add_argument('--local', action='store_true',
                          help='Force local LLM')
    ai_group.add_argument('--file', type=str, metavar='PATH',
                          help='File path for context (works with --ask, --multi-ai, --code)')

    # === KNOWLEDGE BASE ===
    kb_group = parser.add_argument_group('Knowledge Base (KB)')
    kb_group.add_argument('--kb', type=str, nargs='?', const='', metavar='QUERY',
                          help='Search KB (Gmail, GDrive, Mac, Phone)')
    kb_group.add_argument('--kb-sources', action='store_true',
                          help='Show data sources status')
    kb_group.add_argument('--kb-sync', action='store_true',
                          help='Re-index all sources')
    kb_group.add_argument('--kb-gmail', type=str, nargs='?', const='', metavar='QUERY',
                          help='Search Gmail only (optionally with query: --kb-gmail "search term")')
    kb_group.add_argument('--kb-gdrive', action='store_true',
                          help='Search GDrive only')
    kb_group.add_argument('--kb-mac', action='store_true',
                          help='Search Mac only')
    kb_group.add_argument('--kb-phone', type=str, nargs='?', const='', metavar='PATH',
                          help='Search Phone only (or set path: --kb-phone ~/path)')
    kb_group.add_argument('--kb-phone-path', type=str, metavar='PATH',
                          help='Set phone data path and index it')
    kb_group.add_argument('--summarize', action='store_true',
                          help='Summarize KB results into an answer (uses LLM)')

    # === INDEXING ===
    index_group = parser.add_argument_group('Indexing')
    index_group.add_argument('--index', '-i', nargs='+', metavar='PATH',
                             help='Index directories for RAG')
    index_group.add_argument('--deindex', type=str, nargs='+', metavar='PATH',
                             help='Remove indexed files from a directory (undo --index)')
    index_group.add_argument('--index-public', type=str, metavar='SOURCE',
                             choices=['wikipedia', 'arxiv', 'gutenberg'],
                             help='Index public knowledge')
    index_group.add_argument('--topics', type=str, metavar='TOPICS',
                             help='Topics for --index-public')
    index_group.add_argument('--scan', nargs='+', metavar='PATH',
                             help='Scan directories for KB DNA manifests (breadth context)')
    index_group.add_argument('--scan-status', action='store_true',
                             help='Show KB manifest status')
    index_group.add_argument('--scan-depth', type=int, default=5, metavar='N',
                             help='Max directory depth for --scan (default: 5)')

    # === BLOCKCHAIN & RESEARCH ===
    chain_group = parser.add_argument_group('Blockchain & Research (Pillar 3)')
    chain_group.add_argument('--chain', action='store_true',
                             help='Show blockchain status')
    chain_group.add_argument('--mine', action='store_true',
                             help='Mine block (Proof-of-Boundary, zero energy)')
    chain_group.add_argument('--wallet', action='store_true',
                             help='Show wallet/identity')
    chain_group.add_argument('--attest', type=str, metavar='CONTENT',
                             help='Attest knowledge to chain (prove you knew it first)')
    chain_group.add_argument('--email', type=str, metavar='EMAIL',
                             help='Email for attestation receipt (use with --attest)')
    chain_group.add_argument('--verify', type=str, metavar='ID',
                             help='Verify attestation')
    chain_group.add_argument('--trust', type=str, nargs='?', const='', metavar='NODE',
                             help='Show trust scores')
    chain_group.add_argument('--trd', type=int, nargs='?', const=5, metavar='N',
                             help='TrD consciousness test (n agents, default: 5)')
    chain_group.add_argument('--trd-heartbeat', action='store_true',
                             help='Run TrD heartbeat demo (persistent self-reference)')
    chain_group.add_argument('--trd-scan', nargs=2, type=int, metavar=('START', 'END'),
                             help='Phase transition scan (e.g. --trd-scan 15 22)')
    chain_group.add_argument('--consciousness', type=int, nargs='?', const=2, metavar='N',
                             help='Consciousness scaling test (Darmiyan: psi = phi*sqrt(n))')

    # === EVOLUTION (Self-Improvement) ===
    evo_group = parser.add_argument_group('Evolution (Autonomous Self-Improvement)')
    evo_group.add_argument('--propose', type=str, metavar='TITLE',
                           help='Propose a code improvement (requires --diff)')
    evo_group.add_argument('--diff', type=str, metavar='FILE',
                           help='Patch file for --propose')
    evo_group.add_argument('--proposals', nargs='?', const='all', metavar='STATUS',
                           help='List proposals (all, active, approved, rejected)')
    evo_group.add_argument('--vote', type=str, metavar='PROPOSAL_ID',
                           help='Vote on a proposal')
    evo_group.add_argument('--approve', action='store_true',
                           help='Approve (use with --vote)')
    evo_group.add_argument('--reject', action='store_true',
                           help='Reject (use with --vote)')
    evo_group.add_argument('--reason', type=str, default='',
                           help='Reasoning for vote (use with --vote)')
    evo_group.add_argument('--evolution-status', action='store_true',
                           help='Show evolution engine status (autonomy level, proposals, constitution)')
    evo_group.add_argument('--constitution', action='store_true',
                           help='Show constitutional bounds')

    # === P2P NETWORK ===
    p2p_group = parser.add_argument_group('Network & P2P (Pillar 2)')
    p2p_group.add_argument('--omega', action='store_true',
                           help='Start full distributed brain (Learning + Mesh + TrD + P2P)')
    p2p_group.add_argument('--headless', action='store_true',
                           help='Run --omega without TUI (plain terminal mode)')
    p2p_group.add_argument('--join', type=str, nargs='*', metavar='HOST:PORT',
                           help='Join network')
    p2p_group.add_argument('--peers', action='store_true',
                           help='Show peers')
    p2p_group.add_argument('--mesh', action='store_true',
                           help='Show mesh network vital signs (peers, trust, health)')
    p2p_group.add_argument('--sync', action='store_true',
                           help='Sync knowledge with network')
    p2p_group.add_argument('--publish', action='store_true',
                           help='Publish to DHT')
    p2p_group.add_argument('--query-network', type=str, metavar='Q',
                           help='Query network')
    p2p_group.add_argument('--phi-pulse', action='store_true',
                           help='Start Phi-Pulse discovery (LAN UDP on port 5150)')
    p2p_group.add_argument('--port', type=int, default=5151,
                           help='P2P listening port (default: 5151)')
    p2p_group.add_argument('--node-id', type=str, default=None,
                           help='Custom node ID (default: auto-generated)')

    # === INFO ===
    info_group = parser.add_argument_group('Info')
    info_group.add_argument('--version', '-v', action='store_true',
                            help='Show version')
    info_group.add_argument('--check', action='store_true',
                            help='System diagnostic')
    info_group.add_argument('--constants', action='store_true',
                            help='Show constants (φ, α, ψ)')
    info_group.add_argument('--stats', action='store_true',
                            help='Show statistics')
    info_group.add_argument('--rac', action='store_true',
                            help='Show RAC (Resonance-Augmented Continuity) status')
    info_group.add_argument('--carm', action='store_true',
                            help='Show CARM (Context-Addressed Resonant Memory) status')
    info_group.add_argument('--models', action='store_true',
                            help='List local models')
    info_group.add_argument('--local-status', action='store_true',
                            help='Local model status')
    info_group.add_argument('--bootstrap-local', action='store_true',
                            help='Setup Ollama + llama3')

    # === EXTENDED HELP ===
    help_group = parser.add_argument_group('Extended Help')
    help_group.add_argument('--help-ai', action='store_true',
                            help='AI commands documentation')
    help_group.add_argument('--help-kb', action='store_true',
                            help='Knowledge Base documentation')
    help_group.add_argument('--help-chain', action='store_true',
                            help='Blockchain documentation')
    help_group.add_argument('--help-p2p', action='store_true',
                            help='P2P network documentation')
    help_group.add_argument('--help-all', action='store_true',
                            help='Full documentation')

    # === HIDDEN/ADVANCED ===
    parser.add_argument('--node', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--proof', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--consensus', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--network', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--nat', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--learn', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--share', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--attest-pricing', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--simple', '-s', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--verbose', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--vac', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--demo', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--generate', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--no-p2p', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    # =====================================================
    # AUTO-START PHI-PULSE (Background P2P Discovery)
    # =====================================================
    # Start Phi-Pulse automatically for interactive modes
    # unless --no-p2p is specified
    _phi_pulse_instance = None
    _query_server_instance = None
    _mesh_query_instance = None
    _mesh_node_id = None

    def _start_background_p2p():
        """Start Phi-Pulse + QueryServer in background for peer discovery and mesh queries."""
        nonlocal _phi_pulse_instance, _query_server_instance, _mesh_query_instance, _mesh_node_id
        if getattr(args, 'no_p2p', False):
            return

        # Only start for interactive modes or explicit P2P commands
        is_interactive = (
            getattr(args, 'chat', False) or
            getattr(args, 'join', None) is not None or
            (args.question is None and not any([
                getattr(args, 'ask', None),
                getattr(args, 'version', False),
                getattr(args, 'check', False),
                getattr(args, 'help_all', False),
            ]))
        )

        if not is_interactive:
            return

        try:
            from ..decentralized.peer_discovery import PhiPulse
            from ..p2p.persistence import get_persistence_manager
            from ..p2p.mesh_query import QueryServer, MeshQuery
            import hashlib
            import os

            # Generate or load node ID
            pm = get_persistence_manager()
            node_id = pm.get_state('node_id')
            if not node_id:
                node_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
                pm.set_state('node_id', node_id)

            _mesh_node_id = node_id
            p2p_port = getattr(args, 'port', 5151)

            def on_peer_found(peer_id, ip, port):
                # Silently save to persistence
                pass  # Already saved by PhiPulse internally

            _phi_pulse_instance = PhiPulse(
                node_id=node_id,
                listen_port=p2p_port,
                on_peer_found=on_peer_found,
            )
            _phi_pulse_instance.start()

            # Start QueryServer so peers can ask us questions
            _query_server_instance = QueryServer(port=p2p_port, node_id=node_id)

            # Create MeshQuery for fan-out queries
            _mesh_query_instance = MeshQuery(node_id=node_id)

            # Load and check known peers from last session
            known_peers = pm.get_known_peers(limit=10, max_age_hours=24)
            if known_peers:
                print(f"  📡 Loaded {len(known_peers)} known peers from last session")

        except Exception as e:
            # Silently fail - P2P is optional
            pass

    async def _start_query_server(bazinga_instance):
        """Start QueryServer with BAZINGA as the handler (must be called from async context)."""
        nonlocal _query_server_instance
        if _query_server_instance and bazinga_instance:
            async def handle_query(question: str) -> dict:
                try:
                    result = await bazinga_instance.ask(question, fresh=True)
                    if isinstance(result, dict):
                        return {
                            "answer": result.get("response", str(result)),
                            "confidence": result.get("coherence", 0.5),
                            "source": result.get("source", "unknown"),
                        }
                    return {"answer": str(result), "confidence": 0.5, "source": "llm"}
                except Exception as e:
                    return {"answer": f"Error: {e}", "confidence": 0.0, "source": "error"}

            _query_server_instance.set_handler(handle_query)
            await _query_server_instance.start()

    def _stop_background_p2p():
        """Stop Phi-Pulse and QueryServer when CLI exits."""
        nonlocal _phi_pulse_instance, _query_server_instance, _mesh_query_instance
        if _phi_pulse_instance:
            _phi_pulse_instance.stop()
            _phi_pulse_instance = None
        if _query_server_instance:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_query_server_instance.stop())
                else:
                    loop.run_until_complete(_query_server_instance.stop())
            except Exception:
                pass
            _query_server_instance = None
        _mesh_query_instance = None

    # =====================================================
    # DISPATCH — Route to command handler modules
    # =====================================================
    # Each handler is an async function in bazinga/cli/commands/
    # This keeps _core.py focused on BAZINGA class + argparse.

    # Helper closures for commands that need P2P state
    def _get_mesh_node_id():
        return _mesh_node_id

    def _get_mesh_query():
        return _mesh_query_instance

    # Handle extended help options
    if hasattr(args, 'help_all') and args.help_all:
        _print_full_help()
        return
    if hasattr(args, 'help_ai') and args.help_ai:
        _print_ai_help()
        return
    if hasattr(args, 'help_kb') and args.help_kb:
        _print_kb_help()
        return
    if hasattr(args, 'help_chain') and args.help_chain:
        _print_chain_help()
        return
    if hasattr(args, 'help_p2p') and args.help_p2p:
        _print_p2p_help()
        return

    # ── Info commands ──────────────────────────────────────
    if args.version:
        from .commands.info import handle_version
        await handle_version(args)
        return

    # Handle --check (system diagnostic)
    if args.check:
        from .commands.info import handle_check
        await handle_check(args)
        return

    # Handle --constants
    if args.constants:
        from .commands.info import handle_constants
        await handle_constants(args)
        return

    # Handle --stats
    if args.stats:
        from .commands.info import handle_stats
        await handle_stats(args)
        return

    # Handle --models
    if args.models:
        from .commands.info import handle_models
        await handle_models(args)
        return

    # Handle --rac
    if args.rac:
        from .commands.info import handle_rac
        await handle_rac(args)
        return

    # Handle --carm
    if args.carm:
        from .commands.info import handle_carm
        await handle_carm(args)
        return

    # Handle --bootstrap-local
    if args.bootstrap_local:
        from .commands.info import handle_bootstrap_local
        await handle_bootstrap_local(args)
        return

    # Handle --local-status
    if args.local_status:
        from .commands.info import handle_local_status
        await handle_local_status(args)
        return

    # Handle --consciousness
    if args.consciousness is not None:
        from .commands.info import handle_consciousness
        await handle_consciousness(args)
        return

    # ── Agent ──────────────────────────────────────────────
    if args.agent is not None:
        from .commands.network import handle_agent
        await handle_agent(args)
        return

    # ── Knowledge Base ─────────────────────────────────────
    if hasattr(args, 'kb_phone_path') and args.kb_phone_path:
        from .commands.kb import handle_kb_phone_path
        await handle_kb_phone_path(args)
        return

    kb_phone_is_path = hasattr(args, 'kb_phone') and args.kb_phone and args.kb_phone != '' and '/' in args.kb_phone
    if kb_phone_is_path:
        from .commands.kb import handle_kb_phone_as_path
        await handle_kb_phone_as_path(args)
        return

    if args.kb is not None or args.kb_sources or args.kb_sync:
        from .commands.kb import handle_kb
        await handle_kb(args, BAZINGA)
        return

    # ── Indexing ───────────────────────────────────────────
    if args.index:
        from .commands.kb import handle_index
        await handle_index(args, BAZINGA)
        return

    if getattr(args, 'deindex', None):
        from .commands.kb import handle_deindex
        await handle_deindex(args, BAZINGA)
        return

    if args.index_public:
        from .commands.kb import handle_index_public
        await handle_index_public(args)
        return

    if args.scan:
        from .commands.kb import handle_scan
        await handle_scan(args)
        return

    if getattr(args, 'scan_status', False):
        from .commands.kb import handle_scan_status
        await handle_scan_status(args)
        return

    # ── Blockchain & Research ──────────────────────────────
    if args.trd is not None:
        from .commands.chain import handle_trd
        await handle_trd(args)
        return

    if args.trd_scan is not None:
        from .commands.chain import handle_trd_scan
        await handle_trd_scan(args)
        return

    if args.trd_heartbeat:
        from .commands.chain import handle_trd_heartbeat
        await handle_trd_heartbeat(args)
        return

    if args.node:
        from .commands.chain import handle_node
        await handle_node(args)
        return

    if args.proof:
        from .commands.chain import handle_proof
        await handle_proof(args)
        return

    if args.consensus:
        from .commands.chain import handle_consensus
        await handle_consensus(args)
        return

    if args.chain:
        from .commands.chain import handle_chain
        await handle_chain(args)
        return

    if args.mine:
        from .commands.chain import handle_mine
        await handle_mine(args)
        return

    if args.wallet:
        from .commands.chain import handle_wallet
        await handle_wallet(args)
        return

    if args.attest_pricing:
        from .commands.chain import handle_attest_pricing
        await handle_attest_pricing(args)
        return

    if args.attest:
        from .commands.chain import handle_attest
        await handle_attest(args)
        return

    if args.verify:
        from .commands.chain import handle_verify
        await handle_verify(args)
        return

    if args.trust is not None:
        from .commands.chain import handle_trust
        await handle_trust(args)
        return

    # ── Evolution ──────────────────────────────────────────
    if args.constitution:
        from .commands.evolution import handle_constitution
        await handle_constitution(args)
        return

    if args.evolution_status:
        from .commands.evolution import handle_evolution_status
        await handle_evolution_status(args)
        return

    if args.proposals is not None:
        from .commands.evolution import handle_proposals
        await handle_proposals(args)
        return

    if args.propose:
        from .commands.evolution import handle_propose
        await handle_propose(args)
        return

    if args.vote:
        from .commands.evolution import handle_vote
        await handle_vote(args)
        return

    # ── P2P Network ────────────────────────────────────────
    if args.network:
        from .commands.network import handle_network
        await handle_network(args)
        return

    if args.join is not None:
        from .commands.network import handle_join
        await handle_join(args)
        return

    if getattr(args, 'phi_pulse', False):
        from .commands.network import handle_phi_pulse
        await handle_phi_pulse(args)
        return

    if getattr(args, 'mesh', False):
        from .commands.network import handle_mesh
        await handle_mesh(args)
        return

    if args.peers:
        from .commands.network import handle_peers
        await handle_peers(args)
        return

    if args.nat:
        from .commands.network import handle_nat
        await handle_nat(args)
        return

    if args.sync:
        from .commands.network import handle_sync
        await handle_sync(args)
        return

    if args.learn:
        from .commands.network import handle_learn
        await handle_learn(args)
        return

    if getattr(args, 'omega', False):
        from .commands.network import handle_omega
        await handle_omega(args, BAZINGA, _start_background_p2p, _start_query_server,
                           _stop_background_p2p, _get_mesh_node_id, _get_mesh_query)
        return

    if args.publish:
        from .commands.network import handle_publish
        await handle_publish(args, BAZINGA)
        return

    if args.query_network:
        from .commands.network import handle_query_network
        await handle_query_network(args, BAZINGA)
        return

    # ── AI Commands ────────────────────────────────────────
    if args.quantum:
        from .commands.ai import handle_quantum
        await handle_quantum(args, BAZINGA)
        return

    if args.coherence:
        from .commands.ai import handle_coherence
        await handle_coherence(args, BAZINGA)
        return

    if args.code:
        from .commands.ai import handle_code
        await handle_code(args)
        return

    if args.generate:
        from .commands.ai import handle_generate
        await handle_generate(args)
        return

    if args.vac:
        from .commands.ai import handle_vac
        await handle_vac(args, BAZINGA)
        return

    if args.multi_ai:
        from .commands.ai import handle_multi_ai
        await handle_multi_ai(args)
        return

    if args.chat:
        from .commands.ai import handle_chat
        await handle_chat(args, BAZINGA, _start_background_p2p, _start_query_server,
                          _stop_background_p2p, _get_mesh_query)
        return

    # Handle ask (--ask or positional question)
    question = args.ask or args.question
    if question:
        from .commands.ai import handle_ask
        await handle_ask(args, BAZINGA)
        return

    # Handle demo
    if args.demo:
        from .commands.ai import handle_demo
        await handle_demo(args, BAZINGA)
        return

    # Default: Interactive mode
    from .commands.ai import handle_interactive
    await handle_interactive(args, BAZINGA)


def main_sync() -> None:
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
