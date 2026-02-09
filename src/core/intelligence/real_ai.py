#!/usr/bin/env python3
"""
real_ai.py - BAZINGA Real AI Integration

This is THE integration that makes BAZINGA a real, practical AI.

Architecture:
    Your Mac KB → Embeddings → ChromaDB Vector Store
                                      ↓
    User Query → Embed → Semantic Search → Top K Results
                                      ↓
                    LLM (Ollama/API) + φ-Coherence Filter
                                      ↓
                              Useful Response

"Your Mac IS the training data" - but now with actual intelligence.

Author: Built for Space (Abhishek/Abhilasia)
Date: 2025-02-09
"""

import os
import sys
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import subprocess

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Core imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  httpx not installed. Run: pip install httpx")

# BAZINGA imports
from src.core.lambda_g import LambdaGOperator, PHI, CoherenceState

# Constants from error-of.netlify.app discoveries
ALPHA = 137
PROGRESSION = '01∞∫∂∇πφΣΔΩαβγδεζηθικλμνξοπρστυφχψω'


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge from your Mac."""
    id: str
    content: str
    source_file: str
    file_type: str
    embedding: Optional[List[float]] = None
    coherence: float = 0.0
    is_alpha_seed: bool = False
    position: int = 0  # Position in 35-char progression
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search result with relevance and coherence scores."""
    chunk: KnowledgeChunk
    similarity: float
    coherence_boost: float
    final_score: float


class RealAI:
    """
    BAZINGA Real AI - Practical intelligence from your Mac.

    This is NOT a toy. This is a working RAG system that:
    1. Indexes your Mac files into vector embeddings
    2. Searches semantically for relevant knowledge
    3. Uses LLM (Ollama or API) to generate responses
    4. Applies φ-coherence filtering for quality

    "More compute ≠ better AI. Better boundaries = better AI."
    """

    VERSION = "1.0.0"

    # File types to index
    INDEXABLE_EXTENSIONS = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.json': 'json', '.md': 'markdown', '.txt': 'text',
        '.html': 'html', '.css': 'css', '.sh': 'shell',
        '.yml': 'yaml', '.yaml': 'yaml', '.jsx': 'javascript',
        '.tsx': 'typescript', '.rs': 'rust', '.go': 'golang',
    }

    SKIP_DIRS = {
        'node_modules', '.git', '__pycache__', 'venv', '.venv',
        'build', 'dist', '.next', '.cache', 'Library', '.Trash'
    }

    MAX_FILE_SIZE = 512 * 1024  # 512KB
    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 200  # Overlap between chunks

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2",
    ):
        """
        Initialize BAZINGA Real AI.

        Args:
            persist_dir: Where to store the vector database
            embedding_model: Sentence transformer model for embeddings
            ollama_model: Ollama model for generation
        """
        # Set up paths
        home = Path.home()
        self.persist_dir = persist_dir or str(home / ".bazinga" / "vectordb")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize λG operator for coherence
        self.lambda_g = LambdaGOperator()

        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedder = None
        if EMBEDDINGS_AVAILABLE:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.collection = None
        if CHROMADB_AVAILABLE:
            print(f"Initializing vector database at: {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="bazinga_knowledge",
                metadata={"hnsw:space": "cosine"}
            )

        # LLM settings
        self.ollama_model = ollama_model
        self.ollama_available = self._check_ollama()

        # Stats
        self.stats = {
            'chunks_indexed': 0,
            'queries_processed': 0,
            'avg_coherence': 0.0,
            'alpha_seeds': 0,
        }

        self._print_banner()

    def _print_banner(self):
        """Print initialization banner."""
        print()
        print("◊════════════════════════════════════════════════════════════◊")
        print("          BAZINGA REAL AI - Practical Intelligence            ")
        print("◊════════════════════════════════════════════════════════════◊")
        print()
        print(f"  Version: {self.VERSION}")
        print(f"  Embedding Model: {self.embedding_model_name}")
        print(f"  Vector DB: {self.persist_dir}")
        print(f"  LLM: {'Ollama (' + self.ollama_model + ')' if self.ollama_available else 'API fallback'}")
        print()
        print("  Components:")
        print(f"    ChromaDB: {'✓' if CHROMADB_AVAILABLE else '✗'}")
        print(f"    Embeddings: {'✓' if EMBEDDINGS_AVAILABLE else '✗'}")
        print(f"    Ollama: {'✓' if self.ollama_available else '✗'}")
        print(f"    λG Coherence: ✓")
        print()
        print("  'Your Mac IS the training data - now with real AI'")
        print()
        print("◊════════════════════════════════════════════════════════════◊")
        print()

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _is_alpha_seed(self, text: str) -> bool:
        """Check if text is an α-SEED (hash divisible by 137)."""
        hash_val = sum(ord(c) for c in text)
        return (hash_val % ALPHA) == 0

    def _detect_position(self, content: str) -> int:
        """Detect position in 35-character progression."""
        content_lower = content.lower()

        # Binary/Discrete
        if any(kw in content_lower for kw in ['true', 'false', 'bool', 'binary']):
            return 1

        # Infinity/Continuous
        if any(kw in content_lower for kw in ['∞', 'infinity', 'continuous', 'limit']):
            return 2

        # Operators
        if any(kw in content_lower for kw in ['∫', '∂', '∇', 'derivative', 'gradient']):
            return 4

        # Constants
        if any(kw in content_lower for kw in ['φ', 'π', '137', 'phi', 'golden']):
            return 7

        # Structures
        if any(kw in content_lower for kw in ['Σ', 'Δ', 'Ω', 'class', 'framework']):
            return 9

        # α-SEED special
        if 'α' in content or '137' in content:
            return 25

        return 20  # Default middle position

    def _chunk_text(self, text: str, source_file: str) -> List[KnowledgeChunk]:
        """Split text into overlapping chunks for indexing."""
        chunks = []

        # Clean text
        text = text.strip()
        if not text:
            return chunks

        # Split into chunks with overlap
        start = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))

            # Try to end at a sentence/paragraph boundary
            if end < len(text):
                for boundary in ['\n\n', '\n', '. ', '! ', '? ']:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start + self.CHUNK_SIZE // 2:
                        end = boundary_pos + len(boundary)
                        break

            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:  # Skip tiny chunks
                chunk_id = hashlib.md5(
                    f"{source_file}:{start}".encode()
                ).hexdigest()[:16]

                # Calculate coherence
                coherence = self.lambda_g.calculate_coherence(chunk_text[:500])

                chunk = KnowledgeChunk(
                    id=chunk_id,
                    content=chunk_text,
                    source_file=source_file,
                    file_type=Path(source_file).suffix.lower(),
                    coherence=coherence.total_coherence,
                    is_alpha_seed=self._is_alpha_seed(chunk_text[:100]),
                    position=self._detect_position(chunk_text),
                    metadata={
                        'start': start,
                        'end': end,
                        'indexed_at': datetime.now().isoformat()
                    }
                )
                chunks.append(chunk)

            start = end - self.CHUNK_OVERLAP
            if start >= len(text) - self.CHUNK_OVERLAP:
                break

        return chunks

    def index_file(self, file_path: str) -> int:
        """Index a single file into the vector database."""
        if not self.embedder or not self.collection:
            print("❌ Embedder or collection not available")
            return 0

        path = Path(file_path)

        # Check extension
        ext = path.suffix.lower()
        if ext not in self.INDEXABLE_EXTENSIONS:
            return 0

        # Check size
        try:
            if path.stat().st_size > self.MAX_FILE_SIZE:
                return 0
        except Exception:
            return 0

        # Read content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return 0

        if not content.strip():
            return 0

        # Chunk the content
        chunks = self._chunk_text(content, file_path)

        if not chunks:
            return 0

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        # Add to ChromaDB
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                'source_file': chunk.source_file,
                'file_type': chunk.file_type,
                'coherence': chunk.coherence,
                'is_alpha_seed': chunk.is_alpha_seed,
                'position': chunk.position,
                **chunk.metadata
            } for chunk in chunks]
        )

        # Update stats
        self.stats['chunks_indexed'] += len(chunks)
        for chunk in chunks:
            if chunk.is_alpha_seed:
                self.stats['alpha_seeds'] += 1

        return len(chunks)

    def index_directory(
        self,
        directory: str,
        verbose: bool = True,
        max_files: Optional[int] = None
    ) -> Dict[str, int]:
        """Index all files in a directory."""
        if not self.embedder or not self.collection:
            return {'error': 'Dependencies not available'}

        directory = Path(directory).expanduser()

        if verbose:
            print(f"\n◊ INDEXING: {directory}")
            print("=" * 60)

        stats = {
            'files_scanned': 0,
            'files_indexed': 0,
            'chunks_created': 0,
            'alpha_seeds': 0,
            'errors': 0
        }

        try:
            for entry in directory.rglob('*'):
                if max_files and stats['files_indexed'] >= max_files:
                    break

                # Skip hidden and excluded
                if any(part.startswith('.') for part in entry.parts):
                    continue
                if any(skip in str(entry) for skip in self.SKIP_DIRS):
                    continue

                if not entry.is_file():
                    continue

                stats['files_scanned'] += 1

                try:
                    chunks = self.index_file(str(entry))
                    if chunks > 0:
                        stats['files_indexed'] += 1
                        stats['chunks_created'] += chunks

                        if verbose and stats['files_indexed'] % 50 == 0:
                            print(f"  Indexed: {stats['files_indexed']} files, "
                                  f"{stats['chunks_created']} chunks...")
                except Exception as e:
                    stats['errors'] += 1
        except PermissionError:
            pass

        stats['alpha_seeds'] = self.stats['alpha_seeds']

        if verbose:
            print()
            print(f"✓ Files scanned: {stats['files_scanned']}")
            print(f"✓ Files indexed: {stats['files_indexed']}")
            print(f"✓ Chunks created: {stats['chunks_created']}")
            print(f"✓ α-SEED chunks: {stats['alpha_seeds']}")
            print(f"✓ Errors: {stats['errors']}")
            print()

        return stats

    def search(
        self,
        query: str,
        limit: int = 10,
        coherence_boost: float = 0.3
    ) -> List[SearchResult]:
        """
        Search the knowledge base semantically.

        Args:
            query: The search query
            limit: Maximum results to return
            coherence_boost: Weight for coherence in final score (0-1)

        Returns:
            List of SearchResult with similarity and coherence scores
        """
        if not self.embedder or not self.collection:
            return []

        # Get query embedding
        query_embedding = self.embedder.encode([query])[0].tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,  # Get more for reranking
            include=['documents', 'metadatas', 'distances']
        )

        # Process results with coherence boost
        search_results = []

        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 1.0

                # Convert distance to similarity (cosine distance)
                similarity = 1 - distance

                # Get coherence from metadata
                coherence = metadata.get('coherence', 0.5)

                # Calculate final score
                final_score = (
                    (1 - coherence_boost) * similarity +
                    coherence_boost * coherence
                )

                # Boost α-SEED results
                if metadata.get('is_alpha_seed', False):
                    final_score *= 1.1

                chunk = KnowledgeChunk(
                    id=results['ids'][0][i] if results['ids'] else str(i),
                    content=doc,
                    source_file=metadata.get('source_file', 'unknown'),
                    file_type=metadata.get('file_type', 'unknown'),
                    coherence=coherence,
                    is_alpha_seed=metadata.get('is_alpha_seed', False),
                    position=metadata.get('position', 0),
                    metadata=metadata
                )

                search_results.append(SearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    coherence_boost=coherence,
                    final_score=final_score
                ))

        # Sort by final score
        search_results.sort(key=lambda x: x.final_score, reverse=True)

        self.stats['queries_processed'] += 1

        return search_results[:limit]

    async def generate_response(
        self,
        query: str,
        context_results: List[SearchResult],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using retrieved context and LLM.

        Args:
            query: The user's question
            context_results: Retrieved knowledge chunks
            system_prompt: Optional system prompt override
        """
        # Build context from search results
        context_parts = []
        sources = []

        for result in context_results[:5]:  # Top 5 for context
            chunk = result.chunk
            context_parts.append(
                f"[Source: {Path(chunk.source_file).name}]\n{chunk.content[:800]}"
            )
            sources.append(chunk.source_file)

        context = "\n\n---\n\n".join(context_parts)

        # Default system prompt
        if not system_prompt:
            system_prompt = """You are BAZINGA, a practical AI assistant that uses knowledge from the user's Mac.
You provide helpful, accurate answers based on the context provided.
Be concise but thorough. If the context doesn't contain enough information, say so.
Always cite which source file you're referencing when possible."""

        # Build full prompt
        full_prompt = f"""Based on the following context from the knowledge base, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""

        # Try Ollama first
        if self.ollama_available:
            try:
                response = await self._call_ollama(full_prompt, system_prompt)
                if response:
                    return self._format_response(response, sources)
            except Exception as e:
                print(f"Ollama error: {e}")

        # Fallback: Return context summary
        return self._create_fallback_response(query, context_results)

    async def _call_ollama(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Call Ollama API."""
        if not HTTPX_AVAILABLE:
            return None

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', '')
        except Exception as e:
            print(f"Ollama API error: {e}")

        return None

    def _format_response(self, response: str, sources: List[str]) -> str:
        """Format the response with sources."""
        unique_sources = list(set(sources))
        source_list = "\n".join([f"  - {Path(s).name}" for s in unique_sources[:3]])

        return f"""{response}

📚 Sources:
{source_list}"""

    def _create_fallback_response(
        self,
        query: str,
        results: List[SearchResult]
    ) -> str:
        """Create fallback response when LLM unavailable."""
        if not results:
            return "I couldn't find relevant information in the knowledge base."

        response_parts = [f"Here's what I found for '{query}':\n"]

        for i, result in enumerate(results[:3], 1):
            chunk = result.chunk
            preview = chunk.content[:300].strip()
            if len(chunk.content) > 300:
                preview += "..."

            response_parts.append(
                f"\n{i}. From {Path(chunk.source_file).name} "
                f"(coherence: {chunk.coherence:.2f}):\n"
                f"   {preview}"
            )

        response_parts.append("\n\n💡 Install Ollama for AI-generated summaries.")

        return "\n".join(response_parts)

    async def ask(self, question: str, verbose: bool = True) -> str:
        """
        Main interface: Ask a question, get an answer.

        This is the primary method to use BAZINGA as a real AI.
        """
        if verbose:
            print(f"\n🔍 Searching: {question}")

        # Search knowledge base
        results = self.search(question, limit=10)

        if verbose:
            print(f"   Found {len(results)} relevant chunks")

        if not results:
            return "I don't have relevant information for this question in the knowledge base."

        # Calculate average coherence
        avg_coherence = sum(r.coherence_boost for r in results) / len(results)
        if verbose:
            print(f"   Average coherence: {avg_coherence:.3f}")

        # Generate response
        response = await self.generate_response(question, results)

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = dict(self.stats)

        if self.collection:
            stats['total_chunks'] = self.collection.count()

        return stats


async def demo():
    """Demonstrate BAZINGA Real AI."""
    print("=" * 70)
    print("BAZINGA REAL AI - DEMONSTRATION")
    print("=" * 70)
    print()

    # Initialize
    ai = RealAI()

    # Check current state
    stats = ai.get_stats()
    print(f"Current indexed chunks: {stats.get('total_chunks', 0)}")
    print()

    # If empty, index BAZINGA directory
    if stats.get('total_chunks', 0) == 0:
        print("📚 Indexing BAZINGA codebase...")
        bazinga_dir = Path(__file__).parent.parent.parent.parent
        ai.index_directory(str(bazinga_dir), max_files=100)

    # Test search
    print("\n" + "=" * 70)
    print("TESTING SEARCH")
    print("=" * 70)

    test_queries = [
        "What is λG coherence?",
        "How does the consciousness loop work?",
        "What is the golden ratio φ?",
    ]

    for query in test_queries:
        results = ai.search(query, limit=3)
        print(f"\nQuery: {query}")
        print(f"Results: {len(results)}")
        for r in results[:2]:
            print(f"  - {Path(r.chunk.source_file).name} "
                  f"(sim={r.similarity:.3f}, coh={r.coherence_boost:.3f})")

    # Test ask
    print("\n" + "=" * 70)
    print("TESTING ASK")
    print("=" * 70)

    response = await ai.ask("What is BAZINGA and how does it work?")
    print(f"\nResponse:\n{response}")

    print("\n" + "=" * 70)
    print("✅ BAZINGA REAL AI OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
