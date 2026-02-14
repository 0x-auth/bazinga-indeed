#!/usr/bin/env python3
"""
BAZINGA Public Knowledge Indexer
=================================

Bootstrap the network with public knowledge sources.

Supported sources:
- Wikipedia (dumps.wikimedia.org)
- arXiv (arxiv.org) [future]
- Project Gutenberg (gutenberg.org) [future]

"Knowledge shared is knowledge multiplied."

How it works:
1. Download Wikipedia dumps (specific categories)
2. Parse into chunks
3. Index locally (ChromaDB)
4. Publish topics to DHT
5. Now ANY peer can query your knowledge!

Author: Space (Abhishek/Abhilasia) & Claude
License: MIT
"""

import asyncio
import gzip
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from xml.etree import ElementTree as ET

# Constants
PHI = 1.618033988749895
CHUNK_SIZE = 1000  # Characters per chunk
MAX_ARTICLES_PER_CATEGORY = 500  # Limit for testing


@dataclass
class WikiArticle:
    """A Wikipedia article."""
    title: str
    content: str
    categories: List[str] = field(default_factory=list)
    url: str = ""

    def to_chunks(self, chunk_size: int = CHUNK_SIZE) -> List[str]:
        """Split article into chunks for indexing."""
        # Clean content
        text = self._clean_wiki_markup(self.content)
        if not text:
            return []

        # Split into chunks
        chunks = []
        words = text.split()
        current_chunk = []
        current_len = 0

        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1

            if current_len >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                # Add title prefix for context
                chunks.append(f"[{self.title}] {chunk_text}")
                current_chunk = []
                current_len = 0

        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(f"[{self.title}] {chunk_text}")

        return chunks

    def _clean_wiki_markup(self, text: str) -> str:
        """Remove Wikipedia markup."""
        if not text:
            return ""

        # Remove references
        text = re.sub(r'\[\[File:.*?\]\]', '', text)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^/]*?/>', '', text)

        # Convert [[link|text]] to text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

        # Remove templates {{ }}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special markers
        text = re.sub(r"'''?", '', text)
        text = re.sub(r'&[a-z]+;', ' ', text)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text


class WikipediaIndexer:
    """
    Index Wikipedia articles into local RAG.

    Uses Wikipedia API for lightweight access.
    For full dumps, we'd use dumps.wikimedia.org.
    """

    def __init__(self, rag_engine=None, verbose: bool = True):
        """
        Args:
            rag_engine: Local RAG engine (RealAI instance)
            verbose: Print progress
        """
        self.rag = rag_engine
        self.verbose = verbose
        self.stats = {
            "articles_fetched": 0,
            "chunks_indexed": 0,
            "categories_processed": 0,
            "errors": 0,
        }

    async def index_category(
        self,
        category: str,
        limit: int = MAX_ARTICLES_PER_CATEGORY,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Index articles from a Wikipedia category.

        Args:
            category: Wikipedia category name (e.g., "Physics", "Consciousness")
            limit: Maximum articles to fetch
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with indexing stats
        """
        if self.verbose:
            print(f"\n📚 Indexing Wikipedia category: {category}")

        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed. Run: pip install httpx"}

        articles = []
        continue_token = None

        # Fetch articles using Wikipedia API
        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(articles) < limit:
                # Build API URL
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{category}",
                    "cmlimit": min(50, limit - len(articles)),
                    "cmtype": "page",
                    "format": "json",
                }
                if continue_token:
                    params["cmcontinue"] = continue_token

                try:
                    response = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params=params
                    )

                    if response.status_code != 200:
                        break

                    data = response.json()
                    members = data.get("query", {}).get("categorymembers", [])

                    for member in members:
                        title = member.get("title", "")
                        if title and not title.startswith("Category:"):
                            articles.append(title)
                            if progress_callback:
                                progress_callback(f"Found: {title}")

                    # Check for more
                    continue_info = data.get("continue", {})
                    continue_token = continue_info.get("cmcontinue")
                    if not continue_token:
                        break

                except Exception as e:
                    if self.verbose:
                        print(f"  Error fetching category: {e}")
                    self.stats["errors"] += 1
                    break

        if self.verbose:
            print(f"  Found {len(articles)} articles")

        # Fetch and index each article
        chunks_indexed = 0
        for i, title in enumerate(articles):
            try:
                article = await self._fetch_article(client, title)
                if article:
                    chunks = article.to_chunks()
                    for chunk in chunks:
                        if self.rag:
                            # Index the chunk
                            self.rag.add_text(chunk, source=f"wikipedia:{category}/{title}")
                        chunks_indexed += 1

                    self.stats["articles_fetched"] += 1

                    if self.verbose and (i + 1) % 10 == 0:
                        print(f"  Indexed {i + 1}/{len(articles)} articles ({chunks_indexed} chunks)")

                    if progress_callback:
                        progress_callback(f"Indexed: {title}")

            except Exception as e:
                self.stats["errors"] += 1
                if self.verbose:
                    print(f"  Error indexing {title}: {e}")

        self.stats["chunks_indexed"] += chunks_indexed
        self.stats["categories_processed"] += 1

        return {
            "category": category,
            "articles": len(articles),
            "chunks": chunks_indexed,
            "errors": self.stats["errors"],
        }

    async def _fetch_article(self, client, title: str) -> Optional[WikiArticle]:
        """Fetch a single Wikipedia article."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories",
            "explaintext": "true",
            "exlimit": 1,
            "format": "json",
        }

        try:
            response = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params=params
            )

            if response.status_code != 200:
                return None

            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                if page_id == "-1":
                    return None

                content = page.get("extract", "")
                categories = [c.get("title", "").replace("Category:", "")
                             for c in page.get("categories", [])]

                return WikiArticle(
                    title=title,
                    content=content,
                    categories=categories,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                )

        except Exception:
            return None

        return None

    async def index_categories(
        self,
        categories: List[str],
        limit_per_category: int = 100
    ) -> Dict[str, Any]:
        """
        Index multiple Wikipedia categories.

        Args:
            categories: List of category names
            limit_per_category: Max articles per category

        Returns:
            Combined stats
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  BAZINGA Wikipedia Indexer")
            print(f"  Categories: {', '.join(categories)}")
            print(f"  Limit: {limit_per_category} articles per category")
            print(f"{'='*60}\n")

        results = []
        for category in categories:
            result = await self.index_category(category, limit_per_category)
            results.append(result)

        total_articles = sum(r.get("articles", 0) for r in results)
        total_chunks = sum(r.get("chunks", 0) for r in results)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  INDEXING COMPLETE")
            print(f"  Categories: {len(categories)}")
            print(f"  Articles: {total_articles}")
            print(f"  Chunks: {total_chunks}")
            print(f"{'='*60}\n")

        return {
            "categories": len(categories),
            "total_articles": total_articles,
            "total_chunks": total_chunks,
            "results": results,
        }


# =============================================================================
# ARXIV INDEXER
# =============================================================================

@dataclass
class ArxivPaper:
    """An arXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    published: str = ""
    url: str = ""

    def to_chunks(self, chunk_size: int = CHUNK_SIZE) -> List[str]:
        """Split paper into chunks for indexing."""
        # Combine title and abstract
        text = f"{self.title}\n\n{self.abstract}"
        if not text.strip():
            return []

        # For papers, we keep the abstract as one chunk with metadata
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al."

        chunk = f"[arXiv:{self.arxiv_id}] {self.title}\nAuthors: {authors_str}\n\n{self.abstract}"
        return [chunk]


class ArxivIndexer:
    """
    Index arXiv papers into local RAG.

    Uses arXiv API for paper search and metadata.
    """

    def __init__(self, rag_engine=None, verbose: bool = True):
        self.rag = rag_engine
        self.verbose = verbose
        self.stats = {
            "papers_fetched": 0,
            "chunks_indexed": 0,
            "categories_processed": 0,
            "errors": 0,
        }

    async def index_category(
        self,
        category: str,
        limit: int = 100,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Index papers from an arXiv category.

        Args:
            category: arXiv category (e.g., "cs.AI", "physics.gen-ph", "math.NT")
            limit: Maximum papers to fetch
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with indexing stats
        """
        if self.verbose:
            print(f"\n📄 Indexing arXiv category: {category}")

        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed. Run: pip install httpx"}

        papers = []

        # arXiv API query
        # http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=100
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                params = {
                    "search_query": f"cat:{category}",
                    "start": 0,
                    "max_results": min(limit, 200),  # arXiv limits to 200 per request
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }

                response = await client.get(
                    "http://export.arxiv.org/api/query",
                    params=params
                )

                if response.status_code != 200:
                    return {"error": f"arXiv API error: {response.status_code}"}

                # Parse Atom XML response
                papers = self._parse_arxiv_response(response.text)

                if self.verbose:
                    print(f"  Found {len(papers)} papers")

            except Exception as e:
                if self.verbose:
                    print(f"  Error fetching from arXiv: {e}")
                self.stats["errors"] += 1
                return {"error": str(e)}

        # Index each paper
        chunks_indexed = 0
        for i, paper in enumerate(papers):
            try:
                chunks = paper.to_chunks()
                for chunk in chunks:
                    if self.rag:
                        self.rag.add_text(chunk, source=f"arxiv:{category}/{paper.arxiv_id}")
                    chunks_indexed += 1

                self.stats["papers_fetched"] += 1

                if self.verbose and (i + 1) % 20 == 0:
                    print(f"  Indexed {i + 1}/{len(papers)} papers")

                if progress_callback:
                    progress_callback(f"Indexed: {paper.title[:50]}...")

            except Exception as e:
                self.stats["errors"] += 1

        self.stats["chunks_indexed"] += chunks_indexed
        self.stats["categories_processed"] += 1

        return {
            "category": category,
            "papers": len(papers),
            "chunks": chunks_indexed,
            "errors": self.stats["errors"],
        }

    def _parse_arxiv_response(self, xml_text: str) -> List[ArxivPaper]:
        """Parse arXiv Atom XML response."""
        papers = []

        try:
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom',
            }

            root = ET.fromstring(xml_text)

            for entry in root.findall('atom:entry', ns):
                try:
                    # Get ID (extract just the arxiv ID)
                    id_elem = entry.find('atom:id', ns)
                    arxiv_id = ""
                    if id_elem is not None and id_elem.text:
                        # ID is like http://arxiv.org/abs/2301.12345v1
                        arxiv_id = id_elem.text.split('/')[-1]

                    # Get title
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                    # Get abstract (summary)
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""

                    # Get authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text)

                    # Get categories
                    categories = []
                    for cat in entry.findall('arxiv:primary_category', ns):
                        term = cat.get('term', '')
                        if term:
                            categories.append(term)
                    for cat in entry.findall('atom:category', ns):
                        term = cat.get('term', '')
                        if term and term not in categories:
                            categories.append(term)

                    # Get published date
                    published_elem = entry.find('atom:published', ns)
                    published = published_elem.text if published_elem is not None else ""

                    if arxiv_id and (title or abstract):
                        papers.append(ArxivPaper(
                            arxiv_id=arxiv_id,
                            title=title,
                            abstract=abstract,
                            authors=authors,
                            categories=categories,
                            published=published,
                            url=f"https://arxiv.org/abs/{arxiv_id}",
                        ))

                except Exception:
                    continue

        except Exception as e:
            if self.verbose:
                print(f"  Error parsing arXiv XML: {e}")

        return papers

    async def index_categories(
        self,
        categories: List[str],
        limit_per_category: int = 100
    ) -> Dict[str, Any]:
        """
        Index multiple arXiv categories.

        Args:
            categories: List of arXiv category codes
            limit_per_category: Max papers per category

        Returns:
            Combined stats
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  BAZINGA arXiv Indexer")
            print(f"  Categories: {', '.join(categories)}")
            print(f"  Limit: {limit_per_category} papers per category")
            print(f"{'='*60}\n")

        results = []
        for category in categories:
            result = await self.index_category(category, limit_per_category)
            results.append(result)
            # Small delay to be nice to arXiv API
            await asyncio.sleep(1)

        total_papers = sum(r.get("papers", 0) for r in results)
        total_chunks = sum(r.get("chunks", 0) for r in results)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  INDEXING COMPLETE")
            print(f"  Categories: {len(categories)}")
            print(f"  Papers: {total_papers}")
            print(f"  Chunks: {total_chunks}")
            print(f"{'='*60}\n")

        return {
            "categories": len(categories),
            "total_papers": total_papers,
            "total_chunks": total_chunks,
            "results": results,
        }


# =============================================================================
# CLI INTEGRATION
# =============================================================================

async def index_public_knowledge(
    source: str,
    topics: List[str],
    limit: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Index public knowledge sources.

    Args:
        source: "wikipedia", "arxiv", or "gutenberg"
        topics: List of topics/categories to index
        limit: Max items per topic
        verbose: Print progress

    Returns:
        Indexing stats
    """
    # Get or create RAG engine
    rag = None
    try:
        from src.core.intelligence.real_ai import RealAI
        rag = RealAI()
    except ImportError:
        try:
            # Try alternate import path
            from .cli import _get_real_ai
            RealAI = _get_real_ai()
            rag = RealAI()
        except Exception as e:
            return {"error": f"Could not initialize RAG: {e}. Make sure chromadb is installed."}

    if source.lower() == "wikipedia":
        indexer = WikipediaIndexer(rag_engine=rag, verbose=verbose)
        result = await indexer.index_categories(topics, limit)

        # Publish to DHT if available
        try:
            from .p2p.knowledge_sharing import KnowledgePublisher
            from .p2p import get_dht_node

            dht = get_dht_node()
            if dht:
                publisher = KnowledgePublisher(dht, rag)
                publish_result = await publisher.publish_from_index(limit=200)
                result["dht_published"] = publish_result
                if verbose:
                    print(f"\n📡 Published {publish_result.get('topics_published', 0)} topics to DHT")
        except Exception:
            pass

        return result

    elif source.lower() == "arxiv":
        indexer = ArxivIndexer(rag_engine=rag, verbose=verbose)
        result = await indexer.index_categories(topics, limit)

        # Publish to DHT if available
        try:
            from .p2p.knowledge_sharing import KnowledgePublisher
            from .p2p import get_dht_node

            dht = get_dht_node()
            if dht:
                publisher = KnowledgePublisher(dht, rag)
                publish_result = await publisher.publish_from_index(limit=200)
                result["dht_published"] = publish_result
                if verbose:
                    print(f"\n📡 Published {publish_result.get('topics_published', 0)} topics to DHT")
        except Exception:
            pass

        return result

    elif source.lower() == "gutenberg":
        return {"error": "Project Gutenberg indexing not yet implemented. Coming in v4.9.0!"}

    else:
        return {"error": f"Unknown source: {source}. Supported: wikipedia, arxiv, gutenberg"}


def index_public_knowledge_sync(
    source: str,
    topics: List[str],
    limit: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """Synchronous version of index_public_knowledge."""
    return asyncio.run(index_public_knowledge(source, topics, limit, verbose))


# =============================================================================
# PRESET TOPIC COLLECTIONS
# =============================================================================

# Wikipedia presets
TOPIC_PRESETS = {
    "science": [
        "Physics",
        "Mathematics",
        "Chemistry",
        "Biology",
        "Astronomy",
        "Computer_science",
    ],
    "philosophy": [
        "Philosophy_of_mind",
        "Epistemology",
        "Metaphysics",
        "Ethics",
        "Logic",
        "Consciousness",
    ],
    "ai": [
        "Artificial_intelligence",
        "Machine_learning",
        "Neural_networks",
        "Natural_language_processing",
        "Computer_vision",
        "Robotics",
    ],
    "bazinga": [
        "Consciousness",
        "Golden_ratio",
        "Distributed_computing",
        "Peer-to-peer",
        "Blockchain",
        "Cryptography",
    ],
}

# arXiv category presets (use with --index-public arxiv)
ARXIV_PRESETS = {
    "cs": [  # Computer Science
        "cs.AI",      # Artificial Intelligence
        "cs.LG",      # Machine Learning
        "cs.CL",      # Computation and Language (NLP)
        "cs.CV",      # Computer Vision
        "cs.DC",      # Distributed Computing
        "cs.CR",      # Cryptography
    ],
    "physics": [
        "physics.gen-ph",   # General Physics
        "quant-ph",         # Quantum Physics
        "cond-mat",         # Condensed Matter
        "hep-th",           # High Energy Physics - Theory
    ],
    "math": [
        "math.NT",    # Number Theory
        "math.CO",    # Combinatorics
        "math.LO",    # Logic
        "math.PR",    # Probability
    ],
    "ai": [  # AI focused
        "cs.AI",
        "cs.LG",
        "cs.NE",      # Neural and Evolutionary Computing
        "stat.ML",    # Machine Learning (Statistics)
    ],
    "bazinga": [  # BAZINGA-relevant papers
        "cs.DC",      # Distributed Computing
        "cs.CR",      # Cryptography
        "cs.AI",      # AI
        "quant-ph",   # Quantum
        "cs.MA",      # Multi-Agent Systems
    ],
}


def get_preset_topics(preset: str, source: str = "wikipedia") -> List[str]:
    """Get predefined topic list by name."""
    if source.lower() == "arxiv":
        return ARXIV_PRESETS.get(preset.lower(), [])
    return TOPIC_PRESETS.get(preset.lower(), [])


# =============================================================================
# CLI ENTRY
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m bazinga.public_knowledge <source> <topics>")
        print("  source: wikipedia, arxiv, gutenberg")
        print("  topics: comma-separated list OR preset name (science, philosophy, ai, bazinga)")
        print()
        print("Examples:")
        print("  python -m bazinga.public_knowledge wikipedia Physics,Mathematics")
        print("  python -m bazinga.public_knowledge wikipedia science")
        print("  python -m bazinga.public_knowledge wikipedia bazinga")
        sys.exit(1)

    source = sys.argv[1]
    topics_arg = sys.argv[2]

    # Check if it's a preset
    if topics_arg.lower() in TOPIC_PRESETS:
        topics = TOPIC_PRESETS[topics_arg.lower()]
    else:
        topics = [t.strip() for t in topics_arg.split(",")]

    result = index_public_knowledge_sync(source, topics)
    print(json.dumps(result, indent=2))
