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
    try:
        from .ai import RealAI
        rag = RealAI()
    except Exception as e:
        return {"error": f"Could not initialize RAG: {e}"}

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
        return {"error": "arXiv indexing not yet implemented. Coming in v4.9.0!"}

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


def get_preset_topics(preset: str) -> List[str]:
    """Get predefined topic list by name."""
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
