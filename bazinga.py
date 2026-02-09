#!/usr/bin/env python3
"""
BAZINGA - Distributed AI
"Intelligence distributed, not controlled"

Usage:
    python bazinga.py                    # Interactive mode
    python bazinga.py --ask "question"   # Ask a question
    python bazinga.py --index ~/Documents # Index a directory

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
sys.path.insert(0, str(Path(__file__).parent))

from src.core.intelligence.real_ai import RealAI
from src.core.lambda_g import LambdaGOperator, PHI


class BAZINGA:
    """
    BAZINGA - Distributed AI that belongs to everyone.

    Features:
    - Index your files into semantic vector space
    - Search with φ-coherence ranking
    - Generate responses via local Ollama or cloud APIs
    - No central control - runs anywhere
    """

    VERSION = "1.0.0"

    def __init__(self):
        self.lambda_g = LambdaGOperator()
        self.ai = RealAI()
        self.session_start = datetime.now()
        self.queries = []

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
        print("║   Mode: Local (use --distributed for cloud LLMs)                 ║")
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
        """Ask a question and get an answer."""
        self.queries.append(question)
        return await self.ai.ask(question, verbose=verbose)

    async def interactive(self):
        """Run interactive mode."""
        print("◊ BAZINGA INTERACTIVE MODE ◊")
        print("-" * 40)
        print("Commands:")
        print("  /index <path>  - Index a directory")
        print("  /stats         - Show statistics")
        print("  /quit          - Exit")
        print("-" * 40)
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
                    stats = self.ai.get_stats()
                    print(f"\n📊 Stats:")
                    print(f"   Chunks: {stats.get('total_chunks', 0)}")
                    print(f"   Queries: {len(self.queries)}")
                    print()
                    continue

                response = await self.ask(query)
                print(f"\nBAZINGA: {response}\n")

            except KeyboardInterrupt:
                print("\n\n✨ BAZINGA signing off.")
                break
            except EOFError:
                break


async def main():
    parser = argparse.ArgumentParser(
        description="BAZINGA - Distributed AI"
    )
    parser.add_argument('--ask', type=str, help='Ask a question')
    parser.add_argument('--index', nargs='+', help='Directories to index')
    parser.add_argument('--demo', action='store_true', help='Run demo')

    args = parser.parse_args()

    bazinga = BAZINGA()

    if args.index:
        await bazinga.index(args.index)
    elif args.ask:
        response = await bazinga.ask(args.ask)
        print(f"\n{response}\n")
    elif args.demo:
        print("Running demo...")
        await bazinga.index([str(Path(__file__).parent)])
        response = await bazinga.ask("What is BAZINGA?")
        print(f"\n{response}\n")
    else:
        await bazinga.interactive()


if __name__ == "__main__":
    asyncio.run(main())
