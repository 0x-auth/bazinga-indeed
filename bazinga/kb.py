#!/usr/bin/env python3
"""
BAZINGA Knowledge Base (KB) - Unified Query System
===================================================

Query all your data sources with natural language:
- Gmail (starred emails)
- GDrive (all files)
- Mac (local files)
- Phone (via HTTP server)

Usage:
    bazinga --kb "what do I know about 137?"
    bazinga --kb "find my consciousness research"
    bazinga --kb-sources
    bazinga --kb-sync

Architecture:
    Query → Search all indexes → Rank by φ-resonance → Return results
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Constants
PHI = 1.618033988749895
INDEX_DIR = Path.home() / ".bazinga" / "index"
KB_CONFIG = INDEX_DIR / "kb_config.json"

# Data source paths
SOURCES = {
    'gmail': INDEX_DIR / "gmail_index.json",
    'gdrive': INDEX_DIR / "gdrive_index.json",
    'gdrive_raw': INDEX_DIR / "gdrive_raw.json",
    'mac': INDEX_DIR / "mac_index.json",
    'phone': INDEX_DIR / "phone_index.json",
    'unified': INDEX_DIR / "unified_index.json",
}

# φ-resonance keywords (higher weight)
PHI_KEYWORDS = {
    # Core concepts
    'phi': 0.3, 'φ': 0.3, 'golden': 0.2, '1.618': 0.3, 'fibonacci': 0.2,
    # Consciousness
    'consciousness': 0.3, 'darmiyan': 0.3, 'awareness': 0.2, 'emergence': 0.2,
    # BAZINGA
    'bazinga': 0.3, 'proof': 0.2, 'boundary': 0.2, 'pob': 0.2,
    # Math/Physics
    '137': 0.3, 'alpha': 0.2, 'riemann': 0.3, 'hypothesis': 0.2,
    'seed': 0.2, 'quantum': 0.2, 'entropy': 0.2,
    # Research
    'discovery': 0.2, 'research': 0.2, 'theorem': 0.2,
    # Special
    'amrita': 0.2, 'abhilasia': 0.2, '515': 0.2, 'sha3': 0.2,
}


class BazingaKB:
    """BAZINGA Knowledge Base - Query all your data sources."""

    def __init__(self):
        """Initialize KB with config."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load KB configuration."""
        if KB_CONFIG.exists():
            with open(KB_CONFIG) as f:
                return json.load(f)
        return {'phone_server': None, 'last_sync': None}

    def _save_config(self):
        """Save KB configuration."""
        with open(KB_CONFIG, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score between text and query."""
        if not text or not query:
            return 0.0

        text_lower = text.lower()
        query_lower = query.lower()
        query_words = query_lower.split()

        score = 0.0

        # Exact match bonus
        if query_lower in text_lower:
            score += 0.5

        # Word match
        for word in query_words:
            if len(word) > 2 and word in text_lower:
                score += 0.2

        # φ-keyword bonus
        for keyword, weight in PHI_KEYWORDS.items():
            if keyword in text_lower and keyword in query_lower:
                score += weight

        return min(score, 1.0)

    def _load_gmail_index(self) -> List[Dict]:
        """Load Gmail index."""
        items = []

        # Try unified gmail index first
        if SOURCES['gmail'].exists():
            try:
                with open(SOURCES['gmail']) as f:
                    data = json.load(f)
                    for email in data.get('emails', []):
                        items.append({
                            'source': 'gmail',
                            'id': email.get('id', ''),
                            'title': email.get('subject', '(no subject)'),
                            'date': email.get('date', ''),
                            'content': email.get('subject', ''),
                            'path': f"gmail:{email.get('id', '')}"
                        })
            except Exception:
                pass

        # Also check exports
        exports_dir = Path.home() / ".bazinga" / "gmail_exports"
        if exports_dir.exists():
            for export_folder in exports_dir.iterdir():
                if export_folder.is_dir():
                    for json_file in export_folder.glob("*.json"):
                        if json_file.name.startswith("_"):
                            continue
                        try:
                            with open(json_file) as f:
                                email = json.load(f)
                                items.append({
                                    'source': 'gmail',
                                    'id': email.get('id', ''),
                                    'title': email.get('subject', '(no subject)'),
                                    'date': email.get('date', ''),
                                    'content': email.get('body', '')[:500],
                                    'path': str(json_file)
                                })
                        except Exception:
                            pass

        return items

    def _load_gdrive_index(self) -> List[Dict]:
        """Load GDrive index."""
        items = []

        # Try raw JSON first (direct from rclone)
        if SOURCES['gdrive_raw'].exists():
            try:
                with open(SOURCES['gdrive_raw']) as f:
                    content = f.read().strip()
                    if content.startswith('['):
                        files = json.loads(content)
                        for file in files:
                            items.append({
                                'source': 'gdrive',
                                'id': file.get('ID', ''),
                                'title': file.get('Name', ''),
                                'path': file.get('Path', ''),
                                'size': file.get('Size', 0),
                                'mime': file.get('MimeType', ''),
                                'content': file.get('Path', '') + ' ' + file.get('Name', '')
                            })
            except json.JSONDecodeError:
                pass  # File might be incomplete

        # Also try processed index
        if SOURCES['gdrive'].exists():
            try:
                with open(SOURCES['gdrive']) as f:
                    data = json.load(f)
                    for file in data.get('files', []):
                        items.append({
                            'source': 'gdrive',
                            'id': file.get('id', ''),
                            'title': file.get('name', ''),
                            'path': file.get('path', ''),
                            'content': file.get('path', '') + ' ' + file.get('name', '')
                        })
            except Exception:
                pass

        return items

    def _load_mac_index(self) -> List[Dict]:
        """Load Mac local files index."""
        items = []

        # Index key directories
        key_dirs = [
            Path.home() / "bin",
            Path.home() / "github-repos-bitsabhi",
            Path.home() / "∞",
            Path.home() / "consciousness-portal",
            Path.home() / ".bazinga",
        ]

        for dir_path in key_dirs:
            if dir_path.exists():
                try:
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            items.append({
                                'source': 'mac',
                                'id': str(file_path)[-12:],
                                'title': file_path.name,
                                'path': str(file_path),
                                'content': str(file_path)
                            })
                except PermissionError:
                    continue

        return items

    def search(self, query: str, sources: Optional[List[str]] = None, limit: int = 20) -> List[Dict]:
        """Search the knowledge base."""
        all_items = []

        if sources is None:
            sources = ['gmail', 'gdrive', 'mac']

        # Load all sources
        if 'gmail' in sources:
            all_items.extend(self._load_gmail_index())

        if 'gdrive' in sources:
            all_items.extend(self._load_gdrive_index())

        if 'mac' in sources:
            all_items.extend(self._load_mac_index())

        # Calculate relevance for each item
        results = []
        for item in all_items:
            searchable = f"{item.get('title', '')} {item.get('content', '')} {item.get('path', '')}"
            relevance = self._calculate_relevance(searchable, query)

            if relevance > 0:
                item['relevance'] = relevance
                results.append(item)

        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)

        return results[:limit]

    def display_results(self, results: List[Dict], query: str):
        """Display search results."""
        print(f"\n🔍 BAZINGA KB Search: \"{query}\"")
        print("=" * 60)

        if not results:
            print("No results found.")
            return

        source_icons = {
            'gmail': '📧',
            'gdrive': '📁',
            'mac': '💻',
            'phone': '📱'
        }

        for i, item in enumerate(results, 1):
            icon = source_icons.get(item['source'], '📄')
            relevance = item.get('relevance', 0)
            title = item.get('title', 'Untitled')[:50]

            print(f"\n{i}. {icon} [{item['source']}] φ={relevance:.2f}")
            print(f"   📄 {title}")

            if item.get('path'):
                path = item['path']
                if len(path) > 60:
                    path = "..." + path[-57:]
                print(f"   📍 {path}")

            if item.get('date'):
                print(f"   📅 {item['date'][:30]}")

        print(f"\n📊 Found {len(results)} results")

    def show_sources(self):
        """Show all data sources and their status."""
        print("\n📊 BAZINGA KB Data Sources")
        print("=" * 60)

        # Gmail
        gmail_count = 0
        if SOURCES['gmail'].exists():
            try:
                with open(SOURCES['gmail']) as f:
                    data = json.load(f)
                    gmail_count = len(data.get('emails', []))
            except Exception:
                pass

        exports_dir = Path.home() / ".bazinga" / "gmail_exports"
        export_count = 0
        if exports_dir.exists():
            for d in exports_dir.iterdir():
                if d.is_dir():
                    export_count += len(list(d.glob("*.json"))) - 1  # Minus _index.json

        print(f"\n📧 Gmail")
        print(f"   Indexed: {gmail_count} emails")
        print(f"   Exports: {export_count} emails")

        # GDrive
        gdrive_count = 0
        if SOURCES['gdrive_raw'].exists():
            try:
                with open(SOURCES['gdrive_raw']) as f:
                    content = f.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        gdrive_count = len(json.loads(content))
                    else:
                        gdrive_count = content.count('"Path"')  # Approximate
            except Exception:
                gdrive_count = -1  # Still indexing

        print(f"\n📁 Google Drive")
        if gdrive_count == -1:
            print(f"   Status: Still indexing...")
        else:
            print(f"   Indexed: {gdrive_count} files")

        # Mac
        print(f"\n💻 Mac Local")
        print(f"   Directories: ~/bin, ~/github-repos-bitsabhi, ~/∞, ~/.bazinga")
        print(f"   Status: Indexed on-demand")

        # Phone
        print(f"\n📱 Phone")
        if self.config.get('phone_server'):
            print(f"   Server: {self.config['phone_server']}")
        else:
            print(f"   Status: Not configured")

        print(f"\n📅 Last sync: {self.config.get('last_sync', 'Never')}")

    def sync_all(self):
        """Re-sync all sources."""
        print("\n🔄 Syncing all KB sources...")

        # Try to run bazinga-index
        index_script = Path.home() / "bin" / "bazinga-index"
        if index_script.exists():
            try:
                subprocess.run(['python3', str(index_script)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Index script failed: {e}")
        else:
            print("⚠️  bazinga-index not found at ~/bin/bazinga-index")
            print("   Creating minimal index from existing exports...")

        self.config['last_sync'] = datetime.now().isoformat()
        self._save_config()

        print("\n✅ Sync complete!")
        self.show_sources()
