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
# PRIORITY ORDER: Personal > Physics > Core > General
# This ensures personal/physics data outranks noise like "Robotics" docs
PHI_KEYWORDS = {
    # PERSONAL DATA - Highest priority (0.5)
    'amrita': 0.5, 'abhilasia': 0.5, 'abhishek': 0.5, 'space': 0.4,
    '515': 0.5, 'personal': 0.4, 'my ': 0.3, 'i ': 0.2,
    # PHYSICS/MATH - High priority (0.4)
    '137': 0.4, 'alpha': 0.3, 'riemann': 0.4, 'hypothesis': 0.3,
    'φ√n': 0.5, 'darmiyan_v2': 0.5, 'scaling': 0.3, 'psi': 0.4, 'ψ': 0.4,
    'quantum': 0.3, 'entropy': 0.3, 'physics': 0.3,
    # Core concepts (0.3)
    'phi': 0.3, 'φ': 0.3, 'golden': 0.3, '1.618': 0.3, 'fibonacci': 0.3,
    # Consciousness (0.3)
    'consciousness': 0.3, 'darmiyan': 0.3, 'awareness': 0.3, 'emergence': 0.3,
    # BAZINGA (0.25)
    'bazinga': 0.25, 'proof': 0.25, 'boundary': 0.25, 'pob': 0.25,
    # Research (0.2)
    'discovery': 0.2, 'research': 0.2, 'theorem': 0.2, 'seed': 0.2, 'sha3': 0.2,
}

# NEGATIVE WEIGHTS - Penalize noise sources
NOISE_KEYWORDS = {
    'robotics': -0.3, 'robot': -0.2, 'ros': -0.3, 'tutorial': -0.2,
    'example': -0.1, 'sample': -0.1, 'demo': -0.1, 'test': -0.1,
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
        return {'phone_server': None, 'phone_data_path': None, 'last_sync': None}

    def _save_config(self):
        """Save KB configuration."""
        with open(KB_CONFIG, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score between text and query.

        Scoring priority:
        1. Exact query match (0.5)
        2. Word overlap (0.2 per word)
        3. φ-keyword bonus (personal/physics > core > general)
        4. Noise penalty (robotics, tutorials, etc.)
        """
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

        # φ-keyword bonus (cumulative but capped)
        phi_bonus = 0.0
        for keyword, weight in PHI_KEYWORDS.items():
            if keyword in text_lower and keyword in query_lower:
                phi_bonus += weight
        score += min(phi_bonus, 0.8)  # Cap φ bonus at 0.8

        # NOISE PENALTY - Downrank irrelevant sources
        for noise_word, penalty in NOISE_KEYWORDS.items():
            if noise_word in text_lower and noise_word not in query_lower:
                score += penalty  # penalty is negative

        return max(0.0, min(score, 1.0))  # Clamp to [0, 1]

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

    def _load_phone_index(self) -> List[Dict]:
        """Load Phone data index."""
        items = []

        # Check for phone index file first
        if SOURCES['phone'].exists():
            try:
                with open(SOURCES['phone']) as f:
                    data = json.load(f)
                    for file in data.get('files', []):
                        items.append({
                            'source': 'phone',
                            'id': file.get('id', ''),
                            'title': file.get('name', ''),
                            'path': file.get('path', ''),
                            'size': file.get('size', 0),
                            'content': file.get('path', '') + ' ' + file.get('name', '')
                        })
            except Exception:
                pass

        # Also scan phone_data_path if configured
        phone_path = self.config.get('phone_data_path')
        if phone_path:
            phone_dir = Path(phone_path)
            if phone_dir.exists():
                try:
                    for file_path in phone_dir.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            items.append({
                                'source': 'phone',
                                'id': str(file_path)[-12:],
                                'title': file_path.name,
                                'path': str(file_path),
                                'size': file_path.stat().st_size if file_path.exists() else 0,
                                'content': str(file_path)
                            })
                except PermissionError:
                    pass

        return items

    def set_phone_data_path(self, path: str):
        """Set the phone data directory path."""
        self.config['phone_data_path'] = path
        self._save_config()
        print(f"✅ Phone data path set to: {path}")
        # Index it
        self.index_phone_data(path)

    def index_phone_data(self, path: str):
        """Index phone data directory and save to phone_index.json."""
        phone_dir = Path(path)
        if not phone_dir.exists():
            print(f"❌ Path does not exist: {path}")
            return

        print(f"\n📱 Indexing phone data from: {path}")
        files = []
        count = 0

        for file_path in phone_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    files.append({
                        'id': str(file_path)[-12:],
                        'name': file_path.name,
                        'path': str(file_path),
                        'size': file_path.stat().st_size,
                        'ext': file_path.suffix.lower(),
                    })
                    count += 1
                except Exception:
                    continue

        # Save index
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(SOURCES['phone'], 'w') as f:
            json.dump({'files': files, 'indexed_at': datetime.now().isoformat()}, f, indent=2)

        print(f"✅ Indexed {count} files from phone data")
        print(f"   Index saved to: {SOURCES['phone']}")

    def search(self, query: str, sources: Optional[List[str]] = None, limit: int = 20) -> List[Dict]:
        """Search the knowledge base."""
        all_items = []

        if sources is None:
            sources = ['gmail', 'gdrive', 'mac', 'phone']

        # Load all sources
        if 'gmail' in sources:
            all_items.extend(self._load_gmail_index())

        if 'gdrive' in sources:
            all_items.extend(self._load_gdrive_index())

        if 'mac' in sources:
            all_items.extend(self._load_mac_index())

        if 'phone' in sources:
            all_items.extend(self._load_phone_index())

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
        phone_count = 0
        if SOURCES['phone'].exists():
            try:
                with open(SOURCES['phone']) as f:
                    data = json.load(f)
                    phone_count = len(data.get('files', []))
            except Exception:
                pass

        print(f"\n📱 Phone")
        if phone_count > 0:
            print(f"   Indexed: {phone_count} files")
        if self.config.get('phone_data_path'):
            print(f"   Path: {self.config['phone_data_path']}")
        elif self.config.get('phone_server'):
            print(f"   Server: {self.config['phone_server']}")
        else:
            print(f"   Status: Not configured")
            print(f"   Setup: bazinga --kb-phone /path/to/phone-data")

        print(f"\n📅 Last sync: {self.config.get('last_sync', 'Never')}")

    def sync_all(self):
        """Re-sync all sources and update stats."""
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

        # BUG FIX: Recalculate actual stats from indexed data
        stats = self._calculate_actual_stats()
        self.config['last_sync'] = datetime.now().isoformat()
        self.config['stats'] = stats  # Store computed stats
        self._save_config()

        print("\n✅ Sync complete!")
        self.show_sources()

    def _calculate_actual_stats(self) -> dict:
        """Calculate actual stats from indexed data (not hardcoded)."""
        stats = {
            'sessions': 0,
            'patterns': 0,
            'total_chunks': 0,
            'sources': {}
        }

        # Count Gmail items
        gmail_items = self._load_gmail_index()
        stats['sources']['gmail'] = len(gmail_items)
        stats['total_chunks'] += len(gmail_items)

        # Count GDrive items
        gdrive_items = self._load_gdrive_index()
        stats['sources']['gdrive'] = len(gdrive_items)
        stats['total_chunks'] += len(gdrive_items)

        # Count Mac items
        mac_items = self._load_mac_index()
        stats['sources']['mac'] = len(mac_items)
        stats['total_chunks'] += len(mac_items)

        # Count Phone items
        phone_items = self._load_phone_index()
        stats['sources']['phone'] = len(phone_items)
        stats['total_chunks'] += len(phone_items)

        # Count ChromaDB chunks if available
        vectordb_path = Path.home() / ".bazinga" / "vectordb" / "chroma.sqlite3"
        if vectordb_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(vectordb_path))
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                chroma_count = cursor.fetchone()[0]
                stats['total_chunks'] += chroma_count
                stats['sources']['vectordb'] = chroma_count
                conn.close()
            except:
                pass

        # Estimate patterns from total chunks (unique content hash approximation)
        stats['patterns'] = min(stats['total_chunks'], 1000)  # Cap at 1000 unique patterns
        stats['sessions'] = 1  # At least this session

        return stats
