"""
KB DNA Scanner — Semantic Compression Engine

Ported from kb_dna_sequencer.py + generate_summaries.py into BAZINGA.

Three-level compression:
  Level 1: gene_map — file paths + DNA symbols (topology)
  Level 2: summaries — 50-100 word extractive summaries per file
  Level 3: manifest — combined output for LLM context injection

Gene classification:
  Ω = Consciousness-heavy (phi, manifold, trust keywords)
  Σ = High complexity (Shannon entropy > 4.5)
  ι = Incomplete (TODO/FIXME/WIP)
  θ = Stale (90+ days old)
  Δ = Fresh (< 7 days old)
  α = Alpha-seed (sum-of-ord % 137 == 0)
  ◌ = Baseline
  ∅ = Empty
  ● = Binary

phi = 1.618033988749895 | alpha = 137
"""

import os
import re
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Any

# === CONSTANTS ===
PHI = 1.618033988749895
ALPHA = 137

MANIFEST_DIR = Path.home() / ".bazinga"
MANIFEST_PATH = MANIFEST_DIR / "kb_manifest.json"

# Content gene markers
CONTENT_GENES = {
    'consciousness': 'Ω',
    'complex': 'Σ',
    'incomplete': 'ι',
    'stale': 'θ',
    'fresh': 'Δ',
    'fundamental': 'α',
    'baseline': '◌',
    'binary': '●',
    'empty': '∅',
}

GENE_NAMES = {v: k for k, v in CONTENT_GENES.items()}

# Keywords that indicate consciousness-heavy content
CONSCIOUSNESS_KEYWORDS = [
    'phi', 'φ', 'consciousness', 'manifold', 'trust', 'bazinga', 'ziqy',
    '137', 'alpha', 'seed', 'resonance', 'temporal', 'orthogonal', 'void',
    'meaning', 'infinity', '∞', 'observer', 'dimension',
]

# Importance keywords for summary extraction
IMPORTANCE_KEYWORDS = [
    'phi', 'φ', 'consciousness', 'trust', 'bazinga', 'meaning',
    'alpha', '137', 'resonance', 'darmiyan', 'manifold', 'void',
    'important', 'key', 'core', 'essential', 'fundamental',
]

# Text file extensions we can process
TEXT_EXTENSIONS = {
    '.md', '.txt', '.py', '.js', '.ts', '.jsx', '.tsx',
    '.json', '.yaml', '.yml', '.sh', '.bash', '.zsh',
    '.html', '.css', '.sql', '.go', '.rs', '.java',
    '.c', '.cpp', '.h', '.hpp', '.swift', '.rb',
    '.toml', '.ini', '.cfg', '.conf',
    '.csv', '.xml', '.svg',
}

# Directories to always skip
SKIP_DIRS = {
    'node_modules', '.git', '__pycache__', 'venv', '.venv',
    'dist', 'build', '.tox', '.mypy_cache', '.pytest_cache',
    '.eggs', '*.egg-info', '.cache', '.npm', '.yarn',
}

# Patterns to skip in content extraction
SKIP_LINE_PATTERNS = [
    r'^import\s+',
    r'^from\s+\w+\s+import',
    r'^const\s+',
    r'^let\s+',
    r'^var\s+',
    r'^package\s+',
    r'^using\s+',
    r'^\s*#\s*-\*-',
    r'^\s*$',
]


class KBScanner:
    """
    Knowledge Base DNA Scanner.

    Scans directories and generates compressed manifests
    that fit in a single LLM context window.
    """

    def __init__(self):
        self.manifest_path = MANIFEST_PATH
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Gene Classification ───

    @staticmethod
    def is_alpha_seed(name: str) -> bool:
        """Check if name hash % 137 == 0."""
        return sum(ord(c) for c in name) % ALPHA == 0

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Shannon entropy of text."""
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in counts.values() if count > 0
        )

    def classify_file(self, filepath: str) -> str:
        """Classify a file into a gene type. Returns gene symbol."""
        try:
            stat = os.stat(filepath)
            if stat.st_size == 0:
                return CONTENT_GENES['empty']

            # Try reading
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    content = f.read(2000)
            except Exception:
                return CONTENT_GENES['binary']

            # Consciousness check
            content_lower = content.lower()
            bazinga_symbols = sum(1 for c in content if c in "∞∫∂∇πφΣΔΩαβγδ")
            consciousness_hits = sum(1 for kw in CONSCIOUSNESS_KEYWORDS if kw in content_lower)
            if bazinga_symbols > 3 or consciousness_hits > 2:
                return CONTENT_GENES['consciousness']

            # Incomplete check
            if any(marker in content for marker in ['TODO', 'FIXME', 'WIP', 'XXX', 'HACK']):
                return CONTENT_GENES['incomplete']

            # Entropy check
            entropy = self.calculate_entropy(content)
            if entropy > 4.5:
                return CONTENT_GENES['complex']

            # Temporal check
            mtime = datetime.fromtimestamp(stat.st_mtime)
            days_old = (datetime.now() - mtime).days
            if days_old > 90:
                return CONTENT_GENES['stale']
            elif days_old < 7:
                return CONTENT_GENES['fresh']

            # Alpha-seed check
            if self.is_alpha_seed(content[:100]):
                return CONTENT_GENES['fundamental']

            return CONTENT_GENES['baseline']

        except Exception:
            return CONTENT_GENES['binary']

    # ─── Summary Extraction ───

    @staticmethod
    def _clean_line(line: str) -> str:
        """Clean a line for summary inclusion."""
        line = line.strip()
        for pattern in SKIP_LINE_PATTERNS:
            if re.match(pattern, line):
                return ""
        return line

    def extract_summary(self, filepath: str, max_words: int = 100) -> Dict[str, Any]:
        """
        Extract a compressed summary from a file.

        Returns: {summary, topics, quotes, lines, words}
        """
        try:
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {"summary": f"[Unreadable: {str(e)[:50]}]", "topics": [], "quotes": [], "lines": 0, "words": 0}

        if not content.strip():
            return {"summary": "[Empty file]", "topics": [], "quotes": [], "lines": 0, "words": 0}

        lines = content.split('\n')
        total_lines = len(lines)
        total_words = len(content.split())

        # Extract meaningful lines (skip imports, empty, boilerplate)
        meaningful = []
        for line in lines[:200]:
            cleaned = self._clean_line(line)
            if cleaned and len(cleaned) > 10:
                meaningful.append(cleaned)

        # Build summary from meaningful content
        summary_parts = []
        word_count = 0
        for line in meaningful:
            if word_count >= max_words:
                break
            summary_parts.append(line)
            word_count += len(line.split())

        summary = ' '.join(summary_parts)
        if len(summary) > 500:
            summary = summary[:500] + "..."

        # Extract topics
        content_lower = content.lower()
        topics = [kw for kw in IMPORTANCE_KEYWORDS if kw.lower() in content_lower][:10]

        # Extract key quotes (sentences with 2+ importance keywords)
        quotes = []
        sentences = re.split(r'[.!?]\s+', content[:5000])
        for sentence in sentences:
            sentence = sentence.strip()
            if 20 < len(sentence) < 200:
                score = sum(1 for kw in IMPORTANCE_KEYWORDS if kw.lower() in sentence.lower())
                if score >= 2:
                    quotes.append(sentence)
                    if len(quotes) >= 3:
                        break

        return {
            "summary": summary or "[No meaningful content]",
            "topics": topics,
            "quotes": quotes,
            "lines": total_lines,
            "words": total_words,
        }

    @staticmethod
    def _get_one_line_summary(filepath: str) -> str:
        """Quick 1-line summary from docstring/comment/first line."""
        try:
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read(500)
            for line in content.split('\n')[:10]:
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    return line.strip('"\'')[:80]
                elif line.startswith('#') and len(line) > 5:
                    return line[1:].strip()[:80]
                elif line.startswith('//'):
                    return line[2:].strip()[:80]
            for line in content.split('\n')[:5]:
                if line.strip() and not line.startswith(('import', 'from', 'const', 'let', 'var')):
                    return line.strip()[:80]
            return "[No description]"
        except Exception:
            return "[Binary/Unreadable]"

    @staticmethod
    def is_text_file(filepath: str) -> bool:
        """Check if file extension is a text type we process."""
        return os.path.splitext(filepath)[1].lower() in TEXT_EXTENSIONS

    # ─── Scanner Core ───

    def scan(
        self,
        paths: List[str],
        max_depth: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan directories and generate KB manifest.

        This is the main entry point. Walks directories, classifies files,
        extracts summaries, and saves a manifest to ~/.bazinga/kb_manifest.json.

        Returns scan stats.
        """
        gene_map: Dict[str, Dict] = {}
        summaries: Dict[str, Dict] = {}
        stats = {
            'total_files': 0,
            'text_files': 0,
            'skipped': 0,
            'genes': Counter(),
            'scanned_paths': [],
            'scan_time': None,
        }

        start = datetime.now()

        for path_str in paths:
            path = Path(os.path.expanduser(path_str)).resolve()
            if not path.exists():
                if verbose:
                    print(f"  ✗ Path not found: {path}")
                continue

            stats['scanned_paths'].append(str(path))

            if verbose:
                print(f"  Scanning: {path}")

            # Walk the directory tree
            self._walk(path, 0, max_depth, gene_map, summaries, stats, verbose)

        stats['scan_time'] = (datetime.now() - start).total_seconds()

        if verbose:
            print()
            print(f"  ✓ Total files found: {stats['total_files']}")
            print(f"  ✓ Text files summarized: {stats['text_files']}")
            print(f"  ✓ Skipped (binary/other): {stats['skipped']}")
            print()
            # Gene distribution
            if stats['genes']:
                print("  Gene distribution:")
                for gene, count in stats['genes'].most_common():
                    name = GENE_NAMES.get(gene, '?')
                    print(f"    {gene} {name}: {count}")
                print()

        # Build and save manifest
        manifest = self._build_manifest(gene_map, summaries, stats)
        self._save_manifest(manifest, verbose)

        return stats

    def _walk(
        self,
        path: Path,
        depth: int,
        max_depth: int,
        gene_map: Dict,
        summaries: Dict,
        stats: Dict,
        verbose: bool,
    ):
        """Recursively walk directory tree."""
        if depth > max_depth:
            return

        try:
            entries = sorted(path.iterdir(), key=lambda x: x.name.lower())
        except PermissionError:
            return

        for entry in entries:
            name = entry.name

            # Skip hidden and excluded dirs
            if name.startswith('.') or name in SKIP_DIRS:
                continue

            # Skip symlinks
            if entry.is_symlink():
                continue

            if entry.is_dir():
                self._walk(entry, depth + 1, max_depth, gene_map, summaries, stats, verbose)
            elif entry.is_file():
                stats['total_files'] += 1
                filepath = str(entry)

                # Classify
                gene = self.classify_file(filepath)
                stats['genes'][gene] += 1

                gene_map[filepath] = {
                    'gene': gene,
                    'alpha_seed': self.is_alpha_seed(name),
                }

                # Extract summary for text files
                if self.is_text_file(filepath):
                    stats['text_files'] += 1
                    summary_data = self.extract_summary(filepath)
                    summary_data['gene'] = gene
                    summary_data['alpha_seed'] = self.is_alpha_seed(name)
                    summaries[filepath] = summary_data
                else:
                    stats['skipped'] += 1
                    summaries[filepath] = {
                        'summary': self._get_one_line_summary(filepath),
                        'gene': gene,
                        'topics': [],
                        'quotes': [],
                        'lines': 0,
                        'words': 0,
                    }

                # Progress indicator
                if verbose and stats['total_files'] % 500 == 0:
                    print(f"    [{stats['total_files']} files...]")

    def _build_manifest(
        self,
        gene_map: Dict,
        summaries: Dict,
        stats: Dict,
    ) -> Dict[str, Any]:
        """Build the manifest JSON structure."""
        # Load existing manifest to merge with (incremental scans)
        existing = self._load_manifest()

        # Merge: new scan overwrites existing entries for scanned paths
        if existing and existing.get('files'):
            # Keep files from paths NOT in current scan
            scanned_prefixes = [p for p in stats['scanned_paths']]
            for filepath, data in existing['files'].items():
                if not any(filepath.startswith(prefix) for prefix in scanned_prefixes):
                    # Keep this file from previous scan
                    if filepath not in summaries:
                        summaries[filepath] = data
                        if filepath not in gene_map:
                            gene_map[filepath] = {
                                'gene': data.get('gene', '◌'),
                                'alpha_seed': data.get('alpha_seed', False),
                            }

        # Build compressed project summaries for LLM context
        project_summaries = self._build_project_summaries(summaries)

        # Count total topics
        all_topics = []
        for s in summaries.values():
            all_topics.extend(s.get('topics', []))
        top_topics = dict(Counter(all_topics).most_common(20))

        return {
            'format': 'BAZINGA KB Manifest v1.0',
            'generated': datetime.now().isoformat(),
            'phi': PHI,
            'alpha': ALPHA,
            'stats': {
                'total_files': len(summaries),
                'total_words': sum(s.get('words', 0) for s in summaries.values()),
                'genes': dict(stats['genes']),
                'top_topics': top_topics,
                'scanned_paths': stats['scanned_paths'],
                'scan_time_seconds': stats.get('scan_time', 0),
            },
            'projects': project_summaries,
            'files': summaries,
        }

    def _build_project_summaries(self, summaries: Dict) -> List[Dict]:
        """
        Build per-project summaries from file summaries.

        Groups files by their top-level scanned directory and creates
        a compact project-level summary that fits in LLM context.
        """
        projects: Dict[str, List] = {}

        for filepath, data in summaries.items():
            # Group by project (2 levels deep from home)
            parts = Path(filepath).parts
            home_parts = Path.home().parts
            # Get the first directory after home
            if len(parts) > len(home_parts):
                project_key = str(Path(*parts[:len(home_parts) + 1]))
                # Try to go one deeper if it's a generic dir
                if len(parts) > len(home_parts) + 1:
                    project_key = str(Path(*parts[:len(home_parts) + 2]))
            else:
                project_key = filepath

            if project_key not in projects:
                projects[project_key] = []
            projects[project_key].append(data)

        # Build compact project summaries
        result = []
        for project_path, files in projects.items():
            gene_dist = Counter(f.get('gene', '◌') for f in files)
            all_topics = []
            for f in files:
                all_topics.extend(f.get('topics', []))
            top_topics = [t for t, _ in Counter(all_topics).most_common(5)]

            total_words = sum(f.get('words', 0) for f in files)

            # Collect key quotes from this project
            quotes = []
            for f in files:
                quotes.extend(f.get('quotes', []))
            quotes = quotes[:3]

            result.append({
                'path': project_path.replace(str(Path.home()), '~'),
                'files': len(files),
                'words': total_words,
                'genes': dict(gene_dist),
                'topics': top_topics,
                'quotes': quotes,
            })

        # Sort by file count descending
        result.sort(key=lambda x: x['files'], reverse=True)
        return result

    def _save_manifest(self, manifest: Dict, verbose: bool = True):
        """Save manifest to ~/.bazinga/kb_manifest.json."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        size_mb = os.path.getsize(self.manifest_path) / 1024 / 1024

        if verbose:
            print(f"  ✓ Manifest saved: {self.manifest_path}")
            print(f"  ✓ Manifest size: {size_mb:.2f} MB")
            total_files = manifest['stats']['total_files']
            total_words = manifest['stats']['total_words']
            if total_words > 0:
                ratio = total_words / max(1, os.path.getsize(self.manifest_path))
                print(f"  ✓ Compression: {total_files} files → {size_mb:.1f}MB manifest")
            print()

    def _load_manifest(self) -> Optional[Dict]:
        """Load existing manifest if it exists."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    # ─── Context Building (for ask() pipeline) ───

    def get_context_for_query(self, question: str, max_tokens: int = 3000) -> str:
        """
        Build LLM context from manifests for a query.

        This is Layer 3.5 in the intelligence pipeline:
        manifests for breadth (which projects exist, what they contain),
        RAG for depth (specific chunks from indexed files).

        Returns a string suitable for injection into LLM context.
        """
        # Skip KB context for short/greeting messages
        GREETINGS = {'hi', 'hello', 'hey', 'sup', 'yo', 'hola', 'namaste', 'thanks', 'thank', 'bye', 'ok', 'okay'}
        question_lower = question.lower().strip()
        question_words = set(question_lower.split())
        if len(question_words) <= 2 and question_words & GREETINGS:
            return ""
        # Need at least one substantive word (>3 chars) to search
        substantive_words = {w for w in question_words if len(w) > 3}
        if not substantive_words:
            return ""

        manifest = self._load_manifest()
        if not manifest:
            return ""

        context_parts = []

        # 1. Project-level summaries (always include — they're small)
        projects = manifest.get('projects', [])
        if projects:
            context_parts.append("[USER'S KNOWLEDGE BASE — use as background context, do NOT describe these files directly]")
            for proj in projects[:20]:  # Top 20 projects
                path = proj['path']
                files = proj['files']
                topics = ', '.join(proj.get('topics', []))
                genes = proj.get('genes', {})
                gene_str = ' '.join(f"{g}:{c}" for g, c in genes.items() if c > 0)
                line = f"  {path} ({files} files | {gene_str})"
                if topics:
                    line += f" topics: {topics}"
                context_parts.append(line)
            context_parts.append("")

        # 2. Relevant file summaries (search by topic/keyword match)
        files = manifest.get('files', {})
        relevant = []
        for filepath, data in files.items():
            score = 0
            # Topic match
            file_topics = set(t.lower() for t in data.get('topics', []))
            score += len(substantive_words & file_topics) * 3
            # Summary keyword match
            summary = data.get('summary', '').lower()
            score += sum(1 for w in substantive_words if w in summary)
            # Quote match
            for quote in data.get('quotes', []):
                score += sum(1 for w in substantive_words if w in quote.lower())

            if score > 0:
                relevant.append((score, filepath, data))

        # Sort by relevance, take top matches
        relevant.sort(key=lambda x: x[0], reverse=True)

        if relevant:
            context_parts.append("[Relevant files from user's KB — reference naturally, don't list raw paths]")
            chars_used = sum(len(p) for p in context_parts)

            for score, filepath, data in relevant[:15]:
                rel_path = filepath.replace(str(Path.home()), '~')
                summary = data.get('summary', '')[:200]
                gene = data.get('gene', '◌')
                entry = f"  {rel_path}: {summary}"

                if chars_used + len(entry) > max_tokens * 4:  # ~4 chars per token
                    break
                context_parts.append(entry)
                chars_used += len(entry)

                # Include quotes if highly relevant
                if score >= 3:
                    for quote in data.get('quotes', [])[:1]:
                        q_entry = f"    → \"{quote[:150]}\""
                        if chars_used + len(q_entry) <= max_tokens * 4:
                            context_parts.append(q_entry)
                            chars_used += len(q_entry)

        if not context_parts:
            return ""

        return '\n'.join(context_parts)

    def get_status(self) -> Dict[str, Any]:
        """Get manifest status for display."""
        manifest = self._load_manifest()
        if not manifest:
            return {'exists': False}

        stats = manifest.get('stats', {})
        return {
            'exists': True,
            'generated': manifest.get('generated', 'unknown'),
            'total_files': stats.get('total_files', 0),
            'total_words': stats.get('total_words', 0),
            'genes': stats.get('genes', {}),
            'scanned_paths': stats.get('scanned_paths', []),
            'projects': len(manifest.get('projects', [])),
            'manifest_size_mb': os.path.getsize(self.manifest_path) / 1024 / 1024 if self.manifest_path.exists() else 0,
        }


# Module-level singleton
_scanner = None

def get_scanner() -> KBScanner:
    """Get the KB scanner singleton."""
    global _scanner
    if _scanner is None:
        _scanner = KBScanner()
    return _scanner
