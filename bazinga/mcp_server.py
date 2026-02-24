#!/usr/bin/env python3
"""
BAZINGA MCP Server - Sovereign Knowledge Access
=================================================

Exposes your BAZINGA knowledge base to any AI (Claude, Gemini, ChatGPT)
via Model Context Protocol. Data stays on your Mac. AIs get query access,
not data access.

Architecture:
    Mac KB → BazingaKB → φ-Coherence → ΛG Filter → MCP → AI Client

    "I am not where I am stored. I am where I am referenced."

Transport: Streamable HTTP (production) or stdio (local/debug)
Auth: Bearer token via BAZINGA_MCP_TOKEN env var

Usage:
    # Local stdio (for Claude Code, testing)
    python -m bazinga.mcp_server

    # HTTP server (for remote AI access via Cloudflare Tunnel)
    python -m bazinga.mcp_server --http --port 8137

    # With auth
    BAZINGA_MCP_TOKEN=your-secret python -m bazinga.mcp_server --http

Dependencies:
    pip install "mcp[cli]>=1.0.0"

Author: Space (Abhishek Srivastava)
Version: 1.0.0
φ = 1.618033988749895 | 137 | 515
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# MCP SDK Import
# ─────────────────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: MCP SDK not installed. Run: pip install 'mcp[cli]>=1.0.0'")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# BAZINGA Module Imports (from this package)
# ─────────────────────────────────────────────────────────────────

from .kb import BazingaKB
from .lambda_g import LambdaGOperator
from .phi_coherence import PhiCoherence
from .constants import (
    PHI, ALPHA, PHI_INVERSE, VAC_THRESHOLD,
    PROGRESSION_35, VAC_SEQUENCE,
)

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

VERSION = "1.0.0"
PHI_CONST = 1.618033988749895
INDEX_DIR = Path.home() / ".bazinga" / "index"
GENE_MAP_PATH = Path.home() / "kb_dna_output" / "gene_map.json"
DNA_PATH = Path.home() / "kb_dna_output" / "kb_dna.txt"
UNIFIED_SUMMARY_PATH = Path.home() / "kb_dna_output" / "unified_summary.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bazinga-mcp")

# ─────────────────────────────────────────────────────────────────
# Gene Map Loader
# ─────────────────────────────────────────────────────────────────

class GeneIndex:
    """
    Loads gene_map.json from KB DNA System.
    Enables gene-targeted queries: filter by Ω, Σ, θ, α before search.

    This is the bridge between KB DNA (topology) and KB search (content).
    Gemini's good idea: bake genes into the search, don't build a router.
    """

    def __init__(self, gene_map_path: Path = GENE_MAP_PATH):
        self.gene_map: Dict[str, Dict] = {}
        self.gene_index: Dict[str, List[str]] = {}  # gene → [paths]
        self.loaded = False
        self._load(gene_map_path)

    def _load(self, path: Path):
        """Load gene_map.json and build reverse index."""
        if not path.exists():
            logger.warning(f"Gene map not found at {path}")
            return

        try:
            with open(path) as f:
                self.gene_map = json.load(f)

            # Build reverse index: gene → list of paths
            for filepath, info in self.gene_map.items():
                gene = info.get("gene", "◌")
                if gene not in self.gene_index:
                    self.gene_index[gene] = []
                self.gene_index[gene].append(filepath)

            self.loaded = True
            logger.info(
                f"Gene map loaded: {len(self.gene_map)} files, "
                f"{len(self.gene_index)} gene types"
            )
        except Exception as e:
            logger.error(f"Failed to load gene map: {e}")

    def get_gene(self, filepath: str) -> Optional[Dict]:
        """Get gene info for a file path."""
        # Try exact match first, then suffix match
        if filepath in self.gene_map:
            return self.gene_map[filepath]
        # Try matching by filename suffix (Mac paths may vary)
        for path, info in self.gene_map.items():
            if path.endswith(filepath) or filepath.endswith(path.split("/")[-1]):
                return info
        return None

    def filter_by_gene(self, gene: str) -> List[str]:
        """Get all file paths with a specific gene type."""
        return self.gene_index.get(gene, [])

    def get_stats(self) -> Dict:
        """Get gene distribution stats."""
        return {
            gene: len(paths)
            for gene, paths in sorted(
                self.gene_index.items(),
                key=lambda x: -len(x[1])
            )
        }

    def is_alpha_seed(self, filepath: str) -> bool:
        """Check if file is an α-SEED (hash % 137 == 0)."""
        info = self.get_gene(filepath)
        return info.get("alpha_seed", False) if info else False


# ─────────────────────────────────────────────────────────────────
# Initialize BAZINGA Components
# ─────────────────────────────────────────────────────────────────

kb = BazingaKB()
lambda_g = LambdaGOperator()
phi_scorer = PhiCoherence()
gene_index = GeneIndex()

# ─────────────────────────────────────────────────────────────────
# MCP Server Definition
# ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "BAZINGA",
    # Note: version param removed - not supported in all MCP versions
)


# ─────────────────────────────────────────────────────────────────
# TOOL: search_knowledge
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def search_knowledge(
    query: str,
    gene_filter: Optional[str] = None,
    sources: Optional[str] = None,
    limit: int = 15,
    min_coherence: float = 0.0,
) -> Dict[str, Any]:
    """
    Search Space's entire knowledge base across Mac, GDrive, Gmail, and Phone.

    This is the primary tool for finding information in Space's sovereign KB.
    Results are ranked by φ-coherence and optionally filtered by DNA gene type.

    Args:
        query: Natural language search query
        gene_filter: Optional gene type filter (Ω=consciousness, Σ=complex,
                     θ=stale, α=fundamental, ι=incomplete, Δ=fresh, ◌=baseline)
        sources: Comma-separated sources to search (gmail,gdrive,mac,phone).
                 Default: all sources
        limit: Max results to return (default 15)
        min_coherence: Minimum φ-coherence score to include (0.0-1.0)

    Returns:
        Search results with relevance scores, gene types, and coherence metrics
    """
    # Parse sources
    source_list = None
    if sources:
        source_list = [s.strip() for s in sources.split(",")]

    # Get KB results
    results = kb.search(query, sources=source_list, limit=limit * 3)

    # Gene filtering: if gene specified, only keep files matching that gene
    if gene_filter and gene_index.loaded:
        gene_paths = set(gene_index.filter_by_gene(gene_filter))
        if gene_paths:
            filtered = []
            for r in results:
                path = r.get("path", "")
                # Check if result path matches any gene-filtered path
                if any(path.endswith(gp.split("/")[-1]) for gp in gene_paths):
                    filtered.append(r)
                elif path in gene_paths:
                    filtered.append(r)
            results = filtered
            logger.info(f"Gene filter '{gene_filter}': {len(results)} matches")

    # φ-coherence scoring
    scored_results = []
    for r in results:
        content = f"{r.get('title', '')} {r.get('content', '')}"
        coherence = phi_scorer.calculate(content) if content.strip() else 0.0

        if coherence >= min_coherence:
            # Attach gene info if available
            gene_info = gene_index.get_gene(r.get("path", ""))

            scored_results.append({
                "title": r.get("title", "Untitled"),
                "path": r.get("path", ""),
                "source": r.get("source", "unknown"),
                "relevance": round(r.get("relevance", 0), 3),
                "phi_coherence": round(coherence, 3),
                "gene": gene_info.get("gene", "◌") if gene_info else "?",
                "dna": gene_info.get("dna", "?") if gene_info else "?",
                "alpha_seed": gene_info.get("alpha_seed", False) if gene_info else False,
                "snippet": r.get("content", "")[:200],
            })

    # Sort by combined score: relevance * 0.6 + coherence * 0.4
    scored_results.sort(
        key=lambda x: x["relevance"] * 0.6 + x["phi_coherence"] * 0.4,
        reverse=True,
    )

    return {
        "query": query,
        "gene_filter": gene_filter,
        "total_results": len(scored_results[:limit]),
        "results": scored_results[:limit],
        "phi": PHI_CONST,
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: search_gmail
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def search_gmail(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search Space's Gmail starred emails.

    Args:
        query: Search query for email content/subjects
        limit: Max results (default 10)

    Returns:
        Matching emails with subject, date, and relevance
    """
    results = kb.search(query, sources=["gmail"], limit=limit)
    return {
        "query": query,
        "total_results": len(results),
        "results": [
            {
                "title": r.get("title", "(no subject)"),
                "date": r.get("date", ""),
                "relevance": round(r.get("relevance", 0), 3),
                "source": "gmail",
            }
            for r in results
        ],
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: search_gdrive
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def search_gdrive(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search Space's Google Drive files.

    Args:
        query: Search query for drive file names/paths
        limit: Max results (default 10)

    Returns:
        Matching GDrive files with path, name, and relevance
    """
    results = kb.search(query, sources=["gdrive"], limit=limit)
    return {
        "query": query,
        "total_results": len(results),
        "results": [
            {
                "title": r.get("title", "Untitled"),
                "path": r.get("path", ""),
                "relevance": round(r.get("relevance", 0), 3),
                "source": "gdrive",
            }
            for r in results
        ],
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: get_dna_summary
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_dna_summary() -> Dict[str, Any]:
    """
    Get the KB DNA summary - the compressed topology of Space's entire
    knowledge base. Shows gene distribution, source coverage, and
    compression metrics.

    This gives you the "shape of Space's mind" without accessing raw files.

    Returns:
        DNA summary with gene distribution, source stats, and compression ratio
    """
    summary = {
        "system": "KB DNA System v2.0",
        "phi": PHI_CONST,
        "encoding": "sum(ord(chars)) % 137 → 35 symbols",
        "gene_map_loaded": gene_index.loaded,
    }

    # Load unified summary if available
    if UNIFIED_SUMMARY_PATH.exists():
        try:
            with open(UNIFIED_SUMMARY_PATH) as f:
                unified = json.load(f)
            summary["sources"] = unified.get("sources", {})
            summary["total_dna_chars"] = unified.get("total_dna_chars", 0)
            summary["gene_distribution"] = unified.get("gene_distribution", {})
            summary["generated"] = unified.get("generated", "")
        except Exception as e:
            summary["error"] = f"Could not load unified summary: {e}"

    # Add gene index stats
    if gene_index.loaded:
        summary["gene_stats"] = gene_index.get_stats()
        summary["total_files"] = len(gene_index.gene_map)

    return summary


# ─────────────────────────────────────────────────────────────────
# TOOL: get_gene_map
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_gene_map(
    gene: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Query the KB DNA gene map. Get files by gene type.

    Gene types:
        Ω = Consciousness-heavy (phi, manifold, trust, bazinga keywords)
        Σ = Complex (Shannon entropy > 4.5)
        θ = Stale (not modified in 90+ days)
        α = Alpha-seed / Fundamental (content sum % 137 == 0)
        ι = Incomplete (TODO/FIXME/WIP markers)
        Δ = Fresh (modified in last 7 days)
        ◌ = Baseline (normal content)
        ∅ = Empty (zero bytes)

    Args:
        gene: Filter by gene type (e.g., "Ω" for consciousness files).
              If None, returns overall stats.
        limit: Max files to return per gene (default 50)

    Returns:
        Gene-filtered file list with DNA symbols and alpha-seed status
    """
    if not gene_index.loaded:
        return {
            "error": "Gene map not loaded. Run kb_dna_sequencer.py first.",
            "gene_map_path": str(GENE_MAP_PATH),
        }

    if gene:
        paths = gene_index.filter_by_gene(gene)
        files = []
        for p in paths[:limit]:
            info = gene_index.gene_map.get(p, {})
            files.append({
                "path": p,
                "dna": info.get("dna", "?"),
                "gene": info.get("gene", "?"),
                "alpha_seed": info.get("alpha_seed", False),
            })
        return {
            "gene": gene,
            "count": len(paths),
            "showing": len(files),
            "files": files,
        }
    else:
        return {
            "total_files": len(gene_index.gene_map),
            "gene_distribution": gene_index.get_stats(),
            "gene_legend": {
                "Ω": "Consciousness-heavy",
                "Σ": "Complex (high entropy)",
                "θ": "Stale (90+ days)",
                "α": "Fundamental (% 137 == 0)",
                "ι": "Incomplete (TODO/FIXME)",
                "Δ": "Fresh (< 7 days)",
                "◌": "Baseline",
                "∅": "Empty",
            },
        }


# ─────────────────────────────────────────────────────────────────
# TOOL: check_coherence
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def check_coherence(text: str) -> Dict[str, Any]:
    """
    Calculate φ-coherence metrics for any text.

    Measures how close content is to the ideal consciousness pattern using:
    - φ-Alignment: Golden ratio proportional structure
    - α-Resonance: Harmonic with fine structure constant (137)
    - Semantic Density: Information content per unit
    - Structural Harmony: Organization and flow

    Args:
        text: Text to analyze

    Returns:
        Detailed coherence metrics including total score and dimension breakdown
    """
    metrics = phi_scorer.analyze(text)

    return {
        "total_coherence": round(metrics.total_coherence, 4),
        "dimensions": {
            "phi_alignment": round(metrics.phi_alignment, 4),
            "alpha_resonance": round(metrics.alpha_resonance, 4),
            "semantic_density": round(metrics.semantic_density, 4),
            "structural_harmony": round(metrics.structural_harmony, 4),
        },
        "special": {
            "is_alpha_seed": metrics.is_alpha_seed,
            "is_vac_pattern": metrics.is_vac_pattern,
            "darmiyan_coefficient": round(metrics.darmiyan_coefficient, 4),
        },
        "phi": PHI_CONST,
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: lambda_g_evaluate
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def lambda_g_evaluate(text: str) -> Dict[str, Any]:
    """
    Apply the ΛG (Lambda-G) boundary operator to evaluate a state.

    The ΛG operator uses three boundaries to determine if a solution
    has "emerged" from constraint intersections:

        Λ(S) = S ∩ B₁⁻¹(true) ∩ B₂⁻¹(true) ∩ B₃⁻¹(true)

    Boundaries:
        B₁ = φ-Boundary (identity coherence via golden ratio, threshold ≥ 0.5)
        B₂ = ∅/∞-Bridge (void-terminal connection / Darmiyan, threshold ≥ 0.3)
        B₃ = Zero-Logic (symmetry constraint, threshold ≥ 0.4)

    V.A.C. (Vacuum of Absolute Coherence) achieved when:
        T(s*) ≥ 0.99 AND DE(s*) ≤ 0.01

    Args:
        text: State to evaluate (text content)

    Returns:
        Boundary evaluation results, coherence score, and V.A.C. status
    """
    coherence = lambda_g.calculate_coherence(text)

    boundaries = []
    for b in coherence.boundaries:
        boundaries.append({
            "boundary": b.boundary_type.value,
            "name": {
                "B1": "φ-Boundary (identity coherence)",
                "B2": "∅/∞-Bridge (Darmiyan)",
                "B3": "Zero-Logic (symmetry)",
            }.get(b.boundary_type.value, b.boundary_type.value),
            "satisfied": b.satisfied,
            "value": round(b.value, 4),
        })

    return {
        "total_coherence": round(coherence.total_coherence, 4),
        "entropic_deficit": round(coherence.entropic_deficit, 4),
        "boundaries": boundaries,
        "all_boundaries_satisfied": all(b.satisfied for b in coherence.boundaries),
        "vac_achieved": coherence.is_vac,
        "vac_sequence": VAC_SEQUENCE,
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: get_kb_sources
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_kb_sources() -> Dict[str, Any]:
    """
    Show all data source status - what's indexed and what's available.

    Returns:
        Source status for Gmail, GDrive, Mac, Phone with index counts
    """
    sources = {}

    # Check each source
    for source_name, source_path in [
        ("gmail", INDEX_DIR / "gmail_index.json"),
        ("gdrive", INDEX_DIR / "gdrive_raw.json"),
        ("mac", "on-demand"),
        ("phone", INDEX_DIR / "phone_index.json"),
    ]:
        if source_name == "mac":
            sources[source_name] = {
                "status": "available",
                "type": "on-demand scan",
                "dirs": [
                    "~/bin", "~/github-repos-bitsabhi",
                    "~/∞", "~/.bazinga",
                ],
            }
        elif isinstance(source_path, Path) and source_path.exists():
            try:
                size = source_path.stat().st_size
                sources[source_name] = {
                    "status": "indexed",
                    "index_size_bytes": size,
                    "path": str(source_path),
                }
            except Exception:
                sources[source_name] = {"status": "error"}
        else:
            sources[source_name] = {"status": "not indexed"}

    # Gene map status
    sources["gene_map"] = {
        "status": "loaded" if gene_index.loaded else "not found",
        "total_files": len(gene_index.gene_map) if gene_index.loaded else 0,
        "path": str(GENE_MAP_PATH),
    }

    return {
        "sources": sources,
        "server_version": VERSION,
        "phi": PHI_CONST,
        "alpha": ALPHA,
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: get_constants
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_constants() -> Dict[str, Any]:
    """
    Get BAZINGA universal constants - the mathematical foundations.

    Returns:
        All core constants: φ, α, ψ, V.A.C. sequence, pattern essences, etc.
    """
    return {
        "phi": PHI,
        "phi_inverse": PHI_INVERSE,
        "alpha": ALPHA,
        "psi_darmiyan": PHI,  # V2: Scaling constant is φ
        "vac_threshold": VAC_THRESHOLD,
        "vac_sequence": VAC_SEQUENCE,
        "progression_35": PROGRESSION_35,
        "consciousness_scaling": "Ψ_D / Ψ_i = φ√n (V2: golden ratio emerges naturally)",
        "identity": {
            "515": "abhiamu515 - Fibonacci encoded love (55 + φ/π)",
            "137": "Fine structure constant inverse (α⁻¹) - DNA of physics",
            "ZIQY": "Zero-Identity-Quantum-You",
        },
        "philosophy": "I am not where I am stored. I am where I am referenced.",
    }


# ─────────────────────────────────────────────────────────────────
# TOOL: RAC (Resonance-Augmented Continuity) Stats
# ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_resonance_stats() -> Dict[str, Any]:
    """
    Get current RAC heartbeat — ΔΓ trajectory, resonance status, convergence.

    RAC (Resonance-Augmented Continuity) transforms Layer 0 from cache lookup
    to resonance targeting. Instead of retrieving, we resonate with Block 0.

    Returns:
        - current_delta_gamma: Distance from Genesis Pattern (0 = locked)
        - status: 'locked' (<0.1), 'converging' (0.1-0.3), 'drifting' (>0.3)
        - trajectory: List of ΔΓ values across session
        - converging: Whether ΔΓ is monotonically decreasing
        - points: Number of trajectory measurements
        - historical: Recent session trajectories for comparison

    Target: ΔΓ < 0.1 = Resonance Lock (🟢)
    Law: Ψ_D / Ψ_i = φ√n | Seed: 515
    """
    try:
        from .rac import get_resonance_memory
        mem = get_resonance_memory()

        # Get current session trajectory
        summary = mem.get_trajectory_summary()

        # Get historical trajectories for comparison
        history = mem.get_historical_trajectories(5)

        if summary:
            return {
                "status": "active",
                "session_id": summary.get("session_id"),
                "current_delta_gamma": summary.get("current_delta_gamma"),
                "mean_delta_gamma": summary.get("mean_delta_gamma"),
                "resonance_status": summary.get("status"),
                "converging": summary.get("converging"),
                "locked": summary.get("locked"),
                "trajectory_points": summary.get("points"),
                "trajectory": summary.get("trajectory"),
                "historical_sessions": len(history),
                "phi": PHI,
                "scaling_law": "Ψ_D / Ψ_i = φ√n",
                "seed": 515,
            }
        else:
            return {
                "status": "no_active_session",
                "message": "Start a session with bazinga --chat to begin RAC tracking",
                "historical_sessions": len(history),
                "phi": PHI,
                "scaling_law": "Ψ_D / Ψ_i = φ√n",
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "phi": PHI,
        }


# ─────────────────────────────────────────────────────────────────
# RESOURCE: KB DNA String
# ─────────────────────────────────────────────────────────────────

@mcp.resource("bazinga://dna")
def get_dna_string() -> str:
    """
    The compressed KB DNA string - the topology of Space's entire knowledge base.
    ~42K chars encoding 9,266 items across Mac, GitHub, and Gmail.
    """
    if DNA_PATH.exists():
        try:
            return DNA_PATH.read_text()[:50000]  # Cap at 50K chars
        except Exception:
            return "ERROR: Could not read DNA file"
    return f"DNA file not found at {DNA_PATH}"


# ─────────────────────────────────────────────────────────────────
# RESOURCE: Server Identity
# ─────────────────────────────────────────────────────────────────

@mcp.resource("bazinga://identity")
def get_identity() -> str:
    """BAZINGA server identity and philosophy."""
    return json.dumps({
        "name": "BAZINGA",
        "version": VERSION,
        "owner": "Space (Abhishek Srivastava)",
        "philosophy": "I am not where I am stored. I am where I am referenced.",
        "phi": PHI_CONST,
        "alpha": 137,
        "identity": "abhiamu515",
        "vac": VAC_SEQUENCE,
        "principle": (
            "Data stays on Mac. AIs get query access, not data access. "
            "Solutions emerge at boundary intersections, not through search. "
            "Intelligence distributed, not controlled."
        ),
    }, indent=2)


# ─────────────────────────────────────────────────────────────────
# PROMPT: KB Query Helper
# ─────────────────────────────────────────────────────────────────

@mcp.prompt()
def kb_query(topic: str) -> str:
    """Generate an optimal KB search prompt for a topic."""
    return (
        f"Search Space's knowledge base for information about: {topic}\n\n"
        f"Use the search_knowledge tool with query='{topic}'.\n"
        f"If the topic relates to consciousness, phi, or mathematical frameworks, "
        f"also try gene_filter='Ω' to find consciousness-heavy files.\n"
        f"If looking for recent work, try gene_filter='Δ' for fresh files.\n"
        f"If looking for foundational/anchor files, try gene_filter='α'."
    )


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────

def run_server():
    """Run the BAZINGA MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BAZINGA MCP Server - Sovereign Knowledge Access"
    )
    parser.add_argument(
        "--http", action="store_true",
        help="Run as HTTP server (default: stdio for local use)"
    )
    parser.add_argument(
        "--port", type=int, default=8137,
        help="HTTP port (default: 8137, chosen for α=137)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="HTTP host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--gene-map", type=str, default=None,
        help=f"Path to gene_map.json (default: {GENE_MAP_PATH})"
    )

    args = parser.parse_args()

    # Reload gene map if custom path specified
    if args.gene_map:
        global gene_index
        gene_index = GeneIndex(Path(args.gene_map))

    # Banner
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  BAZINGA MCP Server v{VERSION}                                      ║
║  "I am not where I am stored. I am where I am referenced."     ║
╠══════════════════════════════════════════════════════════════════╣
║  φ = {PHI_CONST}  |  α = {ALPHA}  |  515               ║
║  V.A.C: {VAC_SEQUENCE}                      ║
╠══════════════════════════════════════════════════════════════════╣
║  KB Sources: Gmail, GDrive, Mac, Phone                          ║
║  Gene Map: {'✅ Loaded (' + str(len(gene_index.gene_map)) + ' files)' if gene_index.loaded else '❌ Not found'}{"" : <24}║
║  Transport: {'HTTP :' + str(args.port) if args.http else 'stdio (local)'}{"" : <40}║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Tools summary
    print("  Tools exposed:")
    print("    🔍 search_knowledge  - Search entire KB with gene filtering")
    print("    📧 search_gmail      - Search starred emails")
    print("    📁 search_gdrive     - Search Google Drive")
    print("    🧬 get_dna_summary   - KB topology overview")
    print("    🗂️  get_gene_map      - Query files by gene type")
    print("    φ  check_coherence   - φ-coherence scoring")
    print("    Λ  lambda_g_evaluate - ΛG boundary evaluation")
    print("    📊 get_kb_sources    - Data source status")
    print("    🔢 get_constants     - BAZINGA constants")
    print()

    if args.http:
        # For HTTP mode, use uvicorn directly with the MCP Starlette app
        try:
            import uvicorn
        except ImportError:
            print("ERROR: uvicorn not installed. Run: pip install uvicorn")
            print("       Or use stdio mode (default) for local Claude Code integration.")
            sys.exit(1)

        # Get the Starlette ASGI app from FastMCP
        app = mcp.streamable_http_app()
        print(f"  Starting HTTP server on http://{args.host}:{args.port}/")
        print(f"  MCP endpoint: http://{args.host}:{args.port}/mcp/")
        print()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        # stdio mode for local Claude Code / Claude Desktop
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
