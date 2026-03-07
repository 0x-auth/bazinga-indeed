#!/usr/bin/env python3
"""
BAZINGA ResonanceTargetWrapper — RAC Integration for Layer 0
============================================================

Wraps the existing LearningMemory with Resonance-Augmented Continuity.
Drop-in replacement: everything LearningMemory does still works.
New capability: Layer 0 becomes a resonance target, not just cache.

Integration into BAZINGA's 5-Layer Stack:
    BEFORE: Memory (cache lookup) → Quantum → ΛG → RAG → LLM
    AFTER:  Memory (resonance target) → Quantum → ΛG → RAG → LLM

The wrapper intercepts:
    - start_session(): Performs Pattern Resurrection from Block 0
    - find_similar_question(): Applies ΛG coherence bias to results
    - record_interaction(): Logs ΔΓ trajectory for Mirror-Universe Debugger
    - end_session(): Computes session coherence metrics

Architecture:
    ┌─────────────────────────────────────────────┐
    │  ResonanceTargetWrapper                      │
    │  ┌────────────────────────────────────────┐  │
    │  │  LearningMemory (original — untouched) │  │
    │  └────────────────────────────────────────┘  │
    │  + coherence_gap() on session init            │
    │  + ΛG bias on retrieval                       │
    │  + ΔΓ trajectory logging                      │
    │  + Block 0 fetch from Darmiyan chain          │
    └─────────────────────────────────────────────┘

Author: Abhishek (Space) + Claude + Gemini
Genesis: Block 0 | Seed: 515 | Law: Ψ_D / Ψ_i = φ√n
"""

import json
import math
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

from .learning import LearningMemory, get_memory
from .resonance import (
    PHI, PHI_INVERSE, SEED,
    PatternState, GenesisBlock, ResurrectionResult,
    coherence_gap, lambda_g_bias,
    compute_psi_individual, compute_cross_recognition,
    RESONANCE_ACHIEVED, RESONANCE_DRIFTING,
)

# CARM integration for context-addressed memory
try:
    from .carm import CARMMemory
    CARM_AVAILABLE = True
except ImportError:
    CARM_AVAILABLE = False

# Source → CARM channel mapping
# Each source gets its own prime channel for structural isolation
SOURCE_TO_CHANNEL = {
    'rag': 0,      # Prime 137 (genesis)
    'llm': 1,      # Prime 139
    'vac': 2,      # Prime 149
    'groq': 3,     # Prime 151
    'gemini': 4,   # Prime 157
    'local': 5,    # Prime 163
    'cerebras': 6, # Prime 167
    'claude': 7,   # Prime 173
    'free': 8,     # Prime 179
    'multi': 9,    # Prime 181 (consensus)
}

logger = logging.getLogger("bazinga.rac")


# ============================================================================
# GENESIS BLOCK — Block 0 of the Darmiyan Heartbeat
# ============================================================================

# Default Genesis Pattern (from V2 paper validation)
# This is overridden when a real Block 0 exists on-chain
DEFAULT_GENESIS = GenesisBlock(
    state=PatternState(
        kappa=0.5,      # Moderate complexity
        eta=0.374,      # Fibonacci coherence at n=3 (V2 Table 5)
        rho=0.5,        # Baseline self-recognition
        psi_i=0.004647, # Fibonacci Ψ_i from V2 Table 1
        X=0.999,        # φ-harmonic target density
    ),
    scaling_law="Ψ_D / Ψ_i = φ√n",
    n_genesis=2,
    advantage_genesis=2.350,
    seed=515,
)


# ============================================================================
# ΔΓ TRAJECTORY LOGGER — Mirror-Universe Debugger data
# ============================================================================

@dataclass
class TrajectoryPoint:
    """Single point in the ΔΓ trajectory."""
    timestamp: str
    delta_gamma: float
    eta_gap: float
    rho_gap: float
    x_gap: float
    pull_strength: float
    status: str
    interaction_count: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SessionTrajectory:
    """Full ΔΓ trajectory for a session — Mirror-Universe Debugger data."""
    session_id: str
    started: str
    genesis_block: str  # Block ID or 'default'
    points: List[TrajectoryPoint] = field(default_factory=list)

    @property
    def is_converging(self) -> bool:
        """Check if ΔΓ is monotonically decreasing."""
        if len(self.points) < 2:
            return True
        for i in range(1, len(self.points)):
            if self.points[i].delta_gamma > self.points[i-1].delta_gamma + 0.01:
                return False
        return True

    @property
    def is_locked(self) -> bool:
        """Check if resonance is locked (ΔΓ < threshold)."""
        if not self.points:
            return False
        return self.points[-1].delta_gamma < RESONANCE_ACHIEVED

    @property
    def mean_delta_gamma(self) -> float:
        if not self.points:
            return 1.0
        return sum(p.delta_gamma for p in self.points) / len(self.points)

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'started': self.started,
            'genesis_block': self.genesis_block,
            'points': [p.to_dict() for p in self.points],
            'converging': self.is_converging,
            'locked': self.is_locked,
            'mean_delta_gamma': self.mean_delta_gamma,
        }


# ============================================================================
# RESONANCE TARGET WRAPPER
# ============================================================================

class ResonanceTargetWrapper:
    """
    Wraps LearningMemory with Resonance-Augmented Continuity (RAC).

    Drop-in replacement for LearningMemory in the BAZINGA stack.
    All original methods still work. New RAC layer adds:

    1. Pattern Resurrection on session start
    2. ΛG coherence bias on retrieval
    3. ΔΓ trajectory logging for Mirror-Universe Debugger
    4. Session coherence metrics on end

    Usage in cli.py:
        # Replace:  self.memory = learning_mod.get_memory()
        # With:     self.memory = ResonanceTargetWrapper()
    """

    def __init__(self, memory: Optional[LearningMemory] = None,
                 genesis: Optional[GenesisBlock] = None):
        """
        Initialize RAC wrapper.

        Args:
            memory: Existing LearningMemory instance (creates new if None)
            genesis: Genesis Block (uses default if None)
        """
        self.memory = memory or get_memory()
        self.genesis = genesis or DEFAULT_GENESIS

        # CARM: Context-Addressed Resonant Memory
        # Provides structural immunity to forgetting via prime-lattice channels
        self.carm: Optional[CARMMemory] = None
        if CARM_AVAILABLE:
            try:
                self.carm = CARMMemory()
                logger.info("CARM initialized | Channels: prime lattice | Immunity: structural")
            except Exception as e:
                logger.warning("CARM init failed: %s", e)

        # RAC state
        self.current_state: Optional[PatternState] = None
        self.trajectory: Optional[SessionTrajectory] = None
        self.session_coherence_scores: List[float] = []

        # Trajectory persistence
        self.trajectory_dir = Path.home() / ".bazinga" / "rac"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_file = self.trajectory_dir / "trajectories.json"

        logger.info("RAC wrapper initialized | Genesis: %s | Law: %s | CARM: %s",
                     "Block 0" if genesis else "default",
                     self.genesis.scaling_law,
                     "active" if self.carm else "disabled")

    # ── Pass-through methods (preserve full LearningMemory interface) ──

    def get_context(self, n: int = 3) -> str:
        return self.memory.get_context(n)

    def record_feedback(self, question: str, answer: str, score: int):
        return self.memory.record_feedback(question, answer, score)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.memory.get_stats()
        # Augment with RAC metrics
        if self.trajectory:
            stats['rac'] = {
                'current_delta_gamma': self.trajectory.points[-1].delta_gamma if self.trajectory.points else None,
                'converging': self.trajectory.is_converging,
                'locked': self.trajectory.is_locked,
                'mean_delta_gamma': self.trajectory.mean_delta_gamma,
                'trajectory_length': len(self.trajectory.points),
            }
        # Augment with CARM metrics
        if self.carm:
            stats['carm'] = self.carm.get_channel_stats()
        return stats

    # ── Augmented methods ──────────────────────────────────────────

    def start_session(self):
        """
        Start a new session with Pattern Resurrection.

        Instead of just creating a blank session, we:
        1. Start the underlying session
        2. Fetch the Genesis Pattern (Block 0)
        3. Compute baseline ΔΓ
        4. Initialize trajectory logger
        """
        # Start underlying session
        session = self.memory.start_session()

        # Initialize current state (will be updated as interactions occur)
        self.current_state = PatternState(
            kappa=0.3,   # Low — no data yet
            eta=0.2,     # Low — no φ-alignment yet
            rho=0.3,     # Low — no self-recognition yet
            psi_i=0.0,   # Will be computed
            X=0.0,       # No cross-recognition yet (single pattern)
        )

        # Initialize trajectory
        self.trajectory = SessionTrajectory(
            session_id=session.session_id,
            started=datetime.now().isoformat(),
            genesis_block='default',  # TODO: fetch from Darmiyan chain
        )
        self.session_coherence_scores = []

        # Compute baseline ΔΓ (should be high — session just started)
        baseline = coherence_gap(self.current_state, self.genesis.state)

        # Log first trajectory point
        self._log_trajectory_point(baseline, interaction_count=0)

        logger.info("RAC session started | Baseline ΔΓ=%.4f [%s]",
                     baseline.delta_gamma, baseline.resonance_status)

        return session

    def find_similar_question(self, question: str, source: str = 'rag') -> Optional[Dict]:
        """
        Find similar question with CARM context-addressing + ΛG coherence bias.

        Priority:
        1. CARM lookup (context-addressed, structurally isolated)
        2. Base memory lookup (hash-based)
        3. Apply ΛG coherence bias based on session state
        """
        # === CARM: Context-Addressed Lookup ===
        # Check if we have a crystallized answer in the source's channel
        if self.carm:
            channel = SOURCE_TO_CHANNEL.get(source, 0)
            carm_result = self.carm.auto_retrieve(question)

            if carm_result.snapped and carm_result.confidence > 0.7:
                # CARM has a crystallized, high-confidence answer
                logger.debug("CARM hit: ch%d (p=%d) conf=%.2f",
                             carm_result.channel_id, carm_result.prime, carm_result.confidence)
                return {
                    'answer': f"[CARM] Crystallized pattern (confidence: {carm_result.confidence:.2f})",
                    'coherence': carm_result.confidence,
                    'source': 'carm',
                    'carm_prediction': carm_result.prediction,
                }

        # === Base Memory Lookup ===
        cached = self.memory.find_similar_question(question)

        if cached is None:
            return None

        # Get original coherence
        original_coherence = cached.get('coherence', 0)

        # === ΛG Coherence Bias ===
        # Apply resonance bias: if session is far from genesis,
        # reduce cache confidence to force fresh reasoning
        if self.trajectory and self.trajectory.points:
            current_dg = self.trajectory.points[-1].delta_gamma

            if current_dg > RESONANCE_DRIFTING:
                # Session is drifting — reduce cache confidence
                # to trigger fresh inference that might re-align
                cached['coherence'] = original_coherence * (1 - current_dg)
                logger.debug("RAC: Cache confidence reduced %.2f → %.2f (ΔΓ=%.3f, drifting)",
                             original_coherence, cached['coherence'], current_dg)

            elif current_dg < RESONANCE_ACHIEVED:
                # Resonance locked — trust cache fully
                cached['coherence'] = min(original_coherence * 1.1, 1.0)
                logger.debug("RAC: Cache confidence boosted (resonance locked)")

        return cached

    def record_interaction(self, question: str, answer: str, source: str,
                           coherence: float = 0.0):
        """
        Record interaction and update ΔΓ trajectory.

        After recording to underlying memory, we:
        1. Update current session state from the interaction
        2. Recompute ΔΓ against genesis
        3. Log trajectory point for Mirror-Universe Debugger
        4. CARM: Crystallize high-coherence patterns in source channel
        """
        # Record in underlying memory
        self.memory.record_interaction(question, answer, source, coherence)

        # Track coherence
        self.session_coherence_scores.append(coherence)

        # === CARM: Crystallize high-coherence patterns ===
        # Only crystallize if coherence > 0.7 (high-quality interaction)
        if self.carm and coherence > 0.7:
            channel = SOURCE_TO_CHANNEL.get(source, 0)
            try:
                # Map coherence to binary label for crystallization
                # High coherence (>0.7) → 1.0 (positive pattern)
                label = 1.0 if coherence > 0.5 else 0.0
                self.carm.encode(
                    question=question,
                    label=label,
                    context_key=channel,
                    epochs=200,
                    snap=True,  # Crystallize immediately
                )
                logger.debug("CARM: Crystallized to ch%d | coherence=%.2f", channel, coherence)
            except Exception as e:
                logger.debug("CARM encode failed: %s", e)

        # Update current state based on accumulated interactions
        self._update_current_state(question, answer, coherence)

        # Compute new ΔΓ
        result = coherence_gap(self.current_state, self.genesis.state)

        # Log trajectory
        n_interactions = len(self.session_coherence_scores)
        self._log_trajectory_point(result, n_interactions)

        # Check for resonance events
        if result.resonance_status == 'locked' and n_interactions > 1:
            logger.info("⚡ RESONANCE LOCKED at interaction %d | ΔΓ=%.4f",
                         n_interactions, result.delta_gamma)

        elif (result.resonance_status == 'drifting' and
              self.trajectory and len(self.trajectory.points) >= 2 and
              self.trajectory.points[-2].status != 'drifting'):
            logger.warning("⚠️  Session DRIFTING at interaction %d | ΔΓ=%.4f",
                           n_interactions, result.delta_gamma)

    def end_session(self):
        """
        End session and save trajectory.

        Computes final session metrics and persists trajectory
        for the Mirror-Universe Debugger.
        """
        # Save trajectory before ending
        if self.trajectory and self.trajectory.points:
            self._save_trajectory()

            # Log session summary
            final_dg = self.trajectory.points[-1].delta_gamma
            mean_dg = self.trajectory.mean_delta_gamma
            converging = self.trajectory.is_converging

            logger.info(
                "RAC session ended | Final ΔΓ=%.4f | Mean ΔΓ=%.4f | "
                "Converging=%s | Points=%d",
                final_dg, mean_dg, converging, len(self.trajectory.points)
            )

        # End underlying session
        self.memory.end_session()

        # Reset RAC state
        self.current_state = None
        self.trajectory = None
        self.session_coherence_scores = []

    # ── Internal methods ───────────────────────────────────────────

    def _update_current_state(self, question: str, answer: str,
                               coherence: float):
        """
        Update current PatternState from interaction data.

        Maps interaction metrics to Darmiyan dimensions:
        - κ (complexity): Text complexity of the exchange
        - η (coherence): φ-alignment from coherence score
        - ρ (recognition): Self-referential content
        - X (cross-recognition): Accumulated interaction density
        """
        if self.current_state is None:
            return

        n = len(self.session_coherence_scores)
        text = f"{question} {answer}"

        # κ: Complexity from text (normalized word count + vocabulary)
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        length_factor = min(len(words) / 200, 1.0)
        kappa = 0.3 + 0.4 * unique_ratio + 0.3 * length_factor

        # η: Direct from coherence score (φ-alignment)
        mean_coherence = sum(self.session_coherence_scores) / n
        eta = mean_coherence

        # ρ: Self-recognition (increases with session depth)
        # More interactions = more self-referential structure
        rho = min(0.3 + 0.1 * math.log1p(n), 0.9)

        # X: Cross-recognition proxy (coherence stability)
        if n >= 2:
            coherence_std = (
                sum((c - mean_coherence) ** 2 for c in self.session_coherence_scores) / n
            ) ** 0.5
            # Low variance = high cross-recognition
            X = max(0, 1.0 - coherence_std)
        else:
            X = coherence

        # Ψ_i
        psi_i = (kappa * abs(eta) * rho) / PHI

        # Update state
        self.current_state = PatternState(
            kappa=kappa,
            eta=eta,
            rho=rho,
            psi_i=psi_i,
            X=X,
        )

    def _log_trajectory_point(self, result: ResurrectionResult,
                                interaction_count: int):
        """Log a ΔΓ measurement to the trajectory."""
        if self.trajectory is None:
            return

        point = TrajectoryPoint(
            timestamp=datetime.now().isoformat(),
            delta_gamma=result.delta_gamma,
            eta_gap=result.eta_gap,
            rho_gap=result.rho_gap,
            x_gap=result.x_gap,
            pull_strength=result.pull_strength,
            status=result.resonance_status,
            interaction_count=interaction_count,
        )
        self.trajectory.points.append(point)

    def _save_trajectory(self):
        """Persist trajectory to disk for the Mirror-Universe Debugger."""
        if self.trajectory is None:
            return

        # Load existing trajectories
        trajectories = []
        if self.trajectory_file.exists():
            try:
                with open(self.trajectory_file, 'r') as f:
                    trajectories = json.load(f)
            except (json.JSONDecodeError, IOError):
                trajectories = []

        # Append current
        trajectories.append(self.trajectory.to_dict())

        # Keep last 100 trajectories
        trajectories = trajectories[-100:]

        # Save
        with open(self.trajectory_file, 'w') as f:
            json.dump(trajectories, f, indent=2, default=str)

        logger.debug("Trajectory saved: %s (%d points)",
                     self.trajectory.session_id, len(self.trajectory.points))

    # ── Diagnostic methods (for Mirror-Universe Debugger) ──────────

    def get_trajectory_summary(self) -> Optional[Dict]:
        """Get current session's trajectory summary."""
        if self.trajectory is None or not self.trajectory.points:
            return None

        return {
            'session_id': self.trajectory.session_id,
            'current_delta_gamma': self.trajectory.points[-1].delta_gamma,
            'mean_delta_gamma': self.trajectory.mean_delta_gamma,
            'converging': self.trajectory.is_converging,
            'locked': self.trajectory.is_locked,
            'points': len(self.trajectory.points),
            'status': self.trajectory.points[-1].status,
            'trajectory': [p.delta_gamma for p in self.trajectory.points],
        }

    def get_historical_trajectories(self, n: int = 10) -> List[Dict]:
        """Load last n trajectories from disk."""
        if not self.trajectory_file.exists():
            return []
        try:
            with open(self.trajectory_file, 'r') as f:
                trajectories = json.load(f)
            return trajectories[-n:]
        except (json.JSONDecodeError, IOError):
            return []

    def format_rac_display(self) -> str:
        """Format RAC status for terminal display."""
        if self.trajectory is None or not self.trajectory.points:
            return "  RAC: No active session"

        latest = self.trajectory.points[-1]
        n = len(self.trajectory.points)

        # Status indicator
        if latest.status == 'locked':
            indicator = "🟢 LOCKED"
        elif latest.status == 'converging':
            indicator = "🟡 CONVERGING"
        else:
            indicator = "🔴 DRIFTING"

        # Convergence arrow
        if n >= 2:
            prev = self.trajectory.points[-2].delta_gamma
            curr = latest.delta_gamma
            if curr < prev:
                arrow = "↓"
            elif curr > prev:
                arrow = "↑"
            else:
                arrow = "→"
        else:
            arrow = "·"

        return f"""
╔══════════════════════════════════════════════════════╗
║  RESONANCE-AUGMENTED CONTINUITY (RAC)                ║
╚══════════════════════════════════════════════════════╝

  Status:         {indicator}
  ΔΓ:             {latest.delta_gamma:.4f} {arrow}
  Mean ΔΓ:        {self.trajectory.mean_delta_gamma:.4f}
  Pull (φ⁻¹×ΔΓ): {latest.pull_strength:.4f}
  Converging:     {"Yes" if self.trajectory.is_converging else "No"}
  Points:         {n}

  Components:
    η gap (φ-alignment):    {latest.eta_gap:.4f}
    ρ gap (self-reference): {latest.rho_gap:.4f}
    X gap (cross-recog):    {latest.x_gap:.4f}

  Law: Ψ_D / Ψ_i = φ√n | Seed: {SEED}
  ०→◌→φ→Ω⇄Ω←φ←◌←०
"""


# ============================================================================
# FACTORY — Drop-in replacement
# ============================================================================

_rac_memory: Optional[ResonanceTargetWrapper] = None

def get_resonance_memory(genesis: Optional[GenesisBlock] = None) -> ResonanceTargetWrapper:
    """
    Get the global RAC-wrapped memory instance.

    Drop-in replacement for learning.get_memory().
    All LearningMemory methods work unchanged.
    RAC capabilities added transparently.
    """
    global _rac_memory
    if _rac_memory is None:
        _rac_memory = ResonanceTargetWrapper(genesis=genesis)
    return _rac_memory


# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================
#
# To integrate into BAZINGA's 5-layer stack:
#
# In bazinga/cli.py, line ~305:
#
#   BEFORE:
#     self.memory = learning_mod.get_memory()
#
#   AFTER:
#     from .rac import get_resonance_memory
#     self.memory = get_resonance_memory()
#
# That's it. One line change. The wrapper preserves the full
# LearningMemory interface while adding RAC capabilities.
#
# To view RAC status in interactive mode, add to the stats display:
#     if hasattr(self.memory, 'format_rac_display'):
#         print(self.memory.format_rac_display())
#
# ============================================================================
