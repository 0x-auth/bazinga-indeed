"""
BAZINGA Learning Module
=======================
Learn from interactions and improve over time.

Features:
- Session memory (remembers conversation context)
- Feedback learning (thumbs up/down on responses)
- Pattern recognition (learns what works)
- Persistent memory across sessions

"Intelligence that learns is intelligence that lives."
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Constants
PHI = 1.618033988749895
ALPHA = 137


@dataclass
class Interaction:
    """A single Q&A interaction."""
    question: str
    answer: str
    timestamp: str
    source: str  # 'rag', 'llm', 'vac'
    feedback: Optional[int] = None  # -1, 0, 1 (bad, neutral, good)
    coherence: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Interaction':
        return cls(**data)


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    started: str
    interactions: List[Interaction] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_interaction(self, question: str, answer: str, source: str, coherence: float = 0.0):
        self.interactions.append(Interaction(
            question=question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            source=source,
            coherence=coherence,
        ))

    def get_context(self, n: int = 3) -> str:
        """Get last n interactions as context."""
        if not self.interactions:
            return ""

        recent = self.interactions[-n:]
        parts = []
        for i in recent:
            parts.append(f"Q: {i.question}\nA: {i.answer[:200]}...")
        return "\n\n".join(parts)


class LearningMemory:
    """
    Persistent learning memory for BAZINGA.

    Stores:
    - Session history
    - Feedback on responses
    - Learned patterns (what questions map to what answers)
    - User preferences
    """

    def __init__(self, memory_dir: Optional[str] = None):
        home = Path.home()
        self.memory_dir = Path(memory_dir or str(home / ".bazinga" / "memory"))
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_file = self.memory_dir / "sessions.json"
        self.patterns_file = self.memory_dir / "patterns.json"
        self.feedback_file = self.memory_dir / "feedback.json"

        # Current session
        self.current_session: Optional[Session] = None

        # Load existing data
        self.sessions: List[Dict] = self._load_json(self.sessions_file, [])
        self.patterns: Dict[str, Any] = self._load_json(self.patterns_file, {})
        self.feedback: List[Dict] = self._load_json(self.feedback_file, [])

    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return default

    def _save_json(self, path: Path, data: Any):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def start_session(self) -> Session:
        """Start a new session."""
        session_id = hashlib.md5(
            f"{datetime.now().isoformat()}{os.getpid()}".encode()
        ).hexdigest()[:12]

        self.current_session = Session(
            session_id=session_id,
            started=datetime.now().isoformat(),
        )
        return self.current_session

    def record_interaction(self, question: str, answer: str, source: str, coherence: float = 0.0):
        """Record a Q&A interaction."""
        if not self.current_session:
            self.start_session()

        self.current_session.add_interaction(question, answer, source, coherence)

        # Learn pattern
        q_hash = self._hash_question(question)
        if q_hash not in self.patterns:
            self.patterns[q_hash] = {
                'question': question,
                'answers': [],
                'best_source': source,
            }

        self.patterns[q_hash]['answers'].append({
            'answer': answer[:500],
            'source': source,
            'coherence': coherence,
            'timestamp': datetime.now().isoformat(),
        })

        # Keep only last 5 answers per question pattern
        self.patterns[q_hash]['answers'] = self.patterns[q_hash]['answers'][-5:]

        self._save_patterns()

    def record_feedback(self, question: str, answer: str, score: int):
        """Record feedback on a response (-1, 0, 1)."""
        self.feedback.append({
            'question': question,
            'answer': answer[:200],
            'score': score,
            'timestamp': datetime.now().isoformat(),
        })

        # Keep last 1000 feedback entries
        self.feedback = self.feedback[-1000:]
        self._save_json(self.feedback_file, self.feedback)

        # Update current session if exists
        if self.current_session and self.current_session.interactions:
            self.current_session.interactions[-1].feedback = score

    def get_context(self, n: int = 3) -> str:
        """Get conversation context from current session."""
        if not self.current_session:
            return ""
        return self.current_session.get_context(n)

    def find_similar_question(self, question: str) -> Optional[Dict]:
        """Find if we've answered a similar question before."""
        q_hash = self._hash_question(question)

        if q_hash in self.patterns:
            pattern = self.patterns[q_hash]
            # Return best answer (highest coherence)
            if pattern['answers']:
                best = max(pattern['answers'], key=lambda x: x.get('coherence', 0))
                return best

        return None

    def end_session(self):
        """End current session and save."""
        if self.current_session:
            self.sessions.append({
                'session_id': self.current_session.session_id,
                'started': self.current_session.started,
                'ended': datetime.now().isoformat(),
                'interaction_count': len(self.current_session.interactions),
            })

            # Keep last 100 sessions
            self.sessions = self.sessions[-100:]
            self._save_json(self.sessions_file, self.sessions)

            self.current_session = None

    def _hash_question(self, question: str) -> str:
        """Create a hash for question similarity matching."""
        # Normalize: lowercase, remove punctuation, sort words
        words = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in question)
        words = sorted(words.split())
        return hashlib.md5(' '.join(words).encode()).hexdigest()[:16]

    def _save_patterns(self):
        self._save_json(self.patterns_file, self.patterns)

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_feedback = len(self.feedback)
        positive = sum(1 for f in self.feedback if f['score'] > 0)
        negative = sum(1 for f in self.feedback if f['score'] < 0)

        return {
            'total_sessions': len(self.sessions),
            'patterns_learned': len(self.patterns),
            'total_feedback': total_feedback,
            'positive_feedback': positive,
            'negative_feedback': negative,
            'approval_rate': positive / total_feedback if total_feedback > 0 else 0,
        }


# Singleton for easy access
_memory: Optional[LearningMemory] = None

def get_memory() -> LearningMemory:
    """Get the global learning memory instance."""
    global _memory
    if _memory is None:
        _memory = LearningMemory()
    return _memory
