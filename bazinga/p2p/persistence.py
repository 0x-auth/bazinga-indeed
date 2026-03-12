"""
BAZINGA P2P Persistence Manager

SQLite-backed storage for DHT routing table and peer information.
Survives process restarts - your node remembers its friends.

"Memory is the mother of all wisdom." - Aeschylus
"Memory is the φ-resonance of past connections." - BAZINGA
"""

import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# BAZINGA constants
PHI = 1.618033988749895
ALPHA = 137


@dataclass
class PeerRecord:
    """A peer in the network."""
    node_id: str
    ip: str
    port: int
    last_seen: float
    trust_score: float = 0.5
    pob_count: int = 0
    is_bootstrap: bool = False
    capabilities: str = ""  # JSON string of capabilities

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'PeerRecord':
        return cls(
            node_id=row[0],
            ip=row[1],
            port=row[2],
            last_seen=row[3],
            trust_score=row[4],
            pob_count=row[5],
            is_bootstrap=bool(row[6]),
            capabilities=row[7] or ""
        )

    def is_alive(self, timeout: float = 3600) -> bool:
        """Check if peer was seen within timeout (default 1 hour)."""
        return (time.time() - self.last_seen) < timeout

    def age_seconds(self) -> float:
        """How long since we last saw this peer."""
        return time.time() - self.last_seen


@dataclass
class DHTEntry:
    """A DHT routing table entry."""
    key: str  # The DHT key (hash)
    value: str  # JSON-encoded value
    node_id: str  # Node that stored this
    timestamp: float
    ttl: int = 3600  # Time to live in seconds

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


class PersistenceManager:
    """
    SQLite-backed persistence for BAZINGA P2P network.

    Stores:
    - Known peers (node_id, ip, port, trust, last_seen)
    - DHT entries (key-value pairs with TTL)
    - Network state (last known good configuration)

    Usage:
        pm = PersistenceManager()
        pm.save_peer(peer_record)
        known_peers = pm.get_known_peers()
    """

    VERSION = "1.0.0"
    DEFAULT_PATH = Path.home() / ".bazinga" / "network.db"

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize persistence manager."""
        self.db_path = db_path or self.DEFAULT_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Peers table - known nodes in the network
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    node_id TEXT PRIMARY KEY,
                    ip TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    last_seen REAL NOT NULL,
                    trust_score REAL DEFAULT 0.5,
                    pob_count INTEGER DEFAULT 0,
                    is_bootstrap INTEGER DEFAULT 0,
                    capabilities TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)

            # DHT entries - distributed hash table storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dht_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl INTEGER DEFAULT 3600
                )
            """)

            # Network state - configuration and metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS network_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)

            # Phi-Pulse history - track discovery events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    node_id TEXT,
                    ip TEXT,
                    port INTEGER,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    details TEXT
                )
            """)

            # Peer expertise - topic specialization tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peer_expertise (
                    node_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    score REAL DEFAULT 0.5,
                    query_count INTEGER DEFAULT 0,
                    good_answers INTEGER DEFAULT 0,
                    last_queried REAL,
                    PRIMARY KEY (node_id, topic)
                )
            """)

            # Create indices for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_peers_last_seen ON peers(last_seen)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_peers_trust ON peers(trust_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dht_timestamp ON dht_entries(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expertise_topic ON peer_expertise(topic)")

            # Store schema version
            cursor.execute("""
                INSERT OR REPLACE INTO network_state (key, value)
                VALUES ('schema_version', ?)
            """, (self.VERSION,))

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get thread-safe database connection."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0
                )
                self._conn.row_factory = sqlite3.Row
            yield self._conn

    def close(self):
        """Close database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ========== PEER MANAGEMENT ==========

    def save_peer(self, peer: PeerRecord) -> bool:
        """Save or update a peer record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO peers
                    (node_id, ip, port, last_seen, trust_score, pob_count, is_bootstrap, capabilities)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    peer.node_id,
                    peer.ip,
                    peer.port,
                    peer.last_seen,
                    peer.trust_score,
                    peer.pob_count,
                    1 if peer.is_bootstrap else 0,
                    peer.capabilities
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving peer: {e}")
            return False

    def get_peer(self, node_id: str) -> Optional[PeerRecord]:
        """Get a specific peer by node_id."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM peers WHERE node_id = ?",
                    (node_id,)
                )
                row = cursor.fetchone()
                if row:
                    return PeerRecord.from_row(tuple(row))
        except Exception as e:
            print(f"Error getting peer: {e}")
        return None

    def get_known_peers(
        self,
        limit: int = 100,
        min_trust: float = 0.0,
        max_age_hours: float = 24.0
    ) -> List[PeerRecord]:
        """
        Get known peers, sorted by trust score (descending).

        Args:
            limit: Maximum number of peers to return
            min_trust: Minimum trust score filter
            max_age_hours: Only return peers seen within this time
        """
        try:
            cutoff = time.time() - (max_age_hours * 3600)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM peers
                    WHERE trust_score >= ? AND last_seen >= ?
                    ORDER BY trust_score DESC, last_seen DESC
                    LIMIT ?
                """, (min_trust, cutoff, limit))

                return [PeerRecord.from_row(tuple(row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting known peers: {e}")
            return []

    def get_bootstrap_peers(self) -> List[PeerRecord]:
        """Get bootstrap/seed peers."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM peers WHERE is_bootstrap = 1"
                )
                return [PeerRecord.from_row(tuple(row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting bootstrap peers: {e}")
            return []

    def update_peer_seen(self, node_id: str) -> bool:
        """Update last_seen timestamp for a peer."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE peers SET last_seen = ? WHERE node_id = ?",
                    (time.time(), node_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating peer: {e}")
            return False

    def update_peer_trust(self, node_id: str, trust_delta: float) -> bool:
        """Update trust score for a peer (additive)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Clamp trust between 0 and 1
                cursor.execute("""
                    UPDATE peers
                    SET trust_score = MAX(0, MIN(1, trust_score + ?))
                    WHERE node_id = ?
                """, (trust_delta, node_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating peer trust: {e}")
            return False

    def increment_pob_count(self, node_id: str) -> bool:
        """Increment PoB success count for a peer."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE peers SET pob_count = pob_count + 1 WHERE node_id = ?",
                    (node_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error incrementing PoB count: {e}")
            return False

    def remove_stale_peers(self, max_age_hours: float = 168.0) -> int:
        """Remove peers not seen in specified hours (default 1 week)."""
        try:
            cutoff = time.time() - (max_age_hours * 3600)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Don't remove bootstrap peers
                cursor.execute(
                    "DELETE FROM peers WHERE last_seen < ? AND is_bootstrap = 0",
                    (cutoff,)
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Error removing stale peers: {e}")
            return 0

    def count_peers(self) -> int:
        """Get total number of known peers."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM peers")
                return cursor.fetchone()[0]
        except Exception:
            return 0

    # ========== EXPERTISE TRACKING ==========

    def update_expertise(self, node_id: str, topic: str, good_answer: bool) -> bool:
        """
        Update a peer's expertise score for a topic.

        Called after mesh queries to track which peers are good at what.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Upsert: create or update
                cursor.execute("""
                    INSERT INTO peer_expertise (node_id, topic, score, query_count, good_answers, last_queried)
                    VALUES (?, ?, ?, 1, ?, ?)
                    ON CONFLICT(node_id, topic) DO UPDATE SET
                        query_count = query_count + 1,
                        good_answers = good_answers + ?,
                        score = CASE
                            WHEN ? THEN MIN(1.0, score + 0.05 * (1.0 - score))
                            ELSE MAX(0.1, score - 0.02)
                        END,
                        last_queried = ?
                """, (
                    node_id, topic.lower(),
                    0.5 if good_answer else 0.4,  # Initial score
                    1 if good_answer else 0,       # Initial good_answers
                    time.time(),
                    1 if good_answer else 0,       # Increment good_answers
                    good_answer,                    # For CASE condition
                    time.time(),
                ))
                conn.commit()
                return True
        except Exception:
            return False

    def get_experts(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find the best peers for a given topic.

        Returns peers sorted by expertise score, joined with trust score.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.node_id, e.topic, e.score, e.query_count, e.good_answers,
                           p.ip, p.port, p.trust_score, p.last_seen
                    FROM peer_expertise e
                    JOIN peers p ON e.node_id = p.node_id
                    WHERE e.topic = ?
                      AND p.last_seen > ?
                      AND e.score > 0.3
                    ORDER BY (e.score * p.trust_score) DESC
                    LIMIT ?
                """, (topic.lower(), time.time() - 3600, limit))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "node_id": row[0],
                        "topic": row[1],
                        "expertise_score": row[2],
                        "query_count": row[3],
                        "good_answers": row[4],
                        "ip": row[5],
                        "port": row[6],
                        "trust_score": row[7],
                        "combined_score": row[2] * row[7],  # expertise * trust
                    })
                return results
        except Exception:
            return []

    def get_peer_topics(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all topics a peer is an expert in."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT topic, score, query_count, good_answers
                    FROM peer_expertise
                    WHERE node_id = ? AND score > 0.3
                    ORDER BY score DESC
                """, (node_id,))
                return [
                    {"topic": r[0], "score": r[1], "queries": r[2], "good": r[3]}
                    for r in cursor.fetchall()
                ]
        except Exception:
            return []

    # ========== DHT STORAGE ==========

    def dht_put(self, key: str, value: Any, node_id: str, ttl: int = 3600) -> bool:
        """Store a value in the DHT."""
        try:
            value_json = json.dumps(value) if not isinstance(value, str) else value

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO dht_entries (key, value, node_id, timestamp, ttl)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, value_json, node_id, time.time(), ttl))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error in dht_put: {e}")
            return False

    def dht_get(self, key: str) -> Optional[Tuple[Any, str]]:
        """
        Get a value from the DHT.

        Returns: (value, node_id) or None if not found/expired
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value, node_id, timestamp, ttl FROM dht_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row:
                    value, node_id, timestamp, ttl = row
                    # Check if expired
                    if (time.time() - timestamp) > ttl:
                        # Clean up expired entry
                        cursor.execute("DELETE FROM dht_entries WHERE key = ?", (key,))
                        conn.commit()
                        return None

                    # Try to parse as JSON
                    try:
                        return json.loads(value), node_id
                    except json.JSONDecodeError:
                        return value, node_id
        except Exception as e:
            print(f"Error in dht_get: {e}")
        return None

    def dht_delete(self, key: str) -> bool:
        """Delete a DHT entry."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM dht_entries WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error in dht_delete: {e}")
            return False

    def dht_cleanup_expired(self) -> int:
        """Remove expired DHT entries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM dht_entries
                    WHERE (strftime('%s', 'now') - timestamp) > ttl
                """)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Error cleaning up DHT: {e}")
            return 0

    # ========== NETWORK STATE ==========

    def set_state(self, key: str, value: Any) -> bool:
        """Store network state value."""
        try:
            value_json = json.dumps(value) if not isinstance(value, str) else value

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO network_state (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, value_json, time.time()))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error setting state: {e}")
            return False

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get network state value."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value FROM network_state WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row:
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        return row[0]
        except Exception as e:
            print(f"Error getting state: {e}")
        return default

    # ========== DISCOVERY LOG ==========

    def log_discovery(
        self,
        event_type: str,
        node_id: str = None,
        ip: str = None,
        port: int = None,
        details: str = None
    ):
        """Log a discovery event."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO discovery_log (event_type, node_id, ip, port, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (event_type, node_id, ip, port, details))
                conn.commit()
        except Exception as e:
            print(f"Error logging discovery: {e}")

    def get_discovery_log(self, limit: int = 100) -> List[Dict]:
        """Get recent discovery events."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM discovery_log
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting discovery log: {e}")
            return []

    # ========== UTILITIES ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Peer stats
                cursor.execute("SELECT COUNT(*) FROM peers")
                total_peers = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM peers WHERE is_bootstrap = 1")
                bootstrap_peers = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(trust_score) FROM peers")
                avg_trust = cursor.fetchone()[0] or 0

                # DHT stats
                cursor.execute("SELECT COUNT(*) FROM dht_entries")
                dht_entries = cursor.fetchone()[0]

                # Discovery stats
                cursor.execute("SELECT COUNT(*) FROM discovery_log")
                discovery_events = cursor.fetchone()[0]

                return {
                    'total_peers': total_peers,
                    'bootstrap_peers': bootstrap_peers,
                    'avg_trust_score': round(avg_trust, 3),
                    'dht_entries': dht_entries,
                    'discovery_events': discovery_events,
                    'db_path': str(self.db_path),
                    'db_size_kb': self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0
                }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    def export_peers(self) -> List[Dict]:
        """Export all peers as list of dicts (for backup/sharing)."""
        peers = self.get_known_peers(limit=1000, max_age_hours=720)  # 30 days
        return [p.to_dict() for p in peers]

    def import_peers(self, peers: List[Dict]) -> int:
        """Import peers from list of dicts."""
        count = 0
        for p in peers:
            try:
                peer = PeerRecord(**p)
                if self.save_peer(peer):
                    count += 1
            except Exception:
                continue
        return count


# Singleton instance
_persistence_manager: Optional[PersistenceManager] = None


def get_persistence_manager(db_path: Optional[Path] = None) -> PersistenceManager:
    """Get or create the global PersistenceManager instance."""
    global _persistence_manager

    if _persistence_manager is None:
        _persistence_manager = PersistenceManager(db_path)

    return _persistence_manager


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  BAZINGA Persistence Manager Test")
    print("=" * 60)

    pm = PersistenceManager()

    # Test peer storage
    test_peer = PeerRecord(
        node_id="0x-test-node-001",
        ip="192.168.1.100",
        port=5151,
        last_seen=time.time(),
        trust_score=0.75,
        pob_count=5
    )

    print(f"\n1. Saving peer: {test_peer.node_id}")
    pm.save_peer(test_peer)

    print(f"2. Retrieving peer...")
    retrieved = pm.get_peer(test_peer.node_id)
    print(f"   Retrieved: {retrieved}")

    print(f"\n3. Testing DHT storage...")
    pm.dht_put("test-key", {"phi": PHI, "alpha": ALPHA}, "local-node")
    result = pm.dht_get("test-key")
    print(f"   DHT result: {result}")

    print(f"\n4. Network state...")
    pm.set_state("last_sync", time.time())
    pm.set_state("node_config", {"port": 5151, "broadcast": True})
    print(f"   Config: {pm.get_state('node_config')}")

    print(f"\n5. Statistics:")
    stats = pm.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print(f"\n✓ Persistence Manager ready!")
    print(f"  Database: {pm.db_path}")
