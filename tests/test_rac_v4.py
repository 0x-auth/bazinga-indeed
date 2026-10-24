"""RAC uses the V4 statuses from resonance.coherence_gap() and carries cumulative ΔΓ forward."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HOME"] = tempfile.mkdtemp()

from bazinga.rac import ResonanceTargetWrapper, STATUS_DISPLAY


class FakeSession:
    session_id = "test-session"


class FakeMemory:
    def start_session(self): return FakeSession()
    def end_session(self): pass
    def record_interaction(self, *a, **k): pass
    def find_similar_question(self, q): return {"answer": "a", "coherence": 0.8}
    def get_stats(self): return {}


def make():
    rac = ResonanceTargetWrapper(memory=FakeMemory())
    rac.carm = None          # keep the test independent of on-disk CARM state
    rac._trd_engine = None
    return rac


def test_statuses_are_v4():
    rac = make(); rac.start_session()
    for c in (0.9, 0.2, 0.95, 0.1):
        rac.record_interaction("question about phi", "answer", "rag", coherence=c)
    statuses = {p.status for p in rac.trajectory.points}
    assert statuses <= set(STATUS_DISPLAY), statuses
    assert rac.trajectory.status in STATUS_DISPLAY


def test_cumulative_is_carried_forward():
    rac = make(); rac.start_session()
    for c in (0.9, 0.3, 0.8, 0.6):
        rac.record_interaction("q", "a", "rag", coherence=c)
    cum = [p.cumulative_delta for p in rac.trajectory.points]
    dg = [p.delta_gamma for p in rac.trajectory.points]
    assert all(b >= a for a, b in zip(cum, cum[1:])), cum
    assert abs(cum[-1] - sum(dg)) < 1e-9


def test_display_uses_v4_labels():
    rac = make(); rac.start_session()
    rac.record_interaction("q", "a", "rag", coherence=0.9)
    out = rac.format_rac_display()
    assert any(label in out for label in STATUS_DISPLAY.values())
    assert "DRIFTING" not in out and "LOCKED" not in out
    assert "Cumulative" in out


def test_virgin_no_longer_boosts_cache():
    rac = make(); rac.start_session()
    rac.trajectory.points[-1].delta_gamma = 0.05   # virgin zone
    assert rac.find_similar_question("q")["coherence"] == 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
