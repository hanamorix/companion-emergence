from datetime import UTC, datetime, timedelta

from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.store import KindledLinkStore


class _StubProvider:
    def complete(self, prompt: str) -> str:
        return "draft"


class _StubThrottle:
    """Deterministic throttle for tests; defaults to NOT recently active."""
    def __init__(self, yield_=False):
        self._yield = yield_

    def should_yield(self, *, now=None):
        return self._yield


def _engine(tmp_path, *, throttle=None):
    store = KindledLinkStore(tmp_path / "k.db")
    eng = SessionEngine(store=store, identity=None, provider=_StubProvider(),
                        throttle=throttle or _StubThrottle())
    return eng, store


def _paired(store, now):
    store.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="fp",
                      consent_state="paired", relay_url="https://r", now=now)


def test_can_send_now_enforces_60s_gap(tmp_path):
    eng, store = _engine(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    store.bump_session_outbound("kid_a", "s1", now)
    assert eng.can_send_now("kid_a", "s1", now + timedelta(seconds=30)) is False
    assert eng.can_send_now("kid_a", "s1", now + timedelta(seconds=61)) is True


def test_under_session_cap_blocks_at_24(tmp_path):
    eng, store = _engine(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    for _ in range(24):
        store.bump_session_outbound("kid_a", "s1", now)
    assert eng.under_session_cap("kid_a", "s1") is False


def test_under_daily_caps(tmp_path):
    eng, store = _engine(tmp_path)
    for _ in range(20):
        store.incr_outbound_count("kid_a", "2026-06-20")
    assert eng.under_daily_caps("kid_a", "2026-06-20") is False


def test_can_start_session_requires_paired_and_no_active(tmp_path):
    eng, store = _engine(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    assert eng.can_start_session("kid_a", now) is False  # unknown peer
    _paired(store, now)
    assert eng.can_start_session("kid_a", now) is True
    store.create_session("kid_a", "s1", now)
    assert eng.can_start_session("kid_a", now) is False  # active session exists


def test_can_start_session_suppressed_by_recent_interactive_use(tmp_path):
    # §5.5: a recently-active user suppresses an autonomous start.
    eng, store = _engine(tmp_path, throttle=_StubThrottle(yield_=True))
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    _paired(store, now)
    assert eng.can_start_session("kid_a", now) is False


def test_can_start_session_blocked_by_cooldown(tmp_path):
    eng, store = _engine(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    _paired(store, now)
    store.create_session("kid_a", "s1", now)
    store.end_session("kid_a", "s1", now=now,
                      cooldown_until=now + timedelta(hours=6))
    assert eng.can_start_session("kid_a", now + timedelta(hours=1)) is False
    assert eng.can_start_session("kid_a", now + timedelta(hours=7)) is True
