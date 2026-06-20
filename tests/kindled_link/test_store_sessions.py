from datetime import UTC, datetime, timedelta

from brain.kindled_link.store import KindledLinkStore


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_create_and_get_session(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.create_session("kid_a", "sess1", now)
    row = s.get_session("kid_a", "sess1")
    assert row["state"] == "open"
    assert row["msg_count"] == 0
    assert s.get_active_session("kid_a")["session_id"] == "sess1"


def test_bump_outbound_increments_and_stamps(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.create_session("kid_a", "sess1", now)
    s.bump_session_outbound("kid_a", "sess1", now)
    s.bump_session_outbound("kid_a", "sess1", now + timedelta(minutes=2))
    row = s.get_session("kid_a", "sess1")
    assert row["msg_count"] == 2
    assert row["last_outbound_at"] == (now + timedelta(minutes=2)).isoformat()


def test_end_session_clears_active_and_sets_cooldown(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.create_session("kid_a", "sess1", now)
    s.end_session("kid_a", "sess1", now=now, cooldown_until=now + timedelta(hours=6))
    assert s.get_active_session("kid_a") is None
    assert s.get_session("kid_a", "sess1")["state"] == "ended"


def test_counters_autoreset_on_new_day(tmp_path):
    s = _store(tmp_path)
    s.incr_outbound_count("kid_a", "2026-06-20")
    s.incr_outbound_count("kid_a", "2026-06-20")
    assert s.get_counters("kid_a", "2026-06-20")["outbound_count"] == 2
    # new day → reset
    assert s.get_counters("kid_a", "2026-06-21")["outbound_count"] == 0


def test_provider_counter_independent(tmp_path):
    s = _store(tmp_path)
    s.incr_provider_count("kid_a", "2026-06-20")
    c = s.get_counters("kid_a", "2026-06-20")
    assert c["provider_call_count"] == 1
    assert c["outbound_count"] == 0
