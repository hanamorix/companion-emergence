import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge import persisted_cadence as pc


def test_is_due_true_when_never_run_and_advance_sets_next_at():
    now = datetime(2026, 6, 23, 12, tzinfo=UTC)
    assert pc.is_due(pc.CadenceState(next_at=None), now=now) is True
    advanced = pc.advance(now=now, interval_s=3600.0)
    assert advanced.next_at == now + timedelta(seconds=3600.0)
    assert pc.is_due(advanced, now=now) is False
    assert pc.is_due(advanced, now=now + timedelta(seconds=3601)) is True


def test_save_load_round_trip_and_no_tmp_left():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 23, 12, tzinfo=UTC)
        state = pc.advance(now=now, interval_s=3600.0)
        pc.save_cadence(pd, "x_cadence.json", state)
        loaded = pc.load_cadence(pd, "x_cadence.json")
        assert loaded.next_at == state.next_at
        assert not list(pd.glob("*.tmp")), "atomic write must leave no .tmp"


def test_missing_and_corrupt_state_is_due_now():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 23, 12, tzinfo=UTC)
        assert pc.is_due(pc.load_cadence(pd, "nope.json"), now=now) is True
        (pd / "bad.json").write_text("{not json", encoding="utf-8")
        assert pc.is_due(pc.load_cadence(pd, "bad.json"), now=now) is True
        (pd / "list.json").write_text("[1,2,3]", encoding="utf-8")
        assert pc.is_due(pc.load_cadence(pd, "list.json"), now=now) is True
        (pd / "bts.json").write_text('{"next_at": "not-a-date"}', encoding="utf-8")
        assert pc.is_due(pc.load_cadence(pd, "bts.json"), now=now) is True


def test_future_next_at_not_due_until_wall_clock_passes_it():
    # Core defect-fixed proof: due-ness is pure wall-clock, NO monotonic advance.
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        base = datetime(2026, 6, 23, 12, tzinfo=UTC)
        pc.save_cadence(pd, "v.json", pc.advance(now=base, interval_s=86400.0))
        state = pc.load_cadence(pd, "v.json")
        assert pc.is_due(state, now=base + timedelta(hours=1)) is False
        assert pc.is_due(state, now=base + timedelta(hours=25)) is True


def test_save_cadence_is_fail_soft_on_oserror(monkeypatch):
    # A cadence save runs inside supervisor finally-blocks; a raised OSError
    # (disk full / EACCES / Windows AV holding the .tmp during rename) would
    # escape run_folded and kill the whole supervisor. save_cadence must
    # swallow it (best-effort persistence, like load_cadence).
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        state = pc.advance(now=datetime(2026, 6, 23, 12, tzinfo=UTC), interval_s=3600.0)

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        # Must NOT raise.
        pc.save_cadence(pd, "v.json", state)
