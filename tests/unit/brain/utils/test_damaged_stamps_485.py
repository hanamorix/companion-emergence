"""#485: one failure policy for a damaged stored timestamp, written once.

Neighbouring readers handled a damaged stamp (wrong type, not ISO, timezone-naive) three different ways: the kindled session gate let a
bad stamp raise out of the send gate, the maker and notes cooldowns caught only ValueError so a naive or non-string stamp escaped them
with a TypeError, and pending file writes treated a naive `proposed_at` as unusable while the commit reconciler read it as UTC. Now
`brain.utils.time.lenient_utc` takes any object and returns an aware UTC datetime or None (naive means UTC, the repo's rule), and each
caller decides explicitly what None means: due now, skip, or refuse. `resonance.py` stores an aware UTC stamp.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from brain.utils.time import hours_since, lenient_utc

NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


# ---- the lenient reader ----------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-06-15T10:00:00Z", datetime(2026, 6, 15, 10, 0, tzinfo=UTC)),
        ("2026-06-15T10:00:00+00:00", datetime(2026, 6, 15, 10, 0, tzinfo=UTC)),
        ("2026-06-15T10:00:00+02:00", datetime(2026, 6, 15, 8, 0, tzinfo=UTC)),  # converted to UTC
        ("2026-06-15T10:00:00", datetime(2026, 6, 15, 10, 0, tzinfo=UTC)),  # naive means UTC
        (datetime(2026, 6, 15, 10, 0), datetime(2026, 6, 15, 10, 0, tzinfo=UTC)),
        (datetime(2026, 6, 15, 10, 0, tzinfo=timezone(timedelta(hours=-5))), datetime(2026, 6, 15, 15, 0, tzinfo=UTC)),
    ],
)
def test_lenient_utc_returns_an_aware_utc_datetime(value, expected):
    got = lenient_utc(value)

    assert got == expected and got.utcoffset() == timedelta(0)


@pytest.mark.parametrize("bad", [None, "", "garbage", "2026-13-45T00:00:00Z", 5, 1.5, [], {}, b"2026-06-15T10:00:00Z", "0000-00-00"])
def test_lenient_utc_returns_none_for_anything_else(bad):
    assert lenient_utc(bad) is None


def test_hours_since_is_none_for_an_unusable_stamp_and_never_negative():
    assert hours_since("2026-06-15T10:00:00Z", NOW) == pytest.approx(2.0)
    assert hours_since("2026-06-15T10:00:00", NOW) == pytest.approx(2.0)  # naive means UTC
    assert hours_since("2026-06-15T14:00:00Z", NOW) == 0.0  # a stamp in the future is "now", not negative
    assert hours_since("garbage", NOW) is None and hours_since(None, NOW) is None and hours_since(7, NOW) is None


# ---- the kindled send gate: a bad stamp means "due now", like cadence.py ----


def _engine(store, throttle=None):
    from brain.kindled_link.session_engine import SessionEngine

    eng = object.__new__(SessionEngine)
    eng._store = store
    eng._throttle = throttle or SimpleNamespace(should_yield=lambda: False)
    return eng


@pytest.mark.parametrize("stamp", ["garbage", 7, "2026-06-15T11:59:59", ["x"]])
def test_can_send_now_does_not_raise_on_a_damaged_last_outbound_stamp(stamp):
    store = SimpleNamespace(get_session=lambda p, s: {"last_outbound_at": stamp, "msg_count": 0})

    assert _engine(store).can_send_now("p", "s", NOW) in (True, False)  # no raise


def test_can_send_now_treats_an_unusable_stamp_as_due_and_a_naive_one_as_utc():
    garbage = SimpleNamespace(get_session=lambda p, s: {"last_outbound_at": "garbage"})
    naive_recent = SimpleNamespace(get_session=lambda p, s: {"last_outbound_at": "2026-06-15T11:59:59"})
    naive_old = SimpleNamespace(get_session=lambda p, s: {"last_outbound_at": "2026-06-15T08:00:00"})

    assert _engine(garbage).can_send_now("p", "s", NOW) is True  # due now
    assert _engine(naive_recent).can_send_now("p", "s", NOW) is False  # one second ago, read as UTC: still inside the gap
    assert _engine(naive_old).can_send_now("p", "s", NOW) is True


@pytest.mark.parametrize("stamp", ["garbage", 7, "2026-06-15T13:00:00"])
def test_can_start_session_does_not_raise_on_a_damaged_cooldown_stamp(stamp):
    store = SimpleNamespace(
        get_peer=lambda p: {"consent_state": "paired"},
        get_active_session=lambda p: None,
        latest_cooldown_until=lambda p: stamp,
    )

    assert _engine(store).can_start_session("p", NOW) in (True, False)


def test_can_start_session_reads_a_naive_cooldown_as_utc_and_ignores_an_unusable_one():
    def store(stamp):
        return SimpleNamespace(
            get_peer=lambda p: {"consent_state": "paired"}, get_active_session=lambda p: None, latest_cooldown_until=lambda p: stamp
        )

    assert _engine(store("2026-06-15T13:00:00")).can_start_session("p", NOW) is False  # still cooling (naive = UTC)
    assert _engine(store("garbage")).can_start_session("p", NOW) is True  # unusable: the cooldown cannot hold anyone back
    assert _engine(store("2026-06-15T11:00:00Z")).can_start_session("p", NOW) is True


# ---- maker and notes cooldowns: only ValueError was caught ----------------


def test_maker_charge_elapsed_hours_survives_a_naive_or_non_string_stamp():
    from brain.maker.charge import _elapsed_hours

    assert _elapsed_hours("2026-06-15T10:00:00", NOW) == pytest.approx(2.0)  # naive means UTC (was a TypeError)
    assert _elapsed_hours(7, NOW) == 0.0 and _elapsed_hours("garbage", NOW) == 0.0 and _elapsed_hours(None, NOW) == 0.0


@pytest.mark.parametrize("stamp", ["2026-06-15T11:00:00", 7, ["x"], "garbage"])
def test_the_maker_tick_does_not_raise_on_a_damaged_last_fire_stamp(tmp_path, monkeypatch, stamp):
    import brain.maker as maker
    from brain.maker.charge import MakerCharge, save_charge

    save_charge(tmp_path, MakerCharge(charge=99.0, last_tick_ts=NOW.isoformat(), last_fire_ts=stamp, prior_soul_count=0))
    monkeypatch.setattr(maker, "current_emotional_intensity", lambda store: 0.0)
    monkeypatch.setattr(maker, "dreams_since", lambda store, ts: 0)
    monkeypatch.setattr(maker, "soul_pending_count", lambda pd: 0)
    monkeypatch.setattr("brain.maker.wiring.flip_ready_shares", lambda *a, **k: None)
    reached = []
    monkeypatch.setattr("brain.bridge.cli_throttle.slot_available", lambda: reached.append(1) or False)

    maker.run_maker_tick(tmp_path, store=None, provider=None, now=NOW, threshold=1.0, cooldown_hours=0.5)

    # "2026-06-15T11:00:00" (naive = UTC) is one hour ago: past the 0.5 h cooldown, so it reaches the throttle gate; an
    # unusable stamp cannot hold a cooldown, so it reaches the gate too.
    assert reached == [1]


def test_the_notes_tick_does_not_raise_on_a_damaged_last_note_stamp(tmp_path, monkeypatch):
    import brain.notes as notes
    from brain.notes.state import NotesState, save_notes_state

    config = SimpleNamespace(notes_enabled=True, notes_folder=str(tmp_path))
    reached = []
    monkeypatch.setattr("brain.bridge.cli_throttle.slot_available", lambda: reached.append(1) or False)
    for stamp in ("2026-06-15T01:00:00", "garbage"):
        save_notes_state(tmp_path, NotesState(last_note_at=stamp))
        notes.run_notes_tick(tmp_path, config=config, provider=None, silence_hours=99.0, now=NOW, cooldown_hours=1.0, away_hours=1.0)

    assert reached == [1, 1]


# ---- pending file writes: a naive proposed_at is read as UTC, as commit.py does ----


def test_a_naive_proposed_at_is_read_as_utc_by_the_pending_writes():
    from brain.files import pending

    rec = {"id": "r1", "proposed_at": "2026-06-14T09:00:00"}  # naive: 27 hours before NOW if read as UTC

    assert pending.is_expired(rec, NOW) is True
    assert pending._age(rec, NOW) == timedelta(hours=27)


@pytest.mark.parametrize("rec", [{"id": "r1"}, {"id": "r1", "proposed_at": None}, {"id": "r1", "proposed_at": "garbage"}, {"id": "r1", "proposed_at": 7}])
def test_an_unusable_proposed_at_is_still_never_expired_and_skipped(rec):
    # PIN: unchanged policy for what cannot be read at all.
    from brain.files import pending

    assert pending.is_expired(rec, NOW) is False
    assert pending._age(rec, NOW) is None


# ---- the resonance writer stores an aware UTC stamp ----------------------


def test_resonance_stores_an_aware_utc_stamp(tmp_path):
    from brain.initiate.resonance import MemoryActivationBaseline

    store = MemoryActivationBaseline(tmp_path / "resonance.db")
    store.update("m1", 3.0, alpha=0.3)
    row = store.get("m1")
    store.close()

    parsed = datetime.fromisoformat(row.last_updated_ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 60


def test_hours_since_reads_a_naive_now_as_utc_too():
    """Code review 3b, S7: every other call site wraps `now` in `lenient_utc`; `hours_since` subtracted it raw, so a naive `now`
    raised TypeError in the maker, notes and charge paths."""
    naive_now = datetime(2026, 6, 15, 12, 0, 0)  # the repo's rule: naive means UTC
    assert hours_since("2026-06-15T10:00:00Z", naive_now) == pytest.approx(2.0)
    assert hours_since("2026-06-15T10:00:00", naive_now) == pytest.approx(2.0)
    assert hours_since("not a stamp", naive_now) is None
    assert hours_since("2026-06-15T10:00:00Z", None) is None  # an unusable `now` is unusable, not a crash
