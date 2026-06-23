import logging
from datetime import UTC, datetime

from brain.bridge import cli_throttle
from brain.maker import run_maker_tick
from brain.maker.budget import consume_budget
from brain.maker.charge import MakerCharge, load_charge, save_charge
from brain.memory.store import Memory, MemoryStore


def _store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    s.create(Memory.create_new("x", "conversation", "us", emotions={"joy": 9.0}))
    return s


def test_tick_below_threshold_does_not_fire(tmp_path):
    store = _store(tmp_path)
    fired = []
    save_charge(tmp_path, MakerCharge(0.0, "2026-06-14T00:00:00+00:00", None, 0))
    run_maker_tick(
        tmp_path,
        store=store,
        provider=None,
        now=datetime(2026, 6, 14, 0, 1, tzinfo=UTC),
        make_fn=lambda **k: fired.append(1),
        threshold=100.0,
    )  # unreachable
    assert fired == []
    store.close()


def test_tick_flips_ready_eventual_share_to_shared(tmp_path):
    from datetime import timedelta

    from brain.works import Work, make_work_id
    from brain.works.store import WorksStore
    store = _store(tmp_path)
    old = datetime.now(UTC) - timedelta(hours=48)
    w = Work(id=make_work_id("share"), title="t", type="poem", created_at=old,
             session_id=None, word_count=1, summary=None, disposition="eventual_share",
             private_reason=None, origin="maker", charge_sources=None, shared_at=None)
    ws = WorksStore(tmp_path / "works.db")
    ws.insert(w, content="x")
    ws.close()
    save_charge(tmp_path, MakerCharge(0.0, "2026-06-14T00:00:00+00:00", None, 0))
    run_maker_tick(
        tmp_path,
        store=store,
        provider=None,
        now=datetime.now(UTC),
        make_fn=lambda **k: None,
        threshold=100.0,  # below-threshold: flip runs regardless of discharge
    )
    ws = WorksStore(tmp_path / "works.db")
    assert ws.get(make_work_id("share")).shared_at is not None
    ws.close()
    store.close()


def test_tick_at_threshold_fires_and_resets(tmp_path):
    store = _store(tmp_path)
    fired = []
    save_charge(tmp_path, MakerCharge(999.0, "2026-06-14T00:00:00+00:00", None, 0))
    run_maker_tick(
        tmp_path,
        store=store,
        provider=None,
        now=datetime(2026, 6, 14, 0, 1, tzinfo=UTC),
        make_fn=lambda **k: fired.append(1),
        threshold=1.0,
        cooldown_hours=0.0,
        daily_cap=5,
    )
    assert fired == [1]
    assert load_charge(tmp_path).charge == 0.0  # reset after fire
    store.close()


def test_tick_throttle_unavailable_spends_no_budget_or_cooldown(tmp_path, caplog):
    # A throttle defer must cost nothing: no make_fn call, no budget spend, no
    # cooldown engaged, no charge penalty, no ERROR log (Windows v0.0.38 report).
    store = _store(tmp_path)
    fired = []
    save_charge(tmp_path, MakerCharge(999.0, "2026-06-14T00:00:00+00:00", None, 0))
    cli_throttle.mark_interactive_active()  # chat "recent" → slot unavailable
    with caplog.at_level(logging.ERROR):
        run_maker_tick(
            tmp_path,
            store=store,
            provider=None,
            now=datetime(2026, 6, 14, 0, 1, tzinfo=UTC),
            make_fn=lambda **k: fired.append(1),
            threshold=1.0,
            cooldown_hours=0.0,
            daily_cap=1,
        )
    assert fired == []  # making not attempted
    after = load_charge(tmp_path)
    # charge preserved (still above threshold; not reset to 0, not the FAILED
    # penalty) → it retries next tick once the slot frees
    assert after.charge >= 999.0
    assert after.last_fire_ts is None  # cooldown NOT engaged
    # budget untouched → still available today
    assert consume_budget(tmp_path, now=datetime(2026, 6, 14, 0, 2, tzinfo=UTC), cap=1) is True
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)  # no crash log
    store.close()


def test_tick_respects_cooldown(tmp_path):
    store = _store(tmp_path)
    fired = []
    last_fire = (datetime(2026, 6, 14, 0, 0, tzinfo=UTC)).isoformat()
    save_charge(tmp_path, MakerCharge(999.0, "2026-06-14T00:00:00+00:00", last_fire, 0))
    run_maker_tick(
        tmp_path,
        store=store,
        provider=None,
        now=datetime(2026, 6, 14, 0, 30, tzinfo=UTC),  # 0.5h < cooldown
        make_fn=lambda **k: fired.append(1),
        threshold=1.0,
        cooldown_hours=6.0,
        daily_cap=5,
    )
    assert fired == []  # cooldown blocks
    store.close()
