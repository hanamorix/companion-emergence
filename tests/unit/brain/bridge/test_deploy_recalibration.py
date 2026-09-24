"""Tests for F2b's deploy-time one-time floor recalibration (spec §6, #276
inc3) — `brain.bridge.supervisor._run_deploy_recalibration_check` and its
wiring into `run_folded`'s startup one-shot sequence.

§5/§5b re-point F2a's daily calibration fit AND its cold-start/bootstrap
floor sources to the per-query anchor-normalized score scale. §6 covers the
one thing those don't: a floor row PERSISTED before F2b started producing
normalized scores is still raw-scale, and would otherwise sit in effect
(mismatched against the now-normalized gate) until the next scheduled daily
tick happens to re-derive it. `_run_deploy_recalibration_check` closes that
gap — an out-of-cycle, startup-fired, idempotent-by-construction call to
F2a's already-built `floor_calibration.derive_and_persist_floor`.

Mirrors `test_calibration_cadence.py`'s coverage style for the sibling
`_run_calibration_tick` (direct-function tests for the mechanism, a handful
of `run_folded`-level tests for the startup wiring + fault isolation).
"""
from __future__ import annotations

import tempfile
import threading
import time as _time
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider


def _persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    return p


# ---------------------------------------------------------------------------
# Direct-function coverage: `_run_deploy_recalibration_check`'s own logic
# (staleness check -> out-of-cycle derive).
# ---------------------------------------------------------------------------


def test_run_deploy_recalibration_check_importable_and_callable():
    import inspect

    from brain.bridge.supervisor import _run_deploy_recalibration_check

    sig = inspect.signature(_run_deploy_recalibration_check)
    assert list(sig.parameters) == ["persona_dir"]


def test_deploy_recalibration_fires_exactly_once_when_no_row_exists(monkeypatch):
    """AC10: a fresh F2b deploy (no persisted floor row for the current
    model) must run an out-of-cycle `derive_and_persist_floor` exactly
    once, without waiting for the daily cadence, and the persisted row must
    come out on the normalized scale (no longer stale) afterward."""
    from brain.bridge.supervisor import _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.floor_calibration import FloorDerivationOutcome
    from brain.memory.reranker import FakeRerankerProvider
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    calls: list[str] = []

    def fake_derive(store, model_id, **kw):
        calls.append(model_id)
        store.write_reranker_floor(
            model_id, floor=1.0, raw_fit_floor=1.0, sample_pairs=200, is_cold_start=False
        )
        return FloorDerivationOutcome(
            accepted=True, floor=1.0, raw_fit_floor=1.0, sample_pairs=200,
            is_cold_start=False, held_for_data_starvation=False,
        )

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", fake_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        _run_deploy_recalibration_check(pd)

        assert calls == ["fake-reranker"], (
            "must derive a floor for the CURRENT production reranker's model_id, exactly once"
        )
        store = MemoryStore(pd / "memories.db", integrity_check=False)
        assert store.reranker_floor_is_stale("fake-reranker") is False, (
            "the out-of-cycle write must land on the normalized scale"
        )
        store.close()


def test_deploy_recalibration_fires_for_a_legacy_raw_scale_row(monkeypatch):
    """AC10/§6: an EXISTING deployment whose persisted floor predates F2b
    (score_scale='raw') must also trigger the out-of-cycle recalibration —
    staleness is not just 'row absent', it's 'row not on the current
    scale'."""
    from brain.bridge.supervisor import _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.floor_calibration import FloorDerivationOutcome
    from brain.memory.reranker import FakeRerankerProvider
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    calls: list[str] = []

    def fake_derive(store, model_id, **kw):
        calls.append(model_id)
        store.write_reranker_floor(
            model_id, floor=1.0, raw_fit_floor=1.0, sample_pairs=200, is_cold_start=True
        )
        return FloorDerivationOutcome(
            accepted=True, floor=1.0, raw_fit_floor=1.0, sample_pairs=200,
            is_cold_start=True, held_for_data_starvation=False,
        )

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", fake_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        store = MemoryStore(pd / "memories.db", integrity_check=False)
        store.write_reranker_floor(
            "fake-reranker", floor=-3.0, raw_fit_floor=-3.0, sample_pairs=6,
            is_cold_start=True, score_scale="raw",
        )
        store.close()

        _run_deploy_recalibration_check(pd)

        assert calls == ["fake-reranker"]


def test_deploy_recalibration_noops_when_already_normalized(monkeypatch):
    """A row already stamped with the current scale must NOT trigger a
    re-derivation — the no-op path is what makes this idempotent."""
    from brain.bridge.supervisor import _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.reranker import FakeRerankerProvider
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    calls: list[str] = []
    monkeypatch.setattr(
        fc_mod, "derive_and_persist_floor", lambda store, model_id, **kw: calls.append(model_id)
    )

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        store = MemoryStore(pd / "memories.db", integrity_check=False)
        store.write_reranker_floor(
            "fake-reranker", floor=0.5, raw_fit_floor=0.5, sample_pairs=200, is_cold_start=False
        )  # default score_scale='normalized'
        store.close()

        _run_deploy_recalibration_check(pd)

        assert calls == [], "an already-normalized row must not trigger a re-derivation"


def test_deploy_recalibration_does_not_refire_on_a_second_startup(monkeypatch):
    """AC10's other half: calling the check again (simulating a second
    bridge startup) must NOT re-fire, because the first call's accepted
    write already stamped the row normalized. This is enforced by the
    PERSISTED row, not an in-process flag, so it survives a real restart."""
    from brain.bridge.supervisor import _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.floor_calibration import FloorDerivationOutcome
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    calls: list[str] = []

    def fake_derive(store, model_id, **kw):
        calls.append(model_id)
        store.write_reranker_floor(
            model_id, floor=1.0, raw_fit_floor=1.0, sample_pairs=200, is_cold_start=False
        )
        return FloorDerivationOutcome(
            accepted=True, floor=1.0, raw_fit_floor=1.0, sample_pairs=200,
            is_cold_start=False, held_for_data_starvation=False,
        )

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", fake_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        _run_deploy_recalibration_check(pd)  # first "startup" — fires
        _run_deploy_recalibration_check(pd)  # second "startup" — must no-op

        assert calls == ["fake-reranker"], "must fire on the first call only, never twice"


def test_deploy_recalibration_does_not_refire_after_a_normal_daily_tick(monkeypatch):
    """AC10: once the out-of-cycle pass has landed a normalized floor, a
    SUBSEQUENT normal daily-cadence tick (`_run_calibration_tick`, which
    also calls `derive_and_persist_floor` — this time gated on cadence
    due-ness, not staleness) must not cause THIS check to re-fire either."""
    from brain.bridge.supervisor import _run_calibration_tick, _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.floor_calibration import FloorDerivationOutcome
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    deploy_calls: list[str] = []

    def fake_derive(store, model_id, **kw):
        store.write_reranker_floor(
            model_id, floor=1.0, raw_fit_floor=1.0, sample_pairs=200, is_cold_start=False
        )
        return FloorDerivationOutcome(
            accepted=True, floor=1.0, raw_fit_floor=1.0, sample_pairs=200,
            is_cold_start=False, held_for_data_starvation=False,
        )

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", fake_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        _run_deploy_recalibration_check(pd)  # deploy transition — fires once

        # A normal daily tick runs afterward (its own gate, unrelated).
        _run_calibration_tick(pd, judge=FakeRelevanceJudgeProvider())

        def counting_derive(store, model_id, **kw):
            deploy_calls.append(model_id)
            return fake_derive(store, model_id, **kw)

        monkeypatch.setattr(fc_mod, "derive_and_persist_floor", counting_derive)
        _run_deploy_recalibration_check(pd)  # a further "startup" after the daily tick

        assert deploy_calls == [], "the deploy check must stay a no-op after a normal daily tick too"


def test_deploy_recalibration_does_not_crash_on_a_derivation_failure(monkeypatch):
    """Fault isolation (§6, mirroring the daily tick's own floor-derivation
    fault isolation): a `derive_and_persist_floor` failure must not
    propagate out of `_run_deploy_recalibration_check` itself when it
    would otherwise crash the caller — verified at the `run_folded` level
    below (that's where this function's own fault-isolation wrapper
    lives); THIS test pins that the underlying derivation failure is a
    plain exception (no special swallowing inside the function) so the
    caller's try/except is what's actually doing the isolating, not an
    accidental double-catch that would make that wrapper untested."""
    from brain.bridge.supervisor import _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    def _raising_derive(store, model_id, **kw):
        raise RuntimeError("simulated floor-derivation failure")

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", _raising_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        try:
            _run_deploy_recalibration_check(pd)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "expected the raw derivation failure to propagate out of this function — "
                "fault isolation belongs to the run_folded call site, see the wiring test below"
            )


# ---------------------------------------------------------------------------
# Pre-flip revision Change 1's §6 retry gate: a fresh deploy (or Change 1's
# own data-starvation ramp) can sit with NO persisted row at all for a
# while, which reads as stale FOREVER until real data clears the
# threshold. Without a gate, this check would re-attempt `derive_and_
# persist_floor` (a cheap no-op in that state) on EVERY bridge restart
# during the ramp. These tests use the REAL `derive_and_persist_floor` (not
# a scripted fake) so the gate is proven against the actual mechanism it
# has to distinguish, not a stand-in that could hide a wrong wiring.
# ---------------------------------------------------------------------------


def test_deploy_recalibration_gate_does_not_spin_when_no_row_and_insufficient_data(
    monkeypatch,
):
    """AC (a): a fresh-deploy state — no persisted floor row, and fewer
    than FLOOR_FIT_MIN_LABELED_PAIRS labeled pairs — must NOT re-run the
    fit on repeated deploy-checks. Counts REAL `derive_and_persist_floor`
    invocations (delegating to the real implementation) across two
    successive calls simulating two bridge restarts; a genuinely gated
    implementation calls it once, not twice."""
    from brain.bridge.supervisor import _run_deploy_recalibration_check
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.reranker import FakeRerankerProvider
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    call_count = {"n": 0}
    real_derive = fc_mod.derive_and_persist_floor

    def counting_derive(store, model_id):
        call_count["n"] += 1
        return real_derive(store, model_id)

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", counting_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        _run_deploy_recalibration_check(pd)  # first "startup" — attempts once
        _run_deploy_recalibration_check(pd)  # second "startup" — must NOT re-attempt

        assert call_count["n"] == 1, (
            "no-row/insufficient-data state must be attempted at most once per gate window, "
            f"not re-spun on every restart — got {call_count['n']} attempts"
        )
        # And genuinely no-op: still no row, nothing crashed.
        store = MemoryStore(pd / "memories.db", integrity_check=False)
        assert store.get_persisted_reranker_floor("fake-reranker") is None
        store.close()


def test_deploy_recalibration_gate_still_fires_for_an_existing_raw_scale_row_with_enough_data(
    monkeypatch,
):
    """AC (b), the HARD CONSTRAINT: an EXISTING deployment with a raw-scale
    floor row AND >= FLOOR_FIT_MIN_LABELED_PAIRS labeled pairs must still
    get its one-time raw->normalized recalibration on the deploy check,
    UNGATED — the day-boundary gate applies ONLY to the no-row case, never
    to a genuine migration. Uses the REAL `derive_and_persist_floor` end to
    end (real day-scoped fit), proving the migration actually lands, not
    just that some function was called.

    Pre-seeds the §6 retry-gate's OWN cadence file as "not yet due" (as if
    a prior no-row attempt had already spent this window's one retry)
    BEFORE running the check — a gate that (incorrectly) consulted the
    cadence unconditionally, instead of only when no prior row exists,
    would wrongly block this migration; a correctly-scoped gate ignores
    the cadence entirely for the raw-scale-row case and fires anyway."""
    from datetime import UTC, datetime

    from brain.bridge import persisted_cadence
    from brain.bridge.supervisor import (
        _DEPLOY_RECAL_RETRY_CADENCE_FILE,
        _run_deploy_recalibration_check,
    )
    from brain.memory import reranker as reranker_mod
    from brain.memory.reranker import FakeRerankerProvider
    from brain.memory.store import CALIBRATION_SCORE_SCALE, MemoryStore

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        # The gate's own cadence file, freshly "attempted" (NOT due again
        # for a full day) — this is the state the no-row case leaves
        # behind; the migration case must ignore it entirely.
        persisted_cadence.save_cadence(
            pd,
            _DEPLOY_RECAL_RETRY_CADENCE_FILE,
            persisted_cadence.advance(now=datetime.now(UTC), interval_s=86400.0),
        )
        assert not persisted_cadence.is_due(
            persisted_cadence.load_cadence(pd, _DEPLOY_RECAL_RETRY_CADENCE_FILE), now=datetime.now(UTC)
        ), "fixture bug: the pre-seeded cadence must genuinely read as not-yet-due"

        store = MemoryStore(pd / "memories.db", integrity_check=False)
        # An existing pre-F2b raw-scale row (the genuine migration case).
        store.write_reranker_floor(
            "fake-reranker", floor=-3.0, raw_fit_floor=-3.0, sample_pairs=6,
            is_cold_start=True, score_scale="raw",
        )
        # Enough real labeled (normalized-scale) pairs, all on today's
        # day_bucket, for the real fit to actually run rather than hit the
        # data-starvation backstop.
        import json

        for i in range(250):
            label = "relevant" if i % 2 == 0 else "irrelevant"
            store._conn.execute(
                "INSERT INTO calibration_log "
                "(query, candidate_ids, reranker_scores, reranker_model_id, "
                "local_judge_label, haiku_label, score_scale) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"q{i}", json.dumps(["c"]), json.dumps([float(i)]), "fake-reranker",
                    json.dumps([label]), json.dumps([None]), CALIBRATION_SCORE_SCALE,
                ),
            )
        store._conn.commit()
        store.close()

        _run_deploy_recalibration_check(pd)

        store2 = MemoryStore(pd / "memories.db", integrity_check=False)
        assert store2.reranker_floor_is_stale("fake-reranker") is False, (
            "an existing raw-scale row with enough data must migrate to normalized-scale "
            "UNCONDITIONALLY on this restart, never gated"
        )
        migrated = store2.get_persisted_reranker_floor("fake-reranker")
        assert migrated is not None
        assert migrated["floor"] != pytest.approx(-3.0), "must be a REAL fresh fit, not the old raw value"
        store2.close()


def test_deploy_recalibration_gate_retries_after_the_retry_window_elapses(monkeypatch):
    """The no-row gate rate-limits, it does not permanently block: once the
    retry cadence's window has elapsed, a later restart must attempt again
    — proven by directly expiring the persisted cadence file (simulating
    wall-clock time passing) rather than waiting in real time."""
    from datetime import UTC, datetime, timedelta

    from brain.bridge import persisted_cadence
    from brain.bridge.supervisor import (
        _DEPLOY_RECAL_RETRY_CADENCE_FILE,
        _run_deploy_recalibration_check,
    )
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kwargs: FakeRerankerProvider())

    call_count = {"n": 0}
    real_derive = fc_mod.derive_and_persist_floor

    def counting_derive(store, model_id):
        call_count["n"] += 1
        return real_derive(store, model_id)

    monkeypatch.setattr(fc_mod, "derive_and_persist_floor", counting_derive)

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        _run_deploy_recalibration_check(pd)  # attempt #1
        assert call_count["n"] == 1

        # Force the gate's cadence file into the past, as if the retry
        # window had already elapsed by the next restart.
        persisted_cadence.save_cadence(
            pd,
            _DEPLOY_RECAL_RETRY_CADENCE_FILE,
            persisted_cadence.CadenceState(next_at=datetime.now(UTC) - timedelta(seconds=1)),
        )

        _run_deploy_recalibration_check(pd)  # attempt #2 — window elapsed, must fire

        assert call_count["n"] == 2, "must retry once the gate's own window has elapsed"


# ---------------------------------------------------------------------------
# Wiring into `run_folded`'s startup one-shot sequence.
# ---------------------------------------------------------------------------


def test_run_folded_calls_deploy_recalibration_check_once_at_startup(tmp_path: Path) -> None:
    """The startup one-shot sequence must call
    `_run_deploy_recalibration_check` exactly once, distinct from (and in
    addition to) the daily-cadence catch-up calibration tick."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    stop.set()  # already set — run the startup one-shots once, then exit

    calls: list[Path] = []

    def fake_check(pd):
        calls.append(pd)

    with patch("brain.bridge.supervisor._run_deploy_recalibration_check", side_effect=fake_check):
        from brain.bridge.supervisor import run_folded

        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=bus,
            tick_interval_s=0.1,
            heartbeat_interval_s=None,
        )

    assert calls == [persona_dir], "must fire exactly once at startup, for the right persona_dir"


def test_run_folded_skips_deploy_recalibration_when_calibration_disabled(tmp_path: Path) -> None:
    """`calibration_interval_s=None` (the tests/dev disable knob for the
    whole calibration subsystem) must also skip this check — there is no
    floor-gated recall path for it to protect when calibration is off."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    stop.set()

    calls: list[Path] = []

    def fake_check(pd):
        calls.append(pd)

    with patch("brain.bridge.supervisor._run_deploy_recalibration_check", side_effect=fake_check):
        from brain.bridge.supervisor import run_folded

        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=bus,
            tick_interval_s=0.1,
            heartbeat_interval_s=None,
            calibration_interval_s=None,
        )

    assert calls == [], "must be skipped when the calibration subsystem is disabled"


def test_run_folded_deploy_recalibration_failure_does_not_crash_the_bridge(tmp_path: Path) -> None:
    """A failure inside `_run_deploy_recalibration_check` must be caught at
    the `run_folded` call site (mirroring every other one-shot startup
    step) and must not crash bridge startup."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    stop.set()

    def boom(pd):
        raise RuntimeError("simulated deploy recalibration failure")

    with patch("brain.bridge.supervisor._run_deploy_recalibration_check", side_effect=boom):
        from brain.bridge.supervisor import run_folded

        # Must not raise.
        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=bus,
            tick_interval_s=0.1,
            heartbeat_interval_s=None,
        )


def test_run_folded_deploy_recalibration_failure_does_not_block_periodic_loop(
    tmp_path: Path,
) -> None:
    """Belt-and-suspenders on the fault-isolation claim: with the check
    raising on every call, run the loop for a couple of periodic ticks (not
    just the startup one-shot) and confirm the supervisor thread stays
    alive and exits cleanly on stop — a leaked exception in this one-shot
    must never wedge the whole supervisor thread."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    attempts: list[int] = []

    def boom(pd):
        attempts.append(1)
        raise RuntimeError("simulated deploy recalibration failure")

    def runner():
        with patch("brain.bridge.supervisor._run_deploy_recalibration_check", side_effect=boom):
            from brain.bridge.supervisor import run_folded

            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    try:
        deadline = _time.monotonic() + 5.0
        while _time.monotonic() < deadline and len(attempts) < 1:
            _time.sleep(0.02)
        assert len(attempts) >= 1, "the startup one-shot never fired"
    finally:
        stop.set()
        t.join(timeout=5.0)
    assert not t.is_alive(), "supervisor loop did not exit after stop_event"
