"""Tests for brain.memory.judge_selftune — F2c inc2's OFFLINE weekly-tick
SCAFFOLD (spec `f2c-judge-selftune-spec.md` §1 [hardware-tiered mechanism],
§2 [cadence + gate], §7 [scope]).

Covers what inc2 actually builds: the >handful gate, runtime RAM
tier-detection, the cgroup-aware OOM-safety downgrade, the per-persona
last-trained marker + the per-row consumed marker (round-trip via
`MemoryStore` — red-team fixes F-1/F-2 replaced the original MAX(id)
watermark + judge-labeled-row count with a per-row `selftune_consumed_at`
marker and a non-None-`haiku_label`-position count), and
`_run_judge_selftune_tick`'s fault isolation. Cadence WIRING into
`supervisor.run_folded` (is_due/advance/save, startup-catch-up-free shape,
`=None` disables) is covered separately in
`tests/bridge/test_supervisor_cadence_persisted.py`, alongside the other
persisted cadences it mirrors.

All offline — no real model, no network, no torch import anywhere in this
module or its tests (F2c inc2 builds no tuning code yet; see the module's
own TODO(F2c inc3+) marker).
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import mock_open

import pytest

from brain.bridge import persisted_cadence as pc
from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE
from brain.memory import judge_selftune
from brain.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


def _seed_labeled_rows(
    store: MemoryStore, n: int, *, raw_score: float | None = 2.0
) -> list[int]:
    """Insert `n` already judge-labeled `calibration_log` rows (`haiku_label`
    non-null, mirrors `write_calibration_labels`'s always-write contract —
    see that method's own docstring) and return their ids in insertion
    order.

    `raw_score` (F2c inc3): each row's single candidate position also gets
    `local_judge_raw_score=[raw_score]` written by default, so a tick that
    fires against these seeded rows has real (score, label) pairs for the
    knob-refit to train on — without it, `fit_platt_knob` would see zero
    usable pairs and raise (AC3's "no train" fault case, see
    `test_tick_fires_above_the_handful_threshold` and friends below).
    Pass `raw_score=None` to reproduce a legacy/pre-inc1 row that has no
    raw score logged at all (used by the dedicated "no usable pairs"
    bite test).
    """
    ids: list[int] = []
    for i in range(n):
        store.log_calibration_sample(
            query=f"q{i}", candidate_ids=["m"], reranker_scores=[1.0], reranker_model_id="m"
        )
        row_id = store._conn.execute(
            "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
        ).fetchone()["id"]
        local_judge_raw_score = [raw_score] if raw_score is not None else None
        store.write_calibration_labels(
            row_id, ["relevant"], ["relevant"], local_judge_raw_score=local_judge_raw_score
        )
        ids.append(row_id)
    return ids


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_cadence_constants() -> None:
    assert judge_selftune.JUDGE_TUNE_CADENCE_FILE == "judge_selftune_cadence.json"
    assert judge_selftune.JUDGE_TUNE_INTERVAL_HOURS == 168.0


def test_tune_grade_order_is_weak_to_beefy() -> None:
    assert judge_selftune._TUNE_GRADE_ORDER == (
        judge_selftune.TUNE_GRADE_KNOB_REFIT,
        judge_selftune.TUNE_GRADE_LORA,
        judge_selftune.TUNE_GRADE_FULL_FT,
    )


# ---------------------------------------------------------------------------
# _read_total_ram_bytes — mirrors reranker._proc_meminfo_available_bytes's
# shape, MemTotal instead of MemAvailable.
# ---------------------------------------------------------------------------


def test_read_total_ram_bytes_parses_memtotal_kb_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_selftune.sys, "platform", "linux")
    meminfo = "MemTotal:       16333000 kB\nMemFree:         2000000 kB\nMemAvailable:    5000000 kB\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=meminfo))
    assert judge_selftune._read_total_ram_bytes() == 16_333_000 * 1024.0


def test_read_total_ram_bytes_none_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_selftune.sys, "platform", "darwin")
    assert judge_selftune._read_total_ram_bytes() is None


def test_read_total_ram_bytes_none_on_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_selftune.sys, "platform", "linux")

    def _boom(*a, **k):
        raise OSError("no /proc on this box")

    monkeypatch.setattr("builtins.open", _boom)
    assert judge_selftune._read_total_ram_bytes() is None


def test_read_total_ram_bytes_none_when_memtotal_line_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_selftune.sys, "platform", "linux")
    monkeypatch.setattr("builtins.open", mock_open(read_data="MemFree: 2000000 kB\n"))
    assert judge_selftune._read_total_ram_bytes() is None


# ---------------------------------------------------------------------------
# _select_tune_grade_by_ram — AC1 (tier detection): scripted RAM values ->
# the correct tier per the (tunable) thresholds, evaluated at RUNTIME.
# ---------------------------------------------------------------------------


def test_select_tune_grade_weak_below_lora_min() -> None:
    assert (
        judge_selftune._select_tune_grade_by_ram(8.0 * 1024**3)
        == judge_selftune.TUNE_GRADE_KNOB_REFIT
    )


def test_select_tune_grade_lora_at_lora_min_boundary() -> None:
    total = judge_selftune.JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES
    assert judge_selftune._select_tune_grade_by_ram(total) == judge_selftune.TUNE_GRADE_LORA


def test_select_tune_grade_knob_refit_just_below_lora_min_boundary() -> None:
    total = judge_selftune.JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES - 1.0
    assert judge_selftune._select_tune_grade_by_ram(total) == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_select_tune_grade_full_ft_at_full_ft_min_boundary() -> None:
    total = judge_selftune.JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES
    assert judge_selftune._select_tune_grade_by_ram(total) == judge_selftune.TUNE_GRADE_FULL_FT


def test_select_tune_grade_lora_just_below_full_ft_min_boundary() -> None:
    total = judge_selftune.JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES - 1.0
    assert judge_selftune._select_tune_grade_by_ram(total) == judge_selftune.TUNE_GRADE_LORA


def test_select_tune_grade_none_fails_toward_knob_refit() -> None:
    """Detection failure (non-Linux / read error) must fail toward the
    always-safe floor, never toward a grade that might not fit."""
    assert judge_selftune._select_tune_grade_by_ram(None) == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_select_tune_grade_reads_tunable_override_live_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BITE-CHECK the 'evaluated at runtime, not cached' requirement (spec
    §1): the SAME total_ram_bytes input must select a DIFFERENT grade once
    a tunable override changes the threshold, on the very next call — no
    memoization across calls."""
    total = 20.0 * 1024**3  # between the default lora_min (16G) and full_ft_min (64G)
    assert judge_selftune._select_tune_grade_by_ram(total) == judge_selftune.TUNE_GRADE_LORA

    original_get_tunable = judge_selftune.tunables.get_tunable

    def _override(key, default):
        if key == "judge_selftune.ram_tier_lora_min_bytes":
            return 100.0 * 1024**3  # now above `total` -> should downgrade
        return original_get_tunable(key, default)

    monkeypatch.setattr(judge_selftune.tunables, "get_tunable", _override)
    assert judge_selftune._select_tune_grade_by_ram(total) == judge_selftune.TUNE_GRADE_KNOB_REFIT


# ---------------------------------------------------------------------------
# _downgrade_for_oom_safety — AC1's OOM-guard half. BITE: a scripted
# (chosen-tier footprint > effective headroom) case DOWNGRADES; without the
# guard the too-big tier would have been kept.
# ---------------------------------------------------------------------------


def test_oom_guard_keeps_grade_when_headroom_is_generous() -> None:
    headroom = judge_selftune.JUDGE_TUNE_FOOTPRINT_FULL_FT_BYTES + 1.0
    result = judge_selftune._downgrade_for_oom_safety(judge_selftune.TUNE_GRADE_FULL_FT, headroom)
    assert result == judge_selftune.TUNE_GRADE_FULL_FT


def test_oom_guard_downgrades_full_ft_to_lora_when_only_lora_fits() -> None:
    headroom = judge_selftune.JUDGE_TUNE_FOOTPRINT_LORA_BYTES + 1.0
    result = judge_selftune._downgrade_for_oom_safety(judge_selftune.TUNE_GRADE_FULL_FT, headroom)
    assert result == judge_selftune.TUNE_GRADE_LORA


def test_oom_guard_downgrades_to_knob_refit_when_only_the_floor_fits() -> None:
    headroom = judge_selftune.JUDGE_TUNE_FOOTPRINT_KNOB_REFIT_BYTES
    result = judge_selftune._downgrade_for_oom_safety(judge_selftune.TUNE_GRADE_FULL_FT, headroom)
    assert result == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_oom_guard_none_headroom_fails_toward_knob_refit() -> None:
    """No headroom signal at all (every source unavailable) means no
    signal to TRUST the RAM-tier pick with either — same fail-toward-safe
    posture as `_select_tune_grade_by_ram`'s own None branch."""
    result = judge_selftune._downgrade_for_oom_safety(judge_selftune.TUNE_GRADE_FULL_FT, None)
    assert result == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_oom_guard_bites_without_it_the_naive_pick_would_oom() -> None:
    """The actual BITE: a beefy host-total RAM naively selects full_ft, but
    a cgroup-capped effective headroom that only fits knob-refit must
    override that pick. Proves the guard actually changes the outcome, not
    merely that it exists."""
    total_ram = judge_selftune.JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES + 1.0
    naive = judge_selftune._select_tune_grade_by_ram(total_ram)
    assert naive == judge_selftune.TUNE_GRADE_FULL_FT, "the too-big pick, absent the guard"

    guarded = judge_selftune._downgrade_for_oom_safety(
        naive, judge_selftune.JUDGE_TUNE_FOOTPRINT_KNOB_REFIT_BYTES
    )
    assert guarded == judge_selftune.TUNE_GRADE_KNOB_REFIT
    assert guarded != naive, "the guard must have changed the outcome"


def test_oom_guard_knob_refit_footprint_is_the_floor_it_never_downgrades_past() -> None:
    result = judge_selftune._downgrade_for_oom_safety(judge_selftune.TUNE_GRADE_KNOB_REFIT, 1.0)
    assert result == judge_selftune.TUNE_GRADE_KNOB_REFIT


# ---------------------------------------------------------------------------
# _downgrade_for_missing_lora_extra — F2c inc5a's optional-extra guard
# (Opus cold-review round 2, Planning ruling): the SAME downgrade-to-floor
# posture as the OOM guard above, gated on `judge_lora.lora_available()`
# instead of memory headroom.
# ---------------------------------------------------------------------------


def test_lora_extra_guard_keeps_grade_when_lora_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import brain.memory.judge_lora as judge_lora

    monkeypatch.setattr(judge_lora, "lora_available", lambda: True)
    result = judge_selftune._downgrade_for_missing_lora_extra(judge_selftune.TUNE_GRADE_LORA)
    assert result == judge_selftune.TUNE_GRADE_LORA


def test_lora_extra_guard_downgrades_lora_to_knob_refit_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual BITE: `judge_lora.lora_available()` monkeypatched False
    (the optional `f2c-training` extra isn't installed) on a LoRA-grade
    pick must downgrade to the always-safe knob-refit floor -- mirrors
    `test_oom_guard_bites_without_it_the_naive_pick_would_oom`'s "proves
    the guard actually changes the outcome" posture."""
    import brain.memory.judge_lora as judge_lora

    monkeypatch.setattr(judge_lora, "lora_available", lambda: False)
    result = judge_selftune._downgrade_for_missing_lora_extra(judge_selftune.TUNE_GRADE_LORA)
    assert result == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_lora_extra_guard_downgrades_full_ft_to_knob_refit_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import brain.memory.judge_lora as judge_lora

    monkeypatch.setattr(judge_lora, "lora_available", lambda: False)
    result = judge_selftune._downgrade_for_missing_lora_extra(judge_selftune.TUNE_GRADE_FULL_FT)
    assert result == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_lora_extra_guard_knob_refit_never_calls_lora_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """knob_refit short-circuits BEFORE checking availability at all --
    mirrors `_downgrade_for_oom_safety`'s identical short-circuit for this
    grade (the floor never needs to ask "is the optional extra installed"
    since it never uses it)."""
    import brain.memory.judge_lora as judge_lora

    def _must_not_be_called() -> bool:
        raise AssertionError("lora_available() must not be called for knob_refit")

    monkeypatch.setattr(judge_lora, "lora_available", _must_not_be_called)
    result = judge_selftune._downgrade_for_missing_lora_extra(judge_selftune.TUNE_GRADE_KNOB_REFIT)
    assert result == judge_selftune.TUNE_GRADE_KNOB_REFIT


def test_tick_downgrades_to_knob_refit_when_ram_selects_lora_but_extra_unavailable(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end BITE through the real tick: a scripted RAM config that
    naively selects LoRA, with generous OOM headroom (the OOM guard alone
    would KEEP the LoRA pick), but `lora_available()` monkeypatched False
    -- the tick's actual `tune_grade` result must still land on knob-refit.
    """
    import brain.memory.judge_lora as judge_lora

    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    _seed_labeled_rows(store, handful + 1)
    monkeypatch.setattr(
        judge_selftune, "_read_total_ram_bytes",
        lambda: judge_selftune.JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES + 1.0,
    )
    monkeypatch.setattr(
        judge_selftune, "_available_ram_headroom_bytes",
        lambda: judge_selftune.JUDGE_TUNE_FOOTPRINT_LORA_BYTES + 1.0,  # generous: OOM guard alone keeps LoRA
    )
    monkeypatch.setattr(judge_lora, "lora_available", lambda: False)

    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))

    assert result["fired"] is True
    assert result["tune_grade"] == judge_selftune.TUNE_GRADE_KNOB_REFIT


# ---------------------------------------------------------------------------
# MemoryStore.count_new_haiku_decisions / mark_selftune_consumed — the
# >handful gate's data source (spec §2/§3), red-team fixes F-1 (per-row
# consume-once, no MAX(id) watermark) + F-2 (count non-None POSITIONS, not
# judge-labeled ROWS).
# ---------------------------------------------------------------------------


def _seed_row_with_haiku_labels(
    store: MemoryStore, haiku_labels: list[str | None], *, n_candidates: int | None = None
) -> int:
    """Insert ONE judge-labeled `calibration_log` row with an EXPLICIT
    `haiku_label` list (unlike `_seed_labeled_rows`, which always writes a
    single-candidate, single-non-None-position row) — lets a test control
    exactly how many non-None POSITIONS one row contributes. Returns the
    row id."""
    n = n_candidates if n_candidates is not None else len(haiku_labels)
    candidate_ids = [f"m{i}" for i in range(n)]
    store.log_calibration_sample(
        query="q", candidate_ids=candidate_ids, reranker_scores=[1.0] * n, reranker_model_id="m"
    )
    row_id = store._conn.execute(
        "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]
    store.write_calibration_labels(row_id, ["relevant"] * n, haiku_labels)
    return row_id


def test_count_new_haiku_decisions_counts_labeled_rows(store: MemoryStore) -> None:
    _seed_labeled_rows(store, 3)  # 1 non-None position each (see helper docstring)
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 3
    assert sorted(row_ids) == [1, 2, 3]


def test_count_new_haiku_decisions_excludes_unlabeled_rows(store: MemoryStore) -> None:
    store.log_calibration_sample(
        query="q", candidate_ids=[], reranker_scores=[], reranker_model_id="m"
    )
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 0
    assert row_ids == []


def test_count_new_haiku_decisions_counts_positions_not_rows(store: MemoryStore) -> None:
    """AC3 / F-2 bite: a SINGLE row with `haiku_label` `[None, "relevant",
    None]` must count as 1 decision (one non-None POSITION) — not as 1 row
    (the old row-count semantics) and not as 3 (the list's length). Before
    the F-2 fix this method counted `haiku_label IS NOT NULL` ROWS, which
    would have read this as 1 anyway by coincidence of row-count == 1; the
    real break is exposed by the all-None-row test below, which the OLD
    row-counting query could not distinguish from this one."""
    _seed_row_with_haiku_labels(store, [None, "relevant", None])
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 1
    assert len(row_ids) == 1


def test_count_new_haiku_decisions_all_none_rows_do_not_count(store: MemoryStore) -> None:
    """AC3 bite (F-2, the actual break the fix targets): MANY judge-labeled
    rows whose `haiku_label` lists are ALL-`None` (the local judge never
    routed any candidate to Haiku that turn — `write_calibration_labels`
    still writes a non-NULL JSON list of `None`s per its own contract) must
    contribute ZERO to the count. The OLD `haiku_label IS NOT NULL` ROW
    count would have wrongly read this as `handful + 1` and fired the gate
    on zero new Haiku signal — exactly the bug AC3 requires closed."""
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    for _ in range(handful + 1):
        _seed_row_with_haiku_labels(store, [None, None, None])
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 0, "all-None-position rows must not count as Haiku decisions"
    assert len(row_ids) == handful + 1, "the rows are still returned so a firing tick can consume them"


def test_count_new_haiku_decisions_excludes_consumed_rows(store: MemoryStore) -> None:
    ids = _seed_labeled_rows(store, 3)
    store.mark_selftune_consumed([ids[0]], consumed_at=datetime.now(UTC))
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 2
    assert sorted(row_ids) == sorted(ids[1:])


def test_count_new_haiku_decisions_out_of_order_low_id_row_still_counts(store: MemoryStore) -> None:
    """F-1 bite: the exact scenario a MAX(id) watermark strands. Row 1 is
    logged but stays UNLABELED (simulating `sample_unlabeled_calibration_
    rows`'s randomized sampling skipping it this week); row 2 gets labeled
    and consumed this week. A week later, row 1 FINALLY gets labeled (its
    `haiku_label` written out of order, after a higher-id row already
    advanced past it) — it must still be counted. A `MAX(id)`
    watermark set to row 2's id would have permanently excluded row 1
    (`id > since_id` with `since_id == 2` never matches `id == 1`); the
    per-row `selftune_consumed_at` marker has no such ordering dependency."""
    store.log_calibration_sample(
        query="q1", candidate_ids=["m"], reranker_scores=[1.0], reranker_model_id="m"
    )
    row1_id = store._conn.execute(
        "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]
    # row 1 stays unlabeled here — sampled-out this week.
    row2_id = _seed_row_with_haiku_labels(store, ["relevant"])
    store.mark_selftune_consumed([row2_id], consumed_at=datetime.now(UTC))

    # Week later: row 1 finally gets labeled, out of order relative to row 2.
    store.write_calibration_labels(row1_id, ["relevant"], ["relevant"])

    count, row_ids = store.count_new_haiku_decisions()
    assert count == 1, "the out-of-order low-id row's Haiku decision must still be counted"
    assert row_ids == [row1_id]


def test_count_new_haiku_decisions_survives_retention_pruning(store: MemoryStore) -> None:
    """`calibration_log`'s own rolling retention can delete OLD rows
    outright — the count is a live scan of whatever rows physically remain
    (no cursor to corrupt), so deleting one just shrinks the result."""
    ids = _seed_labeled_rows(store, 3)
    store._conn.execute("DELETE FROM calibration_log WHERE id = ?", (ids[0],))
    store._conn.commit()
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 2
    assert sorted(row_ids) == sorted(ids[1:])


def test_mark_selftune_consumed_is_the_only_writer_of_the_marker(store: MemoryStore) -> None:
    ids = _seed_labeled_rows(store, 2)
    now = datetime(2026, 9, 24, tzinfo=UTC)
    store.mark_selftune_consumed(ids, consumed_at=now)
    rows = store._conn.execute(
        "SELECT id, selftune_consumed_at FROM calibration_log ORDER BY id"
    ).fetchall()
    assert [r["selftune_consumed_at"] for r in rows] == [now.isoformat(), now.isoformat()]


def test_mark_selftune_consumed_empty_list_is_a_noop(store: MemoryStore) -> None:
    ids = _seed_labeled_rows(store, 2)
    store.mark_selftune_consumed([], consumed_at=datetime.now(UTC))
    count, row_ids = store.count_new_haiku_decisions()
    assert count == 2
    assert sorted(row_ids) == sorted(ids)


def test_consume_once_a_row_consumed_in_week_one_is_not_recounted_in_week_two(
    store: MemoryStore,
) -> None:
    """AC3 / spec §2 consume-once contract: once a row is marked consumed,
    it must never contribute to a later gate count again, however many
    times `count_new_haiku_decisions` is subsequently called."""
    ids = _seed_labeled_rows(store, 5)
    week1_count, week1_ids = store.count_new_haiku_decisions()
    assert week1_count == 5
    store.mark_selftune_consumed(week1_ids, consumed_at=datetime.now(UTC))

    # Week 2: no new rows logged at all — everything from week 1 is consumed.
    week2_count, week2_ids = store.count_new_haiku_decisions()
    assert week2_count == 0
    assert week2_ids == []

    # Week 3: one genuinely new row arrives — only IT counts, not the 5 already consumed.
    new_id = _seed_row_with_haiku_labels(store, ["relevant"])
    week3_count, week3_ids = store.count_new_haiku_decisions()
    assert week3_count == 1
    assert week3_ids == [new_id]
    assert new_id not in ids, "sanity: the new row is distinct from the consumed ones"


# ---------------------------------------------------------------------------
# MemoryStore.get_judge_selftune_state / write_judge_selftune_state — marker
# upsert/read round-trip (I1: table in memories.db). Red-team fix F-1: this
# marker no longer carries a `consumed_through_id` cursor (see
# `calibration_log.selftune_consumed_at` above) — it is purely the
# cadence-adjacent "last trained" record now.
# ---------------------------------------------------------------------------


def test_judge_selftune_state_absent_reads_as_none(store: MemoryStore) -> None:
    assert store.get_judge_selftune_state("some-model") is None


def test_judge_selftune_state_round_trip(store: MemoryStore) -> None:
    now = datetime(2026, 9, 23, 12, tzinfo=UTC)
    store.write_judge_selftune_state("model-a", last_trained_at=now)
    state = store.get_judge_selftune_state("model-a")
    assert state is not None
    assert state["judge_model_id"] == "model-a"
    assert state["last_trained_at"] == now.isoformat()


def test_judge_selftune_state_upsert_replaces_not_accumulates(store: MemoryStore) -> None:
    now1 = datetime(2026, 9, 1, tzinfo=UTC)
    now2 = datetime(2026, 9, 8, tzinfo=UTC)
    store.write_judge_selftune_state("model-a", last_trained_at=now1)
    store.write_judge_selftune_state("model-a", last_trained_at=now2)
    state = store.get_judge_selftune_state("model-a")
    assert state["last_trained_at"] == now2.isoformat()
    n_rows = store._conn.execute("SELECT COUNT(*) AS n FROM judge_selftune_state").fetchone()["n"]
    assert n_rows == 1, "upsert must replace, never accumulate a history row"


def test_judge_selftune_state_scoped_per_judge_model_id(store: MemoryStore) -> None:
    """No cross-model bleed: a second judge_model_id's row must not be
    visible under the first's key (mirrors reranker_floor_calibration's own
    model_id keying)."""
    now = datetime(2026, 9, 23, tzinfo=UTC)
    store.write_judge_selftune_state("model-a", last_trained_at=now)
    assert store.get_judge_selftune_state("model-b") is None


# ---------------------------------------------------------------------------
# MemoryStore.judge_knob_refit_pairs — F2c inc3's data-assembly step (spec
# §3-5, AC4): reuses labeled_calibration_pairs's exact effective-label
# precedence + unknown/error skip, but reads local_judge_raw_score (the bge
# JUDGE's own score) rather than reranker_scores (jina's).
# ---------------------------------------------------------------------------


def test_judge_knob_refit_pairs_uses_local_label_when_no_haiku_override(store: MemoryStore) -> None:
    ids = _seed_labeled_rows(store, 1, raw_score=3.0)
    # _seed_labeled_rows writes haiku_label == local_judge_label ("relevant"
    # on both) -- exercise the "no override" path with an explicit local-
    # only row instead.
    store.log_calibration_sample(
        query="q", candidate_ids=["a", "b"], reranker_scores=[9.0, 9.0], reranker_model_id="m"
    )
    row_id = store._conn.execute("SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1").fetchone()["id"]
    store.write_calibration_labels(
        row_id, ["relevant", "irrelevant"], [None, None], local_judge_raw_score=[1.0, 2.0]
    )
    pairs = store.judge_knob_refit_pairs([row_id])
    assert sorted(pairs) == sorted([(1.0, "relevant"), (2.0, "irrelevant")])
    assert ids  # sanity: the unrelated seeded row exists and is NOT in this scan


def test_judge_knob_refit_pairs_haiku_label_overrides_local_label(store: MemoryStore) -> None:
    store.log_calibration_sample(
        query="q", candidate_ids=["a"], reranker_scores=[9.0], reranker_model_id="m"
    )
    row_id = store._conn.execute("SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1").fetchone()["id"]
    # Local judge said "relevant" (ambiguous-band); Haiku overrode to "irrelevant".
    store.write_calibration_labels(row_id, ["relevant"], ["irrelevant"], local_judge_raw_score=[1.0])
    assert store.judge_knob_refit_pairs([row_id]) == [(1.0, "irrelevant")]


def test_judge_knob_refit_pairs_skips_unknown_and_error_sentinels(store: MemoryStore) -> None:
    store.log_calibration_sample(
        query="q", candidate_ids=["a", "b", "c"], reranker_scores=[9.0, 9.0, 9.0], reranker_model_id="m"
    )
    row_id = store._conn.execute("SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1").fetchone()["id"]
    store.write_calibration_labels(
        row_id, ["relevant", "unknown", "error"], [None, None, None],
        local_judge_raw_score=[1.0, None, None],
    )
    assert store.judge_knob_refit_pairs([row_id]) == [(1.0, "relevant")]


def test_judge_knob_refit_pairs_skips_legacy_rows_with_no_raw_score_at_all(store: MemoryStore) -> None:
    """A pre-F2c-inc1 row (never had local_judge_raw_score written) must be
    skipped outright, not treated as a 0.0 score."""
    ids = _seed_labeled_rows(store, 1, raw_score=None)
    assert store.judge_knob_refit_pairs(ids) == []


def test_judge_knob_refit_pairs_scoped_to_the_given_row_ids_only(store: MemoryStore) -> None:
    """Only the passed-in row_ids are read -- an unrelated labeled row with
    a real raw score must NOT leak in, mirroring the "exact set the gate
    considered" contract (spec §2)."""
    in_scope = _seed_labeled_rows(store, 1, raw_score=5.0)
    _seed_labeled_rows(store, 1, raw_score=9.0)  # NOT passed to judge_knob_refit_pairs
    pairs = store.judge_knob_refit_pairs(in_scope)
    assert pairs == [(5.0, "relevant")]


def test_judge_knob_refit_pairs_empty_row_ids_returns_empty(store: MemoryStore) -> None:
    assert store.judge_knob_refit_pairs([]) == []


# ---------------------------------------------------------------------------
# MemoryStore.get_judge_knob_calibration / write_judge_knob_calibration —
# marker round-trip (I1: table in memories.db), mirrors
# get_judge_selftune_state / write_judge_selftune_state's own tests above.
# ---------------------------------------------------------------------------


def test_judge_knob_calibration_absent_reads_as_none(store: MemoryStore) -> None:
    assert store.get_judge_knob_calibration("some-model") is None


def test_judge_knob_calibration_round_trip(store: MemoryStore) -> None:
    store.write_judge_knob_calibration("model-a", slope=1.7, intercept=-0.3)
    knob = store.get_judge_knob_calibration("model-a")
    assert knob is not None
    assert knob["judge_model_id"] == "model-a"
    assert knob["slope"] == pytest.approx(1.7)
    assert knob["intercept"] == pytest.approx(-0.3)


def test_judge_knob_calibration_upsert_replaces_not_accumulates(store: MemoryStore) -> None:
    store.write_judge_knob_calibration("model-a", slope=1.0, intercept=0.0)
    store.write_judge_knob_calibration("model-a", slope=2.0, intercept=0.5)
    knob = store.get_judge_knob_calibration("model-a")
    assert knob["slope"] == pytest.approx(2.0)
    assert knob["intercept"] == pytest.approx(0.5)
    n_rows = store._conn.execute("SELECT COUNT(*) AS n FROM judge_knob_calibration").fetchone()["n"]
    assert n_rows == 1, "upsert must replace, never accumulate a history row"


def test_judge_knob_calibration_scoped_per_judge_model_id(store: MemoryStore) -> None:
    """No cross-model bleed within one persona's own db, mirroring
    judge_selftune_state's own model_id keying."""
    store.write_judge_knob_calibration("model-a", slope=1.0, intercept=0.0)
    assert store.get_judge_knob_calibration("model-b") is None


def test_judge_knob_calibration_per_persona_store_isolation() -> None:
    """AC11 (per-persona weight isolation): two DIFFERENT personas' stores
    (each its own memories.db, I1) must never read or overwrite each
    other's fitted knob -- the database FILE is the isolation boundary."""
    store_a = MemoryStore(db_path=":memory:")
    store_b = MemoryStore(db_path=":memory:")
    store_a.write_judge_knob_calibration("shared-model-id", slope=9.0, intercept=9.0)
    assert store_b.get_judge_knob_calibration("shared-model-id") is None, (
        "persona B must not see persona A's fitted knob, even under the same judge_model_id"
    )


# ---------------------------------------------------------------------------
# _run_judge_selftune_tick — the full scaffold: gate + tier-select +
# OOM-downgrade + marker write, and fault isolation.
# ---------------------------------------------------------------------------


def test_run_judge_selftune_tick_importable_and_callable() -> None:
    sig = inspect.signature(judge_selftune._run_judge_selftune_tick)
    params = list(sig.parameters)
    assert params == ["store", "now"]
    for name in params:
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_tick_does_not_fire_at_exactly_the_handful_threshold(store: MemoryStore) -> None:
    """AC3 (gate): NOT more than a handful -> no fire. Exactly at the
    threshold is the boundary case (`> handful`, not `>=`)."""
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    _seed_labeled_rows(store, handful)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert result["fired"] is False
    assert result["new_decisions"] == handful
    assert result["error"] is None
    assert store.get_judge_selftune_state(MODEL_RELEVANCE_JUDGE) is None, "marker untouched on no-fire"


def test_tick_fires_above_the_handful_threshold(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3 (gate): MORE than a handful -> fires, the marker records the
    firing timestamp, and every scanned row is marked consumed (red-team
    fix F-1: per-row consumed marker, not a MAX(id) cursor)."""
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    ids = _seed_labeled_rows(store, handful + 1)
    monkeypatch.setattr(judge_selftune, "_read_total_ram_bytes", lambda: 8.0 * 1024**3)  # weak box
    monkeypatch.setattr(judge_selftune, "_available_ram_headroom_bytes", lambda: None)
    now = datetime.now(UTC)

    result = judge_selftune._run_judge_selftune_tick(store=store, now=now)

    assert result["fired"] is True
    assert result["error"] is None
    assert result["tune_grade"] == judge_selftune.TUNE_GRADE_KNOB_REFIT
    assert result["new_decisions"] == handful + 1
    marker = store.get_judge_selftune_state(MODEL_RELEVANCE_JUDGE)
    assert marker is not None
    assert marker["last_trained_at"] == now.isoformat()
    # F2c inc3: a completed knob-refit must have persisted fitted params.
    knob = store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE)
    assert knob is not None
    # every row this tick scanned must now be consumed — a second count sees nothing.
    post_count, post_row_ids = store.count_new_haiku_decisions()
    assert post_count == 0
    assert post_row_ids == []
    consumed_rows = store._conn.execute(
        "SELECT id FROM calibration_log WHERE selftune_consumed_at IS NOT NULL"
    ).fetchall()
    assert sorted(r["id"] for r in consumed_rows) == sorted(ids)


def test_tick_second_fire_only_counts_rows_after_prior_consumed_cursor(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is re-evaluated against UNCONSUMED rows each tick, not
    against the corpus total — a second consecutive fire without enough
    genuinely NEW (unconsumed) rows must not fire again, and the first
    fire's rows must not be double-counted (consume-once, F-1)."""
    monkeypatch.setattr(judge_selftune, "_read_total_ram_bytes", lambda: 8.0 * 1024**3)
    monkeypatch.setattr(judge_selftune, "_available_ram_headroom_bytes", lambda: None)
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    _seed_labeled_rows(store, handful + 1)

    r1 = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert r1["fired"] is True

    _seed_labeled_rows(store, handful)  # exactly `handful` NEW rows -> not more than a handful
    r2 = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert r2["fired"] is False
    assert r2["new_decisions"] == handful, (
        "must count only the genuinely NEW unconsumed rows, not re-count week 1's"
    )


def test_tick_selects_tier_and_applies_oom_downgrade(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end scaffold BITE: a beefy host-total RAM would naively pick
    full_ft, but a cgroup-capped effective headroom that only fits LoRA
    must downgrade the tick's actual selection."""
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    _seed_labeled_rows(store, handful + 1)
    monkeypatch.setattr(
        judge_selftune, "_read_total_ram_bytes",
        lambda: judge_selftune.JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES + 1.0,
    )
    monkeypatch.setattr(
        judge_selftune, "_available_ram_headroom_bytes",
        lambda: judge_selftune.JUDGE_TUNE_FOOTPRINT_LORA_BYTES + 1.0,
    )

    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))

    assert result["fired"] is True
    assert result["tune_grade"] == judge_selftune.TUNE_GRADE_LORA, "downgraded from full_ft"


def test_tick_is_fault_isolated_never_raises(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "count_new_haiku_decisions", _boom)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert result["error"] is not None
    assert "disk full" in result["error"]
    assert result["fired"] is False


# ---------------------------------------------------------------------------
# AC3 new bite (F2c inc3, spec §2 "consume = trained-on, never fired-on"):
# a tick that fires the >handful GATE but does NOT complete a tune must
# leave the consume marker UNADVANCED, so those rows are still countable
# next week -- the actual behavior change inc3 makes over inc2's
# "consume unconditionally on every fire" scaffold.
# ---------------------------------------------------------------------------


def test_tick_gate_fires_but_no_usable_pairs_leaves_rows_unconsumed(store: MemoryStore) -> None:
    """Scripted 'no-train' case: every seeded row is judge-labeled and has
    a non-None haiku_label position (the gate fires, `new_decisions` is
    correctly counted), but NONE carries a `local_judge_raw_score` (as a
    week of purely legacy/pre-inc1 rows would look) -- `judge_knob_refit_
    pairs` returns `[]`, `fit_platt_knob` raises, and the tick's own
    try/except must catch that and leave every row unconsumed rather than
    silently losing the Haiku signal these rows carry."""
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    ids = _seed_labeled_rows(store, handful + 1, raw_score=None)

    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))

    assert result["fired"] is False, "the gate fired but the tune never completed -- must not report fired"
    assert result["error"] is not None
    assert result["new_decisions"] == handful + 1, "the gate DID see enough decisions to have fired"
    assert store.get_judge_selftune_state(MODEL_RELEVANCE_JUDGE) is None, "marker must stay untouched"
    assert store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE) is None, "no knob was ever fitted"

    # the rows must still be there, UNCONSUMED, for next week's gate to recount.
    post_count, post_row_ids = store.count_new_haiku_decisions()
    assert post_count == handful + 1
    assert sorted(post_row_ids) == sorted(ids)


def test_tick_fault_during_fit_leaves_rows_unconsumed(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same contract: real usable pairs exist (the
    fault is injected directly into the fit step itself, mirroring
    `test_tick_is_fault_isolated_never_raises`'s injection style), and the
    fault must still leave the rows unconsumed -- not just the "zero pairs"
    degenerate path above."""
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    ids = _seed_labeled_rows(store, handful + 1, raw_score=2.0)

    def _boom(pairs):
        raise RuntimeError("simulated fault mid-fit")

    monkeypatch.setattr(judge_selftune, "fit_platt_knob", _boom)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))

    assert result["fired"] is False
    assert "simulated fault mid-fit" in result["error"]
    assert store.get_judge_selftune_state(MODEL_RELEVANCE_JUDGE) is None
    assert store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE) is None
    post_count, post_row_ids = store.count_new_haiku_decisions()
    assert post_count == handful + 1
    assert sorted(post_row_ids) == sorted(ids)


def test_tick_does_not_import_torch_or_sentence_transformers() -> None:
    """I6 / AC8 (off hot path): the scaffold, INCLUDING a FIRING tick that
    actually runs F2c inc3's real knob-refit training code (assembly +
    `fit_platt_knob` + persist), must never pull torch/sentence_transformers
    into `sys.modules` — that stays scoped to the future LoRA/full-FT
    tiers (inc5/6), never to the knob-refit this increment ships.

    Red-team fix F-3 (inc2, still honored here): the prior version of this
    test grepped `judge_selftune`'s own source TEXT for the literal strings
    "import torch" / "import sentence_transformers". That only proves this
    one module has no such import statement in it — it would NOT catch a
    TRANSITIVE pull via some other module `judge_selftune` imports (e.g. a
    future change to `brain.memory.reranker` or `brain.memory.store`
    growing a module-scope torch import). This runs a FRESH subprocess —
    never inheriting whatever this test PROCESS's own earlier tests may
    already have imported into `sys.modules`, which would make an
    in-process `sys.modules` check meaningless (the same caveat
    `test_relevance_judge.py`'s sibling import-scope test documents) — that
    imports `judge_selftune`, seeds enough `calibration_log` rows (WITH raw
    judge scores, so the knob-refit has real pairs to fit — inc3's fire
    path now actually trains, not just consumes) to FIRE the >handful
    gate, actually runs `_run_judge_selftune_tick` end to end, and only
    THEN asserts neither package landed in `sys.modules`.
    """
    script = textwrap.dedent(
        """
        import sys
        from datetime import UTC, datetime

        from brain.memory import judge_selftune
        from brain.memory.store import MemoryStore

        store = MemoryStore(db_path=":memory:")
        handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
        for i in range(handful + 1):
            store.log_calibration_sample(
                query=f"q{i}",
                candidate_ids=["m"],
                reranker_scores=[1.0],
                reranker_model_id="m",
            )
            row_id = store._conn.execute(
                "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
            store.write_calibration_labels(
                row_id, ["relevant"], ["relevant"], local_judge_raw_score=[2.0]
            )

        result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
        assert result["fired"] is True, result
        assert store.get_judge_knob_calibration(judge_selftune.MODEL_RELEVANCE_JUDGE) is not None

        assert "torch" not in sys.modules, sorted(sys.modules)
        assert "sentence_transformers" not in sys.modules, sorted(sys.modules)
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


# ---------------------------------------------------------------------------
# AC9: the durable Haiku-oracle note must be present in F2c's OWN training
# code location (relevance_judge.py's own copy is covered by that module's
# test file) — grep/lint check, not a behavior test.
# ---------------------------------------------------------------------------


def test_haiku_oracle_note_is_present_in_judge_selftune_source() -> None:
    source = inspect.getsource(judge_selftune)
    marker = "# F2c (durable note, spec"
    assert marker in source, "the durable Haiku-oracle note must precede _run_judge_selftune_tick"
    note_start = source.index(marker)
    note_end = source.index("def _run_judge_selftune_tick", note_start)
    note = source[note_start:note_end]
    assert "oracle" in note.lower()
    assert "haiku" in note.lower()
    assert "—" not in note, "no em-dashes in this durable note (plain code comment, no LLM-tells)"


# ---------------------------------------------------------------------------
# Tunables (I3/I7): every threshold this module uses is registered, not a
# bare constant — spot-check the registry has them under the right keys.
# ---------------------------------------------------------------------------


def test_every_judge_selftune_threshold_is_a_registered_tunable() -> None:
    for key in (
        "judge_selftune.gate_handful_decisions",
        "judge_selftune.ram_tier_lora_min_bytes",
        "judge_selftune.ram_tier_full_ft_min_bytes",
        "judge_selftune.footprint_knob_refit_bytes",
        "judge_selftune.footprint_lora_bytes",
        "judge_selftune.footprint_full_ft_bytes",
    ):
        assert key in judge_selftune.tunables._registry, f"{key} must be tunables.register()-ed"


# ---------------------------------------------------------------------------
# Cadence-file constant wiring (round-trip via persisted_cadence, generic
# is_due/advance/save mechanics already covered by persisted_cadence's own
# tests — this just pins that judge_selftune's OWN constants plug in).
# ---------------------------------------------------------------------------


def test_judge_selftune_cadence_due_now_when_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 29, 12, tzinfo=UTC)
        state = pc.load_cadence(pd, judge_selftune.JUDGE_TUNE_CADENCE_FILE)
        assert pc.is_due(state, now=now) is True


def test_judge_selftune_cadence_not_due_after_advance_and_fires_at_interval() -> None:
    now = datetime(2026, 6, 29, 12, tzinfo=UTC)
    interval_s = judge_selftune.JUDGE_TUNE_INTERVAL_HOURS * 3600.0
    state = pc.advance(now=now, interval_s=interval_s)
    assert pc.is_due(state, now=now) is False
    assert pc.is_due(state, now=now + timedelta(seconds=interval_s)) is True


def test_judge_selftune_cadence_save_load_round_trip() -> None:
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 29, 12, tzinfo=UTC)
        state = pc.advance(now=now, interval_s=100.0)
        pc.save_cadence(pd, judge_selftune.JUDGE_TUNE_CADENCE_FILE, state)
        loaded = pc.load_cadence(pd, judge_selftune.JUDGE_TUNE_CADENCE_FILE)
        assert loaded.next_at == state.next_at
        assert not list(pd.glob("*.tmp")), "atomic write must leave no .tmp"


def test_run_folded_accepts_judge_selftune_interval_s() -> None:
    from brain.bridge.supervisor import run_folded

    sig = inspect.signature(run_folded)
    assert "judge_selftune_interval_s" in sig.parameters
    param = sig.parameters["judge_selftune_interval_s"]
    assert param.default == judge_selftune.JUDGE_TUNE_INTERVAL_HOURS * 3600.0


# ---------------------------------------------------------------------------
# fit_platt_knob — F2c inc3's knob-refit fit itself (spec §5, AC2/AC8): a
# deterministic, torch-free, best-separating Platt slope+intercept fit.
# ---------------------------------------------------------------------------


def test_fit_platt_knob_empty_pairs_raises() -> None:
    with pytest.raises(ValueError, match="at least one labeled pair"):
        judge_selftune.fit_platt_knob([])


def test_fit_platt_knob_single_class_returns_identity_mapping() -> None:
    """No separation exists to fit -- falls back to the identity mapping
    (slope=1.0, intercept=0.0), `label_for_score`'s own fixed default,
    mirroring `fit_threshold_fbeta`'s degenerate-input posture rather than
    fitting a meaningless direction from zero contrast."""
    pairs = [(1.0, "relevant"), (2.0, "relevant"), (3.0, "relevant")]
    slope, intercept = judge_selftune.fit_platt_knob(pairs)
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0)


def test_fit_platt_knob_is_deterministic() -> None:
    """AC2/AC8: the SAME pairs must always produce the SAME fitted
    params -- no randomness anywhere in the fit."""
    pairs = [
        (-5.0, "irrelevant"), (-4.0, "irrelevant"), (-3.0, "irrelevant"),
        (3.0, "relevant"), (4.0, "relevant"), (5.0, "relevant"),
    ]
    first = judge_selftune.fit_platt_knob(pairs)
    second = judge_selftune.fit_platt_knob(list(pairs))  # fresh list, same contents
    assert first == second


def test_fit_platt_knob_order_invariant() -> None:
    """The fit is a sum over pairs (order-invariant) -- shuffling the
    input must not change the result."""
    pairs = [
        (-5.0, "irrelevant"), (-4.0, "irrelevant"), (-3.0, "irrelevant"),
        (3.0, "relevant"), (4.0, "relevant"), (5.0, "relevant"),
    ]
    forward = judge_selftune.fit_platt_knob(pairs)
    reversed_order = judge_selftune.fit_platt_knob(list(reversed(pairs)))
    assert forward[0] == pytest.approx(reversed_order[0], abs=1e-9)
    assert forward[1] == pytest.approx(reversed_order[1], abs=1e-9)


def test_fit_platt_knob_best_separates_and_bites_on_label_for_score() -> None:
    """AC2's actual bite: the fitted mapping separates a lopsided sample
    (relevant scores clustered near 1.0, irrelevant scores clustered near
    -3.0 -- an asymmetric split, unlike the fixed sigmoid's boundary at
    raw_score=0.0) and, once applied through `label_for_score`, CHANGES
    the label on a case whose score crossed the new cutoff versus the
    fixed default."""
    pairs = (
        [(1.0, "relevant")] * 5
        + [(-3.0, "irrelevant")] * 5
    )
    slope, intercept = judge_selftune.fit_platt_knob(pairs)

    # The two clusters must separate correctly under the fitted mapping.
    from brain.memory.relevance_judge import label_for_score

    for score, expected in pairs:
        label, _ = label_for_score(score, slope=slope, intercept=intercept)
        assert label == expected, f"fitted mapping must correctly classify its own training pair {score}"

    # BITE: a probe score between the two clusters, closer to the
    # irrelevant cluster, is "irrelevant" under the fixed default
    # (raw_score=-0.5 < 0) but "relevant" under the fitted (shifted
    # toward the lopsided data's true midpoint near -1.0) mapping.
    probe = -0.5
    fixed_label, _ = label_for_score(probe)
    fitted_label, _ = label_for_score(probe, slope=slope, intercept=intercept)
    assert fixed_label == "irrelevant", "sanity: the fixed default puts -0.5 on the irrelevant side"
    assert fitted_label != fixed_label, "the fitted knob must have shifted the cutoff past this probe score"


def test_fit_platt_knob_symmetric_balanced_data_stays_near_identity() -> None:
    """A dataset already well-separated by the EXISTING fixed boundary
    (symmetric around raw_score=0, balanced classes) should fit params
    close to the identity mapping -- the refit generalizes the existing
    knob, it should not gratuitously distort a sample the fixed default
    already handles well."""
    pairs = (
        [(-5.0, "irrelevant"), (-4.0, "irrelevant"), (-3.0, "irrelevant")]
        + [(3.0, "relevant"), (4.0, "relevant"), (5.0, "relevant")]
    )
    slope, intercept = judge_selftune.fit_platt_knob(pairs)
    assert slope > 0.0, "higher raw score must still mean more likely relevant"
    assert abs(intercept) < 1.0, "a symmetric/balanced sample should not push the boundary far off zero"


def test_fit_platt_knob_anti_correlated_data_falls_back_to_identity_not_negative_slope() -> None:
    """Monotonicity guard (Opus cold-review, LOW) BITE: on ANTI-correlated
    (raw_score, effective_label) pairs -- the exact mirror image of
    `test_fit_platt_knob_symmetric_balanced_data_stays_near_identity`'s
    data, with the two labels swapped, so HIGH raw scores are labeled
    "irrelevant" and LOW raw scores "relevant" -- unconstrained
    Newton-Raphson fits a NEGATIVE slope here (confirmed by running this
    exact algorithm without the guard: ~-0.338, the sign-flipped mirror of
    the symmetric test's ~+0.338). The judge's raw score is
    positively-correlated-with-relevance BY CONSTRUCTION, so a negative
    slope would INVERT the judge's relevance direction -- never a valid
    refit. The guard must catch this and return the identity mapping
    instead."""
    pairs = (
        [(-5.0, "relevant"), (-4.0, "relevant"), (-3.0, "relevant")]
        + [(3.0, "irrelevant"), (4.0, "irrelevant"), (5.0, "irrelevant")]
    )
    slope, intercept = judge_selftune.fit_platt_knob(pairs)
    assert slope == pytest.approx(1.0), "must fall back to the identity slope, never a negative one"
    assert intercept == pytest.approx(0.0)


def test_fit_platt_knob_does_not_import_torch() -> None:
    """AC8: the fit function itself, exercised directly (not only via the
    tick), must never import torch -- pure numpy only."""
    script = textwrap.dedent(
        """
        import sys
        from brain.memory import judge_selftune

        pairs = [(-3.0, "irrelevant"), (3.0, "relevant")]
        slope, intercept = judge_selftune.fit_platt_knob(pairs)
        assert isinstance(slope, float) and isinstance(intercept, float)
        assert "torch" not in sys.modules, sorted(sys.modules)
        assert "sentence_transformers" not in sys.modules, sorted(sys.modules)
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
