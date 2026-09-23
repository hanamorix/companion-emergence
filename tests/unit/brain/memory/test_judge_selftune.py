"""Tests for brain.memory.judge_selftune — F2c inc2's OFFLINE weekly-tick
SCAFFOLD (spec `f2c-judge-selftune-spec.md` §1 [hardware-tiered mechanism],
§2 [cadence + gate], §7 [scope]).

Covers what inc2 actually builds: the >handful gate, runtime RAM
tier-detection, the cgroup-aware OOM-safety downgrade, the per-persona
consumed-cursor marker (round-trip via `MemoryStore`), and
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
import tempfile
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


def _seed_labeled_rows(store: MemoryStore, n: int) -> list[int]:
    """Insert `n` already judge-labeled `calibration_log` rows (`haiku_label`
    non-null, mirrors `write_calibration_labels`'s always-write contract —
    see that method's own docstring) and return their ids in insertion
    order."""
    ids: list[int] = []
    for i in range(n):
        store.log_calibration_sample(
            query=f"q{i}", candidate_ids=["m"], reranker_scores=[1.0], reranker_model_id="m"
        )
        row_id = store._conn.execute(
            "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
        ).fetchone()["id"]
        store.write_calibration_labels(row_id, ["relevant"], ["relevant"])
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
# MemoryStore.count_new_haiku_decisions — the >handful gate's data source
# (spec §2/§3).
# ---------------------------------------------------------------------------


def test_count_new_haiku_decisions_counts_labeled_rows_after_cursor(store: MemoryStore) -> None:
    _seed_labeled_rows(store, 3)
    count, max_id = store.count_new_haiku_decisions(0)
    assert count == 3
    assert max_id == 3


def test_count_new_haiku_decisions_excludes_unlabeled_rows(store: MemoryStore) -> None:
    store.log_calibration_sample(
        query="q", candidate_ids=[], reranker_scores=[], reranker_model_id="m"
    )
    count, max_id = store.count_new_haiku_decisions(0)
    assert count == 0
    assert max_id is None


def test_count_new_haiku_decisions_respects_since_id_cursor(store: MemoryStore) -> None:
    ids = _seed_labeled_rows(store, 5)
    count, max_id = store.count_new_haiku_decisions(ids[2])
    assert count == 2
    assert max_id == ids[-1]


def test_count_new_haiku_decisions_survives_retention_pruning(store: MemoryStore) -> None:
    """Cursor semantics compose with calibration_log's own rolling
    retention: id is AUTOINCREMENT/monotonic, so deleting an OLD row below
    the cursor can only shrink the count, never corrupt the `> since_id`
    comparison — pre-checked per the BUILD instructions' flag-back
    criterion, not a design fork (see the method's own docstring)."""
    ids = _seed_labeled_rows(store, 3)
    store._conn.execute("DELETE FROM calibration_log WHERE id = ?", (ids[0],))
    store._conn.commit()
    count, max_id = store.count_new_haiku_decisions(0)
    assert count == 2
    assert max_id == ids[-1]


# ---------------------------------------------------------------------------
# MemoryStore.get_judge_selftune_state / write_judge_selftune_state — marker
# upsert/read round-trip (I1: table in memories.db).
# ---------------------------------------------------------------------------


def test_judge_selftune_state_absent_reads_as_none(store: MemoryStore) -> None:
    assert store.get_judge_selftune_state("some-model") is None


def test_judge_selftune_state_round_trip(store: MemoryStore) -> None:
    now = datetime(2026, 9, 23, 12, tzinfo=UTC)
    store.write_judge_selftune_state("model-a", last_trained_at=now, consumed_through_id=42)
    state = store.get_judge_selftune_state("model-a")
    assert state is not None
    assert state["judge_model_id"] == "model-a"
    assert state["consumed_through_id"] == 42
    assert state["last_trained_at"] == now.isoformat()


def test_judge_selftune_state_upsert_replaces_not_accumulates(store: MemoryStore) -> None:
    now1 = datetime(2026, 9, 1, tzinfo=UTC)
    now2 = datetime(2026, 9, 8, tzinfo=UTC)
    store.write_judge_selftune_state("model-a", last_trained_at=now1, consumed_through_id=10)
    store.write_judge_selftune_state("model-a", last_trained_at=now2, consumed_through_id=25)
    state = store.get_judge_selftune_state("model-a")
    assert state["consumed_through_id"] == 25
    assert state["last_trained_at"] == now2.isoformat()
    n_rows = store._conn.execute("SELECT COUNT(*) AS n FROM judge_selftune_state").fetchone()["n"]
    assert n_rows == 1, "upsert must replace, never accumulate a history row"


def test_judge_selftune_state_scoped_per_judge_model_id(store: MemoryStore) -> None:
    """No cross-model bleed: a second judge_model_id's row must not be
    visible under the first's key (mirrors reranker_floor_calibration's own
    model_id keying)."""
    now = datetime(2026, 9, 23, tzinfo=UTC)
    store.write_judge_selftune_state("model-a", last_trained_at=now, consumed_through_id=5)
    assert store.get_judge_selftune_state("model-b") is None


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
    """AC3 (gate): MORE than a handful -> fires, and the marker's consumed
    cursor advances to the newest counted row."""
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
    assert marker["consumed_through_id"] == ids[-1]
    assert marker["last_trained_at"] == now.isoformat()


def test_tick_second_fire_only_counts_rows_after_prior_consumed_cursor(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is re-evaluated against the marker's own consumed cursor
    each tick, not against the corpus total — a second consecutive fire
    without enough genuinely NEW rows must not fire again."""
    monkeypatch.setattr(judge_selftune, "_read_total_ram_bytes", lambda: 8.0 * 1024**3)
    monkeypatch.setattr(judge_selftune, "_available_ram_headroom_bytes", lambda: None)
    handful = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS
    _seed_labeled_rows(store, handful + 1)

    r1 = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert r1["fired"] is True

    _seed_labeled_rows(store, handful)  # exactly `handful` NEW rows -> not more than a handful
    r2 = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert r2["fired"] is False
    assert r2["new_decisions"] == handful


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


def test_tick_does_not_import_torch_or_sentence_transformers() -> None:
    """I6 / AC8 (off hot path): the scaffold itself must never pull in
    torch/sentence_transformers — that stays scoped to the future TODO(F2c
    inc3+) tuning code, never to the cadence/gate/tier-detect machinery
    this increment ships."""
    source = inspect.getsource(judge_selftune)
    assert "import torch" not in source
    assert "import sentence_transformers" not in source
    assert "from sentence_transformers" not in source


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
