"""F2c inc8 — rolling one-week retention for Haiku-decision `calibration_log`
rows (spec `f2c-judge-selftune-spec.md` §3 "Retention vs the weekly tick",
AC14; AC10 F2a unchanged).

Owner rule: rows that carry a Haiku decision are kept on a rolling one-week
cutoff; the weekly self-tune counts its gate and trains only on the past
week's Haiku decisions; nothing is kept past a week; the accumulated rows are
cleared once the weekly self-tune has trained on them; every other row keeps
F2a's 3-day window; F2a's own calibration inputs are unchanged.

All rows are synthetic and written with explicit `logged_at` / `day_bucket`
ages relative to a fixed `NOW`; no real model, never a live persona (I11).
Criteria ids (R1..R15) refer to
`changes/f2c-inc8-rolling-week-retention/1.5-criteria.md`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE
from brain.memory import floor_calibration, judge_selftune
from brain.memory.store import MemoryStore
from tests.unit.brain.memory.test_judge_selftune_lifecycle import (
    Env,
    _consumed,
    _seed_scenario,
    _set_tier,
    _small_tunables,
)

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
HANDFUL = judge_selftune.JUDGE_TUNE_GATE_HANDFUL_DECISIONS


def _week() -> timedelta:
    return timedelta(hours=judge_selftune.JUDGE_TUNE_INTERVAL_HOURS)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


def _seed(
    store: MemoryStore,
    age: timedelta,
    *,
    haiku: str = "decision",
    consumed: bool = False,
    label: str = "relevant",
    model: str = "m",
    n_candidates: int = 1,
    now: datetime = NOW,
) -> int:
    """Insert one row logged `age` before `now`. `haiku`: "decision" (one
    non-None Haiku position), "all_none" (judge-labeled, no tie-break) or
    "unlabeled" (no labels at all)."""
    ts = now - age
    ids = [f"c{i}" for i in range(n_candidates)]
    scores = [float(i % 7) for i in range(n_candidates)]
    labels = [("relevant" if i % 2 == 0 else "irrelevant") for i in range(n_candidates)]
    labels[0] = label
    raw = [2.0 if lab == "relevant" else -2.0 for lab in labels]
    if haiku == "unlabeled":
        local, hl, raw_json = None, None, None
    else:
        local = json.dumps(labels)
        h = [None] * n_candidates
        if haiku == "decision":
            h[0] = label
        hl = json.dumps(h)
        raw_json = json.dumps(raw)
    cur = store._conn.execute(
        "INSERT INTO calibration_log (logged_at, day_bucket, query, candidate_ids, reranker_scores, "
        "reranker_model_id, local_judge_label, haiku_label, score_scale, local_judge_raw_score, "
        "candidate_docs, selftune_consumed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'normalized', ?, ?, ?)",
        (
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            ts.strftime("%Y-%m-%d"),
            f"q-{age}-{haiku}",
            json.dumps(ids),
            json.dumps(scores),
            model,
            local,
            hl,
            raw_json,
            json.dumps([f"doc-{i}" for i in ids]),
            ts.isoformat() if consumed else None,
        ),
    )
    store._conn.commit()
    return int(cur.lastrowid)


def _seed_many(store: MemoryStore, n: int, age: timedelta) -> list[int]:
    return [
        _seed(store, age, label=("relevant" if i % 2 == 0 else "irrelevant")) for i in range(n)
    ]


def _ids(store: MemoryStore) -> set[int]:
    return {int(r["id"]) for r in store._conn.execute("SELECT id FROM calibration_log").fetchall()}


def _tick(store: MemoryStore, now: datetime = NOW) -> dict:
    return judge_selftune._run_judge_selftune_tick(store=store, now=now)


# ---------------------------------------------------------------------------
# R1 — a 5-day-old Haiku decision survives the daily prune, is counted by the
# weekly gate and is trained on.
# ---------------------------------------------------------------------------


def test_r1_five_day_old_haiku_decisions_survive_prune_are_counted_and_trained_on(store) -> None:
    n = HANDFUL + 5
    ids = _seed_many(store, n, timedelta(days=5))

    store.prune_calibration_log(now=NOW)
    assert set(ids) <= _ids(store), "5-day-old Haiku rows must survive F2a's daily 3-day prune"

    result = _tick(store)
    assert result["error"] is None
    assert result["fired"] is True
    assert result["new_decisions"] == n
    assert store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE) is not None, "knob-refit trained"


# ---------------------------------------------------------------------------
# R2 — nothing older than a week is kept, counted or trained on (rolling).
# ---------------------------------------------------------------------------


def test_r2a_week_old_haiku_decisions_are_not_counted_even_without_a_prune(store) -> None:
    ids = _seed_many(store, HANDFUL + 5, _week() + timedelta(hours=1))

    result = _tick(store)

    assert result["new_decisions"] == 0
    assert result["fired"] is False
    assert not any(_consumed(store, rid) for rid in ids)


def test_r2b_gate_boundary_is_rolling_on_logged_at(store) -> None:
    young = _seed(store, _week() - timedelta(hours=1))
    _seed(store, _week() + timedelta(hours=1))

    result = _tick(store)
    assert result["new_decisions"] == 1

    count, row_ids = store.count_new_haiku_decisions(now=NOW)
    assert (count, row_ids) == (1, [young])


def test_r2c_prune_drops_haiku_rows_past_a_week_and_keeps_those_inside(store) -> None:
    young = _seed(store, _week() - timedelta(hours=1))
    old = _seed(store, _week() + timedelta(hours=1))

    store.prune_calibration_log(now=NOW)

    assert young in _ids(store)
    assert old not in _ids(store)


def test_r2d_logged_at_uses_the_format_the_week_cutoff_compares_against(store) -> None:
    store.log_calibration_sample(query="q", candidate_ids=["a"], reranker_scores=[1.0], reranker_model_id="m")
    value = store._conn.execute("SELECT logged_at FROM calibration_log").fetchone()["logged_at"]
    datetime.strptime(value, "%Y-%m-%d %H:%M:%S")  # raises if the format ever differs


# ---------------------------------------------------------------------------
# R3 — every other row keeps F2a's 3-day window.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("age_days", [4, 5, 6])
def test_r3_rows_without_an_untrained_haiku_decision_keep_the_three_day_window(store, age_days) -> None:
    age = timedelta(days=age_days)
    unlabeled = _seed(store, age, haiku="unlabeled")
    all_none = _seed(store, age, haiku="all_none")
    consumed = _seed(store, age, consumed=True)
    inside = {
        _seed(store, timedelta(days=1), haiku="unlabeled"),
        _seed(store, timedelta(days=1), haiku="all_none"),
        _seed(store, timedelta(days=1), consumed=True),
    }

    store.prune_calibration_log(now=NOW)

    remaining = _ids(store)
    assert {unlabeled, all_none, consumed}.isdisjoint(remaining)
    assert inside <= remaining


# ---------------------------------------------------------------------------
# R4 — the accumulated rows are cleared once the self-tune has trained on them.
# ---------------------------------------------------------------------------


def test_r4_tick_clears_the_held_rows_it_trained_on_and_keeps_in_window_rows(store) -> None:
    # The daily prune has run (it sets F2a's window); the rows it would hold are
    # then inserted directly, so this setup is identical at every commit.
    store.prune_calibration_log(now=NOW)
    held = _seed_many(store, HANDFUL + 5, timedelta(days=5))
    in_window = _seed(store, timedelta(days=1))

    result = _tick(store)

    assert result["fired"] is True and result["error"] is None
    remaining = _ids(store)
    assert set(held).isdisjoint(remaining), "held rows the tick trained on are cleared by the tick"
    assert in_window in remaining, "a consumed row inside F2a's window is not deleted early (AC10)"
    assert result["cleared"] == len(held)
    # ... and it is never counted again.
    assert store.count_new_haiku_decisions(now=NOW) == (0, [])


def test_r4_clear_keeps_an_unconsumed_held_row_until_it_passes_a_week(store) -> None:
    store.prune_calibration_log(now=NOW)
    trained = _seed(store, timedelta(days=5), consumed=True)
    untrained = _seed(store, timedelta(days=5))

    cleared = store.clear_selftune_held_rows(now=NOW)

    assert cleared == 1
    assert trained not in _ids(store) and untrained in _ids(store)
    assert store.count_new_haiku_decisions(now=NOW) == (1, [untrained])
    later = NOW + timedelta(days=2, hours=1)  # the row is now past a week old
    assert store.count_new_haiku_decisions(now=later) == (0, [])
    store.clear_selftune_held_rows(now=later)
    assert untrained not in _ids(store)


def test_r4_clear_is_a_noop_before_any_prune(store) -> None:
    rid = _seed(store, timedelta(days=5), consumed=True)
    assert store.clear_selftune_held_rows(now=NOW) == 0
    assert rid in _ids(store)


# ---------------------------------------------------------------------------
# R4b — 2B through the real mid/beefy ACCEPT dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", [judge_selftune.TUNE_GRADE_LORA, judge_selftune.TUNE_GRADE_FULL_FT])
def test_r4b_accept_week_clears_doc_having_rows_and_holds_the_2b_doc_absent_row(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    logged = NOW - timedelta(days=5)
    store._conn.execute(
        "UPDATE calibration_log SET logged_at = ?, day_bucket = ?",
        (logged.strftime("%Y-%m-%d %H:%M:%S"), logged.strftime("%Y-%m-%d")),
    )
    store._conn.commit()
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    env.accept = True

    store.prune_calibration_log(now=NOW)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=NOW, persona_dir=tmp_path)

    assert result["fired"] is True and result["accepted"] is True and result["error"] is None
    remaining = _ids(store)
    for key in ("r1", "r2", "r3"):
        assert ids[key] not in remaining, f"{key}: trained on (doc-having) and held → cleared"
    doc_absent = ids["r4_docabsent"]
    assert doc_absent in remaining and not _consumed(store, doc_absent)
    count, row_ids = store.count_new_haiku_decisions(now=NOW)
    assert row_ids == [doc_absent] and count == 1
    # Once it passes a week it is neither counted nor kept.
    later = logged + _week() + timedelta(hours=1)
    assert store.count_new_haiku_decisions(now=later) == (0, [])
    store.prune_calibration_log(now=later)
    assert doc_absent not in _ids(store)


# ---------------------------------------------------------------------------
# R6 — F2a sees exactly what it saw before inc8 (AC10).
# ---------------------------------------------------------------------------


def test_r6a_idle_gap_a_held_day_is_invisible_to_the_f2a_floor_fit(store) -> None:
    # A held Haiku row carrying enough labeled pairs that, if F2a could see it,
    # its day would be "the most recent day" and a floor would be fit.
    n = floor_calibration.FLOOR_FIT_MIN_LABELED_PAIRS + 10
    _seed(store, timedelta(days=5), n_candidates=n)

    store.prune_calibration_log(now=NOW)

    assert store.labeled_calibration_pairs("m") == []
    outcome = floor_calibration.derive_and_persist_floor(store, "m")
    assert outcome.accepted is False
    assert store.get_persisted_reranker_floor("m") is None


def test_r6b_f2a_pairs_equal_the_pairs_without_the_held_rows(store) -> None:
    control = MemoryStore(db_path=":memory:")
    for s in (store, control):
        _seed(s, timedelta(days=1), n_candidates=6)
        _seed(s, timedelta(days=1), haiku="all_none", n_candidates=4)
    _seed(store, timedelta(days=5), n_candidates=8)  # held only in `store`

    store.prune_calibration_log(now=NOW)
    control.prune_calibration_log(now=NOW)

    assert store.labeled_calibration_pairs("m") == control.labeled_calibration_pairs("m")
    assert store.labeled_calibration_pairs("m")  # non-trivial comparison
    control.close()


def test_r6c_held_rows_are_never_sampled_for_judge_labeling(store) -> None:
    _seed(store, timedelta(days=5))
    unlabeled = {_seed(store, timedelta(days=1), haiku="unlabeled") for _ in range(3)}

    store.prune_calibration_log(now=NOW)

    assert {r["id"] for r in store.sample_unlabeled_calibration_rows(100)} == unlabeled


def test_r7_f2a_retention_derivation_is_unchanged() -> None:
    assert floor_calibration.derive_retention_window_days() == 3.0


# ---------------------------------------------------------------------------
# R8 — one week is derived from the weekly cadence, not typed.
# ---------------------------------------------------------------------------


def test_r8_week_follows_the_weekly_cadence_constant(store, monkeypatch: pytest.MonkeyPatch) -> None:
    rid = _seed(store, timedelta(days=5))
    assert store.count_new_haiku_decisions(now=NOW) == (1, [rid])

    monkeypatch.setattr(judge_selftune, "JUDGE_TUNE_INTERVAL_HOURS", 96.0)

    assert store.count_new_haiku_decisions(now=NOW) == (0, [])
    store.prune_calibration_log(now=NOW)
    assert rid not in _ids(store)


# ---------------------------------------------------------------------------
# R13 — a clear fault never masks a completed tune.
# ---------------------------------------------------------------------------


def test_r13_clear_fault_does_not_mark_the_tune_failed(store, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = _seed_many(store, HANDFUL + 5, timedelta(days=1))

    def boom(self, *, now=None):
        raise RuntimeError("simulated clear failure")

    monkeypatch.setattr(MemoryStore, "clear_selftune_held_rows", boom)

    result = _tick(store)

    assert result["fired"] is True
    assert result["error"] is None
    assert result["clear_error"] is not None and "simulated clear failure" in result["clear_error"]
    assert all(_consumed(store, rid) for rid in ids)


# ---------------------------------------------------------------------------
# R14 — a DB that predates inc8 (I9).
# ---------------------------------------------------------------------------

# Verbatim from ba2ca98a's brain/memory/store.py (the pre-inc8 schema).
_PRE_INC8_CALIBRATION_LOG_DDL = """
CREATE TABLE IF NOT EXISTS calibration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    day_bucket TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d', 'now')),
    query TEXT NOT NULL,
    candidate_ids TEXT NOT NULL,
    reranker_scores TEXT NOT NULL,
    reranker_model_id TEXT NOT NULL,
    local_judge_label TEXT,
    haiku_label TEXT,
    score_scale TEXT NOT NULL DEFAULT 'raw',
    local_judge_raw_score TEXT,
    candidate_docs TEXT,
    selftune_consumed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_calibration_log_day_bucket ON calibration_log(day_bucket);
"""


def _view(store: MemoryStore) -> str | None:
    row = store._conn.execute(
        "SELECT f2a_view_cutoff_bucket FROM calibration_log_retention_state"
    ).fetchone()
    return None if row is None else row[0]


def test_r14_pre_inc8_database_upgrades_with_unchanged_reads(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    conn = sqlite3.connect(db)
    conn.executescript(_PRE_INC8_CALIBRATION_LOG_DDL)
    for day, scores, labels in (
        ("2026-03-01", [0.1, 0.2], ["irrelevant", "relevant"]),
        ("2026-03-02", [0.7, 0.3], ["relevant", "irrelevant"]),
    ):
        conn.execute(
            "INSERT INTO calibration_log (logged_at, day_bucket, query, candidate_ids, reranker_scores, "
            "reranker_model_id, local_judge_label, haiku_label, score_scale) "
            "VALUES (?, ?, 'q', '[\"a\",\"b\"]', ?, 'm', ?, '[null, null]', 'normalized')",
            (f"{day} 10:00:00", day, json.dumps(scores), json.dumps(labels)),
        )
    held_logged = NOW - timedelta(days=5)
    conn.execute(
        "INSERT INTO calibration_log (logged_at, day_bucket, query, candidate_ids, reranker_scores, "
        "reranker_model_id, local_judge_label, haiku_label, score_scale) "
        "VALUES (?, ?, 'h', '[\"x\"]', '[0.5]', 'other', '[\"relevant\"]', '[\"relevant\"]', 'normalized')",
        (held_logged.strftime("%Y-%m-%d %H:%M:%S"), held_logged.strftime("%Y-%m-%d")),
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db)
    tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "calibration_log_retention_state" in tables
    assert _view(store) is None, "no row seeded at creation"
    # The ba2ca98a contract: the most recent labeled day's (score, effective label) pairs.
    assert store.labeled_calibration_pairs("m") == [(0.7, "relevant"), (0.3, "irrelevant")]

    store.prune_calibration_log(now=NOW)
    assert _view(store) == (NOW - timedelta(days=3)).strftime("%Y-%m-%d")
    remaining = {r["query"] for r in store._conn.execute("SELECT query FROM calibration_log")}
    assert remaining == {"h"}, "old rows without a Haiku decision pruned; the 5-day Haiku row held"
    view_before = _view(store)
    store.close()

    reopened = MemoryStore(db)
    assert _view(reopened) == view_before, "a second open is a no-op"
    reopened.close()


# ---------------------------------------------------------------------------
# R15 — the F2a view cutoff is a high-water mark across window changes.
# ---------------------------------------------------------------------------


def test_r15_view_cutoff_is_high_water_across_window_changes(store) -> None:
    n = floor_calibration.FLOOR_FIT_MIN_LABELED_PAIRS + 10
    _seed(store, timedelta(days=5), n_candidates=n)

    store.prune_calibration_log(now=NOW, window_days=3.0)
    v3 = _view(store)
    assert v3 == (NOW - timedelta(days=3)).strftime("%Y-%m-%d")

    store.prune_calibration_log(now=NOW, window_days=10.0)  # operator widens the window
    assert _view(store) == v3
    assert store.labeled_calibration_pairs("m") == [], "a held row stays invisible to F2a"

    store.prune_calibration_log(now=NOW, window_days=2.0)  # operator narrows it
    assert _view(store) == (NOW - timedelta(days=2)).strftime("%Y-%m-%d")
