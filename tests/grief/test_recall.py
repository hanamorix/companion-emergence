"""test_recall.py — recall-touch intensity + handle_recall_touch."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.grief import policy, recall
from brain.memory.store import MemoryStore


def test_touch_intensity_fresh_loss() -> None:
    # 0.8 emotion × 5.0 × 1.0 (recency at d=0) = 4.0
    result = recall.compute_touch_intensity(
        grave_emotion_max=0.8,
        lived_days_since_loss=0.0,
    )
    assert result == pytest.approx(4.0, abs=0.01)


def test_touch_intensity_recency_decay() -> None:
    fresh = recall.compute_touch_intensity(grave_emotion_max=1.0, lived_days_since_loss=0.0)
    aged = recall.compute_touch_intensity(grave_emotion_max=1.0, lived_days_since_loss=14.0)
    assert aged == pytest.approx(fresh * 0.5, abs=0.01)


def test_touch_intensity_clamped_at_10() -> None:
    # 3.0 emotion (out-of-range) × 5.0 × 1.0 = 15, clamped to 10
    result = recall.compute_touch_intensity(grave_emotion_max=3.0, lived_days_since_loss=0.0)
    assert result == 10.0


def test_touch_intensity_old_loss_low() -> None:
    # 60 lived-days ago → recency 0.5^(60/14) ≈ 0.051
    # Full-score input g=0.9 produces raw ≈ 0.230 — well below 3.0 threshold.
    result = recall.compute_touch_intensity(grave_emotion_max=0.9, lived_days_since_loss=60.0)
    assert result < 0.3


def _make_graveyard_entry(
    *,
    memory_id: str,
    summary: str = "the night by the rooftop",
    emotion_max_normalised: float = 0.8,
    salience_at_drop: float = 0.6,
    lived_age_hours_at_forgetting: float = 24.0,
    forgotten_at_iso: str | None = None,
) -> dict:
    """Construct a graveyard entry dict matching brain/forgetting/graveyard.append.

    Note: salience_at_drop is still in the graveyard schema as part of the audit
    trail, but it is no longer read by handle_recall_touch's intensity formula
    (see spec §3 — salience was dropped from grief intensity calculations).
    Kept on the fixture so test entries match real-world graveyard JSON shape.
    """
    return {
        "memory_id": memory_id,
        "forgotten_at_iso": forgotten_at_iso or datetime.now(UTC).isoformat(),
        "lived_age_hours_at_forgetting": lived_age_hours_at_forgetting,
        "domain": "memory",
        "memory_type": "episodic",
        "created_at_iso": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        "summary": summary,
        "emotion_at_ingest": {"joy": emotion_max_normalised * 10.0},
        "salience_at_drop": salience_at_drop,
        "salience_inputs_at_drop": {
            "emotion": emotion_max_normalised,
            "hebbian": 0.0,
            "recall": 0.0,
            "soul": 0.0,
            "freshness": 0.0,
        },
        "graveyard_reason": "salience<0.2 for 3 consecutive passes",
    }


def test_handle_recall_touch_writes_breadcrumb_above_threshold(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    entry = _make_graveyard_entry(memory_id="mem-x")
    recall.handle_recall_touch(
        touched_ids=["mem-x"],
        graveyard_entries=[entry],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
    )
    rows = store._conn.execute(
        "SELECT id FROM memories WHERE memory_type = 'grief_event'"
    ).fetchall()
    assert len(rows) == 1


def test_handle_recall_touch_skips_below_threshold(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    entry = _make_graveyard_entry(
        memory_id="mem-x", emotion_max_normalised=0.2, salience_at_drop=0.2
    )
    recall.handle_recall_touch(
        touched_ids=["mem-x"],
        graveyard_entries=[entry],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
    )
    rows = store._conn.execute(
        "SELECT id FROM memories WHERE memory_type = 'grief_event'"
    ).fetchall()
    assert rows == []


def test_handle_recall_touch_debounces_within_2_hours(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    entry = _make_graveyard_entry(memory_id="mem-x")
    recall.handle_recall_touch(
        touched_ids=["mem-x"],
        graveyard_entries=[entry],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
    )
    recall.handle_recall_touch(
        touched_ids=["mem-x"],
        graveyard_entries=[entry],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
    )
    rows = store._conn.execute(
        "SELECT id FROM memories WHERE memory_type = 'grief_event'"
    ).fetchall()
    assert len(rows) == 1


def test_handle_recall_touch_attaches_triggering_arc(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    entry = _make_graveyard_entry(memory_id="mem-x")
    recall.handle_recall_touch(
        touched_ids=["mem-x"],
        graveyard_entries=[entry],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
        triggering_arc_id="arc-42",
    )
    row = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchone()
    meta = json.loads(row["metadata_json"])
    assert meta["triggering_arc_id"] == "arc-42"


def test_handle_recall_touch_ignores_untouched_entries(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    entry_x = _make_graveyard_entry(memory_id="mem-x")
    entry_y = _make_graveyard_entry(memory_id="mem-y")
    recall.handle_recall_touch(
        touched_ids=["mem-x"],
        graveyard_entries=[entry_x, entry_y],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
    )
    rows = store._conn.execute("SELECT id FROM memories WHERE memory_type='grief_event'").fetchall()
    assert len(rows) == 1


def test_handle_recall_touch_empty_graveyard_no_writes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    recall.handle_recall_touch(
        touched_ids=["mem-anything"],
        graveyard_entries=[],
        persona_dir=tmp_path,
        store=store,
        lived_age_hours_now=24.0,
    )
    rows = store._conn.execute("SELECT id FROM memories WHERE memory_type='grief_event'").fetchall()
    assert rows == []


# The grief-touch race (#342): two turns recalling the same forgotten memory at once must write ONE breadcrumb.
_RACE_B_HEAD_START_S = 2.0  # how long the second turn gets to run while the first is paused at its insert
_RACE_JOIN_S = 10.0  # the most the test waits for the second turn to finish


def _touch_count(db: Path, referent_id: str) -> int:
    store = MemoryStore(db)
    try:
        rows = store._conn.execute("SELECT metadata_json FROM memories WHERE memory_type = 'grief_event'").fetchall()
        return sum(1 for r in rows if json.loads(r["metadata_json"]).get("grief_referent_id") == referent_id)
    finally:
        store.close()


def test_two_turns_touching_one_forgotten_memory_write_one_breadcrumb(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    entry = _make_graveyard_entry(memory_id="mem-x")
    opened = threading.Event()
    go = threading.Event()
    b_outcome: dict[str, object] = {}

    def turn_b() -> None:
        store_b = MemoryStore(db)  # opened in this thread, before the race starts: the connection is thread-bound
        try:
            opened.set()
            go.wait(_RACE_JOIN_S)
            recall.handle_recall_touch(
                touched_ids=["mem-x"], graveyard_entries=[entry], persona_dir=tmp_path, store=store_b, lived_age_hours_now=24.0
            )
            b_outcome["finished"] = True
        except BaseException as exc:
            b_outcome["error"] = exc
        finally:
            store_b.close()

    thread_b = threading.Thread(target=turn_b, name="grief-turn-b", daemon=True)
    thread_b.start()
    assert opened.wait(_RACE_JOIN_S), "the second turn never opened its store"
    store_a = MemoryStore(db)
    fired: list[str] = []

    def trace(statement: str) -> None:
        # The first turn is about to insert its breadcrumb: let the second turn run its whole touch now.
        if statement.lstrip().startswith("INSERT INTO memories (") and not fired:
            fired.append(statement)
            go.set()
            thread_b.join(_RACE_B_HEAD_START_S)

    store_a._conn.set_trace_callback(trace)
    a_finished = False
    try:
        recall.handle_recall_touch(
            touched_ids=["mem-x"], graveyard_entries=[entry], persona_dir=tmp_path, store=store_a, lived_age_hours_now=24.0
        )
        a_finished = True
    finally:
        go.set()
        thread_b.join(_RACE_JOIN_S)
        store_a._conn.set_trace_callback(None)
    assert not thread_b.is_alive(), "the second turn did not finish"
    assert len(fired) == 1, "the interleaving did not happen: the first turn's insert was never seen"
    assert a_finished and b_outcome == {"finished": True}
    assert _touch_count(db, "mem-x") == 1  # the two turns raced and only one breadcrumb exists

    # the debounce cases, in the same test: inside the window nothing is written; past it one is; another referent gets its own
    try:
        recall.handle_recall_touch(
            touched_ids=["mem-x"], graveyard_entries=[entry], persona_dir=tmp_path, store=store_a, lived_age_hours_now=24.0
        )
        assert _touch_count(db, "mem-x") == 1
        old = (datetime.now(UTC) - timedelta(hours=policy.DEBOUNCE_HOURS + 1)).isoformat()
        store_a._conn.execute("UPDATE memories SET created_at = ? WHERE memory_type = 'grief_event'", (old,))
        store_a._conn.commit()
        recall.handle_recall_touch(
            touched_ids=["mem-x"], graveyard_entries=[entry], persona_dir=tmp_path, store=store_a, lived_age_hours_now=24.0
        )
        assert _touch_count(db, "mem-x") == 2
        recall.handle_recall_touch(
            touched_ids=["mem-y"],
            graveyard_entries=[_make_graveyard_entry(memory_id="mem-y")],
            persona_dir=tmp_path,
            store=store_a,
            lived_age_hours_now=24.0,
        )
        assert _touch_count(db, "mem-y") == 1
    finally:
        store_a.close()
