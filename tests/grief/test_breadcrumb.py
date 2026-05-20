"""test_breadcrumb.py — intensity formulas + content phrases + write path."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.grief import breadcrumb
from brain.memory.store import MemoryStore


def test_compute_drop_intensity_high_emotion() -> None:
    # 0.9 emotion × 7.0 = 6.3 (heavy, breadcrumb territory)
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=0.9)
    assert result == pytest.approx(6.3, abs=0.01)


def test_compute_drop_intensity_low_inputs_under_floor() -> None:
    # 0.2 emotion × 7.0 = 1.4 (under floor)
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=0.2)
    assert result == pytest.approx(1.4, abs=0.01)


def test_compute_drop_intensity_clamped_at_10() -> None:
    # 2.0 emotion (out-of-range) × 7.0 = 14, clamped to 10
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=2.0)
    assert result == 10.0


def test_compute_drop_intensity_clamped_at_zero() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=-0.5)
    assert result == 0.0


def test_compute_arc_close_intensity_heavy_member() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.8)
    assert result == pytest.approx(5.6, abs=0.01)


def test_compute_arc_close_intensity_under_floor() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.2)
    assert result == pytest.approx(1.4, abs=0.01)


def test_compute_arc_close_intensity_zero() -> None:
    assert breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.0) == 0.0


def test_compute_arc_close_intensity_clamped_at_10() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=2.0)
    assert result == 10.0


def test_first_n_words_long_summary() -> None:
    summary = "I drove out to the rooftop morning before the cold rain hit hard."
    assert breadcrumb.first_n_words(summary, 6) == "I drove out to the rooftop"


def test_first_n_words_short_summary() -> None:
    summary = "quiet day"
    assert breadcrumb.first_n_words(summary, 6) == "quiet day"


def test_first_n_words_empty() -> None:
    assert breadcrumb.first_n_words("", 6) == ""


def test_drop_phrase_with_summary() -> None:
    phrase = breadcrumb.drop_phrase("I drove out to the rooftop morning before")
    assert phrase == "the memory of I drove out to the rooftop is gone"


def test_drop_phrase_empty_summary_fallback() -> None:
    phrase = breadcrumb.drop_phrase("", lived_days_ago=2.3)
    assert phrase == "a memory from 2 lived-days ago is gone"


def test_recall_touch_phrase() -> None:
    phrase = breadcrumb.recall_touch_phrase("I drove out to the rooftop morning before")
    assert phrase == "reached for I drove out to the rooftop — gone"


def test_recall_touch_phrase_empty_summary() -> None:
    phrase = breadcrumb.recall_touch_phrase("")
    assert phrase == "reached for a lost memory — gone"


def test_arc_close_phrase() -> None:
    phrase = breadcrumb.arc_close_phrase("first cold week")
    assert phrase == "the arc 'first cold week' has closed"


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db")


def test_write_breadcrumb_drop_with_residue(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    bc_id = breadcrumb.write_breadcrumb(
        store=store,
        intensity=4.4,
        subtype="drop",
        referent_type="memory",
        referent_id="mem-abc-123",
        content="the memory of the rooftop morning is gone",
        residue_emotion=("joy", 7.0),
    )
    saved = store.get(bc_id)
    assert saved is not None
    assert saved.memory_type == "grief_event"
    assert saved.domain == "grief"
    assert saved.emotions == {"memory_grief": 4.4, "joy": 3.5}
    assert saved.metadata["grief_referent_id"] == "mem-abc-123"
    assert saved.metadata["grief_referent_type"] == "memory"
    assert saved.metadata["grief_subtype"] == "drop"
    assert saved.metadata.get("triggering_arc_id") is None
    assert saved.state == "active"


def test_write_breadcrumb_recall_touch_with_triggering_arc(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    bc_id = breadcrumb.write_breadcrumb(
        store=store,
        intensity=3.5,
        subtype="recall_touch",
        referent_type="memory",
        referent_id="mem-xyz",
        content="reached for the night of — gone",
        residue_emotion=None,
        triggering_arc_id="arc-foo",
    )
    saved = store.get(bc_id)
    assert saved is not None
    assert saved.emotions == {"memory_grief": 3.5}
    assert saved.metadata["triggering_arc_id"] == "arc-foo"
    assert saved.metadata["grief_subtype"] == "recall_touch"


# ---------------------------------------------------------------------------
# Phase 5 — public surface (handle_drop / handle_arc_close via brain.grief)
# ---------------------------------------------------------------------------

from brain.memory.store import Memory  # noqa: E402


def _make_dropped_memory(*, memory_id: str = "mem-drop", joy: float = 8.0) -> Memory:
    m = Memory.create_new(
        content="we drove out to the rooftop and watched the morning",
        memory_type="episodic",
        domain="memory",
        emotions={"joy": joy},
    )
    object.__setattr__(m, "id", memory_id)
    return m


def test_handle_drop_writes_breadcrumb_above_threshold(tmp_path: Path) -> None:
    from brain import grief

    store = MemoryStore(tmp_path / "memories.db")
    # joy=9.0 (raw), normalised to 0.9, × 7.0 = 6.3 — above THRESHOLD (3.0).
    # Residue: joy × 0.5 = 4.5.
    memory = _make_dropped_memory(memory_id="mem-drop-1", joy=9.0)

    grief.handle_drop(
        memory=memory,
        persona_dir=tmp_path,
        store=store,
    )
    rows = store._conn.execute(
        "SELECT id, content, emotions_json, metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    assert len(rows) == 1
    import json

    em = json.loads(rows[0]["emotions_json"])
    assert em["memory_grief"] == pytest.approx(6.3, abs=0.01)
    assert em.get("joy", 0.0) == pytest.approx(4.5, abs=0.01)
    meta = json.loads(rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "mem-drop-1"
    assert meta["grief_subtype"] == "drop"


def test_handle_drop_skips_below_threshold(tmp_path: Path) -> None:
    from brain import grief

    store = MemoryStore(tmp_path / "memories.db")
    # joy=2.0 (raw) -> normalised 0.2 -> intensity 0.2 × 7.0 = 1.4, below threshold.
    memory = _make_dropped_memory(memory_id="mem-drop-2", joy=2.0)
    grief.handle_drop(memory=memory, persona_dir=tmp_path, store=store)
    rows = store._conn.execute("SELECT id FROM memories WHERE memory_type='grief_event'").fetchall()
    assert rows == []


def test_handle_arc_close_writes_breadcrumb_above_threshold(tmp_path: Path) -> None:
    from brain import grief

    store = MemoryStore(tmp_path / "memories.db")

    class FakeArc:
        id = "arc-1"
        title = "first cold week"
        max_member_emotion_normalised = 0.85
        dominant_non_grief_emotion = ("sorrow", 7.5)

    grief.handle_arc_close(arc=FakeArc(), persona_dir=tmp_path, store=store)
    rows = store._conn.execute(
        "SELECT id, content, emotions_json, metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    assert len(rows) == 1
    import json

    em = json.loads(rows[0]["emotions_json"])
    assert em["memory_grief"] >= 3.0
    assert em.get("sorrow", 0.0) == pytest.approx(3.75, abs=0.01)  # 7.5 * 0.5
    meta = json.loads(rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "arc-1"
    assert meta["grief_subtype"] == "arc_close"
    content_str = (
        rows[0]["content"] if isinstance(rows[0]["content"], str) else rows[0]["content"].decode()
    )
    assert "first cold week" in content_str
