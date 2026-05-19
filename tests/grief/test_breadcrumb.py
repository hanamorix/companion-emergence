"""test_breadcrumb.py — intensity formulas + content phrases + write path."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.grief import breadcrumb
from brain.memory.store import MemoryStore


def test_compute_drop_intensity_high_emotion_high_salience() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=0.9, salience_at_drop=0.7)
    assert result == pytest.approx(4.41, abs=0.01)


def test_compute_drop_intensity_low_inputs_under_floor() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=0.2, salience_at_drop=0.3)
    assert result == pytest.approx(0.42, abs=0.01)


def test_compute_drop_intensity_clamped_at_10() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=2.0, salience_at_drop=2.0)
    assert result == 10.0


def test_compute_drop_intensity_clamped_at_zero() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=-0.5, salience_at_drop=0.7)
    assert result == 0.0


def test_compute_arc_close_intensity_heavy_member() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.8)
    assert result == pytest.approx(5.6, abs=0.01)


def test_compute_arc_close_intensity_under_floor() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.2)
    assert result == pytest.approx(1.4, abs=0.01)


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
