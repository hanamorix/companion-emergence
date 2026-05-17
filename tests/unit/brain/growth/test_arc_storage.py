"""Tests for arc storage helpers — graveyard + snapshot."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.engines.reflex import ReflexArc
from brain.growth.arc_storage import (
    append_removed_arc,
    read_arc_snapshot,
    read_removed_arcs,
    recently_removed_names,
    write_arc_snapshot,
)


def _make_arc(name: str = "test_arc", created_by: str = "brain_emergence") -> ReflexArc:
    return ReflexArc(
        name=name,
        description=f"description of {name}",
        trigger={"vulnerability": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=12.0,
        action="generate_journal",
        output_memory_type="reflex_journal",
        prompt_template="You are nell. {emotion_summary}",
        created_by=created_by,
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
    )


def test_graveyard_round_trip(tmp_path: Path):
    arc = _make_arc("loneliness_journal")
    append_removed_arc(
        tmp_path,
        arc=arc,
        removed_at=datetime(2026, 4, 28, tzinfo=UTC),
        removed_by="user_edit",
        reasoning=None,
    )
    entries = read_removed_arcs(tmp_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "loneliness_journal"
    assert e["removed_by"] == "user_edit"
    assert e["reasoning"] is None
    assert e["trigger_snapshot"] == {"vulnerability": 8.0}
    assert e["description_snapshot"] == "description of loneliness_journal"
    assert "prompt_template_snapshot" in e


def test_graveyard_appends_multiple(tmp_path: Path):
    for i in range(3):
        append_removed_arc(
            tmp_path,
            arc=_make_arc(f"arc_{i}"),
            removed_at=datetime(2026, 4, 28, tzinfo=UTC),
            removed_by="brain_self_prune",
            reasoning=f"reason {i}",
        )
    entries = read_removed_arcs(tmp_path)
    assert [e["name"] for e in entries] == ["arc_0", "arc_1", "arc_2"]
    assert all(e["removed_by"] == "brain_self_prune" for e in entries)


def test_recently_removed_names_window(tmp_path: Path):
    """Only entries within `grace_days` count as recently-removed."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    append_removed_arc(  # 5 days ago — within window
        tmp_path,
        arc=_make_arc("recent"),
        removed_at=now - timedelta(days=5),
        removed_by="user_edit",
        reasoning=None,
    )
    append_removed_arc(  # 20 days ago — outside 15d window
        tmp_path,
        arc=_make_arc("ancient"),
        removed_at=now - timedelta(days=20),
        removed_by="user_edit",
        reasoning=None,
    )
    names = recently_removed_names(tmp_path, now=now, grace_days=15)
    assert names == {"recent"}


def test_snapshot_round_trip(tmp_path: Path):
    arcs = [_make_arc("a"), _make_arc("b", created_by="og_migration")]
    write_arc_snapshot(tmp_path, arcs=arcs, snapshot_at=datetime(2026, 4, 28, tzinfo=UTC))
    read_back = read_arc_snapshot(tmp_path)
    assert read_back is not None
    assert {a.name for a in read_back} == {"a", "b"}
    by_name = {a.name: a for a in read_back}
    assert by_name["a"].created_by == "brain_emergence"
    assert by_name["b"].created_by == "og_migration"


def test_snapshot_returns_none_when_missing(tmp_path: Path):
    assert read_arc_snapshot(tmp_path) is None


def test_graveyard_handles_corrupt_lines(tmp_path: Path):
    """A corrupt line in removed_arcs.jsonl should be skipped, not crash."""
    g = tmp_path / "removed_arcs.jsonl"
    g.write_text(
        '{"name": "valid1", "removed_at": "2026-04-28T00:00:00+00:00", '
        '"removed_by": "user_edit", "reasoning": null, "trigger_snapshot": {}, '
        '"description_snapshot": "", "prompt_template_snapshot": ""}\n'
        "this is not json\n"
        '{"name": "valid2", "removed_at": "2026-04-28T00:00:00+00:00", '
        '"removed_by": "user_edit", "reasoning": null, "trigger_snapshot": {}, '
        '"description_snapshot": "", "prompt_template_snapshot": ""}\n'
    )
    entries = read_removed_arcs(tmp_path)
    assert [e["name"] for e in entries] == ["valid1", "valid2"]
