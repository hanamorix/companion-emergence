import json

from brain.forgetting.graveyard import (
    GRAVEYARD_FILENAME,
    append,
    search,
)
from brain.forgetting.salience import SalienceInputs
from brain.memory.store import Memory


def _make_memory(*, id="mem_test", content="hello", domain="chat", emotions=None) -> Memory:
    m = Memory.create_new(
        content=content,
        memory_type="episodic",
        domain=domain,
        emotions=emotions or {},
    )
    object.__setattr__(m, "id", id)
    return m


def test_append_writes_jsonl_row(tmp_path):
    m = _make_memory(id="mem_001", content="apple banana", emotions={"sorrow": 7.2, "love": 3.0})
    append(
        tmp_path,
        memory=m,
        salience_at_drop=0.07,
        inputs=SalienceInputs(emotion=0.72, hebbian=0.05, recall=0.0, soul=0.0, freshness=0.0),
        lived_age_hours=612.4,
        reason="salience<0.10 for 2 consecutive passes",
    )

    log = tmp_path / GRAVEYARD_FILENAME
    assert log.exists()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["memory_id"] == "mem_001"
    assert entry["domain"] == "chat"
    assert entry["summary"] == "apple banana"
    assert entry["salience_at_drop"] == 0.07
    assert entry["salience_inputs_at_drop"]["emotion"] == 0.72
    assert entry["lived_age_hours_at_forgetting"] == 612.4
    assert entry["graveyard_reason"] == "salience<0.10 for 2 consecutive passes"


def test_search_returns_substring_matches(tmp_path):
    for i, content in enumerate(["apple pie", "cherry tart", "apple jam"]):
        append(
            tmp_path,
            memory=_make_memory(id=f"mem_{i}", content=content),
            salience_at_drop=0.05,
            inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
            lived_age_hours=100.0,
            reason="x",
        )
    hits = search(tmp_path, "apple")
    assert len(hits) == 2
    summaries = {h["summary"] for h in hits}
    assert summaries == {"apple pie", "apple jam"}


def test_search_returns_most_recent_first(tmp_path):
    # Append three with explicit ordering. read returns chronologically;
    # search should reverse to most-recent-first.
    for i, content in enumerate(["first match", "middle match", "last match"]):
        append(
            tmp_path,
            memory=_make_memory(id=f"mem_{i}", content=content),
            salience_at_drop=0.05,
            inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
            lived_age_hours=100.0,
            reason="x",
        )
    hits = search(tmp_path, "match", limit=10)
    assert [h["summary"] for h in hits] == ["last match", "middle match", "first match"]


def test_search_respects_limit(tmp_path):
    for i in range(10):
        append(
            tmp_path,
            memory=_make_memory(id=f"mem_{i}", content=f"item {i} apple"),
            salience_at_drop=0.05,
            inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
            lived_age_hours=100.0,
            reason="x",
        )
    hits = search(tmp_path, "apple", limit=3)
    assert len(hits) == 3


def test_search_tolerates_corrupt_jsonl_line(tmp_path):
    log = tmp_path / GRAVEYARD_FILENAME
    log.write_text(
        '{"memory_id":"a","summary":"good","domain":"chat","forgotten_at_iso":"2026-01-01T00:00:00+00:00","lived_age_hours_at_forgetting":1.0,"salience_at_drop":0.05,"salience_inputs_at_drop":{},"graveyard_reason":"x"}\n'
        "garbage\n"
        '{"memory_id":"b","summary":"also good","domain":"chat","forgotten_at_iso":"2026-01-02T00:00:00+00:00","lived_age_hours_at_forgetting":1.0,"salience_at_drop":0.05,"salience_inputs_at_drop":{},"graveyard_reason":"x"}\n'
    )
    hits = search(tmp_path, "good")
    assert len(hits) == 2
