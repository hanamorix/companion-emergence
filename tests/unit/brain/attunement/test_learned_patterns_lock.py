"""learned_patterns.jsonl read-then-append gaps closed (#650 finding 5, L2 items 5 and 32).

Pass 2 runs on its own thread, beside the start-up backfill, so every writer that reads the patterns and then appends a
superseding row holds `learned_patterns_lock` across the whole read..append. Synthetic tmp persona dirs, no model call.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from brain.attunement import crystallise, store
from brain.attunement.schemas import Evidence, PatternCandidate
from brain.attunement.store import (
    BufferTurn,
    learned_patterns_lock,
    mark_addressed,
    merge_into_learned,
    read_learned_patterns,
)

NOW = "2026-11-27T12:00:00Z"


def _candidate(quote: str = "I keep coming back to this") -> tuple[PatternCandidate, list[BufferTurn]]:
    turn = BufferTurn(id="t1", content=quote)
    cand = PatternCandidate(
        category="tone",
        canonical_key="coming-back",
        description="returns to a topic",
        evidence=[Evidence(quote=quote, turn_id="t1")],
    )
    return cand, [turn]


def _seed(persona_dir: Path) -> str:
    cand, turns = _candidate()
    merge_into_learned(persona_dir, [cand], turns, now_iso=NOW)
    (pattern,) = read_learned_patterns(persona_dir)
    return pattern.id


@pytest.mark.parametrize("writer", ["merge", "mark_addressed", "apply_contradiction", "crystallise"])
def test_each_read_then_append_writer_waits_for_the_lock_before_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    pid = _seed(tmp_path)
    read_started = threading.Event()
    real_read = store.read_learned_patterns

    def spy(persona_dir):  # noqa: ANN001, ANN202
        read_started.set()
        return real_read(persona_dir)

    monkeypatch.setattr(store, "read_learned_patterns", spy)
    monkeypatch.setattr(crystallise, "read_learned_patterns", spy)
    cand, turns = _candidate()
    calls = {
        "merge": lambda: merge_into_learned(tmp_path, [cand], turns, now_iso=NOW),
        "mark_addressed": lambda: mark_addressed(tmp_path, [pid], now_iso=NOW),
        "apply_contradiction": lambda: store.apply_contradiction(tmp_path, pid, now_iso=NOW),
        "crystallise": lambda: crystallise.check_crystallisations(tmp_path, now_iso=NOW),
    }
    done = threading.Event()

    def run() -> None:
        calls[writer]()
        done.set()

    with learned_patterns_lock(tmp_path):
        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert not read_started.wait(0.3), f"{writer} read the patterns while the lock was held"
    assert done.wait(30), f"{writer} did not finish after the lock was released (a deadlock on its own append?)"
    t.join()
    assert read_started.is_set()


def test_two_writers_racing_do_not_lose_an_update(tmp_path: Path) -> None:
    """Many threads each add one evidence to the same pattern: the final count is every one of them."""
    _seed(tmp_path)
    cand, turns = _candidate()
    threads = [
        threading.Thread(target=lambda: merge_into_learned(tmp_path, [cand], turns, now_iso=NOW), daemon=True)
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (pattern,) = read_learned_patterns(tmp_path)
    assert pattern.evidence_count == 9, "1 seeded + 8 concurrent merges, none superseded by a stale read"
