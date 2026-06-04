"""Soul-review cadence is persisted + self-pacing (regression for the backlog
a user hit after a multi-day Claude session-limit outage).

The old cadence timed off process-relative time.monotonic(), which reset on
every supervisor restart and didn't advance during sleep — so a 6h interval
rarely elapsed on a desktop app and candidates piled up. The cadence now
persists its next-due time (wall clock), so a due-in-the-past state fires on a
fresh process regardless of the interval.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def _neutralise_other_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", lambda *a, **k: {})
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_felt_time_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())


def test_soul_review_fires_from_persisted_due_time_on_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted next_review_at in the past fires soul review immediately on a
    fresh supervisor — even with a 6h interval, which the monotonic-only timer
    would have made wait 6h (the under-firing defect)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "soul_review_state.json").write_text(
        json.dumps(
            {
                "next_review_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                "consecutive_failures": 0,
            }
        )
    )
    _neutralise_other_ticks(monkeypatch)

    calls = [0]
    stop = threading.Event()

    def _counter(*a, **k):
        calls[0] += 1
        stop.set()
        return 0, 0  # (model_failures, eligible_pending)

    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", _counter)

    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=6 * 3600.0,  # 6h: monotonic alone would NOT fire
        finalize_interval_s=None,
    )

    assert calls[0] >= 1, "persisted past-due next_review_at must fire on a fresh process"
    # Clean drain → cadence advanced ~6h ahead and was persisted (save works).
    state = json.loads((persona_dir / "soul_review_state.json").read_text())
    saved_next = datetime.fromisoformat(state["next_review_at"])
    assert saved_next > datetime.now(UTC) + timedelta(hours=5)


def test_draft_cursor_not_advanced_when_no_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When _run_soul_review_tick early-returns because eligible_before == 0,
    the draft cursor file must NOT be created or advanced — drafts must wait
    for the next candidate-bearing tick."""
    from brain.bridge.supervisor import _run_soul_review_tick
    from brain.initiate.draft import append_draft_fragment

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    # Write a draft fragment so there IS something to potentially cursor-advance on.
    append_draft_fragment(
        persona_dir,
        timestamp="2026-06-04T10:00:00",
        source="emotion_spike",
        body="A draft that should not be consumed yet.",
    )

    # Patch count_eligible_pending at the source — it's imported inside the function body.
    monkeypatch.setattr(
        "brain.soul.review.count_eligible_pending",
        lambda *a, **k: 0,
    )

    result = _run_soul_review_tick(
        persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
    )

    assert result == (0, 0), "early-return path must return (0, 0)"
    cursor_path = persona_dir / "draft_space_review_cursor.json"
    assert not cursor_path.exists(), (
        "cursor must NOT be written when there are zero eligible candidates"
    )


def test_draft_cursor_advanced_after_successful_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When _run_soul_review_tick runs a real review pass (eligible > 0),
    the draft cursor file is written after the pass completes."""
    import json as _json

    from brain.bridge.supervisor import _run_soul_review_tick
    from brain.initiate.draft import append_draft_fragment

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    # Write a draft fragment.
    append_draft_fragment(
        persona_dir,
        timestamp="2026-06-04T10:00:00",
        source="emotion_spike",
        body="A draft that should be consumed.",
    )

    # Patch count_eligible_pending to return 1 (has candidates).
    monkeypatch.setattr("brain.soul.review.count_eligible_pending", lambda *a, **k: 1)

    # Patch review_pending_candidates to return a dummy ReviewReport.
    from brain.soul.review import ReviewReport

    monkeypatch.setattr(
        "brain.soul.review.review_pending_candidates",
        lambda *a, **k: ReviewReport(pending_at_start=1, examined=1, deferred=1),
    )

    _run_soul_review_tick(
        persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
    )

    cursor_path = persona_dir / "draft_space_review_cursor.json"
    assert cursor_path.exists(), "cursor file must be written after a review pass"
    data = _json.loads(cursor_path.read_text())
    assert "last_seen" in data, "cursor must contain a last_seen timestamp"
