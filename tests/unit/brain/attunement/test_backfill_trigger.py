"""Tests for backfill first-launch trigger detection."""
from __future__ import annotations

import json
from pathlib import Path

from brain.attunement.backfill import should_run_backfill
from brain.attunement.schemas import SCHEMA_VERSION


def _seed_buffer(persona_dir: Path, n_turns: int) -> None:
    convs = persona_dir / "active_conversations"
    convs.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_turns):
        lines.append(json.dumps({
            "id": f"m-{i}",
            "role": "user",
            "ts": f"2026-05-{(i % 28) + 1:02d}T12:00:00Z",
            "content": f"turn {i}",
        }))
    (convs / "main.jsonl").write_text("\n".join(lines) + "\n")


def test_does_not_trigger_for_empty_buffer(tmp_path: Path):
    assert should_run_backfill(tmp_path) is False


def test_does_not_trigger_below_threshold(tmp_path: Path):
    _seed_buffer(tmp_path, n_turns=9)
    assert should_run_backfill(tmp_path) is False


def test_triggers_for_buffer_above_threshold_no_prior_state(tmp_path: Path):
    _seed_buffer(tmp_path, n_turns=15)
    assert should_run_backfill(tmp_path) is True


def test_does_not_trigger_when_backfill_complete(tmp_path: Path):
    _seed_buffer(tmp_path, n_turns=15)
    (tmp_path / "attunement").mkdir(parents=True, exist_ok=True)
    (tmp_path / "attunement" / "backfill_state.json").write_text(json.dumps({
        "started_at": "2026-05-01T00:00:00Z",
        "total_windows": 5,
        "sampled_windows": 1,
        "processed_windows": 1,
        "patterns_emitted": 0,
        "status": "complete",
        "last_cursor": "window-0",
        "schema_version": SCHEMA_VERSION,
    }))
    assert should_run_backfill(tmp_path) is False


def test_triggers_when_prior_state_was_deferred(tmp_path: Path):
    _seed_buffer(tmp_path, n_turns=15)
    (tmp_path / "attunement").mkdir(parents=True, exist_ok=True)
    (tmp_path / "attunement" / "backfill_state.json").write_text(json.dumps({
        "started_at": "2026-05-01T00:00:00Z",
        "total_windows": 5,
        "sampled_windows": 1,
        "processed_windows": 0,
        "patterns_emitted": 0,
        "status": "deferred_to_next_day",
        "last_cursor": "",
        "schema_version": SCHEMA_VERSION,
    }))
    assert should_run_backfill(tmp_path) is True


def test_triggers_when_prior_state_is_corrupt(tmp_path: Path):
    _seed_buffer(tmp_path, n_turns=15)
    (tmp_path / "attunement").mkdir(parents=True, exist_ok=True)
    (tmp_path / "attunement" / "backfill_state.json").write_text("not valid json {")
    # Corrupt state → treated as missing → trigger fires
    assert should_run_backfill(tmp_path) is True


def test_supervisor_calls_should_run_backfill_during_persona_init(tmp_path: Path) -> None:
    """The supervisor startup hook must call should_run_backfill + run_backfill."""
    import threading
    from unittest.mock import patch

    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    _seed_buffer(persona_dir, n_turns=15)

    stop = threading.Event()
    stop.set()  # exit after startup — don't spin the tick loop

    with patch("brain.bridge.supervisor._attunement_should_run_backfill") as mock_should, \
         patch("brain.bridge.supervisor._attunement_run_backfill") as mock_run:
        mock_should.return_value = True

        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=EventBus(),
            tick_interval_s=0.0,
            heartbeat_interval_s=None,
            soul_review_interval_s=None,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
        )

        mock_should.assert_called_once_with(persona_dir)
        mock_run.assert_called_once_with(persona_dir)
