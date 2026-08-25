"""#133 — the attunement backfill must respect the global CLI concurrency cap.

`run_backfill` loops `detector_fn` once per sampled window, and each call is a
fresh `claude -p` subprocess. It took no `cli_throttle` slot, so it could issue
up to `DAILY_BUDGET_DEFAULT = 150` back-to-back spawns while another background
engine held the single slot — and `_inflight_background` never saw it, so
nothing backed off for it either.

Its sibling `brain/ingest/emotion_backfill.py` already does this correctly
(per-iteration slot, idle yield, inter-call pacing); this is the port.

The subtle half is the *state*: the loop falls through to `status="complete"`.
A naive port that simply `break`s on a denied slot would mark the backfill
complete having skipped windows — permanently, since `should_run_backfill`
returns False once a completed state exists. Yielding must leave the state
resumable.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from brain.attunement.backfill import run_backfill
from brain.attunement.schemas import SCHEMA_VERSION, CurrentRead, DetectorOutput


def _make_buffer_file(persona_dir: Path, n_turns: int) -> None:
    """Seed active_conversations/main.jsonl with n_turns user messages."""
    convs = persona_dir / "active_conversations"
    convs.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "id": f"m-{i}",
            "role": "user",
            "ts": f"2026-05-{(i % 28) + 1:02d}T12:00:00Z",
            "content": f"this is message number {i} in the conversation",
        })
        for i in range(n_turns)
    ]
    (convs / "main.jsonl").write_text("\n".join(lines) + "\n")


def _fake_detector_output() -> DetectorOutput:
    return DetectorOutput(
        current_read=CurrentRead(
            ts="2026-05-31T12:00:00Z",
            source_turn_id="m-0",
            tone_label="warm",
            tone_justification="x",
            cadence_label="measured",
            cadence_justification="y",
            mood_valence=0.0,
            mood_intensity=0.0,
            predicted_arc_shape="z",
            schema_version=SCHEMA_VERSION,
        ),
        pattern_candidates=[],
    )


def test_denied_throttle_slot_stops_the_loop_without_marking_it_complete(
    tmp_path: Path,
):
    """No slot -> no `claude -p` spawn, and the backfill stays resumable."""
    _make_buffer_file(tmp_path, n_turns=25)

    calls: list[object] = []

    def counting_detector(buffer_slice, reply_text):  # noqa: ANN001
        calls.append(buffer_slice)
        return _fake_detector_output()

    # The single background slot is already held by another engine.
    with patch("brain.bridge.cli_throttle.acquire_background", return_value=False):
        state = run_backfill(tmp_path, detector_fn=counting_detector, delay_s=0)

    assert calls == [], (
        f"backfill spawned {len(calls)} claude -p call(s) with no throttle slot"
    )
    assert state.status != "complete", (
        "backfill marked itself complete while yielding — the skipped windows "
        "would never be revisited, because should_run_backfill returns False "
        f"once a completed state exists (status={state.status!r})"
    )


def test_an_active_chat_turn_makes_the_backfill_stand_down(tmp_path: Path):
    """The second guard: yield to interactive chat, disk-based and restart-robust.

    `cli_throttle` keeps its state in a module global on a monotonic clock, so it
    resets on every supervisor restart. `compute_active_session_hours` reads the
    conversation buffer off disk and survives one. Belt and braces, same as
    emotion_backfill — a restart mid-conversation must not hand the backfill a
    clean slate to burst from.
    """
    _make_buffer_file(tmp_path, n_turns=25)

    calls: list[object] = []

    def counting_detector(buffer_slice, reply_text):  # noqa: ANN001
        calls.append(buffer_slice)
        return _fake_detector_output()

    # Throttle would allow it; the user being mid-turn is what must stop it.
    with patch(
        "brain.body.session_hours.compute_active_session_hours", return_value=0.4
    ):
        state = run_backfill(tmp_path, detector_fn=counting_detector, delay_s=0)

    assert calls == [], (
        f"backfill spawned {len(calls)} claude -p call(s) during an active chat turn"
    )
    assert state.status != "complete", (
        f"backfill marked itself complete while yielding (status={state.status!r})"
    )
