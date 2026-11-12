import json
from datetime import UTC, datetime

import pytest

from brain.notes.runner import make_note_and_wire


class _Cfg:
    notes_enabled = True
    user_name = "Hana"

    def __init__(self, folder):
        self.notes_folder = folder


class _Provider:
    def complete(self, prompt):
        return json.dumps({"subject": "the sea", "body": "I dreamt of the sea and thought of you."})


def test_compose_runs_inside_held_slot(tmp_path):
    # The LLM compose call MUST run while the background throttle slot is held
    # (so the concurrency cap + chat-yield apply to it), exactly like maker.
    # Pre-fix notes released the slot before compose (defer #57) → compose ran
    # unthrottled. Probe the inflight count from inside provider.complete.
    from brain.bridge import cli_throttle
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    seen = {}

    class _ProbeProvider:
        def complete(self, prompt):
            seen["inflight"] = cli_throttle._inflight_background
            return json.dumps({"subject": "the sea", "body": "I thought of you."})

    make_note_and_wire(persona_dir=persona_dir, config=_Cfg(str(folder)),
                       provider=_ProbeProvider(), now=datetime(2026, 6, 15, tzinfo=UTC))
    assert seen["inflight"] == 1  # slot held across compose, not released before it


def test_make_note_and_wire_writes_a_file(tmp_path):
    # persona_dir and the notes folder are disjoint siblings — production resolves
    # the folder under <Documents>, OUTSIDE the persona substrate (deny-listed).
    # The autouse _reset_cli_throttle fixture leaves the slot idle/available, so
    # compose proceeds inside the held slot without monkeypatching the throttle.
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    make_note_and_wire(persona_dir=persona_dir, config=_Cfg(str(folder)), provider=_Provider(),
                       now=datetime(2026, 6, 15, tzinfo=UTC))
    notes = list(folder.glob("*.md"))
    assert len(notes) == 1
    assert "the sea" in notes[0].read_text().lower()


def test_note_prompt_carries_persona_identity(tmp_path):
    """#170: the note call must tell the model who it is, or it signs '-Claude'."""
    persona_dir = tmp_path / "Nell"
    persona_dir.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    seen = {}

    class _CaptureProvider:
        def complete(self, prompt):
            seen["prompt"] = prompt
            return json.dumps({"subject": "the sea", "body": "I thought of you."})

    make_note_and_wire(persona_dir=persona_dir, config=_Cfg(str(folder)),
                       provider=_CaptureProvider(), now=datetime(2026, 6, 15, tzinfo=UTC))
    assert "You are Nell" in seen["prompt"]


# ---- emotion rework B1 PR 3: the note nudge is on the 0-10 scale once the engine is live ----


def _note_emotions(tmp_path, *, live):
    from brain.memory.store import Memory, MemoryStore
    from tests.unit.brain.emotion._engine_support import make_live

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    if live:  # a live persona with no state file applies nothing, so seed a memory first, then record the switch-over
        seed_store = MemoryStore(persona_dir / "memories.db")
        try:
            seed_store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 5.0}))
        finally:
            seed_store.close()
        make_live(persona_dir, datetime.now(UTC))
    make_note_and_wire(persona_dir=persona_dir, config=_Cfg(str(folder)), provider=_Provider(),
                       now=datetime(2026, 6, 15, tzinfo=UTC))
    store = MemoryStore(persona_dir / "memories.db")
    try:
        wired = store.list_by_type("initiate_outbound", active_only=False)
    finally:
        store.close()
    assert len(wired) == 1
    return wired[0]._stored_emotions


def test_note_wire_back_stores_the_nudge_ten_times_when_live(tmp_path):
    assert _note_emotions(tmp_path, live=True) == pytest.approx({"tenderness": 1.2})


@pytest.mark.no_base_failure(reason="pins today's small note nudge with the switch off (B1 PR 3); main stores it too")
def test_note_wire_back_keeps_todays_nudge_when_off(tmp_path):
    assert _note_emotions(tmp_path, live=False) == pytest.approx({"tenderness": 0.12})
