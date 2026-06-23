import json
from datetime import UTC, datetime

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
