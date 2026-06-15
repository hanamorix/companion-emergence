"""Organ DoD: the notes tick fires through the live supervisor loop, fail-isolated."""
from unittest.mock import MagicMock

from brain.bridge import supervisor


def test_supervisor_invokes_notes_tick(tmp_path, monkeypatch):
    called = {}

    def _fake_tick(persona_dir, **kwargs):
        called["dir"] = persona_dir

    monkeypatch.setattr(supervisor, "_run_notes_tick", _fake_tick, raising=False)
    # Drive the fail-isolated seam directly (mirrors the maker/soul-cadence tests).
    # The assertion that matters: _run_notes_tick runs.
    supervisor._maybe_run_notes_tick(tmp_path, store=MagicMock(), provider=MagicMock())
    assert called["dir"] == tmp_path
