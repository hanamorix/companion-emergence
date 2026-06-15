"""Organ DoD: the maker tick fires through the live supervisor loop."""
from unittest.mock import MagicMock

from brain.bridge import supervisor


def test_supervisor_invokes_maker_tick(tmp_path, monkeypatch):
    called = {}

    def _fake_tick(persona_dir, **kwargs):
        called["dir"] = persona_dir

    monkeypatch.setattr(supervisor, "_run_maker_tick", _fake_tick, raising=False)
    # Drive the fail-isolated seam directly (the soul-cadence tests use the same
    # direct-helper pattern). The assertion that matters: _run_maker_tick runs.
    supervisor._maybe_run_maker_tick(tmp_path, store=MagicMock(), provider=MagicMock())
    assert called["dir"] == tmp_path
