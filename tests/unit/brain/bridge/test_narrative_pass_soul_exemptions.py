"""#502 (from #431): a failed soul-exemption read during the narrative pass is logged as the narrative pass.

Mirrors test_p3_change4_dedup_sweep.py::test_431_failed_exemption_read_during_a_sweep_is_logged_as_the_sweep, through
the real ``_run_narrative_memory_pass``: a directory where soul_candidates.jsonl should be makes the read fail.
"""

from __future__ import annotations

import logging

from brain.bridge import supervisor
from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider


def test_failed_exemption_read_during_the_narrative_pass_is_logged_as_narrative(tmp_path, caplog):
    persona_dir = tmp_path / "personas" / "canary"
    persona_dir.mkdir(parents=True)
    (persona_dir / "soul_candidates.jsonl").mkdir()

    with caplog.at_level(logging.WARNING):
        supervisor._run_narrative_memory_pass(persona_dir, FakeProvider(), EventBus())

    loader = [r.getMessage() for r in caplog.records if r.name == "brain.forgetting"]
    assert loader and all(m.startswith("narrative: ") for m in loader), loader
    assert not (persona_dir / "memories.db").exists()  # skipped before any store was opened
