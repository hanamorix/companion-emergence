"""The bridge's start-up runs the check on the feeling names in `emotion_state.json` (#801 section 7, criterion 8).

Once per start, after her vocabulary has loaded and before the start-up drain; skipped, with the state file untouched, when the
vocabulary did not load. The pass itself is tested in tests/unit/brain/emotion/test_engine_state_names.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge import server
from brain.bridge.server import build_app
from brain.emotion import engine_files, persona_loader
from brain.emotion.engine_utils import Average, EmotionState, EngineState
from tests.unit.brain.emotion._engine_support import NOW


def _save_state_with_twin_keys(persona_dir: Path) -> None:
    emotions = {"joy": EmotionState(Average(4.0, 1.0), Average(2.0, 1.0)), "Joy": EmotionState(Average(8.0, 3.0), Average(6.0, 1.0))}
    assert engine_files.save_state(persona_dir, EngineState(saved_at=NOW, last_update=NOW, active_day="2026-06-10", emotions=emotions), verify=True)


def test_start_up_runs_the_pass_after_the_vocabulary_load_and_before_the_drain(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_load = server.ensure_persona_vocabulary_loaded
    monkeypatch.setattr(server, "ensure_persona_vocabulary_loaded", lambda *a, **k: (order.append("vocabulary"), real_load(*a, **k))[1])
    monkeypatch.setattr(server, "run_state_names_pass", lambda persona: order.append("state names"))
    real_note_active = emotion_engine_stub.note_active
    emotion_engine_stub.note_active = lambda now: (order.append("drain"), real_note_active(now))[1]
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert order == ["vocabulary", "state names", "drain"]


def test_start_up_merges_a_twin_state_key_through_the_real_pass(persona_dir: Path, emotion_engine_stub) -> None:
    _save_state_with_twin_keys(persona_dir)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    state = engine_files.read_state(persona_dir).state
    assert state is not None
    assert list(state.emotions) == ["joy"]
    assert state.emotions["joy"].short == Average(7.0, 4.0)


@pytest.mark.no_base_failure(reason="pins that a failed vocabulary load leaves the state file alone; the base start-up has no pass at all, so it leaves it alone there too")
def test_start_up_skips_the_pass_and_leaves_the_state_file_alone_when_the_vocabulary_did_not_load(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_state_with_twin_keys(persona_dir)
    before = engine_files.state_path(persona_dir).read_bytes()
    (persona_dir / "emotion_vocabulary.json").write_text('{"version": 1, "emotions": []}', encoding="utf-8")

    def broken(*_args: object, **_kwargs: object) -> tuple:
        raise OSError("the vocabulary cannot be read")

    monkeypatch.setattr(persona_loader, "load_persona_vocabulary_with_anomaly", broken)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert engine_files.state_path(persona_dir).read_bytes() == before  # the real pass would have merged the twins
