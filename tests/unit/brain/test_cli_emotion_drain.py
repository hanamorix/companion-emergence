"""The emotion engine's drain at the `nell` command call sites (Engine 4, #579).

`nell chat --no-bridge` drains after each reply and again after the pass-2 exit drain; `nell heartbeat` drains after the tick
and not under `--dry-run`. The engine is the shared recording stub (`emotion_engine_stub`), so a call that is missing, doubled
or misplaced shows in the order of the record.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brain import cli as brain_cli
from brain.cli import main


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8")
    (d / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def _patch_persona_dir(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain import paths

    monkeypatch.setattr(paths, "get_home", lambda: persona_dir.parent.parent)


@pytest.fixture(autouse=True)
def _reset_sessions():
    from brain.chat.session import reset_registry

    reset_registry()
    yield
    reset_registry()


def _calls(stub) -> list[str]:  # noqa: ANN001
    return [call[0] for call in stub.recorded_calls if call[0] == "note_active"]


def test_a_one_shot_chat_drains_after_the_reply_and_again_after_the_pass_two_exit_drain(
    emotion_engine_stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    order: list[str] = []
    monkeypatch.setattr(brain_cli, "_drain_pass2_at_exit", lambda persona_dir: order.append("pass2"))
    real_note_active = emotion_engine_stub.note_active
    emotion_engine_stub.note_active = lambda now: (order.append("drain"), real_note_active(now))[1]
    assert main(["chat", "--persona", "nell", "--no-bridge", "hello"]) == 0
    assert order == ["drain", "pass2", "drain"]
    assert len(_calls(emotion_engine_stub)) == 2
    assert emotion_engine_stub.recorded_calls[0] == ("ensure_ready",)  # the command makes the engine ready first (Engine 6)
    assert emotion_engine_stub.recorded_calls[1][1].tzinfo is not None  # an aware time


def test_the_repl_drains_after_each_reply_and_after_the_exit_pass_two_drain(
    emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    monkeypatch.setattr(brain_cli, "_drain_pass2_at_exit", lambda persona_dir: order.append("pass2"))
    real_note_active = emotion_engine_stub.note_active
    emotion_engine_stub.note_active = lambda now: (order.append("drain"), real_note_active(now))[1]
    with patch("builtins.input", side_effect=["one", "two", "exit"]):
        assert main(["chat", "--persona", "nell", "--no-bridge"]) == 0
    assert order == ["drain", "drain", "pass2", "drain"]


@pytest.fixture
def heartbeat_persona(persona_dir: Path) -> Path:
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(db_path=persona_dir / "memories.db")
    seed = Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"love": 9.0})
    seed.importance = 8.0
    store.create(seed, fold=False)
    store.close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def test_the_heartbeat_command_drains_after_the_tick(emotion_engine_stub, heartbeat_persona: Path) -> None:
    assert main(["heartbeat", "--persona", "nell", "--trigger", "open", "--provider", "fake"]) == 0
    assert len(_calls(emotion_engine_stub)) == 1


def test_the_heartbeat_command_does_not_drain_under_dry_run(emotion_engine_stub, heartbeat_persona: Path) -> None:
    emotion_engine_stub.live = True  # a dry run needs a switched persona (it cannot do the switch-over, it writes nothing)
    assert main(["heartbeat", "--persona", "nell", "--trigger", "manual", "--provider", "fake", "--dry-run"]) == 0
    assert _calls(emotion_engine_stub) == []
    assert main(["heartbeat", "--persona", "nell", "--trigger", "open", "--provider", "fake"]) == 0  # the control
    assert len(_calls(emotion_engine_stub)) == 1
