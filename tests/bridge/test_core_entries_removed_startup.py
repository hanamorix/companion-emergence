"""#801 section 8, criterion 9: where core feelings are taken out of her vocabulary file.

A core feeling (one of the framework's 30, including the four of #648) is never in her file. Any entry the file holds for one is
removed through the shared writer at every bridge start-up (in the vocabulary load, before the twin merge and before the check on
`emotion_state.json`) and in the rebuild child (`nell supervisor rebuild`, before its rebuild). The `nell` commands' own start
(`_engine_ready_or_refuse`) does nothing to her file. The old one-time swap is gone and its marker file is left alone. Personas here
are tmp directories with synthetic rows and vocabulary files; nothing touches a real persona.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import cli as brain_cli
from brain import dev_constants
from brain.bridge import server
from brain.bridge.server import build_app
from brain.cli import main
from brain.emotion import engine, persona_loader, vocabulary, vocabulary_file
from brain.paths import get_persona_dir
from tests.unit.brain.emotion._engine_support import local_day, local_noon, make_persona, row

T35 = local_noon(local_day(35))
VOCAB = "emotion_vocabulary.json"
OLD_MARKER = "vocabulary_swap_648.json"  # written by the removed swap; never read or written now


def _entry(name: str, description: str = "her own words", days: float | None = 6.0, category: str = "persona_extension") -> dict:
    return {"name": name, "description": description, "category": category, "decay_half_life_days": days, "intensity_clamp": 10.0}


def _write_vocab(persona: Path, entries: list) -> None:
    (persona / VOCAB).write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")


def _names(persona: Path) -> list[str]:
    return [e["name"] for e in json.loads((persona / VOCAB).read_text(encoding="utf-8"))["emotions"]]


def _with_core_entries(persona: Path) -> None:
    """A file holding two core entries (`contentment` is one of the four of #648, `joy` an older core feeling) and one of her own."""
    _write_vocab(persona, [_entry("contentment"), _entry("joy", days=9.0, category="core"), _entry("her own word", days=4.0)])


@pytest.fixture
def cli_persona(tmp_path: Path) -> Path:
    persona = get_persona_dir("nell")
    make_persona(persona, [row(T35, emotions={"joy": 6.0, "contentment": 5.0}, mid="seed")])
    (persona / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}', encoding="utf-8")
    _with_core_entries(persona)
    return persona


def _start_bridge(persona_dir: Path) -> None:
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass


# --- the bridge ------------------------------------------------------------------------------------------------------


def test_the_first_bridge_start_removes_both_core_entries_and_a_second_start_removes_nothing(persona_dir, caplog):
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0, "contentment": 5.0}, mid="seed")])
    _with_core_entries(persona_dir)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_file"):
        _start_bridge(persona_dir)
    assert _names(persona_dir) == ["her own word"]
    assert "removed core feeling entries" in caplog.text
    after_first = (persona_dir / VOCAB).read_bytes()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_file"):
        _start_bridge(persona_dir)
    assert (persona_dir / VOCAB).read_bytes() == after_first
    assert "removed core feeling entries" not in caplog.text


@pytest.mark.no_base_failure(
    reason="the engine already took a core feeling's half-life from the framework since step 4 (the old swap also wrote the framework values into the file); this pins the end state of criterion 9 after the removal"
)
def test_after_the_removal_the_engine_takes_a_core_half_life_from_the_framework(persona_dir):
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0, "contentment": 5.0}, mid="seed")])
    _with_core_entries(persona_dir)
    _start_bridge(persona_dir)
    live = engine.engine_for(persona_dir)
    assert live.half_life("joy") == dev_constants.EMOTION_HALF_LIFE_DAYS["joy"]
    assert live.half_life("contentment") == dev_constants.EMOTION_HALF_LIFE_DAYS["contentment"]


def test_adding_a_core_feeling_through_the_shared_function_is_refused_after_the_start(persona_dir):
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    _with_core_entries(persona_dir)
    _start_bridge(persona_dir)
    assert vocabulary_file.add(persona_dir / VOCAB, [("joy", None)])[0].outcome == "refused"
    assert _names(persona_dir) == ["her own word"]


def test_the_removal_runs_in_the_vocabulary_load_before_the_state_check(persona_dir, emotion_engine_stub, monkeypatch):
    order: list[str] = []
    real_remove = vocabulary_file.remove_core_entries
    monkeypatch.setattr(vocabulary_file, "remove_core_entries", lambda path: (order.append("remove"), real_remove(path))[1])
    monkeypatch.setattr(server, "run_state_names_pass", lambda pd: order.append("state pass"))
    _with_core_entries(persona_dir)
    _start_bridge(persona_dir)
    assert order == ["remove", "state pass"]


def test_the_old_marker_file_is_left_on_disk_and_never_read_or_written(persona_dir):
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    _with_core_entries(persona_dir)
    marker = persona_dir / OLD_MARKER
    marker.write_text("not even json", encoding="utf-8")  # unreadable on purpose: nothing may read it
    _start_bridge(persona_dir)
    assert marker.read_text(encoding="utf-8") == "not even json"
    assert _names(persona_dir) == ["her own word"]


@pytest.mark.no_base_failure(
    reason="control: the old swap also created no file for a persona that has none; this pins that the removal does not create one either"
)
def test_a_start_with_no_vocabulary_file_creates_none(persona_dir):
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    assert not (persona_dir / VOCAB).exists()
    _start_bridge(persona_dir)
    assert not (persona_dir / VOCAB).exists() or "joy" not in _names(persona_dir)


def test_a_removal_that_raises_is_logged_and_the_twin_merge_and_the_start_go_on(persona_dir, monkeypatch, caplog):
    def boom(path):
        raise RuntimeError("a bug")

    monkeypatch.setattr(vocabulary_file, "remove_core_entries", boom)
    _write_vocab(persona_dir, [_entry("tender longing"), _entry("tender_longing")])
    with caplog.at_level(logging.ERROR, logger="brain.bridge.server"):
        _start_bridge(persona_dir)
    assert _names(persona_dir) == ["tender longing"]  # the merge still ran
    assert "the removal of core feelings" in caplog.text


# --- the rebuild child and the nell commands ---------------------------------------------------------------------------


def _file_names_seen_by_ensure(monkeypatch: pytest.MonkeyPatch, seen: list[list[str]], persona: Path) -> None:
    """Record the names her file holds at the moment `ensure_engine_ready` is called, then run the real one."""
    real = engine.ensure_engine_ready

    def recording(persona_dir: Path, **kwargs: object) -> None:
        seen.append(_names(persona))
        real(persona_dir, **kwargs)

    monkeypatch.setattr(engine, "ensure_engine_ready", recording)


def test_the_rebuild_child_removes_the_core_entries_before_its_rebuild(cli_persona, monkeypatch):
    seen: list[list[str]] = []
    _file_names_seen_by_ensure(monkeypatch, seen, cli_persona)
    assert brain_cli._supervisor_rebuild_handler(argparse.Namespace(persona="nell")) == 0
    assert seen == [["her own word"]]


def test_a_removal_that_raises_does_not_stop_the_rebuild_child(cli_persona, monkeypatch, caplog):
    def boom(path):
        raise RuntimeError("a bug")

    monkeypatch.setattr(vocabulary_file, "remove_core_entries", boom)
    seen: list[list[str]] = []
    _file_names_seen_by_ensure(monkeypatch, seen, cli_persona)
    with caplog.at_level(logging.WARNING, logger="brain.cli"):
        assert brain_cli._supervisor_rebuild_handler(argparse.Namespace(persona="nell")) == 0
    assert len(seen) == 1  # the rebuild still ran
    assert "the rebuild goes on" in caplog.text
    assert _names(cli_persona) == ["contentment", "joy", "her own word"]  # the file is as it was


def test_a_command_start_runs_no_vocabulary_step_and_leaves_her_file_alone(cli_persona, emotion_engine_stub, monkeypatch):
    from brain.bridge import daemon

    order: list[str] = []
    monkeypatch.setattr(daemon, "run_rebuild_step", lambda persona_dir: (order.append("rebuild step"), None)[1])
    before = (cli_persona / VOCAB).read_bytes()
    assert brain_cli._engine_ready_or_refuse(cli_persona, False) is None
    assert order == ["rebuild step"]
    assert (cli_persona / VOCAB).read_bytes() == before


@pytest.mark.no_base_failure(
    reason="control: a dry run never ran the old swap and writes nothing now either; it pins that no vocabulary step was added to the dry-run path"
)
def test_a_dry_run_writes_nothing(cli_persona, emotion_engine_stub, capsys):
    emotion_engine_stub.live = False
    before = {str(p.relative_to(cli_persona)): p.read_bytes() for p in cli_persona.rglob("*") if p.is_file()}
    assert brain_cli._engine_ready_or_refuse(cli_persona, True) == 2
    after = {str(p.relative_to(cli_persona)): p.read_bytes() for p in cli_persona.rglob("*") if p.is_file()}
    assert after == before
    capsys.readouterr()


@pytest.mark.no_base_failure(
    reason="control: a read-only load never wrote her file on the old code either; it pins that the removal is not hooked into the loader"
)
def test_a_read_only_vocabulary_load_leaves_the_core_entries_in_the_file(cli_persona):
    persona_loader.load_persona_vocabulary(cli_persona / VOCAB, read_only=True)
    assert _names(cli_persona) == ["contentment", "joy", "her own word"]
    assert vocabulary.get("joy").description == "happiness, delight, warmth"  # the registry keeps the framework's entry


def test_the_command_entry_points_still_start_the_heartbeat_command_without_touching_her_file(cli_persona, monkeypatch):
    from brain.bridge import daemon

    class _StopError(Exception):
        pass

    monkeypatch.setattr(daemon, "run_rebuild_step", lambda persona_dir: None)
    monkeypatch.setattr(brain_cli, "get_provider", lambda *a, **k: (_ for _ in ()).throw(_StopError()))
    before = (cli_persona / VOCAB).read_bytes()
    with pytest.raises(_StopError):
        main(["heartbeat", "--persona", "nell"])
    assert (cli_persona / VOCAB).read_bytes() == before
