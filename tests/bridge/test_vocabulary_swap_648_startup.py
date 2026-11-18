"""#648: where the one-time vocabulary swap runs at start-up.

The swap runs before the rebuild on every start-up path: in the rebuild child (`nell supervisor rebuild`, right before
`ensure_engine_ready`), in the `nell` commands' `_engine_ready_or_refuse` (right before the rebuild step), and in the bridge
lifespan (which no longer rebuilds, S107), so a persona's first engine rebuild already uses the swapped half-lives. A dry run writes nothing and
never calls it, and a read-only vocabulary load does not run it. A swap that fails does not stop start-up. Personas here are tmp
directories with synthetic rows and vocabulary files; nothing touches a real persona.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import cli as brain_cli
from brain.bridge import server
from brain.bridge.server import build_app
from brain.cli import main
from brain.emotion import engine, persona_loader
from brain.paths import get_persona_dir
from tests.unit.brain.emotion._engine_support import local_day, local_noon, make_persona, row

T35 = local_noon(local_day(35))
VOCAB = "emotion_vocabulary.json"
MARKER = "vocabulary_swap_648.json"


def _swap():
    """The swap module, imported inside the tests that patch it, so the tests that only look at the result need no import of it."""
    import importlib

    return importlib.import_module("brain.emotion.vocabulary_swap")


def _vocab_with_own_contentment(persona: Path) -> None:
    """A persona file whose own `contentment` has a half-life of 6 days (the owner's S14 case)."""
    entry = {
        "name": "contentment",
        "description": "her own words",
        "category": "persona_extension",
        "decay_half_life_days": 6.0,
        "intensity_clamp": 10.0,
    }
    (persona / VOCAB).write_text(json.dumps({"version": 1, "emotions": [entry]}), encoding="utf-8")


def _half_life_seen_by_ensure(monkeypatch: pytest.MonkeyPatch, seen: list[float]) -> None:
    """Record what the engine's half-life of contentment is at the moment `ensure_engine_ready` is called, then run the real one."""
    real = engine.ensure_engine_ready

    def recording(persona_dir: Path, **kwargs: object) -> None:
        seen.append(engine.engine_for(persona_dir).half_life("contentment"))
        real(persona_dir, **kwargs)

    monkeypatch.setattr(engine, "ensure_engine_ready", recording)


@pytest.fixture
def cli_persona(tmp_path: Path) -> Path:
    persona = get_persona_dir("nell")
    make_persona(persona, [row(T35, emotions={"joy": 6.0, "contentment": 5.0}, mid="seed")])
    (persona / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}', encoding="utf-8")
    _vocab_with_own_contentment(persona)
    return persona


class _StopError(Exception):
    """Raised in place of a command's first real work, so a test ends the command there."""


def _stop_at_the_first_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def first_work(*_args: object, **_kwargs: object):
        raise _StopError

    monkeypatch.setattr(brain_cli, "get_provider", first_work)


# --- the nell commands -----------------------------------------------------------------------------------------------


def _record_rebuild_step(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """Replace the commands' rebuild step (`daemon.run_rebuild_step`, S107) with a recorder that goes on (None)."""
    from brain.bridge import daemon

    monkeypatch.setattr(daemon, "run_rebuild_step", lambda persona_dir: (order.append("rebuild step"), None)[1])


def test_a_command_runs_the_swap_immediately_before_the_rebuild_step(cli_persona, emotion_engine_stub, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(_swap(), "run_vocabulary_swap_648", lambda persona_dir: order.append("swap"))
    _record_rebuild_step(monkeypatch, order)
    assert brain_cli._engine_ready_or_refuse(cli_persona, False) is None
    assert order == ["swap", "rebuild step"]


def test_a_dry_run_never_runs_the_swap_and_writes_nothing(cli_persona, emotion_engine_stub, monkeypatch, capsys):
    called: list[Path] = []
    monkeypatch.setattr(_swap(), "run_vocabulary_swap_648", lambda persona_dir: called.append(persona_dir))
    emotion_engine_stub.live = False
    before = {str(p.relative_to(cli_persona)) for p in cli_persona.rglob("*")}
    assert brain_cli._engine_ready_or_refuse(cli_persona, True) == 2
    assert called == []
    assert not (cli_persona / MARKER).exists()
    assert {str(p.relative_to(cli_persona)) for p in cli_persona.rglob("*")} == before
    capsys.readouterr()


def test_the_rebuild_step_of_a_command_start_already_sees_the_swapped_half_life(cli_persona, monkeypatch):
    """The rebuild step (which runs the rebuild child when one is needed) is reached only after the swap."""
    from brain.bridge import daemon

    seen: list[float] = []

    def step(persona_dir: Path) -> None:
        seen.append(engine.engine_for(persona_dir).half_life("contentment"))

    monkeypatch.setattr(daemon, "run_rebuild_step", step)
    _stop_at_the_first_work(monkeypatch)
    with pytest.raises(_StopError):
        main(["heartbeat", "--persona", "nell"])
    assert seen == [1.0]  # the persona's own 6 days was replaced before the step
    assert (cli_persona / MARKER).is_file()


def test_the_rebuild_child_runs_the_swap_before_its_rebuild(cli_persona, monkeypatch):
    """`nell supervisor rebuild` (the process every start's rebuild runs in) swaps first, so the first rebuild uses the swapped half-life."""
    import argparse

    seen: list[float] = []
    _half_life_seen_by_ensure(monkeypatch, seen)
    assert brain_cli._supervisor_rebuild_handler(argparse.Namespace(persona="nell")) == 0
    assert seen == [1.0]
    assert (cli_persona / MARKER).is_file()


def test_a_swap_that_fails_does_not_stop_a_command_start(cli_persona, emotion_engine_stub, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise RuntimeError("a bug")

    monkeypatch.setattr(_swap(), "read_json_result", boom)
    order: list[str] = []
    _record_rebuild_step(monkeypatch, order)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.vocabulary_swap"):
        assert brain_cli._engine_ready_or_refuse(cli_persona, False) is None
    assert order == ["rebuild step"]
    assert not (cli_persona / MARKER).exists()


def test_a_swap_that_fails_does_not_stop_the_rebuild_child(cli_persona, monkeypatch):
    import argparse

    def boom(*_a, **_k):
        raise RuntimeError("a bug")

    monkeypatch.setattr(_swap(), "read_json_result", boom)
    seen: list[float] = []
    _half_life_seen_by_ensure(monkeypatch, seen)
    assert brain_cli._supervisor_rebuild_handler(argparse.Namespace(persona="nell")) == 0
    assert len(seen) == 1  # the rebuild still ran
    assert not (cli_persona / MARKER).exists()


@pytest.mark.no_base_failure(
    reason="control for the new swap: a read-only load never wrote a marker on the old code either; it pins that the swap is not hooked into the loader"
)
def test_a_read_only_vocabulary_load_does_not_run_the_swap(cli_persona):
    persona_loader.load_persona_vocabulary(cli_persona / VOCAB, read_only=True)
    assert not (cli_persona / MARKER).exists()
    assert json.loads((cli_persona / VOCAB).read_text(encoding="utf-8"))["emotions"][0]["decay_half_life_days"] == 6.0


# --- the bridge ------------------------------------------------------------------------------------------------------


def test_lifespan_runs_the_swap_after_the_vocabulary_load_and_no_rebuild(persona_dir, emotion_engine_stub, monkeypatch):
    """The bridge no longer rebuilds in its lifespan (S107); the swap still runs there, after the vocabulary load."""
    order: list[str] = []
    real_load = server.ensure_persona_vocabulary_loaded
    monkeypatch.setattr(server, "ensure_persona_vocabulary_loaded", lambda *a, **k: (order.append("vocabulary"), real_load(*a, **k))[1])
    monkeypatch.setattr(_swap(), "run_vocabulary_swap_648", lambda pd: order.append("swap"))
    real = emotion_engine_stub.ensure_ready
    emotion_engine_stub.ensure_ready = lambda **kw: (order.append("ensure"), real(**kw))[1]
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert order == ["vocabulary", "swap"]


def test_the_bridge_start_swaps_the_persona_once(persona_dir):
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0, "contentment": 5.0}, mid="seed")])
    _vocab_with_own_contentment(persona_dir)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert (persona_dir / MARKER).is_file()
    entry = json.loads((persona_dir / VOCAB).read_text(encoding="utf-8"))["emotions"][0]
    assert entry["decay_half_life_days"] == 1.0


def test_a_swap_that_fails_does_not_stop_the_bridge(persona_dir, emotion_engine_stub, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("a bug")

    monkeypatch.setattr(_swap(), "read_json_result", boom)
    _vocab_with_own_contentment(persona_dir)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert not (persona_dir / MARKER).exists()
