"""#523: `nell memory backfill-extraction` (B14, the CLI half of B8) and the CLI chat's exit flush (C11e).

Synthetic persona only (persona Kestrel, user Bob, made-up words); stub providers, no model call.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import cli
from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.ingest import extraction_backfill as eb
from brain.ingest import pipeline
from brain.ingest.buffer import append_archive, ingest_turn
from brain.ingest.ingest_utils import chunk_turns

_CHUNK = 600
_OLD = datetime(2026, 8, 3, 19, 0, 0, tzinfo=UTC)


class _RunawayLoopError(AssertionError):
    """Raised by the hard bounds below, so a broken stop rule FAILS the test at once instead of looping until the CI job times
    out. The pass bound raises from the wrapper around ``run_central_pass``, outside every handler of the code under test; the
    call bound is a second net (inside the extractor it counts as a failed call)."""


_CALL_LIMIT = 60
_PASS_LIMIT = 60


@pytest.fixture(autouse=True)
def _bounded_central_passes(monkeypatch):
    from brain.bridge import central_cadence

    real = central_cadence.run_central_pass
    passes = {"n": 0}

    def _bounded(*a, **k):
        passes["n"] += 1
        if passes["n"] > _PASS_LIMIT:
            raise _RunawayLoopError(f"more than {_PASS_LIMIT} central passes: a stop rule is broken")
        return real(*a, **k)

    monkeypatch.setattr(central_cadence, "run_central_pass", _bounded)


class _Stub(LLMProvider):
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) > _CALL_LIMIT:
            raise _RunawayLoopError(f"more than {_CALL_LIMIT} model calls: a stop rule is broken")
        return "[]"

    def name(self) -> str:
        return "stub-cli"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="ok", tool_calls=())


@pytest.fixture
def persona(tmp_path: Path, monkeypatch) -> Path:
    from brain import paths

    d = tmp_path / "personas" / "Kestrel"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop", "user_name": "Bob"}))
    (d / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": []}))
    monkeypatch.setattr(paths, "get_home", lambda: tmp_path)
    monkeypatch.setattr(eb, "extraction_chunk_chars", lambda: _CHUNK)
    monkeypatch.setattr(pipeline, "extraction_chunk_chars", lambda: _CHUNK, raising=False)
    monkeypatch.setattr(cli, "_bridge_running", lambda _pd: None)
    return d


@pytest.fixture
def stub(monkeypatch) -> _Stub:
    from brain.bridge import model_tier

    s = _Stub()
    monkeypatch.setattr(model_tier, "build_tier_provider", lambda persona_dir, tier: s)
    return s


def _rows(sid: str, n: int) -> list[dict]:
    out = []
    for i in range(n):
        t = _OLD + timedelta(seconds=i * 60)
        out.append({"session_id": sid, "speaker": "user", "text": f"zc{i:03d}u " + "m" * 90, "ts": t.isoformat(timespec="seconds")})
        out.append({"session_id": sid, "speaker": "assistant", "text": f"zc{i:03d}a " + "n" * 90,
                    "ts": (t + timedelta(seconds=1)).isoformat(timespec="seconds")})
    return out


def _archived_history(persona: Path, n: int = 6) -> list[dict]:
    rows = _rows("sess_cli", n)
    append_archive(persona, "sess_cli", rows)
    return rows


def _nell(*args: str) -> int:
    return cli.main(["memory", "backfill-extraction", "--persona", "Kestrel", *args])


def test_b14_the_command_parses_every_flag() -> None:
    parser = cli._build_parser()
    for flags in ([], ["--all"], ["--dry-run"], ["--background"], ["--status"], ["--stop"], ["--restart"], ["--restart", "--all"]):
        ns = parser.parse_args(["memory", "backfill-extraction", "--persona", "Kestrel", *flags])
        assert ns.func is cli._memory_backfill_extraction_handler


def test_b14_the_default_mode_is_heads_and_all_selects_everything(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--background") == 0
    assert eb.load_state(persona)["mode"] == "heads"
    assert _nell("--restart", "--all", "--background") == 0
    assert eb.load_state(persona)["mode"] == "all"
    assert stub.prompts == []


def test_b14_the_other_modes_flag_is_refused_naming_restart(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--background") == 0
    capsys.readouterr()
    assert _nell("--all", "--background") == 1
    assert "--restart" in capsys.readouterr().out
    assert eb.load_state(persona)["mode"] == "heads"


def test_b14_status_stop_and_a_rerun_after_complete(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--background") == 0
    capsys.readouterr()
    assert _nell("--status") == 0
    out = capsys.readouterr().out
    assert "mode=all" in out and "missing=0" in out and "discarded=0" in out
    assert _nell("--stop") == 0
    assert eb.load_state(persona)["status"] == "stopped"
    job = eb.backfill_gated_job(persona, store_ctx=None, hebbian_ctx=None, provider_factory=lambda: stub)
    job.run()  # the next job run does nothing
    assert stub.prompts == []
    assert _nell("--all") == 0  # re-running resumes and, with no bridge, runs in the foreground to the end
    assert eb.load_state(persona)["status"] == "complete"
    calls = len(stub.prompts)
    assert calls == len(chunk_turns(_rows("sess_cli", 6), max_chars=_CHUNK, user_name="Bob", assistant_name="Kestrel"))
    capsys.readouterr()
    assert _nell("--all") == 0
    assert len(stub.prompts) == calls
    assert "complete" in capsys.readouterr().out


def test_b14_dry_run_prints_both_modes_and_writes_nothing(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--dry-run") == 0
    out = capsys.readouterr().out
    assert "mode heads" in out and "mode all" in out and "sess_cli" in out
    assert not eb.state_path(persona).exists()
    assert stub.prompts == []


def test_b8_with_a_bridge_running_the_cli_only_arms_the_plan(persona, stub, monkeypatch, capsys) -> None:
    _archived_history(persona)
    monkeypatch.setattr(cli, "_bridge_running", lambda _pd: object())
    assert _nell("--all") == 0
    out = capsys.readouterr().out
    assert eb.load_state(persona)["status"] == "armed"
    assert stub.prompts == []
    assert "restart" in out


def test_b8_background_says_what_happens_without_a_bridge(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--background") == 0
    out = capsys.readouterr().out
    assert "no bridge" in out and "restart" in out
    assert stub.prompts == []


# ---------------------------------------------------------------------------
# C11(e): the CLI chat's exit flush extracts every chunk
# ---------------------------------------------------------------------------


def test_c11e_the_cli_exit_flush_makes_every_call(persona, monkeypatch) -> None:
    from types import SimpleNamespace

    import brain.chat.engine as engine

    sid = "sess_cli_flush"
    for r in _rows(sid, 6):
        ingest_turn(persona, dict(r, session_id=sid))
    assert len(chunk_turns(_rows(sid, 6), max_chars=_CHUNK, user_name="Bob", assistant_name="Kestrel")) == 3
    stub = _Stub()
    monkeypatch.setattr(cli, "get_provider", lambda _name: stub)
    monkeypatch.setattr(cli, "_start_floor_bootstrap_for_direct_chat", lambda *a, **k: None)
    monkeypatch.setattr(engine, "respond", lambda *a, **k: SimpleNamespace(content="ok", session_id=sid, tool_invocations=[]))
    assert cli.main(["chat", "--persona", "Kestrel", "--no-bridge", "hello"]) == 0
    assert len(stub.prompts) == 3


@pytest.mark.parametrize("flag", ["--status", "--stop", "--dry-run"])
def test_restart_with_a_read_or_stop_action_is_an_error(persona, stub, capsys, flag) -> None:
    _archived_history(persona)
    assert _nell("--restart", flag) == 2
    assert "--restart cannot be combined" in capsys.readouterr().err
    assert not eb.state_path(persona).exists()


def test_stop_with_no_plan_creates_nothing(persona, capsys) -> None:
    before = sorted(p.name for p in persona.iterdir())
    assert _nell("--stop") == 0
    assert "no backfill plan" in capsys.readouterr().out
    assert sorted(p.name for p in persona.iterdir()) == before


def test_restart_of_a_plan_with_done_targets_says_they_are_extracted_again(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all") == 0  # runs to complete in the foreground
    capsys.readouterr()
    assert _nell("--restart", "--all", "--background") == 0
    out = capsys.readouterr().out
    assert "done" in out and "extracted again" in out
