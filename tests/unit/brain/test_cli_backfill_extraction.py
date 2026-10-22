"""#523: `nell memory backfill-extraction` and the CLI chat's exit flush.

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
    monkeypatch.setattr(cli, "_live_bridge_state", lambda _pd: None)
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
    for flags in (
        [], ["--all"], ["--dry-run"], ["--background"], ["--status"], ["--stop"], ["--restart"], ["--restart", "--all"], ["--yes"],
        ["--all", "--background", "--yes"],
    ):
        ns = parser.parse_args(["memory", "backfill-extraction", "--persona", "Kestrel", *flags])
        assert ns.func is cli._memory_backfill_extraction_handler


def test_b14_the_default_mode_is_heads_and_all_selects_everything(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--background", "--yes") == 0
    assert eb.load_state(persona)["mode"] == "heads"
    assert _nell("--restart", "--all", "--background", "--yes") == 0
    assert eb.load_state(persona)["mode"] == "all"
    assert stub.prompts == []


def test_b14_the_other_modes_flag_is_refused_naming_restart(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--background", "--yes") == 0
    capsys.readouterr()
    assert _nell("--all", "--background") == 1
    assert "--restart" in capsys.readouterr().out
    assert eb.load_state(persona)["mode"] == "heads"


def test_b14_status_stop_and_a_rerun_after_complete(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--background", "--yes") == 0
    capsys.readouterr()
    assert _nell("--status") == 0
    out = capsys.readouterr().out
    assert "mode=all" in out and "missing=0" in out and "discarded=0" in out
    assert _nell("--stop") == 0
    assert eb.load_state(persona)["status"] == "stopped"
    job = eb.backfill_gated_job(persona, store_ctx=None, hebbian_ctx=None, provider_factory=lambda: stub)
    job.run()  # the next job run does nothing
    assert stub.prompts == []
    assert _nell("--all", "--yes") == 0  # re-running resumes and, with no bridge, runs in the foreground to the end
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
    monkeypatch.setattr(cli, "_live_bridge_state", lambda _pd: object())
    assert _nell("--all", "--yes") == 0
    out = capsys.readouterr().out
    assert eb.load_state(persona)["status"] == "armed"
    assert stub.prompts == []
    assert "restart" in out


def test_b8_background_says_what_happens_without_a_bridge(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--background", "--yes") == 0
    out = capsys.readouterr().out
    assert "no bridge" in out and "restart" in out
    assert stub.prompts == []


# ---------------------------------------------------------------------------
# the CLI chat's exit flush extracts every chunk
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
    assert _nell("--all", "--yes") == 0  # runs to complete in the foreground
    capsys.readouterr()
    assert _nell("--restart", "--all", "--background", "--yes") == 0
    out = capsys.readouterr().out
    assert "done" in out and "extracted again" in out


# ---------------------------------------------------------------------------
# PR #524 review: 524-1 (persona emotion vocabulary without a bridge) and 524-2 (no spending without --yes)
# ---------------------------------------------------------------------------

_ZEPHYRIA = {
    "name": "zephyria",
    "description": "the lightness before a thing begins",
    "category": "persona_extension",
    "decay_half_life_days": 4.0,
}


@pytest.fixture
def zephyria_in_the_vocabulary(persona):
    """The persona's own vocabulary file holds one extension emotion; the process-wide registry is cleaned afterwards."""
    from brain.emotion import vocabulary

    (persona / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": [_ZEPHYRIA]}))
    assert vocabulary.get("zephyria") is None  # not registered yet: nothing but the CLI can register it
    yield
    vocabulary._unregister("zephyria")


class _EmotionStub(_Stub):
    """Every extraction returns one distinct item carrying the persona's extension emotion."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        super().generate(prompt, system=system)
        n = len(self.prompts)
        return json.dumps(
            [{"text": f"Bob felt a zephyr-light thing, number {n}", "label": "observation", "importance": 5, "emotions": {"zephyria": 7}}]
        )


def _pending_emotions(persona: Path) -> list[dict]:
    from brain.memory.pending import PendingQueue

    path = PendingQueue(persona).path
    return [json.loads(line).get("emotions", {}) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def test_524_1_the_backfill_without_a_bridge_keeps_a_persona_extension_emotion(
    persona, zephyria_in_the_vocabulary, monkeypatch
) -> None:
    from brain.bridge import model_tier

    emotional = _EmotionStub()
    monkeypatch.setattr(model_tier, "build_tier_provider", lambda persona_dir, tier: emotional)
    _archived_history(persona)
    assert _nell("--all", "--yes") == 0  # no bridge: the foreground run in this process
    kept = _pending_emotions(persona)
    assert kept, "the run committed nothing"
    assert all(e.get("zephyria") == 7.0 for e in kept), kept


def _says_calls(text: str) -> int:
    import re

    found = re.findall(r"about (\d+) model call", text)
    assert len(found) == 1, text
    return int(found[0])


def test_524_2_the_foreground_run_prints_the_count_and_with_yes_makes_exactly_that_many_calls(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--yes") == 0
    out = capsys.readouterr().out
    assert _says_calls(out) == len(stub.prompts) > 1


@pytest.mark.parametrize(
    "flags, bridge",
    [
        (["--all"], False),  # foreground
        (["--all", "--background"], False),  # --background
        (["--all"], True),  # the bridge is running: armed for it, which spends at lulls with no further prompt
        (["--all", "--background"], True),
        (["--restart", "--all"], False),
        (["--restart", "--all", "--background"], True),
    ],
)
def test_524_2_no_path_that_starts_spending_runs_without_yes(persona, stub, monkeypatch, capsys, flags, bridge) -> None:
    _archived_history(persona)
    if bridge:
        monkeypatch.setattr(cli, "_live_bridge_state", lambda _pd: object())
    assert _nell(*flags) == 1
    captured = capsys.readouterr()
    assert "--yes" in captured.err  # the refusal names the flag
    assert _says_calls(captured.out) == len(
        chunk_turns(_rows("sess_cli", 6), max_chars=_CHUNK, user_name="Bob", assistant_name="Kestrel")
    )  # and the count was shown first
    assert not eb.state_path(persona).exists()  # nothing was planned or armed
    assert stub.prompts == []


def test_524_2_the_default_lost_starts_mode_is_gated_too(persona, stub, capsys) -> None:
    rows = _rows("sess_cli", 140)  # one lull-free span of about 28,000 chars: its head was cut by the old 24,000 limit
    append_archive(persona, "sess_cli", rows)
    assert _nell("--dry-run") == 0
    assert "mode heads: sessions=1" in capsys.readouterr().out
    assert _nell() == 1
    captured = capsys.readouterr()
    assert _says_calls(captured.out) > 0 and "--yes" in captured.err
    assert not eb.state_path(persona).exists()
    assert stub.prompts == []


def test_524_2_a_restart_without_yes_leaves_the_existing_plan_alone(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--background", "--yes") == 0
    before = eb.load_state(persona)
    assert _nell("--restart", "--all", "--background") == 1
    assert eb.load_state(persona) == before


def test_524_2_a_resumed_plan_shows_the_calls_left_not_the_full_count(persona, stub, capsys) -> None:
    _archived_history(persona)
    chunks = chunk_turns(_rows("sess_cli", 6), max_chars=_CHUNK, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) >= 3
    assert _nell("--all", "--background", "--yes") == 0
    assert _says_calls(capsys.readouterr().out) == len(chunks)

    def _first_chunk_done(cur):
        cur["targets"][0]["done_through_ts"] = chunks[0].cursor_ts
        return cur

    eb.update_state(persona, _first_chunk_done)
    assert _nell("--stop") == 0
    capsys.readouterr()
    assert _nell("--all") == 1  # a stopped plan being resumed spends again: gated, and it shows what is left
    assert _says_calls(capsys.readouterr().out) == len(chunks) - 1
    assert eb.load_state(persona)["status"] == "stopped"
    assert _nell("--all", "--yes") == 0
    assert len(stub.prompts) == len(chunks) - 1


def test_524_2_nothing_to_spend_needs_no_yes(persona, stub, capsys) -> None:
    _archived_history(persona)
    assert _nell("--all", "--yes") == 0  # runs to complete
    capsys.readouterr()
    calls = len(stub.prompts)
    assert _nell("--all") == 0  # a complete plan: nothing to confirm
    assert "complete" in capsys.readouterr().out
    assert len(stub.prompts) == calls


@pytest.mark.parametrize("flag", ["--status", "--stop", "--dry-run"])
def test_524_2_yes_with_a_read_or_stop_action_is_harmless(persona, stub, capsys, flag) -> None:
    _archived_history(persona)
    assert _nell(flag, "--yes") == 0
    assert stub.prompts == []
