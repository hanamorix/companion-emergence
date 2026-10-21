"""#440 - a full model id in the tunables file reaches the Claude CLI's --model argv unchanged (#529: the chat model
is the install-wide `model.interactive_chat` line, not a persona file field)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from brain import tunables
from brain.bridge import provider as provider_mod
from brain.bridge.chat import ChatMessage
from brain.bridge.provider import _MAX_TURN_BUDGET_USD, get_provider


def _persona(tmp_path: Path, model: str) -> Path:
    d = tmp_path / "persona"
    d.mkdir()
    (d / "persona_config.json").write_text(json.dumps({"provider": "claude-cli", "searcher": "noop"}))
    tunables.set_overrides({"model.interactive_chat": model})
    return d


def _proc(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def _model_arg(cmd: list[str]) -> str:
    return cmd[cmd.index("--model") + 1]


@pytest.mark.parametrize("model_id", ["claude-sonnet-5", "claude-opus-4-8", "claude-3-5-haiku-20241022"])
def test_generate_argv_carries_the_full_id(tmp_path: Path, model_id: str):
    _persona(tmp_path, model_id)
    prov = get_provider("claude-cli")
    seen: list[list[str]] = []

    def _run(cmd, **kw):
        seen.append(list(cmd))
        return _proc(json.dumps({"is_error": False, "result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_run):
        prov.generate("hi")
    assert _model_arg(seen[0]) == model_id


def test_chat_argv_carries_the_full_id(tmp_path: Path):
    d = _persona(tmp_path, "claude-opus-4-8")
    prov = get_provider("claude-cli")
    seen: list[list[str]] = []

    def _run(cmd, **kw):
        seen.append(list(cmd))
        return _proc(json.dumps({"result": "ok"}))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_run):
        prov.chat(
            [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="hi")],
            tools=[{"name": "x"}],
            options={"persona_dir": str(d)},
        )
    assert _model_arg(seen[0]) == "claude-opus-4-8"


def test_invalid_id_in_the_file_does_not_reach_argv(tmp_path: Path):
    _persona(tmp_path, "Claude-Sonnet-5")
    prov = get_provider("claude-cli")
    seen: list[list[str]] = []
    with patch(
        "brain.bridge.provider.subprocess.run",
        side_effect=lambda cmd, **kw: seen.append(list(cmd)) or _proc(json.dumps({"is_error": False, "result": "ok"})),
    ):
        prov.generate("hi")
    assert _model_arg(seen[0]) == "claude-sonnet-5"  # the shipped default runs instead of the unusable text


# PINS of unchanged behaviour (they cannot fail from this change; the argv tests above are the evidence).
@pytest.mark.parametrize("alias", ["haiku", "sonnet", "opus"])
def test_budget_ceiling_for_a_full_id_equals_its_alias(alias: str):
    assert _MAX_TURN_BUDGET_USD(f"claude-{alias}-9-9") == _MAX_TURN_BUDGET_USD(alias)


def test_budget_ceiling_falls_back_for_an_unrecognised_family():
    assert _MAX_TURN_BUDGET_USD("claude-zzz-9") == 1.50
    assert provider_mod._BUDGET_FALLBACK == 1.50
