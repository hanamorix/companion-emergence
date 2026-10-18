"""#441 scope addition: `nell init --model` takes a full id. Syntax-only: no Claude CLI call at init."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brain import cli


def _init(monkeypatch, tmp_path: Path, model: str | None, extra: list[str] | None = None, inputs: list[str] | None = None):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    if inputs is not None:
        it = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(it))
    argv = ["init", "--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default"]
    if model is not None:
        argv += ["--model", model]
    with patch("subprocess.run", side_effect=AssertionError("no CLI call at init")), patch(
        "subprocess.Popen", side_effect=AssertionError("no CLI call at init")
    ), patch("shutil.which", side_effect=AssertionError("init must not even look for claude")):
        return cli.main(argv + (extra or []))


def _cfg(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "personas" / "siren" / "persona_config.json").read_text())


@pytest.mark.parametrize("model_id", ["claude-sonnet-5", "claude-opus-4-8", "claude-3-5-haiku-20241022"])
def test_full_id_is_accepted_and_written_verbatim(monkeypatch, tmp_path: Path, capsys, model_id):
    assert _init(monkeypatch, tmp_path, model_id) == 0
    cfg = _cfg(tmp_path)
    assert cfg["model"] == model_id
    assert "tier_models" not in cfg and "derived_unavailable" not in cfg
    assert "checked when first used" in capsys.readouterr().out


@pytest.mark.parametrize("alias", ["sonnet", "opus", "haiku"])
def test_alias_writes_exactly_what_it_wrote_before(monkeypatch, tmp_path: Path, capsys, alias):
    assert _init(monkeypatch, tmp_path, alias) == 0
    cfg = _cfg(tmp_path)
    assert set(cfg) == {
        "provider", "searcher", "mcp_audit_log_level", "user_name", "model", "last_opened_at",
        "user_pronouns", "notes_enabled", "notes_folder", "kindled_link_enabled", "kindled_relay_url",
    }
    assert cfg["model"] == alias
    assert "checked when first used" not in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["gpt4", "Claude-Sonnet-5", "claude-", "claude-sonnet-5\n", "claude-" + "a" * 58])
def test_invalid_model_is_refused_and_nothing_is_written(monkeypatch, tmp_path: Path, capsys, bad):
    with pytest.raises(SystemExit) as exc:  # argparse rejects it with the rule in its message
        _init(monkeypatch, tmp_path, bad)
    assert exc.value.code != 0
    assert "claude-" in capsys.readouterr().err
    assert not (tmp_path / "personas" / "siren").exists()


def test_interactive_prompt_accepts_a_typed_full_id(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    answers = iter(["siren", "Hana", "n", "default", "bogus!", "claude-sonnet-5"])  # the first model answer is invalid
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
    with patch("subprocess.run", side_effect=AssertionError), patch("subprocess.Popen", side_effect=AssertionError):
        assert cli.main(["init"]) == 0
    assert _cfg(tmp_path)["model"] == "claude-sonnet-5"


def test_force_reinit_with_a_new_model_clears_a_stale_unavailable_list(monkeypatch, tmp_path: Path):
    assert _init(monkeypatch, tmp_path, "claude-sonnet-5") == 0
    p = tmp_path / "personas" / "siren" / "persona_config.json"
    d = json.loads(p.read_text())
    d["derived_unavailable"] = ["claude-haiku-5"]
    p.write_text(json.dumps(d))
    assert _init(monkeypatch, tmp_path, "claude-opus-4-8", extra=["--force"]) == 0
    cfg = _cfg(tmp_path)
    assert cfg["model"] == "claude-opus-4-8" and "derived_unavailable" not in cfg
