"""`nell model show | set | set-tier | clear-tier` (#441, #529: C10, C30 c/d, C42 e).

The model is one install-wide setting per job in the tunables file. `set` applies the one write rule to the raw value
and, for a value it is about to write that is not an alias word, runs the picker's one-call check first. There is no
`--persona` and no bridge round trip: every bridge reads the file on every call.
"""

from __future__ import annotations

import json

import pytest

from brain import cli, tunables
from brain.bridge import model_error, server

_CHAT = "model.interactive_chat"
_PATH = lambda: tunables._file_path()  # noqa: E731


def _run(argv: list[str]) -> int:
    return cli.main(["model", *argv])


def _overrides() -> dict:
    p = _PATH()
    return json.loads(p.read_text(encoding="utf-8")).get("overrides", {}) if p.exists() else {}


# ---------------------------------------------------------------------------
# set: the write rule, the probe decisions, the report
# ---------------------------------------------------------------------------


def test_set_sonnet_removes_an_existing_line_with_no_probe(capture_claude_argv, capsys) -> None:
    tunables.set_overrides({_CHAT: "claude-opus-5"})
    assert _run(["set", "sonnet"]) == 0
    assert _CHAT not in _overrides()
    assert capture_claude_argv.argvs == []  # nothing new is chosen, so nothing to check
    out = capsys.readouterr().out
    assert "claude-sonnet-5" in out and "following the shipped default" in out


def test_set_sonnet_with_no_line_leaves_the_file_alone(capture_claude_argv) -> None:
    assert _run(["set", "sonnet"]) == 0
    assert not _PATH().exists()
    tunables.set_overrides({"other.key": 1})
    before = _PATH().stat().st_mtime_ns
    assert _run(["set", "sonnet"]) == 0
    assert _PATH().stat().st_mtime_ns == before  # no lock round trip, no file churn
    assert capture_claude_argv.argvs == []


def test_set_opus_writes_the_opus_id_after_one_probe(capture_claude_argv, capsys) -> None:
    assert _run(["set", "opus"]) == 0
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert capture_claude_argv.models == ["claude-opus-5"]
    assert "claude-opus-5" in capsys.readouterr().out


def test_set_haiku_writes_the_alias_with_no_probe(capture_claude_argv) -> None:
    assert _run(["set", "haiku"]) == 0
    assert _overrides()[_CHAT] == "haiku"
    assert capture_claude_argv.argvs == []


def test_set_a_typed_full_id_is_written_even_when_it_equals_the_default(capture_claude_argv) -> None:
    assert _run(["set", "claude-sonnet-5"]) == 0
    assert _overrides()[_CHAT] == "claude-sonnet-5"
    assert capture_claude_argv.models == ["claude-sonnet-5"]


def test_set_a_rejected_id_writes_nothing_and_exits_one(capture_claude_argv, capsys) -> None:
    capture_claude_argv.reject("claude-sonet-5")
    tunables.set_overrides({"other.key": 1})
    before = _PATH().read_bytes()
    assert _run(["set", "claude-sonet-5"]) == 1
    assert _PATH().read_bytes() == before
    assert "Claude rejected" in capsys.readouterr().err


def test_set_with_an_inconclusive_check_saves_unchecked(monkeypatch, capsys) -> None:
    monkeypatch.setattr(server, "_check_model_blocking", lambda model, timeout_s=None: ("inconclusive", "timeout"))
    assert _run(["set", "claude-opus-4-9"]) == 0
    assert _overrides()[_CHAT] == "claude-opus-4-9"
    assert "saved unchecked" in capsys.readouterr().out
    assert model_error.unconfirmed() == ["claude-opus-4-9"]


@pytest.mark.parametrize("bad", ["gpt4", "claude-", "Claude-Sonnet-5", "claude-x\n"])
def test_set_an_unusable_value_exits_one_and_names_the_rule(bad, capsys) -> None:
    try:
        rc = _run(["set", bad])
    except SystemExit as exc:  # pragma: no cover
        rc = exc.code
    assert rc == 1
    assert "lowercase letters" in capsys.readouterr().err
    assert not _PATH().exists()


def test_set_works_with_no_persona_and_no_bridge(capture_claude_argv) -> None:
    """No persona exists, no bridge runs, no HTTP: the command still writes the install-wide setting."""
    assert not (tunables._file_path().parent / "personas").exists()
    assert _run(["set", "opus"]) == 0


def test_set_has_no_persona_option() -> None:
    with pytest.raises(SystemExit):
        _run(["set", "opus", "--persona", "x"])
    with pytest.raises(SystemExit):
        _run(["show", "--persona", "x"])


def test_set_clears_the_record_of_the_id_it_writes_c30(capture_claude_argv) -> None:
    """C30 (c): re-picking an id that was rejected leaves no record behind."""
    from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT, model_for_tier

    capture_claude_argv.reject("claude-sonnet-4-9")
    tunables.set_overrides({_CHAT: "claude-sonnet-4-9"})
    from brain.bridge.provider import ClaudeCliProvider

    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("x")
    assert model_error.is_rejected("claude-sonnet-4-9")
    capture_claude_argv.accept("claude-sonnet-4-9")
    assert _run(["set", "claude-sonnet-4-9"]) == 0
    assert not model_error.is_rejected("claude-sonnet-4-9")
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == "claude-sonnet-4-9"


# ---------------------------------------------------------------------------
# set-tier and clear-tier
# ---------------------------------------------------------------------------


def test_set_tier_applies_the_same_rule_per_job(capsys) -> None:
    tunables.set_overrides({"model.background_generative": "claude-sonnet-4-6", "model.background_classifier": "claude-haiku-4-5"})
    assert _run(["set-tier", "background_generative", "sonnet"]) == 0
    assert "model.background_generative" not in _overrides()  # equals that job's default: the line is removed
    assert _run(["set-tier", "background-classifier", "haiku"]) == 0
    assert "model.background_classifier" not in _overrides()
    assert _run(["set-tier", "background_classifier", "sonnet"]) == 0
    assert _overrides()["model.background_classifier"] == "claude-sonnet-5"  # differs from haiku: the id is written
    assert _run(["set-tier", "compaction", "claude-haiku-4-5"]) == 0
    assert _overrides()["model.compaction"] == "claude-haiku-4-5"


def test_set_tier_clears_the_record_of_the_id_it_writes_c30(capture_claude_argv) -> None:
    from brain.bridge.model_tier import TIER_COMPACTION
    from brain.bridge.provider import ClaudeCliProvider

    capture_claude_argv.reject("claude-haiku-4-9")
    tunables.set_overrides({"model.compaction": "claude-haiku-4-9"})
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_COMPACTION).generate("x")
    assert model_error.is_rejected("claude-haiku-4-9")
    assert _run(["set-tier", "compaction", "claude-haiku-4-9"]) == 0
    assert not model_error.is_rejected("claude-haiku-4-9")


@pytest.mark.parametrize("bad", ["gpt4", "claude-", "claude-" + "x" * 70])
def test_set_tier_refuses_invalid_ids_and_leaves_the_file_alone(bad, capsys) -> None:
    tunables.set_overrides({"other.key": 1})
    before = _PATH().read_bytes()
    assert _run(["set-tier", "compaction", bad]) == 1
    assert _PATH().read_bytes() == before


def test_an_unknown_job_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _run(["set-tier", "not-a-job", "haiku"])
    with pytest.raises(SystemExit):
        _run(["clear-tier", "not-a-job"])


def test_clear_tier_removes_the_line_including_the_chat_job(capsys) -> None:
    tunables.set_overrides({"model.compaction": "claude-haiku-4-5", _CHAT: "claude-opus-5", "keep.me": 1})
    assert _run(["clear-tier", "compaction"]) == 0
    assert _run(["clear-tier", "interactive-chat"]) == 0
    assert _overrides() == {"keep.me": 1}
    assert "shipped default" in capsys.readouterr().out
    assert _run(["clear-tier", "compaction"]) == 0  # clearing nothing is fine


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_lists_each_job_with_default_or_setting(capsys) -> None:
    tunables.set_overrides({"model.compaction": "claude-haiku-4-5", _CHAT: "claude-opus-5"})
    assert _run(["show"]) == 0
    out = capsys.readouterr().out
    assert "interactive-chat: claude-opus-5 (your setting)" in out
    assert "compaction: claude-haiku-4-5 (your setting)" in out
    assert "background-generative: claude-sonnet-5 (shipped default)" in out
    assert "attunement-detector: claude-haiku-4-5-20251001 (shipped default)" in out
    assert "runtime fallbacks appear in the app" in out
    assert "every persona on this install" in out


def test_show_names_the_effective_id_when_a_fallback_is_active(capture_claude_argv, capsys) -> None:
    from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT
    from brain.bridge.provider import ClaudeCliProvider

    capture_claude_argv.reject("claude-sonnet-4-9")
    tunables.set_overrides({_CHAT: "claude-sonnet-4-9"})
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("x")
    assert _run(["show"]) == 0
    assert "running claude-sonnet-5 now" in capsys.readouterr().out


def test_show_warns_for_an_invalid_setting_and_never_edits_it(capsys) -> None:
    tunables.set_overrides({_CHAT: "banana"})
    before = _PATH().read_bytes()
    assert _run(["show"]) == 0
    out = capsys.readouterr().out
    assert "warning" in out and "'banana'" in out and "claude-sonnet-5 is used instead" in out
    assert _PATH().read_bytes() == before


def test_show_is_read_only_and_says_when_a_migration_is_pending(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    persona = tmp_path / "personas" / "ada"
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text(json.dumps({"provider": "claude-cli", "model": "opus"}))
    assert _run(["show"]) == 0
    assert "still holds a model" in capsys.readouterr().out
    assert not (tmp_path / "model_migration.done").exists()  # show never migrates
    assert not _PATH().exists()


# ---------------------------------------------------------------------------
# C42 (e): an unwritable settings file
# ---------------------------------------------------------------------------


def test_a_damaged_settings_file_prints_the_reason_and_the_path_and_exits_non_zero(capsys) -> None:
    _PATH().parent.mkdir(parents=True, exist_ok=True)
    _PATH().write_text("{not json")
    assert _run(["set", "opus"]) == 1
    err = capsys.readouterr().err
    assert "could not be saved" in err and "tunables.json" in err
    assert _PATH().read_text() == "{not json"
    assert _run(["set-tier", "compaction", "claude-haiku-4-5"]) == 1
    assert _run(["clear-tier", "compaction"]) == 1


def test_a_wedged_lock_makes_the_command_fail_without_writing(wedged_lock, monkeypatch, capsys) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    tunables.set_overrides({"other.key": 1})
    before = _PATH().read_bytes()
    with wedged_lock(_PATH()):
        assert _run(["set-tier", "compaction", "claude-haiku-4-5"]) == 1
    assert _PATH().read_bytes() == before
    assert "could not lock" in capsys.readouterr().err
