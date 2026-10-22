"""`nell model show | set | set-tier | clear-tier` (#441, #529: C10, C30 c/d, C42 e).

The model is one install-wide setting per job in tunables.toml. `set` WRITES the pick as the entry's value (nothing is ever
removed) and, for `opus` and any typed id, runs the picker's one-call check first. There is no `--persona` and no bridge
round trip: every bridge reads the file on every call.
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
    """The settings that differ from their shipped default, plus any ``other.*`` / ``keep.*`` key a test wrote."""
    from tests.tunables_file import differing_from_defaults, read_values

    if not _PATH().exists():
        return {}
    extra = {k: v for k, v in read_values(_PATH().parent).items() if k.startswith(("other.", "keep."))}
    return {**differing_from_defaults(), **extra}


# ---------------------------------------------------------------------------
# set: the write rule, the probe decisions, the report
# ---------------------------------------------------------------------------


def test_set_sonnet_writes_the_chat_default_id_with_no_probe_and_keeps_the_entry(capture_claude_argv, capsys) -> None:
    tunables.set_values({_CHAT: "claude-opus-5"})
    assert _run(["set", "sonnet"]) == 0
    assert tunables.get_raw(_CHAT) == "claude-sonnet-5"  # the entry holds the id the word means; nothing was removed
    assert _CHAT not in _overrides()
    assert capture_claude_argv.argvs == []  # the shipped id was verified when it was pinned, so no check
    out = capsys.readouterr().out
    assert "chat model is now claude-sonnet-5." in out and "following" not in out


def test_set_sonnet_when_the_value_already_is_the_default_says_so_and_leaves_the_file_alone(capture_claude_argv, capsys) -> None:
    assert _run(["set", "sonnet"]) == 0
    assert not _PATH().exists()  # nothing to write and no file yet: not created
    tunables.set_values({"other.key": 1})
    before = _PATH().stat().st_mtime_ns
    capsys.readouterr()
    assert _run(["set", "sonnet"]) == 0
    assert _PATH().stat().st_mtime_ns == before  # no write, no file churn
    assert capture_claude_argv.argvs == []
    assert "already the default" in capsys.readouterr().out


def test_set_opus_writes_the_opus_id_after_one_probe(capture_claude_argv, capsys) -> None:
    assert _run(["set", "opus"]) == 0
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert capture_claude_argv.models == ["claude-opus-5"]
    assert "claude-opus-5" in capsys.readouterr().out


def test_set_haiku_writes_the_alias_with_no_probe(capture_claude_argv) -> None:
    assert _run(["set", "haiku"]) == 0
    assert _overrides()[_CHAT] == "haiku"
    assert capture_claude_argv.argvs == []


def test_set_a_typed_full_id_equal_to_the_default_is_probed_and_follows_the_default(capture_claude_argv) -> None:
    """Typed ids are checked even when they equal a default; the value already is that id, so nothing is written (owner
    decision: a value equal to the default follows later default changes, however it got there)."""
    assert _run(["set", "claude-sonnet-5"]) == 0
    assert _CHAT not in _overrides()
    assert capture_claude_argv.models == ["claude-sonnet-5"]


def test_set_a_rejected_id_writes_nothing_and_exits_one(capture_claude_argv, capsys) -> None:
    capture_claude_argv.reject("claude-sonet-5")
    tunables.set_values({"other.key": 1})
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
    tunables.set_values({_CHAT: "claude-sonnet-4-9"})
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


def test_set_tier_writes_the_value_per_job(capsys) -> None:
    tunables.set_values({"model.background_generative": "claude-sonnet-4-6", "model.background_classifier": "claude-haiku-4-5"})
    assert _run(["set-tier", "background_generative", "sonnet"]) == 0
    assert tunables.get_raw("model.background_generative") == "claude-sonnet-5"  # the word means the chat default id
    assert "model.background_generative" not in _overrides()  # which equals that job's default
    assert _run(["set-tier", "background-classifier", "haiku"]) == 0
    assert tunables.get_raw("model.background_classifier") == "haiku" and "model.background_classifier" not in _overrides()
    assert _run(["set-tier", "background_classifier", "sonnet"]) == 0
    assert _overrides()["model.background_classifier"] == "claude-sonnet-5"  # differs from haiku: the id is written
    assert _run(["set-tier", "compaction", "claude-haiku-4-5"]) == 0
    assert _overrides()["model.compaction"] == "claude-haiku-4-5"
    out = capsys.readouterr().out
    assert "background-generative now uses claude-sonnet-5 (the shipped default)" in out


def test_set_tier_says_already_the_default_when_nothing_changed(capsys) -> None:
    assert _run(["set-tier", "compaction", "haiku"]) == 0
    out = capsys.readouterr().out
    assert "compaction already uses haiku" in out and "already the default" in out


def test_set_tier_clears_the_record_of_the_id_it_writes_c30(capture_claude_argv) -> None:
    from brain.bridge.model_tier import TIER_COMPACTION
    from brain.bridge.provider import ClaudeCliProvider

    capture_claude_argv.reject("claude-haiku-4-9")
    tunables.set_values({"model.compaction": "claude-haiku-4-9"})
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_COMPACTION).generate("x")
    assert model_error.is_rejected("claude-haiku-4-9")
    assert _run(["set-tier", "compaction", "claude-haiku-4-9"]) == 0
    assert not model_error.is_rejected("claude-haiku-4-9")


@pytest.mark.parametrize("bad", ["gpt4", "claude-", "claude-" + "x" * 70])
def test_set_tier_refuses_invalid_ids_and_leaves_the_file_alone(bad, capsys) -> None:
    tunables.set_values({"other.key": 1})
    before = _PATH().read_bytes()
    assert _run(["set-tier", "compaction", bad]) == 1
    assert _PATH().read_bytes() == before


def test_an_unknown_job_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _run(["set-tier", "not-a-job", "haiku"])
    with pytest.raises(SystemExit):
        _run(["clear-tier", "not-a-job"])


def test_clear_tier_writes_the_jobs_default_and_keeps_the_entry_including_the_chat_job(capsys) -> None:
    tunables.set_values({"model.compaction": "claude-haiku-4-5", _CHAT: "claude-opus-5", "keep.me": 1})
    assert _run(["clear-tier", "compaction"]) == 0
    assert _run(["clear-tier", "interactive-chat"]) == 0
    assert tunables.get_raw("model.compaction") == "haiku" and tunables.get_raw(_CHAT) == "claude-sonnet-5"  # entries stay
    assert _overrides() == {"keep.me": 1}
    assert "shipped default" in capsys.readouterr().out
    assert _run(["clear-tier", "compaction"]) == 0  # resetting what is already the default is fine


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_lists_each_job_with_default_or_setting(capsys) -> None:
    tunables.set_values({"model.compaction": "claude-haiku-4-5", _CHAT: "claude-opus-5"})
    assert _run(["show"]) == 0
    out = capsys.readouterr().out
    assert "interactive-chat: claude-opus-5 (your setting)" in out
    assert "compaction: claude-haiku-4-5 (your setting)" in out
    assert "background-generative: claude-sonnet-5 (shipped default)" in out
    assert "attunement-detector: claude-haiku-4-5 (shipped default)" in out
    assert "runtime fallbacks appear in the app" in out
    assert "every persona on this install" in out


def test_show_names_the_effective_id_when_a_fallback_is_active(capture_claude_argv, capsys) -> None:
    from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT
    from brain.bridge.provider import ClaudeCliProvider

    capture_claude_argv.reject("claude-sonnet-4-9")
    tunables.set_values({_CHAT: "claude-sonnet-4-9"})
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("x")
    assert _run(["show"]) == 0
    assert "running claude-sonnet-5 now" in capsys.readouterr().out


def test_show_warns_for_an_invalid_setting_and_never_edits_it(capsys) -> None:
    tunables.set_values({_CHAT: "banana"})
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
    _PATH().write_text("# mine\nb = = 2\n")
    assert _run(["set", "opus"]) == 1
    err = capsys.readouterr().err
    assert "could not be saved" in err and "tunables.toml" in err and "line 2" in err  # the reason names the failing line
    assert _PATH().read_text() == "# mine\nb = = 2\n"
    assert _run(["set-tier", "compaction", "claude-haiku-4-5"]) == 1
    assert _run(["clear-tier", "compaction"]) == 1


def test_a_wedged_lock_makes_the_command_fail_without_writing(wedged_lock, monkeypatch, capsys) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    tunables.set_values({"other.key": 1})
    before = _PATH().read_bytes()
    with wedged_lock(_PATH()):
        assert _run(["set-tier", "compaction", "claude-haiku-4-5"]) == 1
    assert _PATH().read_bytes() == before
    assert "could not lock" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The old tunables.json: a CLI folds all-or-nothing (C53 i2, owner N1). The legacy-shape fixture below is the only one in
# this module; the rest of the old shape lives in test_tunables_migration.py.
# ---------------------------------------------------------------------------


def test_a_cli_never_applies_part_of_an_old_json_so_a_later_clear_tier_is_never_undone(tmp_path, caplog) -> None:
    import logging

    from fastapi.testclient import TestClient

    from brain.bridge.server import build_app

    home = _PATH().parent
    home.mkdir(parents=True, exist_ok=True)
    legacy = home / "tunables.json"
    legacy.write_text(
        json.dumps({"overrides": {"model.compaction": "claude-haiku-4-9", "only.the.bridge.knows": 1}}), encoding="utf-8"
    )
    before = legacy.read_bytes()
    with caplog.at_level(logging.INFO, logger="brain.tunables_migration"):
        assert _run(["set", "sonnet"]) == 0  # CLI command 1
    assert legacy.read_bytes() == before and "model.compaction" not in _overrides()
    assert len([r for r in caplog.records if r.name == "brain.tunables_migration" and r.levelno == logging.INFO]) == 1
    assert _run(["clear-tier", "compaction"]) == 0
    assert _run(["set-tier", "background-classifier", "haiku"]) == 0  # CLI command 2
    assert legacy.read_bytes() == before and tunables.get_raw("model.compaction") == "haiku"  # X was NOT applied
    persona = home / "personas" / "n1"
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8")
    with TestClient(build_app(persona_dir=persona, client_origin="tests")):  # a real bridge boot
        pass
    assert tunables.get_raw("model.compaction") == "claude-haiku-4-9"  # applied once, by the bridge
    assert not legacy.exists() and len(list(home.glob("tunables.json.migrated-*"))) == 1
    assert "only.the.bridge.knows" not in __import__("tests.tunables_file", fromlist=["x"]).read_values(home)  # never inserted
    assert _run(["clear-tier", "compaction"]) == 0
    with TestClient(build_app(persona_dir=persona, client_origin="tests")):  # a second boot
        pass
    assert tunables.get_raw("model.compaction") == "haiku"  # nothing was re-applied


def test_a_cli_with_every_key_registered_folds_the_json_and_renames_it() -> None:
    home = _PATH().parent
    home.mkdir(parents=True, exist_ok=True)
    (home / "tunables.json").write_text(json.dumps({"overrides": {"model.compaction": "claude-haiku-4-9"}}), encoding="utf-8")
    assert _run(["set", "sonnet"]) == 0
    assert tunables.get_raw("model.compaction") == "claude-haiku-4-9"
    assert not (home / "tunables.json").exists() and len(list(home.glob("tunables.json.migrated-*"))) == 1


def test_show_mentions_an_old_json_that_is_still_waiting_and_never_folds_it(capsys) -> None:
    home = _PATH().parent
    home.mkdir(parents=True, exist_ok=True)
    (home / "tunables.json").write_text(json.dumps({"overrides": {}}), encoding="utf-8")
    assert _run(["show"]) == 0
    assert "old tunables.json" in capsys.readouterr().out
    assert (home / "tunables.json").exists() and not _PATH().exists()
