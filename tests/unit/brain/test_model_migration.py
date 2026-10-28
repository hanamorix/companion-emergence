"""The one-time move of saved per-persona models into the install-wide tunables lines (#529: C8 d, C14, C15, C30 e,
C33, C37, C42 e, C45).

Throwaway personas under the per-test home, real locks, the real bridge lifespan for the assembled cases (the claude
CLI is never started: the lifespan builds a provider but spawns nothing).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import dev_constants, model_migration, paths, tunables
from brain.bridge import model_error
from brain.bridge.model_tier import set_job_models
from brain.bridge.server import build_app
from brain.health.json_io import JsonFileError
from brain.model_migration import MARKER_NAME, migrate_persona_models, pending_legacy
from brain.persona_config import PersonaConfig
from tests.thread_support import finish_threads

_CHAT = "model.interactive_chat"
_CHAT_JOB = "interactive-chat"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _home() -> Path:
    return paths.get_home()


def _persona(name: str, *, model=None, tiers=None, opened: str | None = None, extra: dict | None = None, **more) -> Path:
    d = paths.get_persona_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    cfg: dict = {"provider": "claude-cli", "searcher": "noop"}
    if model is not None:
        cfg["model"] = model
    if tiers is not None:
        cfg["tier_models"] = tiers
    if opened is not None:
        cfg["last_opened_at"] = opened
    cfg.update(extra or {})
    cfg.update(more)
    (d / "persona_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return d


def _overrides() -> dict:
    """The settings that differ from their shipped default (the file lists every setting now): a model job whose value is not
    the job's default, plus any ``other.*`` key a test wrote. The defaults come from the job table, not the registry (other
    tests clear the registry)."""
    from brain.bridge.model_tier import _CLAUDE_JOBS
    from brain.utils import toml_io

    p = tunables._file_path()
    if not p.exists():
        return {}
    defaults = dict(_CLAUDE_JOBS.values())
    values = toml_io.flatten_toml(toml_io.read_toml(p))
    return {k: v for k, v in values.items() if (k in defaults and v != defaults[k]) or k.startswith("other.")}


def _marker() -> Path:
    return _home() / MARKER_NAME


def _cfg(name: str) -> dict:
    return json.loads((paths.get_persona_dir(name) / "persona_config.json").read_text(encoding="utf-8"))


def _boot(name: str) -> None:
    """Start the real bridge lifespan for a persona (and stop it again)."""
    with TestClient(build_app(persona_dir=paths.get_persona_dir(name), client_origin="tests")):
        pass


@pytest.fixture(autouse=True)
def _short_lock_wait(monkeypatch):
    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 3.0)


# ---------------------------------------------------------------------------
# C14 (a-d, i): who wins, how it is mapped, what is never written
# ---------------------------------------------------------------------------


def test_the_newest_last_opened_at_wins_and_is_mapped(caplog) -> None:
    _persona("older", model="opus", opened="2026-01-01T00:00:00Z")
    _persona("newest", model="claude-sonnet-4-6", opened="2026-03-01T00:00:00Z")
    _persona("middle", model="haiku", opened="2026-02-01T00:00:00Z")
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-sonnet-4-6"  # a full id verbatim
    assert _marker().exists()
    log = caplog.text
    for name in ("older", "newest", "middle"):
        assert f"candidate {name}" in log  # every candidate is listed with its value and signal
    assert "last_opened_at" in log


@pytest.mark.parametrize(("saved", "expected"), [("opus", "claude-opus-5"), ("haiku", "haiku")])
def test_a_saved_word_maps_to_what_it_means(saved, expected) -> None:
    _persona("only", model=saved, opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == expected


def test_a_saved_sonnet_or_the_default_id_is_not_written_and_still_decides() -> None:
    """C14 (c): a value equal to the shipped default follows the default instead of freezing a copy."""
    _persona("a", model="sonnet", opened="2026-01-01T00:00:00Z")
    _persona("b", model="claude-sonnet-5", opened="2026-00-01T00:00:00Z".replace("2026-00", "2025-12"))
    migrate_persona_models(_home(), None)
    assert _CHAT not in _overrides()
    assert _marker().exists()  # a decision was made, there was just nothing to write


def test_without_last_opened_at_the_file_mtime_decides_and_a_tie_goes_to_the_first_name() -> None:
    a = _persona("alpha", model="opus")
    b = _persona("beta", model="claude-sonnet-4-6")
    os.utime(a / "persona_config.json", (1_700_000_000, 1_700_000_000))
    os.utime(b / "persona_config.json", (1_800_000_000, 1_800_000_000))
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-sonnet-4-6"  # beta is newer by mtime
    _marker().unlink()
    set_job_models({"interactive-chat": None})  # back to the shipped default
    os.utime(b / "persona_config.json", (1_700_000_000, 1_700_000_000))  # tie
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-opus-5"  # alphabetical: alpha wins the tie


def test_tier_models_fold_per_job_with_the_newest_holder_winning() -> None:
    _persona("a", tiers={"compaction": "claude-haiku-4-5", "background-generative": "claude-sonnet-4-6"}, opened="2026-01-01T00:00:00Z")
    _persona("b", tiers={"compaction": "claude-haiku-4-6"}, model="opus", opened="2026-02-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    o = _overrides()
    assert o["model.compaction"] == "claude-haiku-4-6"  # b is newer for this job
    assert o["model.background_generative"] == "claude-sonnet-4-6"  # only a holds it
    assert o[_CHAT] == "claude-opus-5"


def test_an_existing_setting_is_never_overwritten() -> None:
    tunables.set_values({_CHAT: "claude-opus-4-9"})
    _persona("only", model="claude-sonnet-4-6", tiers={"compaction": "claude-haiku-4-5"}, opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-opus-4-9"  # the user already chose
    assert _overrides()["model.compaction"] == "claude-haiku-4-5"
    assert _marker().exists()


def test_an_invalid_saved_model_is_dropped_with_a_log_and_not_written(caplog) -> None:
    d = _persona("typo-persona", model="banana", tiers={"compaction": "nope", "not-a-job": "claude-haiku-4-5"}, opened="2026-01-01T00:00:00Z")
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), d)
    assert _overrides() == {}
    assert not _marker().exists()  # no valid candidate: nothing decided, a persona imported later still migrates
    assert not {"model", "tier_models"} & set(_cfg("typo-persona"))
    discarded = [r for r in caplog.records if "typo-persona" in r.getMessage() and "banana" in r.getMessage()]
    assert discarded, "the discarded invalid text is logged with the persona's name"
    assert all(r.levelno >= logging.WARNING for r in discarded)
    assert "nope" in discarded[0].getMessage()  # the per-job text is named too


def test_a_discarded_invalid_text_is_shown_safely(caplog) -> None:
    d = _persona("odd", model="bad\x1b[31mtext\n", opened="2026-01-01T00:00:00Z")
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), d)
    assert "\x1b" not in caplog.text and "bad[31mtext" in caplog.text


def test_a_valid_value_the_decision_never_scanned_is_logged_when_it_is_discarded(caplog) -> None:
    """A persona directory outside the name grammar is not scanned (list_persona_names skips it), so its valid saved
    model plays no part in the decision made by another persona in the same call. Clearing it must not be silent."""
    _persona("good", model="opus", opened="2026-02-01T00:00:00Z")
    stray = _home() / "personas" / "bad name"
    stray.mkdir(parents=True)
    (stray / "persona_config.json").write_text(json.dumps({"provider": "claude-cli", "model": "claude-sonnet-4-6"}), encoding="utf-8")
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), stray)
    assert _overrides()[_CHAT] == "claude-opus-5"  # decided by "good"
    assert "model" not in json.loads((stray / "persona_config.json").read_text())
    assert any("bad name" in r.getMessage() and "claude-sonnet-4-6" in r.getMessage() for r in caplog.records)


def test_a_non_claude_persona_never_decides_the_install_model() -> None:
    """The saved model of a persona on another provider kind was never a Claude choice; it must not set the Claude
    jobs' model for every other persona, however recently that persona was used."""
    _persona("local", model="opus", opened="2026-09-01T00:00:00Z", provider="ollama")
    _persona("fake-one", model="claude-opus-4-9", opened="2026-09-02T00:00:00Z", provider="fake")
    _persona("claude-one", model="claude-sonnet-4-6", opened="2026-01-01T00:00:00Z")
    assert pending_legacy()
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-sonnet-4-6"
    assert json.loads(_marker().read_text())["source_persona"] == "claude-one"


def test_with_only_non_claude_personas_nothing_is_decided() -> None:
    _persona("local", model="opus", opened="2026-09-01T00:00:00Z", provider="ollama")
    assert not pending_legacy()
    migrate_persona_models(_home(), None)
    assert _overrides() == {} and not _marker().exists()


# ---------------------------------------------------------------------------
# C15: idempotence and the zero-candidate rule
# ---------------------------------------------------------------------------


def test_with_no_legacy_keys_nothing_is_written_and_no_marker() -> None:
    _persona("clean")
    tunables.set_values({"other.key": 1})
    before = tunables._file_path().stat().st_mtime_ns
    migrate_persona_models(_home(), paths.get_persona_dir("clean"))
    assert tunables._file_path().stat().st_mtime_ns == before
    assert not _marker().exists()


def test_a_persona_imported_after_a_zero_candidate_decision_is_still_migrated() -> None:
    _persona("first", model="banana")
    migrate_persona_models(_home(), None)
    assert not _marker().exists()
    _persona("imported", model="opus", opened="2026-05-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-opus-5" and _marker().exists()


def test_a_second_run_changes_nothing() -> None:
    _persona("only", model="opus", opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    first = tunables._file_path().read_bytes()
    stamp = _marker().read_bytes()
    migrate_persona_models(_home(), None)
    assert tunables._file_path().read_bytes() == first and _marker().read_bytes() == stamp


# ---------------------------------------------------------------------------
# C14 (f, g, h, j): the clearing step and the pass-through field
# ---------------------------------------------------------------------------


def test_the_launching_persona_loses_its_legacy_keys_and_the_others_keep_theirs_until_their_own_start() -> None:
    a = _persona("alpha", model="opus", tiers={"compaction": "claude-haiku-4-5"}, opened="2026-02-01T00:00:00Z", derived_unavailable=["claude-haiku-5"])
    b = _persona("beta", model="claude-sonnet-4-6", opened="2026-01-01T00:00:00Z")
    beta_before = (b / "persona_config.json").read_bytes()
    migrate_persona_models(_home(), a)
    cfg = _cfg("alpha")
    assert not {"model", "tier_models", "derived_unavailable"} & set(cfg)
    assert (b / "persona_config.json").read_bytes() == beta_before  # never rewritten by another persona's start
    assert _overrides()[_CHAT] == "claude-opus-5"


def test_a_persona_found_after_the_decision_keeps_the_install_chat_line_and_applies_only_if_default_jobs(caplog) -> None:
    """Backward compatibility is permanent: an old config that turns up later is applied the same way, where the install
    has no setting; where it does, the install's setting stays and the old value is discarded with a log."""
    _persona("first", model="opus", opened="2026-02-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    late = _persona("late", model="claude-sonnet-4-6", tiers={"compaction": "claude-haiku-4-5"}, opened="2026-09-01T00:00:00Z")
    other_before = json.loads((late / "persona_config.json").read_text())
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), late)
    assert _overrides() == {_CHAT: "claude-opus-5", "model.compaction": "claude-haiku-4-5"}
    cfg = _cfg("late")
    assert {k: v for k, v in other_before.items() if k not in ("model", "tier_models", "derived_unavailable")} == cfg
    assert "discarded" in caplog.text and "claude-sonnet-4-6" in caplog.text  # the chat value, not applied


def test_a_persona_found_after_a_decision_that_wrote_no_chat_line_sets_it() -> None:
    """The first decision saw only a saved `sonnet` (the shipped default, so no line). A later old config holding `opus`
    finds the install without a chat line and its model goes in."""
    _persona("first", model="sonnet", opened="2026-02-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert _marker().exists() and _CHAT not in _overrides()
    late = _persona("late", model="opus", opened="2026-09-01T00:00:00Z")
    migrate_persona_models(_home(), late)
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert "model" not in _cfg("late")


def test_the_loser_of_the_first_decision_does_not_overwrite_the_winner_when_it_starts(caplog) -> None:
    """The persona is not a late one: the decision considered it. Newest wins; the loser's value, applied
    with only_if_default against an install that holds no line (the winner saved the default), would reverse that."""
    _persona("newer", model="sonnet", opened="2026-05-01T00:00:00Z")
    older = _persona("older", model="opus", opened="2026-01-01T00:00:00Z")
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), older)
    assert _CHAT not in _overrides()
    assert "model" not in _cfg("older") and _marker().exists()


def test_a_pick_made_after_the_decision_survives_a_considered_persona_restarting() -> None:
    """Maintainer review 530-1, scenario 1: alpha (`opus`, newest) and beta (`haiku`). The decision writes `claude-opus-5`;
    the user then picks Sonnet (the shipped default id); alpha restarts. A persona the decision considered is cleared on its
    next start, never applied: `only_if_default` cannot tell "never chose" from "chose the default"."""
    alpha = _persona("alpha", model="opus", opened="2026-05-01T00:00:00Z")
    _persona("beta", model="haiku", opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)  # the decision (call 1)
    assert _overrides()[_CHAT] == "claude-opus-5"
    set_job_models({_CHAT_JOB: "sonnet"})  # the user's pick, `nell model set sonnet` (the default id)
    assert _CHAT not in _overrides()
    migrate_persona_models(_home(), alpha)  # alpha restarts (call 2)
    assert _CHAT not in _overrides()  # the pick survives
    assert "model" not in _cfg("alpha")


def test_an_older_persona_cannot_win_over_the_decision_by_starting_later() -> None:
    """Maintainer review 530-1, scenario 2: alpha (`sonnet`, newest, equal to the default so nothing is written) and beta
    (`haiku`). Beta's own start must not write `haiku` over the decision: the newest persona won."""
    _persona("alpha", model="sonnet", opened="2026-05-01T00:00:00Z")
    beta = _persona("beta", model="haiku", opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)  # call 1: the decision
    assert _CHAT not in _overrides()
    migrate_persona_models(_home(), beta)  # call 2: beta starts
    assert _CHAT not in _overrides()
    assert "model" not in _cfg("beta")


def test_the_marker_records_every_persona_the_decision_considered() -> None:
    _persona("alpha", model="opus", opened="2026-05-01T00:00:00Z")
    _persona("beta", model="haiku", opened="2026-01-01T00:00:00Z")
    _persona("plain")  # no saved model: not a candidate
    _persona("local", model="opus", opened="2026-02-01T00:00:00Z", provider="ollama")  # another provider kind: not a candidate
    migrate_persona_models(_home(), None)
    marker = json.loads(_marker().read_text(encoding="utf-8"))
    assert marker["personas_considered"] == ["alpha", "beta"]


def test_a_persona_the_decision_did_not_consider_is_still_applied_where_the_install_is_at_its_default() -> None:
    """The ruling-B path is unchanged for a persona that is NOT in the marker (imported after the decision)."""
    _persona("first", model="sonnet", opened="2026-02-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert json.loads(_marker().read_text(encoding="utf-8"))["personas_considered"] == ["first"]
    late = _persona("late", model="opus", opened="2026-09-01T00:00:00Z")
    migrate_persona_models(_home(), late)
    assert _overrides()[_CHAT] == "claude-opus-5" and "model" not in _cfg("late")


def test_a_marker_without_the_list_still_lets_a_late_persona_be_applied() -> None:
    """A marker that carries no ``personas_considered`` (hand-made, or from a build that did not record it) names nobody:
    every persona is a late one, which is the old behaviour."""
    _home().mkdir(parents=True, exist_ok=True)
    _marker().write_text(json.dumps({"decided_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    late = _persona("late", model="opus", opened="2026-09-01T00:00:00Z")
    migrate_persona_models(_home(), late)
    assert _overrides()[_CHAT] == "claude-opus-5"


def test_a_late_persona_on_another_provider_kind_is_cleared_not_applied() -> None:
    _persona("first", model="sonnet", opened="2026-02-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    local = _persona("local", model="opus", opened="2026-09-01T00:00:00Z", provider="ollama")
    migrate_persona_models(_home(), local)
    assert _CHAT not in _overrides() and "model" not in _cfg("local")


def test_a_late_persona_is_not_cleared_when_its_values_could_not_be_written(monkeypatch, caplog) -> None:
    _persona("first", model="sonnet", opened="2026-02-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    late = _persona("late", model="opus", opened="2026-09-01T00:00:00Z")

    def refuse(*a, **kw):
        raise tunables.TunablesWriteError("locked")

    real = model_migration.set_job_models
    monkeypatch.setattr(model_migration, "set_job_models", refuse)
    with caplog.at_level(logging.WARNING):
        migrate_persona_models(_home(), late)
    assert _cfg("late")["model"] == "opus"  # nothing lost: the next start retries
    assert "late" in caplog.text
    monkeypatch.setattr(model_migration, "set_job_models", real)
    migrate_persona_models(_home(), late)
    assert _overrides()[_CHAT] == "claude-opus-5" and "model" not in _cfg("late")


def test_legacy_keys_that_reappear_after_a_backup_restore_are_cleared_and_never_applied() -> None:
    a = _persona("only", model="opus", opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), a)
    cfg_path = a / "persona_config.json"
    data = json.loads(cfg_path.read_text())
    data["model"] = "claude-opus-4-1"  # a .bak restore brings the old key back
    cfg_path.write_text(json.dumps(data))
    migrate_persona_models(_home(), a)
    assert "model" not in _cfg("only")
    assert _overrides()[_CHAT] == "claude-opus-5"


def test_a_saved_opus_survives_a_config_save_by_another_path_before_the_migration() -> None:
    """C14 (h): the bridge's first config touch is a load then a save; the pass-through field keeps the legacy keys."""
    a = _persona("only", model="opus", opened="2026-01-01T00:00:00Z")
    cfg_path = a / "persona_config.json"
    cfg = PersonaConfig.load(cfg_path)
    cfg.touch_last_opened()
    cfg.save(cfg_path)  # without the pass-through field this erased `model`
    assert _cfg("only")["model"] == "opus"
    migrate_persona_models(_home(), a)
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert "model" not in _cfg("only")


@pytest.mark.parametrize("pick", ["sonnet", "haiku"])
def test_an_upgrade_after_a_downgrade_keeps_the_settings_and_folds_what_the_older_brain_wrote(caplog, pick) -> None:
    """The rollback contract of the migration spec (docs/source-spec/2026-10-22-model-settings-migration.md): after an in-app
    revert the older brain ignores tunables.toml and runs on its defaults; the values stay in the file and are used again
    on the next upgrade; a ``tunables.json`` the older brain wrote meanwhile is folded by the legacy-JSON rule (only into a
    setting still at its default); a model the older brain saved in a persona the decision considered is cleared, not
    applied (it never overwrites the install's setting).

    ``pick`` is the model the user chose in the upgraded app. ``sonnet`` is the chat job's SHIPPED DEFAULT id: the case the
    "considered persona is only cleared" rule exists for, because a late apply (``only_if_default``) would take the older
    brain's ``claude-opus-4-1`` over a setting that merely equals the default and this test would then fail. ``haiku`` is a
    pick that differs from the default (the old late-apply path discards that one too, so it proves less on its own)."""
    from brain.bridge.model_tier import resolve_choice, shipped_default
    from brain.tunables_migration import migrate_legacy_json
    from tests.tunables_file import read_values

    def old_brain_writes_settings(overrides: dict) -> None:
        (_home() / "tunables.json").write_text(json.dumps({"overrides": overrides}), encoding="utf-8")

    tunables.register("rt.kept", 1.0, description="Kept across the round trip.")
    tunables.register("rt.changed_by_user", 1.0, description="Changed in the upgraded app.")
    tunables.register("rt.new_while_down", 1.0, description="Set only while downgraded.")
    _home().mkdir(parents=True, exist_ok=True)

    # Upgrade 1: an old install (JSON settings, a persona with a saved model).
    old_brain_writes_settings({"rt.kept": 7.0})
    alpha = _persona("alpha", model="opus", opened="2026-05-01T00:00:00Z")
    migrate_legacy_json(_home(), complete_registry=True)
    migrate_persona_models(_home(), alpha)
    assert read_values(_home())["rt.kept"] == 7.0 and _overrides()[_CHAT] == "claude-opus-5"
    assert not (_home() / "tunables.json").exists() and "model" not in _cfg("alpha")

    # The upgraded app: the user changes a setting and picks a model (only tunables.toml is written).
    set_job_models({_CHAT_JOB: pick})
    tunables.set_values({"rt.changed_by_user": 5.0})
    picked = resolve_choice(pick)
    assert picked == (shipped_default(_CHAT_JOB) if pick == "sonnet" else "haiku")  # the case under test is what it says

    # Downgrade: the older brain never reads tunables.toml. It writes its own files.
    assert read_values(_home())["rt.kept"] == 7.0  # nothing the older brain does touches the file's values
    old_brain_writes_settings({"rt.new_while_down": 9.0, "rt.changed_by_user": 2.0, "rt.kept": 3.0})
    cfg_path = alpha / "persona_config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data["model"] = "claude-opus-4-1"  # the older brain's own picker saves the model in the persona file
    cfg_path.write_text(json.dumps(data), encoding="utf-8")

    # Upgrade 2.
    with caplog.at_level(logging.INFO):
        migrate_legacy_json(_home(), complete_registry=True)
        migrate_persona_models(_home(), alpha)
    values = read_values(_home())
    assert values["rt.kept"] == 7.0  # the upgraded value is intact: the older brain's 3.0 was not folded over it
    assert values["rt.changed_by_user"] == 5.0  # a setting the user changed since is never overwritten
    assert values["rt.new_while_down"] == 9.0  # a setting still at its default takes the older brain's value
    assert values[_CHAT] == picked  # the model pick survives, a default-valued one included
    assert not (_home() / "tunables.json").exists() and len(list(_home().glob("tunables.json.migrated-*"))) == 2
    assert "model" not in _cfg("alpha")  # cleared, never applied


def test_the_pass_through_field_holds_raw_values_and_save_reemits_them_only_while_non_empty(tmp_path) -> None:
    p = tmp_path / "persona_config.json"
    p.write_text(json.dumps({"provider": "claude-cli", "model": "banana", "tier_models": {"x": 1}}))
    cfg = PersonaConfig.load(p)
    assert cfg.legacy_model_settings == {"model": "banana", "tier_models": {"x": 1}}  # never interpreted
    cfg.save(p)
    assert json.loads(p.read_text())["model"] == "banana"
    cfg.legacy_model_settings = {}
    cfg.save(p)
    assert "model" not in json.loads(p.read_text()) and "tier_models" not in json.loads(p.read_text())


def test_a_wizard_fixture_config_with_a_saved_sonnet_migrates_to_the_shipped_default() -> None:
    """The real wizard-validation fixture (a checked-in v0.0.12 persona_config.json with "model": "sonnet")."""
    fixtures = sorted((_REPO_ROOT / "wizard-validation" / "fixtures").glob("**/persona_config.json"))
    assert fixtures, "wizard-validation fixture missing"
    d = paths.get_persona_dir("fixture-copy")
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(fixtures[0].read_text(encoding="utf-8"), encoding="utf-8")
    assert json.loads((d / "persona_config.json").read_text()).get("model") == "sonnet"
    migrate_persona_models(_home(), d)
    assert _CHAT not in _overrides()  # a saved "sonnet" follows the shipped default
    assert "model" not in _cfg("fixture-copy")
    assert _marker().exists()


# ---------------------------------------------------------------------------
# C14 (k, l): damaged files, and the reads happen before any lock
# ---------------------------------------------------------------------------


def test_a_content_damaged_persona_file_is_skipped_and_does_not_block_the_others(caplog) -> None:
    bad = _persona("broken", opened="2026-05-01T00:00:00Z")
    (bad / "persona_config.json").write_text("{not json", encoding="utf-8")
    _persona("fine", model="opus", opened="2026-01-01T00:00:00Z")
    with caplog.at_level(logging.INFO):
        migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert (bad / "persona_config.json").read_text() == "{not json"  # never healed or quarantined by another persona's start
    assert not list(bad.glob("persona_config.json.corrupt-*"))
    assert "broken" in caplog.text and json.loads(_marker().read_text())["personas_skipped"] == ["broken"]


def test_a_persisting_io_fault_is_skipped_with_a_warning_naming_the_persona(monkeypatch, caplog) -> None:
    _persona("unreadable", model="claude-opus-4-9", opened="2026-05-01T00:00:00Z")
    _persona("fine", model="opus", opened="2026-01-01T00:00:00Z")
    real = model_migration.read_json_result

    def flaky(path, **kw):
        if "unreadable" in str(path):
            raise JsonFileError(path, "io", "denied")
        return real(path, **kw)

    monkeypatch.setattr(model_migration, "read_json_result", flaky)
    with caplog.at_level(logging.WARNING):
        migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert any("unreadable" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_no_lock_is_held_while_the_persona_files_are_read(monkeypatch) -> None:
    """C14 (l): the reads precede the migration lock, so a slow read blocks nobody."""
    from brain.utils.file_lock import file_lock

    _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    real = model_migration.read_json_result
    free_during_reads: list[bool] = []

    def probing(path, **kw):
        with file_lock(_marker(), blocking=False) as got_marker, file_lock(tunables._file_path(), blocking=False) as got_tun:
            free_during_reads.append(bool(got_marker and got_tun))
        return real(path, **kw)

    monkeypatch.setattr(model_migration, "read_json_result", probing)
    migrate_persona_models(_home(), None)
    assert free_during_reads and all(free_during_reads)


# ---------------------------------------------------------------------------
# C33 and C8 (d): no deadlock, no lost update
# ---------------------------------------------------------------------------


def test_the_migration_cannot_deadlock_itself(monkeypatch) -> None:
    """C33: it writes through set_job_models while holding ONLY its own lock (a different sidecar). A nesting of the
    tunables lock would make this time out instead of finishing: the join below fails rather than hangs."""
    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 1.0)
    _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    t = threading.Thread(target=migrate_persona_models, args=(_home(), None))
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "migration hung"
    assert _marker().exists() and _overrides()[_CHAT] == "claude-opus-5"


def test_the_lock_order_is_the_migration_lock_then_the_tunables_lock(monkeypatch) -> None:
    from brain.utils import file_lock as fl

    order: list[str] = []
    real = fl.locked_within

    def recording(path, seconds, **kw):
        order.append(Path(path).name)
        return real(path, seconds, **kw)

    monkeypatch.setattr(model_migration, "locked_within", recording)
    monkeypatch.setattr(tunables, "locked_within", recording)
    _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert order == [MARKER_NAME, "tunables.toml"]


def test_an_external_write_between_the_scan_and_the_write_survives(monkeypatch) -> None:
    """C33 (atomicity of only_if_default): a chat model the user sets right after the scan is not overwritten."""
    _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    real_scan = model_migration._scan_candidates

    def scan_then_external_write(skipped):
        found = real_scan(skipped)
        tunables.set_values({_CHAT: "claude-sonnet-4-6"})  # lands after the migration read, before it writes
        return found

    monkeypatch.setattr(model_migration, "_scan_candidates", scan_then_external_write)
    migrate_persona_models(_home(), None)
    assert _overrides()[_CHAT] == "claude-sonnet-4-6"


def test_a_cli_write_during_the_migration_and_the_migration_both_survive(monkeypatch) -> None:
    """C8 (d): a set_job_models from another thread that lands inside the migration's write window waits for it."""
    from brain.bridge.model_tier import TIER_COMPACTION, set_job_models

    _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    cli_thread = threading.Thread(target=lambda: set_job_models({TIER_COMPACTION: "claude-haiku-4-5"}))
    inside = []
    window_owner = threading.get_ident()

    def hook() -> None:
        if threading.get_ident() != window_owner:
            return  # the competing write reaches the same hook once it gets in; only the migration's window is under test
        cli_thread.start()
        cli_thread.join(timeout=0.8)
        inside.append(cli_thread.is_alive())

    tunables._LOCKED_WINDOW_HOOK = hook
    try:
        migrate_persona_models(_home(), None)
    finally:
        tunables._LOCKED_WINDOW_HOOK = None
    finish_threads(cli_thread)
    assert inside == [True]  # it could not get in while the migration's window was open
    assert _overrides() == {_CHAT: "claude-opus-5", "model.compaction": "claude-haiku-4-5"}


def test_the_migration_clears_the_rejection_record_of_the_id_it_writes_c30e(capture_claude_argv) -> None:
    from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT
    from brain.bridge.provider import ClaudeCliProvider

    capture_claude_argv.reject("claude-sonnet-4-9")
    tunables.set_values({_CHAT: "claude-sonnet-4-9"})
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("x")
    tunables.set_values({_CHAT: "claude-sonnet-5"})  # back to the shipped default, by a path that does not forget the record
    assert model_error.is_rejected("claude-sonnet-4-9")
    _persona("a", model="claude-sonnet-4-9", opened="2026-01-01T00:00:00Z")
    migrate_persona_models(_home(), None)
    assert not model_error.is_rejected("claude-sonnet-4-9")


# ---------------------------------------------------------------------------
# C42 (e): a wedged lock means no write, no marker, no clearing
# ---------------------------------------------------------------------------


def test_a_wedged_migration_lock_leaves_no_marker_and_no_persona_file_changed(held_lock, monkeypatch, caplog) -> None:
    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    a = _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    before = (a / "persona_config.json").read_bytes()
    with held_lock(_marker()), caplog.at_level(logging.WARNING):
        migrate_persona_models(_home(), a)
    assert not _marker().exists() and _overrides() == {}
    assert (a / "persona_config.json").read_bytes() == before
    migrate_persona_models(_home(), a)  # the next start retries and succeeds
    assert _overrides()[_CHAT] == "claude-opus-5" and _marker().exists()


def test_a_wedged_tunables_lock_leaves_no_marker_either(held_lock, monkeypatch) -> None:
    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    a = _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    before = (a / "persona_config.json").read_bytes()
    tunables.set_values({"other.key": 1})
    with held_lock(tunables._file_path()):
        migrate_persona_models(_home(), a)
    assert not _marker().exists() and (a / "persona_config.json").read_bytes() == before
    assert _overrides() == {"other.key": 1}


def test_a_damaged_settings_file_means_no_marker_and_nothing_cleared() -> None:
    a = _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    before = (a / "persona_config.json").read_bytes()
    tunables._file_path().write_text("a = = 1\n")
    migrate_persona_models(_home(), a)
    assert not _marker().exists() and (a / "persona_config.json").read_bytes() == before
    assert tunables._file_path().read_text() == "a = = 1\n"  # an unreadable settings file is left alone, never repaired


# ---------------------------------------------------------------------------
# C14 (e), C37, C45: the real bridge lifespan (assembled)
# ---------------------------------------------------------------------------


def test_the_lifespan_runs_sync_then_the_json_fold_then_the_persona_migration_then_the_touch(monkeypatch) -> None:
    """C37: the order is asserted from the calls the REAL lifespan makes, not from a re-implementation of it."""
    import brain.tunables_migration as tm

    _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    calls: list[str] = []
    real_sync = tunables.sync_file
    real_fold = tm.migrate_legacy_json
    real_migrate = model_migration.migrate_persona_models
    real_touch = PersonaConfig.touch_last_opened

    monkeypatch.setattr(tunables, "sync_file", lambda: (calls.append("sync"), real_sync())[1])
    monkeypatch.setattr(tm, "migrate_legacy_json", lambda h, **kw: (calls.append("fold"), real_fold(h, **kw))[1])
    monkeypatch.setattr(model_migration, "migrate_persona_models", lambda h, p: (calls.append("migrate"), real_migrate(h, p))[1])
    monkeypatch.setattr(PersonaConfig, "touch_last_opened", lambda self: (calls.append("touch"), real_touch(self))[1])
    _boot("a")
    assert calls[:4] == ["sync", "fold", "migrate", "touch"]
    assert _overrides()[_CHAT] == "claude-opus-5"


def test_an_unreadable_settings_file_is_left_alone_and_the_next_boot_after_the_fix_migrates() -> None:
    """C14 (e), reshaped by owner ruling E9: nothing repairs an unreadable settings file, so the saved Opus stays in the
    persona's file (nothing is lost) and is folded in at the first boot after the user fixes the file."""
    a = _persona("a", model="opus", opened="2026-01-01T00:00:00Z")
    tunables._file_path().write_text("a = = 1\n")
    _boot("a")
    assert tunables._file_path().read_text() == "a = = 1\n"
    assert not _marker().exists() and _cfg("a")["model"] == "opus"
    assert not [p for p in _home().iterdir() if "corrupt" in p.name or "quarantine" in p.name]  # nothing is set aside
    tunables._file_path().write_text('"other.key" = 1\n')
    _boot("a")
    assert _overrides()[_CHAT] == "claude-opus-5"
    assert _marker().exists() and "model" not in _cfg("a")
    assert a.exists()


def test_the_decision_competes_with_the_launching_personas_previous_open_time() -> None:
    """C45, the stated rule (spec s6.4, MR12): the decision runs BEFORE the launching persona's last_opened_at is
    touched, so a more recently used OTHER persona wins even though a different persona triggered the migration.
    Do not 'fix' this into 'the launching persona wins'."""
    _persona("launching", model="claude-sonnet-4-6", opened="2026-01-01T00:00:00Z")
    _persona("recent", model="claude-opus-4-9", opened="2026-06-01T00:00:00Z")
    _boot("launching")
    assert _overrides()[_CHAT] == "claude-opus-4-9"
    assert "model" not in _cfg("launching")  # its own file is cleared, the other's is untouched until its start
    assert _cfg("recent")["model"] == "claude-opus-4-9"


def test_the_assembled_boot_of_a_second_persona_only_clears_it() -> None:
    _persona("first", model="opus", opened="2026-01-01T00:00:00Z")
    _persona("second", model="claude-sonnet-4-6", extra={"user_name": "Ada"}, opened="2026-00-01T00:00:00Z".replace("2026-00", "2025-12"))
    _boot("first")
    line = tunables._file_path().read_bytes()
    first_override = _overrides()[_CHAT]
    _boot("second")
    assert _overrides()[_CHAT] == first_override == "claude-opus-5"
    second = _cfg("second")
    assert "model" not in second and second["user_name"] == "Ada"
    assert line  # the install line was decided once


def test_pending_legacy_reports_without_migrating() -> None:
    assert not pending_legacy()
    _persona("a", model="opus")
    assert pending_legacy()
    assert not _marker().exists() and not tunables._file_path().exists()
    migrate_persona_models(_home(), None)
    assert pending_legacy()  # the persona still holds its keys until its own start clears them
    migrate_persona_models(_home(), paths.get_persona_dir("a"))
    assert not pending_legacy()
