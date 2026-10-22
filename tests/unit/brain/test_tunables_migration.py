"""Backward compatibility for the old ``tunables.json`` (#529, owner rulings E4 and E6; criteria C53, C44, C49).

An install that still has the old JSON settings file is brought forward the first time it is seen, and a JSON found later
(a restored backup, a copied install) is folded the same way: its values go into ``tunables.toml`` only where the TOML value
still equals the shipped default, then the JSON is renamed ``tunables.json.migrated-<time>``. It also keeps the older
retirement of the pre-lull idle keys (ram-spike-fix INC-6): ``throttle.background_min_idle_seconds`` carries to
``chat.idle_lull_seconds`` when it differs from 300.0, the other three retired keys are dropped.

Legacy-shape JSON fixtures live in this one module; they are the only place the old shape appears.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from brain.tunables_migration import migrate_legacy_json
from tests.tunables_file import FILE_NAME, read_values, write_text

LEGACY = "tunables.json"


@pytest.fixture()
def t(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as mod
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 5.0)
    mod._reset_for_tests()
    mod.register("a.one", 1.0, description="One.")
    mod.register("a.two", "x", description="Two.")
    mod.register("a.three", 3, description="Three.")
    mod.register("chat.idle_lull_seconds", 600.0, description="Seconds of quiet.")
    yield mod
    mod._reset_for_tests()


def _write_legacy(home: Path, overrides: dict, **extra) -> Path:
    path = home / LEGACY
    path.write_text(json.dumps({"_readme": "r", "defaults": {"a.one": 1.0}, "overrides": overrides, **extra}, indent=2), encoding="utf-8")
    return path


def _renamed(home: Path) -> list[Path]:
    return sorted(home.glob(f"{LEGACY}.migrated-*"))


# ---------------------------------------------------------------------------
# C53 (a): the old shape is folded, renamed, and a second start does nothing
# ---------------------------------------------------------------------------


def test_the_old_values_become_entries_and_every_other_key_gets_its_default(t, tmp_path) -> None:
    _write_legacy(tmp_path, {"a.one": 5.0, "a.two": "y"})
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert read_values(tmp_path) == {"a.one": 5.0, "a.two": "y", "a.three": 3, "chat.idle_lull_seconds": 600.0}
    assert not (tmp_path / LEGACY).exists()
    assert len(_renamed(tmp_path)) == 1


def test_a_second_start_does_nothing_no_lock_no_write(t, tmp_path, wedged_lock, monkeypatch) -> None:
    from brain import dev_constants

    _write_legacy(tmp_path, {"a.one": 5.0})
    migrate_legacy_json(tmp_path, complete_registry=True)
    path = tmp_path / FILE_NAME
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.2)
    with wedged_lock(path):  # a second start that took the lock would time out and log
        migrate_legacy_json(tmp_path, complete_registry=True)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime


def test_the_toml_keeps_the_users_comments_and_unknown_keys_when_the_fold_writes(t, tmp_path) -> None:
    write_text(tmp_path, '# mine\n"stranger" = "keep"  # note\n# default: 1.0\n"a.one" = 1.0\n')
    _write_legacy(tmp_path, {"a.one": 4.0})
    migrate_legacy_json(tmp_path, complete_registry=True)
    text = (tmp_path / FILE_NAME).read_text(encoding="utf-8")
    assert text.startswith('# mine\n"stranger" = "keep"  # note\n# default: 1.0\n"a.one" = 4.0\n')


# ---------------------------------------------------------------------------
# C53 (b): the retired idle keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "lull"),
    [
        ({"throttle.background_min_idle_seconds": 450.0}, 450.0),
        ({"throttle.background_min_idle_seconds": 300.0}, 600.0),
        ({"chat.pass2_min_idle_seconds": 5.0, "self_model.articulate_min_idle_seconds": 12.0}, 600.0),
        ({"judge_selftune.gate_handful_decisions": 20}, 600.0),
        ({}, 600.0),
    ],
)
def test_retired_idle_keys_carry_or_drop_and_never_reach_the_toml(t, tmp_path, overrides, lull) -> None:
    _write_legacy(tmp_path, overrides)
    migrate_legacy_json(tmp_path, complete_registry=True)
    values = read_values(tmp_path)
    assert values["chat.idle_lull_seconds"] == lull
    assert not any(k in values for k in overrides if k != "chat.idle_lull_seconds")
    assert len(_renamed(tmp_path)) == 1


# ---------------------------------------------------------------------------
# C53 (c) (d): no JSON is a no-op; a JSON found after a TOML exists folds only where the value is still the default
# ---------------------------------------------------------------------------


def test_no_json_is_a_no_op_and_creates_nothing(t, tmp_path) -> None:
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert list(tmp_path.iterdir()) == []


def test_a_json_found_later_folds_only_where_the_toml_value_is_still_the_default(t, tmp_path, caplog) -> None:
    write_text(tmp_path, '# default: 1.0\n"a.one" = 9.0\n# default: "x"\n"a.two" = "x"\n')
    _write_legacy(tmp_path, {"a.one": 5.0, "a.two": "y"})
    with caplog.at_level(logging.INFO, logger="brain.tunables_migration"):
        migrate_legacy_json(tmp_path, complete_registry=True)
    values = read_values(tmp_path)
    assert values["a.one"] == 9.0  # the user's change is kept
    assert values["a.two"] == "y"  # still the default: takes the old value
    assert any("a.one" in r.getMessage() for r in caplog.records)  # the discarded old value is logged by key
    assert len(_renamed(tmp_path)) == 1


# ---------------------------------------------------------------------------
# C53 (e): an unreadable JSON is left alone
# ---------------------------------------------------------------------------


def test_a_json_with_a_syntax_error_is_left_alone_with_one_warning_naming_the_line(t, tmp_path, caplog) -> None:
    path = tmp_path / LEGACY
    content = b'{\n  "overrides": {\n    "a.one": 5.0,\n    "a.two": oops\n  }\n}\n'
    path.write_bytes(content)
    with caplog.at_level(logging.WARNING, logger="brain.tunables_migration"):
        migrate_legacy_json(tmp_path, complete_registry=True)
        migrate_legacy_json(tmp_path, complete_registry=True)  # the next start retries, quietly (throttled)
    assert path.read_bytes() == content and not _renamed(tmp_path)
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name or ".bak" in p.name]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "line 4" in warnings[0] and LEGACY in warnings[0]
    assert "a.one" not in read_values(tmp_path) or read_values(tmp_path)["a.one"] == 1.0  # nothing carried


def test_a_json_that_is_not_utf8_is_left_alone(t, tmp_path) -> None:
    path = tmp_path / LEGACY
    path.write_bytes(b'{"overrides": {"a.two": "caf\xe9"}}')
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert path.exists() and not _renamed(tmp_path)


@pytest.mark.parametrize("content", [b"", b"{}", b'{"overrides": {}}', b'{"overrides": null}'])
def test_an_empty_json_is_renamed_without_a_warning(t, tmp_path, caplog, content) -> None:
    (tmp_path / LEGACY).write_bytes(content)
    with caplog.at_level(logging.WARNING):
        migrate_legacy_json(tmp_path, complete_registry=True)
    assert not (tmp_path / LEGACY).exists() and len(_renamed(tmp_path)) == 1
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# ---------------------------------------------------------------------------
# C53 (f): one poison value never blocks the others or the rename
# ---------------------------------------------------------------------------


def test_values_the_toml_cannot_hold_are_skipped_by_key_and_the_rest_is_carried(t, tmp_path, caplog) -> None:
    t.register("p.null", 1, description="N.")
    t.register("p.nan", 1.0, description="N.")
    t.register("p.inf", 1.0, description="N.")
    t.register("p.surrogate", "s", description="S.")
    t.register("p.nested", 1, description="N.")
    (tmp_path / LEGACY).write_text(
        '{"overrides": {"p.null": null, "p.nan": NaN, "p.inf": Infinity, "p.surrogate": "a\\ud800b", '
        '"p.nested": {"x": 1}, "a.one": 7.0}}',
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="brain.tunables_migration"):
        migrate_legacy_json(tmp_path, complete_registry=True)
    values = read_values(tmp_path)
    assert values["a.one"] == 7.0
    assert values["p.null"] == 1 and values["p.nan"] == 1.0 and values["p.inf"] == 1.0
    assert values["p.surrogate"] == "s" and values["p.nested"] == 1
    logged = " ".join(r.getMessage() for r in caplog.records)
    for key in ("p.null", "p.nan", "p.inf", "p.surrogate", "p.nested"):
        assert key in logged
    assert len(_renamed(tmp_path)) == 1


def test_a_wrong_typed_value_is_copied_and_the_reader_ignores_it(t, tmp_path) -> None:
    _write_legacy(tmp_path, {"a.one": "fast"})
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert read_values(tmp_path)["a.one"] == "fast"
    assert t.get_tunable("a.one", 1.0) == 1.0  # the type guard, as for a hand-typed value


# ---------------------------------------------------------------------------
# C53 (g) (h): lock timeout and a crash between the write and the rename
# ---------------------------------------------------------------------------


def test_a_lock_timeout_leaves_both_files_as_they_were_and_the_next_start_retries(t, tmp_path, wedged_lock, monkeypatch) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    legacy = _write_legacy(tmp_path, {"a.one": 5.0})
    before = legacy.read_bytes()
    with wedged_lock(tmp_path / FILE_NAME):
        migrate_legacy_json(tmp_path, complete_registry=True)
    assert legacy.read_bytes() == before and not _renamed(tmp_path)
    assert not (tmp_path / FILE_NAME).exists()
    migrate_legacy_json(tmp_path, complete_registry=True)  # the lock is free again
    assert read_values(tmp_path)["a.one"] == 5.0 and len(_renamed(tmp_path)) == 1


def test_a_crash_between_the_write_and_the_rename_converges_on_the_rerun(t, tmp_path, monkeypatch) -> None:
    _write_legacy(tmp_path, {"a.one": 5.0})
    real_rename = Path.rename

    def boom(self, target):
        if self.name == LEGACY:
            raise OSError("simulated crash before the rename")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", boom)
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert (tmp_path / LEGACY).exists() and read_values(tmp_path)["a.one"] == 5.0
    monkeypatch.setattr(Path, "rename", real_rename)
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert not (tmp_path / LEGACY).exists() and len(_renamed(tmp_path)) == 1
    assert read_values(tmp_path)["a.one"] == 5.0


def test_the_rename_never_overwrites_an_earlier_migrated_copy(t, tmp_path) -> None:
    first = _write_legacy(tmp_path, {"a.one": 5.0})
    migrate_legacy_json(tmp_path, complete_registry=True)
    kept = _renamed(tmp_path)[0].read_bytes()
    _write_legacy(tmp_path, {"a.one": 6.0})
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert len(_renamed(tmp_path)) == 2
    assert kept in [p.read_bytes() for p in _renamed(tmp_path)]  # the first migrated copy survived the second migration
    assert not first.exists()


def test_an_unreadable_toml_leaves_the_json_unrenamed_and_the_toml_untouched(t, tmp_path) -> None:
    (tmp_path / FILE_NAME).write_bytes(b"a = = 1\n")
    legacy = _write_legacy(tmp_path, {"a.one": 5.0})
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert (tmp_path / FILE_NAME).read_bytes() == b"a = = 1\n"
    assert legacy.exists() and not _renamed(tmp_path)


# ---------------------------------------------------------------------------
# C53 step 5: the bridge discards unregistered keys; the CLI folds all or nothing (N1)
# ---------------------------------------------------------------------------


def test_the_bridge_discards_an_unregistered_key_and_never_inserts_it(t, tmp_path, caplog) -> None:
    _write_legacy(tmp_path, {"a.one": 5.0, "retired.in.code": 1})
    with caplog.at_level(logging.INFO, logger="brain.tunables_migration"):
        migrate_legacy_json(tmp_path, complete_registry=True)
    assert "retired.in.code" not in read_values(tmp_path)
    assert any("retired.in.code" in r.getMessage() for r in caplog.records)
    assert len(_renamed(tmp_path)) == 1


def test_the_cli_folds_everything_and_renames_when_every_key_is_registered(t, tmp_path) -> None:
    _write_legacy(tmp_path, {"a.one": 5.0, "a.two": "y"})
    migrate_legacy_json(tmp_path, complete_registry=False)
    assert read_values(tmp_path)["a.one"] == 5.0 and read_values(tmp_path)["a.two"] == "y"
    assert len(_renamed(tmp_path)) == 1


def test_the_cli_applies_nothing_when_any_key_is_unregistered_here(t, tmp_path, caplog) -> None:
    legacy = _write_legacy(tmp_path, {"a.one": 5.0, "only.the.bridge.knows": 2})
    before = legacy.read_bytes()
    with caplog.at_level(logging.INFO, logger="brain.tunables_migration"):
        migrate_legacy_json(tmp_path, complete_registry=False)
    assert legacy.read_bytes() == before and not _renamed(tmp_path)
    assert read_values(tmp_path).get("a.one") in (None, 1.0)  # a.one was NOT applied although it is registered here
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1 and "bridge" in infos[0].getMessage()


def test_a_cli_that_deferred_never_undoes_a_change_made_before_the_bridge_runs(t, tmp_path) -> None:
    """N1: the clear-tier then CLI then bridge sequence. A CLI that applied the keys it could would re-apply a.one over the reset."""
    _write_legacy(tmp_path, {"a.one": 5.0, "only.the.bridge.knows": 2})
    migrate_legacy_json(tmp_path, complete_registry=False)  # CLI command 1: defers
    t.set_values({"a.one": None})  # the user resets the setting meanwhile
    migrate_legacy_json(tmp_path, complete_registry=False)  # CLI command 2: still defers
    assert read_values(tmp_path)["a.one"] == 1.0
    t.register("only.the.bridge.knows", 0, description="Known to the bridge.")  # the bridge process registers it
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert read_values(tmp_path)["a.one"] == 5.0  # applied once, now that the whole fold can finish
    assert len(_renamed(tmp_path)) == 1
    t.set_values({"a.one": None})
    migrate_legacy_json(tmp_path, complete_registry=True)  # a second bridge boot: nothing is re-applied
    assert read_values(tmp_path)["a.one"] == 1.0


# ---------------------------------------------------------------------------
# C44 re-expressed: the fold writes through set_values under the lock
# ---------------------------------------------------------------------------


def test_a_set_values_landing_between_the_json_read_and_the_write_survives(t, tmp_path) -> None:
    _write_legacy(tmp_path, {"a.one": 5.0})
    competitor = threading.Thread(target=lambda: t.set_values({"a.three": 99}))
    finished_inside: list[bool] = []

    def hook() -> None:
        competitor.start()
        competitor.join(timeout=1.0)
        finished_inside.append(not competitor.is_alive())

    t._LOCKED_WINDOW_HOOK = hook
    try:
        migrate_legacy_json(tmp_path, complete_registry=True)
    finally:
        t._LOCKED_WINDOW_HOOK = None
    competitor.join(timeout=20)
    assert finished_inside and finished_inside[0] is False  # it waited for the fold's lock window
    values = read_values(tmp_path)
    assert values["a.one"] == 5.0 and values["a.three"] == 99


def test_a_set_values_that_beats_the_fold_to_a_key_keeps_its_value(t, tmp_path) -> None:
    """The fold's write is only-if-default: a value the user set (here, between the sync and the fold) is never overwritten."""
    _write_legacy(tmp_path, {"a.one": 5.0})
    t.sync_file()
    t.set_values({"a.one": 8.0})
    migrate_legacy_json(tmp_path, complete_registry=True)
    assert read_values(tmp_path)["a.one"] == 8.0
    assert len(_renamed(tmp_path)) == 1


def test_the_migration_never_raises_into_startup(t, tmp_path, monkeypatch) -> None:
    _write_legacy(tmp_path, {"a.one": 5.0})
    monkeypatch.setattr("brain.tunables.sync_file", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    migrate_legacy_json(tmp_path, complete_registry=True)  # logged and swallowed
    assert (tmp_path / LEGACY).exists()


def test_the_pre_check_is_one_exists_call_and_takes_no_lock(t, tmp_path, wedged_lock, monkeypatch) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.2)
    write_text(tmp_path, '"a.one" = 1.0\n')
    before = (tmp_path / FILE_NAME).stat().st_mtime_ns
    with wedged_lock(tmp_path / FILE_NAME):
        migrate_legacy_json(tmp_path, complete_registry=True)
    assert (tmp_path / FILE_NAME).stat().st_mtime_ns == before


def test_the_names_of_the_old_migration_are_gone() -> None:
    import brain.tunables_migration as tm

    for gone in ("migrate_idle_keys", "_read_state", "_rewrite", "_BACKUP_SUFFIX"):
        assert not hasattr(tm, gone), gone
    assert hasattr(tm, "_RETIRED_KEYS")
