"""ram-spike-fix INC-6, criterion C12 (AC12, S25/S30/S37/S71): the tunables
migration that retires the pre-lull idle-tuning keys into the single
chat.idle_lull_seconds key."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.tunables_migration import migrate_idle_keys


def _write(home: Path, overrides: dict) -> None:
    (home / "tunables.json").write_text(
        json.dumps({"_readme": "x", "defaults": {}, "overrides": overrides}, indent=2),
        encoding="utf-8",
    )


def _read_overrides(home: Path) -> dict:
    data = json.loads((home / "tunables.json").read_text(encoding="utf-8"))
    return data.get("overrides", {})


def test_override_300_becomes_default_600_old_key_gone(tmp_path: Path) -> None:
    """override 300 (== the OLD default, i.e. "never customized") -> lull
    600 (the new code default — no override key written), old key gone."""
    _write(tmp_path, {"throttle.background_min_idle_seconds": 300.0})
    migrate_idle_keys(tmp_path)
    overrides = _read_overrides(tmp_path)
    assert "throttle.background_min_idle_seconds" not in overrides
    assert "chat.idle_lull_seconds" not in overrides  # falls to code default 600


def test_key_unset_stays_default_600(tmp_path: Path) -> None:
    _write(tmp_path, {})
    migrate_idle_keys(tmp_path)
    overrides = _read_overrides(tmp_path)
    assert "chat.idle_lull_seconds" not in overrides


def test_override_450_carries_forward_as_lull_450_old_key_gone(tmp_path: Path) -> None:
    """A genuinely-customized override (450, != the old default 300) carries
    forward as the new key's override."""
    _write(tmp_path, {"throttle.background_min_idle_seconds": 450.0})
    migrate_idle_keys(tmp_path)
    overrides = _read_overrides(tmp_path)
    assert "throttle.background_min_idle_seconds" not in overrides
    assert overrides["chat.idle_lull_seconds"] == 450.0


def test_pass2_and_articulate_keys_removed_not_carried(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "chat.pass2_min_idle_seconds": 5.0,
            "self_model.articulate_min_idle_seconds": 12.0,
        },
    )
    migrate_idle_keys(tmp_path)
    overrides = _read_overrides(tmp_path)
    assert "chat.pass2_min_idle_seconds" not in overrides
    assert "self_model.articulate_min_idle_seconds" not in overrides
    # Neither value carried anywhere — no lull override created from them.
    assert "chat.idle_lull_seconds" not in overrides


def test_judge_selftune_gate_override_removed_not_carried_s71(tmp_path: Path) -> None:
    _write(tmp_path, {"judge_selftune.gate_handful_decisions": 20})
    migrate_idle_keys(tmp_path)
    overrides = _read_overrides(tmp_path)
    assert "judge_selftune.gate_handful_decisions" not in overrides


def test_second_run_is_a_no_op_idempotent(tmp_path: Path) -> None:
    _write(tmp_path, {"throttle.background_min_idle_seconds": 450.0})
    migrate_idle_keys(tmp_path)
    after_first = (tmp_path / "tunables.json").read_bytes()
    migrate_idle_keys(tmp_path)
    after_second = (tmp_path / "tunables.json").read_bytes()
    assert after_first == after_second


def test_no_retired_keys_present_is_a_true_no_op_no_write(tmp_path: Path) -> None:
    """Idempotent: no old keys -> no write at all (not even a rewritten,
    byte-different "no-op" file)."""
    _write(tmp_path, {"some.unrelated.key": 1.0})
    before = (tmp_path / "tunables.json").read_bytes()
    before_mtime = (tmp_path / "tunables.json").stat().st_mtime_ns
    migrate_idle_keys(tmp_path)
    after = (tmp_path / "tunables.json").read_bytes()
    after_mtime = (tmp_path / "tunables.json").stat().st_mtime_ns
    assert before == after
    assert before_mtime == after_mtime, "file must not be touched at all when nothing to migrate"


def test_missing_file_is_a_no_op(tmp_path: Path) -> None:
    migrate_idle_keys(tmp_path)  # must not raise
    assert not (tmp_path / "tunables.json").exists()


def test_forced_write_failure_restores_original_bytes_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A monkeypatched write raising AFTER the backup is taken -> original
    file bytes intact, failure logged, backup cleaned up (per §7: 'Any
    failure -> restore from the backup if the file changed, keep the
    original, log WARNING')."""
    _write(tmp_path, {"throttle.background_min_idle_seconds": 450.0})
    original_bytes = (tmp_path / "tunables.json").read_bytes()

    import os as os_module

    real_replace = os_module.replace

    def _boom(*_a, **_kw):
        raise OSError("simulated disk failure during migration write")

    monkeypatch.setattr("brain.tunables_migration.os.replace", _boom)

    with caplog.at_level("WARNING"):
        migrate_idle_keys(tmp_path)  # must not raise

    assert (tmp_path / "tunables.json").read_bytes() == original_bytes
    assert not (tmp_path / "tunables.json.bak-migrate").exists(), "backup must be cleaned up"
    assert any("WARNING" == r.levelname for r in caplog.records)
    monkeypatch.setattr("brain.tunables_migration.os.replace", real_replace)


def test_migration_completes_before_supervisor_loop_pass_ordering_in_lifespan() -> None:
    """C12: 'migration completes before the supervisor's first loop pass' —
    static check that server.py's lifespan calls migrate_idle_keys() BEFORE
    spawning the run_folded supervisor thread.

    Matches on the bare call-site prefix (round-2 code red-team nitpick: an
    exact-full-line match broke on a harmless reformat) so this survives
    reasonable edits to the call's arguments while still pinning the call
    itself, not just the string "migrate_idle_keys" (which also appears in
    the import line and comments)."""
    import brain.bridge.server as server_mod

    src = Path(server_mod.__file__).read_text(encoding="utf-8")
    # Skip past "from brain.tunables_migration import migrate_idle_keys" (the
    # import line) to find the actual CALL site.
    import_idx = src.index("import migrate_idle_keys")
    call_idx = src.index("migrate_idle_keys(", import_idx)
    supervisor_idx = src.index("from brain.bridge.supervisor import run_folded")
    assert call_idx < supervisor_idx, (
        "migrate_idle_keys must run before the supervisor thread is spawned"
    )
