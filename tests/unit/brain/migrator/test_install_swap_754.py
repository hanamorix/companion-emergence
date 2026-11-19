"""#754: a forced migrate install takes the persona's start lock, and refuses while a start, rebuild or bridge holds it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from brain.bridge.daemon import acquire_lock, release_lock


def _live_persona(tmp_path: Path) -> Path:
    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    (persona / "memories.db").write_bytes(b"live")
    return persona


def test_refuse_if_in_use_refuses_while_the_start_lock_is_held(tmp_path):
    from brain.migrator.install_swap import PersonaInUseError, refuse_if_in_use

    persona = _live_persona(tmp_path)
    fd = acquire_lock(persona)
    assert fd is not None
    try:
        with pytest.raises(PersonaInUseError, match="nell"):
            refuse_if_in_use(persona)
    finally:
        release_lock(persona, fd)
    refuse_if_in_use(persona)  # free again: no refusal


def _new_build(persona: Path) -> Path:
    work = persona.with_name(f"{persona.name}.new")
    work.mkdir()
    (work / "memories.db").write_bytes(b"new")
    return work


def test_install_over_refuses_while_the_start_lock_is_held_and_moves_nothing(tmp_path):
    from brain.migrator.install_swap import PersonaInUseError, install_over

    persona = _live_persona(tmp_path)
    work = _new_build(persona)
    fd = acquire_lock(persona)
    assert fd is not None
    try:
        with pytest.raises(PersonaInUseError):
            install_over(persona, work)
    finally:
        release_lock(persona, fd)
    assert (persona / "memories.db").read_bytes() == b"live"
    assert (work / "memories.db").read_bytes() == b"new"
    assert not list(persona.parent.glob("nell.backup-*"))


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows lets the lock go before the renames (own test)")
def test_install_over_holds_the_start_lock_across_both_renames(tmp_path, monkeypatch):
    from brain.migrator import install_swap

    persona = _live_persona(tmp_path)
    work = _new_build(persona)
    real_rename = os.rename
    seen: list[bool] = []

    def probe(src, dst):
        # The locked file is in `src` at the first rename (the live folder) and in the backup at the second.
        holder = Path(src) if not seen else next(persona.parent.glob("nell.backup-*"))
        fd = acquire_lock(holder)
        seen.append(fd is None)
        if fd is not None:
            release_lock(holder, fd)
        real_rename(src, dst)

    monkeypatch.setattr(install_swap.os, "rename", probe)
    backup = install_swap.install_over(persona, work)
    monkeypatch.setattr(install_swap.os, "rename", real_rename)

    assert seen == [True, True], "a start could take the lock during the swap"
    assert (persona / "memories.db").read_bytes() == b"new"
    assert (backup / "memories.db").read_bytes() == b"live"
    assert not work.exists()
    for folder in (backup, persona):  # free afterwards
        fd = acquire_lock(folder)
        assert fd is not None
        release_lock(folder, fd)


def test_on_windows_install_over_lets_the_lock_go_before_renaming(tmp_path, monkeypatch):
    """Windows won't rename a folder while a file inside it is open, our own lock handle included."""
    from brain.migrator import install_swap

    persona = _live_persona(tmp_path)
    work = _new_build(persona)
    real_rename = os.rename
    free_at_first_rename: list[bool] = []

    def probe(src, dst):
        if not free_at_first_rename:
            fd = acquire_lock(Path(src))
            free_at_first_rename.append(fd is not None)
            if fd is not None:
                release_lock(Path(src), fd)
        real_rename(src, dst)

    monkeypatch.setattr(install_swap, "_IS_WINDOWS", True)
    monkeypatch.setattr(install_swap.os, "rename", probe)
    install_swap.install_over(persona, work)
    monkeypatch.setattr(install_swap.os, "rename", real_rename)

    assert free_at_first_rename == [True]
    assert (persona / "memories.db").read_bytes() == b"new"


def test_on_windows_a_folder_opened_before_the_rename_is_a_refusal(tmp_path, monkeypatch):
    """A start that opens the folder after the lock is let go makes Windows refuse the rename: nothing moves."""
    from brain.migrator import install_swap

    persona = _live_persona(tmp_path)
    work = _new_build(persona)

    def in_use(src, dst):
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(install_swap, "_IS_WINDOWS", True)
    monkeypatch.setattr(install_swap.os, "rename", in_use)
    with pytest.raises(install_swap.PersonaInUseError):
        install_swap.install_over(persona, work)
    monkeypatch.undo()

    assert (persona / "memories.db").read_bytes() == b"live"
    assert (work / "memories.db").read_bytes() == b"new"


def test_the_windows_refusal_says_a_file_may_be_open_and_gives_the_os_error(tmp_path, monkeypatch):
    """Review of #754: Explorer, the search indexer or antivirus holding a file also fails the rename, so the message
    can't say only "running or starting"."""
    from brain.migrator import install_swap

    persona = _live_persona(tmp_path)
    work = _new_build(persona)

    def in_use(src, dst):
        raise PermissionError(13, "being used by another process", str(src))

    monkeypatch.setattr(install_swap, "_IS_WINDOWS", True)
    monkeypatch.setattr(install_swap.os, "rename", in_use)
    with pytest.raises(install_swap.PersonaInUseError) as refused:
        install_swap.install_over(persona, work)
    message = str(refused.value)
    assert "a file in it is open" in message and "being used by another process" in message
    assert "was not changed" in message


def test_a_start_between_the_two_renames_finds_the_lock_held(tmp_path, monkeypatch):
    """Review of #754: between the renames the persona folder is briefly missing. Every start path checks the folder
    exists first, so a missing one there means it is mid-swap: `acquire_lock` answers "held", not FileNotFoundError."""
    from brain.migrator import install_swap

    persona = _live_persona(tmp_path)
    work = _new_build(persona)
    real_rename = os.rename
    seen: list[object] = []

    def probe(src, dst):
        if Path(src) == work:  # the live folder is already aside
            seen.append(acquire_lock(persona))
        real_rename(src, dst)

    monkeypatch.setattr(install_swap.os, "rename", probe)
    install_swap.install_over(persona, work)
    monkeypatch.undo()
    assert seen == [None]


def test_install_over_with_no_live_folder_renames_the_build_into_place(tmp_path):
    from brain.migrator.install_swap import install_over

    persona = tmp_path / "personas" / "nell"
    persona.parent.mkdir(parents=True)
    work = _new_build(persona)
    assert install_over(persona, work) is None
    assert (persona / "memories.db").read_bytes() == b"new"
    assert not work.exists() and not list(persona.parent.glob("nell.backup-*"))


def test_a_failed_second_rename_puts_the_live_folder_back(tmp_path, monkeypatch):
    """Review of #754: a file held open in the build (antivirus on Windows) fails the second rename. The live folder
    goes back in place instead of staying in the backup with the persona path empty."""
    from brain.migrator import install_swap

    persona = _live_persona(tmp_path)
    work = _new_build(persona)
    real_rename = os.rename
    calls: list[str] = []

    def second_fails(src, dst):
        calls.append(str(src))
        if Path(src) == work:
            raise PermissionError(13, "in use", str(src))
        real_rename(src, dst)

    monkeypatch.setattr(install_swap.os, "rename", second_fails)
    with pytest.raises(PermissionError):
        install_swap.install_over(persona, work)
    monkeypatch.undo()

    assert len(calls) == 3  # aside, in (fails), back
    assert (persona / "memories.db").read_bytes() == b"live"
    assert (work / "memories.db").read_bytes() == b"new"
    assert not list(persona.parent.glob("nell.backup-*"))


# ---- task 2: every migrator's forced install refuses while the lock is held, before any work ----
def _og_source(tmp_path: Path) -> Path:
    import json

    import numpy as np

    og = tmp_path / "og_data"
    og.mkdir()
    memory = {"id": "m1", "content": "first", "memory_type": "conversation", "domain": "us",
              "created_at": "2024-01-01T00:00:00+00:00", "emotions": {"love": 9.0}, "emotion_score": 9.0}
    (og / "memories_v2.json").write_text(json.dumps([memory]), encoding="utf-8")
    (og / "connection_matrix_ids.json").write_text(json.dumps(["m1"]), encoding="utf-8")
    np.save(og / "connection_matrix.npy", np.zeros((1, 1), dtype=np.float32))
    (og / "hebbian_state.json").write_text("{}", encoding="utf-8")
    return og


def _held(persona: Path):
    fd = acquire_lock(persona)
    assert fd is not None
    return fd


def _assert_untouched(persona: Path) -> None:
    assert (persona / "memories.db").read_bytes() == b"live"
    assert not list(persona.parent.glob(f"{persona.name}.backup-*"))
    assert not persona.with_name(f"{persona.name}.new").exists(), "the migration ran before the refusal"


def test_a_forced_nellbrain_install_refuses_while_the_persona_is_in_use(tmp_path, monkeypatch):
    from brain.migrator.cli import MigrateArgs, run_migrate
    from brain.migrator.install_swap import PersonaInUseError

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    fd = _held(persona)
    try:
        with pytest.raises(PersonaInUseError):
            run_migrate(MigrateArgs(input_dir=_og_source(tmp_path), output_dir=None, install_as="nell", force=True))
    finally:
        release_lock(persona, fd)
    _assert_untouched(persona)


def _start_arrives_after_the_early_check(monkeypatch, module, persona: Path) -> list[int]:
    """Patch `module.refuse_if_in_use` so a start takes the lock right after the early check passes."""
    from brain.migrator.install_swap import refuse_if_in_use

    taken: list[int] = []

    def check_then_a_start_arrives(final_dir):
        refuse_if_in_use(final_dir)
        taken.append(_held(persona))

    monkeypatch.setattr(module, "refuse_if_in_use", check_then_a_start_arrives)
    return taken


def test_a_start_during_a_nellbrain_migration_is_refused_at_the_swap(tmp_path, monkeypatch):
    from brain.migrator import cli
    from brain.migrator.install_swap import PersonaInUseError

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    taken = _start_arrives_after_the_early_check(monkeypatch, cli, persona)
    try:
        with pytest.raises(PersonaInUseError):
            cli.run_migrate(cli.MigrateArgs(input_dir=_og_source(tmp_path), output_dir=None, install_as="nell",
                                            force=True))
    finally:
        for fd in taken:
            release_lock(persona, fd)
    assert taken, "the early check never ran"
    assert (persona / "memories.db").read_bytes() == b"live"
    assert not list(persona.parent.glob("nell.backup-*"))


def test_a_nellbrain_migrate_refused_at_the_swap_prints_no_completion_report(tmp_path, monkeypatch, capsys):
    """Review of #754: "Migration complete." then a refusal on stderr reads as two answers; the report waits for the
    swap."""
    from brain.migrator import cli
    from brain.migrator.install_swap import PersonaInUseError

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    taken = _start_arrives_after_the_early_check(monkeypatch, cli, persona)
    try:
        with pytest.raises(PersonaInUseError):
            cli.run_migrate(cli.MigrateArgs(input_dir=_og_source(tmp_path), output_dir=None, install_as="nell",
                                            force=True))
    finally:
        for fd in taken:
            release_lock(persona, fd)
    assert "Migration complete." not in capsys.readouterr().out
    assert (persona.with_name("nell.new") / "migration-report.md").is_file()  # still written with the build


def test_nell_migrate_prints_the_refusal_and_exits_1(tmp_path, monkeypatch, capsys):
    from brain.cli import main

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    fd = _held(persona)
    try:
        rc = main(["migrate", "--input", str(_og_source(tmp_path)), "--install-as", "nell", "--force"])
    finally:
        release_lock(persona, fd)
    err = capsys.readouterr().err
    assert rc == 1
    assert "running or starting" in err and "nell supervisor stop --persona nell" in err
    assert "Traceback" not in err
    _assert_untouched(persona)


def test_a_forced_emergence_kit_install_refuses_while_the_persona_is_in_use(tmp_path, monkeypatch):
    from brain.migrator.emergence_kit import EmergenceKitMigrateArgs, migrate_emergence_kit
    from brain.migrator.install_swap import PersonaInUseError
    from tests.unit.brain.migrator.test_emergence_kit import _seed_kit

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    _seed_kit(tmp_path / "kit")
    fd = _held(persona)
    try:
        with pytest.raises(PersonaInUseError):
            migrate_emergence_kit(EmergenceKitMigrateArgs(input_dir=tmp_path / "kit", output_dir=None,
                                                          install_as="nell", force=True))
    finally:
        release_lock(persona, fd)
    _assert_untouched(persona)


def test_a_start_during_an_emergence_kit_migration_is_refused_at_the_swap(tmp_path, monkeypatch):
    from brain.migrator import emergence_kit
    from brain.migrator.install_swap import PersonaInUseError
    from tests.unit.brain.migrator.test_emergence_kit import _seed_kit

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    _seed_kit(tmp_path / "kit")
    taken = _start_arrives_after_the_early_check(monkeypatch, emergence_kit, persona)
    try:
        with pytest.raises(PersonaInUseError):
            emergence_kit.migrate_emergence_kit(emergence_kit.EmergenceKitMigrateArgs(
                input_dir=tmp_path / "kit", output_dir=None, install_as="nell", force=True))
    finally:
        for fd in taken:
            release_lock(persona, fd)
    assert taken, "the early check never ran"
    assert (persona / "memories.db").read_bytes() == b"live"
    assert not list(persona.parent.glob("nell.backup-*"))


def test_an_emergence_kit_build_is_finished_before_the_swap(tmp_path, monkeypatch):
    """Review of #754: the manifest was written after the swap, by path, outside the lock; it now goes into the build,
    so a refused run leaves a finished `.new` and a successful one installs it."""
    from brain.migrator import emergence_kit
    from brain.migrator.install_swap import PersonaInUseError
    from tests.unit.brain.migrator.test_emergence_kit import _seed_kit

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    _seed_kit(tmp_path / "kit")
    taken = _start_arrives_after_the_early_check(monkeypatch, emergence_kit, persona)
    try:
        with pytest.raises(PersonaInUseError):
            emergence_kit.migrate_emergence_kit(emergence_kit.EmergenceKitMigrateArgs(
                input_dir=tmp_path / "kit", output_dir=None, install_as="nell", force=True))
    finally:
        for fd in taken:
            release_lock(persona, fd)
    assert (persona.with_name("nell.new") / "source-manifest.json").is_file()


def test_a_second_emergence_kit_migrate_for_the_same_name_is_refused(tmp_path, monkeypatch):
    """Like companion-emergence: both runs build in `<name>.new`, so the migrate lock keeps them apart."""
    from brain.migrator.cli import _acquire_migrate_lock, _migrate_lock_path, _release_migrate_lock
    from brain.migrator.emergence_kit import EmergenceKitMigrateArgs, migrate_emergence_kit
    from tests.unit.brain.migrator.test_emergence_kit import _seed_kit

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    _seed_kit(tmp_path / "kit")
    lock_path = _migrate_lock_path(persona)
    first = _acquire_migrate_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="another migrate"):
            migrate_emergence_kit(EmergenceKitMigrateArgs(input_dir=tmp_path / "kit", output_dir=None,
                                                          install_as="nell", force=True))
    finally:
        _release_migrate_lock(lock_path, first)
    _assert_untouched(persona)


def test_a_forced_companion_emergence_install_refuses_while_the_persona_is_in_use(tmp_path, monkeypatch):
    from brain.migrator.companion_emergence import (
        CompanionEmergenceMigrateArgs,
        migrate_companion_emergence,
    )
    from brain.migrator.install_swap import PersonaInUseError
    from tests.unit.brain.migrator.test_companion_emergence import _make_v012_persona

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    src = _make_v012_persona(tmp_path / "src", name="nell")
    fd = _held(persona)
    try:
        with pytest.raises(PersonaInUseError):
            migrate_companion_emergence(CompanionEmergenceMigrateArgs(input_dir=src, install_as="nell", force=True))
    finally:
        release_lock(persona, fd)
    _assert_untouched(persona)


def test_a_start_during_a_companion_emergence_copy_is_refused_at_the_swap(tmp_path, monkeypatch):
    """The copy goes to `<name>.new`, never into the live path, and the swap takes the start lock."""
    from brain.migrator import companion_emergence as ce
    from brain.migrator.install_swap import PersonaInUseError
    from tests.unit.brain.migrator.test_companion_emergence import _make_v012_persona

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    src = _make_v012_persona(tmp_path / "src", name="nell")
    taken = _start_arrives_after_the_early_check(monkeypatch, ce, persona)
    try:
        with pytest.raises(PersonaInUseError):
            ce.migrate_companion_emergence(ce.CompanionEmergenceMigrateArgs(input_dir=src, install_as="nell",
                                                                            force=True))
    finally:
        for fd in taken:
            release_lock(persona, fd)
    assert taken, "the early check never ran"
    assert (persona / "memories.db").read_bytes() == b"live"
    assert not list(persona.parent.glob("nell.backup-*"))
    build = persona.with_name("nell.new")
    assert (build / "source-manifest.json").is_file()  # built and finished in .new, left for the next run


def test_a_forced_companion_emergence_install_swaps_its_build_in(tmp_path, monkeypatch):
    from brain.migrator.companion_emergence import (
        CompanionEmergenceMigrateArgs,
        migrate_companion_emergence,
    )
    from tests.unit.brain.migrator.test_companion_emergence import _make_v012_persona

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    leftover = persona.with_name("nell.new")
    leftover.mkdir()
    (leftover / "stale.txt").write_bytes(b"from a refused run")
    src = _make_v012_persona(tmp_path / "src", name="nell")
    migrate_companion_emergence(CompanionEmergenceMigrateArgs(input_dir=src, install_as="nell", force=True))
    (backup,) = persona.parent.glob("nell.backup-*")
    assert (backup / "memories.db").read_bytes() == b"live"
    assert (persona / "source-manifest.json").is_file() and (persona / "persona_config.json").is_file()
    assert not (persona / "stale.txt").exists() and not leftover.exists()


def test_a_second_companion_emergence_migrate_for_the_same_name_is_refused(tmp_path, monkeypatch):
    """Review of #754: both runs build in `<name>.new`, so a second run would delete and refill the first one's build
    while the first swaps it in. The migrate lock (as `run_migrate` takes) keeps them apart."""
    from brain.migrator.cli import _acquire_migrate_lock, _migrate_lock_path, _release_migrate_lock
    from brain.migrator.companion_emergence import (
        CompanionEmergenceMigrateArgs,
        migrate_companion_emergence,
    )
    from tests.unit.brain.migrator.test_companion_emergence import _make_v012_persona

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona = _live_persona(tmp_path / "home")
    src = _make_v012_persona(tmp_path / "src", name="nell")
    lock_path = _migrate_lock_path(persona)
    first = _acquire_migrate_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="another migrate"):
            migrate_companion_emergence(CompanionEmergenceMigrateArgs(input_dir=src, install_as="nell", force=True))
    finally:
        _release_migrate_lock(lock_path, first)
    _assert_untouched(persona)
