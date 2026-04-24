"""Tests for brain.migrator.cli — subcommand orchestration + safety."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from brain.migrator.cli import MigrateArgs, run_migrate


@pytest.fixture
def og_dir(tmp_path: Path) -> Path:
    og = tmp_path / "og_data"
    og.mkdir()

    memories = [
        {
            "id": "m1",
            "content": "first",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2024-01-01T00:00:00+00:00",
            "emotions": {"love": 9.0},
            "emotion_score": 9.0,
        },
        {
            "id": "m2",
            "content": "second",
            "memory_type": "meta",
            "domain": "work",
            "created_at": "2024-02-01T00:00:00+00:00",
            "emotions": {},
            "emotion_score": 0.0,
        },
    ]
    (og / "memories_v2.json").write_text(json.dumps(memories))
    (og / "connection_matrix_ids.json").write_text(json.dumps(["m1", "m2"]))
    matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    np.save(og / "connection_matrix.npy", matrix)
    (og / "hebbian_state.json").write_text("{}")
    return og


def test_run_migrate_output_mode_writes_expected_files(og_dir: Path, tmp_path: Path) -> None:
    """--output mode produces memories.db, hebbian.db, source-manifest.json, migration-report.md."""
    out = tmp_path / "migrated"
    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    run_migrate(args)

    assert (out / "memories.db").exists()
    assert (out / "hebbian.db").exists()
    assert (out / "source-manifest.json").exists()
    assert (out / "migration-report.md").exists()


def test_run_migrate_refuses_nonempty_output_without_force(og_dir: Path, tmp_path: Path) -> None:
    """Non-empty output dir without --force → error."""
    out = tmp_path / "migrated"
    out.mkdir()
    (out / "pre-existing.txt").write_text("x")

    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    with pytest.raises(FileExistsError, match="non-empty"):
        run_migrate(args)


def test_run_migrate_allows_empty_output_dir(og_dir: Path, tmp_path: Path) -> None:
    """Empty output dir is acceptable — user may pre-create the path."""
    out = tmp_path / "migrated"
    out.mkdir()  # empty

    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    run_migrate(args)  # must not raise


def test_run_migrate_refuses_live_lock(og_dir: Path, tmp_path: Path) -> None:
    """Recent memories_v2.json.lock → LiveLockDetected (migrator aborts)."""
    from brain.migrator.og import LiveLockDetected

    (og_dir / "memories_v2.json.lock").write_bytes(b"")
    time.sleep(0.01)  # ensure fresh mtime

    out = tmp_path / "migrated"
    args = MigrateArgs(input_dir=og_dir, output_dir=out, install_as=None, force=False)
    with pytest.raises(LiveLockDetected):
        run_migrate(args)


def test_migrate_args_rejects_both_output_and_install(og_dir: Path, tmp_path: Path) -> None:
    """Passing both --output and --install-as is rejected."""
    with pytest.raises(ValueError, match="one of"):
        MigrateArgs(
            input_dir=og_dir,
            output_dir=tmp_path / "o",
            install_as="nell",
            force=False,
        )


def test_migrate_args_rejects_neither_output_nor_install(og_dir: Path) -> None:
    """Passing neither --output nor --install-as is rejected."""
    with pytest.raises(ValueError, match="one of"):
        MigrateArgs(input_dir=og_dir, output_dir=None, install_as=None, force=False)


def test_run_migrate_install_as_atomic_swap(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--install-as writes to <persona>.new then atomically renames."""
    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    args = MigrateArgs(input_dir=og_dir, output_dir=None, install_as="nell", force=False)
    run_migrate(args)

    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir("nell")
    assert persona_dir.exists()
    assert (persona_dir / "memories.db").exists()
    assert (persona_dir / "hebbian.db").exists()
    # no leftover temp dir
    assert not persona_dir.with_name("nell.new").exists()


def test_run_migrate_install_as_backs_up_existing(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--install-as --force backs up existing persona dir before overwriting."""
    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    from brain.paths import get_persona_dir

    old = get_persona_dir("nell")
    old.mkdir(parents=True)
    (old / "marker.txt").write_text("old-data")

    args = MigrateArgs(input_dir=og_dir, output_dir=None, install_as="nell", force=True)
    run_migrate(args)

    # new nell is live
    assert (get_persona_dir("nell") / "memories.db").exists()
    # old nell is backed up somewhere as nell.backup-<timestamp>
    parent = get_persona_dir("nell").parent
    backups = [p for p in parent.iterdir() if p.name.startswith("nell.backup-")]
    assert len(backups) == 1
    assert (backups[0] / "marker.txt").read_text() == "old-data"


def test_run_migrate_install_as_refuses_without_force(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing persona + no --force → error."""
    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    from brain.paths import get_persona_dir

    get_persona_dir("nell").mkdir(parents=True)

    args = MigrateArgs(input_dir=og_dir, output_dir=None, install_as="nell", force=False)
    with pytest.raises(FileExistsError, match="persona"):
        run_migrate(args)


def test_migrate_writes_reflex_arcs(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: migrator writes reflex_arcs.json from OG reflex_engine.py."""
    import textwrap

    # og_dir is tmp_path/og_data — reflex_engine.py lives one level up (NellBrain root).
    # The migrator's second candidate path is args.input_dir.parent / "reflex_engine.py".
    reflex_src = textwrap.dedent("""\
        REFLEX_ARCS = {
            "creative_pitch": {
                "trigger": {"creative_hunger": 9},
                "days_since_min": 0,
                "action": "generate_story_pitch",
                "output": "gifts",
                "cooldown_hours": 48,
                "description": "d",
                "prompt_template": "t"
            }
        }
    """)
    (tmp_path / "reflex_engine.py").write_text(reflex_src, encoding="utf-8")

    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    args = MigrateArgs(
        input_dir=og_dir,
        output_dir=None,
        install_as="testpersona",
        force=False,
    )
    report = run_migrate(args)

    from brain.paths import get_persona_dir

    target = get_persona_dir("testpersona") / "reflex_arcs.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["arcs"]) == 1
    assert data["arcs"][0]["name"] == "creative_pitch"
    assert data["arcs"][0]["output_memory_type"] == "reflex_gift"
    assert report.reflex_arcs_migrated == 1
    assert report.reflex_arcs_skipped_reason is None
