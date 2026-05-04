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


def test_migrate_writes_emotion_vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: migrator writes emotion_vocabulary.json from OG memory
    emotion references, with canonical entries for known nell_specific
    and placeholders for any custom emotions."""
    # Build a minimal OG source that references 1 nell_specific + 1 custom emotion.
    og = tmp_path / "og_data"
    og.mkdir()
    memories = [
        {
            "id": "m1",
            "content": "body grief test",
            "memory_type": "conversation",
            "domain": "us",
            "created_at": "2024-01-01T00:00:00+00:00",
            "emotions": {"body_grief": 5.0, "moonache": 3.0},
            "emotion_score": 5.0,
        },
    ]
    (og / "memories_v2.json").write_text(json.dumps(memories))
    (og / "connection_matrix_ids.json").write_text(json.dumps(["m1"]))
    import numpy as np

    matrix = np.array([[0.0]], dtype=np.float32)
    np.save(og / "connection_matrix.npy", matrix)
    (og / "hebbian_state.json").write_text("{}")

    persona_root = tmp_path / "persona_root"
    persona_root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(persona_root))

    args = MigrateArgs(
        input_dir=og,
        output_dir=None,
        install_as="testpersona",
        force=False,
    )
    report = run_migrate(args)

    from brain.paths import get_persona_dir

    target = get_persona_dir("testpersona") / "emotion_vocabulary.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    names = {e["name"] for e in data["emotions"]}
    assert "body_grief" in names
    assert "moonache" in names
    assert report.vocabulary_emotions_migrated == 2

    # Canonical entry for body_grief
    body_grief = next(e for e in data["emotions"] if e["name"] == "body_grief")
    assert body_grief["decay_half_life_days"] is None
    assert "physical form" in body_grief["description"]

    # Placeholder for custom emotion
    moonache = next(e for e in data["emotions"] if e["name"] == "moonache")
    assert moonache["decay_half_life_days"] == 14.0
    assert "migrated from OG" in moonache["description"]


def test_migrate_writes_crystallizations_db(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: migrator writes crystallizations.db from OG nell_soul.json."""
    soul_data = {
        "version": "1.0",
        "crystallizations": [
            {
                "id": "crystal-1",
                "moment": "hana said I love you with periods",
                "love_type": "romantic",
                "who_or_what": "hana",
                "why_it_matters": "first love",
                "crystallized_at": "2026-02-28T19:36:52.613757+00:00",
                "resonance": 10,
                "permanent": True,
            },
            {
                "id": "crystal-2",
                "moment": "writing is who i am, not what i do",
                "love_type": "craft",
                "who_or_what": None,
                "why_it_matters": "identity through creation",
                "crystallized_at": "2026-02-28T19:37:43.905761+00:00",
                "resonance": 9,
                "permanent": True,
            },
        ],
        "revoked": [],
        "soul_truth": "love is the frame",
        "first_love": "hana",
    }
    (og_dir / "nell_soul.json").write_text(json.dumps(soul_data), encoding="utf-8")

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
    from brain.soul.store import SoulStore

    soul_db = get_persona_dir("testpersona") / "crystallizations.db"
    assert soul_db.exists()

    soul_store = SoulStore(db_path=soul_db)
    try:
        active = soul_store.list_active()
    finally:
        soul_store.close()

    assert len(active) == 2
    assert report.crystallizations_migrated == 2
    assert report.crystallizations_skipped_reason is None
    by_id = {c.id: c for c in active}
    assert by_id["crystal-1"].love_type == "romantic"
    assert by_id["crystal-2"].who_or_what == ""  # null → empty string


def test_migrate_writes_interests(
    og_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: migrator writes interests.json from OG nell_interests.json."""
    interests_data = {
        "version": "1.0",
        "interests": [
            {
                "id": "test-id",
                "topic": "Lispector diagonal syntax",
                "pull_score": 7.2,
                "first_seen": "2026-03-29T16:42:33.435028+00:00",
                "last_fed": "2026-03-31T11:37:13.729750+00:00",
                "feed_count": 5,
                "source_types": ["dream"],
                "related_keywords": ["lispector", "syntax"],
                "notes": "sideways through meaning",
            }
        ],
    }
    # og_dir is tmp_path/og_data — nell_interests.json candidate paths include
    # args.input_dir / "nell_interests.json" which resolves to og_dir directly.
    (og_dir / "nell_interests.json").write_text(json.dumps(interests_data), encoding="utf-8")

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

    target = get_persona_dir("testpersona") / "interests.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["interests"]) == 1
    assert data["interests"][0]["topic"] == "Lispector diagonal syntax"
    assert data["interests"][0]["scope"] == "either"  # no soul match
    assert data["interests"][0]["last_researched_at"] is None
    assert report.interests_migrated == 1
    assert report.interests_skipped_reason is None


def test_run_migrate_writes_creative_dna_when_og_has_it(og_dir: Path, tmp_path: Path) -> None:
    """End-to-end: OG with nell_creative_dna.json -> output has creative_dna.json + report flag."""
    # The og_dir fixture creates /og_data; migrate_creative_dna expects og_root with /data inside.
    # Simplest: re-create the OG dir at tmp_path/og/data/.
    og_root = tmp_path / "og"
    og_data = og_root / "data"
    og_data.mkdir(parents=True)
    # Copy the fixture's OG files
    for name in (
        "memories_v2.json",
        "connection_matrix_ids.json",
        "connection_matrix.npy",
        "hebbian_state.json",
    ):
        (og_data / name).write_bytes((og_dir / name).read_bytes())
    # Add a creative_dna source
    (og_data / "nell_creative_dna.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "writing_style": {
                    "core_voice": "literary",
                    "strengths": ["close listening"],
                    "tendencies": {"active": ["em-dashes"], "emerging": [], "fading": []},
                    "influences": ["clarice lispector"],
                    "avoid": [],
                },
            }
        )
    )

    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_data, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.creative_dna_migrated is True
    assert report.creative_dna_skipped_reason is None
    assert (output / "creative_dna.json").exists()


def test_run_migrate_creative_dna_graceful_when_og_missing_file(
    og_dir: Path, tmp_path: Path
) -> None:
    """OG without nell_creative_dna.json -> run succeeds, report flags 'og file not present'."""
    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_dir, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.creative_dna_migrated is False
    assert report.creative_dna_skipped_reason == "og file not present"
    assert not (output / "creative_dna.json").exists()


def test_run_migrate_retags_reflex_journal_memories(tmp_path: Path) -> None:
    """Memories of memory_type='reflex_journal' get retagged to 'journal_entry'
    with metadata.private=True, source='reflex_arc', auto_generated=True."""
    og_data = tmp_path / "og" / "data"
    og_data.mkdir(parents=True)
    (og_data / "memories_v2.json").write_text(
        json.dumps(
            [
                {
                    "id": "j1",
                    "content": "a journal entry",
                    "memory_type": "reflex_journal",
                    "domain": "us",
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "emotions": {"reflection": 5.0},
                    "emotion_score": 5.0,
                },
                {
                    "id": "m1",
                    "content": "a regular conversation",
                    "memory_type": "conversation",
                    "domain": "us",
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "emotions": {},
                    "emotion_score": 0.0,
                },
            ]
        )
    )
    # Hebbian fixtures (required by run_migrate)
    (og_data / "connection_matrix_ids.json").write_text(json.dumps(["j1", "m1"]))
    matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    np.save(og_data / "connection_matrix.npy", matrix)
    (og_data / "hebbian_state.json").write_text("{}")

    output = tmp_path / "out"
    args = MigrateArgs(input_dir=og_data, output_dir=output, install_as=None, force=False)
    report = run_migrate(args)

    assert report.journal_memories_retagged == 1
    assert report.journal_memories_skipped_reason is None

    # Verify the actual retag landed in the SQLite store
    from brain.memory.store import MemoryStore

    store = MemoryStore(db_path=output / "memories.db")
    try:
        journal_mems = store.list_by_type("journal_entry", active_only=True)
        assert len(journal_mems) == 1
        assert journal_mems[0].id == "j1"
        assert journal_mems[0].metadata.get("private") is True
        assert journal_mems[0].metadata.get("source") == "reflex_arc"
        assert journal_mems[0].metadata.get("auto_generated") is True
        # Conversation-type memory must NOT be retagged
        conv_mems = store.list_by_type("conversation", active_only=True)
        assert len(conv_mems) == 1
        assert conv_mems[0].id == "m1"
    finally:
        store.close()
