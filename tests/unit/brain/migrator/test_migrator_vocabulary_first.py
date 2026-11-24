"""#801 section 9 (step 8): the OG migrator writes the new persona's first vocabulary file through the shared function, BEFORE it
creates the migrated memories, so the memories' names are checked against that file.

The migrator is one of the two noted exceptions to "only the bridge writes her vocabulary": it acts as the owner of a work folder
no bridge serves. It writes only through `brain.emotion.vocabulary_file` (criterion 10). Sources and personas here are synthetic
tmp folders (`NELLBRAIN_HOME` points at one); nothing touches a real persona.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from brain.emotion import vocabulary, vocabulary_file
from brain.health import json_io
from brain.memory.store import MemoryStore
from brain.migrator.cli import MigrateArgs, run_migrate
from brain.paths import get_persona_dir

VOCAB = "emotion_vocabulary.json"


def _og_source(tmp_path: Path, memories: list[dict]) -> Path:
    og = tmp_path / "og_data"
    og.mkdir()
    (og / "memories_v2.json").write_text(json.dumps(memories))
    (og / "connection_matrix_ids.json").write_text(json.dumps([m["id"] for m in memories]))
    n = len(memories)
    np.save(og / "connection_matrix.npy", np.zeros((n, n), dtype=np.float32))
    (og / "hebbian_state.json").write_text("{}")
    return og


def _mem(mid: str, emotions: dict[str, float]) -> dict:
    return {
        "id": mid,
        "content": f"content of {mid}",
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2024-01-01T00:00:00+00:00",
        "emotions": emotions,
        "emotion_score": max(emotions.values()) if emotions else 0.0,
    }


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))
    return root


def _migrate(og: Path, *, force: bool = False):
    return run_migrate(MigrateArgs(input_dir=og, output_dir=None, install_as="testpersona", force=force))


def _stored_names(persona_dir: Path) -> dict[str, list[str]]:
    """The names each migrated row holds as stored (raw `emotions_json`, not the read-time check)."""
    conn = sqlite3.connect(persona_dir / "memories.db")
    try:
        rows = conn.execute("SELECT id, emotions_json FROM memories").fetchall()
    finally:
        conn.close()
    return {rid: sorted(json.loads(raw or "{}")) for rid, raw in rows}


def _file_names(persona_dir: Path) -> list[str]:
    return [e["name"] for e in json.loads((persona_dir / VOCAB).read_text(encoding="utf-8"))["emotions"]]


@pytest.fixture
def vocabulary_writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """(module that wrote, file name) of every `write_json` in `brain.*` that targets a path naming her vocabulary file.

    Every module that bound the real `write_json` gets its own spy (a `from ... import write_json` is a separate name), plus
    json_io itself; the real write still happens.
    """
    import brain.bridge.server  # noqa: F401  bound before patching, so no module is first imported (and binds a spy) during the test
    import brain.cli  # noqa: F401

    real = json_io.write_json
    calls: list[tuple[str, str]] = []

    def make_spy(module_name: str):
        def spy(path, *args, **kwargs):
            calls.append((module_name, Path(path).name))
            return real(path, *args, **kwargs)

        return spy

    for name, module in list(sys.modules.items()):
        if name.startswith("brain") and getattr(module, "write_json", None) is real:
            monkeypatch.setattr(module, "write_json", make_spy(name))
    return calls


def test_every_write_of_the_vocabulary_file_in_a_migration_comes_from_the_shared_module(home, tmp_path, vocabulary_writes) -> None:
    og = _og_source(tmp_path, [_mem("m1", {"moonache_vx": 3.0})])
    _migrate(og)
    vocabulary_names = [(module, name) for module, name in vocabulary_writes if name == VOCAB]
    assert vocabulary_names, "the migration wrote no vocabulary file"
    assert {module for module, _name in vocabulary_names} == {"brain.emotion.vocabulary_file"}


def test_the_vocabulary_file_exists_with_the_migrated_names_before_the_first_memory_is_created(home, tmp_path, monkeypatch) -> None:
    og = _og_source(tmp_path, [_mem("m1", {"orderly_vx": 3.0}), _mem("m2", {"orderly_vx": 2.0})])
    assert vocabulary.get("orderly_vx") is None
    seen: list[dict] = []
    real_create = MemoryStore.create

    def spy(self, memory, **kwargs):
        file = self.persona_dir / VOCAB
        seen.append({"file": file.exists(), "names": _file_names(self.persona_dir) if file.exists() else None, "registered": vocabulary.get("orderly_vx") is not None})
        return real_create(self, memory, **kwargs)

    monkeypatch.setattr(MemoryStore, "create", spy)
    _migrate(og)
    assert seen and seen[0] == {"file": True, "names": ["orderly_vx"], "registered": True}


def test_the_migrated_rows_names_are_checked_against_the_file_just_written(home, tmp_path) -> None:
    """`Moon Ache VX` is written to the file tidied (`moon_ache_vx`), so the row is stored under that spelling; `Joy` is a core
    feeling, the file never holds it, and the row keeps it (spelled `joy`)."""
    og = _og_source(tmp_path, [_mem("m1", {"Moon Ache VX": 3.0, "Joy": 5.0})])
    report = _migrate(og)
    persona = get_persona_dir("testpersona")
    assert _file_names(persona) == ["moon_ache_vx"]
    assert _stored_names(persona) == {"m1": ["joy", "moon_ache_vx"]}
    assert report.vocabulary_emotions_migrated == 1
    assert report.vocabulary_skipped_reason is None


def test_the_count_is_the_entries_written_not_the_spellings_seen(home, tmp_path) -> None:
    """Two spellings of one word give one entry; a core spelling (`Joy`) is refused and not counted."""
    og = _og_source(tmp_path, [_mem("m1", {"Twin Word VX": 3.0, "twin_word_vx": 2.0, "Joy": 4.0})])
    report = _migrate(og)
    persona = get_persona_dir("testpersona")
    assert _file_names(persona) == ["twin_word_vx"]
    assert report.vocabulary_emotions_migrated == 1


def test_a_core_spelling_is_never_written_to_the_migrated_file(home, tmp_path) -> None:
    """The extract subtracts only exact core names, so `Joy` reaches the add, which refuses a core feeling."""
    og = _og_source(tmp_path, [_mem("m1", {"Joy": 5.0, "Contentment": 4.0, "kept_word_vx": 2.0})])
    _migrate(og)
    assert _file_names(get_persona_dir("testpersona")) == ["kept_word_vx"]


def test_a_source_with_no_feeling_outside_the_core_set_leaves_no_vocabulary_file(home, tmp_path) -> None:
    """The shared add writes nothing when no name is accepted: the file is absent and the rows keep their core names."""
    og = _og_source(tmp_path, [_mem("m1", {"joy": 5.0})])
    report = _migrate(og)
    persona = get_persona_dir("testpersona")
    assert not (persona / VOCAB).exists()
    assert _stored_names(persona) == {"m1": ["joy"]}
    assert report.vocabulary_emotions_migrated == 0
    assert report.vocabulary_skipped_reason is None


def test_a_name_the_gate_refuses_is_dropped_from_the_migrated_row(home, tmp_path) -> None:
    """Spec section 3, outcome 3: a name that tidies to nothing is discarded with its value (the file is written first)."""
    og = _og_source(tmp_path, [_mem("m1", {"---": 4.0, "held_word_vx": 2.0})])
    _migrate(og)
    persona = get_persona_dir("testpersona")
    assert _stored_names(persona) == {"m1": ["held_word_vx"]}


def test_a_vocabulary_that_cannot_be_written_loses_no_name_from_the_migrated_rows(home, tmp_path, monkeypatch) -> None:
    """With no file the check would discard every name outside the core set, so the rows keep their names as given."""

    def cannot_write(path, items):
        return [vocabulary_file.NameResult(raw, None, "cannot_write") for raw, _fields in items]

    monkeypatch.setattr(vocabulary_file, "add", cannot_write)
    og = _og_source(tmp_path, [_mem("m1", {"Odd Word VX": 3.0, "joy": 1.0})])
    report = _migrate(og)
    persona = get_persona_dir("testpersona")
    assert report.vocabulary_skipped_reason is not None and report.vocabulary_skipped_reason.startswith("migrate_error")
    assert report.vocabulary_emotions_migrated == 0
    assert _stored_names(persona) == {"m1": ["Odd Word VX", "joy"]}


@pytest.mark.no_base_failure(reason="pins that a re-run with --force replaces the old file (the folder is cleared first); the base overwrote it too")
def test_a_forced_re_migration_holds_only_the_new_migrations_names(home, tmp_path) -> None:
    first = _og_source(tmp_path, [_mem("m1", {"first_run_vx": 3.0})])
    _migrate(first)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _og_source(second_dir, [_mem("m1", {"second_run_vx": 3.0})])
    _migrate(second, force=True)
    assert _file_names(get_persona_dir("testpersona")) == ["second_run_vx"]
