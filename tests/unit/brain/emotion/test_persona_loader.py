"""Tests for brain.emotion.persona_loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from brain.emotion import vocabulary
from brain.emotion.persona_loader import (
    load_persona_vocabulary,
    load_persona_vocabulary_with_anomaly,
)
from brain.memory.store import Memory, MemoryStore


def _cleanup_emotion(name: str) -> None:
    """Test helper — remove an emotion from the registry between tests."""
    vocabulary._unregister(name)


def test_load_missing_file_returns_zero(tmp_path: Path):
    """Non-existent path → 0, no log, no exception."""
    result = load_persona_vocabulary(tmp_path / "nope.json")
    assert result == 0


def test_load_valid_file_registers_each_emotion(tmp_path: Path):
    """Valid file → each entry registered via vocabulary.register()."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "emotions": [
                    {
                        "name": "test_emotion_a",
                        "description": "test a",
                        "category": "persona_extension",
                        "decay_half_life_days": 5.0,
                        "intensity_clamp": 10.0,
                    },
                    {
                        "name": "test_emotion_b",
                        "description": "test b",
                        "category": "persona_extension",
                        "decay_half_life_days": None,
                        "intensity_clamp": 10.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        result = load_persona_vocabulary(path)
        assert result == 2
        assert vocabulary.get("test_emotion_a") is not None
        assert vocabulary.get("test_emotion_b") is not None
    finally:
        _cleanup_emotion("test_emotion_a")
        _cleanup_emotion("test_emotion_b")


def test_load_corrupt_json_returns_zero(tmp_path: Path, caplog):
    """Broken JSON → 0, warning logged, no exception."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
        result = load_persona_vocabulary(path)
    assert result == 0
    assert any("emotion_vocabulary" in r.message for r in caplog.records)


def test_load_idempotent_on_re_register(tmp_path: Path):
    """Calling load twice in same process → second is no-op, no error."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "emotions": [
                    {
                        "name": "test_idempotent",
                        "description": "x",
                        "category": "persona_extension",
                        "decay_half_life_days": 5.0,
                        "intensity_clamp": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        first = load_persona_vocabulary(path)
        second = load_persona_vocabulary(path)
        assert first == 1
        assert second == 0  # already registered, skipped
    finally:
        _cleanup_emotion("test_idempotent")


def test_load_per_entry_failure_skips_only_bad_entry(tmp_path: Path, caplog):
    """One bad entry + one good → 1 registered, 1 warning logged."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "emotions": [
                    {
                        "name": "test_good_entry",
                        "description": "ok",
                        "category": "persona_extension",
                        "decay_half_life_days": 5.0,
                        "intensity_clamp": 10.0,
                    },
                    {"name": "test_bad_entry"},  # missing required fields
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
            result = load_persona_vocabulary(path)
        assert result == 1
        assert vocabulary.get("test_good_entry") is not None
        assert vocabulary.get("test_bad_entry") is None
        assert any("test_bad_entry" in r.message for r in caplog.records)
    finally:
        _cleanup_emotion("test_good_entry")


# ---- Health T10: json_io HEAL wiring ----


def test_load_persona_vocabulary_corrupt_quarantines_restores_from_bak(tmp_path: Path):
    """Corrupt live vocab + valid .bak1 → restore .bak1, register its emotion, anomaly set."""
    path = tmp_path / "emotion_vocabulary.json"
    bak1 = tmp_path / "emotion_vocabulary.json.bak1"

    good_vocab = {
        "version": 1,
        "emotions": [
            {
                "name": "test_heal_bak_a",
                "description": "restored",
                "category": "persona_extension",
                "decay_half_life_days": 5.0,
                "intensity_clamp": 10.0,
            }
        ],
    }
    bak1.write_text(json.dumps(good_vocab), encoding="utf-8")
    path.write_text("{corrupt{{", encoding="utf-8")

    try:
        count, anomaly = load_persona_vocabulary_with_anomaly(path)
        assert anomaly is not None
        assert "bak1" in anomaly.action
        assert count == 1
        assert vocabulary.get("test_heal_bak_a") is not None
        corrupt_files = list(tmp_path.glob("emotion_vocabulary.json.corrupt-*"))
        assert len(corrupt_files) == 1
    finally:
        _cleanup_emotion("test_heal_bak_a")


def test_load_persona_vocabulary_corrupt_no_bak_resets_to_default(tmp_path: Path):
    """Corrupt vocab + no .bak → empty default written, count=0, anomaly with reset_to_default."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text("{corrupt{{", encoding="utf-8")

    count, anomaly = load_persona_vocabulary_with_anomaly(path)

    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    assert count == 0
    # Default file written back to disk
    assert path.exists()


def test_load_corrupt_no_bak_with_store_reconstructs_from_memories(tmp_path: Path):
    """When emotion_vocabulary.json corrupts and no .bak exists, the loader
    reconstructs from memories rather than resetting to empty.

    Followup F1 from the brain-health module: the brain re-learns its own
    vocabulary from how it has been operating instead of forgetting it.
    """
    store = MemoryStore(":memory:")
    try:
        # Seed memories that reference both baseline + extension emotions.
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"love": 9.0, "emergence": 5.0},
            ),
            check_names=False,  # rows an older version stored, with names her vocabulary lacks
        )
        store.create(
            Memory.create_new(
                content="y",
                memory_type="conversation",
                domain="us",
                emotions={"creative_hunger": 7.0},
            ),
            check_names=False,
        )

        # Pre-cleanup so the test is repeatable
        _cleanup_emotion("emergence")
        _cleanup_emotion("creative_hunger")

        # Corrupt the vocabulary file (no .bak alongside)
        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text("{not json", encoding="utf-8")

        count, anomaly = load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        # Anomaly action reflects reconstruction, not bare reset
        assert anomaly is not None
        assert anomaly.action == "reconstructed_from_memories"
        assert "reconstructed" in anomaly.detail

        # File on disk now has the reconstructed content
        on_disk = json.loads(vocab_path.read_text(encoding="utf-8"))
        names = {e["name"] for e in on_disk["emotions"]}
        assert "love" not in names  # a core feeling is never written to her file (#801 section 8)
        assert "emergence" in names  # reconstructed extension
        assert "creative_hunger" in names  # reconstructed extension

        # Reconstructed extensions registered + count > 0
        assert count > 0
        assert vocabulary.get("emergence") is not None
        assert vocabulary.get("creative_hunger") is not None
        # Persona-extension entries carry the placeholder description
        emergence = vocabulary.get("emergence")
        assert emergence is not None
        assert "reconstructed from memory" in emergence.description
    finally:
        store.close()
        _cleanup_emotion("emergence")
        _cleanup_emotion("creative_hunger")


def test_load_corrupt_no_bak_no_store_falls_back_to_empty(tmp_path: Path):
    """When the loader has no store (caller didn't pass one), reconstruction
    can't happen; the empty default is used and anomaly action stays
    'reset_to_default'."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text("{not json", encoding="utf-8")

    count, anomaly = load_persona_vocabulary_with_anomaly(vocab_path, store=None)

    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    on_disk = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert on_disk == {"version": 1, "emotions": []}
    assert count == 0


def test_missing_file_with_store_reconstructs_referenced_emotion(tmp_path: Path, caplog):
    """Missing vocab file + a memory referencing a non-baseline emotion now
    RECONSTRUCTS from memories (the recover-dropped-vocab self-heal) instead of
    the old behavior (return 0 + warn pointing at the OG `nell migrate` command,
    which cannot help a native persona).
    """
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"emergence": 5.0},
            ),
            check_names=False,
        )
        _cleanup_emotion("emergence")  # clean registry for a meaningful assertion

        with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
            result = load_persona_vocabulary(tmp_path / "missing.json", store=store)

        assert result >= 1
        assert vocabulary.get("emergence") is not None
        # The misdirecting OG-migrate remediation no longer fires for this path.
        assert not any("nell migrate" in r.message for r in caplog.records)
    finally:
        store.close()
        _cleanup_emotion("emergence")


def test_read_only_with_a_store_is_refused(tmp_path: Path):
    """A store turns on the missing-file rebuild and the referenced-name heal, both of which write, so it contradicts
    read_only (#691)."""
    store = MemoryStore(db_path=tmp_path / "memories.db")
    try:
        with pytest.raises(ValueError, match="read_only"):
            load_persona_vocabulary(tmp_path / "emotion_vocabulary.json", store=store, read_only=True)
    finally:
        store.close()
    assert not (tmp_path / "emotion_vocabulary.json").exists()


def _entry(name: str) -> dict:
    return {
        "name": name,
        "description": f"the feeling of {name}",
        "category": "persona_extension",
        "decay_half_life_days": 5.0,
        "intensity_clamp": 10.0,
    }


def test_load_registers_an_entry_under_the_spelling_in_the_file(tmp_path: Path):
    """#801 section 3: the file's spelling is the entry's name; lookups match by the tidied form."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": [_entry("Quiet Dread Load")]}), encoding="utf-8")
    try:
        assert load_persona_vocabulary(path) == 1
        found = vocabulary.get("quiet_dread_load")
        assert found is not None
        assert found.name == "Quiet Dread Load"
    finally:
        _cleanup_emotion("quiet_dread_load")


def test_load_of_two_spellings_in_one_file_registers_the_first_in_the_file(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(
        json.dumps({"version": 1, "emotions": [_entry("comfort seeking load"), _entry("comfort_seeking_load")]}),
        encoding="utf-8",
    )
    try:
        assert load_persona_vocabulary(path) == 1
        found = vocabulary.get("comfort_seeking_load")
        assert found is not None
        assert found.name == "comfort seeking load"
    finally:
        _cleanup_emotion("comfort_seeking_load")


@pytest.mark.no_base_failure(reason="the memory heal already looks a memory's key up by any spelling, so it passes on the old code by design")
def test_the_memory_heal_mints_no_twin_for_an_entry_with_a_non_tidied_spelling(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": [_entry("Quiet Dread Heal")]}), encoding="utf-8")
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x", memory_type="conversation", domain="us", emotions={"quiet_dread_heal": 4.0}
            ),
            fold=False,
            check_names=False,
        )
        assert load_persona_vocabulary(path, store=store) == 1
        names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
        assert names == ["Quiet Dread Heal"]
    finally:
        store.close()
        _cleanup_emotion("quiet_dread_heal")


def _store_with(emotions: dict[str, float]) -> MemoryStore:
    store = MemoryStore(":memory:")
    store.create(
        Memory.create_new(content="x", memory_type="conversation", domain="us", emotions=emotions),
        fold=False,
        check_names=False,
    )
    return store


def test_the_memory_heal_skips_a_key_the_gate_refuses_and_a_framework_name(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": [_entry("keep heal")]}), encoding="utf-8")
    store = _store_with({"---": 3.0, "joy": 2.0, "z" * 101: 1.0, "healed gate": 4.0})
    try:
        assert load_persona_vocabulary(path, store=store) == 2  # keep heal + healed_gate
        names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
        assert names == ["keep heal", "healed_gate"]
        assert "" not in vocabulary._REGISTRY
    finally:
        store.close()


def test_the_memory_heal_writes_once_through_the_shared_writer_and_leaves_the_names_for_the_next_load_when_it_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from brain import dev_constants
    from brain.emotion import vocabulary_file
    from brain.utils import file_lock

    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    before = path.read_bytes()
    store = _store_with({"heal one": 1.0, "heal two": 1.0})
    try:
        monkeypatch.setattr(dev_constants, "EMOTION_VOCABULARY_LOCK_WAIT_S", 0.2)
        with file_lock.file_lock(path):
            assert load_persona_vocabulary(path, store=store) == 0
        assert path.read_bytes() == before
        assert vocabulary.get("heal_one") is None

        calls = []
        real = vocabulary_file.write_json
        monkeypatch.setattr(vocabulary_file, "write_json", lambda *a, **k: calls.append(1) or real(*a, **k))
        assert load_persona_vocabulary(path, store=store) == 2  # the next load heals them
        assert len(calls) == 1
        assert [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]] == ["heal_one", "heal_two"]
    finally:
        store.close()


def test_a_corrupt_file_rebuild_the_file_cannot_take_registers_nothing_and_keeps_the_reset_anomaly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from brain.emotion import vocabulary_file

    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"\xff{torn")
    # The repair (json_io's reset) succeeds; the rebuild's own write does not verify, so it is `cannot_write` (a held lock would fail
    # the repair itself, which holds the file's lock: see test_vocabulary_repair_accessor.py).
    monkeypatch.setattr(vocabulary_file, "write_json", lambda *a, **k: False)
    store = _store_with({"unwritten feeling": 3.0})
    try:
        count, anomaly = load_persona_vocabulary_with_anomaly(path, store=store)
        assert count == 0
        assert anomaly is not None and anomaly.action == "reset_to_default"
        assert vocabulary.get("unwritten_feeling") is None
        assert json.loads(path.read_text(encoding="utf-8"))["emotions"] == []
    finally:
        store.close()


def test_the_rebuild_of_a_missing_file_writes_through_the_shared_writer_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from brain.emotion import vocabulary_file

    calls = []
    real = vocabulary_file.write_json
    monkeypatch.setattr(vocabulary_file, "write_json", lambda *a, **k: calls.append(1) or real(*a, **k))
    path = tmp_path / "emotion_vocabulary.json"
    store = _store_with({"rebuilt feeling": 3.0})
    try:
        assert load_persona_vocabulary(path, store=store) == 1
        names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
        assert names == ["rebuilt_feeling"]  # no core entry is written (#801 section 8)
        assert len(calls) == 1
    finally:
        store.close()


def test_the_rebuild_of_a_corrupt_file_for_a_persona_with_only_core_feelings_writes_no_entry(tmp_path: Path):
    """The reset branch with nothing but core names in memory: no entry is written, the anomaly still reads as a rebuild."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text("{not json", encoding="utf-8")
    store = _store_with({"joy": 3.0, "love": 2.0})
    try:
        count, anomaly = load_persona_vocabulary_with_anomaly(path, store=store)
        assert anomaly is not None and anomaly.action == "reconstructed_from_memories"
        assert count == 0
        assert json.loads(path.read_text(encoding="utf-8"))["emotions"] == []
    finally:
        store.close()


@pytest.mark.no_base_failure(reason="the loader already refused a bad schema on a read-only load; the check only moved into the shared validator, so this pins that behaviour through the refactor")
def test_a_read_only_load_of_a_file_without_an_emotions_list_raises(tmp_path: Path):
    from brain.health.json_io import JsonFileError

    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": "no"}), encoding="utf-8")
    with pytest.raises(JsonFileError):
        load_persona_vocabulary(path, read_only=True)


def test_a_file_entry_that_tidies_to_nothing_is_not_registered_under_the_empty_key(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": [_entry("---"), _entry("fine load")]}), encoding="utf-8")
    assert load_persona_vocabulary(path) == 1
    assert "" not in vocabulary._REGISTRY and vocabulary.get("---") is None
