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
    register_new_word,
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
                emotions={"love": 9.0, "body_grief": 5.0},
            )
        )
        store.create(
            Memory.create_new(
                content="y",
                memory_type="conversation",
                domain="us",
                emotions={"creative_hunger": 7.0},
            )
        )

        # Pre-cleanup so the test is repeatable
        _cleanup_emotion("body_grief")
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
        assert "love" in names  # baseline
        assert "body_grief" in names  # reconstructed extension
        assert "creative_hunger" in names  # reconstructed extension

        # Reconstructed extensions registered + count > 0
        assert count > 0
        assert vocabulary.get("body_grief") is not None
        assert vocabulary.get("creative_hunger") is not None
        # Persona-extension entries carry the placeholder description
        body_grief = vocabulary.get("body_grief")
        assert body_grief is not None
        assert "reconstructed from memory" in body_grief.description
    finally:
        store.close()
        _cleanup_emotion("body_grief")
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
                emotions={"body_grief": 5.0},
            )
        )
        _cleanup_emotion("body_grief")  # clean registry for a meaningful assertion

        with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
            result = load_persona_vocabulary(tmp_path / "missing.json", store=store)

        assert result >= 1
        assert vocabulary.get("body_grief") is not None
        # The misdirecting OG-migrate remediation no longer fires for this path.
        assert not any("nell migrate" in r.message for r in caplog.records)
    finally:
        store.close()
        _cleanup_emotion("body_grief")


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


def test_register_new_word_appends_and_registers(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    assert register_new_word(path, "gleam_word") == "gleam_word"
    names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
    assert "gleam_word" in names
    emotion = vocabulary.get("gleam_word")
    assert emotion is not None
    assert emotion.decay_half_life_days == 14.0


def test_register_new_word_uses_given_half_life(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    assert register_new_word(path, "slow_word", half_life_days=2.5) == "slow_word"
    emotion = vocabulary.get("slow_word")
    assert emotion is not None
    assert emotion.decay_half_life_days == 2.5


def test_register_new_word_refuses_invalid_name(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text('{"version": 1, "emotions": []}', encoding="utf-8")
    before = path.read_bytes()
    assert register_new_word(path, "Love!!") is None
    assert path.read_bytes() == before


def test_register_new_word_creates_the_file_for_a_fresh_persona(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    assert not path.exists()
    assert register_new_word(path, "fresh_word") == "fresh_word"
    names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
    assert names == ["fresh_word"]


def test_register_new_word_keeps_entry_written_by_another_process(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    assert register_new_word(path, "alpha_word") == "alpha_word"
    vocabulary._unregister("alpha_word")  # as if another process had written it
    assert register_new_word(path, "beta_word") == "beta_word"
    names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
    assert names == ["alpha_word", "beta_word"]


def test_register_new_word_refuses_on_corrupt_file(tmp_path: Path):
    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"{not json")
    assert register_new_word(path, "ghost_word") is None
    assert path.read_bytes() == b"{not json"
    assert vocabulary.get("ghost_word") is None


def test_refresh_if_changed_picks_up_other_process_write(tmp_path: Path):
    from brain.emotion.persona_loader import refresh_if_changed
    from brain.health.json_io import write_json

    path = tmp_path / "emotion_vocabulary.json"
    path.write_text('{"version": 1, "emotions": []}', encoding="utf-8")
    load_persona_vocabulary(path)
    entry = {
        "name": "outside_word",
        "description": "d",
        "category": "persona_extension",
        "decay_half_life_days": 14.0,
        "intensity_clamp": 10.0,
    }
    write_json(path, {"version": 1, "emotions": [entry]})
    assert vocabulary.get("outside_word") is None
    assert refresh_if_changed(path) == 1
    assert vocabulary.get("outside_word") is not None
    assert refresh_if_changed(path) == 0


def test_refresh_if_changed_never_writes_a_corrupt_file(tmp_path: Path, monkeypatch):
    from brain.emotion import persona_loader
    from brain.emotion.persona_loader import refresh_if_changed
    from brain.health.json_io import JsonFileError

    path = tmp_path / "emotion_vocabulary.json"
    path.write_bytes(b"{not json")
    with pytest.raises(JsonFileError):
        refresh_if_changed(path)
    assert path.read_bytes() == b"{not json"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "emotion_vocabulary.json.lock") == [
        "emotion_vocabulary.json"
    ]

    calls: list[Path] = []
    real = persona_loader.load_persona_vocabulary
    monkeypatch.setattr(
        persona_loader,
        "load_persona_vocabulary",
        lambda p, **kw: calls.append(p) or real(p, **kw),
    )
    assert refresh_if_changed(path) == 0
    assert calls == []


def test_refresh_if_changed_on_a_missing_file_returns_zero_and_writes_nothing(tmp_path: Path) -> None:
    from brain.emotion.persona_loader import refresh_if_changed

    path = tmp_path / "emotion_vocabulary.json"
    assert refresh_if_changed(path) == 0
    assert refresh_if_changed(path) == 0
    assert list(tmp_path.iterdir()) == []


def test_load_skips_a_word_another_thread_registered_after_the_check(tmp_path: Path, monkeypatch):
    """The registry gains the word between the loader's check and its register: the load does not raise."""
    from brain.emotion.persona_loader import refresh_if_changed
    from brain.emotion.vocabulary import Emotion

    entry = {
        "name": "racing_word",
        "description": "d",
        "category": "persona_extension",
        "decay_half_life_days": 14.0,
        "intensity_clamp": 10.0,
    }
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({"version": 1, "emotions": [entry]}), encoding="utf-8")
    real_get = vocabulary.get

    def get_then_lose_the_race(name: str):
        found = real_get(name)
        if name == "racing_word" and found is None:
            vocabulary.register(
                Emotion(name="racing_word", description="d", category="persona_extension", decay_half_life_days=14.0)
            )
        return found

    monkeypatch.setattr(vocabulary, "get", get_then_lose_the_race)

    assert refresh_if_changed(path) == 0
    assert real_get("racing_word") is not None
