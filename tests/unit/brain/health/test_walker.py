"""Tests for brain.health.walker."""

from __future__ import annotations

from pathlib import Path

from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def _setup_persona(tmp_path: Path) -> Path:
    persona = tmp_path / "persona"
    persona.mkdir()
    MemoryStore(db_path=persona / "memories.db").close()
    HebbianMatrix(db_path=persona / "hebbian.db").close()
    return persona


def test_walk_clean_persona_no_anomalies(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    anomalies = walk_persona(persona)
    assert anomalies == []


def test_walk_detects_corrupt_atomic_rewrite_file(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    assert len(anomalies) == 1
    assert anomalies[0].file == "user_preferences.json"
    # File is healed (reset to default since no .bak)
    assert (persona / "user_preferences.json").exists()


def test_walk_detects_sqlite_corruption(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "memories.db").write_bytes(b"not sqlite")
    HebbianMatrix(db_path=persona / "hebbian.db").close()  # clean
    anomalies = walk_persona(persona)
    assert any(a.kind == "sqlite_integrity_fail" for a in anomalies)


def test_walk_returns_multiple_anomalies(tmp_path: Path) -> None:
    persona = _setup_persona(tmp_path)
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")
    (persona / "persona_config.json").write_text("{also not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    assert len(anomalies) >= 2
    files = {a.file for a in anomalies}
    assert "user_preferences.json" in files
    assert "persona_config.json" in files


def test_walk_detects_corrupt_crystallizations_db(tmp_path: Path) -> None:
    """Post-audit fix: walker now scans crystallizations.db (added in SP-5).

    Previously, soul corruption went undetected by `nell health check`.
    """
    persona = tmp_path / "persona"
    persona.mkdir()
    MemoryStore(db_path=persona / "memories.db").close()
    HebbianMatrix(db_path=persona / "hebbian.db").close()
    # Corrupt crystallizations.db with invalid SQLite header
    (persona / "crystallizations.db").write_bytes(b"not a sqlite database")

    anomalies = walk_persona(persona)
    soul_anomalies = [a for a in anomalies if a.file == "crystallizations.db"]
    assert len(soul_anomalies) == 1
    assert soul_anomalies[0].kind == "sqlite_integrity_fail"
    assert soul_anomalies[0].action == "alarmed_unrecoverable"


def test_walk_clean_persona_with_soul_db_no_anomalies(tmp_path: Path) -> None:
    """A persona with a clean crystallizations.db should NOT raise an anomaly."""
    from brain.soul.store import SoulStore

    persona = _setup_persona(tmp_path)
    SoulStore(db_path=persona / "crystallizations.db").close()  # init clean db

    anomalies = walk_persona(persona)
    assert anomalies == []


def test_walk_missing_soul_db_no_anomaly(tmp_path: Path) -> None:
    """A persona that has never crystallized has no soul db — that's expected,
    not an anomaly. Walker should skip silently."""
    persona = _setup_persona(tmp_path)
    # No crystallizations.db on disk
    anomalies = walk_persona(persona)
    assert all(a.file != "crystallizations.db" for a in anomalies)


def test_walk_missing_voicecraft_no_anomaly(tmp_path: Path) -> None:
    """voicecraft.md is optional — most personas don't have one. Missing
    should NOT raise an anomaly (that would noise up `nell health check`
    for every persona that doesn't author the voice-craft reference doc)."""
    persona = _setup_persona(tmp_path)
    # No voicecraft.md on disk — common case
    anomalies = walk_persona(persona)
    assert all(a.file != "voicecraft.md" for a in anomalies)


def test_walk_clean_voicecraft_no_anomaly(tmp_path: Path) -> None:
    """voicecraft.md present and well-formed → no anomaly."""
    persona = _setup_persona(tmp_path)
    (persona / "voicecraft.md").write_text(
        "# Voice Craft\n\nReal content here.\n", encoding="utf-8"
    )
    anomalies = walk_persona(persona)
    assert all(a.file != "voicecraft.md" for a in anomalies)


def test_walk_corrupt_voicecraft_quarantines_and_heals(tmp_path: Path) -> None:
    """voicecraft.md emptied (the practical disk-corruption mode for plain
    text) → walker quarantines + heals from .bak1 if present.
    """
    persona = _setup_persona(tmp_path)
    # Healthy backup
    (persona / "voicecraft.md.bak1").write_text(
        "# Voice Craft\n\nGood prior content.\n", encoding="utf-8"
    )
    # Live file emptied (the heal trigger for plain text per attempt_heal_text)
    (persona / "voicecraft.md").write_text("", encoding="utf-8")

    anomalies = walk_persona(persona)
    voicecraft_anomalies = [a for a in anomalies if a.file == "voicecraft.md"]
    assert len(voicecraft_anomalies) == 1
    assert voicecraft_anomalies[0].action == "restored_from_bak1"
    # Live file has been restored from .bak1
    assert "Good prior content" in (persona / "voicecraft.md").read_text(encoding="utf-8")


def test_walk_corrupt_voicecraft_no_bak_resets_to_empty(tmp_path: Path) -> None:
    """voicecraft.md empty + no .bak → walker writes empty default + flags anomaly.

    Empty default (rather than fabricated content) is intentional: voicecraft
    is reference content the user wrote; we don't pretend to reconstruct it.
    The anomaly tells the user it was lost; they restore from VCS.
    """
    persona = _setup_persona(tmp_path)
    (persona / "voicecraft.md").write_text("", encoding="utf-8")  # corrupt, no bak

    anomalies = walk_persona(persona)
    voicecraft_anomalies = [a for a in anomalies if a.file == "voicecraft.md"]
    assert len(voicecraft_anomalies) == 1
    assert voicecraft_anomalies[0].action == "reset_to_default"


def _persona_with_heal_marker_and_words(tmp_path: Path, words: list[str]) -> Path:
    """A persona whose memories carry ``words`` and whose one-time vocabulary heal has already run."""
    from brain.health.json_io import write_json
    from brain.memory.store import Memory
    from brain.paths import cadence_state_path

    persona = _setup_persona(tmp_path)
    store = MemoryStore(db_path=persona / "memories.db")
    try:
        for word in words:
            store.create(
                Memory.create_new(content=word, memory_type="conversation", domain="us", emotions={word: 8.0})
            )
    finally:
        store.close()
    write_json(cadence_state_path(persona, "vocab_heal_final.json"), {"healed_at": "x", "count": 0})
    return persona


def _load_with_store(persona: Path) -> None:
    from brain.emotion.persona_loader import load_persona_vocabulary

    store = MemoryStore(db_path=persona / "memories.db")
    try:
        load_persona_vocabulary(persona / "emotion_vocabulary.json", store=store)
    finally:
        store.close()


def test_walker_reset_of_vocabulary_re_arms_the_heal(tmp_path: Path) -> None:
    """The walker resets a corrupt vocabulary with no store; the next store-carrying load re-learns the words."""
    from brain.emotion import vocabulary

    persona = _persona_with_heal_marker_and_words(tmp_path, ["kappa_feel"])
    (persona / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")
    vocabulary._unregister("kappa_feel")

    anomalies = walk_persona(persona)
    assert any(a.file == "emotion_vocabulary.json" and a.action == "reset_to_default" for a in anomalies)

    _load_with_store(persona)

    assert vocabulary.get("kappa_feel") is not None


def test_walker_bak_restore_re_arms_the_heal(tmp_path: Path) -> None:
    """The walker restores an older .bak1 that lacks the newest word; the next store-carrying load re-learns it."""
    import json

    from brain.emotion import vocabulary

    persona = _persona_with_heal_marker_and_words(tmp_path, ["lambda_feel", "mu_feel"])
    old_entry = {
        "name": "lambda_feel",
        "description": "d",
        "category": "persona_extension",
        "decay_half_life_days": 14.0,
        "intensity_clamp": 10.0,
    }
    (persona / "emotion_vocabulary.json.bak1").write_text(
        json.dumps({"version": 1, "emotions": [old_entry]}), encoding="utf-8"
    )
    (persona / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")
    vocabulary._unregister("lambda_feel")
    vocabulary._unregister("mu_feel")

    anomalies = walk_persona(persona)
    assert any(a.file == "emotion_vocabulary.json" and a.action == "restored_from_bak1" for a in anomalies)

    _load_with_store(persona)

    assert vocabulary.get("lambda_feel") is not None
    assert vocabulary.get("mu_feel") is not None
    names = {e["name"] for e in json.loads((persona / "emotion_vocabulary.json").read_text(encoding="utf-8"))["emotions"]}
    assert names == {"lambda_feel", "mu_feel"}


def test_walker_reset_then_same_process_load_restores_the_word(tmp_path: Path) -> None:
    """The bridge's registry still holds the word when the walker resets the file; the next load must write it back."""
    import json

    from brain.emotion import vocabulary

    persona = _setup_persona(tmp_path)
    store = MemoryStore(db_path=persona / "memories.db")
    try:
        from brain.memory.store import Memory

        store.create(Memory.create_new(content="x", memory_type="conversation", domain="us", emotions={"nu_feel": 8.0}))
    finally:
        store.close()
    vocab_path = persona / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    _load_with_store(persona)
    assert vocabulary.get("nu_feel") is not None
    for bak in persona.glob("emotion_vocabulary.json.bak*"):
        bak.unlink()
    vocab_path.write_text("{not json", encoding="utf-8")

    anomalies = walk_persona(persona)
    assert any(a.file == "emotion_vocabulary.json" and a.action == "reset_to_default" for a in anomalies)

    _load_with_store(persona)  # same process: no _unregister

    names = {e["name"] for e in json.loads(vocab_path.read_text(encoding="utf-8"))["emotions"]}
    assert "nu_feel" in names


def test_walker_keeps_going_when_the_heal_marker_cannot_be_removed(tmp_path: Path, monkeypatch, caplog) -> None:
    """A failed unlink of the heal marker is logged; the walk still checks every file and returns its anomalies."""
    import logging

    persona = _persona_with_heal_marker_and_words(tmp_path, ["omicron_feel"])
    (persona / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")
    (persona / "voicecraft.md").write_text("", encoding="utf-8")  # walked after the JSON files
    real_unlink = Path.unlink

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "vocab_heal_final.json":
            raise PermissionError("marker is locked")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with caplog.at_level(logging.WARNING, logger="brain.health.walker"):
        anomalies = walk_persona(persona)

    assert {"emotion_vocabulary.json", "voicecraft.md"} <= {a.file for a in anomalies}
    assert any(r.name == "brain.health.walker" and "vocab_heal_final.json" in r.getMessage() for r in caplog.records)
