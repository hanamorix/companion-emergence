"""Recover-path vocabulary regression: a MISSING emotion_vocabulary.json whose
memories reference extractor-minted, non-baseline emotions must self-heal by
reconstructing from memories — instead of orphaning the emotions (which broke
soul crystallization) and misdirecting the user to the OG migrate path.

Root cause: brain/recovery/engine.py backs up only memories.db + hebbian.db, so
`nell recover` drops the grown emotion_vocabulary.json; load_persona_vocabulary
then early-returned on the missing file without reconstructing.
"""

from __future__ import annotations

import json
from pathlib import Path

from brain.emotion import vocabulary
from brain.emotion.persona_loader import load_persona_vocabulary_with_anomaly
from brain.memory.store import Memory, MemoryStore


def _cleanup_emotion(name: str) -> None:
    vocabulary._unregister(name)


def test_missing_file_with_store_reconstructs_persona_extensions(tmp_path: Path):
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="[monologue emotion influence: warmth:1.20]",
                memory_type="monologue_emotion",
                domain="monologue",
                emotions={"warmth": 8.0},
            )
        )
        _cleanup_emotion("warmth")  # clean registry for a meaningful assertion

        vocab_path = tmp_path / "emotion_vocabulary.json"
        assert not vocab_path.exists()

        count, anomaly = load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        # 'warmth' is reconstructed + registered (no longer orphaned).
        assert vocabulary.get("warmth") is not None
        assert count >= 1
        # The vocabulary file is now written with the recovered extension.
        assert vocab_path.exists()
        names = {e["name"] for e in json.loads(vocab_path.read_text())["emotions"]}
        assert "warmth" in names
        # A missing file is not an integrity anomaly — silent self-heal.
        assert anomaly is None
    finally:
        store.close()
        _cleanup_emotion("warmth")


def test_existing_file_heals_referenced_unregistered_emotion(tmp_path: Path):
    """File EXISTS but a memory references a non-baseline emotion absent from it
    (an extractor-minted 'warmth') → the loader registers it as a
    persona_extension and appends it to the file, instead of warning + pointing
    at the OG migrate path. Closes the file-exists residual of the recover bug."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="[monologue emotion influence: warmth:1.20]",
                memory_type="monologue_emotion",
                domain="monologue",
                emotions={"warmth": 8.0},
            )
        )
        _cleanup_emotion("warmth")

        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")

        load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        # 'warmth' is healed: registered + persisted to the existing file.
        assert vocabulary.get("warmth") is not None
        names = {e["name"] for e in json.loads(vocab_path.read_text())["emotions"]}
        assert "warmth" in names
    finally:
        store.close()
        _cleanup_emotion("warmth")
