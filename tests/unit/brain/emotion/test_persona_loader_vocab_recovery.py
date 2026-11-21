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
import logging
from datetime import UTC, datetime
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


def test_heal_mints_canonical_name_for_variant_memory_key(tmp_path: Path):
    """#174: a memory keyed 'present-ness' heals as 'present_ness', not a hyphen twin."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"present-ness": 8.0},
            )
        )
        _cleanup_emotion("present_ness")

        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")

        load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        names = {e["name"] for e in json.loads(vocab_path.read_text())["emotions"]}
        assert names == {"present_ness"}
        assert vocabulary.get("present-ness") is not None
    finally:
        store.close()
        _cleanup_emotion("present_ness")


def test_heal_names_a_non_finite_valued_emotion_and_logs_no_drop(tmp_path: Path, caplog):
    """A stored emotion whose value is NaN still names an emotion the memory uses, so the heal registers it.

    The heal reads names only: it neither skips the name nor logs "dropping emotion" on every heartbeat load.
    `create_new` drops non-finite values, so the memory is built the way a loaded row is.
    """
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.from_stored_row(
                id="nan-valued-row",
                content="x",
                memory_type="conversation",
                domain="us",
                created_at=datetime.now(UTC),
                emotions={"Nan-Ache": float("nan"), "finite_ache": 6.0},
            ),
            fold=False,
        )
        _cleanup_emotion("nan_ache")
        _cleanup_emotion("finite_ache")

        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        names = {e["name"] for e in json.loads(vocab_path.read_text())["emotions"]}
        assert names == {"nan_ache", "finite_ache"}
        assert vocabulary.get("nan_ache") is not None
        assert not [r for r in caplog.records if "dropping emotion" in r.getMessage()]
    finally:
        store.close()
        _cleanup_emotion("nan_ache")
        _cleanup_emotion("finite_ache")


def test_reconstruct_mints_canonical_names(tmp_path: Path):
    """#174: reconstruct_vocabulary_from_memories collapses variant keys to one entry."""
    from brain.health.reconstruct import reconstruct_vocabulary_from_memories

    store = MemoryStore(":memory:")
    try:
        for key in ("love_grief_blend", "grief-love_blend"):
            store.create(
                Memory.create_new(
                    content="x", memory_type="conversation", domain="us", emotions={key: 8.0}
                )
            )
        data = reconstruct_vocabulary_from_memories(store)
        ext = [e["name"] for e in data["emotions"] if e["category"] == "persona_extension"]
        assert ext == ["grief_love_blend"]
    finally:
        store.close()


def test_heal_runs_once_then_marker_stops_it(tmp_path: Path):
    """The referenced-but-unregistered scan is a one-time backfill: a marker stops later loads rerunning it."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="a", memory_type="conversation", domain="us", emotions={"zeta_feel": 8.0}
            )
        )
        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")

        load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        assert vocabulary.get("zeta_feel") is not None
        marker = tmp_path / "cadence" / "vocab_heal_final.json"
        assert marker.exists()

        store.create(
            Memory.create_new(
                content="b", memory_type="conversation", domain="us", emotions={"eta_feel": 8.0}
            )
        )
        load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        assert vocabulary.get("eta_feel") is None
    finally:
        store.close()


def test_anomaly_on_a_store_load_heals_even_with_the_marker(tmp_path: Path):
    """A repaired vocabulary file can lose words only memories still carry, so an anomaly re-runs the heal."""
    from brain.health.json_io import write_json
    from brain.paths import cadence_state_path

    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="a", memory_type="conversation", domain="us", emotions={"iota_feel": 8.0}
            )
        )
        write_json(cadence_state_path(tmp_path, "vocab_heal_final.json"), {"healed_at": "x", "count": 0})
        vocab_path = tmp_path / "emotion_vocabulary.json"
        (tmp_path / "emotion_vocabulary.json.bak1").write_text(
            json.dumps({"version": 1, "emotions": []}), encoding="utf-8"
        )
        vocab_path.write_text("{corrupt{{", encoding="utf-8")
        _cleanup_emotion("iota_feel")

        _count, anomaly = load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        assert anomaly is not None
        assert vocabulary.get("iota_feel") is not None
        names = {e["name"] for e in json.loads(vocab_path.read_text(encoding="utf-8"))["emotions"]}
        assert "iota_feel" in names
    finally:
        store.close()


def test_bak_restore_in_the_same_process_writes_the_word_back(tmp_path: Path):
    """The loader restores an older .bak1 while this process's registry still holds the word: the file gets it back."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(content="a", memory_type="conversation", domain="us", emotions={"xi_feel": 8.0})
        )
        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
        load_persona_vocabulary_with_anomaly(vocab_path, store=store)
        assert vocabulary.get("xi_feel") is not None

        (tmp_path / "emotion_vocabulary.json.bak1").write_text(
            json.dumps({"version": 1, "emotions": []}), encoding="utf-8"
        )
        vocab_path.write_text("{corrupt{{", encoding="utf-8")

        _count, anomaly = load_persona_vocabulary_with_anomaly(vocab_path, store=store)  # same process: no _unregister

        assert anomaly is not None and "bak1" in anomaly.action
        names = {e["name"] for e in json.loads(vocab_path.read_text(encoding="utf-8"))["emotions"]}
        assert "xi_feel" in names
        marker = json.loads((tmp_path / "cadence" / "vocab_heal_final.json").read_text(encoding="utf-8"))
        assert marker["count"] == 1
    finally:
        store.close()


def test_missing_file_still_reconstructs_after_heal_marker(tmp_path: Path):
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="a", memory_type="conversation", domain="us", emotions={"theta_feel": 8.0}
            )
        )
        vocab_path = tmp_path / "emotion_vocabulary.json"
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
        load_persona_vocabulary_with_anomaly(vocab_path, store=store)
        assert (tmp_path / "cadence" / "vocab_heal_final.json").exists()

        vocab_path.unlink()
        _cleanup_emotion("theta_feel")
        load_persona_vocabulary_with_anomaly(vocab_path, store=store)

        assert vocabulary.get("theta_feel") is not None
        names = {e["name"] for e in json.loads(vocab_path.read_text(encoding="utf-8"))["emotions"]}
        assert "theta_feel" in names
    finally:
        store.close()
