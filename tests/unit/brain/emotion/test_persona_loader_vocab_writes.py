"""How the vocabulary loader writes `emotion_vocabulary.json` (#784 follow-ups).

Every write the loader makes to the file (the heal, the rebuild of a missing file, the rebuild after a reset) happens under
`file_lock(vocab_path)`, the lock `register_new_word` takes, on data read inside that lock, and with the file's own backup
count. The heal writes only entries the loader can read back.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.emotion import persona_loader, vocabulary
from brain.emotion.persona_loader import load_persona_vocabulary
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock


def _store_with(*names: str) -> MemoryStore:
    store = MemoryStore(":memory:")
    for name in names:
        store.create(
            Memory.create_new(content=f"felt {name}", memory_type="conversation", domain="us", emotions={name: 8.0})
        )
    return store


def _lock_is_held(path: Path) -> bool:
    with file_lock(path, blocking=False) as acquired:
        return not acquired


def _entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["emotions"]


@pytest.mark.parametrize("scenario", ["heal", "missing_file_rebuild", "reset_rebuild"])
def test_every_loader_write_holds_the_vocabulary_lock_and_uses_the_files_backups(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocab_path = tmp_path / "emotion_vocabulary.json"
    if scenario == "heal":
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    elif scenario == "reset_rebuild":
        vocab_path.write_text("{not json", encoding="utf-8")
    vocabulary._unregister("omicron_feel")
    monkeypatch.setattr(persona_loader, "compute_treatment", lambda *_a, **_k: SimpleNamespace(backup_count=7))
    writes: list[tuple[bool, int | None]] = []
    real_write = persona_loader.write_json

    def spy(path, data, **kwargs):  # noqa: ANN001, ANN202
        if Path(path) == vocab_path:
            writes.append((_lock_is_held(vocab_path), kwargs.get("backups")))
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(persona_loader, "write_json", spy)
    store = _store_with("omicron_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert writes and all(write == (True, 7) for write in writes)
    assert "omicron_feel" in {e["name"] for e in _entries(vocab_path)}


def test_a_malformed_entry_does_not_hide_its_word_from_the_heal(tmp_path: Path) -> None:
    """`{"name": x}` can't be registered; the heal replaces it with a whole entry rather than counting it as present."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [{"name": "pi_feel"}]}), encoding="utf-8")
    vocabulary._unregister("pi_feel")
    store = _store_with("pi_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert vocabulary.get("pi_feel") is not None
    pi_entries = [e for e in _entries(vocab_path) if e["name"] == "pi_feel"]
    assert len(pi_entries) == 1 and "description" in pi_entries[0]


def test_a_registered_word_with_a_non_finite_half_life_heals_with_finite_values(tmp_path: Path) -> None:
    """JSON has no NaN: writing the registered fields back as they are would raise on every store load."""
    import math

    from brain.emotion.vocabulary import Emotion

    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    vocabulary._unregister("rho_feel")
    vocabulary.register(Emotion("rho_feel", "d", "persona_extension", float("nan"), float("inf")))
    store = _store_with("rho_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    (rho,) = [e for e in _entries(vocab_path) if e["name"] == "rho_feel"]
    assert math.isfinite(rho["decay_half_life_days"]) and math.isfinite(rho["intensity_clamp"])


def test_a_file_entry_with_a_non_finite_number_is_not_registered_and_is_healed(tmp_path: Path) -> None:
    """Python's JSON reads NaN; a NaN half-life in the registry poisons decay and every later write of the file."""
    import math

    vocab_path = tmp_path / "emotion_vocabulary.json"
    bad = {"name": "sigma_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": "NaN"}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [bad]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("sigma_feel")
    store = _store_with("sigma_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert math.isfinite(vocabulary.get("sigma_feel").decay_half_life_days)
    (sigma,) = [e for e in _entries(vocab_path) if e["name"] == "sigma_feel"]
    assert math.isfinite(sigma["decay_half_life_days"])


def test_register_new_word_refuses_softly_when_the_file_holds_a_non_finite_number(tmp_path: Path) -> None:
    """The file can't be written back as JSON, so the word is refused (None) and the file left as it is, not raised."""
    from brain.emotion.persona_loader import register_new_word

    vocab_path = tmp_path / "emotion_vocabulary.json"
    bad = {"name": "upsilon_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": "NaN"}
    text = json.dumps({"version": 1, "emotions": [bad]}).replace('"NaN"', "NaN")
    vocab_path.write_text(text, encoding="utf-8")
    vocabulary._unregister("tau_feel")

    assert register_new_word(vocab_path, "tau_feel") is None
    assert vocab_path.read_text(encoding="utf-8") == text
    assert vocabulary.get("tau_feel") is None
