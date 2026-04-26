"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` at engine startup and
registers each entry with the vocabulary registry. Idempotent on
re-register so multiple loaders in the same process don't fail.

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion
from brain.memory.store import MemoryStore

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

logger = logging.getLogger(__name__)

_DEFAULT_VOCAB: dict = {"version": 1, "emotions": []}


def _default_vocab_factory() -> dict:
    return {"version": 1, "emotions": []}


def _vocab_schema_validator(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        raise ValueError("emotion_vocabulary schema invalid: missing 'emotions' list")


def load_persona_vocabulary_with_anomaly(
    path: Path,
    *,
    store: MemoryStore | None = None,
) -> tuple[int, BrainAnomaly | None]:
    """Load persona vocabulary with self-healing from .bak rotation if corrupt.

    Returns (registered_count, anomaly_or_None).
      - Missing file → 0, no anomaly.
      - Corrupt file → quarantine, restore from .bak1/.bak2/.bak3 or reset to
        empty default. Anomaly set.
      - Healthy file → counts of newly registered emotions, no anomaly.
    """
    from brain.health.attempt_heal import attempt_heal

    if not path.exists():
        if store is not None:
            _warn_on_referenced_but_unregistered(store)
        return 0, None

    data, anomaly = attempt_heal(
        path, _default_vocab_factory, schema_validator=_vocab_schema_validator
    )

    # Reconstruct from memories when reset_to_default fires on vocabulary.
    # The default factory writes empty `{"version":1,"emotions":[]}` — that's
    # a truthful empty default but it loses the persona-extension entries
    # the brain has been operating with. If we have memory access, the brain
    # can re-learn its own vocabulary from how it has been using emotions.
    if anomaly is not None and anomaly.action == "reset_to_default" and store is not None:
        from brain.health.attempt_heal import save_with_backup
        from brain.health.reconstruct import reconstruct_vocabulary_from_memories

        recon_data = reconstruct_vocabulary_from_memories(store)
        save_with_backup(path, recon_data)
        data = recon_data
        # Replace the anomaly with one whose action reflects the reconstruction.
        # Same kind (json_parse_error / schema_mismatch — that's why we needed
        # to reconstruct) and same forensic quarantine path; the heal path
        # advanced beyond reset.
        from brain.health.anomaly import BrainAnomaly

        anomaly = BrainAnomaly(
            timestamp=anomaly.timestamp,
            file=anomaly.file,
            kind=anomaly.kind,
            action="reconstructed_from_memories",
            quarantine_path=anomaly.quarantine_path,
            likely_cause=anomaly.likely_cause,
            detail=(
                f"{anomaly.detail}; reconstructed "
                f"{len(recon_data['emotions'])} entries from memories"
            ),
        )

    if anomaly is not None:
        logger.warning(
            "emotion_vocabulary anomaly detected: %s action=%s file=%s",
            anomaly.kind,
            anomaly.action,
            anomaly.file,
        )

    registered = _register_from_data(data)

    if store is not None:
        _warn_on_referenced_but_unregistered(store)

    return registered, anomaly


def load_persona_vocabulary(
    path: Path,
    *,
    store: MemoryStore | None = None,
) -> int:
    """Load persona vocabulary from JSON file and register each entry.

    Returns the count of emotions newly registered. Re-registering an
    already-registered name is a silent no-op (idempotent), so calling
    this twice for the same persona returns 0 the second time.

    Missing `path` → returns 0 silently. Fresh personas don't need a file.
    Corrupt JSON → quarantine + heal from .bak, log a warning.
    Per-entry validation failure → that entry skipped + warning,
    other entries proceed.

    If `store` is provided, after registration the loader scans memories
    for emotion names not in the registry and logs a one-time warning
    per missing name pointing the user at `nell migrate --force`.
    """
    count, _anomaly = load_persona_vocabulary_with_anomaly(path, store=store)
    return count


def _register_from_data(data: dict) -> int:
    """Register emotions from a parsed vocab dict; return count of newly registered."""
    registered = 0
    for entry in data.get("emotions", []):
        try:
            emotion = _entry_to_emotion(entry)
        except (KeyError, ValueError, TypeError) as exc:
            entry_name = (
                entry.get("name", "<unnamed>") if isinstance(entry, dict) else "<not a dict>"
            )
            logger.warning("skipping emotion entry %r: %s", entry_name, exc)
            continue

        if vocabulary.get(emotion.name) is not None:
            # Already registered — idempotent skip.
            continue

        vocabulary.register(emotion)
        registered += 1
    return registered


def _entry_to_emotion(entry: dict) -> Emotion:
    """Build an Emotion from a JSON entry. Raises on missing/invalid fields."""
    if not isinstance(entry, dict):
        raise TypeError("entry must be a dict")
    required = ("name", "description", "category", "decay_half_life_days")
    for key in required:
        if key not in entry:
            raise KeyError(f"missing required field {key!r}")

    return Emotion(
        name=str(entry["name"]),
        description=str(entry["description"]),
        category=str(entry["category"]),  # type: ignore[arg-type]
        decay_half_life_days=(
            None if entry["decay_half_life_days"] is None else float(entry["decay_half_life_days"])
        ),
        intensity_clamp=float(entry.get("intensity_clamp", 10.0)),
    )


def _warn_on_referenced_but_unregistered(store: MemoryStore) -> None:
    """Scan all active memories for emotion names not in the registry.

    Logs one warning per unique missing name, pointing the user at the
    upgrade migration command. Used to detect the in-flight upgrade case
    where a pre-split persona is running on the post-split framework
    without re-migration yet.
    """
    seen_missing: set[str] = set()
    for mem in store.search_text("", active_only=True, limit=None):
        for name in mem.emotions:
            if name in seen_missing:
                continue
            if vocabulary.get(name) is None:
                seen_missing.add(name)
                logger.warning(
                    "persona memories reference emotion %r which is not in "
                    "this persona's vocabulary. Run `nell migrate --input "
                    "<og-source> --install-as <persona> --force` to upgrade.",
                    name,
                )
