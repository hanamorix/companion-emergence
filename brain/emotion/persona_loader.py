"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` at engine startup and
registers each entry with the vocabulary registry. Idempotent on
re-register so multiple loaders in the same process don't fail.

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


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
    Corrupt JSON → returns 0, logs a warning.
    Per-entry validation failure → that entry skipped + warning,
    other entries proceed.

    If `store` is provided, after registration the loader scans memories
    for emotion names not in the registry and logs a one-time warning
    per missing name pointing the user at `nell migrate --force`.
    """
    if not path.exists():
        if store is not None:
            _warn_on_referenced_but_unregistered(store)
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("emotion_vocabulary at %s could not be parsed: %.200s", path, exc)
        return 0

    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        logger.warning(
            "emotion_vocabulary at %s has invalid schema (missing 'emotions' list)",
            path,
        )
        return 0

    registered = 0
    for entry in data["emotions"]:
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

    if store is not None:
        _warn_on_referenced_but_unregistered(store)

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
