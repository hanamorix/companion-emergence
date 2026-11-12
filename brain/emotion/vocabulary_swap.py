"""One-time swap of a persona's own vocabulary entries for the four feelings #648 registers.

`contentment`, `calm`, `peace` and `groundedness` became baseline feelings in #648. A persona that already holds an entry
under one of those names (a placeholder healed from its memories, a repaired description, or its own words with its own
half-life) gets that entry rewritten once to the baseline definition (category, description, half-life), so every persona
starts from the same fixed values even if self-calibration changes them later. Afterwards the persona's file wins for any
name, as it always has: a persona's own words keep their own half-lives.

The four names are this swap's own target list, not a standing rule. The general mechanism (reconcile a persona's vocabulary
file when the framework adds a feeling) is #741 and replaces this module.

The run is recorded in `vocabulary_swap_648.json` in the persona directory, written after every completed run (even when the
file held none of the four names, or did not exist) and never after a failed read or write. While the marker exists the swap
does nothing. A failure logs a warning and start-up goes on; the old file is untouched, so the swap tries again at the next
start.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.emotion import vocabulary
from brain.emotion.engine_files import vocabulary_path
from brain.health.json_io import Recovery, read_json, read_json_result, write_json
from brain.utils import file_ops
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)

# The swap's target list: the four feelings #648 registers (matched with `vocabulary.canonical_name`).
_SWAPPED_NAMES = ("contentment", "calm", "peace", "groundedness")
_MARKER_FILENAME = "vocabulary_swap_648.json"
# The vocabulary file keeps three backups, as every other writer of it does (`vocab_repair`, `persona_loader`).
_VOCABULARY_BACKUPS = 3


def run_vocabulary_swap_648(persona_dir: Path) -> None:
    """Swap the persona's entries for the four names to the baseline definition, once; call it right before `ensure_engine_ready`.

    Never raises: a read or write failure (an `OSError` or `ValueError`, which is what `JsonFileError` is both of) logs a warning,
    and so does a bug in this module, because a failed swap must not stop a persona from starting. No marker is written on a
    failure, so the swap tries again at the next start.
    """
    try:
        _run(Path(persona_dir))
    except (OSError, ValueError) as exc:
        logger.warning("vocabulary swap (#648) stopped, start-up goes on and it retries at the next start: %s", exc)
    except Exception as exc:  # noqa: BLE001 - a bug here must not stop start-up (the FTS health check at start-up does the same)
        logger.warning(
            "vocabulary swap (#648) stopped on %s: %s; start-up goes on and it retries at the next start",
            type(exc).__name__,
            exc,
            exc_info=True,
        )


def _run(persona_dir: Path) -> None:
    if not persona_dir.is_dir():
        return  # no persona folder: nothing to swap, and no folder or marker is created for one
    marker = persona_dir / _MARKER_FILENAME
    try:
        done = read_json(marker, default=lambda: None, recover=Recovery.RAISE, expect=dict)
    except (OSError, ValueError) as exc:
        # A marker that exists but cannot be read: the swap cannot know whether it already ran, and a second run would
        # overwrite words the persona wrote since, so nothing is swapped.
        logger.warning(
            "vocabulary swap (#648): the marker %s cannot be read (%s); the vocabulary file is left as it is", marker.name, exc
        )
        return
    if done is not None:
        return
    swapped = _swap_file(vocabulary_path(persona_dir))
    if swapped is None:
        return
    try:
        write_json(marker, {"version": 1, "swapped": swapped, "completed_at": iso_utc(datetime.now(UTC))})
    except (OSError, ValueError) as exc:
        logger.warning(
            "vocabulary swap (#648): the swap landed but the marker %s could not be written (%s); at the next start the swap "
            "runs again and finds the entries already on the baseline definition",
            marker.name,
            exc,
        )


def _swap_file(path: Path) -> int | None:
    """Rewrite the matching entries of the vocabulary file; the number rewritten, or None when the read or the write failed."""
    before = file_ops.stat_path(path)
    try:
        read = read_json_result(path, default=lambda: None, recover=Recovery.RAISE, expect=dict)
    except (OSError, ValueError) as exc:
        logger.warning("vocabulary swap (#648): the vocabulary file %s cannot be read (%s); it is left as it is", path.name, exc)
        return None
    if read.outcome == "missing":
        return 0  # a completed run: there is nothing to swap, and no vocabulary file is created
    data = read.data
    emotions = data.get("emotions")
    if not isinstance(emotions, list):
        logger.warning("vocabulary swap (#648): the vocabulary file %s has no 'emotions' list; it is left as it is", path.name)
        return None
    swapped = 0
    for entry in emotions:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        key = vocabulary.canonical_name(entry["name"])
        baseline = vocabulary.get(key) if key in _SWAPPED_NAMES else None
        if baseline is None:
            continue
        # Only the three fields of the definition change. The name keeps its spelling (the reflex crystallizer's trigger check
        # compares the file's names as written), and any other key of the entry stays.
        definition = {
            "category": baseline.category,
            "description": baseline.description,
            "decay_half_life_days": baseline.decay_half_life_days,
        }
        if all(key_ in entry and entry[key_] == value for key_, value in definition.items()):
            continue
        entry.update(definition)
        swapped += 1
    if swapped == 0:
        return 0
    # Another writer of this file (the vocabulary repair, the growth scheduler) may have saved since our read; writing our older
    # copy would lose its entries. Abort and retry at the next start. The window between this check and the replace stays
    # open and is accepted (P4): those writers run on a tick, and this runs once, before the supervisor's first one.
    after = file_ops.stat_path(path)
    if (before.key if before is not None else None) != (after.key if after is not None else None):
        logger.warning(
            "vocabulary swap (#648): the vocabulary file %s changed while the swap was working; nothing was written, retrying "
            "at the next start",
            path.name,
        )
        return None
    if not write_json(path, data, backups=_VOCABULARY_BACKUPS, verify=True):
        logger.warning(
            "vocabulary swap (#648): the new vocabulary file %s did not verify; the old content was restored from its backup",
            path.name,
        )
        return None
    return swapped
