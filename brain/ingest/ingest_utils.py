"""Ingest-area helpers shared by the extraction pipeline and the extraction backfill (#523).

``chunk_turns`` splits a run of buffer rows into extraction chunks no larger than ``extraction_chunk_chars()``
characters of formatted transcript, cutting only where the extraction cursor can safely stop; it and ``parse_turn_ts``
are pure (no file, no lock, no model call). The ``load_*`` helpers read what an extraction call needs from the persona
(the user's name and pronouns from ``persona_config.json``, the registered emotion names).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain import tunables
from brain.ingest.extract import format_turn_line
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

# The most characters of formatted transcript that one extraction call is given; a longer span is extracted in several
# calls, oldest first. 24,000 is the per-call input bound every extraction has used since fb55643d (2026-04-26), where it
# was written as a token budget times a 4-chars-per-token guess. Keeping the value keeps each call's prompt size and
# per-call behaviour as they were, so chunking changes only how much of a span reaches the extractor. An ops bound on one
# background call's input (like ``files.read_max_bytes``), not physiology. Measured in characters, the unit it bounds.
_CHUNK_CHARS_DEFAULT: int = tunables.register("ingest.extraction_chunk_chars", 24_000)


def extraction_chunk_chars() -> int:
    """The chunk size limit: the ``ingest.extraction_chunk_chars`` override when it is a positive int, else the default.

    Any other override (0, a negative number, a string, ``true``/``false``; Python's ``bool`` is an ``int``, so it is
    rejected by type) is ignored with one warning. No other floor or ceiling: a small or large positive value is the
    operator's choice.
    """
    value = tunables.get_tunable("ingest.extraction_chunk_chars", _CHUNK_CHARS_DEFAULT)
    if type(value) is int and value > 0:
        return value
    warn_throttled(
        logger,
        "ingest.extraction_chunk_chars",
        "tunables: override 'ingest.extraction_chunk_chars'=%r is not a positive integer; using %d",
        value,
        _CHUNK_CHARS_DEFAULT,
        exc_info=False,
    )
    return _CHUNK_CHARS_DEFAULT


def parse_turn_ts(raw: object) -> datetime | None:
    """A buffer row's ``ts`` as an aware datetime (naive = UTC), or None when it is missing or does not parse."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@dataclass(frozen=True)
class TurnChunk:
    """One extraction chunk.

    ``turns``: the chunk's rows, in buffer order. ``text``: their formatted transcript, byte-identical to
    ``format_transcript(turns, user_name=..., assistant_name=...)``. ``cursor_ts``: the value the session cursor may take
    once this chunk's commits succeed (the raw ``ts`` of the newest row up to the chunk's end; None when none of those rows
    has a parseable ts). ``oversize``: the chunk is a single unit longer than the limit, sent whole.
    """

    turns: tuple[dict, ...]
    text: str
    cursor_ts: str | None
    oversize: bool


def _cut_allowed(speaker: object, ts: datetime | None, prefix_max: datetime | None, suffix_min: datetime | None) -> bool:
    """Whether a chunk may end just before a row (the row at the cut, its ts, the max ts of every row before it and the
    min ts of it and every row after it).

    Exchange rule: only before a ``user`` row, so a question and its reply rows stay together. Timestamp rule: every row
    before the cut is strictly older than every row from the cut on, because the next pass reads ``ts > cursor``; rows
    that share a ts are therefore never split. A row with no parseable ts is never a cut point (it joins the row before)."""
    if speaker != "user" or ts is None or prefix_max is None or suffix_min is None:
        return False
    return prefix_max < suffix_min


def _units(rows: Sequence[dict], stamps: Sequence[datetime | None]) -> list[list[int]]:
    """Group row indexes into units: the rows between two allowed cut points."""
    suffix_min: list[datetime | None] = [None] * len(rows)
    running: datetime | None = None
    for i in range(len(rows) - 1, -1, -1):
        if stamps[i] is not None and (running is None or stamps[i] < running):
            running = stamps[i]
        suffix_min[i] = running
    units: list[list[int]] = []
    prefix_max: datetime | None = None
    for i, row in enumerate(rows):
        if i > 0 and _cut_allowed(row.get("speaker"), stamps[i], prefix_max, suffix_min[i]):
            units.append([i])
        elif units:
            units[-1].append(i)
        else:
            units.append([i])
        if stamps[i] is not None and (prefix_max is None or stamps[i] > prefix_max):
            prefix_max = stamps[i]
    return units


def chunk_turns(
    turns: Iterable[dict],
    *,
    max_chars: int,
    user_name: str | None,
    assistant_name: str | None,
) -> list[TurnChunk]:
    """Split ``turns`` into extraction chunks, oldest first, never cutting inside a row.

    Rows between two allowed cut points (see ``_cut_allowed``) form a unit; a chunk is a run of whole units whose text
    stays within ``max_chars``. A single unit longer than ``max_chars`` is its own chunk, sent whole (``oversize``).
    Compaction ``summary`` rows are dropped first: they are never extracted and their ts must never move the cursor.
    Rows with a missing or unparseable ts are kept (they join the unit before them) and logged as
    ``conversation_extract_ts_missing``.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars!r}")
    rows = [t for t in turns if t.get("speaker") != "summary"]
    if not rows:
        return []
    user_label = user_name if user_name else "user"
    assistant_label = assistant_name if assistant_name else "assistant"
    lines = [format_turn_line(t, user_label, assistant_label) or "" for t in rows]
    stamps = [parse_turn_ts(t.get("ts")) for t in rows]
    missing = sum(1 for s in stamps if s is None)
    if missing:
        logger.warning("conversation_extract_ts_missing rows=%d: kept with the row before them, never a chunk boundary", missing)

    def _length(indexes: list[int]) -> int:
        return sum(len(lines[j]) for j in indexes) + len(indexes) - 1

    groups: list[list[int]] = []
    current: list[int] = []
    for unit in _units(rows, stamps):
        if current and _length(current) + 1 + _length(unit) > max_chars:
            groups.append(current)
            current = []
        current = current + unit
    if current:
        groups.append(current)

    chunks: list[TurnChunk] = []
    newest: datetime | None = None
    newest_raw: str | None = None
    for group in groups:
        for j in group:
            if stamps[j] is not None and (newest is None or stamps[j] > newest):
                newest, newest_raw = stamps[j], str(rows[j].get("ts"))
        text = "\n".join(lines[j] for j in group)
        chunks.append(
            TurnChunk(
                turns=tuple(rows[j] for j in group),
                text=text,
                cursor_ts=newest_raw,
                oversize=len(text) > max_chars,
            )
        )
    return chunks


def load_user_name(persona_dir: Path) -> str | None:
    """Look up PersonaConfig.user_name; None if missing or unset.

    Best-effort — never raises. PersonaConfig is the home for the
    user's name; the pipeline reads it once per extraction pass. If the
    config file is corrupt or the field is missing, the extractor
    falls back to the legacy "user:" / "assistant:" prompt path.
    """
    try:
        from brain.persona_config import PersonaConfig

        return PersonaConfig.load(persona_dir / "persona_config.json").user_name
    except Exception:  # noqa: BLE001
        return None


def load_user_pronouns(persona_dir: Path) -> dict | None:
    """PersonaConfig.user_pronouns; None if missing/unset. Best-effort."""
    try:
        from brain.persona_config import PersonaConfig

        return PersonaConfig.load(persona_dir / "persona_config.json").user_pronouns
    except Exception:  # noqa: BLE001
        return None


def load_emotion_vocab() -> tuple[set[str], str]:
    """Return (names_set, comma_separated_sorted_string) for all registered emotions.

    Reads the in-process vocabulary registry (baseline + any persona extensions
    registered at engine startup). Falls back to empty on any error — an
    extraction must never fail because of a vocabulary load issue.

    Returns a tuple so callers get both forms cheaply (set for normalize(),
    string for the prompt).
    """
    try:
        from brain.emotion.vocabulary import list_all

        names = {e.name for e in list_all()}
        return names, ",".join(sorted(names))
    except Exception:  # noqa: BLE001
        logger.warning("load_emotion_vocab: failed to load vocabulary; emotions will be empty")
        return set(), ""
