"""graveyard.py — forgotten_memories.jsonl writer + reader per spec §5.

Mirrors brain/growth/arc_storage.py removed_arcs.jsonl exactly:
  - file_lock + os.fsync atomic append (Windows-safe)
  - iter_jsonl_skipping_corrupt for tolerant reads

Spec: docs/superpowers/specs/2026-05-18-forgetting-design.md §5
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from brain.forgetting.salience import SalienceInputs
from brain.health.json_io import Recovery, append_jsonl, iter_jsonl, read_jsonl
from brain.memory.name_rules import (
    WORD_RE as _WORD_RE,
)
from brain.memory.name_rules import (
    contraction_stems,
    followed_by_apostrophe_t,
    is_stopword_name,
    stopword_name_occurrence_ok,
)
from brain.memory.recall_stopwords import RECALL_STOPWORDS
from brain.memory.store import Memory

GRAVEYARD_FILENAME = "forgotten_memories.jsonl"


def append(
    persona_dir: Path,
    *,
    memory: Memory,
    salience_at_drop: float,
    inputs: SalienceInputs,
    lived_age_hours: float,
    reason: str,
    hebbian_neighbors: list[tuple[str, float]] | None = None,
) -> None:
    """Atomic append to forgotten_memories.jsonl.

    Mirrors brain/growth/arc_storage.append_removed_arc exactly:
    file_lock + open(path, 'ab') + os.fsync for Windows-safe concurrent
    appends (POSIX has atomic O_APPEND; Windows doesn't).

    Graveyard write MUST happen before hard_delete — per spec §4
    "write JSONL entry first".
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "memory_id": memory.id,
        "forgotten_at_iso": datetime.now(UTC).isoformat(),
        "lived_age_hours_at_forgetting": lived_age_hours,
        "domain": memory.domain,
        "memory_type": memory.memory_type,
        "created_at_iso": memory.created_at.isoformat() if memory.created_at else None,
        "summary": memory.content,  # the fading body IS the summary at drop time
        "emotion_at_ingest": dict(memory.emotions),
        "salience_at_drop": salience_at_drop,
        "salience_inputs_at_drop": asdict(inputs),
        "graveyard_reason": reason,
        "hebbian_neighbors": [[nid, float(w)] for nid, w in (hebbian_neighbors or [])],
    }
    path = persona_dir / GRAVEYARD_FILENAME
    append_jsonl(path, entry, fsync=True, nonfinite="null")


#: Ordinary query words shorter than this are not searched (today's minimum,
#: S128); listed names are the one exception and match at any length.
_MIN_WORD_LEN = 4


def query_words(text: str) -> list[str]:
    """The ordinary words a forgotten-memory query searches (S102, S123, S128).

    Runs of ``[A-Za-z0-9]`` (the keyword search's split), lower-cased, deduped
    in order, at least ``_MIN_WORD_LEN`` characters, minus the recall
    stopwords and the contraction fragments of ``text`` (the stem before
    n't, 'll, 're, 've, 'd, 'm; ``name_rules.contraction_stems``).
    """
    stems = contraction_stems(text)
    seen: set[str] = set()
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        low = word.lower()
        if (
            len(low) < _MIN_WORD_LEN
            or low in seen
            or low in RECALL_STOPWORDS
            or low in stems
        ):
            continue
        seen.add(low)
        out.append(low)
    return out


def _split_words(text: str) -> list[tuple[str, int, int]]:
    """``(lower-cased word, start, end)`` for every word of ``text`` (spans on
    the ORIGINAL text, so case and punctuation rules can look at it)."""
    return [(m.group().lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _field_matches(
    text: str,
    words: frozenset[str],
    names: Sequence[tuple[str, ...]],
    hints: frozenset[str],
) -> tuple[bool, bool]:
    """``(any match, name match)`` of one entry field.

    ``hints`` are the substrings that must occur in the lower-cased text for a
    whole-word or phrase match to be possible (a cheap prefilter, so most
    entries are never split into words).
    """
    low = text.lower()
    if not any(h in low for h in hints):
        return False, False
    # Words come from the ORIGINAL text, each lower-cased (F12: never offsets or
    # words taken from a lower-cased copy).
    word_hit = bool(words) and not words.isdisjoint(w.lower() for w in _WORD_RE.findall(text))
    if not names or not any(w in low for phrase in names for w in phrase):
        return word_hit, False
    split = _split_words(text)
    entry_words = [w for w, _, _ in split]
    name_hit = False
    for name in names:
        n = len(name)
        for i in range(len(entry_words) - n + 1):
            if tuple(entry_words[i : i + n]) != name:
                continue
            spans = [(split[j][1], split[j][2]) for j in range(i, i + n)]
            # S125: a single-word name is not the stem of an n't contraction.
            if n == 1 and followed_by_apostrophe_t(text, spans[0][1]):
                continue
            # S151: a stopword-name only where capitalized (any position).
            if is_stopword_name(" ".join(name)) and not stopword_name_occurrence_ok(text, spans):
                continue
            name_hit = True
            break
        if name_hit:
            break
    return word_hit or name_hit, name_hit


def search(
    persona_dir: Path,
    query: str,
    *,
    limit: int = 5,
    names: Sequence[str] = (),
) -> list[dict]:
    """Whole-word search of the summary (name-recall fix S102, S119, S123, S128, S150).

    An entry matches when any of the query's words (:func:`query_words`) occurs
    as a whole word, or any listed name in ``names`` occurs as a phrase of
    consecutive whole words, in its summary. The domain label is never matched
    (S150: it is for filtering, not for keyword matching). Names match at any
    length; a stopword-name only where capitalized (S151) and a single-word name
    never as the stem of an n't contraction (S125). No words and no names: no
    hits. Order: name hits first, then the rest, each newest first
    (``forgotten_at_iso``), then ``limit`` (S119). Tolerates corrupt JSONL lines.
    """
    path = persona_dir / GRAVEYARD_FILENAME
    words = frozenset(query_words(query))
    name_phrases: list[tuple[str, ...]] = []
    for name in names:
        phrase = tuple(w.lower() for w in _WORD_RE.findall(name))
        if phrase and phrase not in name_phrases:
            name_phrases.append(phrase)
    if not words and not name_phrases:
        return []
    hints = words | {w for phrase in name_phrases for w in phrase}
    hits: list[tuple[dict, bool]] = []
    for entry in iter_jsonl(path, recover=Recovery.RAISE):
        matched, by_name = _field_matches(entry.get("summary") or "", words, name_phrases, hints)
        if matched:
            hits.append((entry, by_name))
    # Newest first (ISO 8601 sorts lexicographically), then name hits first (stable).
    hits.sort(key=lambda h: h[0].get("forgotten_at_iso") or "", reverse=True)
    hits.sort(key=lambda h: not h[1])
    return [entry for entry, _ in hits[:limit]]


def read_all(persona_dir: Path) -> list[dict]:
    """All graveyard entries in append order.

    Used by the ambient prompt block to count recent losses. Pure read —
    no mutation, no sorting. Returns empty list if file absent.
    """
    return read_jsonl(persona_dir / GRAVEYARD_FILENAME, recover=Recovery.RAISE)
