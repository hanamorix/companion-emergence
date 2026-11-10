"""Mechanical parse of a research session reply (NOTES/MEMORY/VERDICT markers).

Formatting lives here, never in the creative prompt.
Spec: docs/source-spec/2026-07-13-research-engine-redesign-design.md §5.3.

Marker rule. A BARE marker is ``NAME:`` alone on its line, name in any case; it always splits. An EXTENDED marker is a
marker with its text on the same line (``MEMORY: text``), or in bold or underscore emphasis (``**MEMORY:** text``,
``**MEMORY**: text``, ``__MEMORY:__``), name in any case because a colon follows it; or the name in bold or underscores
alone on its line with no colon (``**MEMORY**``), upper case only. An extended marker splits only when the reply has no
bare marker of that name, it sits in prompt order (NOTES, MEMORY, VERDICT) relative to every accepted marker, and it is the
last of its name; a NOTES marker must also be the first non-blank line, so no preamble is ever cut off, and a VERDICT
marker must be the last content of the reply or carry a verdict the prompt offers (close, continue, spawn:), so a prose
line with text after it cannot end a section and lose that text. Everything else
(``MEMORY's``, ``VERDICT- x``, a marker word mid-line) is text. A reply with all three bare markers parses exactly as it
did before extended markers existed.
"""
from __future__ import annotations

import json
import math
import re
from bisect import bisect_left
from dataclasses import dataclass

_BARE_MARKER = re.compile(r"^(NOTES|MEMORY|VERDICT):\s*$", re.IGNORECASE | re.MULTILINE)
# Extended markers use ASCII case folding: Unicode folding would let a name such as "VERDİCT" match.
_EXT_COLON = re.compile(
    r"^(?:(?P<plain>NOTES|MEMORY|VERDICT):[ \t]*(?=\S)"
    r"|(?:\*\*|__)(?P<bold>NOTES|MEMORY|VERDICT)(?::(?:\*\*|__)|(?:\*\*|__):)[ \t]*)(?P<rest>[^\n]*)$",
    re.IGNORECASE | re.MULTILINE | re.ASCII,
)
_EXT_UPPER = re.compile(
    r"^(?:\*\*|__)(?P<name>NOTES|MEMORY|VERDICT)(?:\*\*|__)[ \t\r]*$", re.MULTILINE | re.ASCII
)
_ORDER = ("NOTES", "MEMORY", "VERDICT")
_SPAWN_CAP = 2
# A felt line (emotion rework B1): a name, optionally in bold or underscores, with the colon inside or outside them.
_FELT_LINE = re.compile(
    r"^(?:\*\*|__)?(?P<name>EMOTIONS|ENGAGEMENT)(?:\*\*|__)?:(?:\*\*|__)?[ \t]*(?P<value>.*)$",
    re.IGNORECASE | re.ASCII,
)
_LEADING_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
_FELT_MAX = 10.0


@dataclass(frozen=True)
class SessionOutput:
    notes: str
    memory: str | None
    verdict: str
    spawn_topics: tuple[str, ...]
    degraded: bool
    # The sections of MEMORY and VERDICT that carry no text in the reply (both for a degraded reply). A reply with notes
    # and a non-empty ``missing`` is a layout the parser did not read; the caller records it.
    missing: tuple[str, ...]


def _is_taken_verdict(raw: str, body_start: int) -> bool:
    """True when an inline VERDICT marker whose text starts at ``body_start`` may end its section without losing text.

    A VERDICT section keeps only its first line, so what follows that line is dropped. The marker is therefore taken when
    nothing but white space follows its value line (the last content of the reply, whatever the value says), or when the
    value is a verdict the prompt offers (close, continue, spawn:). A prose "Verdict: ..." line with text after it is
    neither, stays text, and loses nothing. The value line is the one ``parse_session_output`` keeps (same ``splitlines``).
    """
    lines = raw[body_start:].strip().splitlines()
    if len(lines) <= 1:
        return True
    value = lines[0].strip().lower()
    return value in ("close", "continue") or value.startswith("spawn:")


def _find_markers(raw: str) -> list[tuple[str, int, int]]:
    """The accepted markers as (NAME, start, body_start), sorted by position. O(bare + extended)."""
    markers = [(m.group(1).upper(), m.start(), m.end()) for m in _BARE_MARKER.finditer(raw)]
    bare_names = {name for name, _, _ in markers}

    extended: dict[str, list[tuple[int, int]]] = {name: [] for name in _ORDER}
    for m in _EXT_COLON.finditer(raw):
        extended[(m.group("plain") or m.group("bold")).upper()].append((m.start(), m.start("rest")))
    for m in _EXT_UPPER.finditer(raw):
        extended[m.group("name")].append((m.start(), m.end()))

    # Positions of the bare markers per rank (a bare marker whose name is none of the three, reachable through Unicode
    # folding of the bare pattern, stays a section boundary and takes no part in the ordering, as before).
    bare_start = {name: [s for n, s, _ in markers if n == name] for name in _ORDER}
    accepted_start: dict[str, int] = {}
    for rank, name in enumerate(_ORDER):
        if name in bare_names or not extended[name]:
            continue
        candidates = sorted(extended[name])
        if name == "NOTES":
            # Only the first non-blank line of the reply: a NOTES marker further down would cut the preamble off.
            candidates = candidates[:1] if not raw[: candidates[0][0]].strip() else []
            if not candidates:
                continue
        earlier = [p for n in _ORDER[:rank] for p in (*bare_start[n], *([accepted_start[n]] if n in accepted_start else []))]
        later = [p for n in _ORDER[rank + 1:] for p in bare_start[n]]
        lo = max(earlier, default=-1)
        hi = min(later, default=len(raw) + 1)
        index = bisect_left([start for start, _ in candidates], hi) - 1
        if index < 0 or candidates[index][0] <= lo:
            continue
        start, body_start = candidates[index]
        # Only the last candidate counts, for every name: when it is rejected the earlier ones are prose too, and falling
        # back to one would apply a verdict (a `close`) that the reply's own last verdict line did not give.
        if name == "VERDICT" and not _is_taken_verdict(raw, body_start):
            continue
        accepted_start[name] = start
        markers.append((name, start, body_start))
    markers.sort(key=lambda marker: marker[1])
    return markers


def extract_felt_lines(raw: str) -> tuple[str, dict[str, float] | None, float | None]:
    """Split the EMOTIONS and ENGAGEMENT lines off the end of a research reply: ``(rest, emotions, rating)``.

    Only one-line markers at the very end count: the walk goes back from the last line, skips blank lines, and stops at
    the first line that isn't a felt marker, so a marker word inside the notes is never taken. The nearest line of each
    name wins. ``rest`` is the text before the earliest felt line, or ``raw`` unchanged when there is none.

    ``None`` means missing or broken. For emotions, ``{}`` means a valid object whose values were all dropped: only
    registered names with finite values above 0 are kept, held to 10. The rating is the number at the start of the
    value (``7/10`` gives 7), held to 10; a negative is treated as missing.
    """
    lines = raw.splitlines(keepends=True)
    values: dict[str, str] = {}
    earliest = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        text = lines[index].strip()
        if not text:
            continue
        match = _FELT_LINE.match(text)
        if not match:
            break
        values.setdefault(match.group("name").upper(), match.group("value").strip())
        earliest = index
    if earliest == len(lines):
        return raw, None, None
    rest = "".join(lines[:earliest])
    emotions = _felt_emotions(values["EMOTIONS"]) if "EMOTIONS" in values else None
    rating = _felt_rating(values["ENGAGEMENT"]) if "ENGAGEMENT" in values else None
    return rest, emotions, rating


def _felt_emotions(value: str) -> dict[str, float] | None:
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    kept = {}
    for name, number in parsed.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            continue
        if math.isfinite(number) and number > 0:
            kept[str(name)] = min(float(number), _FELT_MAX)
    from brain.chat.extractor import _filter_to_registered

    return _filter_to_registered(kept)


def _felt_rating(value: str) -> float | None:
    match = _LEADING_NUMBER.match(value)
    if not match:
        return None
    number = float(match.group())
    if not math.isfinite(number) or number < 0:
        return None
    return min(number, _FELT_MAX)


def parse_session_output(raw: str) -> SessionOutput:
    sections: dict[str, str] = {}
    markers = _find_markers(raw)
    for n, (name, _, body_start) in enumerate(markers):
        end = markers[n + 1][1] if n + 1 < len(markers) else len(raw)
        sections[name] = raw[body_start:end].strip()

    if "NOTES" not in sections:
        return SessionOutput(notes=raw.strip(), memory=None, verdict="continue",
                             spawn_topics=(), degraded=True, missing=("MEMORY", "VERDICT"))

    memory = sections.get("MEMORY", "").strip() or None
    verdict_line = sections.get("VERDICT", "").strip().splitlines()
    verdict_raw = verdict_line[0].strip().lower() if verdict_line else ""

    spawn: tuple[str, ...] = ()
    verdict = "continue"
    if verdict_raw == "close":
        verdict = "close"
    elif verdict_raw.startswith("spawn:"):
        topics = [t.strip() for t in verdict_raw[len("spawn:"):].split(";")]
        spawn = tuple(t for t in topics if t)[:_SPAWN_CAP]

    missing = tuple(
        name for name, present in (("MEMORY", memory is not None), ("VERDICT", bool(verdict_line))) if not present
    )
    return SessionOutput(notes=sections["NOTES"], memory=memory, verdict=verdict,
                         spawn_topics=spawn, degraded=False, missing=missing)
