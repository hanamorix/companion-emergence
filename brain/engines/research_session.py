"""Mechanical parse of a research session reply (NOTES/MEMORY/VERDICT markers).

Formatting lives here, never in the creative prompt.
Spec: docs/source-spec/2026-07-13-research-engine-redesign-design.md §5.3.

Marker rule. A BARE marker is ``NAME:`` alone on its line, name in any case; it always splits. An EXTENDED marker is a
marker with its text on the same line (``MEMORY: text``), or in bold or underscore emphasis (``**MEMORY:** text``,
``**MEMORY**: text``, ``__MEMORY:__``), name in any case because a colon follows it; or the name in bold or underscores
alone on its line with no colon (``**MEMORY**``), upper case only. An extended marker splits only when the reply has no
bare marker of that name, it sits in prompt order (NOTES, MEMORY, VERDICT) relative to every accepted marker, and it is the
last of its name; a NOTES marker must also be the first non-blank line, so no preamble is ever cut off. Everything else
(``MEMORY's``, ``VERDICT- x``, a marker word mid-line) is text. A reply with all three bare markers parses exactly as it
did before extended markers existed.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class SessionOutput:
    notes: str
    memory: str | None
    verdict: str
    spawn_topics: tuple[str, ...]
    degraded: bool
    # The sections of MEMORY and VERDICT that carry no text in the reply (both for a degraded reply). A reply with notes
    # and a non-empty ``missing`` is a layout the parser did not read; the caller records it.
    missing: tuple[str, ...] = ()


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
        accepted_start[name] = start
        markers.append((name, start, body_start))
    markers.sort(key=lambda marker: marker[1])
    return markers


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
