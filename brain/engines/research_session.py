"""Mechanical parse of a research session reply (NOTES/MEMORY/VERDICT markers).

Formatting lives here, never in the creative prompt.
Spec: docs/source-spec/2026-07-13-research-engine-redesign-design.md §5.3.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MARKER = re.compile(r"^(NOTES|MEMORY|VERDICT):\s*$", re.IGNORECASE | re.MULTILINE)
_SPAWN_CAP = 2


@dataclass(frozen=True)
class SessionOutput:
    notes: str
    memory: str | None
    verdict: str
    spawn_topics: tuple[str, ...]
    degraded: bool


def parse_session_output(raw: str) -> SessionOutput:
    sections: dict[str, str] = {}
    matches = list(_MARKER.finditer(raw))
    for n, m in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(raw)
        sections[m.group(1).upper()] = raw[m.end():end].strip()

    if "NOTES" not in sections:
        return SessionOutput(notes=raw.strip(), memory=None, verdict="continue",
                             spawn_topics=(), degraded=True)

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

    return SessionOutput(notes=sections["NOTES"], memory=memory, verdict=verdict,
                         spawn_topics=spawn, degraded=False)
