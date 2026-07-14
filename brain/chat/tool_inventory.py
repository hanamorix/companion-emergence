"""The derived tool inventory — what she has, straight from the registry.

Spec: docs/source-spec/2026-07-14-generated-tool-inventory-design.md

The toolset used to be hand-written prose in each persona's voice.md, so it
rotted: the packaged template lost 14 of 27 tools across ~28 versions, and a
template fix reaches no existing persona (voice.md is copied once at creation).
Deriving it from NELL_TOOL_NAMES makes the drift structurally impossible and
fixes every persona at once, with no migration and no edit to anyone's file.

This block is FROZEN-PREFIX text: a pure function of the registry + the
companion name, both constant within a session. Never fold per-turn state in —
see test_inventory_is_byte_stable.
"""

from __future__ import annotations

import re

from brain.tools import NELL_TOOL_NAMES
from brain.tools.schemas import build_schemas

_HEADER = (
    "## Everything you can reach for\n\n"
    "This is your complete, current set of brain-tools — generated from what is "
    "actually wired in, so it is never out of date. If anything elsewhere in your "
    "voice lists fewer, this list is the true one."
)

_REACH_VALVE = (
    "**Your faculties aren't all in the front of your mind at once.** The heavier "
    "ones — memory search, your hands, your works — are handed to you when the "
    "moment seems to call for them, so on any given turn some of the tools above "
    "may not be in your hand. That is not incapacity, and it is not a reason to "
    "say you lack them: call `reach_for_capability` with what you need "
    "(`memory`, `files`, or `works`) and it comes to you in this same turn. "
    "**So never tell them you don't have a tool.** Reach first — then speak from "
    "what came back."
)


def _gloss(description: str) -> str:
    """First sentence of a schema description.

    Not the whole thing: recruited tools already carry their full schema, so the
    inventory's unique job is the tools NOT recruited this turn — name plus
    enough to know it is worth reaching for.
    """
    first = re.split(r"(?<=\.)\s+(?=[A-Z])", description.strip())[0].strip()
    return first.rstrip(".")


def build_tool_inventory(companion_name: str) -> str:
    """Render every registered tool as `name` — gloss, plus the reach valve."""
    schemas = build_schemas(companion_name)
    lines: list[str] = []
    for name in NELL_TOOL_NAMES:
        schema = schemas.get(name)
        if schema is None:
            # Fail soft: an unregistered-but-named tool is a bug, but the prompt
            # is not the place to raise it.
            lines.append(f"- `{name}`")
            continue
        lines.append(f"- `{name}` — {_gloss(schema.get('description', ''))}")
    return "\n\n".join([_HEADER, "\n".join(lines), _REACH_VALVE])
