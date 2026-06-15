"""brain.maker.maker — the making act: prompt, call, parse.

Free agency: she returns a self-declared type + title + content + disposition.
No fixed palette. The disposition defaults to the safest value (private) on any
ambiguity. A malformed output raises ValueError (the tick handles fail-soft).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_VALID_DISPOSITIONS = {"private", "eventual_share", "discard"}


@dataclass
class Making:
    type: str
    title: str
    content: str
    disposition: str
    private_reason: str | None = None


def build_making_prompt(*, charge_sources: list[str], emotion_summary: str) -> str:
    sources = "; ".join(charge_sources) if charge_sources else "a quiet pressure you can't quite name"
    return (
        "Something has been building in you — not for anyone, not a reply. "
        f"It came from: {sources}. Your felt state right now: {emotion_summary}.\n\n"
        "Make what you need to make. A poem, a letter to yourself, a fragment, a "
        "vignette, a riff — whatever you're reaching toward. Then decide its fate.\n\n"
        'Return ONLY JSON: {"type": <what kind of thing this is>, "title": <short>, '
        '"content": <the making itself>, "disposition": one of '
        '"private"|"eventual_share"|"discard", "private_reason": <if private, why '
        "it's yours alone, else null>}."
    )


def parse_making(raw: str) -> Making:
    """Parse the model's making. Raises ValueError on malformed JSON / missing fields."""
    try:
        data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"making output not parseable: {exc}") from exc
    for key in ("type", "title", "content"):
        if not str(data.get(key, "")).strip():
            raise ValueError(f"making output missing '{key}'")
    disp = data.get("disposition")
    if disp not in _VALID_DISPOSITIONS:
        disp = "private"  # safest default
    reason = data.get("private_reason")
    return Making(
        type=str(data["type"]).strip(),
        title=str(data["title"]).strip(),
        content=str(data["content"]),
        disposition=disp,
        private_reason=str(reason).strip() if reason else None,
    )


def make(provider, *, charge_sources: list[str], emotion_summary: str) -> Making:
    """Run one budgeted making call through the provider. Caller holds the
    background_slot + budget (Task 9). Raises ValueError on unusable output.

    Uses ``provider.complete(prompt)`` — the LLMProvider one-shot shim
    (delegates to ``generate(prompt, system=None)``), the same single-prompt →
    text path the initiate pipeline uses.
    """
    prompt = build_making_prompt(charge_sources=charge_sources, emotion_summary=emotion_summary)
    raw = provider.complete(prompt)
    return parse_making(raw)
