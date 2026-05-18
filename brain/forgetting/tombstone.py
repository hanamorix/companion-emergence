"""tombstone.py — deterministic body → summary compression per spec §5.

No LLM call. Honest about its limits: "the first sentence, roughly".
LLM-generated summaries deferred (spec §7).
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RUN = re.compile(r"\s+")


def summarise(body: str, *, max_chars: int = 240) -> str:
    """Compress body to a summary suitable for the fading state.

    1. Collapse whitespace runs.
    2. If len <= max_chars, return as-is.
    3. Otherwise take the first sentence (split on `. `, `? `, `! `).
    4. If even that's too long, hard-truncate at a word boundary and
       append `"…"`.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", body).strip()
    if len(collapsed) <= max_chars:
        return collapsed

    first_sentence = _SENTENCE_SPLIT.split(collapsed, maxsplit=1)[0]
    if len(first_sentence) <= max_chars:
        return first_sentence

    # Word-boundary hard truncation. Reserve 1 char for the ellipsis.
    target = max_chars - 1
    truncated = collapsed[:target]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"
