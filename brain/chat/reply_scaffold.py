"""#227 — strip scaffolding the model leaks onto the tail of its own reply.

Two shapes, both seen in production turn-diag logs:

* the literal ``</s>`` end-of-sequence token;
* the JSON wrapper history is replayed in (``{"speaker", "text", "ts"}``),
  completed by the model: ``", "ts": "2026-09-08T00:10:55-04:00"`` with or
  without the closing ``"}``.

Tail-only on purpose: a mid-reply mention of ``</s>`` is the model's own words,
not a leak. The ``ts`` pattern requires an ISO-8601 date so a JSON example in a
reply is left alone.
"""

from __future__ import annotations

import re

# ponytail: regex on the tail only; a reply that legitimately ENDS with
# `", "ts": "<ISO date>"` would be clipped. Upgrade: gate on history rendering.
_TAIL = re.compile(
    r"""(?:\s*</s>|\s*",\s*"ts"\s*:\s*"\d{4}-\d{2}-\d{2}T[^"\n]*"?\s*\}?)\s*$"""
)


def strip_scaffold_tail(text: str) -> str:
    """Remove leaked ``</s>`` / ``", "ts": "..."`` fragments from the end of ``text``."""
    while (m := _TAIL.search(text)) is not None:
        text = text[: m.start()]
    return text
