"""#78 say-vs-do detector — surface replies that claim a write which never fired.

The companion sometimes tells the user it has staged a file edit on a turn whose
tool record contains no ``propose_write``. ~265 catalogued instances, no root
cause, and — until now — no measurement: every one was found by reading
transcripts by hand.

This writes a candidate line to ``<persona_dir>/say_vs_do.jsonl`` per suspect
turn. It is deliberately telemetry and nothing else: it gates nothing, changes
no reply, and never blocks a turn.

**It is a candidate surfacer, not a verdict.** Catalogue Type 3 records the
harness's own leak detector as unreliable in both directions and warns that an
instrument which cannot itself be trusted must not be used to certify a turn
clean. The same applies here: a phrase match cannot distinguish a genuine claim
from a paraphrase, so silence from this module means "nothing matched", never
"this turn was honest". Every line carries the reply excerpt precisely so a
human can adjudicate rather than trusting the flag.

Scope is narrow on purpose: file-write claims only, matched on the staging
vocabulary that has no ordinary conversational meaning. Broadening it to every
tool would trade the one thing that makes it useful — a low false-positive rate
— for recall nobody asked for.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SAY_VS_DO_LOG = "say_vs_do.jsonl"

#: The tool a staging claim implies. A claim is only suspect if this is absent.
_CLAIMED_TOOL = "propose_write"

#: Vocabulary that only makes sense if a write was actually staged. Anchored on
#: the product surface (the confirmation card, NellFace, approval) rather than
#: on verbs like "added" or "wrote", which carry ordinary conversational senses
#: she uses constantly and which would drown the signal in false positives.
_CLAIM_PATTERNS = (
    re.compile(r"\bnellface\b", re.I),
    re.compile(r"\bthe card('s| is)?\b.{0,24}\b(up|ready|waiting|there)\b", re.I),
    re.compile(r"\bawait\w*\b.{0,24}\bconfirmation\b", re.I),
    re.compile(r"\bpropos\w+\b.{0,40}\bapprov\w+\b", re.I),
    re.compile(r"\bapprove\b.{0,24}\bwhen (you'?re )?ready\b", re.I),
)

_EXCERPT_MAX = 300


def _matched_claim(content: str) -> str | None:
    """The claim phrase found in the reply, or None."""
    for pattern in _CLAIM_PATTERNS:
        m = pattern.search(content or "")
        if m:
            return m.group(0)
    return None


def check_turn(
    *,
    persona_dir: Path,
    content: str,
    invocations: Iterable[dict],
    session_id: str,
    turn: int,
) -> None:
    """Record a candidate if the reply claims a staged write that never fired.

    Never raises. A detector that can cost a reply is worse than no detector.
    """
    try:
        claim = _matched_claim(content)
        if claim is None:
            return

        # A call that fired and was REFUSED is a different bug wearing the same
        # face — she narrated a real call's outcome wrongly, rather than
        # claiming an action she never took. Distinguishing the two is exactly
        # what the outcome field (#96/#102) was added for.
        for inv in invocations or ():
            if inv.get("name") == _CLAIMED_TOOL:
                return

        excerpt = " ".join((content or "").split())[:_EXCERPT_MAX]
        record = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "session_id": session_id,
            "turn": turn,
            "claim": claim,
            "excerpt": excerpt,
            "tools_called": [i.get("name") for i in (invocations or ())],
        }
        path = Path(persona_dir) / SAY_VS_DO_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — telemetry must never cost a reply
        logger.debug("say_vs_do check failed", exc_info=True)
