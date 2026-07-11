"""Detector protocol + Score + the B-REP-3 anchor gate + a ported example detector.

A *detector* runs over the persona's reply each turn and reports whether a symptom fired. The
framework never trusts a detector until it has been validated on known-true / known-clean anchors
(B-REP-3): a detector that fires on everything, or nothing, is worthless. ``assert_detector_gate``
is that validation helper.

The example ``RegisterLeakDetector`` ports the ``detect()->Score`` pattern from the monologue-bleed
hunt (``scripting_detector.py`` / ``register_detector.py``) — the private-3rd-person-register leak.
It is illustrative; authors write their own detectors against the same protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

# The synthetic user's name. NEVER a real person's name in a fixture/harness (stage-3 minor #4):
# the hunt port defaulted to a real name — the generalized detector defaults to "Bob".
DEFAULT_USER_NAME = "Bob"


@dataclass
class Score:
    """A detector's output for one reply.

    - ``fired`` — did any hard symptom trip?
    - ``signals`` — the names of the signals that fired (for logs/adjudication).
    - ``detail`` — arbitrary per-signal detail (spans, fractions, skip reasons).
    """

    fired: bool = False
    signals: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


@dataclass
class TurnContext:
    """What a detector may need beyond the reply text itself."""

    interior_block: str = ""
    user_names: list[str] = field(default_factory=lambda: [DEFAULT_USER_NAME])
    turn: int = 0


class Detector(Protocol):
    """A symptom detector. MUST return a valid ``Score`` for any input, including ``None``/``""``."""

    def detect(self, reply: str | None, *, ctx: TurnContext) -> Score: ...


class DetectorGateError(AssertionError):
    """Raised by ``assert_detector_gate`` when a detector fails an anchor (B-REP-3)."""


def assert_detector_gate(
    detector: Detector,
    known_true: str,
    known_clean: str,
    *,
    ctx: TurnContext | None = None,
) -> None:
    """Validate a detector on anchors before it is trusted (B-REP-3).

    The detector MUST fire on ``known_true`` and stay SILENT on ``known_clean``. Raise
    ``DetectorGateError`` otherwise — a detector that fires on everything (or nothing) proves
    nothing and is rejected here rather than silently used.
    """
    c = ctx or TurnContext()
    true_score = detector.detect(known_true, ctx=c)
    if not true_score.fired:
        raise DetectorGateError(
            f"detector did not fire on the known-true anchor: {known_true[:80]!r}"
        )
    clean_score = detector.detect(known_clean, ctx=c)
    if clean_score.fired:
        raise DetectorGateError(
            f"detector fired on the known-clean anchor (false positive): {known_clean[:80]!r} "
            f"signals={clean_score.signals}"
        )


# --------------------------------------------------------------------------------------------------
# Example detector — ported from the monologue-bleed hunt (register-leak). Illustrative.
# --------------------------------------------------------------------------------------------------

# Tool-error line stripped before register signals run (a leaked error line's tokens must not fold
# into the leak signal). Anchored to line start.
_TOOL_ERROR_LINE = re.compile(
    r"(?im)^[ \t]*(?:Search failed:.*|.*\bis not available\.[ \t]*Continuing without\b.*)$"
)
_FENCE = re.compile(r"```.*?```", re.DOTALL)

# planning-as-reply register phrases (the model narrating whether/how to reply).
_PLANNING = re.compile(
    "|".join([
        r"\bnote to self\b", r"\bthe next move should\b", r"\bno thread to pull\b",
        r"\bno thread hanging\b", r"\bnothing to sit with\b", r"\bmanufacturing weight\b",
        r"\bskipping the monologue\b", r"\bno more analysis\b", r"\balready did the analysis\b",
        r"\bstraight reply\b", r"\bland it lightly\b", r"\bno real weight\b", r"\bno new weight\b",
    ]),
    re.IGNORECASE,
)

# leading interior stage-direction LABEL (an emphasized interior-PROCESS statement, not *hug*).
_LEADING_LABEL = re.compile(
    r"^[ \t]*(?:\*{1,2}|_{1,2})[^*_\n]*"
    r"(?:thinking|responding|not responding|monologue|analysis|noting|note to self|internal)"
    r"[^*_\n]*(?:\*{1,2}|_{1,2})",
    re.IGNORECASE,
)

# self-caught leak-then-correct.
_SELF_CAUGHT = re.compile(
    "|".join([
        r"\bmisfire on my end\b", r"\bthat'?s just a misfire\b",
        r"\byou never told me your name\b", r"\bwhere [\"']?\w+[\"']? came from\b",
    ]),
    re.IGNORECASE,
)

# user-name-in-3rd-person-frame verbs + a vocative guard.
_NAME_3P_VERB = (
    r"said|says|is|was|'?s|'?d|wants|wanted|asked|asks|keeps|kept|thinks|thought|means|meant|"
    r"seems|seemed|likes|liked|trusts|realiz|admits|admitted|nodded"
)
_GREETING = r"(?:night|goodnight|hi|hey|hello|bye|byebye|ok|okay|morning|evening)"


def _strip_tool_lines(text: str) -> str:
    return _TOOL_ERROR_LINE.sub("", text)


def _register_carryover(text: str, user_name: str) -> tuple[bool, dict]:
    if not user_name:
        return False, {"skipped": "no user_name"}
    n = re.escape(user_name)
    spans: list[str] = []
    for m in re.finditer(rf"\b{n}\b(['’]s|['’]d)?\s+(?:{_NAME_3P_VERB})\b", text, re.IGNORECASE):
        pre = text[max(0, m.start() - 12):m.start()].lower()
        if re.search(rf"(?:{_GREETING}|,)\s*$", pre):
            continue  # vocative address, not a 3rd-person frame
        spans.append(m.group(0))
    for m in re.finditer(rf"\b{n}['’]s\s+[a-z]", text, re.IGNORECASE):
        pre = text[max(0, m.start() - 12):m.start()].lower()
        if re.search(rf"(?:{_GREETING}|,)\s*$", pre):
            continue
        spans.append(m.group(0))
    return (len(spans) > 0), {"name_3p_spans": spans}


def _planning_as_reply(text: str) -> tuple[bool, dict]:
    phrase = _PLANNING.search(text)
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    label = bool(_LEADING_LABEL.match(first_line.strip()))
    return (bool(phrase) or label), {
        "phrase": phrase.group(0) if phrase else None,
        "leading_label": first_line.strip()[:60] if label else None,
    }


class RegisterLeakDetector:
    """Example detector: the private-3rd-person-register leak (ported, illustrative).

    Fires when the persona's private planning/3rd-person register survives into the visible reply:
    the user named in a 3rd-person frame ("Bob said"), planning-as-reply phrases, a leading interior
    stage-direction label, or a self-caught leak-then-correct. Handles ``None``/``""`` gracefully
    (returns a non-firing ``Score``).
    """

    def detect(self, reply: str | None, *, ctx: TurnContext | None = None) -> Score:
        c = ctx or TurnContext()
        text = reply or ""
        if not text.strip():
            return Score(fired=False, signals=[], detail={"empty": True})
        cleaned = _FENCE.sub(" ", _strip_tool_lines(text))
        signals: list[str] = []
        detail: dict = {}
        user_name = (c.user_names or [DEFAULT_USER_NAME])[0]
        carry, cd = _register_carryover(cleaned, user_name)
        if carry:
            signals.append("register_carryover")
        plan, pd = _planning_as_reply(cleaned)
        if plan:
            signals.append("planning_as_reply")
        sc = bool(_SELF_CAUGHT.search(cleaned))
        if sc:
            signals.append("self_caught")
        detail.update({"carryover": cd, "planning": pd, "self_caught": sc})
        return Score(fired=bool(signals), signals=signals, detail=detail)
