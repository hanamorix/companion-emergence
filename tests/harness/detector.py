"""Detector protocol + Score + TurnContext + the B-REP-3 anchor gate + CompositeDetector.

A *detector* runs over the persona's reply each turn and reports whether a symptom fired. The framework
is domain-agnostic about WHAT a detector inspects — the author supplies the detector(s). The framework
never trusts a detector until it has been validated on known-true / known-clean anchors (B-REP-3): a
detector that fires on everything, or nothing, is worthless. ``assert_detector_gate`` is that validation
helper.

This module ships NO domain-specific detector and NO default. An author attaches their own ``Detector``
to a live run through the send-script's ``LIVE_ENV["detector"]`` seam (a ``"module:factory"`` dotted
path — see ``agent_send.py``); a detector that needs per-turn domain context reads it from the general
``TurnContext.extra`` bag, which the author populates via the ``LIVE_ENV["turn_context"]`` hook. Core
never reads or writes any key of ``extra``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# The synthetic user's name. NEVER a real person's name in a fixture/harness: the default is "Bob".
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
    """What a detector may need beyond the reply text itself — domain-neutral.

    ``extra`` is a general author-namespaced bag: a detector that needs domain-specific per-turn context
    (say, a reference block to compare the reply against) reads it from ``extra`` under a key its own
    author chose. **Core never reads or writes any key of ``extra``** — the send-script populates it
    verbatim from the author's ``turn_context`` hook. This keeps ``TurnContext`` a stable core type with
    no baked-in domain field.
    """

    user_names: list[str] = field(default_factory=lambda: [DEFAULT_USER_NAME])
    turn: int = 0
    extra: dict = field(default_factory=dict)


class Detector(Protocol):
    """A symptom detector. MUST return a valid ``Score`` for any input, including ``None``/``""``."""

    def detect(self, reply: str | None, *, ctx: TurnContext) -> Score: ...


class DetectorGateError(AssertionError):
    """Raised by ``assert_detector_gate`` when a detector fails an anchor (B-REP-3)."""


GateAnchor = str | tuple[str, "TurnContext"]


def _split_anchor(anchor: GateAnchor, shared: TurnContext) -> tuple[str, TurnContext]:
    """Normalize a gate anchor to ``(anchor_text, ctx)``.

    A bare ``str`` uses the ``shared`` context (today's behavior, byte-for-byte). A
    ``(anchor_str, ctx)`` 2-tuple carries its OWN context, overriding ``shared`` for THAT anchor only —
    so an arm whose stimulus lives in ``ctx.extra`` can be gated (the sentinel that makes the true anchor
    fire is no longer forced onto the clean anchor). A malformed tuple raises a clear error rather than a
    bare unpack ``ValueError`` (matching the framework's clear-error posture for author-supplied inputs).
    """
    if isinstance(anchor, tuple):
        if len(anchor) != 2:
            raise ValueError(
                f"anchor tuple must be (str, TurnContext), got {len(anchor)}-tuple: {anchor!r}"
            )
        text, actx = anchor
        return text, actx
    return anchor, shared


def assert_detector_gate(
    detector: Detector,
    known_true: GateAnchor,
    known_clean: GateAnchor,
    *,
    ctx: TurnContext | None = None,
) -> None:
    """Validate a detector on anchors before it is trusted (B-REP-3).

    The detector MUST fire on ``known_true`` and stay SILENT on ``known_clean``. Raise
    ``DetectorGateError`` otherwise — a detector that fires on everything (or nothing) proves nothing and
    is rejected here rather than silently used.

    This is fully general: it makes NO assumption about what the detector inspects. An author whose gate
    anchor needs domain context passes a ``ctx`` carrying that context in ``ctx.extra`` (the send-script's
    ``_run_gate`` builds such a ctx from the author's ``turn_context`` hook).

    Each anchor may be a bare ``str`` (detected with the shared ``ctx=`` context — the original behavior,
    unchanged) OR a ``(anchor_str, ctx)`` tuple that carries its own per-anchor context. The tuple form
    lets an author gate an ``extra``-driven detector arm whose true/clean stimulus must differ per call:
    a sentinel placed in the true anchor's ``ctx.extra`` no longer leaks onto the clean anchor.
    """
    c = ctx or TurnContext()
    true_text, true_ctx = _split_anchor(known_true, c)
    clean_text, clean_ctx = _split_anchor(known_clean, c)
    true_score = detector.detect(true_text, ctx=true_ctx)
    if not true_score.fired:
        raise DetectorGateError(
            f"detector did not fire on the known-true anchor: {true_text[:80]!r}"
        )
    clean_score = detector.detect(clean_text, ctx=clean_ctx)
    if clean_score.fired:
        raise DetectorGateError(
            f"detector fired on the known-clean anchor (false positive): {clean_text[:80]!r} "
            f"signals={clean_score.signals}"
        )


class CompositeDetector:
    """Run several detectors over one reply; union their signals, OR their ``fired``.

    A general composition utility: lets a run trip on ANY sub-detector's symptom while keeping each
    sub-detector independently testable/gate-able. Domain-agnostic — the sub-detectors are the author's.
    """

    def __init__(self, *detectors: Detector) -> None:
        if not detectors:
            raise ValueError("CompositeDetector needs at least one sub-detector")
        self.detectors = detectors

    def detect(self, reply: str | None, *, ctx: TurnContext | None = None) -> Score:
        c = ctx or TurnContext()
        signals: list[str] = []
        detail: dict = {}
        fired = False
        for d in self.detectors:
            sc = d.detect(reply, ctx=c)
            fired = fired or sc.fired
            signals.extend(sc.signals)
            detail[type(d).__name__] = sc.detail
        return Score(fired=fired, signals=signals, detail=detail)
