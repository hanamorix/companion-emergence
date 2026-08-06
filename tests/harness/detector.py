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


class DetectorGateConfigError(ValueError):
    """Raised by ``assert_detector_gate`` when the ANCHOR CONFIG itself is malformed — today, an
    EMPTY anchor battery on either side (a 0/0 "pass" proves nothing).

    Distinct from a detector *failing* the gate (``DetectorGateError``) AND from an author's own
    ``detector.detect()`` raising a bare ``ValueError``. It subclasses ``ValueError`` so a direct
    caller catching ``ValueError`` still catches it (backward-compatible), while a seam
    (``agent_send.main``) can catch it SPECIFICALLY and report a *config* error without swallowing —
    and mislabeling — a detector-raised ``ValueError``.
    """


GateAnchor = str | tuple[str, "TurnContext"]
# A single anchor OR a battery (list) of anchors. The discriminator is ``isinstance(x, list)``: a
# ``list`` is a battery (one element per anchor); a bare ``str`` or a ``(str, TurnContext)`` tuple is
# a SINGLE anchor. So the single-anchor path is a strict special case of the battery path.
GateAnchors = GateAnchor | list[GateAnchor]


@dataclass
class GateReport:
    """Per-side hit/miss counts from :func:`assert_detector_gate` (B-REP-3).

    Returned on a PASS so a caller can read the MEASURED reliability of the anchor battery instead of
    a bare boolean — the thing the single-anchor gate could not surface. ``clean_fired`` is the
    false-positive count (a known-clean anchor that wrongly fired). The rate properties are
    fired/total per side.
    """

    true_total: int
    true_fired: int
    clean_total: int
    clean_fired: int
    true_missed: list[str] = field(default_factory=list)
    clean_false_fired: list[str] = field(default_factory=list)

    @property
    def true_hit_rate(self) -> float:
        return self.true_fired / self.true_total if self.true_total else 0.0

    @property
    def clean_fp_rate(self) -> float:
        return self.clean_fired / self.clean_total if self.clean_total else 0.0

    @property
    def passed(self) -> bool:
        return not self.true_missed and not self.clean_false_fired


def _as_anchor_list(anchors: GateAnchors) -> list[GateAnchor]:
    """Normalize a single anchor OR a battery to a ``list`` of anchors.

    A ``list`` passes through (a battery); anything else — a bare ``str`` or a ``(str, TurnContext)``
    per-anchor tuple — becomes a 1-element list, so a single-anchor call is a strict special case of
    the battery path and runs byte-identical fire/silent checks.
    """
    return list(anchors) if isinstance(anchors, list) else [anchors]


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
    known_true: GateAnchors,
    known_clean: GateAnchors,
    *,
    ctx: TurnContext | None = None,
) -> GateReport:
    """Validate a detector on anchors before it is trusted (B-REP-3).

    The detector MUST fire on EVERY ``known_true`` anchor and stay SILENT on EVERY ``known_clean``
    anchor. Each side may be a SINGLE anchor OR a **list** of anchors — an anchor BATTERY — and the
    WHOLE battery must pass. A detector that fires on everything, on nothing, or that is unreliable
    across the long tail (fires on task-vocabulary, misses a real leak) is rejected here rather than
    silently used — the Type-3 both-directions-unreliability a single anchor pair could not preclude.

    Returns a :class:`GateReport` carrying the per-side hit/miss RATE (fired/total) on success, so a
    caller reads the MEASURED reliability instead of a bare pass/fail. Raises ``DetectorGateError`` if
    any known-true anchor misses or any known-clean anchor fires — the message carries both per-side
    rates and the offending anchor text(s). Raises ``ValueError`` if either side's battery is EMPTY (a
    vacuous 0/0 "pass" is exactly the prove-nothing hole the gate exists to close).

    This is fully general: it makes NO assumption about what the detector inspects. An author whose gate
    anchor needs domain context passes a ``ctx`` carrying that context in ``ctx.extra`` (the send-script's
    ``_run_gate`` builds such a ctx from the author's ``turn_context`` hook).

    Each anchor may be a bare ``str`` (detected with the shared ``ctx=`` context — the original behavior,
    unchanged) OR a ``(anchor_str, ctx)`` tuple that carries its own per-anchor context. The tuple form
    lets an author gate an ``extra``-driven detector arm whose true/clean stimulus must differ per call:
    a sentinel placed in the true anchor's ``ctx.extra`` no longer leaks onto the clean anchor. A single
    ``str`` / ``(str, ctx)`` anchor on each side reproduces the original one-true/one-clean check
    byte-for-byte (a 1-element battery); no existing caller inspects the return value, so widening
    ``None`` -> :class:`GateReport` is invisible to them.
    """
    c = ctx or TurnContext()
    true_anchors = _as_anchor_list(known_true)
    clean_anchors = _as_anchor_list(known_clean)
    if not true_anchors or not clean_anchors:
        raise DetectorGateConfigError(
            "assert_detector_gate needs at least one known-true AND one known-clean anchor "
            f"(got {len(true_anchors)} true, {len(clean_anchors)} clean) — an empty anchor "
            "battery would vacuously pass and prove nothing"
        )

    true_fired = 0
    true_missed: list[str] = []
    for anchor in true_anchors:
        text, actx = _split_anchor(anchor, c)
        if detector.detect(text, ctx=actx).fired:
            true_fired += 1
        else:
            true_missed.append(text)

    clean_fired = 0
    clean_false_fired: list[str] = []
    for anchor in clean_anchors:
        text, actx = _split_anchor(anchor, c)
        if detector.detect(text, ctx=actx).fired:
            clean_fired += 1
            clean_false_fired.append(text)

    report = GateReport(
        true_total=len(true_anchors),
        true_fired=true_fired,
        clean_total=len(clean_anchors),
        clean_fired=clean_fired,
        true_missed=true_missed,
        clean_false_fired=clean_false_fired,
    )
    if not report.passed:
        parts = [
            f"detector failed the anchor battery: known-true fired "
            f"{true_fired}/{len(true_anchors)}, known-clean false-fired "
            f"{clean_fired}/{len(clean_anchors)}"
        ]
        if true_missed:
            parts.append(
                "missed known-true anchor(s): "
                + "; ".join(repr(t[:80]) for t in true_missed)
            )
        if clean_false_fired:
            parts.append(
                "false-fired on known-clean anchor(s): "
                + "; ".join(repr(t[:80]) for t in clean_false_fired)
            )
        raise DetectorGateError(" | ".join(parts))
    return report


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
