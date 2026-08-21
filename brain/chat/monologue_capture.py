"""Monologue tool-call capture for v0.0.26 inner monologue.

When `record_monologue` is dispatched in the chat tool loop, this module validates
the args and writes the feed digest synchronously to monologue_digest.jsonl. The
returned monologue text is queued for the async pass-2 Haiku extractor (memory +
emotion + soul + reflex_audit).

Best-effort on the digest write: a failed disk write logs to extractor_errors.jsonl
but does not raise (reply must still ship).
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.store import MemoryStore
from brain.monologue.trace import MONOLOGUE_TRACE_TYPE, write_trace_memory

logger = logging.getLogger(__name__)

MONOLOGUE_DIGEST_LOG = "monologue_digest.jsonl"
EXTRACTOR_ERROR_LOG = "extractor_errors.jsonl"

MAX_MONOLOGUE_LEN = 3000
MAX_FEED_DIGEST_LEN = 400


class CaptureRejected(ValueError):  # noqa: N818
    """Args failed validation; the tool dispatcher should treat as not-called."""


def _utcnow_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def capture_monologue(
    *,
    persona_dir: Path,
    store: MemoryStore,
    monologue: str,
    feed_digest: str,
    surface: bool = True,
) -> str:
    """Validate, persist the Tier-2 trace memory + the gated Tier-3 digest line.

    Returns the monologue text. Raises CaptureRejected on validation failure
    (whitespace-only or too-long or non-string). Trace-write and digest-write
    failures are logged to extractor_errors.jsonl best-effort, never raised
    (the reply must still ship).
    """
    if not isinstance(monologue, str) or not monologue.strip():
        raise CaptureRejected("monologue must be a non-whitespace string")
    if not isinstance(feed_digest, str) or not feed_digest.strip():
        raise CaptureRejected("feed_digest must be a non-whitespace string")
    if len(monologue) > MAX_MONOLOGUE_LEN:
        raise CaptureRejected(f"monologue exceeds {MAX_MONOLOGUE_LEN} chars")
    if len(feed_digest) > MAX_FEED_DIGEST_LEN:
        raise CaptureRejected(f"feed_digest exceeds {MAX_FEED_DIGEST_LEN} chars")

    # #93: the recruit-on-reach rerun can call record_monologue twice in one
    # turn. The narrowed toolset meant to prevent that is inert on
    # ClaudeCliProvider — the tools argument is a boolean there and
    # --allowedTools is permissive — so on the real path both calls land and
    # this ran twice, persisting two identical traces and two digest lines.
    #
    # Guard the artefact rather than the timing (same shape as #101/#105): the
    # duplicate is always immediately adjacent, so comparing against the newest
    # trace is sufficient, needs no window constant, and leaves her free to
    # think the same thought again another day.
    #
    # Fail-soft: if the lookup raises, capture anyway — never lose her thought
    # to a dedupe check.
    # INTEGRATION NOTE (temp-gate migration completion — NEEDS-DECISION):
    # monologue_trace is now a GATED type — write_trace_memory routes the trace
    # into the pending-candidate queue, not memories.db. temp-gate moved the
    # other monologue_trace reads (recall.py, ambient.py) to PendingQueue but
    # left THIS same-turn dedup guard reading committed rows, so it went blind to
    # the just-enqueued trace and a duplicate digest line was written (the #93
    # bug this guard exists to prevent). Reading the queue restores the guard,
    # matching temp-gate's own ambient.py/recall.py template. Provisional —
    # flag for owner confirmation of the read source.
    try:
        from brain.memory.pending import PendingQueue

        recent = PendingQueue(store.persona_dir).read_recent(MONOLOGUE_TRACE_TYPE, limit=1)
        if recent and recent[0].content == monologue:
            return monologue
    except Exception:  # noqa: BLE001
        logger.debug("monologue dedupe lookup failed; capturing anyway", exc_info=True)

    # Tier 2 — persist the verbatim trace FIRST (most important; never lose her
    # thought). Best-effort: a failure logs but does not block the reply.
    try:
        write_trace_memory(store, monologue)
    except Exception as exc:  # noqa: BLE001
        logger.warning("monologue trace memory write failed: %s", exc)
        try:
            _append_jsonl(
                persona_dir / EXTRACTOR_ERROR_LOG,
                {"ts": _utcnow_iso_z(), "step": "monologue_trace_write", "error": str(exc)},
            )
        except Exception:  # noqa: BLE001
            pass

    # Tier 3 — gated digest line (surfaced controls Feed visibility).
    try:
        _append_jsonl(
            persona_dir / MONOLOGUE_DIGEST_LOG,
            {"ts": _utcnow_iso_z(), "digest": feed_digest, "surfaced": bool(surface)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("monologue digest write failed: %s", exc)
        try:
            _append_jsonl(
                persona_dir / EXTRACTOR_ERROR_LOG,
                {
                    "ts": _utcnow_iso_z(),
                    "step": "monologue_digest_write",
                    "error": str(exc),
                },
            )
        except Exception:  # noqa: BLE001
            pass

    return monologue
