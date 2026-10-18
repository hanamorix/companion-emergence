"""Recognising a Claude CLI model rejection, and remembering which models were rejected (#440).

When ``claude --model <id>`` is given an id that does not exist, or one the signed-in account cannot
use, the CLI exits 1 and its result frame says (claude 2.1.219, probed 2026-10-17)::

    There's an issue with the selected model (claude-sonet-5). It may not exist or you may not have access to it.

The CLI gives the same sentence for both cases. This module recognises that sentence in the exception
text the provider raises, names the rejected model, and keeps a small process-wide record per rejected
model so a saved model that later stops working is visible (a throttled WARNING and the
``model_rejections`` key of ``/persona/state``) instead of failing silently inside background jobs.

Detection is text matching because the provider forwards only the CLI's result frame, and the frame
carries the sentence but not the CLI's structured ``error: "model_not_found"`` key (that key sits on a
synthetic assistant frame neither parser reads). Whether the CURRENT CLI still words it this way is
checked by tests/bridge/test_model_rejection_live.py.

Shape mirrors :mod:`brain.bridge.provider_auth`: module globals under one lock, every public function
fails OPEN (an internal error never blocks a call or a turn), ``reset()`` for tests.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from brain import tunables
from brain.persona_config import is_valid_model

log = logging.getLogger(__name__)

MODEL_REJECTION_MARKER = "issue with the selected model"
_NAMED_MODEL = re.compile(r"selected model \(([^)\s]+)\)", re.IGNORECASE)

# Same cadence as provider_auth's re-probe interval: one WARNING per rejected model per quarter hour.
WARN_INTERVAL_S: float = tunables.register("model_error.warn_interval_seconds", 900.0)

# Module-level clock so tests inject time.
_monotonic = time.monotonic

_lock = threading.Lock()
_rejected: dict[str, dict[str, Any]] = {}
# Models that were switched away from: an ambient call that read the old id before the hot-swap must
# not re-record it when it fails later.
_ignored: set[str] = set()
# Models saved without the Apply-time check being able to confirm them (a check that timed out, hit a
# limit or found the login expired). Cleared by a clean frame for that model or by switching away.
_unconfirmed: set[str] = set()
# Derived tier ids an Apply could not probe (budget used up, timeout, expired login): "not checked yet".
# Cleared by a clean frame for that id, by a later Apply that settles it, or by reset().
_tier_unconfirmed: set[str] = set()
_local = threading.local()


def is_model_rejection(detail: str) -> bool:
    """True when ``detail`` (an exception's text) is the CLI rejecting the ``--model`` it was given."""
    return MODEL_REJECTION_MARKER in (detail or "").lower()


def rejected_model_name(detail: str, configured: str | None) -> str | None:
    """The model the CLI says it rejected, else ``configured``, else None.

    The name parsed from the text wins over ``configured``: it is the model that was ACTUALLY rejected
    (it survives a hot-swap during a turn), whereas ``configured`` is only the fallback. Only text that
    is valid model grammar (an alias or ``claude-[a-z0-9.-]+``) is ever returned, so no CLI prose is
    relayed to a client.
    """
    m = _NAMED_MODEL.search(detail or "")
    if m and is_valid_model(m.group(1)):
        return m.group(1)
    if configured is not None and is_valid_model(configured):
        return configured
    return None


@contextmanager
def suppressed() -> Iterator[None]:
    """Within this block (this thread only) rejections are neither recorded nor logged.

    The Apply-time check calls the CLI with an id that may be refused and never saved; that must not
    leave a "rejected model" record or WARNING for an id nobody uses.
    """
    previous = getattr(_local, "suppress", False)
    _local.suppress = True
    try:
        yield
    finally:
        _local.suppress = previous


def note_failure(detail: str) -> None:
    """Record a CLI failure text if it is a model rejection. Anything else is a no-op."""
    try:
        if getattr(_local, "suppress", False) or not is_model_rejection(detail):
            return
        model = rejected_model_name(detail, None)
        if model is None:
            return
        # Read the tunable BEFORE taking the lock: get_tunable stats a file under its own lock.
        interval = float(tunables.get_tunable("model_error.warn_interval_seconds", WARN_INTERVAL_S))
        now = _monotonic()
        stamp = datetime.now(UTC).isoformat()
        warn = False
        with _lock:
            if model in _ignored:
                return
            rec = _rejected.get(model)
            if rec is None:
                rec = {"since": stamp, "last_at": stamp, "count": 0, "warned": None}
                _rejected[model] = rec
            rec["count"] += 1
            rec["last_at"] = stamp
            if rec["warned"] is None or (now - rec["warned"]) >= interval:
                rec["warned"] = now
                warn = True
            count = rec["count"]
        if warn:  # log outside the lock so a slow handler never stalls state() readers
            log.warning(
                "Claude rejected the model %r (it may not exist, or this account may not have access to it): "
                "%d failed call(s) so far. Chat and background jobs using it keep failing until the model "
                "is changed in the connection panel.",
                model,
                count,
            )
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.note_failure failed (ignored)", exc_info=True)


def note_success(model: str) -> None:
    """A clean CLI frame for ``model``: its rejection record and unconfirmed mark (if any) are stale."""
    try:
        with _lock:
            _rejected.pop(model, None)
            _unconfirmed.discard(model)
            _tier_unconfirmed.discard(model)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.note_success failed (ignored)", exc_info=True)


def note_unconfirmed(model: str) -> None:
    """``model`` was saved without the Apply-time check being able to confirm it."""
    try:
        with _lock:
            _unconfirmed.add(model)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.note_unconfirmed failed (ignored)", exc_info=True)


def clear_unconfirmed(model: str) -> None:
    """A check just confirmed ``model`` (or it needs no check): drop its unconfirmed mark."""
    try:
        with _lock:
            _unconfirmed.discard(model)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.clear_unconfirmed failed (ignored)", exc_info=True)


def note_tier_unconfirmed(model: str) -> None:
    """A derived tier id the Apply-time check could not settle."""
    try:
        with _lock:
            _tier_unconfirmed.add(model)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.note_tier_unconfirmed failed (ignored)", exc_info=True)


def clear_tier_unconfirmed(model: str) -> None:
    """The Apply-time check settled this derived tier id (confirmed or rejected)."""
    try:
        with _lock:
            _tier_unconfirmed.discard(model)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.clear_tier_unconfirmed failed (ignored)", exc_info=True)


def tier_unconfirmed(only: Iterable[str] | None = None) -> list[str]:
    """Derived tier ids still "not checked yet", sorted; restricted to ``only`` when given."""
    try:
        with _lock:
            out = sorted(_tier_unconfirmed)
        if only is not None:
            allowed = frozenset(only)
            out = [m for m in out if m in allowed]
        return out
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.tier_unconfirmed failed (ignored)", exc_info=True)
        return []


def unconfirmed(only: Iterable[str] | None = None) -> list[str]:
    """Models still saved-but-unconfirmed, sorted; restricted to ``only`` when given."""
    try:
        with _lock:
            out = sorted(_unconfirmed)
        if only is not None:
            allowed = frozenset(only)
            out = [m for m in out if m in allowed]
        return out
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.unconfirmed failed (ignored)", exc_info=True)
        return []


def switch(old: str, new: str, keep: Iterable[str] = ()) -> None:
    """The saved model changed from ``old`` to ``new``.

    Drops ``old``'s record and ignores any later failure for it (an in-flight call that read the old
    id), unless ``old`` is in ``keep`` (a model some tier uses on its own). ``new`` is un-ignored.
    """
    try:
        if old == new:
            return
        keep_set = frozenset(keep)
        with _lock:
            _rejected.pop(old, None)
            _unconfirmed.discard(old)
            if old not in keep_set:
                _ignored.add(old)
            # An id that is still (or again) in use in some other role must never stay ignored, or a real
            # rejection of it would be dropped silently (#441).
            _ignored.difference_update(keep_set)
            _ignored.discard(new)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.switch failed (ignored)", exc_info=True)


def activate(models: Iterable[str]) -> None:
    """``models`` are in use (again): none of them may stay in the ignore set."""
    try:
        ids = frozenset(models)
        with _lock:
            _ignored.difference_update(ids)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.activate failed (ignored)", exc_info=True)


def rejected_models() -> frozenset[str]:
    """The models with a rejection record in this process (a snapshot)."""
    try:
        with _lock:
            return frozenset(_rejected)
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.rejected_models failed (ignored)", exc_info=True)
        return frozenset()


def state(only: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Copies of the rejection records, oldest first; restricted to ``only`` when given."""
    try:
        with _lock:
            recs = [
                {"model": m, "since": r["since"], "last_at": r["last_at"], "count": r["count"]}
                for m, r in _rejected.items()
            ]
        if only is not None:
            allowed = frozenset(only)
            recs = [r for r in recs if r["model"] in allowed]
        return sorted(recs, key=lambda r: r["last_at"])
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.state failed (ignored)", exc_info=True)
        return []


def reset() -> None:
    """Clear everything (tests)."""
    with _lock:
        _rejected.clear()
        _ignored.clear()
        _unconfirmed.clear()
        _tier_unconfirmed.clear()
    _local.suppress = False
