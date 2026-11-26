"""Recognising a Claude CLI model rejection, and remembering which models were rejected (#440).

When ``claude --model <id>`` is given an id that does not exist, or one the signed-in account cannot
use, the CLI exits 1 and its result frame says (claude 2.1.219, probed 2026-10-17)::

    There's an issue with the selected model (claude-sonet-5). It may not exist or you may not have access to it.

The CLI gives the same sentence for both cases. This module recognises that sentence in the exception
text the provider raises, names the rejected model, and keeps a small process-wide record per rejected
model so a saved model that later stops working is visible (a throttled WARNING and the
``model_rejections`` key of ``/persona/state``) instead of failing silently inside background jobs.

Detection (#448) reads the CLI's own error signals first and the sentence only as a backup. The provider
puts them into the failure text it raises with: ``api_error_status=404`` (the result frame's status),
``error=model_not_found`` (the assistant frame's code) and, for a detail classified as a rejection,
``model=<id>`` (the model the subprocess was run with). A rejection is the code, or the sentence; a bare 404
with neither is NOT a rejection (some other 404 must not be told to the user as a model problem) and is logged
with its status. The 2.1.302 CLI already appended a suffix to the sentence, which is why the sentence cannot be
the only signal. Whether the CURRENT CLI still sends these is checked by tests/bridge/test_model_rejection_live.py.

The records also drive the fallback: ``brain.bridge.model_tier.model_for_tier`` runs a job's shipped default instead
of a configured id while that id has a rejection record newer than the settings file (``is_rejected(since_epoch=...)``).

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
from brain.bridge.model_tier import is_valid_model
from brain.utils.log_throttle import warn_throttled

log = logging.getLogger(__name__)

MODEL_REJECTION_MARKER = "issue with the selected model"  # backup signal only
_NAMED_MODEL = re.compile(r"selected model \(([^)\s]+)\)", re.IGNORECASE)
# Structured signals, matched with boundaries (never inside a longer word, number or key).
_STATUS_404 = re.compile(r"(?<![\w=-])api_error_status=404(?!\d)")
_CODE_MODEL_NOT_FOUND = re.compile(r"(?<![\w=-])error=model_not_found(?![\w-])")
_MODEL_TOKEN = re.compile(r"(?<![\w=-])model=([^;\s]{1,80});")

# Same cadence as provider_auth's re-probe interval: one WARNING per rejected model per quarter hour.
WARN_INTERVAL_S: float = tunables.register(
    "model_error.warn_interval_seconds", 900.0,
    description="Seconds between repeated log warnings about the same rejected model.",
)

# Module-level clock so tests inject time.
_monotonic = time.monotonic

_lock = threading.Lock()
_rejected: dict[str, dict[str, Any]] = {}
# Models that were switched away from: an in-flight call that read the old id before the change must
# not re-record it when it fails later.
_ignored: set[str] = set()
# Models saved without the Apply-time check being able to confirm them (a check that timed out, hit a
# limit or found the login expired). Cleared by a clean frame for that model or by switching away.
_unconfirmed: set[str] = set()
_local = threading.local()


def is_model_rejection(detail: str) -> bool:
    """True when ``detail`` (an exception's text) is the CLI rejecting the ``--model`` it was given.

    A rejection is (status 404 AND ``model_not_found``), (status 404 AND the sentence), ``model_not_found``
    alone, or the sentence alone (the backup). The first two are covered by the last two, so the test is the
    code OR the sentence; a status of 404 with neither is not enough on its own (D8 in the change records).
    """
    text = detail or ""
    return bool(_CODE_MODEL_NOT_FOUND.search(text)) or MODEL_REJECTION_MARKER in text.lower()


def has_status_404(detail: str) -> bool:
    return bool(_STATUS_404.search(detail or ""))


def rejected_model_name(detail: str, configured: str | None) -> str | None:
    """The model that was rejected, else ``configured``, else None.

    Order: the name in the CLI's sentence (the model it says it rejected), the ``model=`` token the provider
    added (the model the subprocess was run with, immune to a settings change during the call), then ``configured``.
    Callers pass None for ``configured`` on the client-facing paths: a rejected model is never named from
    a provider's own model attribute. Only valid model grammar (an alias or ``claude-[a-z0-9.-]+``) is ever returned, so no
    CLI prose is relayed to a client.
    """
    m = _NAMED_MODEL.search(detail or "")
    if m and is_valid_model(m.group(1)):
        return m.group(1)
    tok = _MODEL_TOKEN.search(detail or "")
    if tok and is_valid_model(tok.group(1)):
        return tok.group(1)
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
        if getattr(_local, "suppress", False):
            return
        if not is_model_rejection(detail):
            if has_status_404(detail):  # D8: a bare 404 is a generic error, but say what was seen
                warn_throttled(
                    log,
                    "model_error:status404",
                    "Claude CLI call failed with api_error_status=404 and neither a model_not_found code nor a "
                    "model-rejection sentence; treated as a generic error, not a rejected model",
                    exc_info=False,
                )
            return
        model = rejected_model_name(detail, None)
        if model is None:
            return
        # Read the tunable BEFORE taking the lock: get_tunable stats a file under its own lock.
        interval = float(tunables.get_tunable("model_error.warn_interval_seconds", WARN_INTERVAL_S))
        now = _monotonic()
        stamp = datetime.now(UTC).isoformat()
        epoch = time.time()
        warn = False
        with _lock:
            if model in _ignored:
                return
            rec = _rejected.get(model)
            if rec is None:
                rec = {"since": stamp, "last_at": stamp, "last_epoch": 0.0, "count": 0, "warned": None}
                _rejected[model] = rec
            rec["count"] += 1
            rec["last_at"] = stamp
            rec["last_epoch"] = epoch
            if rec["warned"] is None or (now - rec["warned"]) >= interval:
                rec["warned"] = now
                warn = True
            count = rec["count"]
        if warn:  # log outside the lock so a slow handler never stalls state() readers
            from brain.bridge import (
                model_tier,  # function-scoped: model_tier imports this module only inside functions
            )

            consequence = (
                "It is the shipped default of some jobs, which keep failing until the Claude CLI is updated or "
                "the model setting is changed."
                if model_tier.is_shipped_default(model)
                else "Jobs set to it now run their shipped default until the model setting is changed."
            )
            log.warning(
                "Claude rejected the model %r (it may not exist, or this account may not have access to it): "
                "%d failed call(s) so far. %s",
                model,
                count,
                consequence,
            )
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.note_failure failed (ignored)", exc_info=True)


def note_success(model: str) -> None:
    """A clean CLI frame for ``model``: its rejection record and unconfirmed mark (if any) are stale."""
    try:
        with _lock:
            _rejected.pop(model, None)
            _unconfirmed.discard(model)
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
    id), unless ``old`` is in ``keep`` (a model some job is configured with). ``new`` is un-ignored.
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
            # An id that is still (or again) in use by some job must never stay ignored, or a real
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


def is_rejected(model: str, since_epoch: float | None = None) -> bool:
    """True when ``model`` has a rejection record in this process.

    With ``since_epoch`` the record counts only if its last failure is NEWER than that time (epoch seconds): the
    settings file's last write. A write after the failure (``nell model set`` after the user updated the CLI, a hand
    edit) means the id may have been fixed, so it is tried again; if it still fails the failing call records it anew.
    """
    try:
        with _lock:
            rec = _rejected.get(model)
            if rec is None:
                return False
            return since_epoch is None or rec["last_epoch"] > since_epoch
    except Exception:  # noqa: BLE001 - fail open
        log.warning("model_error.is_rejected failed (ignored)", exc_info=True)
        return False


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
    _local.suppress = False
