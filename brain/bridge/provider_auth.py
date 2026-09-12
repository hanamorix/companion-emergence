"""Process-wide auth-expiry state for the Claude CLI provider (#246).

The brain's dedicated ``CLAUDE_CONFIG_DIR`` login can expire; when it does every
``claude -p`` the supervisor spawns fails with ``OAuth session expired`` and each
engine's fail-soft handler logs and moves on — the autonomic layer dies silently.
This module turns that CLI error text into one bridge-owned state the provider
consults before spawning, ``/health`` and ``/persona/state`` report, and chat
errors classify against.

Shape mirrors :mod:`brain.bridge.cli_throttle`: module globals under one lock,
every public function fails OPEN (an internal error never blocks a call),
``reset()`` for tests. In-memory by design — a bridge restart re-authenticates
through the Keychain, so persisting the state would only resurrect a stale flag.

Debounce (spec §6): the observed live expiry text (``oauth session expired``) is
*strong* and flips ``ok -> expired`` on first sight. ``not logged in`` /
``failed to authenticate`` are *weak*: the live log shows ``Not logged in`` 13/13
times as a shutdown artefact (Keychain unreadable during teardown), never as an
expiry, so a weak hit needs a second weak hit >= WEAK_CONFIRM_S later.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime

from brain import tunables
from brain.bridge.cli_throttle import ThrottleDeferred

log = logging.getLogger(__name__)

_REPROBE_S_DEFAULT = tunables.register("provider_auth.reprobe_seconds", 900.0)
REPROBE_S = _REPROBE_S_DEFAULT  # read through _reprobe_s() at call time
WEAK_CONFIRM_S = 60.0

_STRONG = ("oauth session expired",)
_WEAK = ("not logged in", "failed to authenticate")

# Module-level clock so tests inject time without a `now` kwarg on generate().
_monotonic = time.monotonic

_lock = threading.Lock()
_status: str = "ok"
_since: datetime | None = None
_detail: str = ""
_failures: int = 0
_last_probe_monotonic: float = -1e9
_weak_first_at: float | None = None


class ProviderAuthDeferred(ThrottleDeferred):
    """Raised by ``ClaudeCliProvider.generate`` while the login is expired.

    A ``ThrottleDeferred`` subclass so the callers that already special-case a
    quiet defer keep treating it as a no-op, not a failure.
    """


def _reprobe_s() -> float:
    return float(tunables.get_tunable("provider_auth.reprobe_seconds", _REPROBE_S_DEFAULT))


def is_strong_auth_failure(detail: str) -> bool:
    d = (detail or "").lower()
    return any(s in d for s in _STRONG)


def is_auth_failure(detail: str) -> bool:
    d = (detail or "").lower()
    return any(s in d for s in _STRONG + _WEAK)


def note_cli_failure(detail: str) -> None:
    """Record a CLI failure text. Non-auth text is a no-op."""
    global _status, _since, _detail, _failures, _last_probe_monotonic, _weak_first_at
    try:
        if not is_auth_failure(detail):
            return
        strong = is_strong_auth_failure(detail)
        now = _monotonic()
        with _lock:
            if not strong:
                if _weak_first_at is None or (now - _weak_first_at) < WEAK_CONFIRM_S:
                    if _weak_first_at is None:
                        _weak_first_at = now
                    return
            _failures += 1
            _detail = (detail or "")[:200]
            flipped = _status == "ok"
            if flipped:
                _status = "expired"
                _since = datetime.now(UTC)
                _last_probe_monotonic = now  # the next background call waits a full interval
        if flipped:  # log outside the lock so a slow handler never stalls state() readers
            log.warning(
                "provider auth EXPIRED — background LLM calls backed off to one probe per "
                "%.0fs until a call succeeds; re-authorise from the connection panel: %s",
                _reprobe_s(),
                _detail,
            )
    except Exception:  # noqa: BLE001 — fail open
        log.warning("provider_auth.note_cli_failure failed (ignored)", exc_info=True)


def note_cli_success() -> None:
    """A clean CLI frame: clear any expiry and any pending weak hit."""
    global _status, _since, _detail, _failures, _weak_first_at
    try:
        recovered = False
        with _lock:
            _weak_first_at = None
            if _status == "expired":
                _status = "ok"
                _since = None
                _failures = 0
                _detail = ""
                recovered = True
        if recovered:
            log.info("provider auth recovered — background LLM calls resumed")
    except Exception:  # noqa: BLE001 — fail open
        log.warning("provider_auth.note_cli_success failed (ignored)", exc_info=True)


def should_skip_background() -> bool:
    """True when a background call should be deferred; grants one probe per interval."""
    global _last_probe_monotonic
    try:
        with _lock:
            if _status != "expired":
                return False
            now = _monotonic()
            if (now - _last_probe_monotonic) >= _reprobe_s():
                _last_probe_monotonic = now
                return False  # the probe
            return True
    except Exception:  # noqa: BLE001 — fail open
        log.warning("provider_auth.should_skip_background failed; allowing", exc_info=True)
        return False


def state() -> dict:
    with _lock:
        return {
            "status": _status,
            "since": _since.isoformat() if _since else None,
            "detail": _detail,
            "failures": _failures,
        }


def reset() -> None:  # test helper
    global _status, _since, _detail, _failures, _last_probe_monotonic, _weak_first_at
    with _lock:
        _status = "ok"
        _since = None
        _detail = ""
        _failures = 0
        _last_probe_monotonic = -1e9
        _weak_first_at = None
