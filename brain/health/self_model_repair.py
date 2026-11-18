"""One-time reset of the persisted self-model gap (self_model_state.json).

The v0.0.36 derived read used a total-mass-normalised mean that produced a large
uniform-negative gap artifact (magnitude in the hundreds — "almost everything I
claim to feel is arriving at a fraction of the strength") for any populated
persona. The windowed-peak fix corrects the computation, but a persona that ran
the old code has a bogus OPEN gap persisted, carrying a confabulated note. This
clears ``current_gap`` ONCE so the next reflection tick recomputes honestly and
no false "natural resolution" fires from the collapsing artifact. ``gap_history``
is preserved. Marker-gated → runs at most once per persona.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.health.json_io import read_json_result, write_json
from brain.self_model import state as sm_state
from brain.utils.file_lock import LockTimeoutError

logger = logging.getLogger(__name__)

_MARKER_FILENAME = "self_model_repair_state.json"
# A repair that fails this many starts in a row gives up (#737): a fault that never clears must not log a traceback on
# every start.
_MAX_FAILED_ATTEMPTS = 3


def _failed_attempts(persona_dir: Path) -> int | None:
    """The failed starts the marker records, or None when it says the repair is over (done, or given up).

    Marker shapes: missing (0 failures), ``{"done": false, "failed_attempts": n}`` while retrying, ``{"done": true}``
    once finished (``"gave_up": true`` beside it after the last failure). A marker that cannot be read, or holds anything
    else, reads as over, as its mere presence did before #737; so does every shape for a brain older than #737, which
    checks only that the file exists.
    """
    res = read_json_result(persona_dir / _MARKER_FILENAME, default=lambda: None, expect=dict)
    if res.outcome == "missing":
        return 0
    data = res.data
    if not isinstance(data, dict) or data.get("done") is not False:
        return None
    count = data.get("failed_attempts")
    return count if type(count) is int and count >= 0 else None


def should_run_self_model_repair(persona_dir: Path) -> bool:
    """True until the repair has finished, or has failed `_MAX_FAILED_ATTEMPTS` starts in a row."""
    count = _failed_attempts(persona_dir)
    return count is not None and count < _MAX_FAILED_ATTEMPTS


def _write_marker(persona_dir: Path, data: dict) -> None:
    write_json(persona_dir / _MARKER_FILENAME, data, indent=None, fsync=False, on_error="log")


def _record_failure(persona_dir: Path) -> None:
    """Count one more failed start; on the last allowed one, mark the repair over (given up) with one warning."""
    count = (_failed_attempts(persona_dir) or 0) + 1
    if count < _MAX_FAILED_ATTEMPTS:
        _write_marker(persona_dir, {"done": False, "failed_attempts": count})
        return
    logger.warning(
        "self_model_repair: gave up after %d failed starts; a persisted gap is left as it is", count
    )
    _write_marker(persona_dir, {"done": True, "gave_up": True, "failed_attempts": count})


def run_self_model_repair(persona_dir: Path) -> None:
    """Clear any persisted current_gap once, then write the marker. Fail-soft.

    The marker says done only when the repair finished (the gap was cleared, or there was none). A lock that did not come
    is not counted and is retried at the next start; any other failure is counted and retried, up to
    `_MAX_FAILED_ATTEMPTS` starts (#737).
    """
    try:
        # Under the state lock (#558): a reconcile in flight would otherwise write the cleared gap back.
        with sm_state.locked(persona_dir):
            state, _recovered = sm_state.load_or_recover(persona_dir)
            if state.current_gap is not None:
                sm_state.save(
                    persona_dir,
                    sm_state.SelfModelState(current_gap=None, gap_history=state.gap_history),
                )
                logger.info(
                    "self_model_repair: cleared a persisted gap (windowed-peak migration)"
                )
    except LockTimeoutError as exc:
        # Another writer kept the state lock for the whole bounded wait: nothing was cleared, so the marker is not
        # written and the next start runs the repair again.
        logger.warning("self_model_repair: %s; not run, retried at the next start", exc)
        return
    except Exception:
        # Nothing was cleared (#737): the marker counts the failure, and the next start runs the repair again.
        logger.exception("self_model_repair: failed (non-fatal); retried at the next start")
        _record_failure(persona_dir)
        return
    _write_marker(persona_dir, {"done": True})
