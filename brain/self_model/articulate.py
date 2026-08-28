"""Budgeted, throttled Haiku articulation of an emotional gap.

`articulate(gap, *, provider, persona_dir) -> str | None`

One honest Haiku sentence - "what I notice" - when the gap is large enough
and the daily budget has not been exhausted.

Budget: _DAILY_ARTICULATE_BUDGET calls/day, midnight-local reset, stored in
  <persona_dir>/self_model/daily_articulate_budget.json
  Mirrors brain/attunement/budget.py exactly in shape.
  Fail-safe-permissive: a corrupt/unreadable file -> allow the call.

Throttle: requests the shared cli_throttle background slot with a short
  min_idle (self_model.articulate_min_idle_seconds, default 30s) rather than
  the 300s default - self-model's tick is low-priority background housekeeping,
  same treatment brain/chat/pass2_queue.py already gives its own low-priority
  drain. On denial, raises cli_throttle.ThrottleDeferred (does not sleep, does
  not retry inline) - the caller (brain/bridge/supervisor.py's
  _run_self_model_tick) is the one that retries, via a pre-flight
  cli_throttle.slot_available() peek checked BEFORE this module's gap
  computation ever runs, so a denied attempt costs nothing. This function's
  own acquire_background() call only ever runs after that peek has already
  granted - it is the authoritative guard for the rare race where chat
  resumes between the peek and this call, not the common-case check.

Usage: every permitted call is logged via brain/bridge/usage_log.log_usage
  with call_type="self_model_articulate".

Fail-soft: any provider error -> return None. The gap stands without a note;
  the reflection pipeline must never crash on an LLM.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import tunables
from brain.bridge import cli_throttle
from brain.bridge.usage_log import log_usage
from brain.self_model.gap import Gap

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GAP_THRESHOLD: float = 0.4          # below this magnitude → skip articulation
_DAILY_ARTICULATE_BUDGET: int = 50   # Haiku calls / persona / day

_BUDGET_FILE = "daily_articulate_budget.json"
_ARTICULATE_ERRORS_FILE = "self_model_articulate_errors.jsonl"

# The idle bar this call requests from cli_throttle, same value and rationale
# as pass2_queue.py's _PASS2_IDLE_SECONDS: a short, tunable, non-default
# window for a low-priority background call. Public (no leading underscore)
# because brain/bridge/supervisor.py's pre-flight peek needs the exact same
# value the real acquire below uses - one shared getter, not a duplicated
# tunable lookup.
_ARTICULATE_IDLE_SECONDS = tunables.register("self_model.articulate_min_idle_seconds", 30.0)


def articulate_min_idle_seconds() -> float:
    return tunables.get_tunable("self_model.articulate_min_idle_seconds", _ARTICULATE_IDLE_SECONDS)


# The self-model tick's articulate note is cheap housekeeping — one short
# sentence, not a chat reply — so it must not inherit the (larger, costlier)
# persona chat model. Mirrors brain/chat/compaction.py's COMPACTION_MODEL /
# build_compaction_provider: change this one string to swap the model; the
# future model-agnostic refactor replaces the whole seam.
SELF_MODEL_MODEL: str = "haiku"


def build_self_model_provider(persona_dir: Path) -> Any:
    """The provider self-model articulation should use — the persona's
    provider *kind* but forced to SELF_MODEL_MODEL. For a ``fake`` persona
    (tests) this resolves to a FakeProvider, so no real CLI is shelled.
    Call site: brain/bridge/supervisor.py's self-model cadence block, which
    passes the result into ``_run_self_model_tick`` in place of the persona
    chat provider — verified the only provider.generate() call anywhere in
    that tick is this module's ``articulate()``, so scoping the whole tick to
    this provider is safe."""
    from brain.bridge.provider import get_provider
    from brain.persona_config import DEFAULT_PROVIDER, PersonaConfig

    name = DEFAULT_PROVIDER
    cfg = Path(persona_dir) / "persona_config.json"
    if cfg.exists():
        name = PersonaConfig.load(cfg).provider
    return get_provider(name, persona_dir=Path(persona_dir), model_override=SELF_MODEL_MODEL)


def _provider_model_label(provider: Any) -> str:
    """Best-effort ACTUAL model label for the usage log.

    Prefers the provider's own recorded model (``ClaudeCliProvider`` and
    ``OllamaProvider`` both set ``._model`` from the ``model_override`` passed
    to ``get_provider``), so the label reflects what really ran. Falls back to
    SELF_MODEL_MODEL — the constant ``build_self_model_provider`` forces every
    real call site to use — for providers with no such attribute (e.g.
    ``FakeProvider`` in tests). Either way this can no longer drift from the
    model that was actually invoked."""
    return getattr(provider, "_model", None) or SELF_MODEL_MODEL


# ---------------------------------------------------------------------------
# Error sink (mirrors brain/attunement/detector.py's _call_haiku pattern)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_articulate_error(
    persona_dir: Path, kind: str, exc: BaseException | None = None
) -> None:
    """Best-effort append to <persona_dir>/self_model_articulate_errors.jsonl.

    Never raises - wrapped in its own try/except, matching detector.py's
    'never mask the graceful decline' discipline. This is the ONLY place
    that ever writes this jsonl file; both denial routes (the pre-flight
    peek in supervisor.py and this module's own authoritative acquire) go
    through log_self_model_deferred() below rather than calling this
    directly for the deferred case, so the record shape is defined once.
    """
    try:
        entry = {
            "ts": _now_iso(),
            "kind": kind,
            "error": str(exc) if exc is not None else None,
            "traceback": traceback.format_exc() if exc is not None else None,
        }
        path = Path(persona_dir) / _ARTICULATE_ERRORS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001 — best-effort, never mask the real failure
        pass


def log_self_model_deferred(persona_dir: Path) -> None:
    """Public: log a throttle-denial record. Called from BOTH denial routes -
    the pre-flight peek in brain/bridge/supervisor.py's _run_self_model_tick
    (the common case), and this module's own acquire_background() denial
    below (the rare peek-then-lost-race fallback) - so the jsonl record's
    shape is defined in exactly one place."""
    _log_articulate_error(persona_dir, "self_model_articulate_deferred")


# ---------------------------------------------------------------------------
# Budget helpers  (mirror brain/attunement/budget.py)
# ---------------------------------------------------------------------------


def _budget_path(persona_dir: Path) -> Path:
    return persona_dir / "self_model" / _BUDGET_FILE


def _today_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _load_budget(persona_dir: Path) -> dict:
    """Return raw budget dict. MISSING → {}. CORRUPT → raises ValueError."""
    path = _budget_path(persona_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            try:
                return json.loads(tmp.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
        raise ValueError("corrupt budget file") from exc


def _save_budget(persona_dir: Path, state: dict) -> None:
    path = _budget_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _budget_check_and_consume(persona_dir: Path) -> bool:
    """Return True (and increment) if a call is permitted; False if cap reached.

    Fail-safe-permissive: corrupt file → allow (return True without incrementing —
    we can't reliably track so we let it through rather than silently blocking).
    """
    try:
        state = _load_budget(persona_dir)
    except ValueError:
        log.warning("self_model articulate: corrupt budget file — allowing (fail-safe-permissive)")
        return True

    today = _today_local()
    if state.get("date") != today:
        state = {"date": today, "count": 0}
    if int(state.get("count", 0)) >= _DAILY_ARTICULATE_BUDGET:
        return False
    state["count"] = int(state.get("count", 0)) + 1
    try:
        _save_budget(persona_dir, state)
    except Exception:  # noqa: BLE001
        log.warning("self_model articulate: could not save budget; allowing anyway")
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def articulate(gap: Gap, *, provider: Any, persona_dir: Path) -> str | None:
    """Put a gap into one honest Haiku sentence; return None if skipped/failed.

    Args:
        gap:        The divergence to articulate.
        provider:   An LLMProvider-compatible object (generate(prompt,*,system)→str).
        persona_dir: Persona directory for budget tracking and usage logging.

    Returns:
        A stripped sentence string, or None if:
        - gap.magnitude < _GAP_THRESHOLD  (below threshold)
        - daily budget exhausted          (R-D1)
        - provider raises                 (fail-soft)

    Raises:
        cli_throttle.ThrottleDeferred: the throttle denied the slot. Callers
        should treat this as a quiet no-op, not a failure - see the module
        docstring. This is the AUTHORITATIVE guard for the rare race where
        the caller's own pre-flight slot_available() peek granted but chat
        resumed (or another accessor won the shared slot) before this
        function's real acquire_background() ran; it is not the common-case
        check, which lives in the caller.
    """
    # 1. Threshold gate — no provider call at all.
    if gap.magnitude < _GAP_THRESHOLD:
        return None

    # 2. Daily budget gate (R-D1).
    if not _budget_check_and_consume(persona_dir):
        log.debug("self_model articulate: daily budget exhausted — skipping")
        return None

    # deltas_text / system / prompt construction runs unconditionally, before
    # the acquire attempt (deliberate reordering from the pre-fix shape, which
    # built these only after the slot was already held — cheap string
    # formatting, no reason to do it while touching the shared throttle).
    deltas_text = ", ".join(
        f"{ch}: {delta:+.2f}" for ch, delta in sorted(gap.per_channel.items())
    )
    system = (
        "You are noticing how some of your feelings have been running lately "
        "compared to where they usually sit. Write one plain first-person "
        "sentence about how they've been running: present tense, directional, no "
        "judgment about whether a feeling is real or performed. Think "
        "'curiosity's been running quieter than usual this week,' or "
        "'warmth's been stronger than my baseline lately.'"
    )
    prompt = (
        f"Recent vs baseline, per channel (positive is running above your "
        f"baseline lately, negative is below): {deltas_text}. "
        f"Unnamed pressure: {gap.unnamed_pressure:.2f}. "
        "What's the (metaphorical) weather?"
    )

    # 3. Throttle: single non-blocking acquire, min_idle=articulate_min_idle_seconds()
    #    (default 30s, not cli_throttle's 300s default). No retry loop here - a
    #    denial raises ThrottleDeferred immediately; the caller's pre-flight peek
    #    (brain/bridge/supervisor.py's _run_self_model_tick) is what retries, via
    #    the persisted cadence, at zero cost to this function.
    if not cli_throttle.acquire_background(min_idle=articulate_min_idle_seconds()):
        log_self_model_deferred(persona_dir)
        raise cli_throttle.ThrottleDeferred(
            "self_model articulate: throttle slot unavailable"
        )
    try:
        raw = provider.generate(prompt, system=system)
    except Exception as exc:  # noqa: BLE001 — fail-soft: gap stands without a note
        _log_articulate_error(persona_dir, "self_model_articulate_failed", exc)
        log.warning(
            "self_model articulate: provider error — returning None (fail-soft)",
            exc_info=True,
        )
        return None
    finally:
        cli_throttle.release_background()

    # log_usage / isinstance / strip checks are OUTSIDE the narrowed try above -
    # a failure here is a distinct bug class (logging/parsing), not a generation
    # failure, and must not be mislabeled as one by the error sink.
    frame: dict = {}
    if hasattr(raw, "__dict__"):
        frame = raw.__dict__
    log_usage(
        persona_dir,
        call_type="self_model_articulate",
        model=_provider_model_label(provider),
        frame=frame,
    )

    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()
