"""Floor bootstraps, off the reply path (name-recall fix S85 revised, S91, S92;
spec §2).

The COSINE floor and the RERANK floor both have a derived bootstrap value that
serves until the daily tick persists a calibrated row. Neither is ever
computed while a reply is being built: `MemoryStore.get_cosine_floor` /
`get_reranker_floor` only PEEK the process caches. Instead:

* COSINE (S91): once per process, at process start, `compute_missing_floors`
  computes it if it is missing (no calibrated row, not yet cached). Two entry
  points call it: the bridge's startup background thread
  (`brain.bridge.server` lifespan) and `nell chat --no-bridge` session start
  (`brain.cli`), so the no-rerank path "works from the first message" except
  for the moments before this finishes;
* RERANK (S91): computed on FIRST NEED, in the background. A turn that would
  rerank but finds no rerank floor takes the cosine path for that turn and
  calls `request_rerank_bootstrap`, which flags the need and starts the
  bootstrap on a daemon thread (never on the reply path). Nothing loads the
  reranker at boot for a session that never reranks;
* RETRY (S92): a failed bootstrap (either floor) is retried in the background,
  off the reply path, on each incoming chat message until it succeeds
  (`on_incoming_message`, called by `respond()`), and also by the central
  cadence jobs `cosine_floor_bootstrap` / `rerank_floor_bootstrap` at the next
  lull (due again only after chat activity since the failure). No time
  constants;
* never more than ONE bootstrap in flight per floor: every start (startup
  thread, first-need request, message retry, cadence job) goes through
  `try_begin(kind)` / `end(kind)`.

`compute_missing_floors`, `request_rerank_bootstrap` and `on_incoming_message`
never raise: a failure is logged and recorded as a failed attempt. The
`run_*_floor` helpers are fail-soft too; only their tiny id lookups could raise,
and every caller sits inside a guard.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from brain.memory import floor_calibration

logger = logging.getLogger(__name__)

# Kinds ("cosine" / "rerank") with a bootstrap in flight in this process, from
# ANY starter: the startup thread, the first-need request, the message retry or
# the cadence job. One at a time per floor (S91/S92).
_inflight: set[str] = set()
_inflight_lock = threading.Lock()

# Test seam: True stops `request_rerank_bootstrap` / `on_incoming_message` from
# spawning threads (the flag/failure bookkeeping still happens). The test
# suite's autouse fixture sets it so a recall or a `respond()` in an unrelated
# test never leaks a background thread; production never touches it.
_background_inhibited = False

# The chat-activity marker the background starters record on a failure (the
# cadence job's retry rule compares it); set at process start by the bridge and
# the direct chat. Default: no marker.
def _no_marker() -> object:
    return None


_marker_provider: Callable[[], object] = _no_marker


def set_activity_marker_provider(provider: Callable[[], object]) -> None:
    """Register the chat-activity marker source (`cli_throttle.chat_activity_marker`)."""
    global _marker_provider
    _marker_provider = provider


def _current_marker() -> object:
    try:
        return _marker_provider()
    except Exception:  # noqa: BLE001
        return None


def try_begin(kind: str) -> bool:
    """Claim the single in-flight slot of the `kind` floor; False when taken."""
    with _inflight_lock:
        if kind in _inflight:
            return False
        _inflight.add(kind)
        return True


def end(kind: str) -> None:
    """Release the `kind` floor's in-flight slot."""
    with _inflight_lock:
        _inflight.discard(kind)


def inflight(kind: str) -> bool:
    """True while a `kind` bootstrap is running in this process."""
    with _inflight_lock:
        return kind in _inflight


def startup_compute_active() -> bool:
    """True while the cosine bootstrap is in flight (kept for the cadence job)."""
    return inflight("cosine")


def embedder_model_id() -> str:
    """The embedder id the cosine bootstrap and the recall lookup key on: the
    tier's model string, which is also what `FastEmbedProvider.model_id()`
    returns in production (pinned by a test)."""
    from brain.bridge.model_tier import TIER_EMBEDDING, model_for_tier

    return model_for_tier(TIER_EMBEDDING)


def reranker_model_id() -> str:
    """The runtime reranker id the rerank bootstrap and the recall lookup key
    on (`reranker.resolve_reranker_model_id`): cheap, loads nothing."""
    from brain.memory.reranker import resolve_reranker_model_id

    return resolve_reranker_model_id()


def cosine_floor_due(store, *, activity_marker: object = None) -> bool:
    """A cosine bootstrap is worth (re)computing: no calibrated row, not cached
    in this process, and not already tried since the last chat activity."""
    model_id = embedder_model_id()
    if not floor_calibration.cosine_bootstrap_due(model_id, activity_marker=activity_marker):
        return False
    return store.get_persisted_cosine_floor(model_id) is None


def rerank_floor_due(store, *, activity_marker: object = None) -> bool:
    """A rerank bootstrap is worth (re)computing (same rule as the cosine one)."""
    model_id = reranker_model_id()
    if not floor_calibration.rerank_bootstrap_due(model_id, activity_marker=activity_marker):
        return False
    return store.get_persisted_reranker_floor(model_id) is None


def run_cosine_floor(*, activity_marker: object = None) -> dict | None:
    """Compute the cosine bootstrap for the runtime embedder (never raises)."""
    return floor_calibration.run_cosine_bootstrap(
        embedder_model_id(), activity_marker=activity_marker
    )


def run_rerank_floor(*, activity_marker: object = None) -> dict | None:
    """Compute the rerank bootstrap for the runtime reranker (never raises).

    `reranker.build_reranker_provider()` runs first: it registers the fp16
    model and caches the provider for the runtime model id, which the bootstrap
    (`_bootstrap_reranker_provider`) then reuses."""
    model_id = reranker_model_id()
    try:
        from brain.memory import reranker as reranker_mod

        reranker_mod.build_reranker_provider()
    except Exception:  # noqa: BLE001 — recorded as a failed attempt below
        logger.exception("floor_startup: reranker provider build failed for %s", model_id)
        floor_calibration._record_attempt(  # noqa: SLF001
            "rerank", model_id, False, activity_marker
        )
        return None
    return floor_calibration.run_rerank_bootstrap(model_id, activity_marker=activity_marker)


def run_guarded(kind: str, run, *, activity_marker: object = None) -> bool:
    """Run `run(activity_marker=...)` under the `kind` in-flight slot; False
    (nothing run) when another bootstrap of that floor is already in flight."""
    if not try_begin(kind):
        return False
    try:
        run(activity_marker=activity_marker)
    finally:
        end(kind)
    return True


def compute_missing_floors(
    persona_dir: Path,
    *,
    activity_marker: Callable[[], object] = lambda: None,
    _registered: bool = False,
) -> None:
    """The process-start computation (S91): the COSINE bootstrap, if it is
    missing. Opens its own short-lived store to read the calibrated row and
    closes it before any model loads. Never raises. `_registered` = the caller
    already holds the cosine in-flight slot (`start_background`)."""
    from brain.memory.store import MemoryStore

    if not _registered and not try_begin("cosine"):
        return
    try:
        try:
            store = MemoryStore(Path(persona_dir) / "memories.db", integrity_check=False)
        except Exception:  # noqa: BLE001
            logger.exception("floor_startup: could not open the store; floor not computed")
            return
        try:
            cosine_needed = cosine_floor_due(store, activity_marker=activity_marker())
        except Exception:  # noqa: BLE001
            logger.exception("floor_startup: due check failed; floor not computed")
            return
        finally:
            store.close()
        if cosine_needed:
            try:
                run_cosine_floor(activity_marker=activity_marker())
            except Exception:  # noqa: BLE001
                logger.exception("floor_startup: the cosine bootstrap raised")
    finally:
        end("cosine")


def start_background(
    persona_dir: Path,
    *,
    activity_marker: Callable[[], object] = lambda: None,
    name: str = "floor-bootstrap",
) -> threading.Thread | None:
    """Run `compute_missing_floors` on a daemon thread and return it (started),
    or `None` when a cosine bootstrap is already in flight. Daemon: never holds
    up a shutdown; a process that exits mid-compute simply computes again next
    start. Registers the process's activity-marker source for the background
    starters."""
    set_activity_marker_provider(activity_marker)
    # Claim the slot BEFORE the thread exists, so the cadence job can never win
    # a race against the startup computation (`compute_missing_floors` ends it).
    if not try_begin("cosine"):
        return None
    thread = threading.Thread(
        target=compute_missing_floors,
        args=(persona_dir,),
        kwargs={"activity_marker": activity_marker, "_registered": True},
        name=name,
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        end("cosine")
        raise
    return thread


def _spawn(kind: str, body: Callable[[], object], *, name: str) -> threading.Thread | None:
    """Start `body` on a daemon thread under the `kind` in-flight slot; `None`
    when inhibited (tests) or another `kind` bootstrap is already in flight."""
    if _background_inhibited or not try_begin(kind):
        return None

    def _run() -> None:
        try:
            body()
        except Exception:  # noqa: BLE001 — a background bootstrap never raises into the void
            logger.exception("floor_startup: background %s bootstrap raised", kind)
        finally:
            end(kind)

    thread = threading.Thread(target=_run, name=name, daemon=True)
    try:
        thread.start()
    except Exception:
        end(kind)
        raise
    return thread


def request_rerank_bootstrap() -> threading.Thread | None:
    """First-need request (S91): a turn that would rerank found no rerank floor.
    Flags the need and starts the rerank bootstrap in the background, one at a
    time, and returns immediately (never on the reply path). Returns the thread
    started, or `None` (already in flight, or inhibited). Never raises."""
    try:
        floor_calibration.note_bootstrap_needed("rerank", reranker_model_id())
        return _spawn(
            "rerank",
            lambda: run_rerank_floor(activity_marker=_current_marker()),
            name="floor-bootstrap-rerank",
        )
    except Exception:  # noqa: BLE001
        logger.exception("floor_startup: could not request the rerank bootstrap")
        return None


def _retry_body(kind: str, persona_dir: Path) -> None:
    """The message-retry body: skip when a calibrated row exists (the floor then
    exists and the failed record is moot), else run the bootstrap."""
    from brain.memory.store import MemoryStore

    store = MemoryStore(Path(persona_dir) / "memories.db", integrity_check=False)
    try:
        if kind == "cosine":
            calibrated = store.get_persisted_cosine_floor(embedder_model_id()) is not None
        else:
            calibrated = store.get_persisted_reranker_floor(reranker_model_id()) is not None
    finally:
        store.close()
    if calibrated:
        return
    run = run_cosine_floor if kind == "cosine" else run_rerank_floor
    run(activity_marker=_current_marker())


def on_incoming_message(persona_dir: Path) -> list[threading.Thread]:
    """S92: called for each incoming chat message (`respond()`). For each floor
    whose last bootstrap FAILED and that has no floor yet, retry it in the
    background, off the reply path, unless one is already in flight for that
    floor. No time constants: the message itself is the trigger, and it repeats
    on every message until a bootstrap succeeds. Returns the threads started.
    Never raises, never blocks."""
    started: list[threading.Thread] = []
    try:
        for kind, model_id in (("cosine", embedder_model_id()), ("rerank", reranker_model_id())):
            if not floor_calibration.bootstrap_failed(kind, model_id):
                continue
            thread = _spawn(
                kind,
                lambda kind=kind: _retry_body(kind, persona_dir),
                name=f"floor-bootstrap-{kind}-retry",
            )
            if thread is not None:
                started.append(thread)
    except Exception:  # noqa: BLE001
        logger.exception("floor_startup: on-message bootstrap retry failed")
    return started
