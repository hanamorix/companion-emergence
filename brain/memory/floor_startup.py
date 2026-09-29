"""Floor bootstrap at process start, off the reply path (name-recall fix S85,
revised; spec §2).

The COSINE floor and the RERANK floor both have a derived bootstrap value that
serves until the daily tick persists a calibrated row. Neither is ever
computed while a reply is being built: `MemoryStore.get_cosine_floor` /
`get_reranker_floor` only PEEK the process caches. Instead:

* once per process, at process start, `compute_missing_floors` computes every
  bootstrap that is missing (no calibrated row, not yet cached), the cosine one
  first, then the rerank one, one after the other (this module never loads
  both models concurrently; both then stay resident in their provider caches). Two entry points call it: the bridge's startup
  background thread (`brain.bridge.server` lifespan) and `nell chat
  --no-bridge` session start (`brain.cli`), so recall "works from the first
  message" except for the moments before this finishes;
* a failed bootstrap stays missing (that path renders keyword results only),
  and the central cadence jobs `cosine_floor_bootstrap` /
  `rerank_floor_bootstrap` retry it at the next lull: due again only after chat
  activity has happened since the failure (no time constants).

`compute_missing_floors` never raises: a failure is logged and recorded as a
failed attempt (retried at the next lull). The `run_*_floor` helpers are
fail-soft too; only their tiny id lookups (`embedder_model_id`,
`reranker_model_id`) could raise, and every caller sits inside a guard
(`compute_missing_floors`, the cadence pass).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from brain.memory import floor_calibration

logger = logging.getLogger(__name__)

# Set while `compute_missing_floors` runs in this process, so the cadence jobs
# do not queue up behind a startup computation that is already doing their work.
_startup_active = threading.Event()


def startup_compute_active() -> bool:
    """True while the process-start floor computation is running."""
    return _startup_active.is_set()


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


def compute_missing_floors(
    persona_dir: Path,
    *,
    activity_marker: Callable[[], object] = lambda: None,
) -> None:
    """The process-start computation: every missing bootstrap floor, cosine
    then rerank, sequentially. Opens its own short-lived store to read the
    calibrated rows and closes it before any model loads. Never raises."""
    from brain.memory.store import MemoryStore

    _startup_active.set()
    try:
        try:
            store = MemoryStore(Path(persona_dir) / "memories.db", integrity_check=False)
        except Exception:  # noqa: BLE001
            logger.exception("floor_startup: could not open the store; floors not computed")
            return
        try:
            cosine_needed = cosine_floor_due(store, activity_marker=activity_marker())
            rerank_needed = rerank_floor_due(store, activity_marker=activity_marker())
        except Exception:  # noqa: BLE001
            logger.exception("floor_startup: due check failed; floors not computed")
            return
        finally:
            store.close()
        for needed, run in ((cosine_needed, run_cosine_floor), (rerank_needed, run_rerank_floor)):
            if not needed:
                continue
            try:
                run(activity_marker=activity_marker())
            except Exception:  # noqa: BLE001 — one floor failing must not stop the other
                logger.exception("floor_startup: %s raised", run.__name__)
    finally:
        _startup_active.clear()


def start_background(
    persona_dir: Path,
    *,
    activity_marker: Callable[[], object] = lambda: None,
    name: str = "floor-bootstrap",
) -> threading.Thread:
    """Run `compute_missing_floors` on a daemon thread and return it (started).
    Daemon: never holds up a shutdown; a process that exits mid-compute simply
    computes again next start."""
    # Set the flag BEFORE the thread exists, so the cadence job can never win a
    # race against the startup computation (`compute_missing_floors` clears it).
    _startup_active.set()
    thread = threading.Thread(
        target=compute_missing_floors,
        args=(persona_dir,),
        kwargs={"activity_marker": activity_marker},
        name=name,
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        _startup_active.clear()
        raise
    return thread
