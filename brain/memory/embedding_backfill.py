"""Idle-chipped embedding backfill.

Stage 2 of the local semantic-retrieval build (companion-emergence). Most
memories are committed via ``MemoryStore``'s ``create`` method at ~11 call sites
(``brain/tools/impls/add_memory.py``, ``crystallize_soul.py``,
``add_journal.py``, ``brain/recovery/engine.py``,
``brain/kindled_link/relationship.py``, ``brain/migrator/cli.py``,
``brain/body/events.py``, ``brain/soul/review.py``,
``brain/engines/consolidation.py``, ``brain/migrator/emergence_kit.py``,
``brain/grief/breadcrumb.py``, ``brain/memory/pending.py``) that never touch
``EmbeddingCache`` at all — see ``hunts/semantic-retrieval/plan.md`` Part A
#7. Only the ingest pipeline (``brain/ingest/pipeline.py``, via dedupe's
``get_or_compute`` side effect) embeds a memory as a side effect of writing
it. Rather than instrumenting all ~11 sites (invasive, easy to miss a 12th
later — see this module's sibling decision in the Stage 2 report), this
backfill is the single source of truth: it scans ``memories`` on every
supervisor tick and embeds whatever hasn't been covered yet, ingest-path or
not.

Resumable/idempotent by construction: "backlog" is defined directly against
the ``(content_hash, model_id)`` cache (``EmbeddingCache.has``), never a
row's mere presence in a cursor window — a row embedded by a prior (possibly
interrupted) tick is simply absent from the next tick's work. Killing the
process mid-batch loses nothing: everything embedded before the kill is
already committed to ``embeddings.db`` (``get_or_compute`` commits per row);
the next tick's scan just resumes. A persisted cursor (this persona's
``cadence/embedding_backfill_cursor.json``) is a pure scan-cost optimization
— it lets a large corpus avoid re-walking its already-resolved prefix every
tick — NOT the source of correctness; a missing/corrupt/stale cursor file
just means the next tick rescans more than strictly necessary, never that a
row is skipped. The cursor resets automatically on a model swap (its
``model_id`` no longer matches the cache's), so a new model reopens the
whole backlog rather than silently under-covering it.

Stays off the message hot path: ``run_embedding_backfill_tick`` is only
ever called from the supervisor's own per-tick maintenance block
(``brain/bridge/supervisor.py``), never from a chat-turn code path — no
different from the ``embeddings = build_embedding_cache(persona_dir)``
handle that block already opens for ``snapshot_stale_sessions``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from brain.memory.embeddings import EmbeddingCache
from brain.memory.store import MemoryStore
from brain.paths import cadence_state_path

logger = logging.getLogger(__name__)

# Skip very short/low-content rows — a handful of words produces a noisy
# vector that doesn't meaningfully distinguish itself from anything else.
# Mirrors the REASONING behind brain/memory/relevance.py's
# SNIPPET_MIN_CHARS=20 (a different concern — render-snippet floor, not
# embed-worthiness — so kept as its own constant rather than imported, so
# the two thresholds can diverge independently later).
MIN_CHARS_TO_EMBED = 20

# Bounded batch per tick so even a large cold-start backlog only chips away
# a little bit per tick rather than spiking CPU. Two independent caps:
# BATCH_SIZE bounds actual embed *compute* (the expensive part — ~34ms/call
# per the spec's own empirical measurement on this class of hardware);
# SCAN_CAP bounds how many candidate rows are even *examined* this tick
# (cheap: an indexed cache lookup per row), so a corpus with a long run of
# skip-worthy short rows can't spin CPU scanning without also embedding
# anything.
DEFAULT_BATCH_SIZE = 25
DEFAULT_SCAN_CAP = 500

_CURSOR_FILE = "embedding_backfill_cursor.json"


@dataclass(frozen=True)
class BackfillTickResult:
    """What one tick of the backfill accomplished — for logging/tests."""

    scanned: int  # candidate rows examined this tick
    embedded: int  # rows newly embedded (real compute, not a cache hit)
    already_cached: int  # candidates that turned out already embedded
    skipped_short: int  # rows skipped for being under MIN_CHARS_TO_EMBED
    errors: int  # embed attempts that raised (left in the backlog for retry)


def _load_cursor(persona_dir, current_model_id: str) -> str | None:  # noqa: ANN001
    """Best-effort cursor read. Missing/corrupt/model-mismatched -> None
    (start of history) — fail toward re-scanning, never toward silently
    skipping a row. See module docstring: the cursor is an optimization,
    not the correctness mechanism."""
    path = cadence_state_path(persona_dir, _CURSOR_FILE)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("model_id") != current_model_id:
        # A model swap (or first run under this model) invalidates any prior
        # cursor position — the whole backlog needs re-examining under the
        # new model_id, since embedding_cache rows are scoped to model_id.
        return None
    cursor = raw.get("cursor")
    return cursor if isinstance(cursor, str) and cursor else None


def _save_cursor(persona_dir, current_model_id: str, cursor: str | None) -> None:  # noqa: ANN001
    """Best-effort cursor write (temp file + rename). Failure is swallowed —
    mirrors persisted_cadence.save_cadence's posture: a failed save only
    means the NEXT tick re-scans a bit more than necessary, never that a
    row goes unembedded."""
    path = cadence_state_path(persona_dir, _CURSOR_FILE)
    payload = {"model_id": current_model_id, "cursor": cursor}
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.warning("embedding_backfill: could not persist cursor (best-effort)", exc_info=True)


def run_embedding_backfill_tick(
    persona_dir,  # noqa: ANN001 — Path, kept untyped to avoid importing pathlib just for the hint here
    store: MemoryStore,
    embeddings: EmbeddingCache,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    scan_cap: int = DEFAULT_SCAN_CAP,
) -> BackfillTickResult:
    """Embed up to `batch_size` un-embedded active memories, off the hot path.

    Intended to be called once per supervisor tick (see module docstring).
    Never performs more than `batch_size` real embed computations, so even a
    large cold-start backlog only chips away a little bit per call rather
    than spiking CPU; `scan_cap` separately bounds how many rows are even
    read from `memories` this call.

    Resumable/idempotent: a row counts as backlog iff it is missing from
    `embeddings` under the cache's OWN model_id (`EmbeddingCache.has`) — the
    exact key `get_or_compute` reads/writes — so a row embedded by a prior
    (possibly killed) tick, or by the ingest pipeline's own embed-on-write
    side effect, is simply skipped here, never re-embedded or duplicated.

    Fault-isolated: an embed failure (e.g. a transient provider error) is
    logged and the tick stops advancing there — the persisted cursor is
    pinned just before the failing row, so THAT row is retried first on the
    next tick rather than being silently skipped forever. Rows already
    resolved earlier in the same tick (embedded or found already-cached)
    keep their progress either way, since `has()` reflects them regardless
    of where the cursor sits.
    """
    model_id = embeddings.model_id
    cursor = _load_cursor(persona_dir, model_id)
    candidates = store.list_active_since(cursor, limit=scan_cap)

    scanned = 0
    embedded = 0
    already_cached = 0
    skipped_short = 0
    errors = 0
    resolved_up_to = cursor  # last row the cursor can safely advance past

    for memory in candidates:
        scanned += 1

        if len(memory.content) < MIN_CHARS_TO_EMBED:
            skipped_short += 1
            resolved_up_to = memory.created_at.isoformat()
            continue

        try:
            cached = embeddings.has(memory.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding_backfill: cache lookup failed for memory %s: %s", memory.id, exc
            )
            errors += 1
            break  # stop here this tick; cursor stays pinned before this row

        if cached:
            already_cached += 1
            resolved_up_to = memory.created_at.isoformat()
            continue

        if embedded >= batch_size:
            # Batch budget spent — leave this (and anything after it) for
            # next tick. Cursor stays pinned before this row.
            break

        try:
            embeddings.get_or_compute(memory.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding_backfill: embed failed for memory %s: %s", memory.id, exc
            )
            errors += 1
            break  # stop here this tick; cursor stays pinned before this row

        embedded += 1
        resolved_up_to = memory.created_at.isoformat()

    _save_cursor(persona_dir, model_id, resolved_up_to)

    return BackfillTickResult(
        scanned=scanned,
        embedded=embedded,
        already_cached=already_cached,
        skipped_short=skipped_short,
        errors=errors,
    )
