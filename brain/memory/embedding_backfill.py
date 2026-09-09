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

The cursor is a COMPOSITE ``(created_at, id)`` keyset position, not a bare
timestamp — see ``MemoryStore.list_active_since``. A bare-timestamp cursor
made any row sharing its exact ``created_at`` with the pinned row permanently
unreachable (bulk migrator imports routinely produce duplicate/second-
granularity timestamps — ``brain/migrator/emergence_kit.py`` via
``brain/migrator/transform.py``'s ``_coerce_utc``); pairing the timestamp
with the row's own ``id`` gives every row a distinct position in the scan
order, so a shared timestamp can never hide a row from the backfill. A
cursor file written before this change is timestamp-only and is treated as
unparseable — see ``_load_cursor`` — which resets to the top of history
rather than guessing; safe and cheap, since a rescan only re-confirms rows
already in ``embeddings.db`` via ``EmbeddingCache.has()``.

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


def _load_cursor(persona_dir, current_model_id: str) -> tuple[str, str] | None:  # noqa: ANN001
    """Best-effort cursor read. Missing/corrupt/model-mismatched/old-format
    -> None (start of history) — fail toward re-scanning, never toward
    silently skipping a row. See module docstring: the cursor is an
    optimization, not the correctness mechanism.

    The persisted cursor is the COMPOSITE ``{"created_at": ..., "id": ...}``
    form (see ``MemoryStore.list_active_since``). A cursor file written by a
    pre-keyset build of this module is timestamp-only (a bare string) — that
    old format is deliberately NOT half-interpreted (e.g. paired with an
    empty/sentinel id, which would silently reintroduce the same-timestamp
    blind spot this format exists to close); it is treated exactly like any
    other unparseable cursor and reset to ``None``, which just rescans from
    the top. Safe and cheap: ``EmbeddingCache.has()``/``get_or_compute`` are
    idempotent, so re-scanning only re-confirms rows already embedded.
    """
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
    if not isinstance(cursor, dict):
        # Includes the pre-keyset bare-string format, and None/missing.
        return None
    created_at = cursor.get("created_at")
    row_id = cursor.get("id")
    if not isinstance(created_at, str) or not created_at:
        return None
    if not isinstance(row_id, str) or not row_id:
        return None
    return (created_at, row_id)


def _save_cursor(
    persona_dir,  # noqa: ANN001
    current_model_id: str,
    cursor: tuple[str, str] | None,
) -> None:
    """Best-effort cursor write (temp file + rename). Failure is swallowed —
    mirrors persisted_cadence.save_cadence's posture: a failed save only
    means the NEXT tick re-scans a bit more than necessary, never that a
    row goes unembedded."""
    path = cadence_state_path(persona_dir, _CURSOR_FILE)
    cursor_payload = (
        {"created_at": cursor[0], "id": cursor[1]} if cursor is not None else None
    )
    payload = {"model_id": current_model_id, "cursor": cursor_payload}
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

    Fault-isolated WITHOUT starving later rows: a per-row failure (cache
    lookup or embed compute — e.g. a transient provider error, or a row that
    permanently trips a real embedding-runtime limit) is logged and the tick
    CONTINUES to the next candidate rather than stopping there. The
    persisted cursor still only advances up to the position just BEFORE the
    EARLIEST failed row this tick — even if later rows in the same batch
    embed successfully — so that row is retried first on the next tick
    (transient failures get their retry). But because scanning does not stop
    at the first failure, a row that fails on EVERY attempt (a permanent
    failure) never blocks the rows after it from being embedded — it simply
    never lets the cursor advance past itself, and is re-examined (and
    re-skipped-with-a-warning) every tick indefinitely. Rows already
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
    # Once a row fails, resolved_up_to must never advance again THIS tick —
    # a later success at a higher position must not skip the persisted
    # cursor past the earlier, still-unresolved failure.
    failed_this_tick = False

    for memory in candidates:
        scanned += 1
        row_cursor = (memory.created_at.isoformat(), memory.id)

        if len(memory.content) < MIN_CHARS_TO_EMBED:
            skipped_short += 1
            if not failed_this_tick:
                resolved_up_to = row_cursor
            continue

        try:
            cached = embeddings.has(memory.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding_backfill: cache lookup failed for memory %s: %s", memory.id, exc
            )
            errors += 1
            failed_this_tick = True
            continue  # keep scanning — a later row must not be starved

        if cached:
            already_cached += 1
            if not failed_this_tick:
                resolved_up_to = row_cursor
            continue

        if embedded >= batch_size:
            # Batch budget spent — leave this (and anything after it) for
            # next tick. This is ordinary pacing, not a failure, so it's a
            # clean stop rather than a continue.
            break

        try:
            embeddings.get_or_compute(memory.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding_backfill: embed failed for memory %s: %s", memory.id, exc
            )
            errors += 1
            failed_this_tick = True
            continue  # keep scanning — a later row must not be starved

        embedded += 1
        if not failed_this_tick:
            resolved_up_to = row_cursor

    _save_cursor(persona_dir, model_id, resolved_up_to)

    return BackfillTickResult(
        scanned=scanned,
        embedded=embedded,
        already_cached=already_cached,
        skipped_short=skipped_short,
        errors=errors,
    )
