"""Idle-chipped embedding backfill (F1 #259 increment 3 rewrite).

Most memories are committed via ``MemoryStore``'s ``create`` method at ~11
call sites (``brain/tools/impls/add_memory.py``, ``crystallize_soul.py``,
``add_journal.py``, ``brain/recovery/engine.py``,
``brain/kindled_link/relationship.py``, ``brain/migrator/cli.py``,
``brain/body/events.py``, ``brain/soul/review.py``,
``brain/engines/consolidation.py``, ``brain/migrator/emergence_kit.py``,
``brain/grief/breadcrumb.py``, ``brain/memory/pending.py``) that never write
an embedding as a side effect. Only ``brain.engines.consolidation``'s
promote branch does (F1 step 4, embed-on-write at pending-queue -> committed
promotion — see ``MemoryStore.embed_row``). This backfill is the single
source of truth for everything else: it scans ``memories`` on every eligible
supervisor tick and embeds whatever hasn't been covered yet, ingest-path or
not.

BACKLOG DEFINITION (F1 #259 increment 3; model-mismatch clause folded in on
Fixing's inc3 spec-gap, Planning-ruled 2026-09-16): an active memory row with
`embedding IS NULL` OR `embedding_model_id != <current model_id>`, and
`length(content) >= MIN_CHARS_TO_EMBED` — all three filters applied straight
in SQL via ``MemoryStore.list_unembedded_since``, not a content-hash side
cache. This replaced the old ``embeddings.db`` / ``EmbeddingCache.has()``
check: identity is now the memory's own id, not its content hash, so
"already embedded" literally means "this row already carries a vector under
the CURRENT model." A row that goes NULL again after a content mutation
(`fade`/`update(content=...)`/`unfade`'s synchronous re-embed failing — see
``MemoryStore._reembed_or_clear``) simply reappears in the backlog query on
its own; no separate invalidation bookkeeping is needed the way the
content-hash cache required. Likewise, the `embedding_model_id !=` clause
means a `MODEL_EMBEDDING` swap reopens every old-model row for this same
backlog query automatically — no separate re-embed path needed — restoring
the model-scoped self-healing the old content-hash cache had (without this
clause, old-model rows stay non-NULL and the warm matrix filters them out by
model_id, so they'd fall to lexical recall forever). It's a no-op in steady
state (every row already matches the current model).

Each embedded row is written via ``MemoryStore.embed_row``, which persists
`embedding` + `embedding_model_id` on the row AND pushes the vector into the
process's warm ``EmbeddingMatrix`` (``brain.memory.embedding_matrix.
build_embedding_matrix(store.db_path)``) so it is immediately recall-visible
without waiting for a lazy matrix rebuild.

RUNTIME-DERIVED BATCH SIZE (replaces the old hardcoded `DEFAULT_BATCH_SIZE =
25`): the number of real embed computations one tick performs self-derives
from a measured WARM per-embed time on the actual host vs a time budget
(`floor(batch_budget_seconds / per_embed_seconds)`) — see `_get_batch_size`
below, which mirrors ``brain/memory/reranker.py``'s auto-scaling rerank
width. Measured ONCE per process (per model_id) and cached for the process
lifetime — no periodic recompute (approved F1 spec §3/S10): unlike rerank
width, hardware doesn't meaningfully drift mid-process here, and this tick
runs unattended on the supervisor thread where a recompute would just be
extra embed-provider calls for no real gain.

IDLE-GATED: this module's own ``run_embedding_backfill_tick`` does NOT gate
itself — the supervisor call site (``brain/bridge/supervisor.py``) wraps the
call in ``cli_throttle.background_slot()``, mirroring the maintenance +
interest-sweep cadences in that file. Before increment 3 the backfill ran on
every base tick unconditionally, unlike every other background maintenance
cadence — a regression from intent this closes.

Resumable/idempotent by construction: because backlog membership is the
row's own `embedding`/`embedding_model_id` columns (see BACKLOG DEFINITION
above), a row (re-)embedded under the current model by a prior (possibly
interrupted) tick is simply absent from the next tick's candidates — killing
the process mid-batch loses nothing (`embed_row` commits per row). A
persisted cursor (this persona's ``cadence/embedding_backfill_cursor.json``)
is a scan-cost optimization for a LARGE backlog — it lets a tick skip
straight past a prefix already known not to be backlog, rather than
re-querying the same window from the top every time — but it is deliberately
NOT trusted once a tick's query returns fewer rows than `scan_cap` (i.e. the
whole currently-null backlog fit in one scan): see the comment at the bottom
of `run_embedding_backfill_tick` for why a forward cursor is only safe while
the backlog is larger than one scan window. A missing/corrupt/stale cursor
file just means the next tick rescans more than strictly necessary, never
that a row is skipped. The cursor resets automatically on a model swap (its
persisted `model_id` no longer matches the current provider's), so a new
model reopens the whole backlog rather than silently under-covering it.

The cursor is a COMPOSITE ``(created_at, id)`` keyset position, not a bare
timestamp — see ``MemoryStore.list_active_since``/``list_unembedded_since``.
A bare-timestamp cursor made any row sharing its exact ``created_at`` with
the pinned row permanently unreachable (bulk migrator imports routinely
produce duplicate/second-granularity timestamps —
``brain/migrator/emergence_kit.py`` via ``brain/migrator/transform.py``'s
``_coerce_utc``); pairing the timestamp with the row's own ``id`` gives
every row a distinct position in the scan order. A cursor file written
before this pairing existed is timestamp-only and is treated as unparseable
— see ``_load_cursor`` — which resets to the top of history rather than
guessing; safe and cheap, since a rescan only re-confirms rows already
embedded (they no longer satisfy `embedding IS NULL`).

CURSOR-FREEZE FIX (folded into increment 3): a row that fails to embed is
logged and the cursor is allowed to advance PAST it (skip-and-log), instead
of the old behavior where a permanently-failing row froze the cursor and
later ticks re-scanned the same `scan_cap` window forever (rows beyond it
never reached). One bad memory can no longer stall the backlog.

Stays off the message hot path: ``run_embedding_backfill_tick`` is only ever
called from the supervisor's own per-tick maintenance block, never from a
chat-turn code path.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass

from brain import tunables
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

# Bounds how many candidate rows are even *examined* (read from `memories`)
# in one tick — cheap (an indexed WHERE-clause row), independent of
# `batch_size` which bounds actual embed *compute* (the expensive part).
# Keeps a corpus with a long run of skip-worthy short/failing rows from
# spinning CPU scanning without also embedding anything.
DEFAULT_SCAN_CAP = 500

_CURSOR_FILE = "embedding_backfill_cursor.json"

# ---------------------------------------------------------------------------
# Runtime-derived batch size (F1 #259 increment 3) — see module docstring.
# ---------------------------------------------------------------------------

# I7: the tick's own time budget and the assumed supervisor tick cadence are
# ops tunables, not inline literals — registered like the existing
# `throttle.*` keys (see brain/bridge/cli_throttle.py). `batch_budget_seconds`
# leaves ~half of `tick_interval_seconds` as headroom so a slow entry (or a
# tick that ran a little long already) still fits inside one supervisor tick.
_BATCH_BUDGET_SECONDS_DEFAULT = tunables.register(
    "embedding_backfill.batch_budget_seconds", 30.0
)
_TICK_INTERVAL_SECONDS_DEFAULT = tunables.register(
    "embedding_backfill.tick_interval_seconds", 60.0
)


def _batch_budget_seconds() -> float:
    return tunables.get_tunable(
        "embedding_backfill.batch_budget_seconds", _BATCH_BUDGET_SECONDS_DEFAULT
    )


def _tick_interval_seconds() -> float:
    return tunables.get_tunable(
        "embedding_backfill.tick_interval_seconds", _TICK_INTERVAL_SECONDS_DEFAULT
    )


# Cold-cache timing trap ([[single-shot-timing-cold-cache-trap]]): the first
# embed() call on a freshly-constructed provider pays model/ONNX-session
# warm-up cost far above steady-state — discard this many calls before
# starting to time (mirrors reranker.py's _WARMUP_RERANKS).
_WARMUP_EMBEDS = 2
# Average over this many WARM calls (post-discard) rather than trusting a
# single noisy sample (mirrors reranker.py's _MEASURE_RERANKS).
_MEASURE_EMBEDS = 5

# Fixed calibration text, sized like a typical memory rather than a token
# stub, so the measured per-embed figure reflects real embedding cost
# (mirrors reranker.py's _MEASURE_DOCUMENT rationale).
_MEASURE_TEXT = (
    "a representative memory passage, sized similarly to a typical corpus "
    "entry, used only to measure warm per-embed compute time on this host "
    "so the derived backfill batch size reflects real embedding cost rather "
    "than a short placeholder"
)

# Fallback ONLY if a measured per-embed time is degenerate (<= 0s — a
# clock/measurement anomaly, never the normal path). Small and conservative
# rather than an unbounded guess.
_FALLBACK_BATCH_SIZE = 25
_MIN_BATCH_SIZE = 1

# Floor for the MEASURED mean per-embed time (F1 #259 increment-3 red-team
# fix, F2): a fluke-fast measurement (near-zero — a clock-resolution
# artifact, or an unrealistically fast provider) would otherwise divide
# `budget` by a near-zero number and derive an absurd batch size. A real
# warm ONNX embed call takes on the order of single-digit milliseconds at
# the very fastest on real hardware, so anything measured below this floor
# is treated as a measurement artifact, not a genuine host capability, and
# clamped up to it before deriving the batch.
_MIN_PLAUSIBLE_PER_EMBED_SECONDS = 0.001

# model_id -> derived batch size. Process-wide, mirrors
# build_embedding_provider's own per-model_id cache — one measurement per
# model_id, shared across every tick in the process.
_batch_size_cache: dict[str, int] = {}
_batch_size_cache_lock = threading.Lock()
_warned_no_headroom = False


def _measure_per_embed_seconds(provider) -> float:  # noqa: ANN001
    """Mean WARM per-embed seconds for `provider`, after discarding
    `_WARMUP_EMBEDS` cold calls, floored at `_MIN_PLAUSIBLE_PER_EMBED_SECONDS`
    so a fluke-fast (near-zero) measurement can't drive an absurd derived
    batch size (F1 #259 increment-3 red-team fix, F2). Isolated as its own
    function (rather than inlined into `_get_batch_size`) so a test can
    monkeypatch/measure it directly without needing a real or artificially-
    timed provider."""
    for _ in range(_WARMUP_EMBEDS):
        provider.embed(_MEASURE_TEXT)
    samples: list[float] = []
    for _ in range(_MEASURE_EMBEDS):
        start = time.monotonic()
        provider.embed(_MEASURE_TEXT)
        samples.append(time.monotonic() - start)
    mean = sum(samples) / len(samples)
    return max(mean, _MIN_PLAUSIBLE_PER_EMBED_SECONDS)


def _derive_batch_size(per_embed_seconds: float, scan_cap: int) -> int:
    """`floor(batch_budget_seconds / per_embed_seconds)`, clamped to >= 1
    and to <= `scan_cap` (F1 #259 increment-3 red-team fix, F2): a derived
    batch can never usefully exceed the number of rows one tick even scans
    (`candidates` is itself `LIMIT scan_cap`), so an upper clamp keeps the
    reported/cached figure sane even before the per-embed-time floor in
    `_measure_per_embed_seconds` above is considered — belt-and-braces
    against a fluke-fast measurement, while a fluke-SLOW measurement is
    already kept reasonable (>= 1) by the existing low clamp below.

    Logs once (per process) if the configured budget leaves no headroom
    inside the assumed tick interval — a configuration smell, not a fatal
    error, so this never raises.
    """
    global _warned_no_headroom
    budget = _batch_budget_seconds()
    tick = _tick_interval_seconds()
    if budget >= tick and not _warned_no_headroom:
        _warned_no_headroom = True
        logger.warning(
            "embedding_backfill: batch_budget_seconds (%.1f) >= "
            "tick_interval_seconds (%.1f) — the derived batch leaves no "
            "headroom inside the tick",
            budget,
            tick,
        )
    if per_embed_seconds <= 0.0:
        return max(_MIN_BATCH_SIZE, min(_FALLBACK_BATCH_SIZE, scan_cap))
    derived = math.floor(budget / per_embed_seconds)
    return max(_MIN_BATCH_SIZE, min(derived, scan_cap))


def _get_batch_size(provider, scan_cap: int) -> int:  # noqa: ANN001
    """Cached-once-per-process derived batch size for `provider`'s
    model_id. Measures OFF-lock (a real embed call can take real time —
    must not serialize concurrent callers behind it) and caches with
    first-writer-wins (`setdefault`) so a measurement race between two
    threads on the very first call never lets a later, possibly-noisier
    measurement overwrite an already-cached figure. Mirrors
    reranker.py's `_warm_per_doc_latency`, minus its periodic-recompute
    machinery — see module docstring for why this measures only once.

    `scan_cap` is folded into the cached figure via `_derive_batch_size`'s
    upper clamp: in production `run_embedding_backfill_tick` is always
    called with the same (default) `scan_cap`, so this is a stable bound;
    a caller that varied `scan_cap` across calls under the same model_id
    would get the clamp from whichever call populated the cache first —
    the same "measured/derived once per process" policy the per-embed
    timing itself already follows.
    """
    model_id = provider.model_id()
    with _batch_size_cache_lock:
        cached = _batch_size_cache.get(model_id)
        if cached is not None:
            return cached

    per_embed = _measure_per_embed_seconds(provider)
    batch = _derive_batch_size(per_embed, scan_cap)

    with _batch_size_cache_lock:
        _batch_size_cache.setdefault(model_id, batch)
        return _batch_size_cache[model_id]


def _reset_batch_size_cache() -> None:
    """Test-only: clear the measured/derived batch-size cache."""
    global _warned_no_headroom
    with _batch_size_cache_lock:
        _batch_size_cache.clear()
    _warned_no_headroom = False


@dataclass(frozen=True)
class BackfillTickResult:
    """What one tick of the backfill accomplished — for logging/tests."""

    scanned: int  # candidate rows examined this tick
    embedded: int  # rows newly embedded (real compute)
    skipped_short: int  # rows skipped for being under MIN_CHARS_TO_EMBED
    errors: int  # embed attempts that raised (logged, cursor skips past them)
    batch_size: int  # the derived (or caller-overridden) batch size used


def _load_cursor(persona_dir, current_model_id: str) -> tuple[str, str] | None:  # noqa: ANN001
    """Best-effort cursor read. Missing/corrupt/model-mismatched/old-format
    -> None (start of history) — fail toward re-scanning, never toward
    silently skipping a row. See module docstring: the cursor is an
    optimization, not the correctness mechanism.

    The persisted cursor is the COMPOSITE ``{"created_at": ..., "id": ...}``
    form (see ``MemoryStore.list_unembedded_since``). A cursor file written
    by a pre-keyset build of this module is timestamp-only (a bare string)
    — that old format is deliberately NOT half-interpreted (e.g. paired with
    an empty/sentinel id, which would silently reintroduce the same-timestamp
    blind spot this format exists to close); it is treated exactly like any
    other unparseable cursor and reset to ``None``, which just rescans from
    the top. Safe and cheap: a rescan only re-confirms rows already embedded
    under the current model (they no longer satisfy the backlog predicate).
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
        # cursor position, so the next tick rescans from the top rather than
        # trusting a position recorded under a different model_id. As of the
        # model-mismatch backlog clause (F1 #259 increment 3, folded in
        # 2026-09-16), this reset is now doubly correct: a model swap DOES
        # reopen every row embedded under the PRIOR model for this same
        # backlog query (`embedding_model_id != current_model_id`), so
        # trusting a cursor position recorded before the swap could skip
        # straight past exactly the rows the swap just put back in scope.
        # Resetting to None makes the very next tick a full rescan of the
        # now-larger (model-mismatched + still-NULL) backlog.
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
    *,
    batch_size: int | None = None,
    scan_cap: int = DEFAULT_SCAN_CAP,
) -> BackfillTickResult:
    """Embed up to `batch_size` un-embedded active memories, off the hot path.

    Intended to be called once per ELIGIBLE supervisor tick — the caller
    (``brain/bridge/supervisor.py``) is responsible for idle-gating this
    call (see module docstring); this function itself does not check
    ``cli_throttle``.

    `batch_size`, when omitted, is the runtime-derived figure from
    `_get_batch_size` (measured once per process per model_id — see module
    docstring); pass an explicit value to override (tests do this for
    deterministic bounds). `scan_cap` separately bounds how many rows are
    even read from `memories` this call.

    Backlog is exactly `MemoryStore.list_unembedded_since`'s definition:
    active, long-enough rows with `embedding IS NULL` OR
    `embedding_model_id` stale under the current model (see module docstring
    BACKLOG DEFINITION). A row is fault-isolated on failure — logged and the
    tick CONTINUES to the next candidate, never
    starving later rows — and the persisted cursor is allowed to advance
    PAST a failing row (skip-and-log, F1 #259 increment 3's cursor-freeze
    fix), so one permanently-bad row can never stall the backlog the way it
    used to.
    """
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.build_embedding_provider()
    model_id = provider.model_id()
    effective_batch_size = (
        batch_size if batch_size is not None else _get_batch_size(provider, scan_cap)
    )

    cursor = _load_cursor(persona_dir, model_id)
    candidates = store.list_unembedded_since(
        cursor, limit=scan_cap, current_model_id=model_id, min_chars=MIN_CHARS_TO_EMBED
    )

    scanned = 0
    embedded = 0
    skipped_short = 0
    errors = 0
    resolved_up_to = cursor

    for memory in candidates:
        scanned += 1
        row_cursor = (memory.created_at.isoformat(), memory.id)

        if len(memory.content) < MIN_CHARS_TO_EMBED:
            # Defensive/redundant as of F1 #259 increment-3 red-team fix
            # (F1): `list_unembedded_since` now excludes short rows in SQL
            # (`length(content) >= min_chars`), which is the load-bearing
            # exclusion — a short row is no longer even a `candidate` here.
            # This branch is kept as a defense-in-depth backstop only; it
            # should never actually trigger against this method's own query.
            skipped_short += 1
            resolved_up_to = row_cursor
            continue

        if embedded >= effective_batch_size:
            # Batch budget spent — leave this (and anything after it) for
            # next tick. Ordinary pacing, not a failure: the cursor must NOT
            # advance past a row that was never even attempted.
            break

        try:
            store.embed_row(memory.id, memory.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding_backfill: embed failed for memory %s — skipping "
                "and advancing past it (skip-and-log): %s",
                memory.id,
                exc,
            )
            errors += 1
            resolved_up_to = row_cursor  # cursor-freeze fix: advance past it
            continue

        embedded += 1
        resolved_up_to = row_cursor

    # If the DB-side query returned fewer rows than scan_cap, this tick has
    # seen the WHOLE currently-null backlog — persist NO forward cursor
    # (reset to None) rather than `resolved_up_to`. Unlike the old
    # content-hash cache, `embedding IS NULL` is not append-only per id: a
    # row can go null a SECOND time (a later content edit whose synchronous
    # re-embed fails — see MemoryStore._reembed_or_clear) at a `created_at`
    # position the cursor may already have passed. A persisted forward
    # cursor is only a safe scan-cost optimization while the backlog is
    # LARGER than one scan window (the case it exists to make cheap); once
    # it fits in one window, resetting is nearly free and makes the NEXT
    # tick a full, self-healing rescan instead of trusting a stale position.
    cursor_to_persist = None if len(candidates) < scan_cap else resolved_up_to
    _save_cursor(persona_dir, model_id, cursor_to_persist)

    return BackfillTickResult(
        scanned=scanned,
        embedded=embedded,
        skipped_short=skipped_short,
        errors=errors,
        batch_size=effective_batch_size,
    )
