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
NOT trusted once a tick actually EXHAUSTS its fetched candidate window (i.e.
processes every fetched row without stopping early on its own `batch_size`
cap) AND that window came back smaller than `scan_cap` (the whole
currently-null backlog fit in one scan): see the comment at the bottom of
`run_embedding_backfill_tick` for why a forward cursor is only safe while
the backlog is larger than one scan window. (F1 #259 increment 7: this used
to key off the fetched window size alone — `len(candidates) < scan_cap` —
which reset to `None` on EVERY tick whenever `batch_size < backlog <
scan_cap`, even though the tick made genuine progress; a tick that stops
early on its batch cap now persists a forward cursor instead, so a
long-lived persona's no-sleep drain doesn't re-scan its full history every
single tick.) A missing/corrupt/stale cursor file just means the next tick
rescans more than strictly necessary, never that a row is skipped. The
cursor resets automatically on a model swap (its persisted `model_id` no
longer matches the current provider's), so a new model reopens the whole
backlog rather than silently under-covering it.

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

ONE-GO DRAIN (F1 #259 increment 6, spec §5b / S12): ``run_embedding_
backfill_to_completion`` (below) is the separate operator escape hatch
behind the ``nell embed backfill`` CLI command (``brain/cli.py``). It loops
this module's own ``run_embedding_backfill_tick`` back-to-back with no
inter-tick sleep and no idle gate (operator-initiated, not the supervisor
cadence), and adds only the loop's own termination condition on top — see
that function's docstring for why a naive "loop until empty" would spin
forever on a small permanently-failing backlog, and how skip-and-log's
cursor-reset behavior is what the termination check keys off of.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
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
    hit_batch_limit = False

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
            # advance past a row that was never even attempted. Recorded so
            # the cursor-persistence decision below (F1 #259 increment 7)
            # knows this tick did NOT exhaust its fetched candidate window.
            hit_batch_limit = True
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

    # Reset the cursor to None ONLY when this tick genuinely EXHAUSTED its
    # fetched candidate window — processed every row without stopping early
    # on its own `batch_size` cap (`not hit_batch_limit`) — AND that window
    # was smaller than `scan_cap` (the whole remaining backlog fit in one
    # scan). Unlike the old content-hash cache, `embedding IS NULL` is not
    # append-only per id: a row can go null a SECOND time (a later content
    # edit whose synchronous re-embed fails — see
    # MemoryStore._reembed_or_clear) at a `created_at` position the cursor
    # may already have passed. A persisted forward cursor is only a safe
    # scan-cost optimization while there is more of the CURRENTLY FETCHED
    # window left to examine (the `hit_batch_limit` case) or the backlog is
    # LARGER than one scan window; once the window is both fully processed
    # and smaller than `scan_cap`, resetting is nearly free and makes the
    # NEXT tick a full, self-healing rescan instead of trusting a stale
    # position.
    #
    # F1 #259 increment 7 perf fix: this used to key off `len(candidates) <
    # scan_cap` ALONE, which reset to None on every tick whenever
    # `batch_size < remaining_backlog < scan_cap` — even though the tick had
    # just made real progress — forcing the next tick to re-scan the whole
    # backlog from the top instead of continuing from where it left off.
    # Under the no-sleep drain (`run_embedding_backfill_to_completion`) on a
    # long-lived persona this meant every single tick re-scanned full
    # history. Now a tick that stops early on its batch cap persists a
    # forward cursor (`hit_batch_limit` is True, so `window_exhausted` is
    # False regardless of `len(candidates)`), and a genuinely-drained tick
    # (no batch cap hit, window smaller than scan_cap) still resets — the
    # "catch rows that went NULL behind the cursor" self-healing property is
    # unchanged for that case.
    #
    # ACCEPTED BOUND (#259 inc7 red-team F2b): under SUSTAINED ingestion that
    # keeps the backlog permanently larger than `batch_size` (every tick hits
    # `hit_batch_limit`), the cursor may simply never reset for a long time.
    # A row that goes NULL a SECOND time BEHIND the persisted cursor position
    # (a failed synchronous re-embed from `fade`/`update(content=...)`/
    # `unfade` — see `MemoryStore._reembed_or_clear`) is invisible to
    # `list_unembedded_since` until either the backlog finally drains below
    # one scan window (letting a tick exhaust it and reset the cursor) or a
    # `MODEL_EMBEDDING` swap forces a full rescan (`_load_cursor` resets on
    # model-id mismatch). This is bounded and self-healing, not silent data
    # loss — the row's content is unchanged, only its (re-)embedding is
    # delayed — and it is an ACCEPTED tradeoff: a synchronous re-embed
    # failure is rare (the embed call that just succeeded once already;
    # `_reembed_or_clear` only hits this path on a second, later failure),
    # and a companion's ingestion is bursty, not sustained — the backlog
    # drains at idle, which is exactly when this backfill runs.
    window_exhausted = not hit_batch_limit and len(candidates) < scan_cap
    cursor_to_persist = None if window_exhausted else resolved_up_to
    _save_cursor(persona_dir, model_id, cursor_to_persist)

    return BackfillTickResult(
        scanned=scanned,
        embedded=embedded,
        skipped_short=skipped_short,
        errors=errors,
        batch_size=effective_batch_size,
    )


# ---------------------------------------------------------------------------
# Drain-to-completion (F1 #259 increment 6 — the `nell embed backfill` CLI's
# operator escape hatch, spec §5b / S12).
# ---------------------------------------------------------------------------

# `(embedded_so_far, failed_so_far, scanned_so_far)` — cumulative running
# totals across the whole drain, invoked once after EVERY tick (including
# the final one) so a caller (the CLI) can render a live ticker without
# polling the store itself.
ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class BackfillDrainResult:
    """What a full `run_embedding_backfill_to_completion` call accomplished."""

    embedded: int  # rows newly embedded across the whole drain
    failed: int  # rows STILL un-embedded/backlog when the drain stopped —
    # a fresh `MemoryStore.count_unembedded` count taken after the last
    # tick, NOT a sum of per-tick `errors` (see docstring below for why a
    # per-tick sum can double-count a row attempted more than once).
    scanned: int  # cumulative candidate rows examined across all ticks
    ticks: int  # number of `run_embedding_backfill_tick` calls performed
    stopped_reason: str  # "no_more_candidates" | "stalled_no_progress"


def run_embedding_backfill_to_completion(
    persona_dir,  # noqa: ANN001 — Path, kept untyped to mirror run_embedding_backfill_tick
    store: MemoryStore,
    *,
    batch_size: int | None = None,
    scan_cap: int = DEFAULT_SCAN_CAP,
    progress_cb: ProgressCallback | None = None,
) -> BackfillDrainResult:
    """Drain the embedding backlog to completion in ONE call, for the
    operator-initiated `nell embed backfill` CLI command (F1 #259 increment
    6, spec §5b / S12) — NOT for the idle-gated supervisor cadence, which
    stays on `run_embedding_backfill_tick` directly (one bounded tick per
    ELIGIBLE supervisor tick, gated by the caller).

    Loops `run_embedding_backfill_tick` back-to-back with NO inter-tick
    sleep and NO idle gate — flat-out, exactly as the operator asked for by
    running this command at all. `batch_size`/`scan_cap` are passed straight
    through to every tick (same runtime-derived-batch-size + cursor-freeze-
    skip-and-log machinery as the idle path — see module docstring); this
    function adds only the LOOP and its termination condition on top.

    TERMINATION (the part this function adds on top of one tick): a tick's
    own cursor-freeze fix already lets ONE tick skip past a failing row and
    keep going — but a naive `while backlog not empty: tick()` loop can
    still spin forever on a SMALL backlog that is entirely (or down to its
    last row) permanently-failing. Why: whenever a tick embeds ZERO rows, it
    can never have stopped early on its own `batch_size` cap (that cap only
    triggers after `embedded >= 1`), so it necessarily processed every
    fetched candidate — meaning `run_embedding_backfill_tick` resets its
    persisted cursor to `None` whenever a tick both embeds zero rows AND
    sees FEWER than `scan_cap` candidates (i.e. the whole remaining backlog
    fit in one scan and every row in it failed or was skipped — see that
    function's closing comment) — a still-NULL permanently-failing row stays
    in that same small backlog forever, so the VERY NEXT tick would rescan
    from the top, hit the identical row(s), fail identically, and reset the
    cursor to `None` again: infinite, byte-for-byte-identical repetition,
    burning real provider calls for no progress. Detect this directly rather
    than counting on a wall-clock timeout: if a tick embeds ZERO rows AND
    saw fewer than `scan_cap` candidates (`result.scanned < scan_cap` — the
    exact condition under which the tick just reset its cursor to `None`),
    stop — the next tick is guaranteed to reproduce the same result, so
    there is nothing to gain by calling it. (A tick that scans a FULL
    `scan_cap` window of entirely failing rows is NOT a stall: its cursor
    advances PAST that window regardless of success/failure — see the
    tick's own skip-and-log cursor logic — so the NEXT tick genuinely
    examines different, not-yet-seen rows; looping continues in that case.
    Likewise a tick that DOES embed rows before hitting its `batch_size` cap
    also advances its cursor rather than resetting — F1 #259 increment 7 —
    but that case can never satisfy this stall check's `embedded == 0`
    condition in the first place, so the termination guard's correctness is
    unaffected.) A tick that saw zero candidates at all (`scanned == 0`)
    means there is nothing left reachable from the current cursor position
    — also stop.

    Both stop conditions are reached in a BOUNDED number of ticks: the
    keyset cursor only ever moves forward (or resets to re-scan a
    provably-exhausted-of-successes remainder), so the total rows examined
    across the whole drain is bounded by a small constant multiple of the
    backlog size, never unbounded.

    `failed` on the returned result is deliberately NOT a running sum of
    each tick's `errors` field — a row that resets the cursor (small
    backlog, see above) and is retried on a LATER tick before the stall is
    detected would be double-counted by a naive sum. Instead it is a fresh
    `store.count_unembedded()` call taken once after the loop stops: the
    true number of rows still `embedding IS NULL` (or model-stale) in the
    table at that moment, independent of how many times any one of them was
    attempted. Note this can be > 0 even when the loop stopped via
    `"no_more_candidates"`, not only `"stalled_no_progress"`: a large
    backlog with failing rows SCATTERED through it can have those rows sit
    behind an already-advanced forward cursor (see the tick's own docstring)
    and so never reappear as `candidates` again within this run, even though
    they are still genuinely un-embedded — `count_unembedded` catches them
    because it does not consult the cursor at all. This is expected, not a
    bug: those rows permanently fail to embed; the drain's job is to make
    forward progress and terminate, not to guarantee an empty backlog when
    permanent failures exist.

    `progress_cb`, when given, is invoked after EVERY tick (including the
    one that triggers a stop) with cumulative
    `(embedded_so_far, failed_so_far_this_run, scanned_so_far)` — the
    "failed_so_far" arg IS a running sum of `errors` (may over-count a
    retried row by the same small margin described above), since it is only
    ever used for a live progress ticker, not the authoritative final count;
    the authoritative count is this function's own returned `.failed`.
    """
    from brain.memory import embeddings as embeddings_mod

    total_embedded = 0
    total_failed_attempts = 0
    total_scanned = 0
    ticks = 0
    stopped_reason = "no_more_candidates"

    while True:
        result = run_embedding_backfill_tick(
            persona_dir, store, batch_size=batch_size, scan_cap=scan_cap
        )
        ticks += 1
        total_embedded += result.embedded
        total_failed_attempts += result.errors
        total_scanned += result.scanned

        if progress_cb is not None:
            progress_cb(total_embedded, total_failed_attempts, total_scanned)

        if result.scanned == 0:
            stopped_reason = "no_more_candidates"
            break

        if result.embedded == 0 and result.scanned < scan_cap:
            stopped_reason = "stalled_no_progress"
            break

    provider = embeddings_mod.build_embedding_provider()
    final_failed = store.count_unembedded(
        current_model_id=provider.model_id(), min_chars=MIN_CHARS_TO_EMBED
    )

    return BackfillDrainResult(
        embedded=total_embedded,
        failed=final_failed,
        scanned=total_scanned,
        ticks=ticks,
        stopped_reason=stopped_reason,
    )
