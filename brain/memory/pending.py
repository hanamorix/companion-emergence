"""Pending-candidate queue — a separate store OUTSIDE memories.db.

TEMP (Root 2 stopgap — remove when the Phase 5 dream cycle lands to replace it).

Automatically-generated ("firehose") memories are ENQUEUED here as *candidates*
rather than written straight into ``memories.db``. A candidate is **not a stored
memory**: it never appears in recall, forgetting, hebbian, or the graveyard. On
the idle heartbeat tick the consolidation gate (``brain.engines.consolidation``)
drains this queue and, per candidate, discards it (reject), ``store.create``s it
(promote), or folds it into an existing memory (merge). Because a rejected
candidate was never a ``memories.db`` row, none of the main-DB machinery
(grief/forgetting/hebbian) ever applies to it.

Backing store: ``<persona_dir>/pending_candidates.jsonl``, guarded by the same
``file_lock`` pattern used for ``soul_candidates.jsonl`` (see brain/soul/review.py)
so a concurrent enqueue is never lost across a drain's read-modify-truncate.

Write routing (``route_write``) keys on ``memory_type``, NOT the call site: a
type is gated (enqueued) unless it is in ``GATE_BYPASS_TYPES``. This is robust to
the two variable-typed writers (a reflex arc's ``arc.output_memory_type`` — one
value is the *deliberate* ``journal_entry`` — and conversation-ingest's
``item.label``): each row self-classifies by its own type. ``journal_entry``
(deliberate; feeds the weekly self-narrative) and ``initiate_outbound`` (its own
dedup needs immediate DB visibility) bypass the gate and commit directly.

The gate takes its batch with ``claim`` (not ``drain``): the batch is kept in
``pending_candidates.processing.jsonl`` until each item is settled (recorded in
``pending_candidates.settled.jsonl``), so a kill or an exception mid-run loses nothing; the next
claim recovers what is left. A damaged line found in either file is kept aside in
``pending_candidates.damaged.jsonl`` (raw bytes, with evidence), never skipped and rewritten away.
"""

from __future__ import annotations

import base64
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import Recovery, append_jsonl, iter_jsonl_raw, read_jsonl, write_jsonl
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock
from brain.utils.finite import clamp_nonfinite

logger = logging.getLogger(__name__)

_QUEUE_FILENAME = "pending_candidates.jsonl"
_PROCESSING_FILENAME = "pending_candidates.processing.jsonl"
_SETTLED_FILENAME = "pending_candidates.settled.jsonl"
_DAMAGED_FILENAME = "pending_candidates.damaged.jsonl"

# Types that BYPASS the gate and are written straight to memories.db. Everything
# else produced at an automatic write site is enqueued. Denylist (not allowlist)
# because reflex arc output types are open-ended; fail-safe direction is to gate
# an unknown automatic type (delayed/vetted, recoverable) rather than leak it into
# recall. Owner-confirmed BROAD scope (Roy, 2026-08-11): only journal_entry and
# initiate_outbound bypass.
#
# monologue_emotion / self_model_reconcile bypass the consolidation gate so
# per-turn emotional nudges affect felt state immediately (aggregate_state pools
# their emotion vectors) rather than waiting for consolidation promotion.
# Trade-off: these are recallable auto-generated memories, so they enter recall
# without gate review. If the memory-flood / low-value-recall problem (ROOT2)
# resurfaces, THIS bypass is the first place to look. Revisit when the dream
# cycle replaces this interim gate.
GATE_BYPASS_TYPES: frozenset[str] = frozenset(
    {"journal_entry", "initiate_outbound", "monologue_emotion", "self_model_reconcile"}
)

# Only these monologue-EPISODE types are eligible for the Pass-1 salience drop —
# they carry a real 0..10 importance signal (extractor sets importance=salience*10).
# Dreams (importance still auto-derives from the emotion sum when emotion-flat)
# and monologue_trace (now a per-trace-derived value, P3 retention rework
# Change 1 — no longer a flat 0.3) are EXEMPT: they go through dedup only,
# never salience-drop. research/heartbeat/reflex/initiate now carry explicit
# non-deflated importance (Change 1) but were never SALIENCE_ELIGIBLE anyway —
# still exempt, same as before. Owner directive (Roy, 2026-08-11) — a single
# flat floor would nuke legitimate flat content.
SALIENCE_ELIGIBLE_TYPES: frozenset[str] = frozenset(
    {"monologue", "monologue_emotion", "monologue_soul_candidate"}
)


class PendingQueue:
    """Append-only JSONL queue of pending candidates for one persona."""

    def __init__(self, persona_dir: str | Path) -> None:
        self.path = Path(persona_dir) / _QUEUE_FILENAME
        self.processing_path = Path(persona_dir) / _PROCESSING_FILENAME
        self.settled_path = Path(persona_dir) / _SETTLED_FILENAME
        self.damaged_path = Path(persona_dir) / _DAMAGED_FILENAME

    def enqueue(self, mem: Memory, *, source: str) -> None:
        """Append a candidate (the Memory serialised + provenance) under lock."""
        # A NaN score or emotion is clamped to 0.0 here, not written as null, so the gate's parse of the row
        # (Memory.from_dict, which float()s these values) always meets a number. Behaviour delta of the migration: before it, a NaN
        # candidate was queued as the NaN token and then crashed the gate at store.create (SQLite NOT NULL on
        # score) after the queue had been drained; now it is promoted with 0.0 values and one warning is logged.
        entry = clamp_nonfinite(
            {
                **mem.to_dict(),
                "_source": source,
                "_enqueued_at": datetime.now(UTC).isoformat(),
            },
            where=f"pending candidate {mem.id}",
        )
        append_jsonl(self.path, entry)

    def read_recent(self, memory_type: str, *, limit: int) -> list[Memory]:
        """Return up to `limit` most-recent candidates of `memory_type`.

        Lock-free read (tolerates a partial last line from a concurrent
        append via ``read_jsonl``). Newest first — entries
        are appended oldest→newest, so reversed order is most-recent-first.
        Used by interior-continuity (a DIFFERENT path from recall); returns
        ``Memory`` objects for interface parity with ``store.list_by_type``.
        """
        if limit <= 0:
            return []
        # Candidates the gate has CLAIMED but not yet settled (they wait in the processing file, for a whole login outage
        # when the classifier is deferred) are older than anything in the queue and must not become invisible.
        rows = [*self._unsettled_processing(), *read_jsonl(self.path, recover=Recovery.RAISE)]
        out: list[Memory] = []
        seen: set[str] = set()
        for entry in reversed(rows):
            if entry.get("memory_type") != memory_type:
                continue
            try:
                mem = Memory.from_dict(entry)
            except (KeyError, ValueError, TypeError):
                continue
            if mem.id in seen:  # an entry present in both files after a mid-claim crash
                continue
            seen.add(mem.id)
            out.append(mem)
            if len(out) >= limit:
                break
        return out

    def _unsettled_processing(self) -> list[dict]:
        """The entries of the processing file that the settled file does not list. A lock-free read that never raises."""
        if not self.processing_path.exists():
            return []
        settled = {r.get("id") for r in read_jsonl(self.settled_path)} if self.settled_path.exists() else set()
        return [e for e in read_jsonl(self.processing_path) if e.get("_claim_id") not in settled]

    def _read_keeping_damaged(self, path: Path) -> list[dict]:
        """Read a queue-shaped JSONL file; an I/O fault raises. A damaged line is KEPT ASIDE, not skipped.

        A plain read skips a damaged line, and the caller then rewrites the file without it: the line, the only copy of a
        claimed entry, would be gone. Here each damaged line is appended to ``pending_candidates.damaged.jsonl`` (its raw bytes
        base64-encoded, plus the file it came from, its position among the non-blank lines and a timestamp) BEFORE the
        caller changes anything; if that append fails the read raises and nothing has changed.
        """
        entries: list[dict] = []
        damaged: list[tuple[int, str]] = []
        for position, (text, record) in enumerate(
            iter_jsonl_raw(path, recover=Recovery.RAISE, bad_lines=True), start=1
        ):
            if record is None:
                damaged.append((position, text))
            else:
                entries.append(record)
        if damaged:
            now = datetime.now(UTC).isoformat()
            records = []
            for position, text in damaged:
                raw = text.encode("utf-8", "surrogateescape")
                records.append(
                    {
                        "kept_at": now,
                        "from": path.name,
                        "line": position,
                        "raw": raw.decode("utf-8", "replace"),
                        "raw_b64": base64.b64encode(raw).decode("ascii"),
                    }
                )
            append_jsonl(self.damaged_path, records, nonfinite="null")
            logger.warning(
                "pending queue: %d damaged line(s) in %s kept aside in %s (positions %s)",
                len(records), path.name, self.damaged_path.name, ", ".join(str(r["line"]) for r in records),
            )
        return entries

    def enqueue_reappraisal(self, memory_id: str, *, source: str, include_held: bool = True) -> None:
        """Enqueue an existing-memory importance re-appraise request (P3
        retention rework, Change 3).

        A minimal item, NOT a full Memory payload: just the target id plus
        a top-level `_route` discriminator, so the queue file stays small.
        `drain()` returns it as a raw dict like any other candidate; the
        gate (`engines.consolidation._run_locked`) branches on `_route` to
        separate re-appraise items from normal candidates BEFORE Pass 1/2.
        Off the hot path: this is a cheap file append under lock, same as
        `enqueue` — no appraisal and no provider call happen here. The gate
        re-appraises on its own consolidation tick and UPDATES the existing
        row in place (never a new row).

        Single-id convenience wrapper around `enqueue_reappraisals` — see
        that method if enqueuing more than one id (e.g. every surfaced
        recall hit in one turn); calling this in a loop reacquires the lock
        and reopens the file once per id.
        """
        self.enqueue_reappraisals([memory_id], source=source, include_held=include_held)

    def enqueue_reappraisals(self, memory_ids: list[str], *, source: str, include_held: bool = True) -> int:
        """Enqueue MULTIPLE existing-memory re-appraise requests under a
        SINGLE `file_lock` + single open + single write (P3 retention
        rework, Change 3 hot-path fix).

        The recall hook (`brain.chat.prompt._build_recall_block`) surfaces
        up to ~2*limit memory ids per turn and used to call
        `enqueue_reappraisal` once per id — each call independently
        acquiring the lock and opening/writing the file, i.e. N lock/open/
        write syscalls on the recall hot path. This batches the whole list
        into one lock acquisition and one file write.

        Each written line has the EXACT same shape as a single
        `enqueue_reappraisal` call (`_route`/`memory_id`/`_source`/
        `_enqueued_at`) — only the I/O is batched, not the item schema.
        `drain()` and the consolidation gate see no difference from N
        individual calls, except that an id that already has a re-appraise item
        (queued, or held unsettled by a claimed run) is not enqueued twice. No-op (returns 0, does not touch the file) on an
        empty list. Returns the number of lines written.
        """
        if not memory_ids:
            return 0
        now = datetime.now(UTC).isoformat()
        items = [
            {
                "_route": "reappraise_importance",
                "memory_id": memory_id,
                "_source": source,
                "_enqueued_at": now,
            }
            for memory_id in memory_ids
        ]
        # No pile-up (#459): an id that already has a re-appraise item in the queue, or held unsettled in the processing
        # file (an outage keeps them there), is not enqueued again. Duplicates inside this call collapse too. The check is a
        # lock-free read followed by ONE locked write (the hot path stays at one lock acquisition); two enqueuers racing can
        # still add the same id twice, which `claim` collapses.
        # ``include_held=False`` is for the gate putting a CLAIMED item back after an environment failure: that item is itself
        # unsettled in the processing file (it is settled right after), so it must not be deduplicated against itself.
        held = self._unsettled_processing() if include_held else []
        pending = {
            e.get("memory_id")
            for e in [*read_jsonl(self.path), *held]
            if e.get("_route") == "reappraise_importance"
        }
        fresh = []
        for item in items:
            if item["memory_id"] in pending:
                continue
            pending.add(item["memory_id"])
            fresh.append(item)
        if not fresh:
            return 0
        append_jsonl(self.path, fresh)  # one write for the whole batch
        return len(fresh)

    def drain(self) -> list[dict]:
        """Atomically take the whole queue: read all entries, then truncate.

        Held under ``file_lock`` over the read+truncate window (the proven
        soul-candidate pattern) so a candidate enqueued concurrently lands
        either in this batch or in the emptied file (next tick) — never lost.
        The lock is released before the (slow) gate processing runs on the
        returned batch. Returns raw dicts (carrying ``_source``/``_enqueued_at``);
        the gate reconstructs ``Memory.from_dict`` for promotion.
        """
        if not self.path.exists():
            return []
        with file_lock(self.path):
            # RAISE: the file is emptied right after this read, so a failed read must raise, not drain nothing.
            rows = self._read_keeping_damaged(self.path)
            write_jsonl(self.path, [], fsync=False)  # empty the data file (the lock sidecar is separate)
        return rows


    def claim(self) -> ClaimedBatch:
        """Take the whole queue for processing, crash-safely (#459).

        ``drain`` empties the queue BEFORE any item is processed, so a kill, or an exception, during the slow per-item work
        loses every item not yet dealt with. ``claim`` keeps the batch in a processing file until each item is settled:

        1. under ``file_lock`` (the one ``enqueue`` takes): read the leftovers of an earlier, interrupted claim (the entries of
           the processing file that the settled file does not list) and the queue; stamp every entry with a fresh
           ``_claim_id`` (so ids left over in an old settled file can never match a new batch);
        2. write the processing file (atomically) FIRST, then reset the settled file, THEN empty the queue. A crash after
           the first write leaves the entries in both files (the next claim sees them twice; the gate drops an exact repeat),
           never in neither.

        Leftovers come first, then the queue. A processing file left over with the queue file absent is recovered too (unlike
        ``drain``, which returns ``[]`` when the queue file is missing). Raises on a read fault, before changing anything.
        """
        with file_lock(self.path):
            leftovers: list[dict] = []
            had_files = self.processing_path.exists() or self.settled_path.exists()
            if self.processing_path.exists():
                settled = (
                    {r.get("id") for r in read_jsonl(self.settled_path, recover=Recovery.RAISE)}
                    if self.settled_path.exists()
                    else set()
                )
                leftovers = [
                    e for e in self._read_keeping_damaged(self.processing_path) if e.get("_claim_id") not in settled
                ]
            queued = self._read_keeping_damaged(self.path) if self.path.exists() else []
            # One re-appraise item per memory id: an outage piles them up (every recall enqueues them) and each tick would
            # otherwise re-stamp and rewrite the whole pile.
            seen_ids: set = set()
            combined = []
            for e in [*leftovers, *queued]:
                if e.get("_route") == "reappraise_importance":
                    if e.get("memory_id") in seen_ids:
                        continue
                    seen_ids.add(e.get("memory_id"))
                combined.append(e)
            entries = [{**e, "_claim_id": uuid.uuid4().hex} for e in combined]
            if not entries:
                if had_files:  # everything in them was settled: clear the stale files
                    self.processing_path.unlink(missing_ok=True)
                    self.settled_path.unlink(missing_ok=True)
                return ClaimedBatch(self, [])
            write_jsonl(self.processing_path, entries, nonfinite="null")  # durable BEFORE the queue is emptied
            write_jsonl(self.settled_path, [], fsync=False)
            if queued:
                write_jsonl(self.path, [], fsync=False)  # empty the data file (the lock sidecar is separate)
        return ClaimedBatch(self, entries)


class ClaimedBatch:
    """A batch taken by :meth:`PendingQueue.claim`: ``entries`` to work through, ``settle`` each one, then ``close``."""

    def __init__(self, queue: PendingQueue, entries: list[dict]) -> None:
        self._queue = queue
        self.entries = entries
        self._settled: set[str] = set()

    def settle(self, entry: dict) -> None:
        """Record that ``entry`` is dealt with (promoted, dropped on purpose, requeued or set aside).

        An ``OSError`` writing the record is logged and the entry stays unsettled: it is replayed by the next claim,
        which is the safe direction.
        """
        claim_id = entry.get("_claim_id")
        if not claim_id or claim_id in self._settled:
            return
        try:
            append_jsonl(self._queue.settled_path, {"id": claim_id}, nonfinite="null")
        except OSError as exc:
            logger.warning(
                "pending queue: could not record entry id=%s (claim %s) as settled (%s: %s); it stays unsettled and "
                "will be replayed by the next run",
                entry.get("id") or entry.get("memory_id"), claim_id, type(exc).__name__, exc,
            )
            return
        self._settled.add(claim_id)

    def close(self) -> None:
        """Delete the processing then the settled file iff every entry is settled; otherwise leave both for the next claim.

        The processing file goes first: a kill between the two leaves a settled file with no processing file, which is harmless
        (deleting the settled file first would replay everything).
        """
        if any(e.get("_claim_id") not in self._settled for e in self.entries):
            return
        self._queue.processing_path.unlink(missing_ok=True)
        self._queue.settled_path.unlink(missing_ok=True)


def route_write(store: MemoryStore, mem: Memory, *, source: str) -> str:
    """Route an automatic write by memory_type: enqueue if gated, else create.

    A gated type (anything not in ``GATE_BYPASS_TYPES``) is enqueued as a
    candidate; a bypass type is written straight to ``memories.db``. Locates the
    persona's queue via ``store.persona_dir`` so callers need only the store +
    the Memory they already hold. Returns ``mem.id`` (matching ``store.create``).
    """
    if mem.memory_type in GATE_BYPASS_TYPES:
        return store.create(mem)
    PendingQueue(store.persona_dir).enqueue(mem, source=source)
    return mem.id
