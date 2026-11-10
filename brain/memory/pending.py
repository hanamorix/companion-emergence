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

from brain import dev_constants
from brain.health.json_io import (
    Recovery,
    append_jsonl,
    encode_jsonl,
    iter_jsonl_raw,
    read_jsonl,
    write_jsonl,
    write_jsonl_bytes,
)
from brain.memory.store import (
    Memory,
    MemoryStore,
    has_feeling_at_or_above,
    scaled_small_emotions,
    times_if_finite,
)
from brain.utils import file_ops
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

# The one type of the x10 fix (`dev_constants.NUDGE_SCALE_FIX_MEMORY_TYPES`) that waits in this queue: `initiate_outbound` is a
# gate-bypass type and `kindled_peer` rows are written straight to the database, so only a `making` can be a candidate.
_X10_QUEUED_TYPE = "making"


def _scaled_candidate(entry: dict, factor: float, threshold: float, marker_key: str) -> dict | None:
    """The queue entry `entry` with its feelings multiplied and the marker set (`scale_up_small_making_candidates`), or None to leave it.

    None, without a log line, for an entry that is not selected: not a `making`, already marked, or a feeling at or above the
    threshold (the largest value is not below it). None WITH a warning naming the entry's id (never content) for an entry that
    cannot be multiplied safely: feelings that cannot be multiplied (`store.scaled_small_emotions`), a `score` that is present
    and not a number or whose product is not finite, or `metadata` that is not a mapping. An absent or null `score` is not
    multiplied (`Memory.from_dict` reads null as missing); absent emotions are the empty mapping, which is selected.
    """
    if entry.get("memory_type") != _X10_QUEUED_TYPE:
        return None
    where = f"pending candidate {entry.get('id')}"
    metadata = entry.get("metadata") or {}  # as `Memory.from_dict` reads it
    if not isinstance(metadata, dict):
        logger.warning("x10 fix: %s skipped, its metadata is not a mapping, so it cannot be marked", where)
        return None
    if metadata.get(marker_key):
        return None
    emotions = entry.get("emotions", {})
    if has_feeling_at_or_above(emotions, threshold):
        return None  # the largest value is not below the threshold: an entry already on the 0-10 scale (no warning: most are)
    multiplied = scaled_small_emotions(emotions, factor=factor, threshold=threshold, where=where)
    if multiplied is None:
        return None
    scaled = {**entry, "emotions": multiplied, "metadata": {**metadata, marker_key: True}}
    if entry.get("score") is not None:
        new_score = times_if_finite(entry["score"], factor)
        if new_score is None:
            logger.warning("x10 fix: %s skipped, its score is not a number, or its score times %s is not a finite number", where, factor)
            return None
        scaled["score"] = new_score
    return scaled


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

    def _read_keeping_damaged(self, path: Path) -> tuple[list[dict], bool]:
        """Read a queue-shaped JSONL file; an I/O fault raises. A damaged line is KEPT ASIDE, not skipped.

        A plain read skips a damaged line, and the caller then rewrites the file without it: the line, the only copy of a
        claimed entry, would be gone. Here each damaged line is appended to ``pending_candidates.damaged.jsonl`` (its raw bytes
        base64-encoded, plus the file it came from, its position among the non-blank lines and a timestamp) BEFORE the
        caller changes anything; if that append fails the read raises and nothing has changed. Returns the good entries and
        whether any line was damaged: a file whose only content was damaged lines must still be emptied by the caller once
        they are kept aside, or every tick would keep the same lines aside again.
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
        return entries, bool(damaged)

    def enqueue_reappraisal(self, memory_id: str, *, source: str) -> None:
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
        self.enqueue_reappraisals([memory_id], source=source)

    def enqueue_reappraisals(self, memory_ids: list[str], *, source: str) -> int:
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
        individual calls. No-op (returns 0, does not touch the file) on an
        empty list. Returns the number of lines written.

        This reads NOTHING: it is one locked append whatever the size of the queue or of the processing file, because it
        runs on the prompt-build path twice per chat turn (#470 review). Repeated ids are not removed here: ``claim`` keeps
        ONE re-appraise item per memory id and drops the rest from both files, so an outage cannot pile them up.
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
        append_jsonl(self.path, items)  # one write for the whole batch
        return len(items)

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
            rows, _ = self._read_keeping_damaged(self.path)
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
                kept, _ = self._read_keeping_damaged(self.processing_path)
                leftovers = [e for e in kept if e.get("_claim_id") not in settled]
            queued, queue_damaged = self._read_keeping_damaged(self.path) if self.path.exists() else ([], False)
            # One re-appraise item per memory id, here and nowhere else (the hot-path enqueue only appends): an outage piles
            # them up (every recall enqueues them) and each tick would otherwise re-stamp and rewrite the whole pile. A
            # duplicate is dropped from the processing file this claim rewrites and from the queue file it empties, so it is
            # in neither file afterwards and can hold nothing open.
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
                if queue_damaged:  # only damaged lines: kept aside above, so empty the file or the next tick keeps them again
                    write_jsonl(self.path, [], fsync=False)
                return ClaimedBatch(self, [])
            write_jsonl(self.processing_path, entries, nonfinite="null")  # durable BEFORE the queue is emptied
            write_jsonl(self.settled_path, [], fsync=False)
            if queued or queue_damaged:
                write_jsonl(self.path, [], fsync=False)  # empty the data file (the lock sidecar is separate)
        return ClaimedBatch(self, entries)

    def scale_up_small_making_candidates(self) -> int:
        """The x10 fix of small feelings for the candidates still waiting here: returns how many entries it multiplied.

        The emotion engine's first start (`brain/emotion/nudge_scale_fix.py`, step (2b) after the database half) calls this once,
        because a `making` candidate that waits in the queue is not a `memories.db` row yet and the fix of the rows cannot reach
        it: the gate would promote it later at the old small size (owner, Planning B21 Q10c: "They're memories so they should get
        treated as such"). Both files are reached, the queue and the claimed batch in the processing file, under the queue's own
        lock (the one `enqueue` and `claim` take), each replaced atomically through the shared writer, so the queue keeps one
        writer.

        An entry is selected when its type is `making`, it is not marked, and the largest of its emotion values is below the
        threshold (the peak is not carried in an entry, so it is computed here; an empty or absent emotions mapping counts as 0
        and is selected). The whole entry's emotions are multiplied, negatives included, and its `score` if it carries one
        (`store.scaled_small_emotions` and `store.times_if_finite`: the rule of the database half). The marker key goes in the
        entry's `metadata` (a top-level key would be lost: `Memory.from_dict` ignores unknown keys), which the promoted row
        carries, so a start-up that fails later and is run again multiplies no entry twice and still catches one queued between.
        An entry that cannot be multiplied safely is skipped with a warning naming its id and left as it was; a line that cannot
        be parsed, a line that is not an object, and every entry not selected are written back as the shared reader yielded them
        (their bytes without the line terminator: a blank line, a carriage return before the newline and a leading byte-order
        mark are not kept; the app writes none of them). A file in which nothing changed is not rewritten, and with neither file
        present nothing is created, not even the lock sidecar. The count and the ids of each file's multiplied entries are logged
        after that file's write. A read or write fault raises with the file as it was: the caller stops the start-up, and the next start
        runs this again (the marker makes that safe).
        """
        if file_ops.stat_path(self.path) is None and file_ops.stat_path(self.processing_path) is None:
            return 0
        factor = dev_constants.NUDGE_SCALE_FIX_FACTOR
        threshold = dev_constants.NUDGE_SCALE_FIX_THRESHOLD
        marker_key = dev_constants.NUDGE_SCALE_FIX_MARKER_KEY
        count = 0
        with file_lock(self.path):
            for path in (self.path, self.processing_path):
                count += self._scale_up_file(path, factor, threshold, marker_key)
        return count

    def _scale_up_file(self, path: Path, factor: float, threshold: float, marker_key: str) -> int:
        """Rewrite one queue-shaped file with its small `making` entries multiplied; how many it multiplied (the caller holds the lock).

        The count and every multiplied entry's id are logged once, right after this file's write, so the entries can be found by
        hand if the fix ever has to be undone, even when the other file's pass then fails.
        """
        if file_ops.stat_path(path) is None:
            return 0
        lines: list[bytes] = []
        multiplied: list[str] = []
        # RAISE: a file that cannot be read must stop the start-up, not be skipped and rewritten without its lines.
        for text, entry in iter_jsonl_raw(path, recover=Recovery.RAISE, bad_lines=True):
            scaled = None if entry is None else _scaled_candidate(entry, factor, threshold, marker_key)
            if scaled is None:
                lines.append(text.encode("utf-8", "surrogateescape") + b"\n")
            else:
                lines.append(encode_jsonl(scaled, nonfinite="null"))
                multiplied.append(str(entry.get("id")))  # type: ignore[union-attr]
        if multiplied:
            # Not `write_jsonl`: it decodes its payload as UTF-8, and a line kept byte for byte may not be valid UTF-8.
            write_jsonl_bytes(path, b"".join(lines))  # atomic and durable: a failure leaves the file as it was
            logger.info(
                "pending queue: multiplied the feelings of %d candidate(s) in %s by %s (ids: %s)",
                len(multiplied), path.name, dev_constants.NUDGE_SCALE_FIX_FACTOR, ", ".join(multiplied),
            )
        return len(multiplied)


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
