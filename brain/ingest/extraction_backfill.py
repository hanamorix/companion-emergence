"""Extraction backfill (#523): re-extract conversation text the old extraction cut never showed the extractor.

Until #523 an extraction pass sent only the last 24,000 characters of the span after the session's cursor and then moved
the cursor to the span's end, so the head of every longer span was recorded as extracted without being seen. The text
itself is kept: in ``archived_conversations/`` (compaction archives only rows behind the cursor) and in the active buffers
behind their cursor. This module plans which of it to re-extract and runs the plan one chunk at a time.

Two modes (owner decision D1):

* ``heads`` (the default, "lost starts"): rebuild the historical passes with the historical lull per era and take, for
  every inferred span whose formatted length exceeds the historical cap, the rows that were cut. Built to be a superset of
  what the lull MODEL says was cut (longest plausible speaker labels, the upper bound of each era's lull), NOT proven a
  superset of what the code cut: a span that ran longer because the bridge was not running, or because a pass was deferred
  or failed, is not seen (``1-spec.md`` §3.8 of the #523 run lists what the model misses).
* ``all``: every archived row and every active row behind its cursor, oldest first.

Backfilled memories carry ``metadata.extraction_backfill = {"from": <first row ts>, "to": <last row ts>}`` and
``source_summary = "conversation:<sid>"``, and are BACK-DATED to the chunk's last user turn (owner decision D2), so the time
since the user last spoke never moves past the real newest user turn. Soul candidates are queued in ``heads`` mode only
(owner decision D5).

The plan and its progress live in ``<persona>/extraction_backfill_state.json``; every read-modify-write holds
``file_lock(<state>)`` and never spans a model call. A chunk's progress is written only after its commits, and only if the
plan it started under is still the plan (``plan_id``), so a stop or a kill loses nothing and a restart cannot be overwritten
by an old chunk. One runner at a time (a non-blocking ``file_lock`` on ``extraction_backfill.run``).

It runs through the central cadence function (owner answer 3): ``backfill_gated_job`` is the ``extraction_backfill`` job of
the bridge's job table, and the CLI's foreground mode (no bridge running) calls ``run_central_pass`` with the same job in its
own process. One chunk (one model call) per pass. It never touches a session's cursor, buffer, backoff or extraction lock.
Reading for a plan or a dry run creates no directory, no file and no lock. Each session is planned up to its own cutoff: the
earlier of the plan's creation and the first row the chunked code extracted in THAT session (the per-session marker the pipeline
writes on a session's first chunked pass), so text the chunked pipeline extracted in full is never planned while every head the
old code cut, in any session, still is (main ruling M4 and follow-ups).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge import central_cadence, cli_throttle
from brain.bridge.central_cadence import GatedJob, JobOutcome
from brain.bridge.provider import LLMProvider
from brain.health.json_io import read_json, read_jsonl, write_json
from brain.ingest.extract import format_turn_line
from brain.ingest.ingest_utils import (
    chunk_turns,
    chunked_extraction_since,
    extraction_chunk_chars,
    load_emotion_vocab,
    load_user_pronouns,
)
from brain.ingest.pipeline import _BACKOFF_FAILURE_THRESHOLD, ChunkResult, run_chunk_stages
from brain.ingest.types import IngestReport
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.file_lock import file_lock
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)

# Historical constants of the cut being repaired (they describe the past and are not tunables).
# The per-pass cap: the last 24,000 characters were kept, since fb55643d (2026-04-26).
_HISTORICAL_CAP_CHARS = 24_000
# The lull that triggered a pass: 300 s on a 60 s supervisor tick before 703ba764 (2026-09-07T20:51:12Z), 600 s after
# (tunable from 1f69f189). The upper bound of each (the rule plus one tick) merges the most gaps, so the inferred spans
# are the longest the model allows and their cut heads the largest.
_LULL_BOUNDARY = datetime(2026, 9, 7, 20, 51, 12, tzinfo=UTC)
_PRE_LULL_UPPER_S = 360
_POST_LULL_UPPER_S = 660

MODES = ("heads", "all")
_STATE_FILE = "extraction_backfill_state.json"
_RUNNER_LOCK = "extraction_backfill.run"
_SCHEMA_VERSION = 1
_COUNTERS = ("chunks", "extracted", "enqueued", "deduped", "errors", "missing", "discarded")


# ---------------------------------------------------------------------------
# Side-effect-free source readers (no mkdir, no lock: buffer.py's readers create directories)
# ---------------------------------------------------------------------------


def _active_dir(persona_dir: Path) -> Path:
    return persona_dir / "active_conversations"


def _archive_dir(persona_dir: Path) -> Path:
    return persona_dir / "archived_conversations"


def _read_rows(path: Path) -> list[dict]:
    return read_jsonl(path) if path.is_file() else []


def _archive_files(persona_dir: Path, sid: str) -> list[Path]:
    """The archive files of ``sid`` in provenance order: the legacy single file, then the numbered segments."""
    d = _archive_dir(persona_dir)
    if not d.is_dir():
        return []
    files = [d / f"{sid}.jsonl"] if (d / f"{sid}.jsonl").is_file() else []
    segments = []
    for p in d.glob(f"{sid}.*.jsonl"):
        mid = p.name[len(sid) + 1 : -len(".jsonl")]
        if mid.isdigit():
            segments.append((int(mid), p))
    return files + [p for _, p in sorted(segments)]


def _read_cursor(persona_dir: Path, sid: str) -> datetime | None:
    path = _active_dir(persona_dir) / f"{sid}.cursor"
    if not path.is_file():
        return None
    try:
        return lenient_utc(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _rolled_to(persona_dir: Path, sid: str) -> str | None:
    path = _active_dir(persona_dir) / f"{sid}.rolled_to"
    if not path.is_file():
        return None
    data = read_json(path, default=dict, expect=dict)
    succ = data.get("successor")
    return succ if isinstance(succ, str) and succ else None


def _session_ids(persona_dir: Path) -> list[str]:
    sids: set[str] = set()
    if _active_dir(persona_dir).is_dir():
        sids.update(p.stem for p in _active_dir(persona_dir).iterdir() if p.is_file() and p.suffix == ".jsonl")
    if _archive_dir(persona_dir).is_dir():
        sids.update(p.name.split(".", 1)[0] for p in _archive_dir(persona_dir).iterdir() if p.is_file() and p.suffix == ".jsonl")
    return sorted(sids)


def _row_key(row: dict) -> tuple:
    return (row.get("ts"), row.get("speaker"), row.get("text"))


def _rows_of(persona_dir: Path, sid: str, cutoff: datetime) -> list[dict]:
    """``sid``'s extractable rows held under its own id: archived rows, then active rows at or behind its cursor; no summary
    rows, nothing at or after ``cutoff`` (a session's marker cutoff IS the first row the chunked code extracted), nothing without a parseable ts; a row present twice (same ts, speaker and text) once."""
    cursor = _read_cursor(persona_dir, sid)
    rows = []
    for path in _archive_files(persona_dir, sid):
        rows.extend(_read_rows(path))
    if cursor is not None:
        for row in _read_rows(_active_dir(persona_dir) / f"{sid}.jsonl"):
            ts = lenient_utc(row.get("ts"))
            if ts is not None and ts <= cursor:
                rows.append(row)
    seen: set[tuple] = set()
    out = []
    for row in rows:
        ts = lenient_utc(row.get("ts"))
        if row.get("speaker") == "summary" or ts is None or ts >= cutoff or _row_key(row) in seen:
            continue
        seen.add(_row_key(row))
        out.append(row)
    out.sort(key=lambda r: lenient_utc(r.get("ts")))
    return out


def _user_name(persona_dir: Path) -> str | None:
    """``persona_config.json``'s ``user_name``, read without the config loader's heal (which may write): a plan and a dry
    run write nothing."""
    path = persona_dir / "persona_config.json"
    if not path.is_file():
        return None
    name = read_json(path, default=dict, expect=dict).get("user_name")
    return name if isinstance(name, str) and name else None


def _labels(persona_dir: Path) -> tuple[str | None, str]:
    return _user_name(persona_dir), persona_dir.name


# ---------------------------------------------------------------------------
# Planners
# ---------------------------------------------------------------------------


def _lull_upper_s(gap_start: datetime) -> int:
    """The historical lull (upper bound) for a gap starting at ``gap_start``; a gap whose window reaches the boundary takes
    the larger one (more merging, larger heads)."""
    if gap_start + timedelta(seconds=_PRE_LULL_UPPER_S) < _LULL_BOUNDARY:
        return _PRE_LULL_UPPER_S
    return _POST_LULL_UPPER_S


def _historical_spans(rows: list[dict]) -> list[list[dict]]:
    """Split a session's rows where the silence was long enough that a pass had run (per the era's lull upper bound)."""
    spans: list[list[dict]] = []
    for row in rows:
        ts = lenient_utc(row.get("ts"))
        if spans:
            prev = lenient_utc(spans[-1][-1].get("ts"))
            if (ts - prev).total_seconds() <= _lull_upper_s(prev):
                spans[-1].append(row)
                continue
        spans.append([row])
    return spans


def _cut_rows(span: list[dict], user_label: str, assistant_label: str) -> list[dict]:
    """The rows of ``span`` the old tail cut did not show in full: a row is cut iff it and every later row of the span,
    formatted and joined with newlines, are longer than the historical cap. The cut rows are a prefix of the span."""
    lengths = [len(format_turn_line(r, user_label, assistant_label)) for r in span]
    cut = []
    suffix = -1  # length of "\n".join(lines[i:]) built from the end
    suffix_lengths = [0] * len(span)
    for i in range(len(span) - 1, -1, -1):
        suffix = lengths[i] + (suffix + 1 if suffix >= 0 else 0)
        suffix_lengths[i] = suffix
    for i, row in enumerate(span):
        if suffix_lengths[i] > _HISTORICAL_CAP_CHARS:
            cut.append(row)
        else:
            break
    return cut


def _target(sid: str, rows: list[dict], cutoff_ts: str | None = None) -> dict:
    return {
        "sid": sid,
        "cutoff_ts": cutoff_ts,
        "from_ts": rows[0]["ts"],
        "to_ts": rows[-1]["ts"],
        "rows": len(rows),
        "seen": 0,
        "done_through_ts": None,
        "failures": 0,
        "status": "open",
    }


def _plan_labels(persona_dir: Path) -> tuple[str, str]:
    """The labels the heads planner measures with: the LONGEST plausible ones (the longer of ``user`` and the configured user
    name; of ``assistant`` and the persona dir name). Longer labels only lengthen every suffix, so the set of rows "cut" can
    only grow."""
    user_name, assistant_name = _labels(persona_dir)
    return max(("user", user_name or ""), key=len), max(("assistant", assistant_name), key=len)


def _session_cutoff(cutoff_ts: str, session_cutoffs: dict[str, str] | None, sid: str) -> str:
    """The session's own cutoff when it has one (its marker), else the plan's (its creation time). Rows at or after it are out."""
    return (session_cutoffs or {}).get(sid, cutoff_ts)


def plan_heads(persona_dir: Path, cutoff_ts: str, session_cutoffs: dict[str, str] | None = None) -> list[dict]:
    """Mode ``heads``: one target per inferred historical span whose head the old cap cut (see the module docstring). A
    session's rows are planned up to its own cutoff (``session_cutoffs``, else ``cutoff_ts``)."""
    user_label, assistant_label = _plan_labels(persona_dir)
    targets = []
    for sid in _session_ids(persona_dir):
        cutoff = _session_cutoff(cutoff_ts, session_cutoffs, sid)
        for span in _historical_spans(_rows_of(persona_dir, sid, lenient_utc(cutoff))):
            cut = _cut_rows(span, user_label, assistant_label)
            if cut:
                targets.append(_target(sid, cut, cutoff))
    targets.sort(key=lambda t: lenient_utc(t["from_ts"]))
    return targets


def plan_all(persona_dir: Path, cutoff_ts: str, session_cutoffs: dict[str, str] | None = None) -> list[dict]:
    """Mode ``all``: one target per session holding every extractable row (archived, and active at or behind its cursor), up to
    the session's own cutoff (``session_cutoffs``, else ``cutoff_ts``)."""
    targets = []
    for sid in _session_ids(persona_dir):
        cutoff = _session_cutoff(cutoff_ts, session_cutoffs, sid)
        rows = _rows_of(persona_dir, sid, lenient_utc(cutoff))
        if rows:
            targets.append(_target(sid, rows, cutoff))
    targets.sort(key=lambda t: lenient_utc(t["from_ts"]))
    return targets


def _plan(persona_dir: Path, mode: str, cutoff_ts: str, session_cutoffs: dict[str, str] | None = None) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"unknown backfill mode {mode!r}")
    planner = plan_heads if mode == "heads" else plan_all
    return planner(persona_dir, cutoff_ts, session_cutoffs)


def _target_rows(persona_dir: Path, target: dict, cutoff_ts: str) -> list[dict]:
    """The rows of ``target`` not yet extracted, looked up under its sid and then along the ``rolled_to`` chain (a rollover
    carries the last rows into the successor with the same ts); a row found twice counts once."""
    cutoff = lenient_utc(target.get("cutoff_ts") or cutoff_ts)  # the session's own cutoff when the plan recorded one
    lo, hi = lenient_utc(target["from_ts"]), lenient_utc(target["to_ts"])
    done = lenient_utc(target.get("done_through_ts"))
    rows: list[dict] = []
    seen: set[tuple] = set()
    sid: str | None = target["sid"]
    visited: set[str] = set()
    while sid is not None and sid not in visited:
        visited.add(sid)
        for row in _rows_of(persona_dir, sid, cutoff):
            ts = lenient_utc(row.get("ts"))
            key = _row_key(row)
            if lo <= ts <= hi and (done is None or ts > done) and key not in seen:
                seen.add(key)
                rows.append(row)
        sid = _rolled_to(persona_dir, sid)
    rows.sort(key=lambda r: lenient_utc(r.get("ts")))
    return rows


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILE


def load_state(persona_dir: Path) -> dict | None:
    """The plan, or None. Lock-free: the file is replaced atomically, so a reader sees a whole old or new version."""
    path = state_path(persona_dir)
    if not path.is_file():
        return None
    data = read_json(path, default=dict, expect=dict)
    return data if data.get("schema_version") == _SCHEMA_VERSION and data.get("plan_id") else None


def update_state(
    persona_dir: Path, fn: Callable[[dict | None], dict | None], *, _between_read_and_write: Callable[[], None] | None = None
) -> dict | None:
    """Read-modify-write the state under ``file_lock(<state>)``: ``fn(current)`` returns the new state (None leaves the
    file alone). Never called across a model call. ``_between_read_and_write`` is a test seam."""
    path = state_path(persona_dir)
    with file_lock(path):
        new = fn(load_state(persona_dir))
        if _between_read_and_write is not None:
            _between_read_and_write()
        if new is not None:
            write_json(path, new, indent=2)
        return new


def _session_cutoffs(persona_dir: Path, now: datetime) -> dict[str, str]:
    """Each session's cutoff (main ruling M4 and follow-ups): the earlier of ``now`` (the plan's creation) and the ts of the first
    row the chunked code extracted in that session (``ingest_utils.chunked_extraction_since``). That session's rows from there on
    were extracted in full: neither mode plans them. A session the chunked code has not touched is planned up to ``now``."""
    return {sid: (since if since < now else now).isoformat() for sid, since in chunked_extraction_since(persona_dir).items()}


def _new_state(persona_dir: Path, mode: str) -> dict:
    now = datetime.now(UTC)
    cutoff = now.isoformat()
    session_cutoffs = _session_cutoffs(persona_dir, now)
    targets = _plan(persona_dir, mode, cutoff, session_cutoffs)
    return {
        "schema_version": _SCHEMA_VERSION,
        "plan_id": uuid.uuid4().hex,
        "status": "armed" if targets else "complete",
        "mode": mode,
        "cutoff_ts": cutoff,
        "sessions_cut_earlier": len(session_cutoffs),
        "created_at": now.isoformat(timespec="seconds"),
        "targets": targets,
        "counters": dict.fromkeys(_COUNTERS, 0),
    }


def _open_targets(state: dict) -> list[dict]:
    return [t for t in state.get("targets", []) if t.get("status") == "open"]


def has_work(persona_dir: Path) -> bool:
    """The job's "has work" probe: an armed plan with an open target. No lock, no model call."""
    state = load_state(persona_dir)
    return state is not None and state.get("status") == "armed" and bool(_open_targets(state))



# ---------------------------------------------------------------------------
# One chunk of work
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """What one run did: ``advanced`` when a target's ``done_through_ts`` moved or a target was newly closed (done or
    failed) or the plan completed; ``reason`` names it (the foreground loop prints it when nothing advanced)."""

    advanced: bool
    reason: str


def _all_rows(persona_dir: Path, sid: str) -> Iterator[dict]:
    for path in _archive_files(persona_dir, sid):
        yield from _read_rows(path)
    yield from _read_rows(_active_dir(persona_dir) / f"{sid}.jsonl")


def _newest_user_ts(persona_dir: Path) -> datetime | None:
    newest = None
    for sid in _session_ids(persona_dir):
        for row in _all_rows(persona_dir, sid):
            ts = lenient_utc(row.get("ts")) if row.get("speaker") == "user" else None
            if ts is not None and (newest is None or ts > newest):
                newest = ts
    return newest


def _backdate(persona_dir: Path, rows: tuple[dict, ...], cursor_ts: str | None, now: datetime) -> datetime:
    """The creation time of a chunk's memories (owner decision D2): the ts of the chunk's last ``user`` row, clamped to now.
    A reply row is stamped after its user row (seconds, or minutes on a tool-using turn), so the chunk's end could be later
    than the user's last turn; the user row's ts cannot. A chunk with no user row (only rows before a session's first user
    turn) takes the earliest of its end, the newest user turn of the persona's sessions, and now. Either way a back-dated
    memory is never newer than the real newest user turn, so the time since the user last spoke never moves past it."""
    users = [lenient_utc(r.get("ts")) for r in rows if r.get("speaker") == "user"]
    users = [u for u in users if u is not None]
    if users:
        return min(max(users), now)
    candidates = [now]
    if cursor_ts is not None:
        candidates.append(lenient_utc(cursor_ts))
    newest_user = _newest_user_ts(persona_dir)
    if newest_user is not None:
        candidates.append(newest_user)
    return min(c for c in candidates if c is not None)


def run_one_chunk(
    persona_dir: Path,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    now: datetime | None = None,
) -> Step:
    """Extract the next chunk of the first open target and record its progress (the caller holds the runner lock).

    The chunk goes through ``run_chunk_stages`` with ``session_id`` = the target's sid (so ``source_summary`` is
    ``conversation:<sid>``), the source marker, the back-date and ``queue_soul`` on in ``heads`` mode only. Then, under the
    state lock and only if the plan is still the one the chunk started under: OK moves ``done_through_ts`` to the chunk's
    end and adds the counters; a failed extraction or a commit failure adds one failure, and the third marks the target
    ``failed`` (its rows stay in the archive). A deferral counts nothing (#473). A plan replaced meanwhile only counts
    ``discarded`` (the memories are already committed; a later plan meets them in dedup).
    """
    now = now or datetime.now(UTC)
    state = load_state(persona_dir)
    if state is None:
        return Step(False, "no plan")
    if state.get("status") != "armed":
        return Step(False, str(state.get("status")))
    plan_id, mode, cutoff_ts = state["plan_id"], state["mode"], state["cutoff_ts"]
    open_targets = _open_targets(state)
    if not open_targets:
        update_state(persona_dir, lambda cur: _complete(cur, plan_id))
        return Step(True, "complete")
    target = open_targets[0]
    index = state["targets"].index(target)
    rows = _target_rows(persona_dir, target, cutoff_ts)
    if not rows:
        update_state(persona_dir, lambda cur: _close_target(cur, plan_id, index))
        return Step(True, "target done")

    user_name, assistant_name = _labels(persona_dir)
    chunk = chunk_turns(rows, max_chars=extraction_chunk_chars(), user_name=user_name, assistant_name=assistant_name)[0]
    is_last = len(chunk.turns) == len(rows)
    created_at = _backdate(persona_dir, chunk.turns, chunk.cursor_ts, now)
    marker = {"extraction_backfill": {"from": chunk.turns[0]["ts"], "to": chunk.turns[-1]["ts"]}}
    report = IngestReport(session_id=target["sid"])
    valid_emotions, emotion_vocab = load_emotion_vocab()
    user_pronouns = load_user_pronouns(persona_dir)
    try:
        # One call per piece (one for a normal chunk; an over-long single exchange goes in pieces within the limit, all in
        # this run). Progress is written only after the last piece commits.
        result = ChunkResult.OK
        for piece in chunk.pieces:
            result = run_chunk_stages(
                persona_dir,
                target["sid"],
                piece,
                store=store,
                hebbian=hebbian,
                provider=provider,
                cfg={},
                report=report,
                user_name=user_name,
                assistant_name=assistant_name,
                valid_emotions=valid_emotions,
                emotion_vocab=emotion_vocab,
                user_pronouns=user_pronouns,
                queue_soul=mode == "heads",
                extra_metadata=marker,
                created_at=created_at,
            )
            if result is not ChunkResult.OK:
                break
    except cli_throttle.ShutdownDeferred:
        raise
    except Exception:  # noqa: BLE001 - a store or commit fault counts as a failed attempt, so it cannot retry forever
        logger.exception("extraction_backfill: chunk raised sid=%s", target["sid"])
        report.errors += 1
        result = ChunkResult.FAILED
    logger.info(
        "extraction_backfill_chunk sid=%s mode=%s from=%s to=%s result=%s extracted=%d deduped=%d",
        target["sid"], mode, marker["extraction_backfill"]["from"], marker["extraction_backfill"]["to"], result.value,
        report.extracted, report.deduped,
    )
    if result is ChunkResult.DEFERRED:
        return Step(False, "deferred")

    step: list[Step] = []

    def _record(cur: dict | None) -> dict | None:
        if cur is None:
            step.append(Step(False, "no plan"))
            return None
        if not _same_plan(cur, plan_id):
            cur["counters"]["discarded"] += 1
            step.append(Step(False, "plan changed"))
            return cur
        t, c = cur["targets"][index], cur["counters"]
        if result is ChunkResult.OK:
            t["done_through_ts"] = chunk.cursor_ts
            t["seen"] += len(chunk.turns)
            c["chunks"] += 1
            c["extracted"] += report.extracted
            c["enqueued"] += report.enqueued
            c["deduped"] += report.deduped
            if is_last:
                _finish(cur, t)
            step.append(Step(True, "chunk"))
        else:
            t["failures"] += 1
            c["errors"] += max(report.errors, 1)
            if t["failures"] >= _BACKOFF_FAILURE_THRESHOLD:
                t["status"] = "failed"
                step.append(Step(True, "target failed"))
            else:
                step.append(Step(False, f"failed (attempt {t['failures']} of {_BACKOFF_FAILURE_THRESHOLD})"))
        if not _open_targets(cur):
            cur["status"] = "complete"
        return cur

    update_state(persona_dir, _record, _between_read_and_write=_TEST_SEAMS.get("between_read_and_write"))
    return step[0]


# Test seam for B5 (a hook INSIDE the progress read-modify-write); never set in production.
_TEST_SEAMS: dict[str, Callable[[], None]] = {}


def _same_plan(state: dict, plan_id: str) -> bool:
    """A chunk's progress applies only to the plan it started under (a ``--restart`` meanwhile made a new plan)."""
    return state.get("plan_id") == plan_id


def _finish(state: dict, target: dict) -> None:
    target["status"] = "done"
    state["counters"]["missing"] += max(0, target["rows"] - target["seen"])


def _close_target(cur: dict | None, plan_id: str, index: int) -> dict | None:
    if cur is None or cur.get("plan_id") != plan_id:
        return None
    _finish(cur, cur["targets"][index])
    if not _open_targets(cur):
        cur["status"] = "complete"
    return cur


def _complete(cur: dict | None, plan_id: str) -> dict | None:
    if cur is None or cur.get("plan_id") != plan_id:
        return None
    cur["status"] = "complete"
    return cur


# ---------------------------------------------------------------------------
# The gated job (bridge and CLI foreground) and the foreground loop
# ---------------------------------------------------------------------------


def _runner_lock(persona_dir: Path) -> AbstractContextManager[bool]:
    return file_lock(persona_dir / _RUNNER_LOCK, blocking=False)


def backfill_gated_job(
    persona_dir: Path,
    *,
    store_ctx: Callable[[], AbstractContextManager[MemoryStore]],
    hebbian_ctx: Callable[[], AbstractContextManager[HebbianMatrix]],
    provider_factory: Callable[[], LLMProvider],
    outcomes: list[Step] | None = None,
) -> GatedJob:
    """The ``extraction_backfill`` job of the central cadence function (spec §3.8): due while an armed plan has an open
    target; each run takes the runner lock and the background slot (either busy: SKIPPED), and runs EXACTLY ONE chunk (one
    model call), then returns COMPLETED so the pass goes on to the later jobs. ``outcomes`` collects each run's ``Step``."""

    def _note(step: Step) -> None:
        if outcomes is not None:
            outcomes.append(step)

    def _run() -> JobOutcome:
        with _runner_lock(persona_dir) as runner:
            if not runner:
                _note(Step(False, "runner busy"))
                return JobOutcome.SKIPPED
            with cli_throttle.background_slot() as slot:
                if not slot:
                    _note(Step(False, "no slot"))
                    return JobOutcome.SKIPPED
                state = load_state(persona_dir)
                if state is None or state.get("status") != "armed":
                    _note(Step(False, "stopped" if state is not None else "no plan"))
                    return JobOutcome.COMPLETED
                with store_ctx() as store, hebbian_ctx() as hebbian:
                    _note(run_one_chunk(persona_dir, store=store, hebbian=hebbian, provider=provider_factory()))
                return JobOutcome.COMPLETED

    return GatedJob("extraction_backfill", run=_run, has_work=lambda: has_work(persona_dir))


def _progress(state: dict | None) -> tuple:
    if state is None:
        return (None,)
    return (state.get("plan_id"), state.get("status"), tuple((t.get("done_through_ts"), t.get("status")) for t in state["targets"]))


_STOP_REASONS = {
    "skip-no-lull": "chat is not idle: a saved message is newer than the lull (a buffer whose last line cannot be read "
    "counts as active too)",
    "skip-slot": "the background slot is busy",
    "skip-deferred": "the model call was refused for now (for example an expired login)",
    "deferred": "the model call was refused for now (for example an expired login)",
    "runner busy": "another backfill run is in progress",
    "no slot": "the background slot is busy",
    "stopped": "the plan was stopped",
    "plan changed": "the plan was replaced while a chunk ran",
    "no plan": "there is no plan",
}


def _keep_going(before: tuple, after: dict | None, decisions: list, outcomes: list[Step]) -> bool:
    """The foreground stop rule: go on only after a pass in which the job advanced a target (a ``done_through_ts`` moved,
    a target closed). Any other pass (no lull, slot or runner busy, a deferral, a failed attempt, the plan stopped or
    replaced from another terminal) ends the loop, so a deferral cannot make it spin: the CLI has no 60 s tick."""
    return _progress(after) != before and bool(outcomes) and outcomes[-1].advanced


def _stop_reason(decisions: list, outcomes: list[Step]) -> str:
    for d in decisions:
        if d.job == "extraction_backfill" and d.action in ("skip-no-lull", "skip-slot", "skip-deferred"):
            return _STOP_REASONS[d.action]
    if outcomes:
        reason = outcomes[-1].reason
        return _STOP_REASONS.get(reason, reason)
    return "nothing to do"


def run_foreground(
    persona_dir: Path,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    bridge_running: Callable[[], bool],
    out: Callable[[str], None] = print,
) -> str:
    """Run the armed plan in this process through ``run_central_pass`` (the CLI with no bridge running), one chunk per pass.

    Before each pass: a bridge that has started takes over (stop, say so); this process's idle anchor is re-seeded from the
    saved buffers (the bridge's start rule, S82), so ``run_central_pass``'s own ``is_chat_idle`` check applies. The loop ends
    after any pass that advanced no target (a deferral, no lull, the slot or the runner lock busy, the plan stopped or
    replaced from another terminal, a failed attempt), naming why; progress is kept, and running the command again resumes.
    Returns a short reason code.
    """
    from contextlib import nullcontext

    outcomes: list[Step] = []
    job = backfill_gated_job(
        persona_dir,
        store_ctx=lambda: nullcontext(store),
        hebbian_ctx=lambda: nullcontext(hebbian),
        provider_factory=lambda: provider,
        outcomes=outcomes,
    )
    while True:
        if bridge_running():
            out("A bridge for this persona is running now: it runs the rest of the plan during lulls. Progress is kept.")
            return "bridge"
        state = load_state(persona_dir)
        if state is None:
            out("There is no backfill plan.")
            return "no plan"
        if state.get("status") == "complete":
            out("The backfill plan is complete.")
            return "complete"
        if state.get("status") != "armed":
            out("The backfill plan is stopped. Progress is kept; run the command again to resume.")
            return "stopped"
        cli_throttle.seed_last_message_from_active_conversations(persona_dir)
        before = _progress(state)
        outcomes.clear()
        decisions = central_cadence.run_central_pass(persona_dir, [job])
        after = load_state(persona_dir)
        if after is not None and after.get("status") == "complete":
            out("The backfill plan is complete.")
            return "complete"
        if _keep_going(before, after, decisions, outcomes):
            c = after["counters"] if after else {}
            out(f"chunk done: chunks={c.get('chunks', 0)} extracted={c.get('extracted', 0)} enqueued={c.get('enqueued', 0)}")
            continue
        out(f"Stopped: {_stop_reason(decisions, outcomes)}. Progress is kept; run the command again to continue.")
        return "not advanced"


# ---------------------------------------------------------------------------
# Plan management for the CLI (`nell memory backfill-extraction`)
# ---------------------------------------------------------------------------


def arm(persona_dir: Path, mode: str) -> tuple[str, str]:
    """Create the plan (mode ``mode``) or resume the existing one. Returns (outcome, message), outcome ``armed`` (there is
    work to run), ``complete`` (nothing to run: the plan is done or empty) or ``refused`` (the existing plan has the other
    mode, which needs ``--restart``). A stopped plan is armed again."""
    message: list[str] = []

    def fn(cur: dict | None) -> dict | None:
        if cur is None:
            new = _new_state(persona_dir, mode)
            message.append(f"planned {len(new['targets'])} target(s), mode {mode}")
            return new
        if cur.get("mode") != mode:
            message.append(
                f"the existing plan is mode {cur.get('mode')}; to plan mode {mode} instead, run again with --restart"
            )
            return None
        if cur.get("status") == "complete":
            message.append("the plan is already complete; --restart plans afresh")
            return None
        if cur.get("status") == "stopped":
            cur["status"] = "armed"
            message.append("resuming the stopped plan")
            return cur
        message.append("resuming the plan")
        return None

    update_state(persona_dir, fn)
    state = load_state(persona_dir)
    if state is None or state.get("mode") != mode:
        outcome = "refused"
    else:
        outcome = "armed" if state.get("status") == "armed" else "complete"
    return outcome, message[0] if message else ""


def restart(persona_dir: Path, mode: str) -> tuple[str, str]:
    """Discard the existing plan, armed, stopped or complete (main ruling M5), and plan afresh (the cutoff is recomputed).
    Returns (outcome, message) like ``arm``; the message says when targets the replaced plan had done are extracted again.
    ``refused``, changing nothing, while a run holds the runner lock (so no chunk is in flight when the plan is replaced)."""
    with _runner_lock(persona_dir) as free:
        if not free:
            return "refused", "a backfill run is in progress; stop it (or wait for it) before --restart"
        replaced: list[dict] = []

        def fn(cur: dict | None) -> dict:
            if cur is not None:
                replaced.append(cur)
            return _new_state(persona_dir, mode)

        new = update_state(persona_dir, fn)
    message = f"planned {len(new['targets'])} target(s), mode {mode}"
    done = sum(1 for t in (replaced[0]["targets"] if replaced else []) if t.get("status") == "done")
    if done:
        message += (
            f"; the replaced plan had {done} target(s) done: their text is extracted again (dedup catches most of it"
            + ("; soul candidates are queued again" if mode == "heads" else "")
            + ")"
        )
    return ("armed" if new["status"] == "armed" else "complete"), message


def stop(persona_dir: Path) -> str:
    """Set the plan ``stopped``: the job does nothing from its next run; progress is kept. With no plan nothing is created
    (not even the state lock's sidecar)."""
    if load_state(persona_dir) is None:
        return "there is no backfill plan"

    def fn(cur: dict | None) -> dict | None:
        if cur is None or cur.get("status") != "armed":
            return None
        cur["status"] = "stopped"
        return cur

    new = update_state(persona_dir, fn)
    if new is not None:
        return "stopped; progress is kept and running the command again resumes"
    state = load_state(persona_dir)
    return "there is no backfill plan" if state is None else f"nothing to stop (the plan is {state.get('status')})"


def status_text(persona_dir: Path) -> str:
    state = load_state(persona_dir)
    if state is None:
        return "no backfill plan"
    targets = state.get("targets", [])
    count = {s: sum(1 for t in targets if t.get("status") == s) for s in ("open", "done", "failed")}
    c = state.get("counters", {})
    return (
        f"status={state.get('status')} mode={state.get('mode')} cutoff={state.get('cutoff_ts')} "
        f"({state.get('sessions_cut_earlier', 0)} session(s) stop earlier, where chunked extraction began in them) "
        f"targets: open={count['open']} done={count['done']} failed={count['failed']}\n"
        + " ".join(f"{k}={c.get(k, 0)}" for k in _COUNTERS)
    )


def dry_run_text(persona_dir: Path) -> str:
    """Both modes' plans, built in memory: per session the targets, characters, chunks (= model calls) and date range.
    No model call, no file or directory created, no lock taken."""
    now = datetime.now(UTC)
    cutoff = now.isoformat()
    session_cutoffs = _session_cutoffs(persona_dir, now)
    user_name, assistant_name = _labels(persona_dir)
    max_chars = extraction_chunk_chars()
    lines = [
        f"dry run (cutoff {cutoff}; {len(session_cutoffs)} session(s) stop earlier, where chunked extraction began in them; "
        f"chunk size {max_chars} chars); nothing is written and no model is called"
    ]
    for mode in MODES:
        per_sid: dict[str, list[int | str]] = {}
        for target in _plan(persona_dir, mode, cutoff, session_cutoffs):
            rows = _target_rows(persona_dir, target, cutoff)
            text = "\n".join(format_turn_line(r, user_name or "user", assistant_name) for r in rows)
            calls = sum(
                len(c.pieces) for c in chunk_turns(rows, max_chars=max_chars, user_name=user_name, assistant_name=assistant_name)
            )
            entry = per_sid.setdefault(target["sid"], [0, 0, 0, target["from_ts"], target["to_ts"]])
            entry[0] += 1
            entry[1] += len(text)
            entry[2] += calls
            entry[4] = target["to_ts"]
        totals = [sum(e[i] for e in per_sid.values()) for i in range(3)]
        lines.append(f"mode {mode}: sessions={len(per_sid)} targets={totals[0]} chars={totals[1]} calls={totals[2]}")
        for sid, (n, chars, calls, first, last) in per_sid.items():
            lines.append(f"  {sid}: targets={n} chars={chars} calls={calls} from={first} to={last}")
    return "\n".join(lines)
