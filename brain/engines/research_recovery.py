"""One-time recovery of the research memories that were lost to an unread reply layout (#591).

Until the parser learned to read a marker and its text on one line, a research reply written that way kept its
``MEMORY: <text>`` line inside the saved notes and the fire was logged with no ``output_memory_id``: the memory was never
created. The text survives in ``research/<interest_id>.md`` (header ``## Session YYYY-MM-DD``, the fire's own UTC date).
This module makes the memory the fire would have made, for every such fire, once, with no user action.

* Which fires: log fires with no ``output_memory_id`` whose interest's notes hold a session block for the fire's UTC date
  that the shared parser reads as a memory AND a verdict (the stored block is the notes section, so it is parsed as
  ``"NOTES:\\n" + block``). An interest has at most one fire per UTC date (24 h cooldown). A date is paired only when every
  fire of the interest on that date is memory-less and the number of recoverable blocks equals the number of fires;
  otherwise it is skipped and counted, never guessed.
* What is made: the memory the live fire builds (``build_research_memory``), dated at the fire (``created_at``), written
  through ``route_write(source="research")`` exactly as the live path does, so the consolidation gate decides and the memory
  store stays the only writer of memories.db. ``metadata["recovered_at"]`` lets the forgetting pass count its recent buffer
  from now (brain/forgetting/policy.py). The close verdict, the spawned topics and the research-completion initiate
  candidate of those old fires are NOT replayed: the interest's state has moved on since and an old ``close`` could shut a
  live thread.
* Pacing: the gate classifies each candidate with one model call and has no batch cap, so a pass hands it ONE interest's
  fires, and ``has_recovery_work`` is false while any recovered candidate is still waiting in the queue.
* Idempotence: every enqueue is followed at once by a line in ``research_recovery_written.jsonl``; a journaled fire is only
  linked, never written again, whatever the gate did with its candidate. The link (``output_memory_id``) is the natural
  marker: a linked fire is never a candidate. A crash anywhere resumes without duplicates; the one window left is a crash
  between an enqueue and its journal line, with the gate then folding that single candidate before the next pass.
* Never rewrites or deletes: the notes and ``interests.json`` are only read, and the research log gains only the links (its
  first form is copied to ``research_log.pre_recovery.json`` before the first change).
"""
from __future__ import annotations

import dataclasses
import logging
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.engines.research import ResearchFire, ResearchLog, build_research_memory
from brain.engines.research_notes import notes_path
from brain.engines.research_session import parse_session_output
from brain.health.json_io import Recovery, append_jsonl, read_json, read_jsonl, write_json
from brain.memory.pending import PendingQueue, route_write
from brain.memory.store import MemoryStore
from brain.utils.atomic_write import save_with_backup_text
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)

STATE_FILE = "research_recovery_state.json"
JOURNAL_FILE = "research_recovery_written.jsonl"
PRE_RECOVERY_LOG = "research_log.pre_recovery.json"

_HEADER = re.compile(r"^## Session (\d{4}-\d{2}-\d{2})[ \t]*$", re.MULTILINE)
_COUNTERS = ("recovered", "resumed", "ambiguous", "no_block", "interest_missing", "unreadable_notes", "errors")

# An item that failed (a notes file that cannot be read, a write that raised) is left aside for the rest of this process
# and retried at the next start, so a persistent fault cannot make the job loop at every idle pass; the other interests
# carry on. ``_DEFERRED`` holds those fires' keys per persona; ``_STALLED`` the personas whose only work left is deferred
# (or whose pass raised), for which the job is not due again until a new process.
_DEFERRED: dict[str, set[str]] = {}
_STALLED: set[str] = set()


@dataclass(frozen=True)
class _Pair:
    fire: ResearchFire
    memory_text: str


def _fire_key(fire: ResearchFire) -> str:
    return f"{fire.interest_id}|{iso_utc(fire.fired_at)}"


def _memory_id(fire: ResearchFire) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"research-recovery/{_fire_key(fire)}"))


def _fire_date(fire: ResearchFire) -> str:
    return fire.fired_at.astimezone(UTC).strftime("%Y-%m-%d")


def _load_state(persona_dir: Path) -> dict:
    """The state file, read tolerantly: an unknown key is ignored and a value of the wrong type counts as absent."""
    data = read_json(persona_dir / STATE_FILE, default=dict)
    if not isinstance(data, dict):
        data = {}
    counts = data.get("counts")
    skipped = data.get("skipped")
    return {
        "done": data.get("done") is True,
        "counts": counts if isinstance(counts, dict) else {},
        "skipped": [k for k in skipped if isinstance(k, str)] if isinstance(skipped, list) else [],
    }


def _queued_recovered(persona_dir: Path) -> list:
    return [
        m for m in PendingQueue(persona_dir).read_recent("research", limit=sys.maxsize)
        if m.metadata.get("recovered_from_notes")
    ]


def has_recovery_work(persona_dir: Path) -> bool:
    """True while recovery has not finished and the gate has dealt with what the last pass handed it."""
    if str(persona_dir) in _STALLED or _load_state(persona_dir).get("done"):
        return False
    try:
        return not _queued_recovered(persona_dir)
    except (OSError, ValueError) as exc:
        logger.warning("research recovery waits: the pending queue could not be read (%s)", exc)
        return False


def _recoverable_blocks(persona_dir: Path, interest_id: str) -> dict[str, list[str]]:
    """Memory texts of the interest's stored session blocks that the parser reads as a memory and a verdict, by date."""
    path = notes_path(persona_dir, interest_id)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    headers = list(_HEADER.finditer(text))
    found: dict[str, list[str]] = {}
    for n, header in enumerate(headers):
        end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
        session = parse_session_output("NOTES:\n" + text[header.end():end].strip())
        if session.memory and "VERDICT" not in session.missing:
            found.setdefault(header.group(1), []).append(session.memory)
    return found


def _scopes(persona_dir: Path) -> dict[str, str]:
    """Interest id -> scope, read as plain data: nothing is healed, quarantined or defaulted."""
    data = read_json(persona_dir / "interests.json", default=dict, recover=Recovery.DEFAULT)
    items = data.get("interests", []) if isinstance(data, dict) else []
    return {i["id"]: i["scope"] for i in items if isinstance(i, dict) and "id" in i and isinstance(i.get("scope"), str)}


def run_recovery(persona_dir: Path, store: MemoryStore, *, now: datetime | None = None) -> dict[str, int]:
    """One pass: recover one interest's lost memories, link them, and mark the job done when nothing is left.

    Returns this pass's counts. An unexpected error stalls the persona for this process and propagates.
    """
    try:
        return _run_pass(persona_dir, store, now or datetime.now(UTC))
    except Exception:
        _STALLED.add(str(persona_dir))
        raise


def _run_pass(persona_dir: Path, store: MemoryStore, now: datetime) -> dict[str, int]:
    state = _load_state(persona_dir)
    if state.get("done"):
        return dict.fromkeys(_COUNTERS, 0)
    log_path = persona_dir / "research_log.json"
    log = ResearchLog.load(log_path) if log_path.exists() else ResearchLog()
    journal_path = persona_dir / JOURNAL_FILE
    journal = {r["key"]: r["memory_id"] for r in read_jsonl(journal_path) if "key" in r and "memory_id" in r} \
        if journal_path.exists() else {}
    skipped = set(state.get("skipped", []))
    deferred = _DEFERRED.setdefault(str(persona_dir), set())
    counts = dict.fromkeys(_COUNTERS, 0)

    open_fires = [
        f for f in log.fires if f.output_memory_id is None and _fire_key(f) not in skipped | deferred
    ]
    interest_id = open_fires[0].interest_id if open_fires else None
    linked: dict[str, str] = {}  # fire key -> memory id, to write into the log

    if interest_id is not None:
        mine = [f for f in open_fires if f.interest_id == interest_id]
        fresh = []
        for fire in mine:
            if _fire_key(fire) in journal:
                linked[_fire_key(fire)] = journal[_fire_key(fire)]
                counts["resumed"] += 1
            else:
                fresh.append(fire)
        if fresh:
            pairs = _pair(persona_dir, interest_id, fresh, log, skipped, deferred, counts)
            _write_pairs(persona_dir, store, pairs, now, linked, journal_path, deferred, counts, _scopes(persona_dir))
    if linked:
        _link(persona_dir, log_path, linked)

    remaining = [
        f for f in ResearchLog.load(log_path).fires if f.output_memory_id is None and _fire_key(f) not in skipped
    ] if log_path.exists() else []
    before = state["counts"]
    totals = {k: (before[k] if isinstance(before.get(k), int) else 0) + counts[k] for k in _COUNTERS}
    done = not remaining
    write_json(
        persona_dir / STATE_FILE,
        {
            "version": 1,
            "done": done,
            "finished_at": iso_utc(now) if done else None,
            "counts": totals,
            "skipped": sorted(skipped),
        },
    )
    logger.info(
        "research recovery pass: recovered=%d resumed=%d ambiguous=%d no_block=%d interest_missing=%d "
        "unreadable_notes=%d errors=%d%s",
        *(counts[k] for k in _COUNTERS), " (finished)" if done else "",
    )
    if remaining and all(_fire_key(f) in deferred for f in remaining):
        _STALLED.add(str(persona_dir))
    return counts


def _pair(
    persona_dir: Path,
    interest_id: str,
    fires: list[ResearchFire],
    log: ResearchLog,
    skipped: set[str],
    deferred: set[str],
    counts: dict[str, int],
) -> list[_Pair]:
    """Pair this interest's memory-less fires with recoverable blocks by UTC date; skip (and count) what is unsure."""
    try:
        blocks = _recoverable_blocks(persona_dir, interest_id)
    except UnicodeDecodeError:
        counts["unreadable_notes"] += len(fires)
        skipped.update(_fire_key(f) for f in fires)
        return []
    except OSError as exc:
        logger.warning("research recovery: notes of %s could not be read (%s)", interest_id, exc)
        counts["errors"] += 1
        deferred.update(_fire_key(f) for f in fires)
        return []

    all_by_date: dict[str, list[ResearchFire]] = {}
    for fire in log.fires:
        if fire.interest_id == interest_id:
            all_by_date.setdefault(_fire_date(fire), []).append(fire)
    fresh_by_date: dict[str, list[ResearchFire]] = {}
    for fire in fires:
        fresh_by_date.setdefault(_fire_date(fire), []).append(fire)

    pairs: list[_Pair] = []
    for date, cell in fresh_by_date.items():
        texts = blocks.get(date, [])
        if not texts:
            counts["no_block"] += len(cell)
            skipped.update(_fire_key(f) for f in cell)
        elif len(texts) != len(cell) or len(all_by_date[date]) != len(cell):
            counts["ambiguous"] += len(cell)
            skipped.update(_fire_key(f) for f in cell)
        else:
            pairs.extend(_Pair(f, t) for f, t in zip(sorted(cell, key=lambda f: f.fired_at), texts, strict=True))
    return pairs


def _write_pairs(
    persona_dir: Path,
    store: MemoryStore,
    pairs: list[_Pair],
    now: datetime,
    linked: dict[str, str],
    journal_path: Path,
    deferred: set[str],
    counts: dict[str, int],
    scopes: dict[str, str],
) -> None:
    if not pairs:
        return
    queued_ids = {m.id for m in PendingQueue(persona_dir).read_recent("research", limit=sys.maxsize)}
    for pair in pairs:
        fire = pair.fire
        memory_id = _memory_id(fire)
        try:
            if memory_id in queued_ids or store.get(memory_id, bump=False) is not None:
                counts["resumed"] += 1
            else:
                if fire.interest_id not in scopes:
                    counts["interest_missing"] += 1
                memory = build_research_memory(
                    content=pair.memory_text,
                    interest_id=fire.interest_id,
                    interest_topic=fire.topic,
                    scope=scopes.get(fire.interest_id, "either"),
                    web_urls=[],
                    web_result_count=fire.web_result_count,
                    web_used=fire.web_used,
                    trigger=fire.trigger,
                    provider_name=None,
                    searcher_name=None,
                )
                memory = dataclasses.replace(
                    memory,
                    id=memory_id,
                    created_at=fire.fired_at,
                    metadata={**memory.metadata, "recovered_from_notes": True, "recovered_at": iso_utc(now)},
                )
                route_write(store, memory, source="research")
                append_jsonl(journal_path, {"key": _fire_key(fire), "memory_id": memory_id})
                counts["recovered"] += 1
            linked[_fire_key(fire)] = memory_id
        except (OSError, sqlite3.Error) as exc:
            logger.warning("research recovery: fire %s could not be recovered (%s: %s)", _fire_key(fire), type(exc).__name__, exc)
            counts["errors"] += 1
            deferred.add(_fire_key(fire))


def _link(persona_dir: Path, log_path: Path, linked: dict[str, str]) -> None:
    """Set ``output_memory_id`` on the fires that got a memory. The log's first form is kept once beside it."""
    pre = persona_dir / PRE_RECOVERY_LOG
    if not pre.exists():
        save_with_backup_text(pre, log_path.read_text(encoding="utf-8"), keep_backups=False, newline="\n")
    log = ResearchLog.load(log_path)
    fires = tuple(
        dataclasses.replace(f, output_memory_id=linked[_fire_key(f)])
        if f.output_memory_id is None and _fire_key(f) in linked else f
        for f in log.fires
    )
    ResearchLog(fires=fires).save(log_path)
