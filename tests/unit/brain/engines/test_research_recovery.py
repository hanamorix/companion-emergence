"""Tests for the one-time recovery of research memories lost to an unread reply layout (#591).

Every persona here is built from scratch in ``tmp_path`` with the project's own writers; the notes blocks have the shapes
seen in the field (a marker and its text on one line) with invented text. Nothing is read from any real persona.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines import research_recovery
from brain.engines.consolidation import Decision, run_consolidation
from brain.engines.research import ResearchFire, ResearchLog, build_research_memory
from brain.engines.research_notes import append_session_notes, notes_path
from brain.memory.pending import PendingQueue
from brain.memory.store import MemoryStore

_PROMOTE_ALL = lambda _c, _ctx: Decision("new")  # noqa: E731
_DROP_ALL = lambda _c, _ctx: Decision("duplicate")  # noqa: E731

_BASE = datetime(2026, 9, 11, 14, 39, tzinfo=UTC)

# Stored notes blocks, in the shapes seen in the field (text invented). `{x}` is replaced per fire.
_INLINE = "read about {x}\nsecond finding on {x}\nMEMORY: It felt like {x} was answering back.\nVERDICT: continue"
_DEGRADED = "NOTES: read about {x}\nMEMORY: A degraded-shape memory about {x}.\nVERDICT: close"
_BOLD = "read about {x}\n**MEMORY:** A bold-shape memory about {x}.\n**VERDICT:** continue"
_CLEAN = "read about {x}\nthe clean notes of a fire written after the fix"
_NO_VERDICT = "read about {x}\nMEMORY: a memory about {x} with no verdict line"
_EXPECT_MEMORY = {
    _INLINE: "It felt like {x} was answering back.",
    _DEGRADED: "A degraded-shape memory about {x}.",
    _BOLD: "A bold-shape memory about {x}.",
}


@pytest.fixture(autouse=True)
def _fresh_process_state():
    research_recovery._DEFERRED.clear()
    research_recovery._STALLED.clear()
    yield
    research_recovery._DEFERRED.clear()
    research_recovery._STALLED.clear()


def _fire(interest_id: str, day: int, *, memory_id: str | None = None, web_used: bool = False) -> ResearchFire:
    return ResearchFire(
        interest_id=interest_id,
        topic=f"topic of {interest_id}",
        fired_at=_BASE + timedelta(days=day),
        trigger="days_since_human",
        web_used=web_used,
        web_result_count=3 if web_used else 0,
        output_memory_id=memory_id,
    )


def _build(tmp_path: Path, spec: dict[str, list[tuple[ResearchFire, str | None]]]) -> list[ResearchFire]:
    """Write interests.json, research_log.json and the notes files. spec: interest id -> [(fire, notes block or None)]."""
    interests = [
        {
            "id": iid, "topic": f"topic of {iid}", "pull_score": 7.0, "scope": "external", "related_keywords": [],
            "notes": "", "first_seen": "2026-08-01T10:00:00Z", "last_fed": "2026-08-01T10:00:00Z",
            "last_researched_at": None, "feed_count": 1, "source_types": ["manual"],
        }
        for iid in spec
    ]
    (tmp_path / "interests.json").write_text(json.dumps({"version": 1, "interests": interests}), encoding="utf-8")
    fires = []
    for iid, rows in spec.items():
        for fire, block in rows:
            fires.append(fire)
            if block is not None:
                append_session_notes(tmp_path, iid, block.format(x=f"{iid}-{fire.fired_at:%m%d}"), now=fire.fired_at)
    ResearchLog(fires=tuple(sorted(fires, key=lambda f: f.fired_at))).save(tmp_path / "research_log.json")
    return fires


def _log(tmp_path: Path) -> dict[tuple[str, datetime], str | None]:
    return {(f.interest_id, f.fired_at): f.output_memory_id for f in ResearchLog.load(tmp_path / "research_log.json").fires}


def _drain(store: MemoryStore, classifier=_PROMOTE_ALL) -> None:
    run_consolidation(store, persona_dir=store.persona_dir, classifier=classifier)


def _run_to_the_end(store: MemoryStore, *, classifier=_PROMOTE_ALL, limit: int = 60) -> int:
    """Run passes, letting the gate settle the queue between them, until the job is no longer due. Returns the pass count."""
    passes = 0
    while research_recovery.has_recovery_work(store.persona_dir):
        research_recovery.run_recovery(store.persona_dir, store)
        _drain(store, classifier)
        passes += 1
        assert passes < limit, "recovery does not terminate"
    return passes


def _queued_ids(persona_dir: Path) -> list[str]:
    return [m.id for m in PendingQueue(persona_dir).read_recent("research", limit=10_000)]


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _snapshot(persona_dir: Path) -> dict[str, bytes]:
    return {str(p.relative_to(persona_dir)): p.read_bytes() for p in sorted(persona_dir.rglob("*")) if p.is_file() and p.suffix != ".db"
            and not p.name.startswith("memories.db")}


# ---- C16: every lost memory is created, as the live fire would have made it -----------------------------------------------


def test_every_lost_memory_is_recovered_as_the_live_fire_would_have_made_it(tmp_path, store):
    fires = _build(tmp_path, {
        "i1": [(_fire("i1", 0, web_used=True), _INLINE), (_fire("i1", 1), _INLINE)],
        "i2": [(_fire("i2", 0), _DEGRADED)],
        "i3": [(_fire("i3", 2), _BOLD)],
    })
    _run_to_the_end(store)

    links = _log(tmp_path)
    live_keys = set(build_research_memory(
        content="x", interest_id="i", interest_topic="t", scope="s", web_urls=[], web_result_count=0,
        web_used=False, trigger="x", provider_name=None, searcher_name=None,
    ).metadata)
    blocks = {"i1": _INLINE, "i2": _DEGRADED, "i3": _BOLD}
    for fire in fires:
        memory_id = links[(fire.interest_id, fire.fired_at)]
        assert memory_id is not None
        memory = store.get(memory_id, bump=False)
        assert memory is not None
        expected = _EXPECT_MEMORY[blocks[fire.interest_id]].format(x=f"{fire.interest_id}-{fire.fired_at:%m%d}")
        assert memory.content == expected
        assert memory.memory_type == "research" and memory.domain == "us" and memory.importance == 5.0
        assert memory.created_at == fire.fired_at
        assert set(memory.metadata) == live_keys | {"recovered_from_notes", "recovered_at"}
        assert memory.metadata["interest_id"] == fire.interest_id
        assert memory.metadata["interest_topic"] == fire.topic
        assert memory.metadata["scope"] == "external"
        assert memory.metadata["triggered_by"] == "days_since_human"
        assert memory.metadata["web_used"] is fire.web_used
        assert memory.metadata["web_result_count"] == fire.web_result_count
        assert memory.metadata["recovered_from_notes"] is True
    assert store.count() == len(fires)
    assert _queued_ids(tmp_path) == []


# ---- C17: only what should be touched is touched --------------------------------------------------------------------------


def test_only_memory_less_fires_with_an_inline_memory_and_verdict_in_their_own_block_are_recovered(tmp_path, store):
    spec = {
        "iA": [
            (_fire("iA", 0), _INLINE),                       # recoverable: the one that proves the run does something
            (_fire("iA", 1, memory_id="live-1"), _INLINE),   # already has a memory (its block holds incidental lines)
            (_fire("iA", 2), _CLEAN),                        # a block written after the fix
            (_fire("iA", 3), None),                          # no block at all
            (_fire("iA", 4), _NO_VERDICT),                   # an inline memory but no verdict line
        ],
        "iB": [
            (_fire("iB", 0), _INLINE),                       # two fires on one date, one of them linked: unsure
            (dataclasses.replace(_fire("iB", 0, memory_id="live-2"), fired_at=_BASE + timedelta(hours=3)), None),
        ],
        "iC": [(_fire("iC", 0), _INLINE), (_fire("iC", 5), _CLEAN)],
    }
    _build(tmp_path, spec)
    before_notes = {p: p.read_bytes() for p in (tmp_path / "research").iterdir()}
    _run_to_the_end(store)

    links = _log(tmp_path)
    recovered = {k for k, v in links.items() if v is not None and not v.startswith("live-")}
    assert recovered == {("iA", _BASE), ("iC", _BASE)}
    assert links[("iA", _BASE + timedelta(days=1))] == "live-1"
    for day in (2, 3, 4):
        assert links[("iA", _BASE + timedelta(days=day))] is None
    assert links[("iB", _BASE)] is None
    state = json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))
    assert state["done"] is True
    assert state["counts"]["recovered"] == 2
    assert state["counts"]["ambiguous"] == 1       # iB's date
    assert state["counts"]["no_block"] == 4        # iA day 2 (clean), 3 (none), 4 (no verdict); iC day 5 (clean)
    assert {p: p.read_bytes() for p in (tmp_path / "research").iterdir()} == before_notes


def test_a_date_whose_recoverable_blocks_do_not_match_its_fires_is_skipped(tmp_path, store):
    # Two blocks of one date (an echoed header) but one fire: unsure, so nothing is made for that date.
    fire = _fire("iA", 0)
    other = _fire("iA", 3)
    _build(tmp_path, {"iA": [(fire, _INLINE), (other, _INLINE)]})
    append_session_notes(tmp_path, "iA", _INLINE.format(x="echo"), now=fire.fired_at)
    _run_to_the_end(store)
    links = _log(tmp_path)
    assert links[("iA", fire.fired_at)] is None
    assert links[("iA", other.fired_at)] is not None


# ---- C18: idempotent and crash safe -----------------------------------------------------------------------------------------


def _three_interests(tmp_path: Path) -> list[ResearchFire]:
    return _build(tmp_path, {
        f"i{n}": [(_fire(f"i{n}", d), _INLINE) for d in range(2)] for n in (1, 2, 3)
    })


def test_a_second_full_run_creates_nothing(tmp_path, store):
    _three_interests(tmp_path)
    _run_to_the_end(store)
    assert research_recovery.has_recovery_work(tmp_path) is False
    before = (_snapshot(tmp_path), store.count(), _queued_ids(tmp_path))
    counts = research_recovery.run_recovery(tmp_path, store)
    assert sum(counts.values()) == 0
    assert (_snapshot(tmp_path), store.count(), _queued_ids(tmp_path)) == before


def test_a_crash_between_the_queue_write_and_the_link_resumes_without_duplicates(tmp_path, store, monkeypatch):
    fires = _three_interests(tmp_path)
    real_save = ResearchLog.save
    state = {"raise": True}

    def flaky_save(self, path):
        if state["raise"]:
            state["raise"] = False
            raise OSError("disk went away")
        return real_save(self, path)

    monkeypatch.setattr(ResearchLog, "save", flaky_save)
    with pytest.raises(OSError):
        research_recovery.run_recovery(tmp_path, store)
    assert research_recovery.has_recovery_work(tmp_path) is False  # stalled for this process
    first = _queued_ids(tmp_path)
    assert len(first) == 2
    assert all(v is None for v in _log(tmp_path).values())

    research_recovery._STALLED.clear()  # a new process; the candidates are still waiting for the gate
    counts = research_recovery.run_recovery(tmp_path, store)
    assert counts["resumed"] == 2 and counts["recovered"] == 0
    assert _queued_ids(tmp_path) == first           # nothing was enqueued twice
    assert sum(v is not None for v in _log(tmp_path).values()) == 2
    _drain(store)
    _run_to_the_end(store)
    ids = list(_log(tmp_path).values())
    assert all(ids) and len(set(ids)) == len(fires)
    assert store.count() == len(fires)


def test_a_candidate_the_gate_already_promoted_is_linked_not_written_twice(tmp_path, store, monkeypatch):
    fires = _three_interests(tmp_path)
    real_save = ResearchLog.save
    state = {"raise": True}

    def flaky_save(self, path):
        if state["raise"]:
            state["raise"] = False
            raise OSError("crash")
        return real_save(self, path)

    monkeypatch.setattr(ResearchLog, "save", flaky_save)
    with pytest.raises(OSError):
        research_recovery.run_recovery(tmp_path, store)
    _drain(store)  # the gate promoted the first interest's two candidates before the restart
    assert store.count() == 2
    research_recovery._STALLED.clear()
    _run_to_the_end(store)
    assert store.count() == len(fires)
    assert len(set(_log(tmp_path).values())) == len(fires)


def test_a_candidate_the_gate_dropped_is_not_enqueued_again(tmp_path, store, monkeypatch):
    fires = _three_interests(tmp_path)
    real_save = ResearchLog.save
    state = {"raise": True}

    def flaky_save(self, path):
        if state["raise"]:
            state["raise"] = False
            raise OSError("crash")
        return real_save(self, path)

    monkeypatch.setattr(ResearchLog, "save", flaky_save)
    with pytest.raises(OSError):
        research_recovery.run_recovery(tmp_path, store)
    _drain(store, _DROP_ALL)  # folded or dropped as a duplicate: in neither the queue nor the store
    assert store.count() == 0 and _queued_ids(tmp_path) == []
    research_recovery._STALLED.clear()
    _run_to_the_end(store)
    journal = (tmp_path / research_recovery.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()
    assert len(journal) == len(fires)                       # each fire enqueued exactly once, ever
    assert all(v is not None for v in _log(tmp_path).values())


def test_a_queue_file_that_cannot_be_read_stops_the_pass_and_marks_nothing_done(tmp_path, store):
    _three_interests(tmp_path)
    PendingQueue(tmp_path).path.mkdir()  # an I/O fault on read (a malformed line is only skipped, with a warning)
    assert research_recovery.has_recovery_work(tmp_path) is False  # the queue cannot be read: wait
    with pytest.raises(OSError):
        research_recovery.run_recovery(tmp_path, store)
    assert not (tmp_path / research_recovery.STATE_FILE).exists()
    assert all(v is None for v in _log(tmp_path).values())


def test_a_failed_item_is_left_for_the_next_start_and_the_others_carry_on(tmp_path, store, monkeypatch):
    fires = _three_interests(tmp_path)
    real = research_recovery.route_write

    def failing(store_, memory, *, source):
        if memory.metadata["interest_id"] == "i2":
            raise OSError("queue file locked")
        return real(store_, memory, source=source)

    monkeypatch.setattr(research_recovery, "route_write", failing)
    _run_to_the_end(store)
    links = _log(tmp_path)
    assert all(links[(f.interest_id, f.fired_at)] is None for f in fires if f.interest_id == "i2")
    assert all(links[(f.interest_id, f.fired_at)] is not None for f in fires if f.interest_id != "i2")
    assert not json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))["done"]
    assert research_recovery.has_recovery_work(tmp_path) is False  # only deferred work is left: not again in this process

    monkeypatch.setattr(research_recovery, "route_write", real)
    research_recovery._DEFERRED.clear()  # a new process
    research_recovery._STALLED.clear()
    assert research_recovery.has_recovery_work(tmp_path) is True
    _run_to_the_end(store)
    assert all(v is not None for v in _log(tmp_path).values())
    assert json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))["done"] is True
    assert store.count() == len(fires)


# ---- C19: nothing the user has is rewritten or removed ----------------------------------------------------------------------


def test_notes_interests_and_every_other_log_field_are_left_alone(tmp_path, store):
    fires = _three_interests(tmp_path)
    (tmp_path / "interests.json").write_text("{ this interests file is corrupt", encoding="utf-8")  # must not be healed
    before = _snapshot(tmp_path)
    original_log = (tmp_path / "research_log.json").read_bytes()
    _run_to_the_end(store)
    after = _snapshot(tmp_path)

    for name, data in before.items():
        if name.startswith("research/") or name == "interests.json":
            assert after[name] == data, name
    assert [p for p in after if p.startswith("interests") and p != "interests.json"] == []  # no quarantine copy
    assert (tmp_path / research_recovery.PRE_RECOVERY_LOG).read_bytes() == original_log

    old = {(f.interest_id, f.fired_at): f for f in fires}
    new = {(f.interest_id, f.fired_at): f for f in ResearchLog.load(tmp_path / "research_log.json").fires}
    assert old.keys() == new.keys()
    for key, fire in old.items():
        assert fire.__dict__ == {**new[key].__dict__, "output_memory_id": None}
        assert new[key].output_memory_id is not None
    scopes = {store.get(v, bump=False).metadata["scope"] for v in (f.output_memory_id for f in new.values())}
    assert scopes == {"either"}  # the unreadable interests file is not guessed at


# ---- C20: counts are logged ------------------------------------------------------------------------------------------------


def test_each_pass_and_the_state_file_report_counts(tmp_path, store, caplog):
    _build(tmp_path, {"i1": [(_fire("i1", 0), _INLINE), (_fire("i1", 1), _CLEAN)]})
    with caplog.at_level(logging.INFO, logger="brain.engines.research_recovery"):
        _run_to_the_end(store)
    messages = [r.getMessage() for r in caplog.records if r.name == "brain.engines.research_recovery"]
    assert any("recovered=1" in m and "no_block=1" in m and "(finished)" in m for m in messages), messages
    state = json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))
    assert state["counts"]["recovered"] == 1 and state["counts"]["no_block"] == 1


# ---- C22: odd personas -----------------------------------------------------------------------------------------------------


def test_a_persona_with_nothing_to_recover_finishes_at_once(tmp_path, store):
    assert research_recovery.has_recovery_work(tmp_path) is True
    counts = research_recovery.run_recovery(tmp_path, store)
    assert sum(counts.values()) == 0
    assert research_recovery.has_recovery_work(tmp_path) is False


def test_an_empty_log_and_a_missing_interests_file_finish_at_once(tmp_path, store):
    ResearchLog().save(tmp_path / "research_log.json")
    research_recovery.run_recovery(tmp_path, store)
    assert research_recovery.has_recovery_work(tmp_path) is False


def test_a_notes_file_with_invalid_utf8_is_skipped_and_the_others_are_recovered(tmp_path, store):
    fires = _build(tmp_path, {"bad": [(_fire("bad", 0), _INLINE)], "good": [(_fire("good", 0), _INLINE)]})
    notes_path(tmp_path, "bad").write_bytes(b"## Session 2026-09-11\n\n\xff\xfe broken \x80\n")
    _run_to_the_end(store)
    links = _log(tmp_path)
    assert links[("bad", fires[0].fired_at)] is None
    assert links[("good", fires[1].fired_at)] is not None
    state = json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))
    assert state["done"] is True and state["counts"]["unreadable_notes"] == 1


def test_a_fire_whose_interest_is_no_longer_listed_is_still_recovered(tmp_path, store):
    fires = _build(tmp_path, {"gone": [(_fire("gone", 0), _INLINE)]})
    (tmp_path / "interests.json").write_text(json.dumps({"version": 1, "interests": []}), encoding="utf-8")
    _run_to_the_end(store)
    memory = store.get(_log(tmp_path)[("gone", fires[0].fired_at)], bump=False)
    assert memory.metadata["scope"] == "either"
    state = json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))
    assert state["counts"]["interest_missing"] == 1


def test_the_state_file_is_read_tolerantly(tmp_path, store):
    _build(tmp_path, {"i1": [(_fire("i1", 0), _INLINE)]})
    (tmp_path / research_recovery.STATE_FILE).write_text(
        json.dumps({"version": 99, "a_key_from_a_newer_brain": [1, 2]}), encoding="utf-8"
    )
    assert research_recovery.has_recovery_work(tmp_path) is True
    _run_to_the_end(store)
    assert store.count() == 1


# ---- C24 / C25: paced, and persona-sized -------------------------------------------------------------------------------------


def test_one_interest_is_handed_to_the_gate_per_pass_and_the_next_waits_for_the_queue_to_drain(tmp_path, store):
    _three_interests(tmp_path)
    for passes_done in range(1, 4):
        assert research_recovery.has_recovery_work(tmp_path) is True
        research_recovery.run_recovery(tmp_path, store)
        assert len(_queued_ids(tmp_path)) == 2                     # exactly one interest's two fires
        assert research_recovery.has_recovery_work(tmp_path) is False  # waits for the gate
        _drain(store)
        assert store.count() == 2 * passes_done
    assert research_recovery.has_recovery_work(tmp_path) is False  # all three interests done


def test_a_persona_sized_case_recovers_every_fire_exactly_once(tmp_path, store):
    spec = {f"i{n:02d}": [(_fire(f"i{n:02d}", d), _INLINE) for d in range(11 if n < 15 else 12)] for n in range(17)}
    fires = _build(tmp_path, spec)
    assert len(fires) == 15 * 11 + 2 * 12
    passes = 0
    while research_recovery.has_recovery_work(tmp_path):
        research_recovery.run_recovery(tmp_path, store)
        PendingQueue(tmp_path).drain()  # the gate's part is not under test here: it just empties the queue
        passes += 1
        assert passes < 100
    journal = [json.loads(line) for line in (tmp_path / research_recovery.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()]
    assert len(journal) == len(fires) == len({row["memory_id"] for row in journal})
    assert all(v is not None for v in _log(tmp_path).values())
    assert passes == 17
    assert json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))["done"] is True


def test_a_notes_file_that_cannot_be_read_is_deferred_and_the_others_are_recovered(tmp_path, store, monkeypatch):
    fires = _three_interests(tmp_path)
    real = research_recovery._recoverable_blocks

    def reading(persona_dir, interest_id):
        if interest_id == "i2":
            raise PermissionError("notes locked")
        return real(persona_dir, interest_id)

    monkeypatch.setattr(research_recovery, "_recoverable_blocks", reading)
    _run_to_the_end(store)
    links = _log(tmp_path)
    assert all(links[(f.interest_id, f.fired_at)] is None for f in fires if f.interest_id == "i2")
    assert all(links[(f.interest_id, f.fired_at)] is not None for f in fires if f.interest_id != "i2")
    state = json.loads((tmp_path / research_recovery.STATE_FILE).read_text(encoding="utf-8"))
    assert state["done"] is False and state["counts"]["errors"] == 1
    assert research_recovery.has_recovery_work(tmp_path) is False
