"""End-to-end ArcUpdatePass orchestrator coverage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from brain.narrative_memory import run_pass
from brain.narrative_memory.state import (
    LOG_FILENAME,
    ArcsState,
    load_or_recover,
    save_state,
)


@dataclass
class FakeAnchor:
    type: str
    ref: str
    label: str
    ts_iso: str
    lived_age_hours: float
    seed_memory_ids: tuple[str, ...]


@dataclass
class FakeMemory:
    id: str
    created_at_iso: str = "2026-05-19T10:00:00+00:00"
    state: str = "active"


@dataclass
class FakeFeltTimeState:
    lived_age_hours: float = 500.0


@dataclass
class _StubEventBus:
    events: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.events = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _stub_anchor_sweep(anchors: list[FakeAnchor]):
    def _sweep(persona_dir: Path, last_pass_ts_iso: str | None) -> list[FakeAnchor]:
        return list(anchors)

    return _sweep


def _stub_candidate_pool(candidates: list[FakeMemory]):
    def _pool(persona_dir: Path, *, opened_at_iso: str) -> list[FakeMemory]:
        return [c for c in candidates if c.created_at_iso > opened_at_iso]

    return _pool


def _stub_salience(value: float = 0.5):
    def _score(memory: FakeMemory, *, ctx: Any) -> float:
        return value

    return _score


def _stub_is_exempt_none(memory: FakeMemory) -> bool:
    return False


class _StubHebbian:
    def __init__(self, weights: dict[tuple[str, str], float] | None = None):
        self.weights = weights or {}

    def weight(self, a: str, b: str) -> float:
        return self.weights.get((a, b), self.weights.get((b, a), 0.0))


class _StubEmbeddings:
    def __init__(self, vectors: dict[str, np.ndarray] | None = None):
        self.vectors = vectors or {}

    def get(self, memory_id: str) -> np.ndarray | None:
        return self.vectors.get(memory_id)


def test_orchestrator_opens_new_arc_from_anchor(tmp_path: Path):
    bus = _StubEventBus()
    anchors = [
        FakeAnchor(
            type="dream",
            ref="dream_1",
            label="the boat one",
            ts_iso="2026-05-19T10:00:00+00:00",
            lived_age_hours=412.0,
            seed_memory_ids=("mem_seed",),
        )
    ]
    result = run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep(anchors),
        candidate_pool=_stub_candidate_pool([]),
        salience_score=_stub_salience(),
        is_exempt=_stub_is_exempt_none,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(),
    )

    assert result["opened"] == 1
    assert result["extended"] == 0
    assert result["closed"] == 0
    state = load_or_recover(tmp_path)
    assert len(state.open) == 1
    arc = next(iter(state.open.values()))
    assert arc.title == "the boat one"
    assert arc.members[0].memory_id == "mem_seed"


def test_orchestrator_extends_existing_arc_when_seed_already_member(tmp_path: Path):
    bus = _StubEventBus()
    # First pass — open arc with mem_seed
    anchors_1 = [
        FakeAnchor(
            "dream", "dream_1", "the boat one", "2026-05-19T10:00:00+00:00", 412.0, ("mem_seed",)
        )
    ]
    run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep(anchors_1),
        candidate_pool=_stub_candidate_pool([]),
        salience_score=_stub_salience(),
        is_exempt=_stub_is_exempt_none,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(),
    )
    # Second pass — a growth anchor whose seed includes mem_seed should EXTEND
    anchors_2 = [
        FakeAnchor(
            "growth",
            "growth_1",
            "the same theme",
            "2026-05-19T11:00:00+00:00",
            413.0,
            ("mem_seed", "mem_new"),
        ),
    ]
    result = run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep(anchors_2),
        candidate_pool=_stub_candidate_pool([]),
        salience_score=_stub_salience(),
        is_exempt=_stub_is_exempt_none,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(),
    )

    assert result["opened"] == 0
    assert result["extended"] >= 1
    state = load_or_recover(tmp_path)
    arc = next(iter(state.open.values()))
    assert {m.memory_id for m in arc.members} == {"mem_seed", "mem_new"}


def test_orchestrator_adds_candidate_via_hebbian(tmp_path: Path):
    bus = _StubEventBus()
    # Open arc seeded
    anchors = [
        FakeAnchor(
            "dream", "dream_1", "the boat one", "2026-05-19T10:00:00+00:00", 412.0, ("mem_seed",)
        )
    ]
    candidate = FakeMemory(id="mem_c", created_at_iso="2026-05-19T10:30:00+00:00")
    # Hebbian weight >= MEMBER_HEBBIAN_THRESHOLD (3.0)
    hebbian = _StubHebbian(weights={("mem_c", "mem_seed"): 3.0})
    run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep(anchors),
        candidate_pool=_stub_candidate_pool([candidate]),
        salience_score=_stub_salience(0.6),
        is_exempt=_stub_is_exempt_none,
        hebbian=hebbian,
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(),
    )
    state = load_or_recover(tmp_path)
    arc = next(iter(state.open.values()))
    assert any(m.memory_id == "mem_c" for m in arc.members)
    # via=hebbian should appear in log
    log_lines = (tmp_path / LOG_FILENAME).read_text().splitlines()
    added_events = [
        json.loads(line) for line in log_lines if json.loads(line)["event"] == "member_added"
    ]
    assert any(e["memory_id"] == "mem_c" and e["via"] == "hebbian" for e in added_events)


def _read_arc_events(persona_dir: Path) -> list[dict]:
    lines = (persona_dir / "arcs.log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_orchestrator_closes_stale_arc(tmp_path: Path):
    bus = _StubEventBus()
    # Pre-seed an arc whose last_extended is 100 lived-hours ago
    anchors = [
        FakeAnchor("dream", "dream_1", "old arc", "2026-05-15T10:00:00+00:00", 300.0, ("mem_seed",))
    ]
    run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep(anchors),
        candidate_pool=_stub_candidate_pool([]),
        salience_score=_stub_salience(),
        is_exempt=_stub_is_exempt_none,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(lived_age_hours=300.0),
    )
    # Second pass — no new anchors, no new candidates, lived_age jumps 100 hours
    result = run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep([]),
        candidate_pool=_stub_candidate_pool([]),
        salience_score=_stub_salience(),
        is_exempt=_stub_is_exempt_none,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(lived_age_hours=400.0),
    )
    assert result["closed"] == 1
    state = load_or_recover(tmp_path)
    assert state.open == {}
    assert len(state.recently_closed) == 1
    assert state.recently_closed[0].state == "closed"
    closed_events = [e for e in _read_arc_events(tmp_path) if e["event"] == "arc_closed"]
    assert closed_events, "expected an arc_closed event in arcs.log.jsonl"
    assert closed_events[0].get("title"), "arc_closed event must carry the arc title"


def test_orchestrator_member_cap_evicts_lowest_salience(tmp_path: Path):
    bus = _StubEventBus()
    # Hand-craft a state where an open arc has MAX_ARC_MEMBERS=50 members and
    # one new candidate is added with higher salience than the lowest.
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.policy import MAX_ARC_MEMBERS

    seed_member = ArcMember(
        memory_id="mem_seed",
        joined_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_join=412.0,
        salience_at_join=0.99,
    )
    fillers = tuple(
        ArcMember(
            memory_id=f"mem_filler_{i}",
            joined_at_iso="2026-05-19T10:00:00+00:00",
            lived_age_at_join=412.0,
            salience_at_join=0.10 + i * 0.001,  # increasing
        )
        for i in range(MAX_ARC_MEMBERS - 1)
    )
    initial_arc = Arc(
        id="arc_full",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_1",
        seed_memory_ids=("mem_seed",),
        title="full arc",
        opened_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_open=412.0,
        last_extended_at_iso="2026-05-19T11:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(seed_member,) + fillers,
    )
    initial = ArcsState(
        open={"arc_full": initial_arc},
        last_pass_ts_iso="2026-05-19T11:00:00+00:00",
    )
    save_state(tmp_path, initial)

    # New high-salience candidate joins via hebbian
    candidate = FakeMemory(id="mem_high", created_at_iso="2026-05-19T11:30:00+00:00")
    hebbian = _StubHebbian(weights={("mem_high", "mem_seed"): 5.0})
    run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep([]),
        candidate_pool=_stub_candidate_pool([candidate]),
        salience_score=_stub_salience(0.95),
        is_exempt=_stub_is_exempt_none,
        hebbian=hebbian,
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(lived_age_hours=413.0),
    )
    state = load_or_recover(tmp_path)
    arc = state.open["arc_full"]
    assert len(arc.members) == MAX_ARC_MEMBERS
    assert any(m.memory_id == "mem_high" for m in arc.members)
    # Lowest-salience filler (filler_0 with salience 0.10) should be gone
    assert not any(m.memory_id == "mem_filler_0" for m in arc.members)


def test_orchestrator_skips_exempt_memories(tmp_path: Path):
    bus = _StubEventBus()
    anchors = [
        FakeAnchor(
            "dream", "dream_1", "the boat one", "2026-05-19T10:00:00+00:00", 412.0, ("mem_seed",)
        )
    ]
    candidate = FakeMemory(id="mem_c", created_at_iso="2026-05-19T10:30:00+00:00")
    hebbian = _StubHebbian(weights={("mem_c", "mem_seed"): 5.0})
    run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep(anchors),
        candidate_pool=_stub_candidate_pool([candidate]),
        salience_score=_stub_salience(),
        is_exempt=lambda m: m.id == "mem_c",  # exempt the candidate
        hebbian=hebbian,
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(),
    )
    state = load_or_recover(tmp_path)
    arc = next(iter(state.open.values()))
    assert not any(m.memory_id == "mem_c" for m in arc.members)


def test_orchestrator_auto_closes_arc_with_all_members_lost(tmp_path: Path):
    """Open arc whose ALL members got hard-deleted -> close with reason=all_members_lost."""
    from brain.narrative_memory.arc import Arc

    initial_arc = Arc(
        id="arc_empty",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_1",
        seed_memory_ids=("mem_seed",),
        title="empty arc",
        opened_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_open=412.0,
        last_extended_at_iso="2026-05-19T11:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(),  # all members got hard-deleted
    )
    save_state(
        tmp_path,
        ArcsState(open={"arc_empty": initial_arc}, last_pass_ts_iso="2026-05-19T11:00:00+00:00"),
    )

    bus = _StubEventBus()
    run_pass(
        tmp_path,
        event_bus=bus,
        anchor_sweep=_stub_anchor_sweep([]),
        candidate_pool=_stub_candidate_pool([]),
        salience_score=_stub_salience(),
        is_exempt=_stub_is_exempt_none,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=FakeFeltTimeState(),
    )
    state = load_or_recover(tmp_path)
    assert "arc_empty" not in state.open
    assert any(a.id == "arc_empty" for a in state.recently_closed)
    log_lines = (tmp_path / LOG_FILENAME).read_text().splitlines()
    closes = [json.loads(line) for line in log_lines if json.loads(line)["event"] == "arc_closed"]
    assert any(e["arc_id"] == "arc_empty" and e["reason"] == "all_members_lost" for e in closes)


# A non-finite lived age (from the held felt-time state) is unknown, like a missing one: the arc-state save is strict
# and used to raise on any tick that stamped it (a member joining, an arc closing with no members).
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_orchestrator_member_join_with_a_non_finite_lived_age_saves_and_warns(tmp_path: Path, caplog, bad):
    anchors = [
        FakeAnchor("dream", "dream_1", "the boat one", "2026-05-19T10:00:00+00:00", 412.0, ("mem_seed",))
    ]
    candidate = FakeMemory(id="mem_c", created_at_iso="2026-05-19T10:30:00+00:00")
    with caplog.at_level(logging.WARNING, logger="brain.narrative_memory"):
        run_pass(
            tmp_path,
            event_bus=_StubEventBus(),
            anchor_sweep=_stub_anchor_sweep(anchors),
            candidate_pool=_stub_candidate_pool([candidate]),
            salience_score=_stub_salience(0.6),
            is_exempt=_stub_is_exempt_none,
            hebbian=_StubHebbian(weights={("mem_c", "mem_seed"): 3.0}),
            embeddings=_StubEmbeddings(),
            felt_time_state=FakeFeltTimeState(lived_age_hours=bad),
        )
    arc = next(iter(load_or_recover(tmp_path).open.values()))
    joined = next(m for m in arc.members if m.memory_id == "mem_c")
    assert joined.lived_age_at_join == 0.0  # the same value a missing age gets
    assert sum("lived age" in r.getMessage() and "not finite" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_orchestrator_no_member_close_with_a_non_finite_lived_age_saves_and_warns(tmp_path: Path, caplog, bad):
    from brain.narrative_memory.arc import Arc

    save_state(
        tmp_path,
        ArcsState(
            open={
                "arc_empty": Arc(
                    id="arc_empty", state="open", seed_anchor_type="dream", seed_anchor_ref="dream_1",
                    seed_memory_ids=("mem_seed",), title="empty arc", opened_at_iso="2026-05-19T10:00:00+00:00",
                    lived_age_at_open=412.0, last_extended_at_iso="2026-05-19T11:00:00+00:00", closed_at_iso=None,
                    lived_age_at_close=None, members=(),
                )
            },
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="brain.narrative_memory"):
        run_pass(
            tmp_path,
            event_bus=_StubEventBus(),
            anchor_sweep=_stub_anchor_sweep([]),
            candidate_pool=_stub_candidate_pool([]),
            salience_score=_stub_salience(),
            is_exempt=_stub_is_exempt_none,
            hebbian=_StubHebbian(),
            embeddings=_StubEmbeddings(),
            felt_time_state=FakeFeltTimeState(lived_age_hours=bad),
        )
    state = load_or_recover(tmp_path)
    (closed,) = [a for a in state.recently_closed if a.id == "arc_empty"]
    assert closed.lived_age_at_close is None  # unknown, as for a missing felt-time state
    (event,) = [e for e in _read_arc_events(tmp_path) if e["event"] == "arc_closed"]
    assert event["lived_age_hours"] == 0.0 and event["reason"] == "all_members_lost"
    assert sum("lived age" in r.getMessage() and "not finite" in r.getMessage() for r in caplog.records) == 1


# The same for the age an anchor carries (the held felt-time anchors reader can hand over a NaN): it is stamped on the
# arc and its members, which the strict state save writes.
def _pass_with(tmp_path: Path, anchors, caplog, *, lived_age: float = 500.0) -> None:
    with caplog.at_level(logging.WARNING, logger="brain.narrative_memory"):
        run_pass(
            tmp_path,
            event_bus=_StubEventBus(),
            anchor_sweep=_stub_anchor_sweep(anchors),
            candidate_pool=_stub_candidate_pool([]),
            salience_score=_stub_salience(),
            is_exempt=_stub_is_exempt_none,
            hebbian=_StubHebbian(),
            embeddings=_StubEmbeddings(),
            felt_time_state=FakeFeltTimeState(lived_age_hours=lived_age),
        )


def _anchor_warnings(caplog) -> int:
    return sum("anchor" in r.getMessage() and "lived age" in r.getMessage() and "not finite" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_orchestrator_opening_an_arc_from_an_anchor_with_a_non_finite_lived_age_saves_and_warns(
    tmp_path: Path, caplog, bad
):
    anchors = [FakeAnchor("dream", "dream_1", "the boat one", "2026-05-19T10:00:00+00:00", bad, ("mem_seed",))]
    _pass_with(tmp_path, anchors, caplog)
    arc = next(iter(load_or_recover(tmp_path).open.values()))
    assert arc.lived_age_at_open == 0.0 and [m.lived_age_at_join for m in arc.members] == [0.0]
    assert _anchor_warnings(caplog) == 1
    assert all(e["lived_age_hours"] == 0.0 for e in _read_arc_events(tmp_path))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_orchestrator_extending_an_arc_from_an_anchor_with_a_non_finite_lived_age_saves_and_warns(
    tmp_path: Path, caplog, bad
):
    first = [FakeAnchor("dream", "dream_1", "the boat one", "2026-05-19T10:00:00+00:00", 412.0, ("mem_seed",))]
    _pass_with(tmp_path, first, caplog)
    caplog.clear()
    second = [
        FakeAnchor("dream", "dream_2", "same arc", "2026-05-19T11:00:00+00:00", bad, ("mem_seed", "mem_new"))
    ]
    _pass_with(tmp_path, second, caplog)
    arc = next(iter(load_or_recover(tmp_path).open.values()))
    joined = {m.memory_id: m.lived_age_at_join for m in arc.members}
    assert joined == {"mem_seed": 412.0, "mem_new": 0.0}  # the earlier member keeps its real age
    assert _anchor_warnings(caplog) == 1
