"""Same-type matching in the dedup sweep (#619).

The sweep merges rows only within one merge type (`monologue`, `monologue_trace` and `monologue_soul_candidate` are one merge
type), except that one text may hold `fact` and `observation` rows together: then the kept row is a `fact`, and an
`observation` is the row removed, never the `fact`. Synthetic persona only: persona "Canary", user "Bob".
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from brain.engines.consolidation import _ARCHIVE_FILENAME
from brain.engines.dedup_sweep import run_dedup_sweep
from brain.memory.store import Memory, MemoryStore
from tests.test_p3_change4_dedup_sweep import _archived_ids, _crystallise, _on_first_archive


def _mem(content: str, memory_type: str, *, importance: float = 3.0, days_ago: int = 0) -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="us", importance=importance)
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=days_ago))
    return m


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _ids(store: MemoryStore) -> set[str]:
    return {m.id for m in store.list_active()}


def _archive_lines(persona_dir) -> list[dict]:
    path = persona_dir / _ARCHIVE_FILENAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _counting_judge(verdict: str = "duplicate"):
    calls: list[tuple[str, str]] = []

    def _judge(other: Memory, cand: Memory) -> str:
        calls.append((other.id, cand.id))
        return verdict

    return _judge, calls


# ------------------------------------------------------------------ exact path
def test_exact_path_does_not_merge_rows_of_different_types(tmp_path, store):
    a = _mem("Bob took the long way home.", "fact", importance=6.0, days_ago=2)
    b = _mem("bob took the long way home.", "dream", importance=2.0, days_ago=1)
    c = _mem("BOB TOOK THE LONG WAY HOME.", "monologue_emotion", importance=1.0)
    for m in (a, b, c):
        store.create(m)

    report = run_dedup_sweep(store, tmp_path)

    assert _ids(store) == {a.id, b.id, c.id}
    assert (report.groups_merged, report.rows_removed, report.merges) == (0, 0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()


def test_exact_path_keeps_the_fact_of_a_mixed_fact_observation_group(tmp_path, store):
    """The observation is older and more important than the fact, and still goes: the fact is the kept row."""
    obs = _mem("Bob's sister lives in Leeds.", "observation", importance=9.0, days_ago=3)
    fact = _mem("bob's sister lives in leeds.", "fact", importance=4.0, days_ago=1)
    for m in (obs, fact):
        store.create(m)

    report = run_dedup_sweep(store, tmp_path)

    assert _ids(store) == {fact.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(fact.id, obs.id)]
    assert (report.groups_merged, report.rows_removed) == (1, 1)
    assert store.get(fact.id, bump=False).importance == pytest.approx(9.0)
    assert _archived_ids(tmp_path) == [obs.id]
    # A second sweep finds nothing left to merge.
    again = run_dedup_sweep(store, tmp_path)
    assert (again.groups_merged, again.rows_removed) == (0, 0)
    assert _ids(store) == {fact.id}


def test_exact_path_mixed_group_of_two_facts_and_an_observation_is_one_group(tmp_path, store):
    obs = _mem("Bob keeps bees.", "observation", importance=9.5, days_ago=4)
    f_old = _mem("Bob keeps bees.", "fact", importance=5.0, days_ago=3)
    f_new = _mem("bob keeps bees.", "fact", importance=7.0, days_ago=1)
    for m in (obs, f_old, f_new):
        store.create(m)

    report = run_dedup_sweep(store, tmp_path)

    assert _ids(store) == {f_new.id}, "the kept row is the most important fact"
    assert (report.groups_merged, report.rows_removed) == (1, 2)
    assert {m.removed_id for m in report.merges} == {obs.id, f_old.id}
    assert store.get(f_new.id, bump=False).importance == pytest.approx(9.5)


def test_exact_path_leaves_a_soul_linked_observation_beside_the_fact(tmp_path, store):
    obs = _mem("Bob keeps bees.", "observation", importance=2.0, days_ago=2)
    fact = _mem("Bob keeps bees.", "fact", importance=8.0, days_ago=1)
    for m in (obs, fact):
        store.create(m)
    _crystallise(tmp_path, obs.id)

    report = run_dedup_sweep(store, tmp_path)

    assert _ids(store) == {obs.id, fact.id}
    assert (report.groups_merged, report.rows_removed, report.merges) == (0, 0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()


@pytest.mark.no_base_failure(reason="same-merge-type sweeps behave as before: two observations of one text still merge")
def test_exact_path_still_merges_two_observations(tmp_path, store):
    a = _mem("Bob hums while he cooks.", "observation", importance=3.0, days_ago=2)
    b = _mem("Bob hums while he cooks.", "observation", importance=5.0, days_ago=1)
    for m in (a, b):
        store.create(m)

    run_dedup_sweep(store, tmp_path)

    assert _ids(store) == {b.id}


@pytest.mark.no_base_failure(reason="the three monologue types were one group on base too; this pins that they stay one merge type")
def test_exact_path_merges_the_monologue_types_and_keeps_a_soul_linked_row(tmp_path, store):
    plain = _mem("A quiet evening.", "monologue", importance=4.0, days_ago=3)
    trace = _mem("a quiet evening.", "monologue_trace", importance=6.0, days_ago=2)
    cand = _mem("A QUIET EVENING.", "monologue_soul_candidate", importance=1.0, days_ago=1)
    for m in (plain, trace, cand):
        store.create(m)
    _crystallise(tmp_path, cand.id)

    run_dedup_sweep(store, tmp_path)

    assert _ids(store) == {cand.id}, "the soul-linked row stays and the other two go"


# --------------------------------------------------------------- near-dup path
def test_near_dup_pass_does_not_judge_or_merge_rows_of_different_types(tmp_path, store):
    a = _mem("Bob painted the fence.", "fact", days_ago=2)
    b = _mem("Bob was painting the fence.", "dream", days_ago=1)
    for m in (a, b):
        store.create(m)
    judge, calls = _counting_judge()

    report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert calls == []
    assert _ids(store) == {a.id, b.id}
    assert report.rows_removed == 0


def test_near_dup_fact_and_observation_removes_the_observation_when_it_is_older(tmp_path, store):
    obs = _mem("Bob painted the fence.", "observation", importance=8.0, days_ago=2)
    fact = _mem("Bob was painting the fence.", "fact", importance=3.0, days_ago=1)
    for m in (obs, fact):
        store.create(m)
    judge, calls = _counting_judge()

    report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert len(calls) == 1
    assert _ids(store) == {fact.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(fact.id, obs.id)]
    assert store.get(fact.id, bump=False).importance == pytest.approx(8.0)


@pytest.mark.no_base_failure(reason="when the observation is the later row, base also removes it; this pins the same outcome")
def test_near_dup_fact_and_observation_removes_the_observation_when_it_is_later(tmp_path, store):
    fact = _mem("Bob painted the fence.", "fact", importance=3.0, days_ago=2)
    obs = _mem("Bob was painting the fence.", "observation", importance=8.0, days_ago=1)
    for m in (fact, obs):
        store.create(m)

    run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _ids(store) == {fact.id}
    assert store.get(fact.id, bump=False).importance == pytest.approx(8.0)


def test_near_dup_pass_leaves_a_soul_linked_observation_and_its_fact_alone(tmp_path, store):
    """The soul-linked observation is the later row: the usual swap would drop the earlier fact, which is never allowed."""
    fact = _mem("Bob painted the fence.", "fact", days_ago=2)
    obs = _mem("Bob was painting the fence.", "observation", days_ago=1)
    for m in (fact, obs):
        store.create(m)
    _crystallise(tmp_path, obs.id)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _ids(store) == {fact.id, obs.id}
    assert report.rows_removed == 0
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()


@pytest.mark.no_base_failure(reason="a soul-linked fact later than a plain observation drops the observation on base too")
def test_near_dup_pass_drops_a_plain_observation_beside_a_soul_linked_fact(tmp_path, store):
    obs = _mem("Bob painted the fence.", "observation", days_ago=2)
    fact = _mem("Bob was painting the fence.", "fact", days_ago=1)
    for m in (obs, fact):
        store.create(m)
    _crystallise(tmp_path, fact.id)

    run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _ids(store) == {fact.id}


@pytest.mark.no_base_failure(reason="a pair of soul-linked rows is left alone on base as well")
def test_near_dup_pass_leaves_a_soul_linked_fact_and_observation_alone(tmp_path, store):
    obs = _mem("Bob painted the fence.", "observation", days_ago=2)
    fact = _mem("Bob was painting the fence.", "fact", days_ago=1)
    for m in (obs, fact):
        store.create(m)
    _crystallise(tmp_path, obs.id)
    _crystallise(tmp_path, fact.id)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _ids(store) == {obs.id, fact.id}
    assert report.rows_removed == 0


def test_near_dup_swap_to_keep_the_fact_retargets_earlier_merges_into_the_observation(tmp_path, store):
    """An observation that already absorbed an exact duplicate is then removed for the fact: that earlier merge is
    reported against the fact, the row that now holds it."""
    o1 = _mem("Bob mended the gate.", "observation", importance=5.0, days_ago=4)
    o2 = _mem("bob mended the gate.", "observation", importance=1.0, days_ago=3)
    fact = _mem("Bob has mended the gate twice.", "fact", importance=2.0, days_ago=1)
    for m in (o1, o2, fact):
        store.create(m)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _ids(store) == {fact.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(fact.id, o2.id), (fact.id, o1.id)]
    assert store.get(fact.id, bump=False).importance == pytest.approx(5.0)


@pytest.mark.no_base_failure(reason="pairs of one merge type work as before: the later of the pair goes")
def test_near_dup_pass_same_type_pair_removes_the_later_row(tmp_path, store):
    a = _mem("Bob likes tea.", "observation", importance=3.0, days_ago=2)
    b = _mem("Bob likes tea with honey.", "observation", importance=7.0, days_ago=1)
    for m in (a, b):
        store.create(m)

    run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _ids(store) == {a.id}
    assert store.get(a.id, bump=False).importance == pytest.approx(7.0)


# --------------------------------------------------------------------- rechecks
def test_near_dup_pass_skips_a_pair_retyped_so_it_may_not_merge_mid_pass(tmp_path, store, caplog):
    fact = _mem("Bob painted the fence.", "fact", days_ago=2)
    obs = _mem("Bob was painting the fence.", "observation", days_ago=1)
    for m in (fact, obs):
        store.create(m)

    def _judge_while_the_fact_is_retyped(_other: Memory, _cand: Memory) -> str:
        store.update(fact.id, memory_type="dream")  # another writer, between the listing and the re-read
        return "duplicate"

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=_judge_while_the_fact_is_retyped)

    assert _ids(store) == {fact.id, obs.id}
    assert report.rows_removed == 0
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(fact.id in m and obs.id in m for m in warnings), warnings


def _three_facts(store: MemoryStore) -> tuple[Memory, Memory, Memory]:
    """Same layout as the #505 tests: K (oldest, most important, kept), D2, then D1 (newest, removed first)."""
    k = _mem("Bob locked the shed.", "fact", importance=9.0, days_ago=3)
    d2 = _mem("bob locked the shed.", "fact", importance=1.0, days_ago=2)
    d1 = _mem("BOB LOCKED THE SHED.", "fact", importance=5.0, days_ago=1)
    for m in (k, d2, d1):
        store.create(m)
    return k, d2, d1


def test_exact_pass_skips_a_duplicate_retyped_mid_group(tmp_path, store, monkeypatch, caplog):
    k, d2, d1 = _three_facts(store)
    _on_first_archive(monkeypatch, lambda: store.update(d2.id, memory_type="dream"))

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path)

    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(k.id, d1.id)]
    assert _archived_ids(tmp_path) == [d1.id]
    assert _ids(store) == {k.id, d2.id}
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and d2.id in warnings[0], warnings


def test_exact_pass_stops_merging_into_a_canonical_retyped_mid_group(tmp_path, store, monkeypatch, caplog):
    k, d2, d1 = _three_facts(store)
    _on_first_archive(monkeypatch, lambda: store.update(k.id, memory_type="dream"))

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path)

    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(k.id, d1.id)]
    assert _archived_ids(tmp_path) == [d1.id]
    assert _ids(store) == {k.id, d2.id}, "no fact is merged into the row that is no longer a fact"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and d2.id in warnings[0], warnings


# ---------------------------------------------------------------- archive lines
@pytest.mark.no_base_failure(reason="the sweep's archive line already carries the removed row's memory_type; this pins its format")
def test_sweep_archive_line_keeps_its_format_and_carries_the_removed_rows_type(tmp_path, store):
    fact = _mem("Bob keeps bees.", "fact", importance=8.0, days_ago=2)
    obs = _mem("Bob keeps bees.", "observation", importance=2.0, days_ago=1)
    for m in (fact, obs):
        store.create(m)

    run_dedup_sweep(store, tmp_path)

    (line,) = _archive_lines(tmp_path)
    assert set(line) == {"archived_at", "reason", "target"}
    assert line["reason"] == "dedup_sweep"
    assert line["target"]["id"] == obs.id
    assert line["target"]["memory_type"] == "observation"
