"""Name-recall fix S85 (spec §2): the cosine-floor bootstrap is computed ONCE
per process, OFF the hot path, by the central cadence function in the first
lull, with back-off on failure. Until a cosine floor exists (bootstrap or
calibrated) the no-rerank path renders keyword results only.

All offline: the R2 scripted embedder, an in-tmp store, an injected clock.
Synthetic data only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.bridge import central_cadence, model_tier
from brain.memory import floor_calibration
from brain.memory.semantic_recall import run_semantic_recall
from brain.memory.store import MemoryStore
from tests.unit.brain.bridge.test_central_cadence import (
    NOW,
    _granting_slot,
    _persona,
    _real_jobs,
)
from tests.unit.brain.memory.test_no_rerank_cosine_path import (
    _EMBEDDER_ID,
    _QUERY,
    _REAL_COSINE_BOOTSTRAP,
    _seed_pool,
    _write_cosine_floor,
)

_JOB = "cosine_floor_bootstrap"


def _counting_bootstrap(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> list[str]:
    """Replace the computing function with a counter; on success it caches like
    the real one so `peek_cosine_bootstrap_floor` sees it."""
    calls: list[str] = []

    def _fake(embedder_model_id: str):
        calls.append(embedder_model_id)
        if fail:
            return None
        result = {
            "embedder_model_id": embedder_model_id,
            "floor": 0.5,
            "raw_fit_floor": 0.5,
            "sample_pairs": 6,
            "is_cold_start": True,
            "updated_at": None,
        }
        floor_calibration._cosine_bootstrap_floor_cache[embedder_model_id] = result  # noqa: SLF001
        return dict(result)

    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _fake)
    return calls


# ---------------------------------------------------------------------------
# The hot path never computes; keyword-only until a floor exists
# ---------------------------------------------------------------------------


def test_a_turn_before_the_lull_is_keyword_only_and_computes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        floor_calibration,
        "get_cosine_bootstrap_floor",
        lambda _id: pytest.fail("the recall hot path computed the cosine bootstrap"),
    )
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])  # < 5 candidates: the no-rerank path

    for _ in range(3):
        assert run_semantic_recall(store, tmp_path, _QUERY) is None, "keyword-only, every turn"


def test_after_the_job_ran_the_cosine_path_is_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _REAL_COSINE_BOOTSTRAP)
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])
    assert run_semantic_recall(store, tmp_path, _QUERY) is None, "before the job: keyword-only"

    assert floor_calibration.run_cosine_bootstrap(_EMBEDDER_ID, now=0.0) is not None

    result = run_semantic_recall(store, tmp_path, _QUERY)
    assert result is not None and result.path == "cosine"
    assert mems[0].id in {m.id for m in [*result.full, *result.snippet]}
    n = store._conn.execute("SELECT COUNT(*) AS n FROM cosine_floor_calibration").fetchone()["n"]  # noqa: SLF001
    assert n == 0, "the bootstrap is never persisted"


def test_a_calibrated_row_gates_the_cosine_path_without_any_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        floor_calibration,
        "get_cosine_bootstrap_floor",
        lambda _id: pytest.fail("a calibrated floor exists: nothing to bootstrap"),
    )
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])
    _write_cosine_floor(store, 0.5)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"
    assert mems[0].id in {m.id for m in [*result.full, *result.snippet]}


# ---------------------------------------------------------------------------
# Back-off on failure (injected clock), no per-turn retries
# ---------------------------------------------------------------------------


def test_a_failing_bootstrap_backs_off_exponentially_with_a_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _counting_bootstrap(monkeypatch, fail=True)
    first = floor_calibration.COSINE_BOOTSTRAP_BACKOFF_INITIAL_S
    cap = floor_calibration.COSINE_BOOTSTRAP_BACKOFF_MAX_S
    assert floor_calibration.cosine_bootstrap_due("emb", now=0.0), "never tried: due"

    now = 100.0
    delays = []
    for _ in range(9):
        assert floor_calibration.run_cosine_bootstrap("emb", now=now) is None
        delay = min(first * 2 ** len(delays), cap)
        delays.append(delay)
        assert not floor_calibration.cosine_bootstrap_due("emb", now=now + delay - 0.001)
        assert floor_calibration.cosine_bootstrap_due("emb", now=now + delay)
        now += delay

    assert delays[:3] == [first, 2 * first, 4 * first]
    assert delays[-1] == cap, "bounded"
    assert len(calls) == 9


def test_recall_turns_never_retry_a_failed_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _counting_bootstrap(monkeypatch, fail=True)
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])
    assert floor_calibration.run_cosine_bootstrap(_EMBEDDER_ID, now=0.0) is None
    assert len(calls) == 1

    for _ in range(5):
        assert run_semantic_recall(store, tmp_path, _QUERY) is None

    assert len(calls) == 1, "no per-turn retry: only the job computes"


def test_a_success_clears_the_back_off_and_is_never_due_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _counting_bootstrap(monkeypatch, fail=True)
    assert floor_calibration.run_cosine_bootstrap("emb", now=0.0) is None
    assert not floor_calibration.cosine_bootstrap_due("emb", now=1.0)

    calls_ok = _counting_bootstrap(monkeypatch)
    assert floor_calibration.run_cosine_bootstrap("emb", now=61.0) is not None
    assert floor_calibration.peek_cosine_bootstrap_floor("emb") is not None
    assert not floor_calibration.cosine_bootstrap_due("emb", now=10_000_000.0), "computed once"
    assert len(calls) == 1 and len(calls_ok) == 1


def test_run_cosine_bootstrap_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_id):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _boom)

    assert floor_calibration.run_cosine_bootstrap("emb", now=0.0) is None
    assert not floor_calibration.cosine_bootstrap_due("emb", now=1.0)


# ---------------------------------------------------------------------------
# The central-cadence job: first lull, once, not while a floor exists
# ---------------------------------------------------------------------------


def _bootstrap_only(persona_dir: Path):
    return [j for j in _real_jobs(persona_dir) if j.name == _JOB]


def _pass(persona_dir, jobs, *, idle):
    return central_cadence.run_central_pass(
        persona_dir,
        jobs,
        is_idle=lambda: idle,
        slot_available=lambda: True,
        now_wall=lambda: NOW,
    )


def test_the_job_runs_once_at_the_first_lull_and_never_inside_a_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    calls = _counting_bootstrap(monkeypatch)
    persona_dir = _persona(tmp_path)
    jobs = _bootstrap_only(persona_dir)
    assert [j.name for j in jobs] == [_JOB]

    _pass(persona_dir, jobs, idle=False)
    assert calls == [], "not inside a chat"

    _pass(persona_dir, jobs, idle=True)
    assert calls == [model_tier.model_for_tier(model_tier.TIER_EMBEDDING)], "the first lull"

    for _ in range(3):
        _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1, "computed once per process"


def test_the_job_is_not_due_while_a_calibrated_floor_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    calls = _counting_bootstrap(monkeypatch)
    persona_dir = _persona(tmp_path)
    embedder_id = model_tier.model_for_tier(model_tier.TIER_EMBEDDING)
    store = MemoryStore(persona_dir / "memories.db")
    store.write_cosine_floor(
        embedder_id, floor=0.5, raw_fit_floor=0.5, sample_pairs=300, is_cold_start=False
    )
    store.close()

    _pass(persona_dir, _bootstrap_only(persona_dir), idle=True)

    assert calls == []


def test_a_failed_job_run_backs_off_and_is_not_retried_on_the_next_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    calls = _counting_bootstrap(monkeypatch, fail=True)
    clock = {"t": 1000.0}
    monkeypatch.setattr(floor_calibration.time, "monotonic", lambda: clock["t"])
    persona_dir = _persona(tmp_path)
    jobs = _bootstrap_only(persona_dir)

    _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1
    for _ in range(5):
        clock["t"] += 10.0  # well inside the first back-off window
        _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1, "no retry inside the back-off window"

    clock["t"] += floor_calibration.COSINE_BOOTSTRAP_BACKOFF_INITIAL_S
    _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 2, "one retry once the window has elapsed"
