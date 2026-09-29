"""Name-recall fix S85 (spec §2, revised): the cosine-floor bootstrap is never
computed on the reply path. It is computed once per process at process start
(`test_floor_startup.py`); a failed one is retried at the next lull by the
central cadence job `cosine_floor_bootstrap` (due again only after chat has
happened since the failure; no time constants). Until a cosine floor exists
(bootstrap or calibrated) the no-rerank path renders keyword results only.

All offline: the R2 scripted embedder, an in-tmp store, an injected activity
marker. Synthetic data only.
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


def test_the_no_floor_turn_logs_nothing_above_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every keyword-only turn before the job ran passes through the no-floor
    branch: it must not spam the log (DEBUG only)."""
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])

    with caplog.at_level("INFO", logger="brain.memory.semantic_recall"):
        assert run_semantic_recall(store, tmp_path, _QUERY) is None

    floor_lines = [
        r
        for r in caplog.records
        if r.name == "brain.memory.semantic_recall" and "cosine floor" in r.getMessage()
    ]
    assert floor_lines == []


def test_after_the_job_ran_the_cosine_path_is_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _REAL_COSINE_BOOTSTRAP)
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])
    assert run_semantic_recall(store, tmp_path, _QUERY) is None, "before the job: keyword-only"

    assert floor_calibration.run_cosine_bootstrap(_EMBEDDER_ID) is not None

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


def test_there_are_no_back_off_constants() -> None:
    """S85 (revised): the retry rule has no time constants at all."""
    assert not [n for n in dir(floor_calibration) if "BACKOFF" in n.upper()]


def test_a_failed_bootstrap_is_retried_only_after_chat_activity_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _counting_bootstrap(monkeypatch, fail=True)
    assert floor_calibration.cosine_bootstrap_due("emb", activity_marker=1.0), "never tried"

    assert floor_calibration.run_cosine_bootstrap("emb", activity_marker=1.0) is None

    assert not floor_calibration.cosine_bootstrap_due("emb", activity_marker=1.0), "same lull"
    assert floor_calibration.cosine_bootstrap_due("emb", activity_marker=2.0), "chat happened"
    assert floor_calibration.run_cosine_bootstrap("emb", activity_marker=2.0) is None
    assert not floor_calibration.cosine_bootstrap_due("emb", activity_marker=2.0)
    assert len(calls) == 2


def test_a_persisted_row_outranks_a_cached_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Once the daily tick has written a calibrated row it supersedes the
    process-cached bootstrap, even though the cache is still populated."""
    _counting_bootstrap(monkeypatch)
    assert floor_calibration.run_cosine_bootstrap("emb") is not None
    assert floor_calibration.peek_cosine_bootstrap_floor("emb")["floor"] == pytest.approx(0.5)
    store = MemoryStore(tmp_path / "memories.db")
    assert store.get_cosine_floor("emb")["floor"] == pytest.approx(0.5), "bootstrap until a row"

    store.write_cosine_floor(
        "emb", floor=0.33, raw_fit_floor=0.33, sample_pairs=300, is_cold_start=False
    )

    got = store.get_cosine_floor("emb")
    assert got["floor"] == pytest.approx(0.33) and got["is_cold_start"] is False


def test_recall_turns_never_retry_a_failed_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _counting_bootstrap(monkeypatch, fail=True)
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])
    assert floor_calibration.run_cosine_bootstrap(_EMBEDDER_ID) is None
    assert len(calls) == 1

    for _ in range(5):
        assert run_semantic_recall(store, tmp_path, _QUERY) is None

    assert len(calls) == 1, "no per-turn retry: only the job computes"


def test_a_success_is_never_due_again(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting_bootstrap(monkeypatch, fail=True)
    assert floor_calibration.run_cosine_bootstrap("emb", activity_marker=1.0) is None
    assert not floor_calibration.cosine_bootstrap_due("emb", activity_marker=1.0)

    calls_ok = _counting_bootstrap(monkeypatch)
    assert floor_calibration.run_cosine_bootstrap("emb", activity_marker=2.0) is not None
    assert floor_calibration.peek_cosine_bootstrap_floor("emb") is not None
    for marker in (1.0, 2.0, 3.0):
        assert not floor_calibration.cosine_bootstrap_due("emb", activity_marker=marker)
    assert len(calls) == 1 and len(calls_ok) == 1


def test_run_cosine_bootstrap_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_id):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _boom)

    assert floor_calibration.run_cosine_bootstrap("emb", activity_marker=1.0) is None
    assert not floor_calibration.cosine_bootstrap_due("emb", activity_marker=1.0)


# ---------------------------------------------------------------------------
# The central-cadence job: first lull, once, not while a floor exists
# ---------------------------------------------------------------------------


def _bootstrap_only(persona_dir: Path, name: str = _JOB):
    return [j for j in _real_jobs(persona_dir) if j.name == name]


def _pass(persona_dir, jobs, *, idle):
    return central_cadence.run_central_pass(
        persona_dir,
        jobs,
        is_idle=lambda: idle,
        slot_available=lambda: True,
        now_wall=lambda: NOW,
    )


def _chat_activity(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Drive `cli_throttle.chat_activity_marker` from the test."""
    marker = {"m": 1.0}
    from brain.bridge import cli_throttle

    monkeypatch.setattr(cli_throttle, "chat_activity_marker", lambda: marker["m"])
    return marker


def test_the_job_is_the_retry_path_it_runs_at_a_lull_and_never_inside_a_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    _chat_activity(monkeypatch)
    calls = _counting_bootstrap(monkeypatch)
    persona_dir = _persona(tmp_path)
    jobs = _bootstrap_only(persona_dir)
    assert [j.name for j in jobs] == [_JOB]

    _pass(persona_dir, jobs, idle=False)
    assert calls == [], "not inside a chat"

    _pass(persona_dir, jobs, idle=True)
    assert calls == [model_tier.model_for_tier(model_tier.TIER_EMBEDDING)], "the lull"

    for _ in range(3):
        _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1, "computed; a floor exists, nothing left to retry"


def test_the_job_is_not_due_while_a_calibrated_floor_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    _chat_activity(monkeypatch)
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


def test_a_failed_bootstrap_is_retried_once_per_lull_not_every_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    marker = _chat_activity(monkeypatch)
    calls = _counting_bootstrap(monkeypatch, fail=True)
    persona_dir = _persona(tmp_path)
    jobs = _bootstrap_only(persona_dir)

    _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1
    for _ in range(5):
        _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1, "still the same lull: no retry however many passes"

    marker["m"] = 2.0  # the user chatted; the next lull is a new one
    _pass(persona_dir, jobs, idle=False)
    assert len(calls) == 1, "not during the chat"
    _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 2, "one retry at the next lull"
    for _ in range(3):
        _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 2


def test_the_job_yields_while_the_startup_computation_is_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor
    from brain.memory import floor_startup

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    _chat_activity(monkeypatch)
    calls = _counting_bootstrap(monkeypatch)
    persona_dir = _persona(tmp_path)
    jobs = _bootstrap_only(persona_dir)

    floor_startup._startup_active.set()  # noqa: SLF001
    try:
        _pass(persona_dir, jobs, idle=True)
        assert calls == [], "the startup thread is already doing this work"
    finally:
        floor_startup._startup_active.clear()  # noqa: SLF001
    _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1
