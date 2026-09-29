"""Name-recall fix R2 (spec §2; criteria C2a-C2c, C2e): the NO-RERANK path.

When fewer than 5 real candidates fit the rerank budget or exist, the
reranker fails to load/score, or its anchor normalization falls back, a turn
ranks its coarse cut by cosine and gates it by a COSINE floor (never
keyword-only, never an ungated ranking). Passive recall logs the cosine
path's examined candidates to the calibration log with scale "cosine"; the
search tool logs nothing (S56).

All offline: a scripted embedder, `FakeRerankerProvider`-based rerankers, an
in-tmp store. Synthetic data only.
"""

from __future__ import annotations

import json
import logging
import math
import zlib
from pathlib import Path

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.memory import floor_calibration
from brain.memory import reranker as reranker_mod
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.floor_calibration import (
    FLOOR_FIT_BETA,
    fit_threshold_fbeta,
    threshold_separates,
)
from brain.memory.hebbian import HebbianMatrix
from brain.memory.reranker import (
    _FP16_GATE_PAIRS,
    ANCHOR_POOL,
    FakeRerankerProvider,
    RerankerProvider,
)
from brain.memory.semantic_recall import MAX_STANDOUT_COUNT, run_semantic_recall
from brain.memory.store import (
    CALIBRATION_SCORE_SCALE,
    COSINE_SCORE_SCALE,
    Memory,
    MemoryStore,
)
from brain.tools.dispatch import dispatch

# Captured at import (before any fixture runs): conftest's autouse fixture
# replaces the module attribute with a never-clearing stub, but a name bound
# here still refers to the real function.
_REAL_COSINE_BOOTSTRAP = floor_calibration.get_cosine_bootstrap_floor

_DIM = 384
_EMBEDDER_ID = "r2-test-embedder"
_QUERY = "r2 cosine path query"
_RERANKER_ID = "fake-reranker"


class _Embedder(EmbeddingProvider):
    """Returns e0 for the literal query, a scripted vector for any text in
    `table`, else a deterministic pseudo-random unit vector (so the bundled
    bootstrap pairs get stable, non-degenerate cosines)."""

    def __init__(self, table: dict[str, np.ndarray] | None = None, *, batch_raises: bool = False):
        self._table = table or {}
        self._batch_raises = batch_raises
        self.batch_calls = 0

    def embed(self, text: str) -> np.ndarray:
        if text == _QUERY:
            return _query_vec()
        if text in self._table:
            return self._table[text].astype(np.float32)
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        v = rng.normal(size=_DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts):
        self.batch_calls += 1
        # Fails the bootstrap's batch (the bundled pairs) only: since
        # name-recall fix R6 the recall query is embedded by `embed_batch`
        # too, and that embed must still succeed here.
        if self._batch_raises and _QUERY not in texts:
            raise RuntimeError("simulated embed_batch failure")
        return [self.embed(t) for t in texts]

    def embedding_dim(self) -> int:
        return _DIM

    def model_id(self) -> str:
        return _EMBEDDER_ID


def _query_vec() -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def _vec_with_cosine(c: float) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = c
    v[1] = math.sqrt(max(0.0, 1.0 - c * c))
    return v


def _mem(store: MemoryStore, content: str, *, memory_type: str = "event") -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="d")
    store.create(m)
    return m


def _seed_row_vector(store: MemoryStore, memory_id: str, vec: np.ndarray) -> None:
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (np.asarray(vec, dtype=np.float32).tobytes(), _EMBEDDER_ID, memory_id),
    )
    store._conn.commit()  # noqa: SLF001


def _seed_pool(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, cosines: list[float], **embedder_kwargs
) -> list[Memory]:
    """One active embedded memory per cosine (index order = cosine order when
    `cosines` is descending), the scripted embedder patched in, and the
    embedding tier aligned so the matrix finds the seeded rows."""
    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider", lambda: _Embedder(**embedder_kwargs)
    )
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _EMBEDDER_ID)
    mems = []
    for i, c in enumerate(cosines):
        m = _mem(store, f"r2 candidate memory number {i}")
        _seed_row_vector(store, m.id, _vec_with_cosine(c))
        mems.append(m)
    return mems


def _write_cosine_floor(store: MemoryStore, floor: float) -> None:
    store.write_cosine_floor(
        _EMBEDDER_ID, floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


class _Recording(RerankerProvider):
    def __init__(self, scores: dict[str, float] | None = None, default: float = -1_000.0):
        self._fake = FakeRerankerProvider(
            scores={**dict.fromkeys(ANCHOR_POOL, 0.0), **(scores or {})}, default=default
        )
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        return self._fake.rerank(query, documents)

    def model_id(self) -> str:
        return _RERANKER_ID

    def scored_calls(self) -> list[list[str]]:
        """Calls that scored more than the single-document warm-ups."""
        return [c for c in self.calls if len(c) > 1]


def _cal_rows(store: MemoryStore) -> list:
    return store._conn.execute(  # noqa: SLF001
        "SELECT query, candidate_ids, reranker_scores, reranker_model_id, score_scale, "
        "candidate_docs FROM calibration_log ORDER BY id"
    ).fetchall()


def _ctx(tmp_path: Path, store: MemoryStore) -> dict:
    return {"store": store, "hebbian": HebbianMatrix(":memory:"), "persona_dir": tmp_path}


def _semantic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the tool's cosine-path GATING under test: name-recall fix R4
    merges the tool's keyword hits below the semantic results, and every
    seeded memory here shares the query token, so the keyword side would
    legitimately add the very memories the floor gates out. The merge itself is
    covered in `test_search_memories_keyword_merge.py`."""
    monkeypatch.setattr(
        "brain.tools.impls.search_memories._keyword_candidates", lambda *a, **k: []
    )


def _no_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kwargs):
        raise RuntimeError("simulated reranker construction failure")

    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", _boom)


# ---------------------------------------------------------------------------
# C2a: fewer than 5 real candidates -> cosine ranking gated by the cosine floor
# ---------------------------------------------------------------------------


def test_pool_below_minimum_takes_the_cosine_path_with_the_persisted_cosine_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.80, 0.45, 0.30])  # 4 < 5
    _write_cosine_floor(store, 0.5)
    rec = _Recording()
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert result.path == "cosine"
    assert result.scale == COSINE_SCORE_SCALE
    assert result.pass_mark == pytest.approx(0.5)
    assert [m.id for m in result.full] == [mems[0].id, mems[1].id], "cosine order, floor-gated"
    assert result.snippet == []
    assert result.scores[mems[0].id] == pytest.approx(0.95, abs=1e-5)
    assert rec.calls == [], "fewer than 5 real candidates: no rerank of any kind"


def test_budget_share_below_five_takes_the_cosine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pool of 6 (>= 5) but the measured cost says fewer than 5 fit the
    budget: no scoring rerank, cosine path."""
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.9, 0.8, 0.7, 0.6, 0.55, 0.1])
    _write_cosine_floor(store, 0.5)
    rec = _Recording(scores={m.content: 50.0 for m in mems})
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)
    reranker_mod._record_rerank_cost(_RERANKER_ID, 1_000, 1.0, None)  # 1 ms per padded token
    monkeypatch.setattr(reranker_mod, "LATENCY_BUDGET_SECONDS", 1e-6)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"
    assert [m.id for m in result.full] == [m.id for m in mems[:5]]
    assert rec.scored_calls() == [], "the budget fit < 5 real: nothing was scored"


def test_uncalibrated_cosine_floor_is_the_fbeta_fit_over_the_bundled_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No persisted cosine row: the gate is the bootstrap, the F-beta fit over
    the cosines of `_FP16_GATE_PAIRS[:6]` (3 relevant, 3 irrelevant) under the
    production embedder, recomputed here with plain numpy; nothing persisted."""
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _REAL_COSINE_BOOTSTRAP)
    embedder = _Embedder()
    labeled = _FP16_GATE_PAIRS[:6]
    cosines = []
    for q, d in labeled:
        a, b = embedder.embed(q), embedder.embed(d)
        cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    labeled_cosines = list(zip(cosines, ["relevant"] * 3 + ["irrelevant"] * 3, strict=True))
    expected = fit_threshold_fbeta(labeled_cosines, beta=FLOOR_FIT_BETA)
    assert threshold_separates(labeled_cosines, expected), "test precondition: a gating bootstrap"

    store = MemoryStore(tmp_path / "memories.db")
    high = [0.9, 0.8]
    clearing_low = expected + 0.01
    below = expected - 0.02
    mems = _seed_pool(store, monkeypatch, [*high, clearing_low, below])
    # S85 (revised): the bootstrap is computed at process start, not by the recall.
    assert floor_calibration.run_cosine_bootstrap(_EMBEDDER_ID) is not None

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert result.pass_mark == pytest.approx(expected)
    cleared = {m.id for m in result.full}
    assert mems[0].id in cleared and mems[1].id in cleared
    assert mems[3].id not in cleared, "a cosine below the bootstrap floor never surfaces"
    n = store._conn.execute("SELECT COUNT(*) AS n FROM cosine_floor_calibration").fetchone()["n"]  # noqa: SLF001
    assert n == 0, "the bootstrap is never persisted"


def test_cosine_bootstrap_failure_yields_no_semantic_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed cosine bootstrap means no gate is possible: no semantic
    results (keyword-only turn), never an ungated cosine ranking."""
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _REAL_COSINE_BOOTSTRAP)
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97], batch_raises=True)
    assert floor_calibration.run_cosine_bootstrap(_EMBEDDER_ID) is None

    assert run_semantic_recall(store, tmp_path, _QUERY) is None

    (row,) = _cal_rows(store)  # the examined candidates are still training data
    assert row["score_scale"] == COSINE_SCORE_SCALE


# ---------------------------------------------------------------------------
# C2b: reranker forced to fail -> the cosine path, not keyword-only
# ---------------------------------------------------------------------------


class _ScoreRaises(_Recording):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise RuntimeError("simulated reranker scoring failure")


@pytest.mark.parametrize("failure", ["construction", "scoring"])
def test_reranker_failure_takes_the_cosine_path_not_keyword_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.9, 0.85, 0.8, 0.75, 0.3, 0.2])  # 7 >= 5
    _write_cosine_floor(store, 0.5)
    if failure == "construction":
        _no_reranker(monkeypatch)
    else:
        monkeypatch.setattr(
            "brain.memory.reranker.build_reranker_provider", lambda **kw: _ScoreRaises()
        )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None, "a reranker failure must not demote the turn to keyword-only"
    assert result.path == "cosine" and result.scale == COSINE_SCORE_SCALE
    assert [m.id for m in [*result.full, *result.snippet]] == [m.id for m in mems[:5]]


def test_reranker_failure_on_the_tool_takes_the_cosine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.9, 0.85, 0.8, 0.75, 0.3, 0.2])
    _write_cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)
    _semantic_only(monkeypatch)

    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic", "limit": 9}, **_ctx(tmp_path, store))

    assert res["mode"] == "semantic"
    assert [m["id"] for m in res["memories"]] == [m.id for m in mems[:5]]


def test_rerank_floor_bootstrap_failure_takes_the_cosine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No persisted rerank floor and the rerank bootstrap itself fails: the
    reranker cannot gate this turn, so it is a cosine-path turn. The rerank
    that DID score is still logged (as before R2: its normalized scores are the
    data the rerank floor's own fit needs, review F4), then the cosine path logs
    its own cosine-scale row: two rows, each on its own true scale."""
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.9, 0.85, 0.8, 0.75, 0.3])
    _write_cosine_floor(store, 0.5)
    rec = _Recording({m.content: 50.0 for m in mems})
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)

    def _boom(model_id: str):
        raise RuntimeError("simulated rerank bootstrap failure")

    monkeypatch.setattr("brain.memory.reranker._bootstrap_reranker_provider", _boom)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"
    rows = _cal_rows(store)
    assert [(r["score_scale"], r["reranker_model_id"]) for r in rows] == [
        (CALIBRATION_SCORE_SCALE, _RERANKER_ID),
        (COSINE_SCORE_SCALE, _EMBEDDER_ID),
    ]


# ---------------------------------------------------------------------------
# C2c fallback clause: a failed normalization is the cosine path, never a raw gate
# ---------------------------------------------------------------------------


def test_normalization_fallback_takes_the_cosine_path_and_logs_no_normalized_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.95, 0.9, 0.85, 0.8, 0.75, 0.3])
    _write_cosine_floor(store, 0.5)
    _seed_rerank_floor(store)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: _Recording())
    monkeypatch.setattr(
        reranker_mod,
        "rerank_for_recall",
        lambda *a, **kw: reranker_mod.RecallRerank(
            width=5, reranked=False, hand_off="normalization", normalization=None, measured=False
        ),
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"
    scales = [r["score_scale"] for r in _cal_rows(store)]
    assert scales == [COSINE_SCORE_SCALE], "no 'normalized' row for a paragraph that took the cosine path"


def _seed_rerank_floor(store: MemoryStore, floor: float = 0.0) -> None:
    store.write_reranker_floor(
        _RERANKER_ID, floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


def test_reranked_path_stamps_the_normalized_scale_and_reports_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The row's scale equals the scale its scores are on, and the result
    reports the scale and pass mark actually applied."""
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.9, 0.85, 0.8, 0.75, 0.3])
    _seed_rerank_floor(store, floor=1.0)
    rec = _Recording({mems[0].content: 5.0, mems[1].content: 3.0}, default=-50.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "reranked"
    assert result.scale == CALIBRATION_SCORE_SCALE and result.pass_mark == pytest.approx(1.0)
    assert [m.id for m in result.full] == [mems[0].id, mems[1].id]
    (row,) = _cal_rows(store)
    assert row["score_scale"] == CALIBRATION_SCORE_SCALE and row["reranker_model_id"] == _RERANKER_ID


# ---------------------------------------------------------------------------
# C2e (S60, S56): the cosine-path calibration row; the tool writes none
# ---------------------------------------------------------------------------


def test_cosine_path_logs_one_row_of_the_first_nine_examined_pass_or_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    cosines = [0.99 - 0.01 * i for i in range(50)]  # 50 candidates, descending
    mems = _seed_pool(store, monkeypatch, cosines)
    _write_cosine_floor(store, 0.965)  # only the first 3 (0.99, 0.98, 0.97) clear
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert [m.id for m in [*result.full, *result.snippet]] == [m.id for m in mems[:3]]
    (row,) = _cal_rows(store)
    ids = json.loads(row["candidate_ids"])
    scores = json.loads(row["reranker_scores"])
    docs = json.loads(row["candidate_docs"])
    assert len(ids) == MAX_STANDOUT_COUNT == 9, "9 logged of 50, pass or fail (3 cleared)"
    assert ids == [m.id for m in mems[:9]], "the path's own order, best cosine first"
    assert scores == pytest.approx(cosines[:9], abs=1e-5)
    assert docs == [m.content for m in mems[:9]], "documents aligned 1:1 with ids"
    assert row["score_scale"] == COSINE_SCORE_SCALE
    assert row["reranker_model_id"] == _EMBEDDER_ID
    assert row["query"] == _QUERY


def test_cosine_path_logs_every_candidate_when_fewer_than_nine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.9, 0.8, 0.4, 0.3])
    _write_cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    run_semantic_recall(store, tmp_path, _QUERY)

    (row,) = _cal_rows(store)
    assert json.loads(row["candidate_ids"]) == [m.id for m in mems]


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_the_tool_writes_no_calibration_row_on_either_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.9, 0.85, 0.8, 0.75, 0.3])
    _write_cosine_floor(store, 0.5)
    _seed_rerank_floor(store, floor=1.0)
    if path == "cosine":
        _no_reranker(monkeypatch)
    else:
        rec = _Recording({m.content: 5.0 for m in mems}, default=-50.0)
        monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)
    before = len(_cal_rows(store))

    for mode in ("semantic", "lexical"):
        dispatch("search_memories", {"query": _QUERY, "mode": mode}, **_ctx(tmp_path, store))

    assert len(_cal_rows(store)) == before == 0


def test_the_tool_gates_the_cosine_path_by_the_cosine_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.95, 0.9, 0.7, 0.45, 0.3])  # 5, all real
    _write_cosine_floor(store, 0.6)
    _no_reranker(monkeypatch)
    _semantic_only(monkeypatch)

    res = dispatch("search_memories", {"query": _QUERY, "mode": "semantic", "limit": 9}, **_ctx(tmp_path, store))

    assert res["mode"] == "semantic"
    assert [m["id"] for m in res["memories"]] == [m.id for m in mems[:3]]


def test_nothing_clearing_the_cosine_floor_falls_through_to_keyword(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.4, 0.3, 0.2])
    _write_cosine_floor(store, 0.9)
    assert run_semantic_recall(store, tmp_path, _QUERY) is None


def test_the_cosine_path_is_taken_with_an_info_log_not_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _seed_pool(store, monkeypatch, [0.9, 0.8, 0.4])
    _write_cosine_floor(store, 0.5)
    with caplog.at_level(logging.INFO):
        run_semantic_recall(store, tmp_path, _QUERY)
    assert any("taking the cosine path" in r.getMessage() for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
