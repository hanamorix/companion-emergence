"""Name-recall fix R2 (spec §2; C2d, INV-I9, S25/S38/S60): the cosine floor.

The floor the no-rerank path gates on lives in its own memories.db table keyed
by the EMBEDDER model id (never a row in `reranker_floor_calibration`), is
bootstrapped by the same F-beta fit over the same bundled pairs until the
daily tick calibrates it, and is fit from `cosine`-scale calibration rows
only: floor fits never mix scales. Offline; synthetic data only.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from brain.bridge.supervisor import _run_calibration_tick
from brain.memory import floor_calibration
from brain.memory.embeddings import FakeEmbeddingProvider
from brain.memory.floor_calibration import (
    FLOOR_FIT_BETA,
    FLOOR_FIT_MIN_LABELED_PAIRS,
    derive_and_persist_cosine_floor,
    fit_threshold_fbeta,
)
from brain.memory.relevance_judge import FakeRelevanceJudgeProvider, label_calibration_sample
from brain.memory.reranker import _FP16_GATE_PAIRS
from brain.memory.store import (
    CALIBRATION_SCORE_SCALE,
    COSINE_SCORE_SCALE,
    Memory,
    MemoryStore,
)

_REAL_COSINE_BOOTSTRAP = floor_calibration.get_cosine_bootstrap_floor

_EMBEDDER_ID = "fake-256"  # FakeEmbeddingProvider().model_id(), what the suite's fake embedder reports
_RERANKER_ID = "fake-reranker"


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


_SEP_ID = "separable-embedder"
_SEP_DIM = 16


def _sep_table(*, relevant: list[float], irrelevant: list[float]) -> tuple[dict[str, np.ndarray], list[float]]:
    """Vectors for the six bundled pairs whose cosines are exactly `relevant`
    (pairs 1-3) and `irrelevant` (pairs 4-6), with the texts the pairs share
    (the calm-down query is also pair 4's document and pair 6's query) placed
    so every pair keeps its own cosine. Returns (table, cosines in pair order)."""
    pairs = _FP16_GATE_PAIRS[:6]
    (q1, d1), (q2, d2), (q3, d3), (q4, d4), (q5, d5), (q6, d6) = pairs
    assert d4 == q1 and q6 == q1, "the bundled set's shared text this fixture relies on"

    def unit(i: int) -> np.ndarray:
        v = np.zeros(_SEP_DIM, dtype=np.float32)
        v[i] = 1.0
        return v

    def near(i: int, j: int, c: float) -> np.ndarray:
        return (c * unit(i) + float(np.sqrt(1.0 - c * c)) * unit(j)).astype(np.float32)

    table = {
        q1: unit(0),
        d1: near(0, 1, relevant[0]),
        q2: unit(2), d2: near(2, 3, relevant[1]),
        q3: unit(4), d3: near(4, 5, relevant[2]),
        q4: near(0, 6, irrelevant[0]),
        q5: unit(7), d5: near(7, 8, irrelevant[1]),
        d6: near(0, 9, irrelevant[2]),
    }
    return table, [*relevant, *irrelevant]


class _SeparableEmbedder(FakeEmbeddingProvider):
    def __init__(self, table: dict[str, np.ndarray]) -> None:
        super().__init__(dim=_SEP_DIM)
        self._table = table

    def embed(self, text: str) -> np.ndarray:
        return self._table[text]

    def model_id(self) -> str:
        return _SEP_ID


def _insert_row(
    store: MemoryStore,
    scores: list[float],
    labels: list[str] | None,
    *,
    model_id: str,
    scale: str,
    day_bucket: str | None = None,
) -> None:
    label_json = json.dumps(labels) if labels is not None else None
    cols = "query, candidate_ids, reranker_scores, reranker_model_id, local_judge_label, score_scale"
    vals: list = [
        "q",
        json.dumps([f"m{i}" for i in range(len(scores))]),
        json.dumps(scores),
        model_id,
        label_json,
        scale,
    ]
    if day_bucket is not None:
        cols = "day_bucket, " + cols
        vals = [day_bucket, *vals]
    store._conn.execute(  # noqa: SLF001
        f"INSERT INTO calibration_log ({cols}) VALUES ({','.join('?' * len(vals))})", vals
    )
    store._conn.commit()  # noqa: SLF001


# ---------------------------------------------------------------------------
# The table: own table, keyed by embedder id, never a reranker_floor row
# ---------------------------------------------------------------------------


def test_cosine_floor_round_trips_in_its_own_table_keyed_by_embedder_id(store: MemoryStore) -> None:
    assert store.get_persisted_cosine_floor("emb-a") is None
    store.write_cosine_floor("emb-a", floor=0.41, raw_fit_floor=0.41, sample_pairs=300, is_cold_start=False)
    store.write_cosine_floor("emb-b", floor=0.77, raw_fit_floor=0.77, sample_pairs=250, is_cold_start=False)
    a = store.get_persisted_cosine_floor("emb-a")
    assert a is not None and a["floor"] == pytest.approx(0.41) and a["sample_pairs"] == 300
    assert store.get_persisted_cosine_floor("emb-b")["floor"] == pytest.approx(0.77)
    # upsert replaces wholesale
    store.write_cosine_floor("emb-a", floor=0.5, raw_fit_floor=0.5, sample_pairs=400, is_cold_start=False)
    assert store.get_persisted_cosine_floor("emb-a")["floor"] == pytest.approx(0.5)
    n = store._conn.execute("SELECT COUNT(*) AS n FROM reranker_floor_calibration").fetchone()["n"]  # noqa: SLF001
    assert n == 0, "a cosine floor is never a row in reranker_floor_calibration (its stale-scale refit)"


def test_get_cosine_floor_prefers_the_persisted_row_over_the_bootstrap(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", lambda _id: pytest.fail("bootstrap used"))
    store.write_cosine_floor(_EMBEDDER_ID, floor=0.33, raw_fit_floor=0.33, sample_pairs=9, is_cold_start=False)
    assert store.get_cosine_floor(_EMBEDDER_ID)["floor"] == pytest.approx(0.33)


def test_get_cosine_floor_never_computes_the_bootstrap_on_a_miss(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S85: the hot-path read only peeks the process cache. With neither a
    persisted row nor a cached bootstrap it returns None and computes nothing."""
    monkeypatch.setattr(
        floor_calibration,
        "get_cosine_bootstrap_floor",
        lambda _id: pytest.fail("the hot path computed the bootstrap"),
    )
    assert store.get_cosine_floor(_SEP_ID) is None


def test_get_cosine_floor_serves_the_never_persisted_bootstrap_once_the_job_computed_it(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, _ = _sep_table(relevant=[0.9, 0.85, 0.8], irrelevant=[0.2, 0.1, 0.2])
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: _SeparableEmbedder(table))
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _REAL_COSINE_BOOTSTRAP)
    assert floor_calibration.run_cosine_bootstrap(_SEP_ID) is not None
    floor = store.get_cosine_floor(_SEP_ID)
    assert floor is not None and floor["is_cold_start"] is True and floor["updated_at"] is None
    assert store.get_persisted_cosine_floor(_SEP_ID) is None


def test_a_persisted_cosine_row_does_not_trip_the_rerank_floor_stale_scale_check(store: MemoryStore) -> None:
    """The reason for a separate table: `reranker_floor_is_stale` reads only
    `reranker_floor_calibration`, so a cosine row can never be read as a raw
    (stale-scale) rerank floor."""
    store.write_cosine_floor(_RERANKER_ID, floor=0.4, raw_fit_floor=0.4, sample_pairs=9, is_cold_start=False)
    assert store.reranker_floor_is_stale(_RERANKER_ID) is True  # absent, as before: unaffected
    assert store.get_persisted_reranker_floor(_RERANKER_ID) is None


# ---------------------------------------------------------------------------
# INV-I9: the idempotent, verified on-open migration
# ---------------------------------------------------------------------------


def _table_counts(db: Path, skip: set[str]) -> dict[str, int]:
    con = sqlite3.connect(db)
    try:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "AND name NOT LIKE 'memories_fts%'"
            )
        ]
        return {n: con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names if n not in skip}
    finally:
        con.close()


def _memories_checksum(db: Path) -> str:
    con = sqlite3.connect(db)
    try:
        h = hashlib.sha256()
        for row in con.execute(
            "SELECT id, content, importance, recall_count, embedding FROM memories ORDER BY id"
        ):
            h.update(repr(row).encode())
        return h.hexdigest()
    finally:
        con.close()


def test_inv_i9_opening_a_legacy_db_creates_the_table_idempotently_and_touches_nothing(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    legacy = MemoryStore(db)
    for i in range(3):
        m = Memory.create_new(content=f"legacy memory {i}", memory_type="event", domain="d")
        legacy.create(m)
        legacy._conn.execute(  # noqa: SLF001
            "UPDATE memories SET embedding = ?, embedding_model_id = 'x', recall_count = ? WHERE id = ?",
            (np.arange(4, dtype=np.float32).tobytes(), float(i), m.id),
        )
    legacy._conn.commit()  # noqa: SLF001
    legacy.write_reranker_floor("old-reranker", floor=-1.5, raw_fit_floor=-1.5, sample_pairs=7, is_cold_start=False)
    legacy.log_calibration_sample("q", ["a"], [1.0], "old-reranker")
    # Make it a LEGACY-schema database: no cosine-floor table.
    legacy._conn.execute("DROP TABLE cosine_floor_calibration")  # noqa: SLF001
    legacy._conn.commit()  # noqa: SLF001
    legacy.close()
    skip = {"cosine_floor_calibration"}
    counts_before = _table_counts(db, skip)
    checksum_before = _memories_checksum(db)
    floor_rows_before = sqlite3.connect(db).execute("SELECT * FROM reranker_floor_calibration").fetchall()
    assert "cosine_floor_calibration" not in _table_counts(db, set())

    first = MemoryStore(db)
    first.close()
    assert "cosine_floor_calibration" in _table_counts(db, set()), "opening the legacy DB creates the table"
    second = MemoryStore(db)  # reopening is a no-op
    second.close()

    assert _table_counts(db, skip) == counts_before, "no pre-existing table's row count changed"
    assert _memories_checksum(db) == checksum_before
    assert sqlite3.connect(db).execute("SELECT * FROM reranker_floor_calibration").fetchall() == floor_rows_before
    assert _table_counts(db, set())["cosine_floor_calibration"] == 0


# ---------------------------------------------------------------------------
# log_calibration_sample stamps the scale it is given
# ---------------------------------------------------------------------------


def test_log_calibration_sample_stamps_the_true_scale(store: MemoryStore) -> None:
    store.log_calibration_sample("q", ["a"], [0.5], _EMBEDDER_ID, score_scale=COSINE_SCORE_SCALE)
    store.log_calibration_sample("q", ["a"], [1.5], _RERANKER_ID)  # default: normalized
    rows = store._conn.execute("SELECT score_scale FROM calibration_log ORDER BY id").fetchall()  # noqa: SLF001
    assert [r["score_scale"] for r in rows] == [COSINE_SCORE_SCALE, CALIBRATION_SCORE_SCALE]
    with pytest.raises(ValueError):
        store.log_calibration_sample("q", ["a"], [1.5], _RERANKER_ID, score_scale="raw")


def test_labeled_pairs_read_only_the_requested_scale(store: MemoryStore) -> None:
    _insert_row(store, [1.0, 2.0], ["relevant", "irrelevant"], model_id="m", scale=CALIBRATION_SCORE_SCALE)
    _insert_row(store, [0.9], ["relevant"], model_id="m", scale=COSINE_SCORE_SCALE)
    assert sorted(s for s, _ in store.labeled_calibration_pairs("m")) == [1.0, 2.0]
    assert [s for s, _ in store.labeled_calibration_pairs("m", COSINE_SCORE_SCALE)] == [0.9]


# ---------------------------------------------------------------------------
# The cosine bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_is_the_fbeta_fit_over_the_bundled_pairs_cosines_cached_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table, cosines = _sep_table(relevant=[0.9, 0.85, 0.8], irrelevant=[0.2, 0.1, 0.2])
    embedder = _SeparableEmbedder(table)
    calls = {"batch": 0}
    real_batch = embedder.embed_batch

    def counting_batch(texts):
        calls["batch"] += 1
        return real_batch(texts)

    monkeypatch.setattr(embedder, "embed_batch", counting_batch)
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: embedder)
    # Recomputed with plain numpy from the vectors the embedder hands out.
    labeled = _FP16_GATE_PAIRS[:6]
    measured = []
    for q, d in labeled:
        a, b = table[q], table[d]
        measured.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert measured == pytest.approx(cosines, abs=1e-6)
    expected = fit_threshold_fbeta(
        list(zip(measured, ["relevant"] * 3 + ["irrelevant"] * 3, strict=True)), beta=FLOOR_FIT_BETA
    )
    assert 0.2 < expected <= 0.8, "test precondition: the fit is a real threshold between the classes"

    first = _REAL_COSINE_BOOTSTRAP(_SEP_ID)
    second = _REAL_COSINE_BOOTSTRAP(_SEP_ID)

    assert first is not None and first["floor"] == pytest.approx(expected)
    assert first["embedder_model_id"] == _SEP_ID and first["is_cold_start"] is True
    assert first["sample_pairs"] == 6 and second == first
    assert calls["batch"] == 1, "one embed_batch for the whole bundled set, computed once per process"


def test_bootstrap_for_a_different_embedder_id_is_none_and_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    table, _ = _sep_table(relevant=[0.9, 0.85, 0.8], irrelevant=[0.2, 0.1, 0.2])
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: _SeparableEmbedder(table))
    assert _REAL_COSINE_BOOTSTRAP("some-other-embedder") is None
    assert _REAL_COSINE_BOOTSTRAP(_SEP_ID) is not None, "a mismatch is not cached against the real id"


def test_bootstrap_failure_returns_none_and_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    table, _ = _sep_table(relevant=[0.9, 0.85, 0.8], irrelevant=[0.2, 0.1, 0.2])

    class _Boom(_SeparableEmbedder):
        fail = True

        def embed_batch(self, texts):
            if self.fail:
                raise RuntimeError("simulated embed failure")
            return super().embed_batch(texts)

    embedder = _Boom(table)
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: embedder)
    assert _REAL_COSINE_BOOTSTRAP(_SEP_ID) is None
    embedder.fail = False
    assert _REAL_COSINE_BOOTSTRAP(_SEP_ID) is not None


def test_bootstrap_refuses_a_fit_that_gates_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An embedder that scores the decoys at or above the relevant pairs fits
    'serve everything' (a threshold below every cosine): no gate is possible,
    so no bootstrap is served (keyword-only), never a silent pass-all floor."""
    table, _ = _sep_table(relevant=[0.80, 0.79, 0.78], irrelevant=[0.85, 0.86, 0.87])
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: _SeparableEmbedder(table))
    assert _REAL_COSINE_BOOTSTRAP(_SEP_ID) is None


# ---------------------------------------------------------------------------
# C2d: the daily tick fits each scale from its own rows only
# ---------------------------------------------------------------------------


def _seed_scale(store: MemoryStore, rng, *, n: int, model_id: str, scale: str, rel: float, irr: float, sd: float):
    pairs = []
    for s in rng.normal(rel, sd, n // 2):
        _insert_row(store, [float(s)], ["relevant"], model_id=model_id, scale=scale)
        pairs.append((float(s), "relevant"))
    for s in rng.normal(irr, sd, n - n // 2):
        _insert_row(store, [float(s)], ["irrelevant"], model_id=model_id, scale=scale)
        pairs.append((float(s), "irrelevant"))
    return pairs


def test_c2d_the_tick_fits_each_scale_from_its_own_rows_never_pooled() -> None:
    """Both scales are seeded under BOTH model ids (as if the ids collided), so
    only the scale filter, not the id filter, keeps a fit from being pooled."""
    n = FLOOR_FIT_MIN_LABELED_PAIRS + 20
    rng = np.random.default_rng(7)
    norm = {"rel": 3.0, "irr": 1.0, "sd": 0.7}
    cos = {"rel": 0.9, "irr": 0.1, "sd": 0.1}
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        seed = MemoryStore(pd / "memories.db", integrity_check=False)
        norm_pairs = _seed_scale(
            seed, rng, n=n, model_id=_RERANKER_ID, scale=CALIBRATION_SCORE_SCALE, **norm
        )
        _seed_scale(seed, rng, n=n, model_id=_EMBEDDER_ID, scale=CALIBRATION_SCORE_SCALE, **norm)
        cos_pairs = _seed_scale(seed, rng, n=n, model_id=_EMBEDDER_ID, scale=COSINE_SCORE_SCALE, **cos)
        _seed_scale(seed, rng, n=n, model_id=_RERANKER_ID, scale=COSINE_SCORE_SCALE, **cos)
        seed.close()

        expected_norm = fit_threshold_fbeta(norm_pairs, beta=FLOOR_FIT_BETA)
        expected_cos = fit_threshold_fbeta(cos_pairs, beta=FLOOR_FIT_BETA)
        pooled = fit_threshold_fbeta(norm_pairs + cos_pairs, beta=FLOOR_FIT_BETA)
        assert pooled != pytest.approx(expected_norm) and pooled != pytest.approx(expected_cos), (
            "test precondition: a pooled fit would differ from both per-scale fits"
        )

        assert _run_calibration_tick(pd) is True

        check = MemoryStore(pd / "memories.db", integrity_check=False)
        rerank_row = check.get_persisted_reranker_floor(_RERANKER_ID)
        cosine_row = check.get_persisted_cosine_floor(_EMBEDDER_ID)
        check.close()
    assert rerank_row is not None and rerank_row["floor"] == pytest.approx(expected_norm)
    assert cosine_row is not None and cosine_row["floor"] == pytest.approx(expected_cos)
    assert cosine_row["is_cold_start"] is False and cosine_row["sample_pairs"] == len(cos_pairs)


def test_cosine_floor_derivation_holds_the_prior_below_the_minimum_and_writes_nothing_without_one(
    store: MemoryStore,
) -> None:
    for i in range(5):
        _insert_row(store, [0.5 + 0.01 * i], ["relevant"], model_id=_EMBEDDER_ID, scale=COSINE_SCORE_SCALE)
    starved = derive_and_persist_cosine_floor(store, _EMBEDDER_ID)
    assert starved.accepted is False and starved.held_for_data_starvation and starved.floor is None
    assert store.get_persisted_cosine_floor(_EMBEDDER_ID) is None

    store.write_cosine_floor(_EMBEDDER_ID, floor=0.42, raw_fit_floor=0.42, sample_pairs=999, is_cold_start=False)
    held = derive_and_persist_cosine_floor(store, _EMBEDDER_ID)
    assert held.accepted is False and held.floor == pytest.approx(0.42)
    assert store.get_persisted_cosine_floor(_EMBEDDER_ID)["floor"] == pytest.approx(0.42)


def test_cosine_fit_ignores_normalized_rows_of_the_same_model_id(store: MemoryStore) -> None:
    n = FLOOR_FIT_MIN_LABELED_PAIRS
    rng = np.random.default_rng(3)
    cos_pairs = _seed_scale(store, rng, n=n, model_id="same-id", scale=COSINE_SCORE_SCALE, rel=0.6, irr=0.3, sd=0.05)
    _seed_scale(store, rng, n=n, model_id="same-id", scale=CALIBRATION_SCORE_SCALE, rel=5.0, irr=-5.0, sd=1.0)
    out = derive_and_persist_cosine_floor(store, "same-id")
    assert out.accepted and out.floor == pytest.approx(fit_threshold_fbeta(cos_pairs, beta=FLOOR_FIT_BETA))


# ---------------------------------------------------------------------------
# The tick labels the two scales separately (S25)
# ---------------------------------------------------------------------------


def test_labeling_samples_each_scale_separately(store: MemoryStore) -> None:
    for _ in range(4):
        _insert_row(store, [1.0], None, model_id=_RERANKER_ID, scale=CALIBRATION_SCORE_SCALE)
    for _ in range(4):
        _insert_row(store, [0.5], None, model_id=_EMBEDDER_ID, scale=COSINE_SCORE_SCALE)

    labeled = label_calibration_sample(store, judge=FakeRelevanceJudgeProvider(), sample_rows=2)

    assert labeled == 4, "2 rows of each scale, neither starving the other"
    by_scale = dict(
        store._conn.execute(  # noqa: SLF001
            "SELECT score_scale, COUNT(*) FROM calibration_log WHERE local_judge_label IS NOT NULL GROUP BY 1"
        ).fetchall()
    )
    assert by_scale == {CALIBRATION_SCORE_SCALE: 2, COSINE_SCORE_SCALE: 2}


# ---------------------------------------------------------------------------
# Review F1: a single-class day has no threshold to fit; never persist a
# sentinel outside the cosine range
# ---------------------------------------------------------------------------


def _seed_single_class_day(store: MemoryStore, label: str, *, model_id: str = _EMBEDDER_ID) -> None:
    rows = -(-FLOOR_FIT_MIN_LABELED_PAIRS // 9)  # rows of 9 candidates, enough for the minimum
    for _ in range(rows):
        _insert_row(
            store,
            [0.80 + 0.005 * i for i in range(9)],
            [label] * 9,
            model_id=model_id,
            scale=COSINE_SCORE_SCALE,
        )


@pytest.mark.parametrize("label", ["irrelevant", "relevant"])
def test_a_single_class_cosine_day_writes_nothing_without_a_prior(store: MemoryStore, label: str) -> None:
    _seed_single_class_day(store, label)
    out = derive_and_persist_cosine_floor(store, _EMBEDDER_ID)
    assert out.accepted is False and out.held_for_data_starvation and out.floor is None
    assert store.get_persisted_cosine_floor(_EMBEDDER_ID) is None, (
        "no sentinel outside [-1, 1] may be persisted; the bootstrap keeps serving"
    )


@pytest.mark.parametrize("label", ["irrelevant", "relevant"])
def test_a_single_class_cosine_day_holds_the_prior_row(store: MemoryStore, label: str) -> None:
    store.write_cosine_floor(_EMBEDDER_ID, floor=0.83, raw_fit_floor=0.83, sample_pairs=300, is_cold_start=False)
    _seed_single_class_day(store, label)
    out = derive_and_persist_cosine_floor(store, _EMBEDDER_ID)
    assert out.accepted is False and out.floor == pytest.approx(0.83)
    assert store.get_persisted_cosine_floor(_EMBEDDER_ID)["floor"] == pytest.approx(0.83)


def _seed_skewed_day(store: MemoryStore, *, relevant_fraction: float, seed: int) -> None:
    """The reviewer's regime: overlapping cosines, most labels one class. The
    recall-leaning F-beta then picks 'serve everything' (a threshold below the
    day's minimum), which gates nothing."""
    rng = np.random.default_rng(seed)
    n = FLOOR_FIT_MIN_LABELED_PAIRS + 16
    for _ in range(n):
        relevant = rng.random() < relevant_fraction
        score = float(rng.normal(0.84 if relevant else 0.82, 0.03))
        _insert_row(store, [score], ["relevant" if relevant else "irrelevant"], model_id=_EMBEDDER_ID,
                    scale=COSINE_SCORE_SCALE)


@pytest.mark.parametrize(("fraction", "seed"), [(0.7, 1), (0.8, 3), (0.95, 2)])
def test_a_skewed_overlapping_cosine_day_never_persists_a_gates_nothing_floor(
    store: MemoryStore, fraction: float, seed: int
) -> None:
    _seed_skewed_day(store, relevant_fraction=fraction, seed=seed)
    pairs = store.labeled_calibration_pairs(_EMBEDDER_ID, COSINE_SCORE_SCALE)
    raw = fit_threshold_fbeta(pairs, beta=FLOOR_FIT_BETA)
    assert raw <= min(s for s, _ in pairs) or raw > max(s for s, _ in pairs), (
        "test precondition: this day's raw fit is a pass-all/pass-none sentinel"
    )
    out = derive_and_persist_cosine_floor(store, _EMBEDDER_ID)
    assert out.accepted is False and out.held_for_data_starvation
    assert store.get_persisted_cosine_floor(_EMBEDDER_ID) is None


def test_the_persisted_cosine_floor_stays_inside_the_cosine_range_for_two_class_days(store: MemoryStore) -> None:
    rng = np.random.default_rng(11)
    _seed_scale(store, rng, n=FLOOR_FIT_MIN_LABELED_PAIRS, model_id=_EMBEDDER_ID, scale=COSINE_SCORE_SCALE,
                rel=0.88, irr=0.80, sd=0.02)
    out = derive_and_persist_cosine_floor(store, _EMBEDDER_ID)
    assert out.accepted and -1.0 <= out.floor <= 1.0


# ---------------------------------------------------------------------------
# Review F2: each scale's fit reads ITS OWN most recent labeled day
# ---------------------------------------------------------------------------


def test_each_scale_reads_its_own_most_recent_labeled_day(store: MemoryStore) -> None:
    """The latest normalized-labeled day is later than the latest cosine-labeled
    day: the cosine fit must still find its own (earlier) day, not read zero
    rows because a normalized row set the 'most recent day'."""
    _insert_row(store, [0.9, 0.2], ["relevant", "irrelevant"], model_id="m", scale=COSINE_SCORE_SCALE,
                day_bucket="2026-09-27")
    _insert_row(store, [0.85], ["relevant"], model_id="m", scale=COSINE_SCORE_SCALE, day_bucket="2026-09-27")
    _insert_row(store, [4.0, -4.0], ["relevant", "irrelevant"], model_id="m", scale=CALIBRATION_SCORE_SCALE,
                day_bucket="2026-09-28")
    assert sorted(s for s, _ in store.labeled_calibration_pairs("m", COSINE_SCORE_SCALE)) == [0.2, 0.85, 0.9]
    assert sorted(s for s, _ in store.labeled_calibration_pairs("m")) == [-4.0, 4.0]


# ---------------------------------------------------------------------------
# Review F5: the tick's fault isolation and pause behaviour around the cosine step
# ---------------------------------------------------------------------------


def _seed_cosine_fit_day(pd: Path) -> None:
    rng = np.random.default_rng(5)
    seed = MemoryStore(pd / "memories.db", integrity_check=False)
    _seed_scale(seed, rng, n=FLOOR_FIT_MIN_LABELED_PAIRS + 20, model_id=_EMBEDDER_ID,
                scale=COSINE_SCORE_SCALE, rel=0.9, irr=0.1, sd=0.1)
    seed.close()


def _persisted(pd: Path):
    check = MemoryStore(pd / "memories.db", integrity_check=False)
    try:
        return check.get_persisted_reranker_floor(_RERANKER_ID), check.get_persisted_cosine_floor(_EMBEDDER_ID)
    finally:
        check.close()


def test_a_failing_rerank_floor_step_does_not_skip_the_cosine_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **kw):
        raise RuntimeError("simulated rerank floor derivation failure")

    monkeypatch.setattr(floor_calibration, "derive_and_persist_floor", _boom)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_cosine_fit_day(pd)
        assert _run_calibration_tick(pd) is True
        rerank_row, cosine_row = _persisted(pd)
    assert rerank_row is None and cosine_row is not None


def test_a_failing_cosine_floor_step_does_not_fail_the_tick_or_undo_the_rerank_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a, **kw):
        raise RuntimeError("simulated cosine floor derivation failure")

    monkeypatch.setattr(floor_calibration, "derive_and_persist_cosine_floor", _boom)
    n = FLOOR_FIT_MIN_LABELED_PAIRS + 20
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        seed = MemoryStore(pd / "memories.db", integrity_check=False)
        _seed_scale(seed, np.random.default_rng(6), n=n, model_id=_RERANKER_ID,
                    scale=CALIBRATION_SCORE_SCALE, rel=3.0, irr=-3.0, sd=1.0)
        seed.close()
        assert _run_calibration_tick(pd) is True
        rerank_row, cosine_row = _persisted(pd)
    assert rerank_row is not None and cosine_row is None


def test_a_paused_labeling_pass_defers_the_cosine_floor_too(monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.memory import relevance_judge

    monkeypatch.setattr(relevance_judge, "build_judge_provider", lambda full_model_dir=None: FakeRelevanceJudgeProvider())
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_cosine_fit_day(pd)
        seed = MemoryStore(pd / "memories.db", integrity_check=False)
        for _ in range(2):  # two unlabeled rows: labeling pauses between them
            _insert_row(seed, [0.5], None, model_id=_EMBEDDER_ID, scale=COSINE_SCORE_SCALE)
        seed.close()
        assert _run_calibration_tick(pd, should_pause=lambda: True) is None
        _, cosine_row = _persisted(pd)
    assert cosine_row is None, "a paused tick derives no floor of either scale"
