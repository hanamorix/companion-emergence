"""The calibration tick and the deploy recalibration job learn the runtime model
ids without building a provider (name-recall fix; CI's judge RSS check).

Building a `FastEmbedProvider` downloads and materializes the whole embedding
model when it is not cached, and building a `CrossEncoderProvider` downloads
the reranker the same way (fastembed calls `download_model()` in both
constructors). A fresh install or CI runner paid that inside the calibration
tick just to read a model id string. These tests make every provider
constructor raise and record the attempt (the tick's steps swallow exceptions,
so the record, not the exception, is what is asserted) while checking the
floors are still derived and persisted under the right ids. Offline; synthetic
data only.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from brain.bridge import supervisor
from brain.memory import embeddings, reranker
from brain.memory.floor_calibration import (
    FLOOR_FIT_BETA,
    FLOOR_FIT_MIN_LABELED_PAIRS,
    fit_threshold_fbeta,
)
from brain.memory.store import CALIBRATION_SCORE_SCALE, COSINE_SCORE_SCALE, MemoryStore
from tests.unit.brain.memory.test_cosine_floor import _seed_scale


@pytest.fixture
def constructions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every way to build an embedding or reranker provider raises and is
    recorded. The returned list is empty iff nothing tried."""
    hits: list[str] = []

    def _trap(name: str):
        def _raise(*args, **kwargs):
            hits.append(name)
            raise AssertionError(f"{name} must not be called to learn a model id")

        return _raise

    monkeypatch.setattr(embeddings, "build_embedding_provider", _trap("build_embedding_provider"))
    monkeypatch.setattr(embeddings.FastEmbedProvider, "__init__", _trap("FastEmbedProvider.__init__"))
    monkeypatch.setattr(reranker, "build_reranker_provider", _trap("build_reranker_provider"))
    monkeypatch.setattr(reranker, "_bootstrap_reranker_provider", _trap("_bootstrap_reranker_provider"))
    monkeypatch.setattr(reranker.CrossEncoderProvider, "__init__", _trap("CrossEncoderProvider.__init__"))
    return hits


def test_the_tick_derives_both_floors_under_the_runtime_ids_without_building_a_provider(
    constructions: list[str],
) -> None:
    embedder_id = embeddings.resolve_embedding_model_id()
    reranker_id = reranker.resolve_reranker_model_id()
    n = FLOOR_FIT_MIN_LABELED_PAIRS + 20
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        seed = MemoryStore(pd / "memories.db", integrity_check=False)
        norm_pairs = _seed_scale(
            seed, rng, n=n, model_id=reranker_id, scale=CALIBRATION_SCORE_SCALE, rel=3.0, irr=1.0, sd=0.7
        )
        cos_pairs = _seed_scale(
            seed, rng, n=n, model_id=embedder_id, scale=COSINE_SCORE_SCALE, rel=0.9, irr=0.1, sd=0.1
        )
        seed.close()

        assert supervisor._run_calibration_tick(pd) is True

        check = MemoryStore(pd / "memories.db", integrity_check=False)
        rerank_row = check.get_persisted_reranker_floor(reranker_id)
        cosine_row = check.get_persisted_cosine_floor(embedder_id)
        check.close()

    assert constructions == []
    assert rerank_row is not None
    assert rerank_row["floor"] == pytest.approx(fit_threshold_fbeta(norm_pairs, beta=FLOOR_FIT_BETA))
    assert cosine_row is not None
    assert cosine_row["floor"] == pytest.approx(fit_threshold_fbeta(cos_pairs, beta=FLOOR_FIT_BETA))


def test_the_deploy_recalibration_job_reads_the_reranker_id_without_building_a_provider(
    constructions: list[str],
) -> None:
    reranker_id = reranker.resolve_reranker_model_id()
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        MemoryStore(pd / "memories.db", integrity_check=False).close()

        due = supervisor._deploy_recalibration_due(pd)
        supervisor._run_deploy_recalibration(pd)

        check = MemoryStore(pd / "memories.db", integrity_check=False)
        stale = check.reranker_floor_is_stale(reranker_id)
        check.close()

    assert due is True, "a store with no floor row for the runtime reranker is due"
    assert stale is True, "no data to fit: the run held, it did not build a provider to do it"
    assert constructions == []


def test_the_embedding_id_accessor_is_the_tier_id_and_builds_nothing(constructions: list[str]) -> None:
    """The id read without a provider is the id `FastEmbedProvider` reports in
    production (pinned in test_floor_startup.py against a real construction
    path): the tier's model string."""
    from brain.bridge.model_tier import TIER_EMBEDDING, model_for_tier

    assert embeddings.resolve_embedding_model_id() == model_for_tier(TIER_EMBEDDING)
    assert constructions == []
