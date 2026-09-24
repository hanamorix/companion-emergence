"""Shared pytest fixtures and configuration for companion-emergence tests."""

from __future__ import annotations

import functools
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.bridge import cli_throttle, provider_auth
from brain.chat import pass2_queue

pytest_plugins = ["pytester"]


@functools.cache
def _symlinks_available() -> bool:
    """Probe once per session whether this host lets us create a symlink (#262).

    Non-elevated Windows without Developer Mode raises ``OSError: [WinError 1314]``; GitHub's
    ``windows-latest`` runner is elevated, so CI never sees it — a contributor's stock checkout does.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "target"
        target.mkdir()
        try:
            os.symlink(target, Path(tmp) / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            return False
    return True


@pytest.fixture
def requires_symlinks() -> None:
    """Skip (not fail) a test whose setup creates a symlink, where the host denies it (#262)."""
    if not _symlinks_available():
        pytest.skip("os.symlink needs an elevated shell or Developer Mode on Windows")


@pytest.fixture(autouse=True)
def _reset_cli_throttle() -> Iterator[None]:
    """Reset cli_throttle global state before each test.

    background_slot() reads process-global monotonic timestamps. Without
    this reset, a test that calls mark_interactive_active() contaminates
    subsequent tests in the same process — causing background-engine calls
    to be gated when the test expects them to fire.
    """
    cli_throttle.reset()
    provider_auth.reset()  # #246: the auth-expiry hooks live inside the CLI detail helpers
    yield
    cli_throttle.reset()
    provider_auth.reset()


@pytest.fixture(autouse=True)
def _inhibit_bridge_background_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep build_app's lifespan from starting the supervisor + migration threads.

    Every bridge endpoint test enters build_app's lifespan; the real supervisor
    thread it started ran a startup catch-up compaction that rolled over the
    months-old session the test had just seeded and deleted its buffer, racing
    the test's own request (hunts/bridge-order-pollution-flakes/diagnosis.md —
    #155, #161, and the c8 / WinError-32 CI reds). Mirrors _reset_pass2_queue:
    tests that genuinely exercise those threads pass background_threads=True.
    """
    from brain.bridge import server

    # raising=False: a bisect onto a pre-flag commit must not error every test at setup.
    monkeypatch.setattr(server, "_background_threads_inhibited", True, raising=False)


@pytest.fixture(autouse=True)
def _reset_pass2_queue() -> Iterator[None]:
    """Reset pass2_queue global state before and after each test.

    Mirrors _reset_cli_throttle.  Without this, tests that call enqueue()
    leave items in the queue (or a running worker thread) that contaminate
    subsequent tests in the same process.

    Also inhibits the daemon worker so enqueue() never spawns a background
    thread during tests — tests drive drain_pending() synchronously, and a
    live worker would race those drains.
    """
    pass2_queue._worker_inhibited = True
    pass2_queue.reset()
    yield
    pass2_queue.reset()


@pytest.fixture(autouse=True)
def _fake_embedding_provider_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force brain.memory.embeddings.build_embedding_provider() to the
    deterministic, offline FakeEmbeddingProvider for the whole suite by
    default.

    build_embedding_provider() is the PRODUCTION default (FastEmbedProvider —
    a real local ONNX model, downloaded once over the network into a shared
    cache dir). Every production call site that embeds anything (recall's
    query embed, dedupe's candidate embed, embed-on-write) goes through this
    one function (Stage 1 of the local semantic-retrieval build; the old
    build_embedding_cache()/EmbeddingCache layer in front of it is gone as
    of F1 #259 increment 8), so ANY test that exercises those
    code paths — even indirectly, via a background thread the test itself
    never awaits — would otherwise attempt a real model download: slow,
    network-dependent, and (seen while landing this fixture) capable of
    retrying for minutes past the test's own teardown in a now-deleted tmp
    dir. A test that genuinely needs the real provider opts out with
    `@pytest.mark.requires_network`.
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import embeddings

    monkeypatch.setattr(
        embeddings, "build_embedding_provider", lambda: embeddings.FakeEmbeddingProvider(dim=256)
    )


@pytest.fixture(autouse=True)
def _reset_embedding_provider_cache() -> Iterator[None]:
    """Reset embeddings.build_embedding_provider()'s process-level provider
    cache before and after each test.

    Most tests never touch this cache at all — the fake-provider override
    above replaces build_embedding_provider() wholesale (cache included), so
    the fake path never reads or writes it. But a couple of tests in
    test_embeddings.py import `build_embedding_provider` by NAME and call the
    real function directly to exercise its own model_tier wiring, which
    bypasses that monkeypatch entirely (the import binds the original
    function object before any fixture runs). Without this reset, whichever
    such test ran first would cache a provider that a later one — expecting
    to build its own, under its own patched tmp_path / stubbed TextEmbedding
    — would get served back instead.
    """
    from brain.memory import embeddings

    embeddings._reset_embedding_provider_cache()
    yield
    embeddings._reset_embedding_provider_cache()


@pytest.fixture(autouse=True)
def _reset_embedding_matrix_cache() -> Iterator[None]:
    """Reset embedding_matrix.build_embedding_matrix()'s process-level matrix
    cache before and after each test — mirrors `_reset_embedding_provider_cache`
    above for the same reason: the cache is process-global and keyed by
    `str(db_path)`, so without a reset a matrix built (and possibly warmed)
    by one test against a given path could leak into a later test that
    happens to reuse that path, or hold a stale reference across tests that
    each expect a fresh singleton for their own tmp_path db."""
    from brain.memory import embedding_matrix

    embedding_matrix._reset_embedding_matrix_cache()
    yield
    embedding_matrix._reset_embedding_matrix_cache()


@pytest.fixture(autouse=True)
def _reset_embedding_backfill_batch_cache() -> Iterator[None]:
    """Reset embedding_backfill's measured/derived batch-size cache before
    and after each test — mirrors `_reset_embedding_provider_cache` above:
    the cache is process-global and keyed by model_id, measured once per
    process (F1 #259 increment 3), so without a reset a batch size measured
    (and possibly deliberately controlled/monkeypatched) by one test could
    leak into a later test expecting its own fresh measurement."""
    from brain.memory import embedding_backfill

    embedding_backfill._reset_batch_size_cache()
    yield
    embedding_backfill._reset_batch_size_cache()


@pytest.fixture(autouse=True)
def _fake_reranker_provider_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force brain.memory.reranker.build_reranker_provider() to the
    deterministic, offline FakeRerankerProvider for the whole suite by
    default (#231 offline-test discipline, mirrors
    `_fake_embedding_provider_by_default` above).

    build_reranker_provider() is the PRODUCTION default (CrossEncoderProvider
    — a real local ONNX cross-encoder model, downloaded once over the
    network into a shared cache dir). Any test that exercises
    run_semantic_recall / search_memories(mode="semantic") — even
    indirectly — would otherwise attempt a real model download. A test that
    genuinely needs the real provider opts out with
    `@pytest.mark.requires_network`.

    FakeRerankerProvider defaults every UNSCRIPTED document to a score far
    below any plausible calibrated floor (see that class's docstring) — so a
    test that never scripts reranker scores gets the same "semantic
    inconclusive -> lexical fallback" behavior it would have gotten from an
    empty/orthogonal cosine result pre-#231, rather than an arbitrary
    reranker score accidentally clearing the floor. Tests that need a
    CONCLUSIVE reranked result construct their own
    `FakeRerankerProvider(scores={...})` and monkeypatch this function
    directly, mirroring how `_ScriptedProvider` overrides the embedding
    fixture above for the same reason.

    Also patches `reranker._bootstrap_reranker_provider` (F2a inc8, #250 §7
    UPDATED — the bootstrap-floor ruling) the same way: that function is the
    OTHER production entry point that constructs a real `CrossEncoderProvider`
    (deliberately independent of `build_reranker_provider` itself — see its
    own docstring on why), used by `floor_calibration.get_bootstrap_floor`
    whenever `store.get_reranker_floor` is asked about a model_id with no
    persisted row yet. Without this, ANY test whose store has no persisted
    floor row (the common case for a fresh in-memory/tmp_path store) would
    attempt a REAL fastembed model load the first time `get_reranker_floor`
    is called — even tests that never intentionally touch the reranker at
    all.

    UNLIKE `build_reranker_provider`'s fake, this one is scripted (not left
    fully unscripted): the bootstrap fits a THRESHOLD from whatever scores
    its provider returns for the bundled `_FP16_GATE_PAIRS[:6]` pairs, so an
    UNSCRIPTED default-everywhere provider would fit a floor of roughly
    `_DEFAULT_UNSCORED - 1.0` (every pair ties at the same sentinel score,
    and `fit_threshold_fbeta` picks the threshold just below it) — only
    ONE unit below `FakeRerankerProvider`'s own `_DEFAULT_UNSCORED`, not
    "far below" it. That would silently break the "unscripted reranker
    score never clears an unscripted floor -> INCONCLUSIVE" invariant every
    other test in this suite relies on for its OWN default (an unscripted
    -1000.0 candidate score would clear a -1001.0 bootstrap floor). Scripted
    here to a realistic, well-SEPARATED pair of scores instead (relevant
    pairs high, irrelevant pairs low) so the default bootstrap floor lands
    near 0.0 — comfortably above `_DEFAULT_UNSCORED`, restoring that
    invariant for every test that never scripts its OWN bootstrap provider.
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import reranker

    # Both consuming call sites (brain.memory.semantic_recall,
    # brain.tools.impls.search_memories) import the module itself
    # (`from brain.memory import reranker as reranker_mod`) and call
    # `reranker_mod.build_reranker_provider()` — a dynamic attribute lookup
    # at call time, exactly like every embedding-provider call site's
    # identical dynamic lookup on `embeddings.build_embedding_provider` — so
    # patching this ONE module attribute is sufficient to intercept every
    # call site.
    monkeypatch.setattr(
        reranker, "build_reranker_provider", lambda *, store=None: reranker.FakeRerankerProvider()
    )
    default_bootstrap_scores = {
        doc: (5.0 if i < 3 else -5.0) for i, (_query, doc) in enumerate(reranker._FP16_GATE_PAIRS[:6])
    }
    monkeypatch.setattr(
        reranker,
        "_bootstrap_reranker_provider",
        lambda model_id: reranker.FakeRerankerProvider(scores=default_bootstrap_scores),
    )


@pytest.fixture(autouse=True)
def _reset_reranker_provider_cache() -> Iterator[None]:
    """Reset reranker.build_reranker_provider()'s process-level provider
    cache, its warm-latency cache, its warm-memory cache (pre-flip revision
    Change 3 — mirrors the warm-latency cache, same rationale), and
    floor_calibration's bootstrap-floor cache (F2a inc8, #250 §7 UPDATED)
    before and after each test — mirrors `_reset_embedding_provider_cache`
    above for the same reason (a test that calls the REAL `build_reranker_
    provider()` / `get_reranker_floor()` directly must not read or leak a
    provider/bootstrap/measurement a prior/later test's call happened to
    cache)."""
    from brain.memory import floor_calibration, reranker

    reranker._reset_reranker_provider_cache()
    reranker._reset_latency_cache()
    reranker._reset_memory_cache()
    floor_calibration._reset_bootstrap_floor_cache()
    yield
    reranker._reset_reranker_provider_cache()
    reranker._reset_latency_cache()
    reranker._reset_memory_cache()
    floor_calibration._reset_bootstrap_floor_cache()


@pytest.fixture(autouse=True)
def _fake_relevance_judge_provider_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force brain.memory.relevance_judge.build_judge_provider() to the
    deterministic, offline FakeRelevanceJudgeProvider for the whole suite by
    default (F2a #250 inc6, mirrors `_fake_reranker_provider_by_default`
    above).

    build_judge_provider() is the PRODUCTION default (TorchCrossEncoderJudge
    — a real local torch/sentence-transformers cross-encoder, downloaded
    once over the network into a shared cache dir). Any test that exercises
    the daily calibration tick's judge-labeling pass — even indirectly via
    `_run_calibration_tick` — would otherwise attempt a real model download
    and a real torch import. A test that genuinely needs the real judge
    opts out with `@pytest.mark.requires_network`.

    Every candidate the FakeRelevanceJudgeProvider default scores lands far
    below any plausible ambiguous band (its unscripted-pair default score,
    see that class's docstring) — mirroring FakeRerankerProvider's "never
    accidentally clears the floor" posture, this default never accidentally
    routes an unscripted pair to a stubbed Haiku call either. Tests that
    need a specific label/ambiguous-band outcome construct their own
    `FakeRelevanceJudgeProvider(scores={...})` and pass it directly to
    `label_calibration_sample`/`_run_calibration_tick` (the `judge=`
    parameter), mirroring how tests construct their own
    `FakeRerankerProvider(scores={...})` for the reranker.
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import relevance_judge

    monkeypatch.setattr(
        relevance_judge, "build_judge_provider", lambda: relevance_judge.FakeRelevanceJudgeProvider()
    )


@pytest.fixture(autouse=True)
def _reset_judge_provider_cache() -> Iterator[None]:
    """Reset relevance_judge.build_judge_provider()'s process-level provider
    cache before and after each test — mirrors `_reset_reranker_provider_cache`
    above for the same reason (a test that calls the REAL
    `build_judge_provider()` directly must not read or leak a provider a
    prior/later test's call happened to cache)."""
    from brain.memory import relevance_judge

    relevance_judge._reset_judge_provider_cache()
    yield
    relevance_judge._reset_judge_provider_cache()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Walk upward from this file to find the repo root (pyproject.toml).

    Replaces brittle `Path(__file__).parents[N]` patterns in tests that
    need absolute paths to checked-in resources.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove companion-emergence-relevant env vars for isolation.

    Used by tests for brain.paths and brain.config (Tasks 2+).
    Each key here corresponds to an env var the framework reads at runtime.
    """
    for key in [
        "NELLBRAIN_HOME",
        "NELL_IPC_JID",
        "BRIDGE_BIND",
        "PROVIDER",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
