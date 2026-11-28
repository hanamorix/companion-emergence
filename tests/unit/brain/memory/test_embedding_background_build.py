"""#833: the embedding model's download no longer blocks its callers in the bridge.

The bridge's lifespan enables the background build (`floor_startup.enable_background_embedding_build`). From then on, while the model
is not built, no caller on the reply or supervisor thread waits for the download: each takes the fallback it already has, and one
background thread (the floor start-up thread, the message retry, the lull job or a caller's request, all through the one "cosine"
in-flight claim) builds the model. A failed build is recorded in the existing failed-attempt note and retried exactly as before.
Processes that do not enable it keep waiting. A notice (`embedding_download_notice`) says "downloading" or "failed".

All offline: the real provider is never built; the build is replaced by a recorder; the hub-layout cache is a tmp dir.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from brain.engines import consolidation
from brain.ingest import dedupe
from brain.memory import embeddings as embeddings_mod
from brain.memory import floor_calibration, floor_startup
from brain.memory import reranker as reranker_mod
from brain.memory.embeddings import EmbeddingNotReadyError, FakeEmbeddingProvider
from brain.memory.semantic_recall import run_semantic_recall
from brain.memory.store import MemoryStore
from tests.unit.brain.engines.test_consolidation_same_type_619 import (  # noqa: F401
    _capture_context,
    _mem,
    _run,
    persona,
)
from tests.unit.brain.ingest.test_dedupe import _seed_embedded_memory, _use_384_fake_provider
from tests.unit.brain.memory.test_no_rerank_cosine_path import (
    _QUERY,
    _seed_pool,
    _write_cosine_floor,
)
from tests.unit.brain.memory.test_reranker import _hf_cache

_MODEL = "intfloat/multilingual-e5-large"
_REPO = "qdrant/multilingual-e5-large-onnx"


@pytest.fixture
def bridge_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """The bridge's background embedding build is enabled, the model is not built, and building it on the calling thread is a
    test failure. Returns the log of `request_embedding_build` calls."""
    requests: list[str] = []
    monkeypatch.setattr(floor_startup, "_background_build_persona_dir", tmp_path)
    monkeypatch.setattr(floor_startup, "request_embedding_build", lambda: requests.append("asked"))

    def _no_build():
        raise AssertionError("a caller built the embedding model on its own thread")

    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", _no_build)
    return requests


# ---------------------------------------------------------------------------
# The accessor
# ---------------------------------------------------------------------------


def test_a_caller_gets_not_ready_at_once_and_the_build_is_asked_for(bridge_mode: list[str]) -> None:
    with pytest.raises(EmbeddingNotReadyError):
        embeddings_mod.get_ready_embedding_provider()
    assert bridge_mode == ["asked"]


def test_a_process_that_did_not_enable_the_background_build_keeps_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: sentinel)
    assert floor_startup.background_embedding_build_enabled() is False
    assert embeddings_mod.get_ready_embedding_provider() is sentinel


def test_once_the_model_is_built_the_caller_gets_it_and_asks_for_nothing(
    bridge_mode: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeEmbeddingProvider(dim=8)
    monkeypatch.setitem(embeddings_mod._provider_cache, embeddings_mod.resolve_embedding_model_id(), provider)  # noqa: SLF001
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    assert embeddings_mod.get_ready_embedding_provider() is provider
    assert bridge_mode == []


def test_enabling_does_nothing_while_background_threads_are_inhibited(tmp_path: Path) -> None:
    assert floor_startup._background_inhibited is True  # noqa: SLF001 - the suite default
    floor_startup.enable_background_embedding_build(tmp_path)
    assert floor_startup.background_embedding_build_enabled() is False


# ---------------------------------------------------------------------------
# Each caller takes the fallback it already has
# ---------------------------------------------------------------------------


def test_recall_falls_back_to_keyword_without_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mems = _seed_pool(store, monkeypatch, [0.99, 0.98, 0.97])
    _write_cosine_floor(store, 0.5)
    ready = run_semantic_recall(store, tmp_path, _QUERY)
    assert ready is not None and mems[0].id in {m.id for m in [*ready.full, *ready.snippet]}, "the model built: cosine recall"

    requests: list[str] = []
    monkeypatch.setattr(floor_startup, "_background_build_persona_dir", tmp_path)
    monkeypatch.setattr(floor_startup, "request_embedding_build", lambda: requests.append("asked"))
    monkeypatch.setattr(
        embeddings_mod, "build_embedding_provider", lambda: pytest.fail("recall built the model on the reply path")
    )
    with caplog.at_level(logging.INFO):
        assert run_semantic_recall(store, tmp_path, _QUERY) is None
    assert requests == ["asked"]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], "a quiet fallback, no traceback per turn"


def test_dedupe_lets_the_item_through_without_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    _use_384_fake_provider(monkeypatch)
    text = "Bob keeps bees behind the harbour"
    _seed_embedded_memory(store, text)
    assert dedupe.is_duplicate(text, store=store) is True, "the model built: an exact copy is a duplicate"

    requests: list[str] = []
    monkeypatch.setattr(floor_startup, "_background_build_persona_dir", tmp_path)
    monkeypatch.setattr(floor_startup, "request_embedding_build", lambda: requests.append("asked"))
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: pytest.fail("dedupe built the model"))
    with caplog.at_level(logging.INFO):
        assert dedupe.is_duplicate(text, store=store) is False
    assert requests == ["asked"]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_the_sorting_steps_embed_gives_no_vector_without_building(bridge_mode: list[str]) -> None:
    assert consolidation._embed_candidate("Bob keeps bees") is None  # noqa: SLF001
    assert bridge_mode == ["asked"]


def test_saving_an_embedding_raises_at_once_without_building(bridge_mode: list[str], tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    m = _mem("Bob keeps bees behind the harbour", "fact")
    store.create(m)
    with pytest.raises(EmbeddingNotReadyError):
        store.embed_row(m.id, m.content)
    assert bridge_mode == ["asked"]


def test_an_edited_memory_has_its_stale_vector_cleared_quietly(
    bridge_mode: list[str], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    m = _mem("Bob keeps bees behind the harbour", "fact")
    store.create(m)
    store.write_embedding(m.id, FakeEmbeddingProvider(dim=8).embed("x"), "fake")
    with caplog.at_level(logging.INFO):
        store._reembed_or_clear(m.id, "Bob keeps bees on the roof")  # noqa: SLF001
    row = store._conn.execute("SELECT embedding FROM memories WHERE id = ?", (m.id,)).fetchone()  # noqa: SLF001
    assert row["embedding"] is None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_promoted_candidate_is_stored_without_a_vector_and_without_a_warning(
    persona, bridge_mode: list[str], caplog: pytest.LogCaptureFixture  # noqa: F811
) -> None:
    _tmp, store, _heb, queue = persona
    queue.enqueue(_mem("Bob keeps bees behind the harbour", "fact"), source="x")
    with caplog.at_level(logging.INFO):
        res = _run(persona, _capture_context({}))
    assert res.promoted == 1
    (row,) = store._conn.execute("SELECT embedding FROM memories").fetchall()  # noqa: SLF001
    assert row["embedding"] is None, "left for the idle backfill"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# ---------------------------------------------------------------------------
# The supervisor thread: the backfill job skips, the lull job only starts the thread
# ---------------------------------------------------------------------------


def _job(persona_dir: Path, name: str):
    from tests.unit.brain.memory.test_floor_startup import _bootstrap_job_named

    (job,) = _bootstrap_job_named(persona_dir, name)
    return job


def test_the_backfill_job_skips_the_pass_while_the_model_is_not_built(
    bridge_mode: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor
    from brain.bridge.central_cadence import JobOutcome
    from tests.unit.brain.bridge.test_central_cadence import _granting_slot, _persona

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    monkeypatch.setattr(
        supervisor, "_embedding_backfill_run_tick", lambda *a, **k: pytest.fail("the backfill ticked without the model")
    )
    job = _job(_persona(tmp_path), "embedding_backfill")
    assert job.run() is JobOutcome.SKIPPED
    assert bridge_mode == ["asked"]


def test_the_lull_job_starts_the_background_thread_and_never_builds_itself(
    bridge_mode: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge.central_cadence import JobOutcome
    from tests.unit.brain.bridge.test_central_cadence import _persona

    started: list[Path] = []
    monkeypatch.setattr(floor_startup, "start_cosine_retry", lambda d: started.append(d) or object())
    monkeypatch.setattr(floor_startup, "run_cosine_floor", lambda **k: pytest.fail("the lull job ran the floor itself"))
    persona_dir = _persona(tmp_path)
    job = _job(persona_dir, "cosine_floor_bootstrap")
    assert job.run() is JobOutcome.COMPLETED
    assert started == [persona_dir]
    monkeypatch.setattr(floor_startup, "start_cosine_retry", lambda d: None)
    assert job.run() is JobOutcome.SKIPPED, "one is already running"


def test_the_lull_job_is_due_while_the_model_is_not_built_even_with_a_saved_floor_row(
    bridge_mode: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_startup, "embedder_model_id", lambda: "m")
    store = MemoryStore(tmp_path / "memories.db")
    monkeypatch.setattr(store, "get_persisted_cosine_floor", lambda _m: {"floor": 0.5})
    assert floor_startup.cosine_floor_due(store, activity_marker=1.0) is True
    floor_calibration._record_attempt("cosine", "m", False, 1.0)  # noqa: SLF001
    assert floor_startup.cosine_floor_due(store, activity_marker=1.0) is False, "not retried within the same chat activity"
    assert floor_startup.cosine_floor_due(store, activity_marker=2.0) is True, "retried once chat has happened"


# ---------------------------------------------------------------------------
# One background build at a time, and the failure rules
# ---------------------------------------------------------------------------


@pytest.fixture
def building(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Bridge mode with a recording build; `fail` makes it raise. The floor compute is replaced by a recorder."""
    state: dict = {"builds": 0, "fail": False, "floors": 0}
    monkeypatch.setattr(floor_startup, "_background_build_persona_dir", tmp_path)
    monkeypatch.setattr(floor_startup, "embedder_model_id", lambda: "m")

    def _build():
        state["builds"] += 1
        if state["fail"]:
            raise OSError("no network")
        provider = FakeEmbeddingProvider(dim=8)
        embeddings_mod._provider_cache[embeddings_mod.resolve_embedding_model_id()] = provider  # noqa: SLF001
        return provider

    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", _build)

    def _floor(*, activity_marker=None):
        state["floors"] += 1

    monkeypatch.setattr(floor_startup, "run_cosine_floor", _floor)
    return state


def test_start_up_builds_the_model_even_when_the_floor_row_is_saved(
    building: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MemoryStore, "get_persisted_cosine_floor", lambda self, _m: {"floor": 0.5})
    floor_startup.compute_missing_floors(tmp_path, activity_marker=lambda: 1.0)
    assert building["builds"] == 1, "the model is built although no floor is due"
    assert building["floors"] == 0
    assert floor_calibration.bootstrap_failed("cosine", "m") is False


def test_a_failed_build_is_recorded_under_the_floors_key_and_makes_no_second_attempt(building: dict, tmp_path: Path) -> None:
    building["fail"] = True
    floor_startup.compute_missing_floors(tmp_path, activity_marker=lambda: 7.0)
    assert building["builds"] == 1, "the floor does not start a second download"
    assert building["floors"] == 0
    assert floor_calibration.bootstrap_failed("cosine", "m") is True
    assert floor_calibration._bootstrap_failed_at[("cosine", "m")] == 7.0  # noqa: SLF001 - the marker of the failed attempt
    assert floor_startup.inflight("cosine") is False


def test_a_new_message_retries_a_failed_build_on_every_message_not_once_per_marker(
    building: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_startup, "_background_inhibited", False)
    monkeypatch.setattr(floor_startup, "reranker_model_id", lambda: "r")
    monkeypatch.setattr(floor_startup, "_current_marker", lambda: 7.0)
    building["fail"] = True
    floor_calibration._record_attempt("cosine", "m", False, 7.0)  # noqa: SLF001
    for expected in (1, 2, 3):
        (thread,) = floor_startup.on_incoming_message(tmp_path)
        thread.join(timeout=10)
        assert building["builds"] == expected, "the same marker, every message retries"
    building["fail"] = False
    (thread,) = floor_startup.on_incoming_message(tmp_path)
    thread.join(timeout=10)
    assert floor_calibration.bootstrap_failed("cosine", "m") is False, "a built model clears the note"
    assert floor_startup.on_incoming_message(tmp_path) == [], "and the messages stop retrying"


def test_a_caller_that_finds_the_model_not_built_follows_the_marker_rule(
    building: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []
    monkeypatch.setattr(floor_startup, "_start_retry", lambda kind, d: started.append(kind) or object())
    marker = {"m": 1.0}
    monkeypatch.setattr(floor_startup, "_current_marker", lambda: marker["m"])
    assert floor_startup.request_embedding_build() is not None, "first ask: nothing failed yet"
    floor_calibration._record_attempt("cosine", "m", False, 1.0)  # noqa: SLF001
    assert floor_startup.request_embedding_build() is None, "failed under this marker: not again until chat"
    marker["m"] = 2.0
    assert floor_startup.request_embedding_build() is not None
    assert started == ["cosine", "cosine"]


def test_asking_without_the_background_build_enabled_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(floor_startup, "_start_retry", lambda kind, d: pytest.fail("started a thread"))
    assert floor_startup.request_embedding_build() is None


def test_two_callers_start_one_thread_through_the_one_claim(
    building: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_startup, "_background_inhibited", False)
    gate = threading.Event()
    entered = threading.Event()

    def _slow_build():
        building["builds"] += 1
        entered.set()
        gate.wait(timeout=10)
        raise OSError("stopped")

    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", _slow_build)
    first = floor_startup.request_embedding_build()
    assert first is not None and entered.wait(timeout=10)
    assert floor_startup.request_embedding_build() is None, "a build is in flight"
    assert floor_startup.start_cosine_retry(tmp_path) is None
    gate.set()
    first.join(timeout=10)
    assert building["builds"] == 1


def test_the_retry_body_builds_before_the_saved_floor_check(building: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.memory.store import MemoryStore as _Store

    monkeypatch.setattr(_Store, "get_persisted_cosine_floor", lambda self, _m: {"floor": 0.5})
    floor_calibration._record_attempt("cosine", "m", False, 1.0)  # noqa: SLF001
    floor_startup._retry_body("cosine", tmp_path)  # noqa: SLF001
    assert building["builds"] == 1 and building["floors"] == 0
    assert floor_calibration.bootstrap_failed("cosine", "m") is False


def test_a_process_that_waits_does_not_build_in_the_ensure_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: pytest.fail("built"))
    assert floor_startup.ensure_embedder_built(1.0) is True


# ---------------------------------------------------------------------------
# The notice
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(floor_startup, "_background_build_persona_dir", tmp_path)
    monkeypatch.setattr(floor_startup, "embedder_model_id", lambda: _MODEL)
    return cache_dir


def test_the_registry_check_sees_the_embedding_model_and_every_one_of_its_files(cache: Path) -> None:
    assert reranker_mod._registry_entry(_MODEL)["additional_files"] == ["model.onnx_data"]  # noqa: SLF001
    assert reranker_mod._model_file_cached(_MODEL, cache) is False  # noqa: SLF001
    _hf_cache(cache, _REPO, ["model.onnx"])
    assert reranker_mod._model_file_cached(_MODEL, cache) is False, "the 2.2 GB second file is missing"  # noqa: SLF001
    (cache / ("models--" + _REPO.replace("/", "--")) / "snapshots" / "abc123" / "model.onnx_data").write_text("x")
    assert reranker_mod._model_file_cached(_MODEL, cache) is True  # noqa: SLF001


def test_no_notice_unless_the_bridge_builds_in_the_background(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(floor_startup, "_background_build_persona_dir", None)
    assert floor_startup.try_begin("cosine")
    assert floor_startup.embedding_download_notice() is None


def test_downloading_while_the_claim_is_held_and_the_files_are_missing(cache: Path) -> None:
    assert floor_startup.embedding_download_notice() is None, "nothing is running"
    assert floor_startup.try_begin("cosine")
    notice = floor_startup.embedding_download_notice()
    assert notice == {"state": "downloading", "size_gb": 2.24}
    _hf_cache(cache, _REPO, ["model.onnx"])
    assert floor_startup.embedding_download_notice() == notice, "half of the model is not the model"


def test_nothing_is_shown_after_a_restart_with_the_model_on_disk(cache: Path) -> None:
    _hf_cache(cache, _REPO, ["model.onnx", "model.onnx_data"])
    assert floor_startup.try_begin("cosine"), "the start-up thread holds the claim while it loads the cached model"
    assert floor_startup.embedding_download_notice() is None


def test_nothing_is_shown_once_the_model_is_built(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert floor_startup.try_begin("cosine")
    monkeypatch.setitem(embeddings_mod._provider_cache, embeddings_mod.resolve_embedding_model_id(), object())  # noqa: SLF001
    assert floor_startup.embedding_download_notice() is None


def test_failed_when_the_note_exists_nothing_runs_and_the_files_are_missing(cache: Path) -> None:
    floor_calibration._record_attempt("cosine", _MODEL, False, 1.0)  # noqa: SLF001
    assert floor_startup.embedding_download_notice() == {"state": "failed", "size_gb": 2.24}
    _hf_cache(cache, _REPO, ["model.onnx", "model.onnx_data"])
    assert floor_startup.embedding_download_notice() is None, "on disk: the failure was not the download"


def test_nothing_is_shown_when_the_check_cannot_decide(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(floor_startup, "embedder_model_id", lambda: "nobody/not-in-the-registry")
    assert floor_startup.try_begin("cosine")
    assert floor_startup.embedding_download_notice() is None


def test_the_state_carries_the_notice_before_the_two_other_notices_and_survives_a_broken_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brain.tunables as tunables
    from brain.bridge.persona_state import build_persona_state

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()  # noqa: SLF001
    persona_dir = tmp_path / "personas" / "canary"
    persona_dir.mkdir(parents=True)
    state = build_persona_state(persona_dir)
    assert list(state)[-3:] == ["embedding_download_notice", "rebuild_notice", "tunables_notice"]
    assert state["embedding_download_notice"] is None
    wanted = {"state": "downloading", "size_gb": 2.24}
    monkeypatch.setattr(floor_startup, "embedding_download_notice", lambda: wanted)
    assert build_persona_state(persona_dir)["embedding_download_notice"] == wanted

    def _boom():
        raise RuntimeError("broken")

    monkeypatch.setattr(floor_startup, "embedding_download_notice", _boom)
    assert build_persona_state(persona_dir)["embedding_download_notice"] is None
