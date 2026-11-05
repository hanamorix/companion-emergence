"""S200, S220: no re-scoring until the machine's memory model exists (name-recall fix, fixes
the S193 first-rerank regression).

On a machine with no usable memory model a recall turn takes the plain meaning-match (cosine)
path with nothing of the reranker built, loaded, warmed up or probed on the reply path, and flags
the need; the `rerank_ram_figure` lull job (or the `--no-bridge` thread) then runs the ladder
(`test_ram_ladder.py`), and the first real rerank is sized from the fitted model. The older
width-5 seeds and the S182 re-seed are retired (S203, S212, S220). This file covers the gate, the
lull job's due/run logic and the background thread (S201, S210).

No model is loaded: fake providers. `reranker._require_ram_figure` is switched on
per test (the suite's autouse fixture switches it off).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from brain.memory import reranker as reranker_mod
from brain.memory.reranker import RerankerProvider
from brain.memory.semantic_recall import COSINE_PATH, RERANKED_PATH, run_semantic_recall
from brain.memory.store import MemoryStore
from tests.thread_support import finish_threads, release_at_teardown, wait_at_gate
from tests.unit.brain.memory._ram_support import ScriptedLadderProvider, seed_window
from tests.unit.brain.memory.test_monologue_last import (
    _RERANKER_ID,
    _cosine_floor,
    _rerank_floor,
    _warm,
)
from tests.unit.brain.memory.test_per_paragraph_recall import _AxisEmbedder, _install, _mem

_QUERY = "kite over the harbour"
_FULL = 1024


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _RERANKER_ID)
    # S245: a ladder pass that the live headroom stops records no completion marker, so these tests must
    # not depend on the host's free memory; the tests that care patch their own figure.
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)  # S261: ample (unreadable now means no rerank)
    # S253/S257: the first rung and the model load need the model file's size (none exists for these doubles); 1 byte.
    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", lambda model_id: 1.0)


class _CappedReranker(RerankerProvider):
    """Token count = characters / 4 per pair, capped at the model's input limit
    (`cap`), like a truncating tokenizer. `rerank_timed` reports the scripted
    seconds and RSS growth. Records every call."""

    def __init__(self, *, cap: int | None = _FULL, seconds: float = 2.0, rss: float | None = 245e6):
        self.cap, self.seconds, self.rss = cap, seconds, rss
        self.plain: list[tuple[str, list[str]]] = []
        self.timed: list[tuple[str, list[str]]] = []
        self.probes = 0

    def rerank(self, query, documents):
        self.plain.append((query, list(documents)))
        return [0.0] * len(documents)

    def rerank_timed(self, query, documents):
        self.timed.append((query, list(documents)))
        return [5.0 if d.startswith("s200") else 0.0 for d in documents], self.seconds, self.rss

    def pair_token_lengths(self, query, documents):
        self.probes += 1
        counts = [(len(query) + len(d)) // 4 for d in documents]
        return counts if self.cap is None else [min(self.cap, c) for c in counts]

    def model_id(self) -> str:
        return _RERANKER_ID


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db")


def _recall_setup(monkeypatch, tmp_path):
    embedder = _AxisEmbedder({_QUERY: 0})
    store = _store(tmp_path)
    memories = [_mem(store, f"s200 memory {i}", {0: 0.9 - 0.01 * i}) for i in range(8)]
    _install(monkeypatch, embedder)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    return store, memories


# ---------------------------------------------------------------------------
# F-B1 / F-B3: the reply path with no figure
# ---------------------------------------------------------------------------


def test_with_no_figure_a_turn_takes_the_cosine_path_and_touches_nothing_of_the_reranker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, memories = _recall_setup(monkeypatch, tmp_path)
    built: list[object] = []

    def _factory(**kwargs):
        built.append(kwargs)
        raise AssertionError("the reranker must not be built on the reply path with no RAM figure")

    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", _factory)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == COSINE_PATH
    assert result.paragraphs[0].gated.real_width == 0
    assert built == []
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}


def test_a_turn_with_fewer_than_five_candidates_flags_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The need is flagged only by a turn that would rerank (5 or more candidates):
    a small pool never asks for the reranker, so it never asks for a figure."""
    embedder = _AxisEmbedder({_QUERY: 0})
    store = _store(tmp_path)
    for i in range(3):
        _mem(store, f"s200 small pool {i}", {0: 0.9 - 0.01 * i})
    _install(monkeypatch, embedder)
    _cosine_floor(store, 0.1)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == COSINE_PATH
    assert reranker_mod._ram_figure_needed == set()


def test_the_same_turn_reranks_once_a_window_figure_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Able to fail: a gate that is always open (or keyed on the wrong thing) would
    not pass the first test; a gate that is always closed would not pass this one."""
    store, memories = _recall_setup(monkeypatch, tmp_path)
    rec = _CappedReranker()
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)
    _warm(store)
    seed_window(store, _RERANKER_ID)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    assert reranker_mod._ram_figure_needed == set()


def test_the_shipped_default_requires_a_ram_figure() -> None:
    """The suite switches the requirement off (autouse); nothing else pins the
    production default, so a flipped default would disable S200 unnoticed."""
    import inspect
    import re

    source = inspect.getsource(reranker_mod)
    assert re.search(r"^_require_ram_figure = True$", source, re.MULTILINE)


# ---------------------------------------------------------------------------
# F-B2: the measurement
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The lull job: due once per chat activity marker, never builds a thread
# ---------------------------------------------------------------------------


def _factory(store: MemoryStore):
    @contextmanager
    def _open() -> Iterator[MemoryStore]:
        yield store

    return _open


def test_a_provider_failure_in_the_job_run_propagates_for_the_central_pass_to_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)

    def _boom(**kwargs):
        raise RuntimeError("model cannot be built")

    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", _boom)
    with pytest.raises(RuntimeError):
        reranker_mod.run_ram_figure("m1", store)
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}


# ---------------------------------------------------------------------------
# The gate (S200, S220, H4b): which states let a rerank run
# ---------------------------------------------------------------------------


def test_a_kept_fit_serves_a_running_process_whose_window_lost_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """H4b: a running process that once computed a usable fit keeps it when the window stops
    giving one (a quiet day: the full-length sample aged out). The gate stays open, the need is
    flagged so the next lull refreshes the window; a fresh process in the same state is held back."""
    store = _store(tmp_path)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False, "nothing yet"
    seed_window(store, _RERANKER_ID)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True
    assert _RERANKER_ID not in reranker_mod._ram_figure_needed, "a window fit needs no refresh"
    # the window loses its samples (all aged out)
    monkeypatch.setattr(reranker_mod._ram_shapes, "shape_groups", lambda *_a, **_k: [])
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True
    assert _RERANKER_ID in reranker_mod._ram_figure_needed, "kept: the lull is asked to refresh it"
    estimate_fit, kept = reranker_mod._ram_fit_for(store, _RERANKER_ID)
    assert estimate_fit is not None and kept is True
    # a fresh process (kept fits forgotten) in the same state is held back
    reranker_mod._reset_rerank_cost_model()
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


def test_a_turn_served_by_a_kept_fit_asks_the_direct_chat_to_refresh_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """H4b: the turn still reranks (the process keeps its last fit), and under `nell chat
    --no-bridge` (no lull job) the refresh is requested for after the reply."""
    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    rec = _CappedReranker()
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)
    _warm(store)
    seed_window(store, _RERANKER_ID)
    floor_startup.enable_background_ram_figure(tmp_path)
    first = run_semantic_recall(store, tmp_path, _QUERY)
    assert first is not None and first.path == RERANKED_PATH
    assert floor_startup._ram_figure_requested is False  # noqa: SLF001, the window still gives the model

    monkeypatch.setattr(reranker_mod._ram_shapes, "shape_groups", lambda *_a, **_k: [])  # the window lost it
    second = run_semantic_recall(store, tmp_path, _QUERY)
    assert second is not None and second.path == RERANKED_PATH, "the kept fit keeps the turn reranking"
    assert floor_startup._ram_figure_requested is True  # noqa: SLF001
    assert _RERANKER_ID in reranker_mod._ram_figure_needed


def test_the_kept_fit_is_keyed_by_database_and_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed_window(store, _RERANKER_ID)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True
    monkeypatch.setattr(reranker_mod._ram_shapes, "shape_groups", lambda *_a, **_k: [])
    assert reranker_mod.ram_figure_ready("another-model", store) is False
    other = MemoryStore(tmp_path / "other.db")
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, other) is False
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True


def test_a_window_of_zero_growth_is_no_model(tmp_path: Path) -> None:
    """S204 in its new form: an always-zero peak is a failed measurement, not a model."""
    store = _store(tmp_path)
    seed_window(store, _RERANKER_ID, bytes_per_token=0.0)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


def test_four_short_shapes_without_the_full_length_sample_are_no_model(tmp_path: Path) -> None:
    """The reviewer's case: short shapes only say nothing about a long first batch (S193)."""
    store = _store(tmp_path)
    for pairs, length in [(1, 128), (1, 256), (1, 384), (4, 256)]:
        reranker_mod._record_rerank_cost(_RERANKER_ID, store, pairs * length, 0.1, 5e6, pairs=pairs, longest=length)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


def test_an_unreadable_peak_leaves_the_older_behaviour(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, _store(tmp_path)) is True


def test_a_failed_database_read_is_no_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(reranker_mod._ram_shapes, "shape_groups", lambda *_a, **_k: None)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


# ---------------------------------------------------------------------------
# The lull job: due once per chat activity marker, never builds a thread
# ---------------------------------------------------------------------------


def test_the_job_is_due_only_while_needed_unmeasured_and_once_per_marker(tmp_path: Path) -> None:
    store = _store(tmp_path)
    open_store = _factory(store)
    assert reranker_mod.ram_figure_due("m1", open_store) is False, "no turn flagged the need"
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    assert reranker_mod.ram_figure_due("m1", open_store) is True
    reranker_mod._ram_figure_attempt[_RERANKER_ID] = "m1"
    assert reranker_mod.ram_figure_due("m1", open_store) is False, "once per chat activity marker"
    assert reranker_mod.ram_figure_due("m2", open_store) is True, "a later lull retries"
    seed_window(store, _RERANKER_ID)
    assert reranker_mod.ram_figure_due("m3", open_store) is False
    assert _RERANKER_ID not in reranker_mod._ram_figure_needed, "a flag whose model exists is dropped"


def test_a_kept_fit_alone_leaves_the_job_due_until_the_window_is_refreshed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    seed_window(store, _RERANKER_ID)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True
    monkeypatch.setattr(reranker_mod._ram_shapes, "shape_groups", lambda *_a, **_k: [])
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True  # flags the need
    assert reranker_mod.ram_figure_due("m1", _factory(store)) is True


def test_the_job_run_builds_the_ladder_clears_the_flag_and_a_failed_run_keeps_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    bad = ScriptedLadderProvider(growth_fn=lambda *_a: None, model=_RERANKER_ID)  # the peak cannot be read
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: bad)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}
    assert reranker_mod._ram_figure_attempt[_RERANKER_ID] == "m1"

    good = ScriptedLadderProvider(model=_RERANKER_ID)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: good)
    assert reranker_mod.run_ram_figure("m2", store) is True
    assert reranker_mod._ram_figure_needed == set()
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True


def test_the_job_run_with_a_model_already_present_measures_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    seed_window(store, _RERANKER_ID)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    boom = ScriptedLadderProvider(model=_RERANKER_ID)
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kw: pytest.fail("a model exists: nothing is built")
    )
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert (boom.probes, boom.timed) == (0, [])
    assert reranker_mod._ram_figure_needed == set()


# ---------------------------------------------------------------------------
# S201: `nell chat --no-bridge` measures the figure itself, in the background
# ---------------------------------------------------------------------------


@pytest.fixture
def _background_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.memory import floor_startup

    monkeypatch.setattr(floor_startup, "_background_inhibited", False)


def _store_factory(store: MemoryStore):
    @contextmanager
    def _factory() -> Iterator[MemoryStore]:
        yield store

    return _factory


def test_s201_with_no_figure_and_a_registered_direct_chat_the_measurement_runs_on_a_background_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _background_on: None, request: pytest.FixtureRequest
) -> None:
    """S210: the flagged turn starts nothing; the thread starts only when the REPL calls
    `start_requested_ram_figure` after the reply."""
    import threading

    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    floor_startup.enable_background_ram_figure(tmp_path)
    ran: list[tuple[str, object]] = []
    gate = threading.Event()
    owner = threading.current_thread()
    started: list = []
    release_at_teardown(request, gate, started)

    def _run(marker, st, **_kw):
        ran.append((threading.current_thread().name, st.db_path))
        wait_at_gate(gate, owner)
        seed_window(st, _RERANKER_ID)
        return True

    monkeypatch.setattr(reranker_mod, "run_ram_figure", _run)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == COSINE_PATH, "the flagged turn takes the cosine path"
    assert not floor_startup.inflight("ram-figure"), "S210: nothing runs while the flagged reply is built"
    assert ran == []
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}

    thread = floor_startup.start_requested_ram_figure()  # the REPL, after the reply is printed
    started.append(thread)
    assert thread is not None
    assert floor_startup.inflight("ram-figure"), "the measurement is in flight in the background"
    assert floor_startup.start_requested_ram_figure() is None, "the recorded need was consumed"
    gate.set()
    finish_threads(thread)
    assert [name for name, _ in ran] == ["ram-figure-measure"], "one thread, not the calling one"
    assert ran[0][1] == store.db_path, "on the persona's own database (a store of its own)"
    assert not floor_startup.inflight("ram-figure")
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store), "the figure exists afterwards"
    assert floor_startup.start_requested_ram_figure() is None, "the request was consumed by the start"


def test_s210_the_recall_only_records_the_need_and_a_second_flagged_turn_starts_nothing_either(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _background_on: None
) -> None:
    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    floor_startup.enable_background_ram_figure(tmp_path)
    monkeypatch.setattr(reranker_mod, "run_ram_figure", lambda *a, **k: pytest.fail("S210: started during the turn"))
    for _ in range(2):
        run_semantic_recall(store, tmp_path, _QUERY)
    assert floor_startup._ram_figure_requested is True  # noqa: SLF001
    assert not floor_startup.inflight("ram-figure")
    started = []
    monkeypatch.setattr(floor_startup, "_spawn", lambda kind, body, *, name: started.append(kind))
    floor_startup.start_requested_ram_figure()
    assert started == ["ram-figure"], "it starts only when the REPL asks, after the reply"


def test_s210_nothing_recorded_nothing_started(tmp_path: Path, _background_on: None) -> None:
    from brain.memory import floor_startup

    floor_startup.enable_background_ram_figure(tmp_path)
    assert floor_startup.start_requested_ram_figure() is None, "no turn flagged a need"
    floor_startup.enable_background_ram_figure(None)
    assert floor_startup.request_ram_figure() is False
    assert floor_startup.start_requested_ram_figure() is None


def test_s210_registering_again_forgets_a_recorded_need(tmp_path: Path, _background_on: None) -> None:
    from brain.memory import floor_startup

    floor_startup.enable_background_ram_figure(tmp_path)
    assert floor_startup.request_ram_figure() is True
    floor_startup.enable_background_ram_figure(tmp_path)
    assert floor_startup.start_requested_ram_figure() is None


def test_s201_without_a_registered_direct_chat_no_thread_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _background_on: None
) -> None:
    """The bridge never registers: its lull job (`rerank_ram_figure`) measures the figure."""
    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    assert floor_startup._ram_figure_persona_dir is None  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "run_ram_figure", lambda *a, **k: pytest.fail("no background measurement"))
    result = run_semantic_recall(store, tmp_path, _QUERY)
    assert result is not None and result.path == COSINE_PATH
    assert not floor_startup.inflight("ram-figure")
    assert floor_startup.request_ram_figure() is False
    assert floor_startup.start_requested_ram_figure() is None


def test_s201_with_a_saved_figure_nothing_is_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _background_on: None
) -> None:
    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    seed_window(store, _RERANKER_ID, bytes_per_token=250_000.0)
    floor_startup.enable_background_ram_figure(tmp_path)
    monkeypatch.setattr(floor_startup, "request_ram_figure", lambda: pytest.fail("a figure exists"))
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **k: _CappedReranker())
    run_semantic_recall(store, tmp_path, _QUERY)
    assert reranker_mod._ram_figure_needed == set()


def test_s201_the_body_measures_once_per_activity_marker_with_a_store_of_its_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    store.close()
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    monkeypatch.setattr(floor_startup, "_marker_provider", lambda: "m1")
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **k: _CappedReranker())
    reranker_mod._warmed_model_ids.add(_RERANKER_ID)
    floor_startup._ram_figure_body(tmp_path)  # noqa: SLF001
    assert reranker_mod._ram_figure_attempt[_RERANKER_ID] == "m1"
    reranker_mod._ram_figure_needed.add(_RERANKER_ID)  # flagged again, same marker: not due
    monkeypatch.setattr(reranker_mod, "run_ram_figure", lambda *a, **k: pytest.fail("once per marker"))
    floor_startup._ram_figure_body(tmp_path)  # noqa: SLF001


def test_s201_the_marker_rule_holds_when_the_first_attempt_left_no_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Able to fail on its own: no figure exists after the first attempt, so only the once-per-marker
    rule keeps the second body run from measuring again."""
    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    store.close()
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    monkeypatch.setattr(floor_startup, "_marker_provider", lambda: "m1")
    bad = _CappedReranker(rss=None)  # the peak cannot be read: the first rung ends the pass
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **k: bad)
    reranker_mod._warmed_model_ids.add(_RERANKER_ID)
    floor_startup._ram_figure_body(tmp_path)  # noqa: SLF001
    assert len(bad.timed) == 1 and reranker_mod._ram_figure_needed == {_RERANKER_ID}
    floor_startup._ram_figure_body(tmp_path)  # noqa: SLF001
    assert len(bad.timed) == 1, "same activity marker: not measured again"
    monkeypatch.setattr(floor_startup, "_marker_provider", lambda: "m2")
    floor_startup._ram_figure_body(tmp_path)  # noqa: SLF001
    assert len(bad.timed) == 2, "a later lull retries"


def test_s201_a_failing_measurement_is_logged_not_raised_and_frees_the_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _background_on: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from brain.memory import floor_startup

    closed: list[int] = []
    real_close = MemoryStore.close
    monkeypatch.setattr(MemoryStore, "close", lambda self: (closed.append(1), real_close(self))[1])

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    store.close()
    floor_startup.enable_background_ram_figure(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)

    def _boom(*a, **k):
        raise RuntimeError("the model would not load")

    monkeypatch.setattr(reranker_mod, "run_ram_figure", _boom)
    assert floor_startup.request_ram_figure() is True
    closed.clear()  # only the thread's own store counts
    with caplog.at_level("ERROR"):
        thread = floor_startup.start_requested_ram_figure()
        assert thread is not None
        finish_threads(thread)
    assert not floor_startup.inflight("ram-figure")
    assert any("ram-figure bootstrap raised" in r.getMessage() for r in caplog.records), "logged, not raised"
    assert closed, "the thread's own store was closed even though the measurement raised"


def test_s201_only_the_interactive_session_registers_the_background_measurement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain import cli
    from brain.memory import floor_startup

    monkeypatch.setattr(floor_startup, "compute_missing_floors", lambda *a, **k: None)
    monkeypatch.setattr(floor_startup, "start_background", lambda *a, **k: None)
    cli._start_floor_bootstrap_for_direct_chat(tmp_path, blocking=False)  # noqa: SLF001
    assert floor_startup._ram_figure_persona_dir == tmp_path  # noqa: SLF001
    cli._start_floor_bootstrap_for_direct_chat(tmp_path, blocking=True)  # noqa: SLF001
    assert floor_startup._ram_figure_persona_dir is None, "the one-shot form's thread would be cut off"  # noqa: SLF001


