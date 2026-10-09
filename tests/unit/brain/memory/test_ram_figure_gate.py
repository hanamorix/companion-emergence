"""S200: no re-scoring until a RAM figure exists (name-recall fix, fixes the S193
first-rerank regression).

On a machine with no saved RAM figure a recall turn takes the plain meaning-match
(cosine) path with nothing of the reranker built, loaded, warmed up or probed on
the reply path, and flags the need; the `rerank_ram_figure` lull job then measures
the figure from ONE pair at the model's full input length, and the first real
rerank is sized from it (not the S5 width-5 seed). The older width-5 seeds of S158,
S178 and S180 are retired (S203); the S182 re-seed in `remeasure_stuck_width` stays
until the batch-sizing build (S212).

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
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _RERANKER_ID)


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
    reranker_mod._ram_cost.record(store, _RERANKER_ID, 1_000, 245_000.0)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    assert reranker_mod._ram_figure_needed == set()


def test_a_kept_figure_counts_only_in_a_process_that_has_recorded_a_rerank(tmp_path: Path) -> None:
    """S180: the sizing code uses a kept figure only for such a process; a kept figure
    alone would open the gate for a rerank sized as unseeded (the S193 shape)."""
    store = _store(tmp_path)
    key = (reranker_mod.cost_series._store_key(store), _RERANKER_ID)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False
    reranker_mod._ram_kept[key] = 100.0
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False, "kept alone is not enough"
    reranker_mod._ram_state[_RERANKER_ID] = reranker_mod._RamState(peak_x=_FULL)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True


def test_the_shipped_default_requires_a_ram_figure() -> None:
    """The suite switches the requirement off (autouse); nothing else pins the
    production default, so a flipped default would disable S200 unnoticed."""
    import inspect
    import re

    source = inspect.getsource(reranker_mod)
    assert re.search(r"^_require_ram_figure = True$", source, re.MULTILINE)


def test_a_recorded_rerank_with_no_ram_growth_is_not_a_figure(tmp_path: Path) -> None:
    """Review L1: membership in the process's measured set is not a figure."""
    store = _store(tmp_path)
    reranker_mod._record_rerank_cost(_RERANKER_ID, store, _FULL, 1.0, 0.0)
    reranker_mod._record_rerank_cost(_RERANKER_ID, store, _FULL // 2, 1.0, None)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


def test_unreadable_rss_leaves_the_older_behaviour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No figure can ever exist where RSS is unreadable: the gate does not apply
    (question to Planning)."""
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, _store(tmp_path)) is True


def test_a_failed_database_read_is_no_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(reranker_mod._ram_cost, "fit", lambda *_a, **_k: None)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


# ---------------------------------------------------------------------------
# F-B2: the measurement
# ---------------------------------------------------------------------------


def test_the_measurement_runs_two_short_warm_ups_then_one_full_length_pair(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider = _CappedReranker()

    assert reranker_mod.measure_ram_figure(provider, store) is True

    assert provider.plain == [("warm up", ["warm up"])] * 2
    ((query, documents),) = provider.timed
    assert len(documents) == 1
    assert provider.pair_token_lengths(query, documents) == [_FULL]
    # one RAM sample and one time sample, both at the pair's token count
    ram = reranker_mod._ram_cost.state(store, _RERANKER_ID).sums
    assert (ram.n, ram.sum_x, ram.sum_y) == (1, _FULL, 245e6)
    time_sums = reranker_mod._rerank_cost.state(store, _RERANKER_ID).sums
    assert (time_sums.n, time_sums.sum_x, time_sums.sum_y) == (1, _FULL, 2.0)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True


def test_the_first_real_rerank_after_the_measurement_is_sized_from_the_figure_not_the_seed(
    tmp_path: Path,
) -> None:
    """Able to fail: the width-5 seed would rerank exactly 5 real candidates."""
    store = _store(tmp_path)
    provider = _CappedReranker()
    reranker_mod.measure_ram_figure(provider, store)
    estimate = reranker_mod.rerank_cost_estimate(_RERANKER_ID, store)
    assert estimate is not None and estimate.ram_bytes_per_token == pytest.approx(245e6 / _FULL)
    provider.timed.clear()

    documents = [f"a short memory {i}" for i in range(12)]
    outcome = reranker_mod.rerank_for_recall(
        provider, "a short query", documents, store=store, budget_seconds=100.0
    )

    assert outcome.reranked and outcome.width == 12
    ((_, sent),) = provider.timed
    assert len(sent) == 12 + reranker_mod.anchor_count(12)


def test_a_pair_with_no_ram_growth_records_nothing_and_is_not_a_figure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for rss in (0.0, -5.0, None):
        provider = _CappedReranker(rss=rss)
        assert reranker_mod.measure_ram_figure(provider, store) is False
    assert reranker_mod._ram_cost.state(store, _RERANKER_ID).sums.n == 0
    assert reranker_mod._rerank_cost.state(store, _RERANKER_ID).sums.n == 0
    assert _RERANKER_ID not in reranker_mod._ram_state
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


def test_a_process_that_already_ran_a_batch_this_large_does_not_rerun_the_pair(
    tmp_path: Path,
) -> None:
    """No RSS growth can show past the high water, so the pair is not even run (it
    would cost a full-length rerank every lull for nothing), and it is not a figure."""
    store = _store(tmp_path)
    reranker_mod._ram_state[_RERANKER_ID] = reranker_mod._RamState(peak_x=_FULL)
    provider = _CappedReranker()
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == []
    assert provider.plain == [], "S213: refused before the warm-up reranks"


def test_s213_a_known_full_pair_length_refuses_before_the_provider_is_even_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The job remembers the pair's token count per model; once the high water covers it, a
    lull builds, loads and warms up nothing (and the attempt is still stamped for this lull)."""
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    first = _CappedReranker(rss=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: first)
    assert reranker_mod.run_ram_figure("m1", store) is False  # no growth: not a figure, tokens learned
    assert reranker_mod._full_pair_tokens[_RERANKER_ID] == _FULL

    reranker_mod._ram_state[_RERANKER_ID] = reranker_mod._RamState(peak_x=_FULL)
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda **kw: pytest.fail("S213: provider built")
    )
    assert reranker_mod.run_ram_figure("m2", store) is False
    assert reranker_mod._ram_figure_attempt[_RERANKER_ID] == "m2"
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}, "the flag stays"


def test_s213_a_known_length_and_a_covering_high_water_refuse_before_even_the_pair_is_probed(
    tmp_path: Path,
) -> None:
    """The first refusal in `measure_ram_figure`: no tokenization, no warm-up, no pair."""
    store = _store(tmp_path)
    reranker_mod._full_pair_tokens[_RERANKER_ID] = _FULL
    reranker_mod._ram_state[_RERANKER_ID] = reranker_mod._RamState(peak_x=_FULL)
    provider = _CappedReranker()
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert (provider.probes, provider.plain, provider.timed) == (0, [], [])


def test_s213_a_lower_high_water_still_measures_after_the_length_is_known(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reranker_mod._full_pair_tokens[_RERANKER_ID] = _FULL
    reranker_mod._ram_state[_RERANKER_ID] = reranker_mod._RamState(peak_x=_FULL - 1)
    provider = _CappedReranker()
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert len(provider.timed) == 1


def test_a_pair_whose_ram_sample_was_not_taken_is_not_reported_as_measured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The return value is whether the window then holds a figure."""
    store = _store(tmp_path)
    monkeypatch.setattr(reranker_mod, "_record_rerank_cost", lambda *a, **k: None)
    assert reranker_mod.measure_ram_figure(_CappedReranker(), store) is False


def test_a_pair_that_never_reaches_the_input_limit_is_not_measured(tmp_path: Path) -> None:
    """A tokenizer whose count keeps growing with the text never shows the limit:
    the pair is not full length, so no (too low) figure is recorded."""
    store = _store(tmp_path)
    provider = _CappedReranker(cap=None)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == []


# ---------------------------------------------------------------------------
# The lull job: due once per chat activity marker, never builds a thread
# ---------------------------------------------------------------------------


def _factory(store: MemoryStore):
    @contextmanager
    def _open() -> Iterator[MemoryStore]:
        yield store

    return _open


def test_the_job_is_due_only_while_needed_unmeasured_and_once_per_marker(tmp_path: Path) -> None:
    store = _store(tmp_path)
    open_store = _factory(store)
    assert reranker_mod.ram_figure_due("m1", open_store) is False, "no turn flagged the need"
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    assert reranker_mod.ram_figure_due("m1", open_store) is True
    reranker_mod._ram_figure_attempt[_RERANKER_ID] = "m1"
    assert reranker_mod.ram_figure_due("m1", open_store) is False, "once per chat activity marker"
    assert reranker_mod.ram_figure_due("m2", open_store) is True, "a later lull retries"
    reranker_mod._ram_kept[(reranker_mod.cost_series._store_key(store), _RERANKER_ID)] = 5.0
    reranker_mod._ram_state[_RERANKER_ID] = reranker_mod._RamState(peak_x=_FULL)
    assert reranker_mod.ram_figure_due("m3", open_store) is False
    assert _RERANKER_ID not in reranker_mod._ram_figure_needed, "a flag whose figure exists is dropped"


def test_the_job_run_measures_clears_the_flag_and_a_failed_run_keeps_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    bad = _CappedReranker(rss=0.0)
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: bad)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}
    assert reranker_mod._ram_figure_attempt[_RERANKER_ID] == "m1"

    good = _CappedReranker()
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: good)
    assert reranker_mod.run_ram_figure("m2", store) is True
    assert reranker_mod._ram_figure_needed == set()
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is True


def test_the_job_run_with_a_figure_already_present_measures_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod._ram_cost.record(store, _RERANKER_ID, 1_000, 245_000.0)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    boom = _CappedReranker()
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: boom)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert boom.timed == [] and boom.plain == []


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _background_on: None
) -> None:
    """S210: the flagged turn starts nothing; the thread starts only when the REPL calls
    `start_requested_ram_figure` after the reply."""
    import threading

    from brain.memory import floor_startup

    store, _memories = _recall_setup(monkeypatch, tmp_path)
    floor_startup.enable_background_ram_figure(tmp_path)
    ran: list[tuple[str, object]] = []
    gate = threading.Event()

    def _run(marker, st, **_kw):
        ran.append((threading.current_thread().name, st.db_path))
        gate.wait(10)
        reranker_mod._record_rerank_cost(_RERANKER_ID, st, _FULL, 2.0, 245e6)
        return True

    monkeypatch.setattr(reranker_mod, "run_ram_figure", _run)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == COSINE_PATH, "the flagged turn takes the cosine path"
    assert not floor_startup.inflight("ram-figure"), "S210: nothing runs while the flagged reply is built"
    assert ran == []
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}

    thread = floor_startup.start_requested_ram_figure()  # the REPL, after the reply is printed
    assert thread is not None
    assert floor_startup.inflight("ram-figure"), "the measurement is in flight in the background"
    assert floor_startup.start_requested_ram_figure() is None, "the recorded need was consumed"
    gate.set()
    thread.join(timeout=10)
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
    reranker_mod._ram_cost.record(store, _RERANKER_ID, _FULL, _FULL * 250_000.0)
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
    bad = _CappedReranker(rss=0.0)
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
        thread.join(timeout=10)
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


def test_s206_the_rss_is_read_through_psutil_and_an_unreadable_one_is_none_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import psutil

    monkeypatch.undo()  # the autouse gate fixture patched the reader: use the real one
    value = reranker_mod._current_rss_bytes()  # noqa: SLF001
    assert value is not None and value > 1e6
    assert value == pytest.approx(psutil.Process().memory_info().rss, rel=0.5)

    def _fail(self):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil.Process, "memory_info", _fail)
    reranker_mod._ram_figure_warned.clear()  # noqa: SLF001
    with caplog.at_level("DEBUG"):
        assert reranker_mod._current_rss_bytes() is None  # noqa: SLF001
        assert reranker_mod._current_rss_bytes() is None  # noqa: SLF001
    warns = [r for r in caplog.records if r.levelname == "WARNING" and "RSS could not be read" in r.getMessage()]
    assert len(warns) == 1, "warned once per process"
