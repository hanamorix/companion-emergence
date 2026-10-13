"""S208/S220: the ladder that makes the machine's memory model. Phase 1 measures single pairs, the
full input length first and then strictly shorter ones, until the window gives a usable fit; phase 2
measures batches of increasing size at the time-budget frontier. Scripted provider with a
truncating tokenizer and a true memory law; the real `measure_ram_figure` / `run_ram_figure`
against a real MemoryStore. No model."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import dev_constants
from brain.memory import cost_series, ram_fit
from brain.memory import reranker as reranker_mod
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.store import MemoryStore
from tests.unit.brain.memory._ram_support import (
    FULL,
    ScriptedLadderProvider,
    full_marked,
    shapes_in_window,
    true_growth,
)

_MODEL = "scripted-ladder"
_BUDGET = 2.0
_REAL_HEADROOM = reranker_mod._available_ram_headroom_bytes  # noqa: SLF001 - the fixture below replaces it


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: _BUDGET)
    # S261: free memory that cannot be read at all stops everything, so the default is ample free memory (it was
    # "unreadable", which used to mean "no memory bound"); the tests of the unreadable case say so themselves.
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)
    # S253: the first rung (and the model load) need free memory of at least the model file's size when no
    # fit is kept; this scripted model has no file, so it reads as 1.0 byte (not above the smallest positive
    # headroom these tests use). The bound itself is exercised in test_model_load_memory_floor.py.
    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", lambda model_id: 1.0)


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    yield s
    s.close()


def _frontier(pairs: int, overhead: float, rate: float, budget: float = _BUDGET) -> int:
    """Independent: the longest padded length B pairs may have and still fit the budget."""
    return min(FULL, int((budget - overhead) / rate) // pairs)


def _batch_sizes() -> list[int]:
    top = CANDIDATE_POOL + reranker_mod.P
    sizes, size = [], 2
    while size < top:
        sizes.append(size)
        size *= 2
    return [*sizes, top]


def _marker_rows(store) -> list[tuple[int, float]]:
    rows = reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    return [(n, t) for x, n, t in rows if x == ram_fit.COMPLETE_MARKER]


def test_phase_one_runs_the_full_pair_then_strictly_shorter_single_pairs(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)  # no phase 2 here
    provider = ScriptedLadderProvider(model=_MODEL)
    # S245: single pairs alone are not a finished pass (the result was True before R15): no marker
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, 1024), (1, 512), (1, 256), (1, 128)]
    assert full_marked(store, _MODEL)
    assert _marker_rows(store) == []
    marked = [ram_fit.decode_shape(x) for x, _n, _t in reranker_mod._ram_shapes.shape_groups(store, _MODEL) if x != ram_fit.COMPLETE_MARKER and ram_fit.decode_shape(x)[2]]  # noqa: SLF001
    assert marked == [(1, FULL, True)], "only the full-length pair carries the marker"
    assert len(provider.plain) == 2, "the two warm-ups ran once, before the first rung"


def test_every_later_phase_one_rung_is_dominated_by_a_shape_already_measured(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    for index, (pairs, length) in enumerate(provider.timed[1:], start=1):
        assert ram_fit.dominated(pairs, length, provider.timed[:index]), (pairs, length)


def test_phase_two_doubles_the_batch_up_to_what_a_recall_sends_at_the_time_budget_frontier(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    phase_two = provider.timed[4:]
    expected = [(b, _frontier(b, 0.1, 1e-3)) for b in _batch_sizes()]
    assert phase_two == expected
    assert [b for b, _ in expected] == [2, 4, 8, 16, 32, CANDIDATE_POOL + reranker_mod.P]
    # Every real call fits the time budget, so it lies on or under the measured frontier and inside
    # the measured ranges: the fit INTERPOLATES for it (a rung size is dominated outright).
    fit = ram_fit.fit_ram(ram_fit.groups_from_rows(reranker_mod._ram_shapes.shape_groups(store, _MODEL)))  # noqa: SLF001
    assert fit is not None
    shapes = set(shapes_in_window(store, _MODEL))
    for pairs in range(1, 59):
        longest = _frontier(pairs, 0.1, 1e-3)
        assert pairs <= fit.max_pairs and longest <= fit.max_length, (pairs, longest)
    for pairs, longest in expected:
        assert ram_fit.dominated(pairs, longest, shapes)


def test_the_pass_records_a_time_sample_and_a_shape_sample_for_every_rung(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    window = reranker_mod._rerank_cost.state(store, _MODEL).sums  # noqa: SLF001
    assert window.n == len(provider.timed) == sum(n for n, _t in shapes_in_window(store, _MODEL).values())


def test_no_rung_repeats_a_shape(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert len(set(provider.timed)) == len(provider.timed)


def test_a_single_pair_rung_runs_only_when_the_shape_that_bounds_it_fits_the_headroom(store, monkeypatch) -> None:
    """S245 (was: headroom 1.0 ran all four single pairs and the pass completed). The first rung has no measured
    shape to bound it, so it needs the model file's size (S253; this file's env reads 1 byte, so a headroom of
    1 byte lets it run; the bound itself is tested in test_model_load_memory_floor.py); the next, shorter rung
    is bounded by the measured full-length pair and a headroom of 1 byte does not hold it."""
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, 1024)], "the first rung ran; the shorter rung is refused on its measured bound"
    assert _marker_rows(store) == []


def test_a_headroom_of_exactly_zero_runs_nothing_and_records_nothing(store, monkeypatch) -> None:
    """S239/S245 (was: phase 2 stopped at its first rung and the pass still completed with the marker). A cgroup
    at its memory.high reads a headroom of 0.0: no warm-up, no length probe, no rung, no marker."""
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 0.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.plain == [] and provider.timed == [] and provider.probes == 0
    assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    assert _marker_rows(store) == []


def test_phase_two_stops_at_the_first_rung_that_does_not_fit_the_headroom(store, monkeypatch) -> None:
    """S208/S220: each larger step only when its prediction fits. The first batch rung (2 pairs
    at 949 tokens) is predicted at about 400 MB on this scripted machine: not 300. S245: that pass
    ends without a batch of more than one pair, so it records no marker and the window is no model;
    the next pass, with room, resumes and completes."""
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 300e6)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, 1024), (1, 512), (1, 256), (1, 128)]
    assert _marker_rows(store) == [] and reranker_mod._window_ready(store, _MODEL) is False  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 500e6)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert [p for p, _l in provider.timed[4:]] == _batch_sizes(), "with room, the batches run"
    assert _marker_rows(store) == [(1, 0.0)]


def test_a_second_pass_on_a_complete_window_runs_nothing(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    ran = len(provider.timed)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert len(provider.timed) == ran, "idempotent: every rung is already measured"


def test_a_pass_resumes_where_the_window_stands(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    for pairs, length, full in [(1, FULL, True), (1, 512, False)]:
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, pairs * length, 0.1, 1e6, pairs=pairs, longest=length, full_length=full
        )
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert provider.timed == [(1, 256), (1, 128)], "the full pair and the half length are not repeated"


def test_a_window_of_short_shapes_only_gets_the_full_length_pair_first(store, monkeypatch) -> None:
    """The reviewer's case: four independent short shapes pass the count and rank tests but say
    nothing about a long first batch; the ladder still measures the full-length pair."""
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    for pairs, length in [(1, 128), (1, 256), (1, 384), (4, 256)]:
        reranker_mod._record_rerank_cost(_MODEL, store, pairs * length, 0.1, 5e6, pairs=pairs, longest=length)  # noqa: SLF001
    assert reranker_mod._window_ready(store, _MODEL) is False  # noqa: SLF001
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert provider.timed == [(1, FULL)]
    assert full_marked(store, _MODEL)


def test_a_tokenizer_that_cannot_produce_a_shorter_pair_ends_the_pass_without_looping(store) -> None:
    class _NeverShorter(ScriptedLadderProvider):
        def _tokens(self, query, document):  # every pair is at the limit
            return FULL

    provider = _NeverShorter(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, FULL)], "the full rung ran; the next could not be built shorter"


def test_an_unreadable_peak_ends_the_pass_and_records_nothing(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL, growth_fn=lambda *_a: None)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, FULL)]
    assert shapes_in_window(store, _MODEL) == {}


def test_a_pair_that_never_reaches_the_input_limit_is_not_measured(store) -> None:
    class _Growing(ScriptedLadderProvider):
        def pair_token_lengths(self, query, documents):
            self.probes += 1
            return [len(query) + len(d) for d in documents]  # keeps growing with the text

    provider = _Growing(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [] and shapes_in_window(store, _MODEL) == {}


def test_text_of_length_is_the_longest_prefix_not_above_the_target(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    base = "x" * 8192
    text, length = reranker_mod._text_of_length(provider, base, 300)  # noqa: SLF001
    assert length == 300 and len(text) in (600, 601)
    assert provider.pair_token_lengths(text, [text]) == [300]
    longer = base[: len(text) + 1]
    assert provider.pair_token_lengths(longer, [longer])[0] > 300, "one more character would pass the target"
    assert reranker_mod._text_of_length(provider, base, 0) is None  # noqa: SLF001
    text, length = reranker_mod._text_of_length(provider, base, 5000)  # noqa: SLF001
    assert length == FULL and text == base[: len(text)], "saturated: the cap is the longest"


def test_the_ladder_is_the_gate_a_fresh_machine_waits_for(store) -> None:
    assert reranker_mod.ram_figure_ready(_MODEL, store) is False
    reranker_mod.measure_ram_figure(ScriptedLadderProvider(model=_MODEL), store)
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True
    estimate = reranker_mod.rerank_cost_estimate(_MODEL, store)
    assert estimate is not None and estimate.ram_fit is not None and estimate.ram_fit_kept is False
    # the model agrees with the scripted machine at the full-length pair (a shape it measured)
    cost_series.flush(store)
    measured = estimate.ram_fit.predict(1, FULL)
    assert measured == pytest.approx(219.04e6, rel=0.2)


def test_a_pass_records_the_completion_marker_once_and_a_complete_window_is_ready(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    reranker_mod.measure_ram_figure(provider, store)
    rows = reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    assert [(n, t) for x, n, t in rows if x == ram_fit.COMPLETE_MARKER] == [(1, 0.0)]
    assert reranker_mod._window_ready(store, _MODEL) is True  # noqa: SLF001


def test_a_window_with_enough_shapes_but_no_completion_marker_is_no_model(store) -> None:
    """The M1 case: single pairs alone would size the first batches (under-predicting them)."""
    for pairs, length, full in [(1, FULL, True), (1, 512, False), (1, 256, False), (1, 128, False)]:
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, pairs * length, 0.1, 5e6 * length, pairs=pairs, longest=length, full_length=full
        )
    assert ram_fit.enough_shapes(ram_fit.groups_from_rows(reranker_mod._ram_shapes.shape_groups(store, _MODEL)))  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is False
    assert reranker_mod.rerank_cost_estimate(_MODEL, store) is None or reranker_mod.rerank_cost_estimate(_MODEL, store).ram_fit is None


def test_a_pass_interrupted_in_phase_two_leaves_no_model_and_the_next_pass_finishes_it(store) -> None:
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    watch_clear = [True]

    def _trip(pairs: int, length: int) -> None:
        if pairs == 4:
            watch_clear[0] = False

    provider.on_timed = _trip
    with pytest.raises(reranker_mod.RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=lambda: watch_clear[0])
    assert [p for p, _l in provider.timed] == [1, 1, 1, 1, 2, 4], "interrupted during the 4-pair rung"
    assert reranker_mod._window_ready(store, _MODEL) is False  # noqa: SLF001
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.ram_figure_due("later", lambda: _open(store)) is True, "the lull picks it up again"
    provider.on_timed = None
    provider.timed.clear()
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert [p for p, _l in provider.timed] == [4, 8, 16, 32, 58], "resumed from the 4-pair rung"
    assert reranker_mod._window_ready(store, _MODEL) is True  # noqa: SLF001


def test_a_phase_two_rung_that_cannot_be_read_leaves_the_pass_incomplete(store) -> None:
    provider = ScriptedLadderProvider(
        model=_MODEL, overhead=0.1, rate=1e-3, growth_fn=lambda p, ln: true_growth(p, ln) if p == 1 else None
    )
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert [p for p, _l in provider.timed] == [1, 1, 1, 1, 2], "the pass stopped at the first unreadable batch"
    assert reranker_mod._window_ready(store, _MODEL) is False  # noqa: SLF001


def test_a_rung_that_records_nothing_ends_the_pass_instead_of_repeating(store, monkeypatch) -> None:
    """A shape that cannot be packed records no sample; the pass must not run it again and again."""
    monkeypatch.setattr(ram_fit, "MAX_LENGTH", 100)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, FULL)]


def test_the_frontier_length_is_capped_at_the_model_limit(store, monkeypatch) -> None:
    """A budget far above the cost of a full-length pair would put the frontier beyond the model
    limit: the rung is capped at the limit, so it is measured once and then counts as measured."""
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 100.0)
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    reranker_mod.measure_ram_figure(provider, store)
    assert provider.timed[4] == (2, FULL), "2 pairs at the limit, not beyond it"
    again = len(provider.timed)
    reranker_mod.measure_ram_figure(provider, store)
    assert len(provider.timed) == again


def test_where_the_peak_cannot_be_read_no_lull_attempt_is_ever_due(store, monkeypatch) -> None:
    """No RAM bound exists there (S206 posture): the window counts as ready, so a flagged need
    is dropped at once and no provider is built per lull."""
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.ram_figure_due("m1", lambda: _open(store)) is False
    assert _MODEL not in reranker_mod._ram_figure_needed  # noqa: SLF001


def _open(store):
    from contextlib import nullcontext

    return nullcontext(store)


def test_the_marker_arriving_after_the_shapes_opens_the_gate_at_once(store) -> None:
    """The pass records the marker LAST: the groups are unchanged when it lands, so a stale
    memo of 'no model' (keyed on the groups alone) would keep the gate shut."""
    for pairs, length, full in [(1, FULL, True), (1, 512, False), (1, 256, False), (2, 128, False)]:
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, pairs * length, 0.1, 5e6 * length, pairs=pairs, longest=length, full_length=full
        )
    assert reranker_mod.ram_figure_ready(_MODEL, store) is False
    reranker_mod._ram_shapes.record(store, _MODEL, ram_fit.COMPLETE_MARKER, 0.0)  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True


# ---- S242: a pass that stopped before its first batch leaves a fit that holds single pairs only ----

_QUERY = "which name did we settle on for the dog"


def _docs(n: int, chars: int = 400) -> list[str]:
    return [f"{i} " + "word " * (chars // 5) for i in range(n)]


def _headroom_stopped_pass(store, monkeypatch) -> ScriptedLadderProvider:
    """A window with a completion marker and single pairs only: the window an earlier build left (it recorded
    the marker for a pass the headroom stopped before its first batch). Since S245 a ladder pass can no longer
    produce it, so it is written directly: the four phase-1 shapes at the scripted machine's true growth and
    time, then the marker. Afterwards the live headroom is generous."""
    for length, full in [(FULL, True), (512, False), (256, False), (128, False)]:
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, length, 0.1 + 1e-3 * length, true_growth(1, length), pairs=1, longest=length, full_length=full
        )
    reranker_mod._ram_shapes.record(store, _MODEL, ram_fit.COMPLETE_MARKER, 0.0)  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e12)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod._warm_up_once(provider, "warm up", ["warm up"])  # noqa: SLF001 - the pass the window came from warmed it up
    return provider


def test_a_marked_window_of_single_pairs_gives_a_fit_that_holds_single_pairs_only(store, monkeypatch) -> None:
    _headroom_stopped_pass(store, monkeypatch)
    rows = reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    assert ram_fit.ladder_complete(rows), "the marker is there"
    fit = reranker_mod._window_ram_fit(store, _MODEL)  # noqa: SLF001
    assert fit is not None and fit.max_pairs == 1 and fit.max_length == FULL


def test_a_recall_on_a_marked_single_pair_window_takes_the_cosine_path_and_a_complete_pass_reranks(
    store, monkeypatch, tmp_path
) -> None:
    provider = _headroom_stopped_pass(store, monkeypatch)
    before = len(provider.timed)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off, out.measured) == (False, "budget", False)
    assert len(provider.timed) == before, "no rerank ran"
    # the control: the same recall on a store whose ladder ran to the end (ample free memory = the rungs run)
    other = MemoryStore(tmp_path / "other.db", integrity_check=False)
    try:
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)
        control = ScriptedLadderProvider(model=_MODEL)
        assert reranker_mod.measure_ram_figure(control, other) is True
        fit = reranker_mod._window_ram_fit(other, _MODEL)  # noqa: SLF001
        assert fit is not None and fit.max_pairs > 1
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e12)
        done = len(control.timed)
        out = reranker_mod.rerank_for_recall(control, _QUERY, _docs(50), store=other, budget_seconds=4.0)
        assert out.reranked is True and out.width >= reranker_mod.RERANK_MIN_REAL_CANDIDATES
        assert len(control.timed) == done + 1
    finally:
        other.close()


def test_a_measured_batch_of_two_pairs_lifts_the_cap_for_the_same_window(store, monkeypatch) -> None:
    provider = _headroom_stopped_pass(store, monkeypatch)
    reranker_mod._record_rerank_cost(_MODEL, store, 2 * 128, 0.1, 5e6, pairs=2, longest=128)  # noqa: SLF001
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert out.reranked is True


def test_a_pass_ended_by_the_time_budget_records_no_marker_and_the_window_is_no_model(store, monkeypatch) -> None:
    """S245 (was: a budget at or below the call overhead recorded the marker and left a capped single-pair fit
    for 24 hours, the cosine path either way). Now the pass records no marker, the window gives no model, the
    recall takes the ram_figure hand-off, and the lull retries."""
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert _marker_rows(store) == []
    assert reranker_mod._window_ram_fit(store, _MODEL) is None  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: _BUDGET)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e12)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off) == (False, "ram_figure")
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.ram_figure_due("next-lull", lambda: _open(store)) is True


def test_the_ladders_own_batch_rungs_are_not_refused_by_the_cap(store, monkeypatch) -> None:
    """The cap holds back real reranks, not the calls that measure larger batches: a later pass on the capped
    (legacy marked) window, with room, runs its first batch rung of 2 pairs."""
    provider = _headroom_stopped_pass(store, monkeypatch)
    before = len(provider.timed)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert provider.timed[before][0] == 2 and len(provider.timed) > before
    fit = reranker_mod._window_ram_fit(store, _MODEL)  # noqa: SLF001
    assert fit is not None and fit.max_pairs > 1


def _stuck_miss() -> reranker_mod._BudgetMiss:
    return reranker_mod._BudgetMiss(  # noqa: SLF001
        query=_QUERY,
        documents=tuple(_docs(5)),
        candidate_tokens=(1000,) * 5,
        anchor_tokens=(50,) * len(reranker_mod.ANCHOR_POOL),
    )


def test_the_stuck_width_re_measure_does_not_run_a_batch_larger_than_any_measured(store, monkeypatch, caplog) -> None:
    """S242 in the second caller: the miss is time-stuck (5 long candidates exceed the full budget), the window is
    capped, the live headroom is generous: the job skips (the miss stays, no rerank ran). On a window with a
    measured batch it runs."""
    provider = _headroom_stopped_pass(store, monkeypatch)
    reranker_mod._budget_misses[_MODEL] = _stuck_miss()  # noqa: SLF001
    reranker_mod._provider_cache[_MODEL] = provider  # noqa: SLF001
    try:
        before = len(provider.timed)
        with caplog.at_level("INFO"):
            assert reranker_mod.remeasure_stuck_width("marker-1", store) is False
        assert len(provider.timed) == before
        assert reranker_mod._budget_misses.get(_MODEL) is not None  # noqa: SLF001
        assert any("larger than any shape measured" in r.getMessage() for r in caplog.records)
        reranker_mod._record_rerank_cost(_MODEL, store, 2 * 128, 0.1, 5e6, pairs=2, longest=128)  # noqa: SLF001
        assert reranker_mod.remeasure_stuck_width("marker-2", store) is True
        assert len(provider.timed) == before + 1
    finally:
        reranker_mod._provider_cache.pop(_MODEL, None)  # noqa: SLF001


def test_the_stuck_width_re_measure_skips_on_the_smaller_of_the_cgroup_figure_and_psutil(store, monkeypatch, caplog) -> None:
    """S241 in the second caller, through the real headroom function: a window with a measured batch (cap lifted),
    a cgroup cap far above any host, and a host with 1 byte available: the batch does not fit, the job skips."""
    provider = _headroom_stopped_pass(store, monkeypatch)
    reranker_mod._record_rerank_cost(_MODEL, store, 2 * 128, 0.1, 5e6, pairs=2, longest=128)  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", _REAL_HEADROOM)
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: 5e12)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 1.0)
    reranker_mod._budget_misses[_MODEL] = _stuck_miss()  # noqa: SLF001
    reranker_mod._provider_cache[_MODEL] = provider  # noqa: SLF001
    try:
        before = len(provider.timed)
        with caplog.at_level("INFO"):
            assert reranker_mod.remeasure_stuck_width("marker-1", store) is False
        assert len(provider.timed) == before
        assert any("does not fit the RAM headroom" in r.getMessage() for r in caplog.records)
        monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 5e12)
        assert reranker_mod.remeasure_stuck_width("marker-2", store) is True
    finally:
        reranker_mod._provider_cache.pop(_MODEL, None)  # noqa: SLF001


# ---- S241 through the ladder: the real headroom function, its sources patched ----


def test_the_ladder_stops_on_the_smaller_of_the_cgroup_figure_and_psutil(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", _REAL_HEADROOM)
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: 5e12)  # a cap above any host
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 1.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    # S245: psutil's 1 byte (the smaller figure) lets the unbounded first rung run and holds the next one (was: 4 rungs and True)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, 1024)], "psutil's 1 byte stopped the pass after the first rung"


def test_the_ladder_runs_its_batches_when_both_sources_have_room(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", _REAL_HEADROOM)
    monkeypatch.setattr(reranker_mod, "_cgroup_v2_memory_headroom_bytes", lambda: 5e12)
    monkeypatch.setattr(reranker_mod, "_cgroup_v1_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(reranker_mod, "_psutil_available_bytes", lambda: 5e12)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert [p for p, _l in provider.timed[4:]] == _batch_sizes()



# ---- S245: the headroom is checked before the model load and before every rung; the marker needs a batch ----


def _build_spy(monkeypatch, provider) -> list[int]:
    built: list[int] = []
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: built.append(1) or provider)
    return built


@pytest.mark.parametrize("headroom", [0.0, -1.0])
def test_run_ram_figure_builds_and_loads_nothing_when_the_headroom_is_known_and_not_positive(
    store, monkeypatch, headroom
) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    built = _build_spy(monkeypatch, provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert built == [] and provider.plain == [] and provider.timed == [] and provider.probes == 0
    assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    assert _MODEL in reranker_mod._ram_figure_needed, "the need flag is kept"
    assert reranker_mod.ram_figure_due("m1", lambda: _open(store)) is False, "once per chat activity marker"
    assert reranker_mod.ram_figure_due("m2", lambda: _open(store)) is True, "the next lull retries"


@pytest.mark.parametrize("headroom", [1.0, 5e12])
def test_run_ram_figure_builds_the_provider_when_the_headroom_is_positive(
    store, monkeypatch, headroom
) -> None:
    """S245's rule on its own: the S253 model-file bound is off in this file (the suite default), so only a
    zero headroom stops the build here; the file-size bound is tested in test_model_load_memory_floor.py."""
    provider = ScriptedLadderProvider(model=_MODEL)
    built = _build_spy(monkeypatch, provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    reranker_mod.run_ram_figure("m1", store)
    assert built == [1]


def test_run_ram_figure_builds_and_loads_nothing_when_free_memory_cannot_be_read(store, monkeypatch) -> None:
    """S261 (was: an unreadable headroom builds the provider, S245): the simple safe fallback."""
    provider = ScriptedLadderProvider(model=_MODEL)
    built = _build_spy(monkeypatch, provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert built == [] and provider.plain == [] and provider.timed == []
    assert _MODEL in reranker_mod._ram_figure_needed, "the need flag is kept"  # noqa: SLF001


@pytest.mark.parametrize(("delta", "rungs"), [(0.0, 4), (-1.0, 1)])
def test_a_phase_one_rung_runs_when_the_dominating_shapes_growth_equals_the_headroom_and_not_below(
    store, monkeypatch, delta, rungs
) -> None:
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    headroom = true_growth(1, FULL) + delta
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert len(provider.timed) == rungs


def test_a_phase_one_rung_is_bounded_by_the_smallest_shape_that_dominates_it(store, monkeypatch) -> None:
    """A real recall's 4-pair group at the full length (2 GB here) dominates every shorter single pair too, but
    the full-length single pair (about 220 MB) is the tighter bound: with 300 MB the shorter rungs run."""
    reranker_mod._record_rerank_cost(  # noqa: SLF001
        _MODEL, store, FULL, 0.1 + 1e-3 * FULL, true_growth(1, FULL), pairs=1, longest=FULL, full_length=True
    )
    reranker_mod._record_rerank_cost(_MODEL, store, 4 * FULL, 0.1 + 4e-3 * FULL, 2e9, pairs=4, longest=FULL)  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 300e6)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert provider.timed == [(1, 512), (1, 256)], "both shorter rungs ran (four shapes then make a usable fit)"


def test_a_shorter_low_growth_shape_does_not_bound_a_longer_rung(store, monkeypatch) -> None:
    """Only a shape that is at least as long (and has at least as many pairs) bounds a rung: a small 2-pair
    shape at 128 tokens says nothing about a 512-token pair, so the full-length pair's growth (about 220 MB)
    holds the rung when only 100 MB is free."""
    reranker_mod._record_rerank_cost(  # noqa: SLF001
        _MODEL, store, FULL, 0.1 + 1e-3 * FULL, true_growth(1, FULL), pairs=1, longest=FULL, full_length=True
    )
    reranker_mod._record_rerank_cost(_MODEL, store, 2 * 128, 0.1 + 1e-3 * 256, 1e6, pairs=2, longest=128)  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 100e6)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert provider.timed == []


def test_the_dominating_shapes_growth_is_its_mean_over_its_samples(store, monkeypatch) -> None:
    """Two samples of the full-length pair, each about 220 MB: the bound is 220 MB (the mean), not 440 MB (the sum)."""
    for _ in range(2):
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, FULL, 0.1 + 1e-3 * FULL, true_growth(1, FULL), pairs=1, longest=FULL, full_length=True
        )
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.5 * true_growth(1, FULL))
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert provider.timed and provider.timed[0] == (1, 512)


def test_the_headroom_is_read_fresh_before_every_rung(store, monkeypatch) -> None:
    readings = iter([5e12, 5e12])  # the warm-up check and the first rung's; every later read is 0.0
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: next(readings, 0.0))
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert provider.timed == [(1, 1024)]


def test_an_unreadable_headroom_stops_the_ladder_before_anything_runs(store, monkeypatch) -> None:
    """S261 (was: an unreadable headroom does not stop the ladder): no warm-up, no rung, nothing recorded."""
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.plain == [] and provider.timed == [] and provider.probes == 0
    assert _marker_rows(store) == [] and not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001


def test_free_memory_that_becomes_unreadable_mid_pass_stops_it_at_the_next_rung(store, monkeypatch) -> None:
    readings = iter([1e15, 1e15])  # the warm-up check and the first rung's check; then nothing can be read
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: next(readings, None))
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.timed == [(1, FULL)], "the first rung ran, the second was stopped"


def test_a_pass_stopped_at_a_later_batch_rung_on_a_window_that_holds_a_batch_still_records_the_marker(
    store, monkeypatch
) -> None:
    """S245 reads 'a multi-pair rung' against the window (an earlier pass, or a recall, took the 2-pair
    batch). The window of {1,2}-pair shapes is then complete and its cap is lifted: S244's literal wording,
    the disclosed R5b."""
    for length, full in [(FULL, True), (512, False), (256, False), (128, False)]:
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, length, 0.1 + 1e-3 * length, true_growth(1, length), pairs=1, longest=length, full_length=full
        )
    reranker_mod._record_rerank_cost(_MODEL, store, 2 * 949, 0.1 + 1e-3 * 2 * 949, true_growth(2, 949), pairs=2, longest=949)  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 100e6)  # refuses the 4-pair rung
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert provider.timed == [], "nothing new was run: the single pairs and the 2-pair batch were measured, the 4-pair refused"
    assert _marker_rows(store) == [(1, 0.0)]
    fit = reranker_mod._window_ram_fit(store, _MODEL)  # noqa: SLF001
    assert fit is not None and fit.max_pairs == 2 and fit.within_measured(58, 1024)


def test_a_starved_lull_is_retried_and_the_next_lull_finishes_the_ladder_so_the_recall_reranks(store, monkeypatch) -> None:
    """S245's purpose, executed: a pass the headroom stopped at the first batch rung leaves no marker, so
    nothing is pinned to the cosine path for 24 hours: the next lull, with room, completes the window and
    the recall reranks (R14: the capped single-pair fit stayed until the window aged out)."""
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    built = _build_spy(monkeypatch, provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 300e6)  # the 2-pair rung needs about 400 MB
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert _marker_rows(store) == [] and reranker_mod.ram_figure_ready(_MODEL, store) is False
    assert reranker_mod.ram_figure_due("m1", lambda: _open(store)) is False
    assert reranker_mod.ram_figure_due("m2", lambda: _open(store)) is True
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    assert reranker_mod.run_ram_figure("m2", store) is True
    assert built == [1, 1] and _marker_rows(store) == [(1, 0.0)]
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert out.reranked is True and out.width >= reranker_mod.RERANK_MIN_REAL_CANDIDATES


# ---- S247: the seed-width recall rerank, the recall-side warm-up and the bundled normalization ----


def _fit_with(*, a=0.0, b=0.0, c=0.0, max_pairs=58, max_length=FULL) -> ram_fit.RamFit:
    return ram_fit.RamFit(a=a, b=b, c=c, shapes=5, samples=5, max_pairs=max_pairs, max_length=max_length)


def test_the_seed_width_must_fit_the_memory_model_and_the_headroom() -> None:
    candidates, anchors = [100] * 10, [50] * reranker_mod.P
    fit = _fit_with(b=1000.0)  # 1000 bytes per padded token
    pairs = reranker_mod.RERANK_MIN_REAL_CANDIDATES + reranker_mod.anchor_count(reranker_mod.RERANK_MIN_REAL_CANDIDATES)
    need = fit.predict(pairs, 100)
    assert need == pairs * 100 * 1000.0
    width = reranker_mod.fit_rerank_width
    assert width(candidates, anchors, None, 4.0, need, seed_ram_fit=fit) == reranker_mod.RERANK_MIN_REAL_CANDIDATES
    assert width(candidates, anchors, None, 4.0, need - 1.0, seed_ram_fit=fit) == 0
    # unchanged without a memory model or without a headroom reading
    assert width(candidates, anchors, None, 4.0, need - 1.0) == reranker_mod.RERANK_MIN_REAL_CANDIDATES
    assert width(candidates, anchors, None, 4.0, None, seed_ram_fit=fit) == reranker_mod.RERANK_MIN_REAL_CANDIDATES
    # the S242 cap applies to the seed batch too
    capped = _fit_with(b=1000.0, max_pairs=1)
    assert width(candidates, anchors, None, 4.0, 1e12, seed_ram_fit=capped) == 0
    assert width(candidates, anchors, None, 4.0, 1e12, seed_ram_fit=_fit_with(b=1000.0, max_pairs=2)) == 5


def _full_window(store, monkeypatch) -> tuple[ScriptedLadderProvider, ram_fit.RamFit]:
    """A complete ladder pass (ample free memory, every rung runs) and the process's kept fit it produced."""
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    fit = reranker_mod._window_ram_fit(store, _MODEL)  # noqa: SLF001
    assert fit is not None and fit.max_pairs > 1
    return provider, fit


def test_a_recall_with_no_time_estimate_runs_its_seed_batch_only_inside_the_ram_bound(store, monkeypatch) -> None:
    """Plan review M1: the 24-hour windows aged out (a quiet day) while the process keeps its memory model
    (H4b). The seed batch (5 real + 2 anchors) was sized by nothing; now it has to fit the model and the headroom."""
    import time

    provider, _fit = _full_window(store, monkeypatch)
    later = time.time() + 25 * 3600
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: later)
    assert reranker_mod.rerank_cost_estimate(_MODEL, store) is None, "the time window is empty"
    assert reranker_mod._window_ram_fit(store, _MODEL) is None, "and so is the memory window"  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True, "the kept fit still opens the gate"
    before = len(provider.timed)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50, chars=3000), store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off, out.width) == (False, "budget", 0)
    assert len(provider.timed) == before, "no rerank ran"
    assert reranker_mod._budget_misses.get(_MODEL) is None, "no time-stuck miss is recorded for a RAM refusal"  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50, chars=3000), store=store, budget_seconds=4.0)
    assert out.reranked is True and out.width == reranker_mod.RERANK_MIN_REAL_CANDIDATES
    assert len(provider.timed) == before + 1


def _unwarmed(provider) -> int:
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._warmed_model_ids.discard(_MODEL)  # noqa: SLF001
    return len(provider.plain)


def test_the_recall_warm_up_runs_only_when_each_one_pair_batch_fits(store, monkeypatch) -> None:
    provider, fit = _full_window(store, monkeypatch)
    docs = _docs(3, chars=3000)  # about 750 tokens: a one-pair call that grows the process
    tokens = provider.pair_token_lengths(_QUERY, [docs[0]])[0]
    need = fit.predict(1, tokens)
    assert need > 0.0
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need - 1.0)
    reranker_mod._warm_up_once(provider, _QUERY, docs, bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before and _MODEL not in reranker_mod._warmed_model_ids  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need)
    reranker_mod._warm_up_once(provider, _QUERY, docs, bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before + 2 and _MODEL in reranker_mod._warmed_model_ids  # noqa: SLF001


def test_the_warm_up_is_not_bounded_for_the_ladder_nor_where_no_bound_applies_and_is_refused_with_no_headroom_reading(
    store, monkeypatch
) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    docs = _docs(3)
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 0.0)
    reranker_mod._warm_up_once(provider, "warm up", ["warm up"])  # noqa: SLF001 - the ladder's own call: bounded=False
    assert len(provider.plain) == before + 2
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)  # no RAM bound on this machine
    reranker_mod._warm_up_once(provider, _QUERY, docs, bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before + 2
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    reranker_mod._warm_up_once(provider, _QUERY, docs, bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before, "S261: no headroom reading: the bounded warm-up is refused (it used to run)"


def test_a_warm_up_with_no_memory_model_in_the_process_waits(store, monkeypatch) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    before = _unwarmed(provider)
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept.clear()  # noqa: SLF001
    reranker_mod._warm_up_once(provider, _QUERY, _docs(3), bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before, "no memory model: the one-pair batch cannot be sized"


def test_a_warmed_model_does_no_bound_work_even_with_a_memory_model_held(store, monkeypatch) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    assert reranker_mod._kept_ram_fits(_MODEL), "the memory model is held, so a bound check WOULD tokenize"  # noqa: SLF001
    assert _MODEL in reranker_mod._warmed_model_ids  # noqa: SLF001
    probes, before = provider.probes, len(provider.plain)
    reranker_mod._warm_up_once(provider, _QUERY, _docs(3), bounded=True)  # noqa: SLF001
    assert provider.probes == probes and len(provider.plain) == before


def test_an_unreadable_headroom_refuses_without_tokenizing_and_every_warm_up_document_is_checked(store, monkeypatch) -> None:
    provider, fit = _full_window(store, monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    _unwarmed(provider)
    probes = provider.probes
    reranker_mod._warm_up_once(provider, _QUERY, _docs(3, chars=3000), bounded=True)  # noqa: SLF001
    assert provider.probes == probes, "no headroom reading (S261): refused, and not even tokenized"
    assert _MODEL not in reranker_mod._warmed_model_ids  # noqa: SLF001
    # the second warm-up document is the long one: its batch does not fit although the first one's does
    short, long_ = _docs(1, chars=40)[0], _docs(1, chars=3000)[0]
    need_long = fit.predict(1, provider.pair_token_lengths(_QUERY, [long_])[0])
    assert need_long > fit.predict(1, provider.pair_token_lengths(_QUERY, [short])[0])
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need_long - 1.0)
    reranker_mod._warm_up_once(provider, _QUERY, [short, long_], bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before, "the long second document held both warm-ups back"


def test_a_starved_recall_runs_no_warm_up_call_but_still_gets_its_pair_sizes_and_warms_up_once_there_is_room(
    store, monkeypatch
) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    docs = _docs(5, chars=3000)
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    reranker_mod.warm_up_for_recall(provider, _QUERY, docs)
    sizes = reranker_mod.recall_pair_sizes(provider, _QUERY, docs)
    assert sizes.lengths is not None and len(sizes.lengths) == len(docs) + reranker_mod.P
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50, chars=3000), store=store, budget_seconds=4.0)
    assert out.reranked is False
    assert len(provider.plain) == before, "none of the three entry points ran a warm-up call"
    assert _MODEL not in reranker_mod._warmed_model_ids  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    reranker_mod.recall_pair_sizes(provider, _QUERY, docs)
    assert len(provider.plain) == before + 2 and _MODEL in reranker_mod._warmed_model_ids  # noqa: SLF001


class _ScoringProvider(ScriptedLadderProvider):
    """Scores differ per document, so a bundled run's output can be compared."""

    def rerank(self, query, documents):
        self.plain.append((query, list(documents)))
        return [float(len(d) % 7) for d in documents]


_BUNDLE = [("which name did we settle on", "we settled on a name for the dog"), ("quiet evening", "the printer jammed again")]


def _bundle_need(provider, fit, pairs=_BUNDLE) -> float:
    longest = max(
        max(provider.pair_token_lengths(q, [d, *reranker_mod.ANCHOR_POOL])) for q, d in pairs
    )
    return fit.predict(1 + len(reranker_mod.ANCHOR_POOL), longest)


def test_the_bundled_normalization_runs_when_its_batches_fit_and_equals_the_unbounded_run(store, monkeypatch) -> None:
    _built, fit = _full_window(store, monkeypatch)
    provider = _ScoringProvider(model=_MODEL)
    need = _bundle_need(provider, fit)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need)  # equal fits
    bounded = reranker_mod.normalize_bundled_pairs_against_anchors(provider, _BUNDLE)
    assert len(provider.plain) == len(_BUNDLE)
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", False)
    unbounded = reranker_mod.normalize_bundled_pairs_against_anchors(_ScoringProvider(model=_MODEL), _BUNDLE)
    assert bounded == unbounded


def test_the_bundled_normalization_defers_before_any_call_when_the_batch_does_not_fit(store, monkeypatch) -> None:
    _built, fit = _full_window(store, monkeypatch)
    provider = _ScoringProvider(model=_MODEL)
    need = _bundle_need(provider, fit)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need - 1.0)
    with pytest.raises(reranker_mod.RerankDeferred):
        reranker_mod.normalize_bundled_pairs_against_anchors(provider, _BUNDLE)
    assert provider.plain == [], "deferred before the first call"


def test_the_bundled_normalization_defers_with_no_memory_model_or_no_pair_sizes_or_a_capped_model(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    provider = _ScoringProvider(model=_MODEL)
    with pytest.raises(reranker_mod.RerankDeferred):  # (c) no kept fit in this process
        reranker_mod.normalize_bundled_pairs_against_anchors(provider, _BUNDLE)
    assert provider.plain == []
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("k", _MODEL)] = _fit_with(a=1.0)  # noqa: SLF001
    sizeless = _ScoringProvider(model=_MODEL)
    sizeless.pair_token_lengths = lambda query, documents: None  # type: ignore[method-assign]
    with pytest.raises(reranker_mod.RerankDeferred):  # (d) the batch cannot be sized
        reranker_mod.normalize_bundled_pairs_against_anchors(sizeless, _BUNDLE)
    assert sizeless.plain == []
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("k", _MODEL)] = _fit_with(a=1.0, max_pairs=1)  # noqa: SLF001
    with pytest.raises(reranker_mod.RerankDeferred):  # (e) single-pair fit: a 9-document batch is larger than measured
        reranker_mod.normalize_bundled_pairs_against_anchors(_ScoringProvider(model=_MODEL), _BUNDLE)
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("k", _MODEL)] = _fit_with(a=1.0, max_pairs=2)  # noqa: SLF001
    ran = _ScoringProvider(model=_MODEL)
    reranker_mod.normalize_bundled_pairs_against_anchors(ran, _BUNDLE)
    assert len(ran.plain) == len(_BUNDLE)


def test_the_bundled_normalization_defers_with_no_headroom_reading_and_runs_where_no_bound_applies(store, monkeypatch) -> None:
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("k", _MODEL)] = _fit_with(a=1e15)  # NOTHING would fit
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    first = _ScoringProvider(model=_MODEL)
    with pytest.raises(reranker_mod.RerankDeferred):  # (f) S261: free memory cannot be read: refused (it used to run)
        reranker_mod.normalize_bundled_pairs_against_anchors(first, _BUNDLE)
    assert first.plain == []
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)
    second = _ScoringProvider(model=_MODEL)
    reranker_mod.normalize_bundled_pairs_against_anchors(second, _BUNDLE)  # (g)
    assert len(second.plain) == len(_BUNDLE)
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept.clear()  # noqa: SLF001
    third = _ScoringProvider(model=_MODEL)
    reranker_mod.normalize_bundled_pairs_against_anchors(third, _BUNDLE)  # no memory model, but no bound either
    assert len(third.plain) == len(_BUNDLE)


def test_the_bundled_normalization_checks_every_pair_against_a_fresh_headroom(store, monkeypatch) -> None:
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("k", _MODEL)] = _fit_with(a=1.0)  # noqa: SLF001
    readings = iter([5e12])
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: next(readings, 0.0))
    provider = _ScoringProvider(model=_MODEL)
    with pytest.raises(reranker_mod.RerankDeferred):
        reranker_mod.normalize_bundled_pairs_against_anchors(provider, _BUNDLE)
    assert len(provider.plain) == 1, "the first pair ran; the second was held back when the headroom fell"


def test_every_kept_memory_model_of_the_model_id_has_to_allow_the_bundle(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e9)
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("a", _MODEL)] = _fit_with(a=1.0)  # noqa: SLF001
        reranker_mod._ram_fit_kept[("b", "another-model")] = _fit_with(a=1e15)  # noqa: SLF001
    first = _ScoringProvider(model=_MODEL)
    reranker_mod.normalize_bundled_pairs_against_anchors(first, _BUNDLE)  # another model's fit does not count
    assert len(first.plain) == len(_BUNDLE)
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[("c", _MODEL)] = _fit_with(a=1e15)  # noqa: SLF001
    with pytest.raises(reranker_mod.RerankDeferred):
        reranker_mod.normalize_bundled_pairs_against_anchors(_ScoringProvider(model=_MODEL), _BUNDLE)


# ---- S244 through the public paths: a window the usable-fit conditions refuse gives no model, a batch or not ----


def _marked_window(store, shapes: list[tuple[int, int, bool]]) -> None:
    for pairs, length, full in shapes:
        reranker_mod._record_rerank_cost(  # noqa: SLF001
            _MODEL, store, pairs * length, 0.1 + 1e-3 * pairs * length, true_growth(pairs, length),
            pairs=pairs, longest=length, full_length=full,
        )
    reranker_mod._ram_shapes.record(store, _MODEL, ram_fit.COMPLETE_MARKER, 0.0)  # noqa: SLF001


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param([(1, FULL, True), (1, 512, False), (2, 256, False)], id="three-shapes"),
        pytest.param([(1, 512, False), (1, 256, False), (1, 128, False), (2, 128, False)], id="no-full-length-sample"),
        pytest.param([(1, 100, True), (2, 100, False), (3, 100, False), (4, 100, False)], id="rank-deficient"),
    ],
)
def test_a_marked_window_with_a_batch_but_without_the_usable_fit_conditions_is_no_model_and_no_lift(
    store, shapes
) -> None:
    _marked_window(store, shapes)
    assert reranker_mod._window_ram_fit(store, _MODEL) is None  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is False
    provider = ScriptedLadderProvider(model=_MODEL)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off) == (False, "ram_figure")
    assert provider.timed == []


# ---- S249: the RAM fit is never learned in a process that downloaded the model; it does not rerank until it restarts ----


def test_a_process_that_downloaded_the_model_does_not_rerank_until_it_restarts(store, monkeypatch) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True
    reranker_mod._note_model_downloaded(_MODEL)  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is False, "even with a complete window fit"
    before = len(provider.timed)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off) == (False, "ram_figure") and len(provider.timed) == before
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.ram_figure_due("a-later-lull", lambda: _open(store)) is False, "no ladder is due"
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._downloaded_model_ids.clear()  # noqa: SLF001 - a restart
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert out.reranked is True


def test_the_ladder_builds_nothing_and_runs_nothing_in_a_process_that_downloaded_the_model(store, monkeypatch) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    built = _build_spy(monkeypatch, provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    reranker_mod.note_ram_figure_needed(_MODEL)
    reranker_mod._note_model_downloaded(_MODEL)  # noqa: SLF001
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert built == [] and provider.plain == [] and provider.timed == []
    assert reranker_mod.measure_ram_figure(provider, store) is False
    assert provider.plain == [] and provider.timed == [] and provider.probes == 0
    assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001


def test_a_build_that_itself_downloads_the_model_stops_the_pass_before_its_warm_up(store, monkeypatch) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)

    def _download_while_building(**kwargs):
        reranker_mod._note_model_downloaded(_MODEL)  # noqa: SLF001
        return provider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", _download_while_building)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 5e12)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert provider.plain == [] and provider.timed == []
    assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001


def test_a_downloaded_model_records_its_time_but_no_memory_sample(store) -> None:
    reranker_mod._note_model_downloaded(_MODEL)  # noqa: SLF001
    reranker_mod._record_rerank_cost(_MODEL, store, 100, 0.5, 5e6, pairs=1, longest=100)  # noqa: SLF001
    assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    assert reranker_mod._rerank_cost.state(store, _MODEL).sums.n == 1  # noqa: SLF001
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._downloaded_model_ids.clear()  # noqa: SLF001
    reranker_mod._record_rerank_cost(_MODEL, store, 100, 0.5, 5e6, pairs=1, longest=100)  # noqa: SLF001
    assert reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001


def test_where_no_ram_bound_applies_a_download_changes_nothing(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)
    reranker_mod._note_model_downloaded(_MODEL)  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True


def test_a_downloaded_process_has_no_ladder_due_even_with_an_empty_window(store) -> None:
    """The due check itself (a window that gives no model would otherwise make the ladder due)."""
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.ram_figure_due("lull-1", lambda: _open(store)) is True
    reranker_mod._note_model_downloaded(_MODEL)  # noqa: SLF001
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.ram_figure_due("lull-2", lambda: _open(store)) is False


# ---- S255: every batch sized from the fit is sized by the prediction times the safety factor ----

_FACTOR = dev_constants.RAM_FIT_SAFETY_FACTOR  # the shipped value, read at import (the suite default is 1.0)


@pytest.fixture
def shipped_factor(monkeypatch) -> float:
    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", _FACTOR)
    assert 1.0777 <= _FACTOR < 1.0778
    return _FACTOR


def test_the_width_scan_sizes_by_the_prediction_times_the_factor(shipped_factor) -> None:
    fit = _fit_with(b=1000.0)
    estimate = reranker_mod.RerankCostEstimate(
        overhead_seconds=0.0, seconds_per_token=0.0, ram_fit=fit, measured_batches=5
    )
    candidates, anchors = [100] * 10, [50] * reranker_mod.P
    need = fit.predict(7, 100)  # 5 real + 2 anchors, padded to 100
    assert need == 7 * 100 * 1000.0
    width = reranker_mod.fit_rerank_width
    assert width(candidates, anchors, estimate, 10.0, need * shipped_factor) == 5
    assert width(candidates, anchors, estimate, 10.0, need * shipped_factor - 1.0) == 4
    assert width(candidates, anchors, estimate, 10.0, need) < 5, "the bare prediction no longer fits"


def test_the_seed_batch_is_sized_by_the_prediction_times_the_factor(shipped_factor) -> None:
    candidates, anchors = [100] * 10, [50] * reranker_mod.P
    fit = _fit_with(b=1000.0)
    pairs = reranker_mod.RERANK_MIN_REAL_CANDIDATES + reranker_mod.anchor_count(reranker_mod.RERANK_MIN_REAL_CANDIDATES)
    need = fit.predict(pairs, 100) * shipped_factor
    width = reranker_mod.fit_rerank_width
    assert width(candidates, anchors, None, 4.0, need, seed_ram_fit=fit) == reranker_mod.RERANK_MIN_REAL_CANDIDATES
    assert width(candidates, anchors, None, 4.0, need - 1.0, seed_ram_fit=fit) == 0
    assert width(candidates, anchors, None, 4.0, fit.predict(pairs, 100), seed_ram_fit=fit) == 0


def _first_batch_rung_prediction(store, monkeypatch) -> float:
    """The prediction the ladder compares with the headroom before its first batch rung (spy; headroom ample)."""
    seen: list[tuple[float | None, str]] = []
    real = reranker_mod._ladder_headroom_stop  # noqa: SLF001

    def spy(predicted, what):
        seen.append((predicted, what))
        return real(predicted, what)

    with monkeypatch.context() as scoped:
        scoped.setattr(reranker_mod, "_ladder_headroom_stop", spy)
        scoped.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e12)
        assert reranker_mod.measure_ram_figure(ScriptedLadderProvider(model=_MODEL), store) is True
    rungs = [p for p, what in seen if what.startswith("a rung of")]
    assert rungs and rungs[0] is not None
    return rungs[0]


def test_the_phase_two_rungs_are_sized_by_the_prediction_times_the_factor(tmp_path, monkeypatch) -> None:
    def pass_at(name: str, factor: float, headroom: float | None) -> ScriptedLadderProvider | float:
        monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", factor)
        fresh = MemoryStore(tmp_path / name, integrity_check=False)
        try:
            if headroom is None:
                return _first_batch_rung_prediction(fresh, monkeypatch)
            monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
            provider = ScriptedLadderProvider(model=_MODEL)
            reranker_mod.measure_ram_figure(provider, fresh)
            return provider
        finally:
            fresh.close()

    bare = pass_at("bare.db", 1.0, None)
    scaled = pass_at("scaled.db", _FACTOR, None)
    assert 1.0777 <= _FACTOR < 1.0778 and bare > 0.0 and scaled == pytest.approx(bare * _FACTOR, rel=1e-9)
    # the stop is decided on the scaled figure: between the two the first batch rung does not run, at the scaled figure it does
    stopped = pass_at("stop.db", _FACTOR, (bare + scaled) / 2.0)
    assert [pairs for pairs, _length in stopped.timed if pairs > 1] == []
    ran = pass_at("ok.db", _FACTOR, scaled)
    assert [pairs for pairs, _length in ran.timed if pairs > 1][:1] == [2]


def test_the_stuck_width_re_measure_sizes_its_batch_by_the_prediction_times_the_factor(
    store, monkeypatch, shipped_factor
) -> None:
    provider, fit = _full_window(store, monkeypatch)
    pairs = 5 + reranker_mod.anchor_count(5)
    need = fit.predict(pairs, 1000) * shipped_factor
    assert need > 0.0
    reranker_mod._budget_misses[_MODEL] = _stuck_miss()  # noqa: SLF001
    reranker_mod._provider_cache[_MODEL] = provider  # noqa: SLF001
    try:
        before = len(provider.timed)
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need - 1.0)
        assert reranker_mod.remeasure_stuck_width("marker-a", store) is False
        assert len(provider.timed) == before, "does not fit once scaled"
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need)
        assert reranker_mod.remeasure_stuck_width("marker-b", store) is True
        assert len(provider.timed) == before + 1
    finally:
        reranker_mod._provider_cache.pop(_MODEL, None)  # noqa: SLF001


def test_the_warm_up_batch_check_sizes_by_the_prediction_times_the_factor(store, monkeypatch, shipped_factor) -> None:
    provider, fit = _full_window(store, monkeypatch)
    docs = _docs(3, chars=3000)
    tokens = provider.pair_token_lengths(_QUERY, [docs[0]])[0]
    need = fit.predict(1, tokens) * shipped_factor
    before = _unwarmed(provider)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need - 1.0)
    reranker_mod._warm_up_once(provider, _QUERY, docs, bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need)
    reranker_mod._warm_up_once(provider, _QUERY, docs, bounded=True)  # noqa: SLF001
    assert len(provider.plain) == before + 2


def test_the_floor_calibrations_bundled_batches_are_sized_by_the_prediction_times_the_factor(
    store, monkeypatch, shipped_factor
) -> None:
    _built, fit = _full_window(store, monkeypatch)
    provider = _ScoringProvider(model=_MODEL)
    need = _bundle_need(provider, fit) * shipped_factor
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need)
    reranker_mod.normalize_bundled_pairs_against_anchors(provider, _BUNDLE)
    assert len(provider.plain) == len(_BUNDLE)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: need - 1.0)
    with pytest.raises(reranker_mod.RerankDeferred):
        reranker_mod.normalize_bundled_pairs_against_anchors(_ScoringProvider(model=_MODEL), _BUNDLE)


@pytest.mark.parametrize(("delta", "rungs"), [(0.0, 4), (-1.0, 1)])
def test_a_phase_one_rung_bounded_by_a_measured_shape_is_judged_on_the_unscaled_measurement(
    store, monkeypatch, shipped_factor, delta, rungs
) -> None:
    """The factor scales PREDICTIONS. A rung a measured shape bounds is judged on that measurement as it is."""
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: true_growth(1, FULL) + delta)
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    assert len(provider.timed) == rungs


# ---- S261: free memory that cannot be read at all takes the cheaper path everywhere ----


def test_a_recall_takes_the_cosine_path_before_any_warm_up_probe_or_rerank_when_free_memory_cannot_be_read(
    store, monkeypatch
) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    _unwarmed(provider)
    probes, timed, plain = provider.probes, len(provider.timed), len(provider.plain)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off, out.width) == (False, "budget", 0)
    assert (provider.probes, len(provider.timed), len(provider.plain)) == (probes, timed, plain)
    assert reranker_mod._budget_misses.get(_MODEL) is None  # noqa: SLF001
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)
    assert reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50), store=store, budget_seconds=4.0).reranked is True


def test_the_stuck_width_re_measure_is_skipped_when_free_memory_cannot_be_read(store, monkeypatch) -> None:
    provider, _fit = _full_window(store, monkeypatch)
    reranker_mod._budget_misses[_MODEL] = _stuck_miss()  # noqa: SLF001
    reranker_mod._provider_cache[_MODEL] = provider  # noqa: SLF001
    try:
        before = len(provider.timed)
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
        assert reranker_mod.remeasure_stuck_width("marker-s261", store) is False
        assert len(provider.timed) == before
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)
        assert reranker_mod.remeasure_stuck_width("marker-s261-b", store) is True
    finally:
        reranker_mod._provider_cache.pop(_MODEL, None)  # noqa: SLF001
