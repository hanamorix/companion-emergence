"""S208/S220: the ladder that makes the machine's memory model. Phase 1 measures single pairs, the
full input length first and then strictly shorter ones, until the window gives a usable fit; phase 2
measures batches of increasing size at the time-budget frontier. Scripted provider with a
truncating tokenizer and a true memory law; the real `measure_ram_figure` / `run_ram_figure`
against a real MemoryStore. No model."""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: _BUDGET)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)


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


def test_phase_one_runs_the_full_pair_then_strictly_shorter_single_pairs(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)  # no phase 2 here
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert provider.timed == [(1, 1024), (1, 512), (1, 256), (1, 128)]
    assert full_marked(store, _MODEL)
    marked = [ram_fit.decode_shape(x) for x, _n, _t in reranker_mod._ram_shapes.shape_groups(store, _MODEL) if ram_fit.decode_shape(x)[2]]  # noqa: SLF001
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


def test_a_rung_runs_only_when_the_fit_says_it_fits_the_headroom(store, monkeypatch) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)
    provider = ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert provider.timed == [(1, 1024), (1, 512), (1, 256), (1, 128)], "phase 2 stopped at its first rung"


def test_phase_two_stops_at_the_first_rung_that_does_not_fit_the_headroom(store, monkeypatch) -> None:
    """S208/S220: each larger step only when its prediction fits. The first batch rung (2 pairs
    at 949 tokens) is predicted at about 400 MB on this scripted machine: not 300."""
    provider = ScriptedLadderProvider(model=_MODEL, overhead=0.1, rate=1e-3)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 300e6)
    assert reranker_mod.measure_ram_figure(provider, store) is True
    assert provider.timed == [(1, 1024), (1, 512), (1, 256), (1, 128)]
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 500e6)
    reranker_mod.measure_ram_figure(provider, store)
    assert [p for p, _l in provider.timed[4:]] == _batch_sizes(), "with room, the batches run"


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
