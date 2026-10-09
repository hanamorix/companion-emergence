"""Name-recall fix R1-recovery: the length-aware paragraph embed cost and the
timing rules around it (spec §3, §7; S108, S109, S110, S113, S115, S117;
criteria C15b (end to end), C17, C18, C19, C20 in
`changes/name-recall-fix/R1-recovery/1.5-criteria.md`).

Offline: the axis embedder and scripted clock of `test_per_paragraph_recall.py`,
extended with a cost that grows with (texts x longest text), a load and a
first-inference cost, and in-lock timings. Token counts are the providers'
stand-in unit (one per character). Expected values are derived by hand in each
test.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import brain.memory.embeddings as embeddings_mod
import brain.memory.reranker as reranker_mod
from brain.memory import semantic_recall as sr
from brain.memory.embeddings import FastEmbedProvider
from brain.memory.reranker import CrossEncoderProvider
from brain.memory.semantic_recall import allocate_widths, run_semantic_recall, search_paragraphs
from brain.memory.store import MemoryStore
from tests.unit.brain.memory.test_monologue_last import _EMBEDDER_ID, _RERANKER_ID
from tests.unit.brain.memory.test_per_paragraph_recall import (
    _OTHER,
    _AxisEmbedder,
    _Clock,
    _cosine_floor,
    _install,
    _mem,
    _no_reranker,
    _rerank_floor,
    _Reranker,
    _warm,
)
from tests.unit.brain.memory.test_per_paragraph_recall import _budget as _set_budget

_O, _R = 0.1, 0.002  # the fake embedder's true cost: 0.1 s + 0.002 s per padded token


class _CostEmbedder(_AxisEmbedder):
    """Each batch costs `_O + _R x texts x longest text` on the clock (the S108
    shape); `load` seconds on the first model use (`ensure_loaded`, or the
    first embed if never loaded); `first` seconds on the first inference;
    optionally a lock `wait` before each timed call, reported apart through
    the timed methods (`in_lock=True`)."""

    def __init__(self, clock: _Clock, *, load=0.0, first=0.0, wait=0.0, in_lock=False, lengths=True):
        super().__init__({}, clock)
        self.load, self.first, self.wait, self.in_lock, self.lengths = load, first, wait, in_lock, lengths
        self.loaded = False
        self.inferred = False
        self.single_embeds: list[str] = []

    def _use(self) -> None:
        if not self.loaded:
            self.clock.advance(self.load)
            self.loaded = True
        if not self.inferred:
            self.clock.advance(self.first)
            self.inferred = True

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.clock.advance(self.load)
            self.loaded = True

    def embed(self, text):
        self.single_embeds.append(text)
        self._use()
        return super().embed(text)

    def embed_batch(self, texts):
        self._use()
        self.batches.append(list(texts))
        self.clock.advance(_O + _R * len(texts) * max(len(t) for t in texts))
        return [_AxisEmbedder.embed(self, t) for t in texts]

    def embed_batch_timed(self, texts):
        self.clock.advance(self.wait)
        before = self.clock.now
        vectors = self.embed_batch(texts)
        return vectors, (self.clock.now - before if self.in_lock else None)

    def token_lengths(self, texts):
        return [len(t) for t in texts] if self.lengths else None

    def token_lengths_timed(self, texts):
        self.clock.advance(self.wait)
        before = self.clock.now
        self.clock.advance(0.01)
        lengths = self.token_lengths(texts)
        return lengths, (self.clock.now - before if self.in_lock else None)


def _store(monkeypatch, tmp_path, embedder) -> MemoryStore:
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    _mem(store, "a memory of the whole message", {_OTHER: 0.9})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)
    return store


def _clocked(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    return clock


def _warm_embed_fit(store) -> None:
    """Short one-line messages of two lengths measured the true cost."""
    for n in (20, 40):
        sr._embed_cost.record(store, _EMBEDDER_ID, n, _O + _R * n)
    sr._scan_time.record(store, "", 1, 0.0)


# ---------------------------------------------------------------------------
# C17b: the fit decision predicts a paste from its own length (R6 F1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("long_chars", "per_paragraph"), [(700, False), (500, True)])
def test_c17b_a_long_paste_is_predicted_from_its_own_length(
    monkeypatch, tmp_path: Path, long_chars: int, per_paragraph: bool
) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _warm_embed_fit(store)
    sr._message_overhead.record(store, "", 1, 0.0)
    p1, p3 = "short one words", "short two words"
    long_p = ("lorem ipsum " * 100)[:long_chars]
    paste = f"{p1}\n{long_p}\n{p3}"
    # Independent: the length-aware prediction vs the old count x average rule.
    length_aware = _O + _R * 3 * long_chars
    count_times_average = 3 * ((_O + _R * 20) + (_O + _R * 40)) / 2
    assert count_times_average <= 4.0, "R6's rule would keep the paragraphs"
    assert (length_aware <= 4.0) is per_paragraph

    result = run_semantic_recall(store, tmp_path, paste)

    assert result is not None
    # S195: when the paragraphs do not fit, paragraphs 1 and 3 (the long middle one
    # skipped) are searched, still from the paste's own length: they fit here.
    expected = [[p1, long_p, p3]] if per_paragraph else [[p1, p3]]
    assert embedder.batches == expected
    assert result.whole_message_fallback is False


# ---------------------------------------------------------------------------
# C17a / C17c: the sample and the rerank budget.
# ---------------------------------------------------------------------------


def test_c17a_c17c_one_sample_per_batch_and_the_rerank_budget(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _warm_embed_fit(store)
    p1, p2 = "alpha words here", "beta words here and more"
    result = run_semantic_recall(store, tmp_path, f"{p1}\n{p2}")
    assert result is not None and embedder.batches == [[p1, p2]]
    state = sr._embed_cost.state(store, _EMBEDDER_ID)
    assert state.sums.n == 3 and state.sums.sum_x == 20 + 40 + 2 * len(p2), (
        "one sample, x = texts x the longest text's tokens"
    )
    fit = sr._embed_cost.fit(store, _EMBEDDER_ID)
    assert fit.overhead == pytest.approx(_O) and fit.rate == pytest.approx(_R)
    t_m = sr._message_overhead.mean(store, "")
    scan = sr._scan_time.mean(store, "")
    expected = 4.0 - t_m - (_O + _R * 2 * len(p2)) - 2 * scan
    assert result.rerank_budget == pytest.approx(expected)


# ---------------------------------------------------------------------------
# C17d: token counts unknown (and it wins over the first-message rule).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warm", [True, False])
def test_c17d_unknown_lengths_search_one_query_and_record_no_sample(
    monkeypatch, tmp_path: Path, warm: bool
) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock, lengths=False)
    store = _store(monkeypatch, tmp_path, embedder)
    if warm:
        _warm_embed_fit(store)
    three = "alpha one words\nbeta two words\ngamma three words"
    result = run_semantic_recall(store, tmp_path, three)
    assert result is not None
    assert embedder.batches == [[three]], "one query; the first-message rule needs lengths"
    assert result.whole_message_fallback is True
    n_before = 2 if warm else 0
    state = sr._embed_cost.state(store, _EMBEDDER_ID)
    assert (state.sums.n if state else 0) == n_before, "no sample without token counts"
    measured = _O + _R * len(three)
    t_m = sr._message_overhead.mean(store, "")
    assert result.rerank_budget == pytest.approx(4.0 - t_m - measured - sr._scan_time.mean(store, ""))


# ---------------------------------------------------------------------------
# C17f / S115: the first-message rule stays literal.
# ---------------------------------------------------------------------------


def test_c17f_the_first_message_rule_is_count_times_the_first_time(monkeypatch, tmp_path: Path) -> None:
    """First paragraph 10 tokens costs 1.0 s; 3 x 1.0 fits 4 s (literal), while
    the fit from that one sample (0.1 s per token) predicts 3 x 100 tokens =
    30 s for the long paragraph: the literal rule keeps the paragraphs."""
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)

    class _Flat(_CostEmbedder):
        def embed_batch(self, texts):
            self._use()
            self.batches.append(list(texts))
            self.clock.advance(1.0 * len(texts))
            return [_AxisEmbedder.embed(self, t) for t in texts]

    embedder = _Flat(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    p1, p2, p3 = "ten chars!", "x" * 100, "short three"
    result = run_semantic_recall(store, tmp_path, f"{p1}\n{p2}\n{p3}")
    assert result is not None
    assert embedder.batches == [[p1], [p2, p3]]
    assert result.whole_message_fallback is False and len(result.paragraphs) == 3


# ---------------------------------------------------------------------------
# Cost-series increment: the three paragraph-time series persist (CS2, CS3).
# ---------------------------------------------------------------------------

_THREE_PARAGRAPHS = ("alpha one words", "beta two words", "gamma three words")


def _restart(store) -> None:
    """A new process on the same database: the buffer saved, then every
    in-process piece of state gone."""
    from brain.memory import cost_series

    cost_series.flush(store)
    cost_series._reset_pending()
    sr._reset_paragraph_time_model()


def test_cs2_cs3_a_restart_does_not_take_the_first_message_rule_and_another_machine_does(
    monkeypatch, tmp_path: Path
) -> None:
    """S115's first-message rule (embed the first paragraph alone, then decide)
    is for a first startup. After a restart the saved embed fit and the saved
    `T_s` / `T_m` decide, in one batch; on another machine (a copied database)
    the series are empty again and the first-message rule returns."""
    from brain.memory import cost_series

    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    p1, p2, p3 = _THREE_PARAGRAPHS
    message = "\n".join(_THREE_PARAGRAPHS)

    run_semantic_recall(store, tmp_path, message)
    assert embedder.batches == [[p1], [p2, p3]], "first startup: the S115 rule"

    _restart(store)
    embedder.batches.clear()
    run_semantic_recall(store, tmp_path, message)
    assert embedder.batches == [[p1, p2, p3]], "after a restart: the saved fit decides"

    _restart(store)
    embedder.batches.clear()
    monkeypatch.setattr(cost_series, "_machine_override", "another-machine")
    run_semantic_recall(store, tmp_path, message)
    assert embedder.batches == [[p1], [p2, p3]], "another machine: a first startup there"


def test_cs3b_after_a_downtime_the_carried_estimates_decide_for_the_three_series(
    monkeypatch, tmp_path: Path
) -> None:
    from brain.memory import cost_series

    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    message = "\n".join(_THREE_PARAGRAPHS)
    run_semantic_recall(store, tmp_path, message)
    _restart(store)
    real = cost_series._wall_clock
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + cost_series.window_seconds() + 60.0)
    for series, key in ((sr._embed_cost, _EMBEDDER_ID), (sr._message_overhead, ""), (sr._scan_time, "")):
        state = series.state(store, key)
        assert state is not None and state.sums.n == 0 and state.carried is not None, series.name
    embedder.batches.clear()
    run_semantic_recall(store, tmp_path, message)
    assert embedder.batches == [list(_THREE_PARAGRAPHS)], "the carried estimates, not the first-message rule"


def test_cs1_the_three_series_are_not_reset_by_anything_but_the_window(
    monkeypatch, tmp_path: Path
) -> None:
    """S146: no tick: a sample is counted for exactly 24 hours."""
    from brain.memory import cost_series

    clock = _clocked(monkeypatch)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    sr._scan_time.record(store, "", 1, 0.5)
    sr._message_overhead.record(store, "", 1, 0.25)
    wall[0] += cost_series.window_seconds() - 1
    assert sr._scan_time.mean(store, "") == pytest.approx(0.5)
    assert sr._message_overhead.mean(store, "") == pytest.approx(0.25)
    sr._reset_paragraph_time_model()  # process state only: the series are the database's
    assert sr._scan_time.mean(store, "") == pytest.approx(0.5)
    wall[0] += 1
    state = sr._scan_time.state(store, "")
    assert state is not None and state.sums.n == 0


# ---------------------------------------------------------------------------
# C18: the load and the first embed are in no sample (S109, S117).
# ---------------------------------------------------------------------------


def test_c18_load_and_first_embed_are_kept_out_of_every_sample(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock, load=10.0, first=5.0)
    store = _store(monkeypatch, tmp_path, embedder)
    text = "kite over the harbour"

    run_semantic_recall(store, tmp_path, text)
    run_semantic_recall(store, tmp_path, text)

    assert embedder.single_embeds == [text], "one discarded warm-up embed per model id per process"
    state = sr._embed_cost.state(store, _EMBEDDER_ID)
    assert state.sums.n == 2
    assert state.sums.sum_y == pytest.approx(2 * (_O + _R * len(text))), "no load, no first inference"
    assert sr._message_overhead.mean(store, "") == pytest.approx(0.01), "the token probe only"
    assert sr._scan_time.mean(store, "") == pytest.approx(0.0)


def test_c18_a_failed_warm_up_is_retried_and_does_not_stop_recall(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    fail = {"on": True}
    real_embed = embedder.embed

    def _embed(text):
        if fail["on"]:
            raise RuntimeError("warm-up failed")
        return real_embed(text)

    monkeypatch.setattr(embedder, "embed", _embed)
    assert run_semantic_recall(store, tmp_path, "kite over the harbour") is not None, (
        "a failed warm-up does not stop recall (review F2)"
    )
    assert sr._embed_warmed == set(), "not marked warm after a failure"
    fail["on"] = False
    assert run_semantic_recall(store, tmp_path, "kite over the harbour") is not None
    assert embedder.single_embeds == ["kite over the harbour"], "retried once, on the next message"
    assert sr._embed_warmed == {_EMBEDDER_ID}


def test_c16_the_paragraph_embed_series_keeps_its_fit_when_the_window_empties_and_rescales(
    monkeypatch, tmp_path: Path
) -> None:
    """S146/S114 on the real series: the saved fit stands when every sample has
    left the window, and one new size rescales it."""
    from brain.memory import cost_series

    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    sr._embed_cost.record(store, _EMBEDDER_ID, 20, _O + _R * 20)
    sr._embed_cost.record(store, _EMBEDDER_ID, 40, _O + _R * 40)
    carried = sr._embed_cost.fit(store, _EMBEDDER_ID)
    assert cost_series.flush(store) == 2
    real = cost_series._wall_clock
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + cost_series.window_seconds() + 60.0)
    state = sr._embed_cost.state(store, _EMBEDDER_ID)
    assert state.sums.n == 0 and state.carried is not None and state.carried.fit == carried
    sr._embed_cost.record(store, _EMBEDDER_ID, 30, 2 * (_O + _R * 30))
    fit = sr._embed_cost.fit(store, _EMBEDDER_ID)
    assert fit.predict(30) == pytest.approx(2 * (_O + _R * 30))
    assert fit.rate == pytest.approx(2 * _R) and fit.overhead == pytest.approx(2 * _O)


def test_f2_a_failing_ensure_loaded_does_not_disable_semantic_recall(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)

    def _broken():
        raise AttributeError("fastembed internals renamed")

    monkeypatch.setattr(embedder, "ensure_loaded", _broken)
    assert run_semantic_recall(store, tmp_path, "kite over the harbour") is not None


# ---------------------------------------------------------------------------
# C20 (i): lock waits are not timed as work.
# ---------------------------------------------------------------------------


def test_c20_waits_before_the_lock_are_not_timed(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 40.0)
    embedder = _CostEmbedder(clock, wait=7.0, in_lock=True)
    store = MemoryStore(tmp_path / "memories.db")
    mems = [_mem(store, f"rc memory {i}", {0: 0.9 - 0.01 * i}) for i in range(6)]

    class _WaitingReranker(_Reranker):
        def pair_token_lengths_timed(self, query, documents):
            clock.advance(7.0)  # waiting behind the re-measure job's rerank
            before = clock.now
            clock.advance(0.125)
            return self.pair_token_lengths(query, documents), clock.now - before

    rec = _WaitingReranker({m.content: 5.0 for m in mems})
    embedder.axes = {"kite over the harbour": 0}
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.5)
    _warm(store)

    run_semantic_recall(store, tmp_path, "kite over the harbour")

    assert sr._message_overhead.mean(store, "") == pytest.approx(0.01), "token probe: in-lock only"
    assert sr._embed_cost.state(store, _EMBEDDER_ID).sums.sum_y == pytest.approx(
        _O + _R * len("kite over the harbour")
    ), "embed: in-lock only"
    assert sr._scan_time.mean(store, "") == pytest.approx(0.125), "pair-size probe: in-lock only"


# ---------------------------------------------------------------------------
# C20 (ii): the real providers read the clock inside their lock.
# ---------------------------------------------------------------------------


class _StepClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _held(lock: threading.Lock, clock: _StepClock, call):
    """Hold `lock`, start `call` on a thread (it blocks), advance the clock by
    5 s of waiting, release, and return the call's result."""
    out: list = []
    lock.acquire()
    t = threading.Thread(target=lambda: out.append(call()))
    t.start()
    t.join(0.2)
    clock.now += 5.0
    lock.release()
    t.join(5.0)
    return out[0]


def test_c20_cross_encoder_pair_probe_times_inside_the_lock(monkeypatch) -> None:
    clock = _StepClock()
    monkeypatch.setattr(reranker_mod, "_clock", clock)
    provider = object.__new__(CrossEncoderProvider)
    provider._rerank_lock = threading.Lock()

    def encode_batch(pairs):
        clock.now += 0.25
        return [SimpleNamespace(attention_mask=[1] * 3) for _ in pairs]

    tokenizer = SimpleNamespace(encode_batch=encode_batch)
    provider._model = SimpleNamespace(model=SimpleNamespace(model=object(), tokenizer=tokenizer))
    lengths, seconds = _held(
        provider._rerank_lock, clock, lambda: provider.pair_token_lengths_timed("q", ["a", "b"])
    )
    assert lengths == [3, 3] and seconds == pytest.approx(0.25)


def _fastembed_stub(clock: _StepClock) -> FastEmbedProvider:
    provider = object.__new__(FastEmbedProvider)
    provider._embed_lock = threading.Lock()
    provider._real_dim = None
    provider._declared_dim = None
    provider._model_id = "stub"

    def embed(batch):
        for _ in batch:  # a lazy generator, as fastembed's
            clock.now += 0.25
            yield np.ones(4, dtype=np.float32)

    def tokenize(texts):
        clock.now += 0.125
        return [SimpleNamespace(attention_mask=[1] * len(t) + [0] * 2) for t in texts]

    inner = SimpleNamespace(model=object(), tokenize=tokenize, load_onnx_model=lambda: None)
    provider._model = SimpleNamespace(embed=embed, model=inner)
    return provider


def test_c20_fastembed_embed_and_token_probe_time_inside_the_lock(monkeypatch) -> None:
    clock = _StepClock()
    monkeypatch.setattr(embeddings_mod, "_clock", clock)
    provider = _fastembed_stub(clock)
    vectors, seconds = _held(provider._embed_lock, clock, lambda: provider.embed_batch_timed(["a", "b"]))
    assert len(vectors) == 2 and seconds == pytest.approx(0.5), "generator consumed inside the timing"
    lengths, seconds = _held(
        provider._embed_lock, clock, lambda: provider.token_lengths_timed(["abc", "de"])
    )
    assert lengths == [3, 2], "padding not counted"
    assert seconds == pytest.approx(0.125)


def test_c18_fastembed_ensure_loaded_loads_once() -> None:
    provider = _fastembed_stub(_StepClock())
    loads: list[int] = []
    inner = provider._model.model
    inner.model = None

    def load():
        loads.append(1)
        inner.model = object()

    inner.load_onnx_model = load
    provider.ensure_loaded()
    session = inner.model
    provider.ensure_loaded()
    assert loads == [1] and inner.model is session


# ---------------------------------------------------------------------------
# C19 / S110: the last paragraph served gets a partial share when >= 5 fit.
# ---------------------------------------------------------------------------


def test_c19_allocation_gives_the_last_paragraph_a_partial_share() -> None:
    assert allocate_widths([(9, 0.9 - 0.01 * i) for i in range(6)]) == [9, 9, 9, 9, 9, 5]
    assert allocate_widths([(8, 0.9 - 0.01 * i) for i in range(7)]) == [8, 8, 8, 8, 8, 8, 0]


def test_c19_end_to_end_the_sixth_paragraph_reranks_five(monkeypatch, tmp_path: Path) -> None:
    bests = [0.61, 0.95, 0.70, 0.88, 0.52, 0.79]
    texts = [f"paragraph {chr(97 + i)} words" for i in range(6)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = []
    for i, best in enumerate(bests):
        for j in range(9):
            contents.append(_mem(store, f"rr para {i} memory {j}", {i: best - 0.001 * j}).content)
    rec = _Reranker(dict.fromkeys(contents, 5.0))
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    _warm(store)

    search = search_paragraphs(store, "\n".join(texts), log_calibration=False)

    assert search is not None
    order = sorted(range(6), key=lambda i: -bests[i])
    widths = [o.gated.real_width for o in search.outcomes]
    expected = [0] * 6
    for rank, i in enumerate(order):
        expected[i] = 9 if rank < 5 else 5
    assert widths == expected
    assert search.total_width == 50


# ---------------------------------------------------------------------------
# C15b end to end: a small paragraph share on a healthy model is not stuck.
# ---------------------------------------------------------------------------


def test_c15b_a_many_paragraph_message_on_a_healthy_model_leaves_no_stuck_state(
    monkeypatch, tmp_path: Path
) -> None:
    """Four paragraphs; each share (1 s) is below the 5 + 2 cost (1.2 s), the
    full budget (4 s) fits it: each paragraph takes the cosine path for lack of
    budget and is recorded, but nothing is stuck and the job is not due."""
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    texts = [f"paragraph {chr(97 + i)} words" for i in range(4)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    mems = [_mem(store, f"rs para {i} memory {j}", {i: 0.9 - 0.001 * j}) for i in range(4) for j in range(6)]
    rec = _Reranker({m.content: 5.0 for m in mems})
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _RERANKER_ID)
    monkeypatch.setitem(reranker_mod._provider_cache, _RERANKER_ID, rec)
    with reranker_mod._cost_lock:
        reranker_mod._warmed_model_ids.add(_RERANKER_ID)
    # A healthy estimate: a 7-document batch of these pairs costs 1.2 s.
    pair = len(texts[0]) + len("rs para 0 memory 0")
    longest = max(pair, *(len(texts[0]) + len(a) for a in reranker_mod.ANCHOR_POOL[:2]))
    for x in (7 * longest, 14 * longest):
        reranker_mod._record_rerank_cost(_RERANKER_ID, store, x, 1.2 * x / (7 * longest), None)
    assert clock.now == 0.0

    search = search_paragraphs(store, "\n".join(texts), log_calibration=False)

    assert search is not None and search.total_width == 0
    assert _RERANKER_ID in reranker_mod._budget_misses, "recorded (S113)"
    assert reranker_mod.stuck_width_due(1.0, lambda: nullcontext(store)) is False, "5 fit the full budget: not stuck"


def test_f2_a_broken_load_warns_once_then_logs_at_debug(monkeypatch, tmp_path: Path, caplog) -> None:
    """Round-2 review N2: a load that fails on every message (renamed fastembed
    internals) warns with a traceback once per model id, then at debug."""
    import logging

    clock = _clocked(monkeypatch)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)

    def _broken():
        raise AttributeError("fastembed internals renamed")

    monkeypatch.setattr(embedder, "ensure_loaded", _broken)
    with caplog.at_level(logging.DEBUG, logger="brain.memory.semantic_recall"):
        for _ in range(3):
            run_semantic_recall(store, tmp_path, "kite over the harbour")
    failures = [r for r in caplog.records if "load/warm-up failed" in r.getMessage()]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG, logging.DEBUG]


# ---------------------------------------------------------------------------
# S202: a single paragraph too long to embed in the time left is cut like the whole message
# ---------------------------------------------------------------------------

_ONE_PARAGRAPH = ("lorem ipsum " * 200)[:1000]  # no line break: one paragraph; 1 token per character here


def test_s202_a_single_paragraph_that_does_not_fit_is_cut_to_the_longest_prefix_that_does(
    monkeypatch, tmp_path: Path
) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 1.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _warm_embed_fit(store)
    sr._message_overhead.record(store, "", 1, 0.0)
    assert _O + _R * len(_ONE_PARAGRAPH) > 1.0, "embedding all of it would not fit (not vacuous)"

    result = run_semantic_recall(store, tmp_path, _ONE_PARAGRAPH)

    assert result is not None
    (batch,) = embedder.batches
    assert len(batch) == 1 and _ONE_PARAGRAPH.startswith(batch[0]) and batch[0] != _ONE_PARAGRAPH
    cap = int((1.0 - _O) / _R)  # independent: 450 tokens fit the 1.0 s budget after the 0.1 s overhead
    assert cap - 8 <= len(batch[0]) <= cap, "the longest word-boundary prefix within the cap"
    assert _O + _R * len(batch[0]) <= 1.0 + 1e-9


def test_s202_a_single_paragraph_that_fits_is_embedded_whole(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 4.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _warm_embed_fit(store)
    sr._message_overhead.record(store, "", 1, 0.0)
    run_semantic_recall(store, tmp_path, _ONE_PARAGRAPH)
    assert embedder.batches == [[_ONE_PARAGRAPH]]


def test_s202_with_no_embed_fit_yet_a_single_paragraph_is_embedded_whole_as_built(
    monkeypatch, tmp_path: Path
) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 1.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)  # no embed samples: the first message of a process
    run_semantic_recall(store, tmp_path, _ONE_PARAGRAPH)
    assert embedder.batches == [[_ONE_PARAGRAPH]]


def test_s202_when_nothing_fits_the_single_paragraph_is_embedded_as_it_is(monkeypatch, tmp_path: Path) -> None:
    """S61: the budget cannot hold even the embed's fixed cost: no cut can help, so the paragraph is what a
    one-paragraph message has always been."""
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 0.05)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _warm_embed_fit(store)
    sr._message_overhead.record(store, "", 1, 0.0)
    run_semantic_recall(store, tmp_path, _ONE_PARAGRAPH)
    assert embedder.batches == [[_ONE_PARAGRAPH]]


def _fit_only(store, *, overhead: float | None, scan: float | None) -> None:
    for n in (20, 40):
        sr._embed_cost.record(store, _EMBEDDER_ID, n, _O + _R * n)
    for _ in range(20):  # many samples, so the run's own sample barely moves the mean
        if overhead is not None:
            sr._message_overhead.record(store, "", 1, overhead)
        if scan is not None:
            sr._scan_time.record(store, "", 1, scan)


_FOUR_HUNDRED = ("lorem ipsum " * 40)[:400]  # one paragraph, 400 tokens here: embed alone 0.1 + 0.8 = 0.9 s


def test_s202_the_message_overhead_counts_against_the_room_for_the_embed(monkeypatch, tmp_path: Path) -> None:
    """Able to fail: the embed alone (0.9 s) fits the 1.0 s budget; only with the 0.3 s setup does it not."""
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 1.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _fit_only(store, overhead=0.3, scan=0.0)
    run_semantic_recall(store, tmp_path, _FOUR_HUNDRED)
    (batch,) = embedder.batches
    assert len(batch[0]) < len(_FOUR_HUNDRED), "cut: the setup leaves less room"
    setup = sr._message_overhead.mean(store, "")  # the value the decision used (the run's own sample is in it)
    assert 0.25 < setup <= 0.3
    cap = int((1.0 - setup - _O) / _R)
    assert cap - 8 <= len(batch[0]) <= cap


def test_s202_the_scan_time_counts_against_the_room_for_the_embed(monkeypatch, tmp_path: Path) -> None:
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 1.0)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _fit_only(store, overhead=0.0, scan=0.3)
    run_semantic_recall(store, tmp_path, _FOUR_HUNDRED)
    (batch,) = embedder.batches
    assert len(batch[0]) < len(_FOUR_HUNDRED), "cut: the scan leaves less room"
    cap = int((1.0 - 0.3 - _O) / _R)
    assert cap - 8 <= len(batch[0]) <= cap


def test_s202_with_no_scan_sample_yet_a_single_paragraph_is_embedded_whole(monkeypatch, tmp_path: Path) -> None:
    """The first message of a process: an embed fit exists (restored) but no scan time: nothing to
    predict the room from, so the paragraph is embedded as it is (as built)."""
    clock = _clocked(monkeypatch)
    _set_budget(monkeypatch, 0.5)
    embedder = _CostEmbedder(clock)
    store = _store(monkeypatch, tmp_path, embedder)
    _fit_only(store, overhead=0.0, scan=None)
    run_semantic_recall(store, tmp_path, _FOUR_HUNDRED)
    assert embedder.batches == [[_FOUR_HUNDRED]]
