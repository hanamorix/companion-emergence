"""Name-recall fix R1: the per-message rerank width (spec §1; criteria C1a,
C1c, CONC-1, C2c part 1, INV-I13a, ADV-9 in
`changes/name-recall-fix/1.5-criteria.md`), on the cost-series increment's
saved 24-hour window (S146, S158, S164; criteria CS1, CS2, CS3, CS15).

The hourly width sample (one query's top-5 documents, cached ~1 h,
diagnosis H8) is replaced by a cost model learned from every recall-time
rerank: seconds = overhead + rate * (pairs * longest pair), the size in
reranker tokens (S75), both fitted by
least squares over the last 24 hours' samples (saved in memories.db), per
reranker model id; each message fits its own width to its own candidates.

All offline and deterministic: timing comes from a scripted clock injected
through `reranker._clock`, advanced by a fake reranker whose per-call cost is
exactly `overhead + rate * len(documents) * longest pair` (in tokens; the
fake providers' stand-in for a token is one character); the per-call peak growth
comes from a scripted `peak_rss`. No model, no network.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import brain.memory.reranker as reranker_mod
from brain.dev_constants import RERANK_MIN_REAL_CANDIDATES
from brain.memory import cost_series, ram_fit
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.reranker import (
    ANCHOR_POOL,
    CrossEncoderProvider,
    FakeRerankerProvider,
    RerankCostEstimate,
    fit_rerank_width,
    surviving_pair_token_lengths,
)
from brain.memory.store import MemoryStore

_MODEL = "cost-scripted"
_QUERY = "q"

# The store this test's recall calls run against (an autouse fixture below
# opens one per test); the module-level helpers add it to every call.
_STORE: MemoryStore


@pytest.fixture(autouse=True)
def _store(tmp_path: Path):
    global _STORE
    _STORE = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    yield _STORE
    _STORE.close()


def rerank_for_recall(provider, query, documents, **kwargs):
    return reranker_mod.rerank_for_recall(provider, query, documents, store=_STORE, **kwargs)


def rerank_cost_estimate(model_id):
    return reranker_mod.rerank_cost_estimate(model_id, _STORE)


def _window(model_id: str = _MODEL) -> cost_series.LinearSums:
    """The window's sums for the rerank series (database + this process's buffer)."""
    state = reranker_mod._rerank_cost.state(_STORE, model_id)
    assert state is not None
    return state.sums


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _CostScriptedProvider(FakeRerankerProvider):
    """A fake reranker whose every `rerank()` call advances the injected clock
    by exactly `overhead + rate * len(documents) * longest pair` (the S67
    model, so padding to the longest pair is part of the fixture). Records
    the documents of every call."""

    def __init__(self, clock: _Clock, overhead: float, rate: float, model: str = _MODEL) -> None:
        super().__init__(default=0.0)
        self.clock = clock
        self.overhead = overhead
        self.rate = rate
        self._model = model
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        longest = max(self.pair_token_lengths(query, documents))
        self.clock.t += self.overhead + self.rate * len(documents) * longest
        return super().rerank(query, documents)

    def model_id(self) -> str:
        return self._model


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(reranker_mod, "_clock", c)
    # Deterministic default: no peak-growth signal (the suite's autouse fixture switches the
    # reader off), so the RAM term is skipped unless a test scripts the peak itself.
    return c


def _docs(n: int, pair_tokens: int, tag: str = "d") -> list[str]:
    """`n` distinct documents whose (query `_QUERY`, doc) pair is exactly
    `pair_tokens` tokens (the fake providers count one token per character)."""
    body = pair_tokens - len(_QUERY)
    return [(f"{tag}{i:03d}" + "x" * body)[:body] for i in range(n)]


def _anchor_pair_tokens(query: str = _QUERY) -> list[int]:
    return [len(query) + len(a) for a in ANCHOR_POOL]


def _reference_width(
    cand: list[int],
    anchors: list[int],
    overhead: float,
    rate: float,
    budget: float,
    *,
    memory: ram_fit.RamFit | None = None,
    headroom: float | None = None,
    cap: int = CANDIDATE_POOL,
) -> int:
    """Independent recomputation of the width (spec §1 / S67 / S23): the
    LARGEST n whose batch of n real + min(8, n // 2) anchors, padded to its
    longest pair, is predicted to fit the budget, and whose predicted peak
    growth (the memory model `a + b*T + c*B*L^2`, recomputed here from its
    coefficients, clamped at 0) fits the headroom. Scans every n (no early stop), so it
    does not lean on the implementation's monotonicity argument."""
    best = 0
    for n in range(1, min(len(cand), cap) + 1):
        k = min(8, n // 2)
        longest = max([*cand[:n], *anchors[:k]])
        padded = (n + k) * longest
        if overhead + rate * padded > budget:
            continue
        if memory is not None and headroom is not None:
            predicted = max(0.0, memory.a + memory.b * padded + memory.c * (n + k) * longest * longest)
            if predicted > headroom:
                continue
        best = n
    return best


def _sum_model_width(cand: list[int], anchors: list[int], rate: float, budget: float) -> int:
    """The OLD model's shape (a plain per-token sum, no padding, no
    overhead) — used only to show the padding fixture discriminates."""
    best = 0
    for n in range(1, len(cand) + 1):
        k = min(8, n // 2)
        if rate * (sum(cand[:n]) + sum(anchors[:k])) <= budget:
            best = n
    return best


# ---------------------------------------------------------------------------
# C1a (i): before any measurement the width is 5 real + 2 anchors, after
# exactly two discarded warm-up reranks; the fit inputs are the measured
# call only.
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason='1.6 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_c1a_i_first_rerank_is_five_real_plus_two_anchors_after_two_discarded_warmups(
    clock: _Clock,
) -> None:
    provider = _CostScriptedProvider(clock, overhead=0.2, rate=2e-4)
    docs = _docs(50, 1200)

    out = rerank_for_recall(provider, _QUERY, docs, budget_seconds=4.0)

    assert out.reranked and out.width == RERANK_MIN_REAL_CANDIDATES == 5
    assert [len(c) for c in provider.calls] == [1, 1, 5 + 2], (
        "two single-document warm-ups, then ONE combined call of 5 real + 2 anchors"
    )
    assert provider.calls[2][:5] == docs[:5]
    assert provider.calls[2][5:] == ANCHOR_POOL[:2]

    # Fit inputs == the measured call only (recomputed here): x = 7 * 1200.
    x = 7 * 1200
    y = 0.2 + 2e-4 * x
    sums = _window()
    assert (sums.n, sums.sum_x, sums.sum_xx) == (1, x, x * x)
    assert sums.sum_y == pytest.approx(y)
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.measured_batches == 1
    # One batch: overhead 0, rate = ratio of sums.
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_token == pytest.approx(y / x)


def test_c1a_i_warmups_run_once_per_model_id(clock: _Clock) -> None:
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-6)
    rerank_for_recall(provider, _QUERY, _docs(20, 100), budget_seconds=4.0)
    rerank_for_recall(provider, _QUERY, _docs(20, 100), budget_seconds=4.0)
    single_doc_calls = [c for c in provider.calls if len(c) == 1]
    assert len(single_doc_calls) == reranker_mod._WARMUP_RERANKS == 2

    other = _CostScriptedProvider(clock, overhead=0.0, rate=1e-6, model="another-model")
    rerank_for_recall(other, _QUERY, _docs(20, 100), budget_seconds=4.0)
    assert len([c for c in other.calls if len(c) == 1]) == 2, "keyed by reranker model id"
    assert rerank_cost_estimate("another-model").measured_batches == 1
    assert rerank_cost_estimate(_MODEL).measured_batches == 2


# ---------------------------------------------------------------------------
# C1a (ii): least squares over running sums recovers overhead and rate;
# degenerate fits fall back to the through-origin ratio.
# ---------------------------------------------------------------------------


def test_c1a_ii_two_padded_sizes_recover_overhead_and_rate(clock: _Clock) -> None:
    o0, r0 = 0.2, 2e-4
    provider = _CostScriptedProvider(clock, overhead=o0, rate=r0)
    rerank_for_recall(provider, _QUERY, _docs(50, 1200, "L"), budget_seconds=4.0)  # M_long
    rerank_for_recall(provider, _QUERY, _docs(50, 150, "S"), budget_seconds=4.0)  # M_short

    sums = _window()
    assert sums.n == 2
    # Independent ordinary least squares over the measured (x, y) pairs.
    measured = [c for c in provider.calls if len(c) > 1]
    xs = [len(c) * max(len(_QUERY) + len(d) for d in c) for c in measured]
    ys = [o0 + r0 * x for x in xs]
    assert len(set(xs)) == 2, "fixture precondition: two different padded sizes"
    slope, intercept = np.polyfit(xs, ys, 1)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == pytest.approx(intercept, rel=1e-9) == pytest.approx(o0, rel=1e-9)
    assert est.seconds_per_token == pytest.approx(slope, rel=1e-9) == pytest.approx(r0, rel=1e-9)
    # Able to fail: the plain ratio-of-sums model (no overhead term) is wrong here.
    assert sum(ys) / sum(xs) != pytest.approx(r0, rel=1e-3)


def test_c1a_ii_same_padded_size_is_degenerate_ratio_of_sums() -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 700, 1.0, None)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 700, 2.0, None)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_token == pytest.approx(3.0 / 1400)


def test_c1a_ii_negative_fitted_overhead_falls_back_to_ratio() -> None:
    # slope 0.002, intercept -0.1 -> negative overhead -> ratio of sums.
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.1, None)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 200, 0.3, None)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_token == pytest.approx(0.4 / 300)


def test_c1a_ii_non_positive_fitted_rate_falls_back_to_ratio() -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.3, None)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 200, 0.1, None)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_token == pytest.approx(0.4 / 300)


def test_no_estimate_before_any_measurement() -> None:
    assert rerank_cost_estimate(_MODEL) is None


# ---------------------------------------------------------------------------
# C1a (iii): the width is the longest prefix whose PADDED batch fits.
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason='1.6 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_c1a_iii_long_candidate_inside_short_prefix_stops_the_width(clock: _Clock) -> None:
    """Six short candidates, then one long one: padding the batch to the long
    pair makes the 7-candidate prefix cost 10 x 1000 tokens, although the
    plain sum of its pair lengths would have fit."""
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 10_000, 1.0, None)  # overhead 0, rate 1e-4
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    cand = [250] * 6 + [1000] + [250] * 5
    docs = [_docs(1, n, f"c{i}")[0] for i, n in enumerate(cand)]
    budget = 0.5

    out = rerank_for_recall(provider, _QUERY, docs, budget_seconds=budget)

    anchors = _anchor_pair_tokens()
    expected = _reference_width(cand, anchors, 0.0, 1e-4, budget)
    assert out.width == expected == 6
    assert _sum_model_width(cand, anchors, 1e-4, budget) > 6, (
        "fixture discriminates: the old sum-of-pair-lengths model would have taken the long candidate"
    )
    combined = [c for c in provider.calls if len(c) > 1]
    assert combined[-1] == docs[:6] + ANCHOR_POOL[:3]


_PER_TOKEN = ram_fit.RamFit(a=0.0, b=1_000.0, c=0.0, shapes=4, samples=4, max_pairs=2, max_length=1024)
_QUADRATIC = ram_fit.RamFit(a=-50.0, b=60.0, c=0.17, shapes=22, samples=22, max_pairs=7, max_length=1024)


@pytest.mark.parametrize(
    ("cand", "overhead", "rate", "budget", "memory", "headroom"),
    [
        ([120] * 50, 0.05, 1e-4, 4.0, None, None),
        ([900, 120, 300, 2000, 80, 80, 80, 1500] + [200] * 30, 0.3, 5e-5, 2.0, None, None),
        ([400] * 50, 0.0, 1e-5, 4.0, _PER_TOKEN, 2_000_000.0),
        ([400] * 50, 0.0, 1e-5, 4.0, _PER_TOKEN, 6_000_000.0),
        ([400] * 50, 0.0, 1e-5, 4.0, _PER_TOKEN, 1.0),
        ([400] * 50, 0.0, 1e-5, 4.0, _QUADRATIC, 3_000_000.0),
        ([60] * 12, 1.5, 1e-3, 4.0, None, None),
        ([3000] * 50, 0.1, 1e-4, 4.0, None, None),
        ([400] * 50, 0.0, 1e-5, 4.0, _PER_TOKEN, None),
    ],
)
def test_c1a_iii_fit_matches_the_independent_reference(
    cand: list[int],
    overhead: float,
    rate: float,
    budget: float,
    memory: ram_fit.RamFit | None,
    headroom: float | None,
) -> None:
    est = RerankCostEstimate(overhead, rate, memory, measured_batches=3)
    anchors = _anchor_pair_tokens()
    got = fit_rerank_width(cand, anchors, est, budget, headroom)
    assert got == _reference_width(
        cand, anchors, overhead, rate, budget, memory=memory, headroom=headroom
    )


@pytest.mark.slow(reason='5.1 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_fit_is_the_minimum_before_the_first_measurement() -> None:
    assert fit_rerank_width([5000] * 50, _anchor_pair_tokens(), None, 0.001, None) == 5
    assert fit_rerank_width([10] * 3, _anchor_pair_tokens(), None, 4.0, None) == 3


def test_fit_is_capped_at_fifty_real_and_at_max_real() -> None:
    est = RerankCostEstimate(0.0, 1e-12, None, measured_batches=2)
    assert fit_rerank_width([10] * 80, _anchor_pair_tokens(), est, 4.0, None) == CANDIDATE_POOL == 50
    assert fit_rerank_width([10] * 80, _anchor_pair_tokens(), est, 4.0, None, max_real=12) == 12


# ---------------------------------------------------------------------------
# C1a (iv): long -> short -> long -> short; each message's own width.
# ---------------------------------------------------------------------------


def test_c1a_iv_short_message_after_a_long_one_gets_a_wider_rerank(clock: _Clock) -> None:
    o0, r0, budget = 0.2, 2e-4, 4.0
    provider = _CostScriptedProvider(clock, overhead=o0, rate=r0)
    long_docs, short_docs = _docs(50, 1200, "L"), _docs(50, 150, "S")
    anchors = _anchor_pair_tokens()

    widths = []
    for docs, pair in ((long_docs, 1200), (short_docs, 150), (long_docs, 1200), (short_docs, 150)):
        est = rerank_cost_estimate(_MODEL)
        out = rerank_for_recall(provider, _QUERY, docs, budget_seconds=budget)
        if est is None:
            expected = 5
        else:
            expected = _reference_width(
                [pair] * 50, anchors, est.overhead_seconds, est.seconds_per_token, budget
            )
        assert out.width == expected
        widths.append(out.width)

    w_long_first, w_short_first, w_long, w_short = widths
    assert w_long_first == 5
    assert w_short > w_long, f"the short message must get a wider rerank: {widths}"
    assert w_long == 10 and w_short == 50  # exact S67 model recovered after two sizes


# ---------------------------------------------------------------------------
# C1a (v): the RAM headroom reader stays. (The hourly width machinery that was removed is listed in
# tests/unit/test_removed_names_stay_removed.py.)
# ---------------------------------------------------------------------------


def test_c1a_v_ram_headroom_reader_kept_for_judge_selftune() -> None:
    from brain.memory import judge_selftune  # imports _available_ram_headroom_bytes

    assert callable(reranker_mod._available_ram_headroom_bytes)
    assert judge_selftune is not None


# ---------------------------------------------------------------------------
# Calibration / bootstrap scoring never feeds the cost model (S24).
# ---------------------------------------------------------------------------


def test_bundled_pair_scoring_is_not_a_recall_measurement(clock: _Clock) -> None:
    provider = _CostScriptedProvider(clock, overhead=0.1, rate=1e-4)
    reranker_mod.normalize_bundled_pairs_against_anchors(provider, [("q1", "d1"), ("q2", "d2")])
    assert rerank_cost_estimate(_MODEL) is None


# ---------------------------------------------------------------------------
# C1c: pair lengths in reranker tokens, at their surviving (post-truncation)
# count (S62, S75).
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS = 4  # the real tokenizer's pair template: <s> q </s></s> d </s>
_ASCII_PER_TOKEN = 10  # the fake tokenizer: a run of up to 10 ASCII characters is one token


def _fake_segment_tokens(text: str) -> int:
    """The fake tokenizer's token count for one segment: ASCII text packs
    `_ASCII_PER_TOKEN` characters per token (Latin words), every other
    character (CJK, emoji) costs one token of its own, as with the shipped
    tokenizer's roughly 0.2 / 0.5 / 1.0 tokens per character."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return -(-ascii_chars // _ASCII_PER_TOKEN) + (len(text) - ascii_chars)


def _fake_encodings(pairs, max_tokens: int) -> list[SimpleNamespace]:
    """Pair encodings shaped like `tokenizers.Encoding` after `encode_batch`
    with truncation at `max_tokens` and padding to the batch's longest pair:
    `attention_mask` is 1 for every token the model runs (special tokens
    included) and 0 for padding. A pair over the maximum keeps its query and
    cuts the document (the query is short in these fixtures)."""
    kept = []
    for query, doc in pairs:
        doc_tokens = min(_fake_segment_tokens(doc), max_tokens - _SPECIAL_TOKENS - _fake_segment_tokens(query))
        kept.append(_SPECIAL_TOKENS + _fake_segment_tokens(query) + doc_tokens)
    longest = max(kept, default=0)
    return [
        SimpleNamespace(attention_mask=[1] * n + [0] * (longest - n), ids=[7] * longest) for n in kept
    ]


class _FakeTokenizer:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.batches: list[list[tuple[str, str]]] = []

    def encode_batch(self, pairs):
        self.batches.append(list(pairs))
        return _fake_encodings(self.batches[-1], self.max_tokens)


def test_c1c_truncated_pair_counts_at_the_model_maximum_and_padding_is_not_counted() -> None:
    query = "what is up"
    long_doc, short_doc = "y" * 5000, "z" * 300
    encodings = _fake_encodings([(query, long_doc), (query, short_doc)], max_tokens=100)
    assert len(encodings[1].attention_mask) == 100, "fixture: the short pair is padded to the long one"

    lengths = surviving_pair_token_lengths(encodings)

    assert lengths[0] == 100, "the cut pair counts at the model maximum (tokens surviving truncation)"
    assert lengths[1] == _SPECIAL_TOKENS + 1 + 30, "the short pair counts its own tokens, not its padding"


def _encoder_provider(tokenizer) -> CrossEncoderProvider:
    """A CrossEncoderProvider built without fastembed (no model files): its
    `_model.model` stands in for fastembed's loaded cross-encoder."""
    provider = object.__new__(CrossEncoderProvider)
    provider._model_id = "fake-encoder"
    provider._rerank_lock = threading.Lock()
    inner = SimpleNamespace(model=object(), tokenizer=tokenizer, loads=0)
    provider._model = SimpleNamespace(model=inner)
    return provider


def test_c1c_cross_encoder_provider_counts_its_tokenizers_tokens() -> None:
    tokenizer = _FakeTokenizer(max_tokens=100)
    provider = _encoder_provider(tokenizer)
    docs = ["y" * 5000, "z" * 300]

    lengths = provider.pair_token_lengths("what is up", docs)

    assert lengths == [100, _SPECIAL_TOKENS + 1 + 30], "tokens, not the 5,010 / 310 characters"
    assert tokenizer.batches == [[("what is up", d) for d in docs]], "one encode_batch over the pairs"


def test_c1c_tokenizer_loaded_on_first_use() -> None:
    tokenizer = _FakeTokenizer(max_tokens=100)
    provider = _encoder_provider(tokenizer)
    inner = provider._model.model
    inner.model = None

    def _load() -> None:
        inner.loads += 1
        inner.model = object()

    inner.load_onnx_model = _load
    provider.pair_token_lengths("q", ["abc"])
    provider.pair_token_lengths("q", ["abc"])
    assert inner.loads == 1


def test_c1c_tokenizer_failure_gives_no_lengths_never_characters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Boom:
        def encode_batch(self, pairs):
            raise RuntimeError("simulated tokenizer failure")

    provider = _encoder_provider(_Boom())
    with caplog.at_level(logging.WARNING, logger=reranker_mod.__name__):
        lengths = provider.pair_token_lengths("q", ["y" * 5000])
    assert lengths is None, "a character count must never reach the token-based cost model"
    assert any("token lengths unavailable" in r.message for r in caplog.records)


def test_c1c_no_token_lengths_means_no_rerank_and_no_recorded_sample(clock: _Clock) -> None:
    """A batch of unknown padded size must not run unbudgeted (hand-off
    "sizes": the no-rerank path serves the query), and no sample is
    recorded (a character count must never reach the token-based sums)."""

    class _NoLengths(_CostScriptedProvider):
        def pair_token_lengths(self, query, documents):
            return None

        def rerank(self, query, documents):
            self.calls.append(list(documents))
            return FakeRerankerProvider.rerank(self, query, documents)

    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 1.0, None)  # a prior token-unit sample
    before = _window()
    provider = _NoLengths(clock, overhead=0.0, rate=0.0)

    out = rerank_for_recall(provider, _QUERY, _docs(30, 400), budget_seconds=4.0)

    assert out.hand_off == "sizes" and not out.reranked and out.normalization is None
    assert out.measured is False
    assert [c for c in provider.calls if len(c) > 1] == [], "no combined rerank runs"
    assert _window() == before, "no sample without a known padded size"


def test_c1c_width_fit_and_measurement_use_the_surviving_lengths(clock: _Clock) -> None:
    class _Truncating(_CostScriptedProvider):
        def pair_token_lengths(self, query, documents):
            return [len(query) + min(len(d), 1200) for d in documents]

    provider = _Truncating(clock, overhead=0.0, rate=1e-5)
    rerank_for_recall(provider, _QUERY, ["w" * 5000] * 10, budget_seconds=4.0)

    sums = _window()
    assert sums.sum_x == 7 * (1 + 1200), "x uses the surviving pair length, not 7 * 5001"


class _TokenizingProvider(_CostScriptedProvider):
    """A scripted-clock provider that sizes pairs with the fake tokenizer
    (through the same `surviving_pair_token_lengths` the real provider
    uses), so a document's cost follows its TOKENS, as the real reranker's
    does."""

    def __init__(self, clock: _Clock, overhead: float, rate: float, max_tokens: int = 100_000) -> None:
        super().__init__(clock, overhead, rate)
        self.tokenizer = _FakeTokenizer(max_tokens)

    def pair_token_lengths(self, query, documents):
        return surviving_pair_token_lengths(self.tokenizer.encode_batch([(query, d) for d in documents]))


def test_s75_cjk_and_emoji_candidates_are_costed_by_tokens_not_characters(clock: _Clock) -> None:
    """The misestimate S75 fixes: equal characters, very different tokens.
    600 Latin characters are 60 tokens, 600 Chinese or emoji characters are
    600 tokens, ten times the padded size (and, at a per-token rate, the
    cost). A character-based model sizes all three alike."""
    latin = "a" * 600
    cjk = "\u5496" * 600
    emoji = "\U0001f600" * 600
    provider = _TokenizingProvider(clock, overhead=0.0, rate=1e-3)

    latin_n, cjk_n, emoji_n = (provider.pair_token_lengths(_QUERY, [d])[0] for d in (latin, cjk, emoji))
    assert len({len(latin), len(cjk), len(emoji)}) == 1, "fixture: the same character count"
    assert latin_n == _SPECIAL_TOKENS + 1 + 60
    assert cjk_n == emoji_n == _SPECIAL_TOKENS + 1 + 600

    # The fit: one budget, the same 12 candidates, Latin or CJK. By characters
    # the two pools are identical, so a character model gives one width for
    # both; by tokens the Latin pool fits whole and the CJK pool is cut.
    anchors = provider.pair_token_lengths(_QUERY, ANCHOR_POOL)
    est = RerankCostEstimate(0.0, 1e-3, None, measured_batches=2)
    budget = 5.0  # seconds
    width_latin = fit_rerank_width([latin_n] * 12, anchors, est, budget, None)
    width_cjk = fit_rerank_width([cjk_n] * 12, anchors, est, budget, None)
    assert width_latin == _reference_width([latin_n] * 12, anchors, 0.0, 1e-3, budget) == 12
    assert width_cjk == _reference_width([cjk_n] * 12, anchors, 0.0, 1e-3, budget)
    assert width_cjk < width_latin, "a CJK-heavy pool gets a narrower rerank than a Latin one"


@pytest.mark.slow(reason='5.4 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_s75_rerank_for_recall_sizes_the_width_in_tokens(clock: _Clock) -> None:
    """End to end through `rerank_for_recall` with a seeded estimate: the same
    ten candidates, equal in characters, get a wide rerank as Latin text and a
    narrow one as CJK text, because the width is sized from their tokens."""
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 1.0, None)  # 1 ms per padded token
    latin, cjk = "a" * 600, "\u5496" * 600
    budget = 2.0

    provider = _TokenizingProvider(clock, overhead=0.0, rate=1e-3)
    out_latin = rerank_for_recall(provider, _QUERY, [latin] * 30, budget_seconds=budget)
    provider = _TokenizingProvider(clock, overhead=0.0, rate=1e-3)
    out_cjk = rerank_for_recall(provider, _QUERY, [cjk] * 30, budget_seconds=budget)

    assert out_latin.reranked and out_latin.width > RERANK_MIN_REAL_CANDIDATES
    assert out_cjk.width < out_latin.width, "equal characters, ten times the tokens"


def test_s75_the_recorded_sample_is_in_tokens(clock: _Clock) -> None:
    provider = _TokenizingProvider(clock, overhead=0.0, rate=1e-3)
    cjk = "\u5496" * 600

    out = rerank_for_recall(provider, _QUERY, [cjk] * 10, budget_seconds=1_000.0)

    k = out.normalization.anchor_count
    anchors = provider.pair_token_lengths(_QUERY, ANCHOR_POOL)[:k]
    longest = max(_SPECIAL_TOKENS + 1 + 600, *anchors)
    assert _window().sum_x == (out.width + k) * longest
    assert longest == 605, "600 CJK characters are 600 tokens; a character count would record 601"


# ---------------------------------------------------------------------------
# C2c part 1: every rerank carries >= 5 real + >= 2 anchors and is normalized.
# ---------------------------------------------------------------------------


# Pools below the minimum return before any cost is read (so one seed rate each); from 5 up, the seed rate sets the
# width: 1.0 fits fewer than 5 (no rerank), 1e-3 a few more than 5, and (60, 1e-7) is the one case that hits both the
# 50-document cap and the 8-anchor cap.
_C2C_CASES = (
    [(0, None), (4, None)]
    + [(pool, rate) for pool in (5, 6, 9, 60) for rate in (None, 1e-3, 1.0)]
    + [(60, 1e-7)]
)


@pytest.mark.no_base_failure(reason='#540 rewrite of an existing test (new id from a rename or a new parameter) asserting what its predecessor asserted; the old code passes it')
@pytest.mark.parametrize(("pool", "seed_rate"), _C2C_CASES)
def test_c2c_every_rerank_has_five_real_and_two_anchors_and_is_normalized(
    clock: _Clock, pool: int, seed_rate: float | None
) -> None:
    if seed_rate is not None:
        reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, seed_rate * 1_000, None)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-6)

    out = rerank_for_recall(provider, _QUERY, _docs(pool, 400), budget_seconds=4.0)

    combined = [c for c in provider.calls if len(c) > 1]
    if pool < RERANK_MIN_REAL_CANDIDATES:
        assert out.hand_off == "pool" and not out.reranked
        assert provider.calls == [], "no warm-up and no rerank for a pool below the minimum"
        return
    if out.reranked:
        (sent,) = combined
        k = min(8, out.width // 2)
        assert out.width >= 5 and k >= 2
        assert sent[out.width :] == ANCHOR_POOL[:k] and len(sent) == out.width + k
        assert out.normalization.did_normalize is True
        assert out.normalization.real_width == out.width
    else:
        assert out.hand_off == "budget" and out.width < 5
        assert combined == [], "nothing is scored (so nothing can be gated raw) when fewer than 5 fit"


# ---------------------------------------------------------------------------
# INV-I13a: RSS unreadable (non-Linux) -> RAM term skipped, time-bound only;
# a readable RSS feeds the RAM bound.
# ---------------------------------------------------------------------------


def test_inv_i13a_an_unreadable_peak_skips_the_ram_term(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak reader is off (non-Linux, or a failed reset): no memory sample, no model, so the
    RAM term is skipped and the width is time-bound only."""
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)  # absurdly tight
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-6)
    rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_fit is None
    assert reranker_mod._ram_shapes.shape_groups(_STORE, _MODEL) == []

    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.width == 50, "no memory model -> width is time-bound only, headroom ignored"


def test_a_readable_peak_records_the_shape_of_every_measured_call(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """S220: the third `rerank_timed` value is the call's peak growth; the sample is keyed by the
    call's shape (pairs, longest pair), whatever the size relative to earlier calls."""
    growth = iter([4_000.0, 0.0, 9_000.0])
    monkeypatch.setattr(reranker_mod.peak_rss, "begin", lambda: 1.0)
    monkeypatch.setattr(reranker_mod.peak_rss, "growth_since", lambda before: next(growth))
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0, max_real=5)  # 7 pairs x 400
    rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0, max_real=5)  # again, same shape
    rerank_for_recall(provider, _QUERY, _docs(50, 200), budget_seconds=4.0, max_real=5)
    groups = reranker_mod._ram_shapes.shape_groups(_STORE, _MODEL)
    by_shape = {ram_fit.decode_shape(x)[:2]: (n, total) for x, n, total in groups}
    longest = lambda tokens: max(tokens, *_anchor_pair_tokens()[:2])  # noqa: E731
    assert by_shape == {(7, longest(400)): (2, 4_000.0), (7, longest(200)): (1, 9_000.0)}


def _seed_ram(
    model: str = _MODEL,
    bytes_per_token: float = 1_000.0,
    store: MemoryStore | None = None,
    *,
    fresh_process: bool = True,
) -> None:
    """A previous process measured four shapes (growth = bytes_per_token x padded tokens, the full-length
    one marked) and saved them; this process then starts with no in-memory state."""
    store = store if store is not None else _STORE
    for pairs, length, full in [(1, 1024, True), (1, 512, False), (1, 256, False), (2, 128, False)]:
        x = ram_fit.encode_shape(pairs, length, full=full)
        reranker_mod._ram_shapes.record(store, model, x, bytes_per_token * pairs * length)
    reranker_mod._ram_shapes.record(store, model, ram_fit.COMPLETE_MARKER, 0.0)  # a finished ladder pass
    assert cost_series.flush(store) >= 5
    if fresh_process:
        reranker_mod._reset_rerank_cost_model()
        cost_series._reset_pending()


def test_ram_bound_from_the_memory_model(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    """A usable model sizes the batch so its predicted growth (the WHOLE growth: nothing is
    already resident with the memory pool off) fits the live headroom."""
    _ram_figure_required(monkeypatch)
    _persist_estimate()
    _seed_ram(bytes_per_token=1_000.0)
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.ram_fit is not None and est.ram_fit_kept is False
    assert (est.ram_fit.a, est.ram_fit.b, est.ram_fit.c) == pytest.approx((0.0, 1_000.0, 0.0), abs=1e-6)

    # Room for 12 padded documents (8 real + 4 anchors) and not for 13 (9 real + 4).
    headroom = 1_000.0 * 12 * 400
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0)
    assert out.width == _reference_width(
        [400] * 50, _anchor_pair_tokens(), 0.0, est.seconds_per_token, 4.0, memory=est.ram_fit, headroom=headroom
    ) == 8


def test_negative_growth_is_recorded_as_zero(clock: _Clock) -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, -5_000.0, pairs=1, longest=1_000)
    ((x, n, total),) = reranker_mod._ram_shapes.shape_groups(_STORE, _MODEL)
    assert (ram_fit.decode_shape(x), n, total) == ((1, 1_000, False), 1, 0.0)


def test_a_growth_with_no_shape_or_an_unpackable_shape_is_not_recorded(caplog: pytest.LogCaptureFixture) -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, 5.0)  # no pairs / longest given
    assert reranker_mod._ram_shapes.shape_groups(_STORE, _MODEL) == []
    with caplog.at_level("WARNING"):
        reranker_mod._record_rerank_cost(_MODEL, _STORE, 9, 0.1, 5.0, pairs=1, longest=ram_fit.MAX_LENGTH + 1)
    assert reranker_mod._ram_shapes.shape_groups(_STORE, _MODEL) == []
    assert any("cannot be recorded" in r.getMessage() for r in caplog.records)
    assert _window().n == 2, "the time samples are recorded regardless"


def test_recorded_samples_from_many_threads_are_all_counted_once() -> None:
    """The buffer is shared by the reply thread, the search tool's worker and the lull job."""
    def worker(index: int) -> None:
        for i in range(50):
            reranker_mod._record_rerank_cost(
                _MODEL, _STORE, 100, 0.1, 10.0, pairs=1 + (index + i) % 3, longest=100
            )

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    groups = reranker_mod._ram_shapes.shape_groups(_STORE, _MODEL)
    assert sum(n for _x, n, _t in groups) == 400 and _window().n == 400


# ---------------------------------------------------------------------------
# The absorbing state that S100's re-measure job recovers from (its recovery
# is `test_rerank_recovery.py`, criterion C15f).
# ---------------------------------------------------------------------------


def test_inflated_first_sample_absorbs_every_message_without_the_remeasure_job(
    clock: _Clock,
) -> None:
    """One inflated measurement makes every later message hand off ("budget")
    with no rerank, so messages alone never take a new measurement and the
    estimate never moves; only the lull re-measure job (S100) or the next
    compaction period (S101/S114) changes it."""
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 1_000.0, None)  # 10 s per token
    before = _window()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)  # the host is actually fast

    for _ in range(5):
        out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
        assert out.hand_off == "budget" and not out.measured

    assert _window() == before
    assert [c for c in provider.calls if len(c) > 1] == []


# ---------------------------------------------------------------------------
# Round-2 additions (stage-6 findings): the provider's timing bracket, the
# normalization hand-off, the one-time fallback warning.
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason='3.9 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_cross_encoder_rerank_timed_excludes_time_waiting_for_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P-3: the time a recall spends WAITING for another thread's rerank
    (the provider lock) is not part of its measured cost. The clock advances
    10 s while the lock is held elsewhere, and the scoring itself takes
    0.5 s: the measurement must be 0.5 s."""
    clock = _Clock()
    monkeypatch.setattr(reranker_mod, "_clock", clock)

    class _Model:
        def rerank(self, query, documents):
            clock.t += 0.5
            return [0.0 for _ in documents]

    provider = object.__new__(CrossEncoderProvider)
    provider._model_id = "fake-encoder"
    provider._rerank_lock = threading.Lock()
    provider._model = _Model()
    result: list[tuple] = []

    provider._rerank_lock.acquire()
    worker = threading.Thread(target=lambda: result.append(provider.rerank_timed("q", ["a", "b"])))
    worker.start()
    worker.join(timeout=0.2)  # the worker is now blocked on the lock
    clock.t += 10.0  # another thread's rerank holding the lock
    provider._rerank_lock.release()
    worker.join(timeout=5.0)

    (scores, seconds, growth) = result[0]
    assert scores == [0.0, 0.0]
    assert seconds == pytest.approx(0.5), "lock wait must not be counted as this call's cost"
    assert growth is None, "the peak reader is off in the suite"


def test_normalization_hand_off_carries_no_scores(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the anchor median could not be taken, the raw scores never reach
    the caller (nothing can gate them): `normalization` is None."""

    def _raw(provider, query, real_documents):
        scores, seconds, rss = provider.rerank_timed(query, list(real_documents))
        return reranker_mod.AnchorNormalizationResult(
            scores=scores, real_width=len(scores), did_normalize=False, seconds=seconds, peak_growth_bytes=rss
        )

    monkeypatch.setattr(reranker_mod, "normalize_against_anchors", _raw)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-6)

    out = rerank_for_recall(provider, _QUERY, _docs(10, 100), budget_seconds=4.0)

    assert out.hand_off == "normalization" and not out.reranked
    assert out.normalization is None


def test_c1c_tokenizer_failure_warns_once_per_provider(caplog: pytest.LogCaptureFixture) -> None:
    class _Boom:
        def encode_batch(self, pairs):
            raise RuntimeError("simulated tokenizer failure")

    provider = _encoder_provider(_Boom())
    with caplog.at_level(logging.DEBUG, logger=reranker_mod.__name__):
        provider.pair_token_lengths("q", ["abc"])
        provider.pair_token_lengths("q", ["abc"])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "one warning (with traceback), later failures at debug level"


# ---------------------------------------------------------------------------
# Cost-series increment: the saved window (CS1-CS3 for the rerank series) and
# the per-restart RAM seed (CS15, S158).
# ---------------------------------------------------------------------------


def _persist_estimate(model: str = _MODEL) -> None:
    """A previous process measured two sizes (rate 1e-4 s per padded token, no
    overhead) and saved them; this process then starts with no in-memory state."""
    reranker_mod._rerank_cost.record(_STORE, model, 700, 0.07)
    reranker_mod._rerank_cost.record(_STORE, model, 1400, 0.14)
    assert cost_series.flush(_STORE) == 2
    reranker_mod._reset_rerank_cost_model()
    cost_series._reset_pending()


def test_cs2_a_restart_keeps_the_estimate_and_the_width_it_gives(clock: _Clock) -> None:
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    est_before = None
    _persist_estimate()
    est_before = rerank_cost_estimate(_MODEL)
    assert est_before is not None and est_before.seconds_per_token == pytest.approx(1e-4)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    # The estimate sizes this process's FIRST recall rerank too (RSS is
    # unreadable here, S158 does not apply): 100 tokens a pair, 1e-4 s a token.
    assert out.width == _reference_width([100] * 50, _anchor_pair_tokens(), 0.0, 1e-4, 4.0)
    assert out.width > RERANK_MIN_REAL_CANDIDATES


def test_cs3a_the_first_startup_on_a_machine_sizes_the_first_rerank_at_five(clock: _Clock) -> None:
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    assert rerank_cost_estimate(_MODEL) is None
    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.width == 5
    assert _window().n == 1, "its sample is buffered (S24)"


def test_cs3b_after_a_downtime_the_carried_estimate_applies(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    _persist_estimate()
    wall[0] += cost_series.window_seconds() + 60.0
    state = reranker_mod._rerank_cost.state(_STORE, _MODEL)
    assert state is not None and state.sums.n == 0, "every sample is older than the window"
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.seconds_per_token == pytest.approx(1e-4)


def test_cs3c_another_machines_samples_do_not_size_this_one(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cost_series, "_machine_override", "another-machine")
    _persist_estimate()
    monkeypatch.setattr(cost_series, "_machine_override", "test-machine")
    assert rerank_cost_estimate(_MODEL) is None


def _ram_figure_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """S200/S203: the peak readable and the memory model required (the suite default is off)."""
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)


def _assert_held_back(provider: _CostScriptedProvider, out, model: str = _MODEL) -> None:
    """S203: no width-5 seeding batch. The rerank is not run, nothing is warmed up or recorded,
    the need is flagged for the RAM ladder (S200, S220)."""
    assert out.reranked is False and out.width == 0 and out.hand_off == "ram_figure"
    assert model not in reranker_mod._warmed_model_ids
    assert model in reranker_mod._ram_figure_needed, "the need is flagged"
    assert list(provider.calls) == [], "the provider was not called at all"


def test_s203_a_process_about_to_rerank_with_no_ram_figure_takes_the_cosine_handoff_and_flags_the_need(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaces the S158 width-5 seeding rerank (cs15): a saved time estimate and no memory model."""
    _ram_figure_required(monkeypatch)
    _persist_estimate()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    _assert_held_back(provider, rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0))


def test_s203_the_hold_back_is_per_model_id(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    _ram_figure_required(monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)  # S261: ample (unreadable now means no rerank)
    _persist_estimate()
    _persist_estimate("another-model")
    _seed_ram()  # a model for _MODEL only
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    other = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4, model="another-model")
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).reranked
    _assert_held_back(other, rerank_for_recall(other, _QUERY, _docs(50, 100), budget_seconds=4.0), "another-model")


def test_s203_with_an_empty_database_and_a_readable_rss_the_first_rerank_waits_for_the_figure(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ram_figure_required(monkeypatch)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    _assert_held_back(provider, rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0))


@pytest.mark.slow(reason='5.2 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_s206_with_an_unreadable_rss_the_turn_is_not_held_back_and_the_first_rerank_is_time_bound(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stated fallback: the per-call peak cannot be read -> no model can exist, the gate lets the
    turn through, the width is sized at the S5 minimum on a first-ever startup (nothing to size from)."""
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.reranked and out.width == 5
    assert reranker_mod.ram_figure_ready(_MODEL, _STORE) is True


# ---------------------------------------------------------------------------
# S171, S176, S220 follow-up: the memory samples are a saved series (CS15 revised, CS18).
# ---------------------------------------------------------------------------


def test_cs15_with_a_saved_ram_figure_the_first_measured_rerank_is_sized_by_the_estimate(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ram_figure_required(monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)  # S261: ample (unreadable now means no rerank)
    _persist_estimate()
    _seed_ram()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    first = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert first.width == _reference_width([100] * 50, _anchor_pair_tokens(), 0.0, 1e-4, 4.0) > 5, (
        "S171: a restart after the first-ever startup on this host needs no width-5 seeding rerank"
    )


def test_cs15_a_saved_figure_of_zero_is_no_figure_and_the_rerank_waits(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ram_figure_required(monkeypatch)
    _persist_estimate()
    _seed_ram(bytes_per_token=0.0)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    _assert_held_back(provider, rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0))


def test_cs15_a_figure_whose_samples_aged_out_is_no_figure_and_the_rerank_waits(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S178, S203: after a downtime longer than the window the saved figure's samples are gone, so a
    fresh process has no model: it takes the cosine path and the ladder makes a new one
    (S200), while the time series still answers its carried fit (CS3b)."""
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    _ram_figure_required(monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)  # S261: ample (unreadable now means no rerank)
    _persist_estimate()
    _seed_ram()
    wall[0] += cost_series.window_seconds() + 60.0  # downtime: every sample is past the window
    est = reranker_mod.rerank_cost_estimate(_MODEL, _STORE)
    assert est is not None and est.ram_fit is None, "the time fit is carried, the memory model is not"
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    _assert_held_back(provider, rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0))


def test_cs15_a_figure_saved_on_another_machine_does_not_count(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ram_figure_required(monkeypatch)
    _persist_estimate()
    monkeypatch.setattr(cost_series, "_machine_override", "another-machine")
    _seed_ram()
    monkeypatch.setattr(cost_series, "_machine_override", "test-machine")
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    _assert_held_back(provider, rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0))


def test_cs15_a_saved_figure_applies_the_ram_bound_from_the_first_rerank_of_a_fresh_process(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With too little headroom for even five candidates the batch does not rerank (the S5
    rule) and records nothing: the whole predicted growth counts, nothing is already resident."""
    _ram_figure_required(monkeypatch)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1_000.0)
    _persist_estimate()
    _seed_ram()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.reranked is False and out.hand_off == "budget"








def test_cs18e_recording_a_rerank_writes_nothing_to_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_STORE, "record_cost_samples", lambda *a, **k: calls.append("write"))
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, 1_000_000.0)
    assert calls == [] and cost_series.has_pending(_STORE.db_path)






# ---------------------------------------------------------------------------
# S220: the saved memory model, its persistence and the running process's kept fit are covered in
# test_ram_shape_series.py, test_ram_figure_gate.py and test_ram_ladder.py.
# ---------------------------------------------------------------------------














def _age_out(wall: list[float]) -> None:
    wall[0] += cost_series.window_seconds() + 60.0
    reranker_mod._rerank_cost.record(_STORE, _MODEL, 700, 0.07)  # keeps the time series answering












class _OwnerLock:
    """A lock that records which thread holds it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: int | None = None

    def __enter__(self):
        self._lock.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *exc) -> bool:
        self.owner = None
        self._lock.release()
        return False


class _GuardedDict(dict):
    """A dict that records a violation whenever a thread touches it without holding `lock`, and
    which threads touched it at all."""

    def __init__(self, lock: _OwnerLock, violations: list[str]) -> None:
        super().__init__()
        self._owner_lock = lock
        self._violations = violations
        self.touched_by: set[int] = set()

    def _check(self, what: str) -> None:
        self.touched_by.add(threading.get_ident())
        if self._owner_lock.owner != threading.get_ident():
            self._violations.append(what)

    def __getitem__(self, key):
        self._check("get")
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check("get")
        return super().get(key, default)

    def __contains__(self, key):
        self._check("in")
        return super().__contains__(key)

    def __setitem__(self, key, value):
        self._check("set")
        super().__setitem__(key, value)

    def pop(self, *args):
        self._check("pop")
        return super().pop(*args)

    def clear(self):
        self._check("clear")
        super().clear()

    def __len__(self):
        self._check("len")
        return super().__len__()


def test_cross_encoder_rerank_timed_reads_the_peak_inside_the_lock_and_returns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S220/S222: the production provider resets the peak before the call and reads it after,
    both INSIDE its lock (so a rerank waiting on the lock never has its reading reset), and
    returns the growth as the third value. Able to fail: a provider that dropped the reading, or
    took it outside the lock, would not pass."""
    events: list[tuple[str, bool]] = []
    provider = object.__new__(CrossEncoderProvider)
    provider._model_id = "fake-encoder"
    provider._rerank_lock = threading.Lock()

    class _Model:
        def rerank(self, query, documents):
            events.append(("rerank", provider._rerank_lock.locked()))
            return [0.0 for _ in documents]

    provider._model = _Model()

    def _begin():
        events.append(("begin", provider._rerank_lock.locked()))
        return 1_000.0

    def _growth(before):
        events.append((f"growth since {before}", provider._rerank_lock.locked()))
        return 7_654_321.0

    monkeypatch.setattr(reranker_mod.peak_rss, "begin", _begin)
    monkeypatch.setattr(reranker_mod.peak_rss, "growth_since", _growth)
    scores, _seconds, growth = provider.rerank_timed("q", ["a", "b"])
    assert growth == 7_654_321.0 and scores == [0.0, 0.0]
    assert events == [("begin", True), ("rerank", True), ("growth since 1000.0", True)]


def test_cross_encoder_rerank_timed_checks_the_live_session_options_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object.__new__(CrossEncoderProvider)
    provider._model_id = "fake-encoder"
    provider._rerank_lock = threading.Lock()
    provider._model = SimpleNamespace(rerank=lambda q, d: [0.0 for _ in d])
    checks: list[int] = []
    monkeypatch.setattr(CrossEncoderProvider, "_check_session_options_once", lambda self: checks.append(1))
    provider.rerank_timed("q", ["a"])
    provider.rerank_timed("q", ["a"])
    assert len(checks) == 2, "called on every timed call; the check itself runs once (its own flag)"
