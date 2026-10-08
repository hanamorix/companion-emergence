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
fake providers' stand-in for a token is one character); RSS comes from a
scripted `_current_rss_bytes`. No model, no network.
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
from brain.memory import cost_series
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
    # Deterministic default: no RSS signal, so the RAM term is skipped unless
    # a test scripts RSS itself.
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)
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
    ram_per_token: float | None = None,
    headroom: float | None = None,
    peak: int = 0,
    cap: int = CANDIDATE_POOL,
) -> int:
    """Independent recomputation of the width (spec §1 / S67 / S23): the
    LARGEST n whose batch of n real + min(8, n // 2) anchors, padded to its
    longest pair, is predicted to fit the budget, and whose RSS growth past
    the measured high water `peak` (RAM per token times the padded size above
    `peak`) fits the headroom. Scans every n (no early stop), so it does not
    lean on the implementation's monotonicity argument."""
    best = 0
    for n in range(1, min(len(cand), cap) + 1):
        k = min(8, n // 2)
        longest = max([*cand[:n], *anchors[:k]])
        padded = (n + k) * longest
        if overhead + rate * padded > budget:
            continue
        if ram_per_token and headroom is not None and ram_per_token * max(0, padded - peak) > headroom:
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


@pytest.mark.parametrize(
    ("cand", "overhead", "rate", "budget", "ram_per_token", "headroom", "peak"),
    [
        ([120] * 50, 0.05, 1e-4, 4.0, None, None, 0),
        ([900, 120, 300, 2000, 80, 80, 80, 1500] + [200] * 30, 0.3, 5e-5, 2.0, None, None, 0),
        ([400] * 50, 0.0, 1e-5, 4.0, 1_000.0, 2_000_000.0, 0),
        ([400] * 50, 0.0, 1e-5, 4.0, 1_000.0, 2_000_000.0, 6_000),
        ([400] * 50, 0.0, 1e-5, 4.0, 1_000.0, 1.0, 6_000),
        ([60] * 12, 1.5, 1e-3, 4.0, None, None, 0),
        ([3000] * 50, 0.1, 1e-4, 4.0, None, None, 0),
    ],
)
def test_c1a_iii_fit_matches_the_independent_reference(
    cand: list[int],
    overhead: float,
    rate: float,
    budget: float,
    ram_per_token: float | None,
    headroom: float | None,
    peak: int,
) -> None:
    est = RerankCostEstimate(overhead, rate, ram_per_token, measured_batches=3, peak_padded_tokens=peak)
    anchors = _anchor_pair_tokens()
    got = fit_rerank_width(cand, anchors, est, budget, headroom)
    assert got == _reference_width(
        cand, anchors, overhead, rate, budget, ram_per_token=ram_per_token, headroom=headroom, peak=peak
    )


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
# C1a (v): the hourly machinery is gone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "get_rerank_width",
        "_latency_cache",
        "_memory_cache",
        "_warm_per_doc_latency",
        "_warm_per_doc_memory",
        "_measure_warm_per_doc_latency",
        "_measure_warm_per_doc_memory",
        "_LATENCY_RECOMPUTE_INTERVAL_SECONDS",
        "_MEMORY_RECOMPUTE_INTERVAL_SECONDS",
        "_MEASURE_RERANKS",
        "CALIBRATION_SAMPLE_SIZE",
    ],
)
def test_c1a_v_hourly_machinery_removed(name: str) -> None:
    assert not hasattr(reranker_mod, name)


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


@pytest.mark.parametrize("pool", [0, 1, 4, 5, 6, 9, 30, 60])
@pytest.mark.parametrize("seed_rate", [None, 1e-7, 1e-3, 1.0])
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


def test_inv_i13a_rss_unreadable_skips_the_ram_term(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0)  # absurdly tight
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-6)
    rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token is None

    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.width == 50, "no RAM figure -> width is time-bound only, headroom ignored"


def test_ram_bound_from_rss_delta_per_padded_token(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    rss = iter([1_000.0, 1_000.0 + 7 * 400 * 1_000.0])  # +1,000 bytes per padded token
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: next(rss))
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx(1_000.0)

    assert est.peak_padded_tokens == 7 * 400

    # Growth room for 5 more padded documents past the 7 already run:
    # 12 documents = 8 real + 4 anchors fit, 13 (9 real + 4) do not.
    headroom = 1_000.0 * (12 * 400 - 7 * 400)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0)
    assert out.width == _reference_width(
        [400] * 50, _anchor_pair_tokens(), 0.0, est.seconds_per_token, 4.0,
        ram_per_token=1_000.0, headroom=headroom, peak=7 * 400,
    ) == 8


def test_ram_figure_does_not_dilute_with_calls_inside_the_high_water() -> None:
    """Stage-6 finding F1: the runtime reuses the memory a batch needed, so
    calls within the high water show ~0 RSS growth. They must not dilute
    RAM per token towards 0 (which would switch the bound off): only
    growth past the high water is counted, per token of that growth."""
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 2_800, 0.1, 2_800 * 1_000.0)  # first batch: 1,000 B/token
    for _ in range(100):
        reranker_mod._record_rerank_cost(_MODEL, _STORE, 2_800, 0.1, 0.0)  # reuse, no growth
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.05, 5_000_000.0)  # another thread's allocation
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx(1_000.0), "neither diluted nor contaminated"
    assert est.peak_padded_tokens == 2_800

    reranker_mod._record_rerank_cost(_MODEL, _STORE, 4_800, 0.2, 2_000 * 500.0)  # past the high water
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx((2_800_000.0 + 1_000_000.0) / 4_800)
    assert est.peak_padded_tokens == 4_800


def test_ram_term_never_blocks_a_batch_within_the_high_water() -> None:
    """Within the high water no growth is predicted, so even a tiny headroom
    leaves every batch up to the size already run; only growth past it is
    checked against the headroom."""
    est = RerankCostEstimate(0.0, 1e-9, 1_000.0, measured_batches=4, peak_padded_tokens=12 * 400)
    width = fit_rerank_width([400] * 50, _anchor_pair_tokens(), est, 4.0, headroom_bytes=1.0)
    assert width == 8, "12 padded documents (8 real + 4 anchors) are within the high water; 13 are not"


def test_negative_rss_delta_clamps_to_zero_in_the_ram_sum() -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, -5_000.0)  # past high water 0: +1,000 tokens, 0 B
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 2_000, 0.2, 4_000.0)  # past high water 1,000: +1,000 tokens
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx(4_000.0 / 2_000)


# ---------------------------------------------------------------------------
# CONC-1: an update injected between another update's read and write is not
# lost; a reader never sees half an update. Fails with the lock removed.
# ---------------------------------------------------------------------------


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_interleaved_updates(
    monkeypatch: pytest.MonkeyPatch, second_update_wait: float
) -> tuple[reranker_mod._RamState, RerankCostEstimate | None]:
    """Two updates of the process's high water interleaved: the second runs
    between the first's read and write (the process-memory part of a measured
    rerank; the samples themselves only append to the locked buffer)."""
    second_started = threading.Event()
    second_done = threading.Event()
    reader_result: list[RerankCostEstimate | None] = []
    fired = []

    def _second_update() -> None:
        second_started.set()
        reranker_mod._record_rerank_cost(_MODEL, _STORE, 300, 0.3, 2_000.0)  # +200 tokens past 100
        second_done.set()

    def _reader() -> None:
        reader_result.append(rerank_cost_estimate(_MODEL))

    threads: list[threading.Thread] = []

    def _hook() -> None:
        if fired:
            return
        fired.append(True)
        for target in (_second_update, _reader):
            t = threading.Thread(target=target)
            threads.append(t)
            t.start()
        second_started.wait(timeout=2.0)
        # Guarded: the second update blocks on the lock, so this times out.
        # Unguarded: it completes inside the first update's read->write gap
        # (the wait returns as soon as it does).
        second_done.wait(timeout=second_update_wait)

    monkeypatch.setattr(reranker_mod, "_cost_update_hook", _hook)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.1, 1_000.0)  # +100 tokens past 0
    for t in threads:
        t.join(timeout=5.0)
    return reranker_mod._ram_state[_MODEL], (reader_result[0] if reader_result else None)


def test_conc1_interleaved_update_is_counted_and_reader_sees_whole_updates(  # against writers; a flush's take-out gap can still hide a buffered sample for a moment (cost_series module docstring)
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ram, seen = _run_interleaved_updates(monkeypatch, second_update_wait=0.3)
    assert ram.peak_x == 300, "both updates counted in the high water"
    figure = reranker_mod._ram_cost.fit(_STORE, _MODEL)
    assert figure is not None and figure.rate == pytest.approx(10.0), "and both RAM samples are in the saved series"
    assert _window().n == 2, "and both time samples are buffered"
    # The reader ran while the first update held the lock, so it saw a whole
    # state (the first update's or both), never a mix of halves; both have
    # 10 bytes per token of growth.
    assert seen is None or seen.ram_bytes_per_token == pytest.approx(10.0)


def test_conc1_able_to_fail_without_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cost_lock", _NoLock())
    # A long wait: without the lock the second update finishes inside the
    # window however slow the host, so this never flakes.
    ram, _ = _run_interleaved_updates(monkeypatch, second_update_wait=30.0)
    assert ram.peak_x == 100, "without the lock the interleaved update is lost (the guard is what CONC-1 tests)"


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


def test_cross_encoder_rerank_timed_excludes_time_waiting_for_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P-3: the time a recall spends WAITING for another thread's rerank
    (the provider lock) is not part of its measured cost. The clock advances
    10 s while the lock is held elsewhere, and the scoring itself takes
    0.5 s: the measurement must be 0.5 s."""
    clock = _Clock()
    monkeypatch.setattr(reranker_mod, "_clock", clock)
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)

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

    (scores, seconds, rss_delta) = result[0]
    assert scores == [0.0, 0.0]
    assert seconds == pytest.approx(0.5), "lock wait must not be counted as this call's cost"
    assert rss_delta is None


def test_normalization_hand_off_carries_no_scores(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the anchor median could not be taken, the raw scores never reach
    the caller (nothing can gate them): `normalization` is None."""

    def _raw(provider, query, real_documents):
        scores, seconds, rss = provider.rerank_timed(query, list(real_documents))
        return reranker_mod.AnchorNormalizationResult(
            scores=scores, real_width=len(scores), did_normalize=False, seconds=seconds, rss_delta_bytes=rss
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


def test_cs15_the_first_measured_rerank_of_a_process_runs_at_five_where_rss_is_readable(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    _persist_estimate()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    first = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert first.width == 5, "seeds the RAM high water, whatever the saved estimate says"
    second = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert second.width == _reference_width([100] * 50, _anchor_pair_tokens(), 0.0, 1e-4, 4.0) > 5


def test_cs15_the_cap_is_per_model_id(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    _persist_estimate()
    _persist_estimate("another-model")
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    other = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4, model="another-model")
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5
    assert rerank_for_recall(other, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width > 5


def test_cs15_the_cap_holds_until_a_rerank_was_recorded(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first rerank that fails (the provider raises on the measured call, after
    the warm-ups) records nothing, so the next one is still the seeding rerank."""
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    _persist_estimate()

    class _FailsOnce(_CostScriptedProvider):
        failures = 1

        def rerank(self, query, documents):
            if len(documents) > 1 and self.failures:
                self.failures -= 1
                raise RuntimeError("the measured rerank failed")
            return super().rerank(query, documents)

    provider = _FailsOnce(clock, overhead=0.0, rate=1e-4)
    with pytest.raises(RuntimeError, match="measured rerank failed"):
        rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5, (
        "nothing was measured yet, so the first measured rerank is still the seed"
    )
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width > 5


def test_cs15_with_an_empty_database_the_first_rerank_is_five_as_today(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5


# ---------------------------------------------------------------------------
# S171, S176 follow-up: the RAM figure is a saved series (CS15 revised, CS18).
# ---------------------------------------------------------------------------


def _persist_ram(model: str = _MODEL, bytes_per_token: float = 1_000.0, tokens: int = 1_000) -> None:
    """A previous process measured RSS growth past its high water and saved it."""
    reranker_mod._ram_cost.record(_STORE, model, tokens, tokens * bytes_per_token)
    assert cost_series.flush(_STORE) == 1
    reranker_mod._reset_rerank_cost_model()
    cost_series._reset_pending()


def test_cs15_with_a_saved_ram_figure_the_first_measured_rerank_is_sized_by_the_estimate(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    _persist_estimate()
    _persist_ram()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    first = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert first.width == _reference_width([100] * 50, _anchor_pair_tokens(), 0.0, 1e-4, 4.0) > 5, (
        "S171: a restart after the first-ever startup on this host needs no width-5 seeding rerank"
    )


@pytest.mark.parametrize("figure", [0.0])
def test_cs15_a_saved_figure_of_zero_is_no_figure_and_the_seed_runs(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, figure: float
) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    _persist_estimate()
    _persist_ram(bytes_per_token=figure)
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5


def test_cs15_a_figure_whose_samples_aged_out_is_no_figure_and_the_seed_runs(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S178 (replaces the carried-ratio answer of the S171 build): after a downtime longer than
    the window the saved figure's samples are gone, so the ratio series has no figure and the
    first measured rerank of the process is the width-5 seed again, while the time series still
    answers its carried fit (CS3b)."""
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    _persist_estimate()
    _persist_ram()
    wall[0] += cost_series.window_seconds() + 60.0  # downtime: every sample is past the window
    est = reranker_mod.rerank_cost_estimate(_MODEL, _STORE)
    assert est is not None and est.ram_bytes_per_token is None, "the time fit is carried, the ratio is not"
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5, (
        "an aged-out figure is no figure: the seed runs as on a first-ever startup"
    )


def test_cs15_a_figure_saved_on_another_machine_does_not_count(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    _persist_estimate()
    monkeypatch.setattr(cost_series, "_machine_override", "another-machine")
    _persist_ram()
    monkeypatch.setattr(cost_series, "_machine_override", "test-machine")
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).width == 5


def test_cs15_a_saved_figure_applies_the_ram_bound_from_the_first_rerank_of_a_fresh_process(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known consequence (review F7b, reported): the fresh process has no high water,
    so the whole batch counts as new memory; with too little headroom for even five
    candidates it does not rerank (the S5 rule) and records nothing."""
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1_000.0)
    _persist_estimate()
    _persist_ram()
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.reranked is False and out.hand_off == "budget"


def test_cs18b_the_figure_survives_a_restart_a_second_connection_counts_and_downtime_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    reranker_mod._ram_cost.record(_STORE, _MODEL, 1_000, 1_000_000.0)  # 1,000 B/token
    reranker_mod._ram_cost.record(_STORE, _MODEL, 3_000, 5_000_000.0)  # 1,667 B/token: the ratio is weighted
    assert cost_series.flush(_STORE) == 2
    weighted = 6_000_000.0 / 4_000
    reranker_mod._reset_rerank_cost_model()
    cost_series._reset_pending()
    persist_time = reranker_mod._rerank_cost
    persist_time.record(_STORE, _MODEL, 700, 0.07)
    persist_time.record(_STORE, _MODEL, 1400, 0.14)
    cost_series.flush(_STORE)
    reopened = MemoryStore(_STORE.db_path, integrity_check=False)
    try:
        est = reranker_mod.rerank_cost_estimate(_MODEL, reopened)
        assert est is not None and est.ram_bytes_per_token == pytest.approx(weighted)
        assert est.peak_padded_tokens == 0, "the high water is process memory: a fresh process starts at 0"
        other = MemoryStore(_STORE.db_path, integrity_check=False)  # a second connection flushes a sample
        try:
            reranker_mod._ram_cost.record(other, _MODEL, 1_000, 3_000_000.0)
            cost_series.flush(other)
        finally:
            other.close()
        est = reranker_mod.rerank_cost_estimate(_MODEL, reopened)
        assert est.ram_bytes_per_token == pytest.approx(9_000_000.0 / 5_000), "its sample counts at once"
        wall[0] += cost_series.window_seconds() + 60.0  # downtime: every sample is past the window
        reranker_mod._rerank_cost.record(reopened, _MODEL, 700, 0.07)  # keeps the time series alive
        est = reranker_mod.rerank_cost_estimate(_MODEL, reopened)
        assert est is not None and est.ram_bytes_per_token is None, "S178: no sample in the window, no figure"
    finally:
        reopened.close()


def test_cs18c_the_figure_is_keyed_by_model_id_and_ages_out_of_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    reranker_mod._ram_cost.record(_STORE, _MODEL, 1_000, 2_000_000.0)
    wall[0] += cost_series.window_seconds() / 2
    reranker_mod._ram_cost.record(_STORE, _MODEL, 1_000, 4_000_000.0)
    reranker_mod._ram_cost.record(_STORE, "another-model", 1_000, 9_000_000.0)
    figure = lambda model: reranker_mod._ram_cost.fit(_STORE, model).rate  # noqa: E731
    assert figure(_MODEL) == pytest.approx(3_000.0) and figure("another-model") == pytest.approx(9_000.0)
    wall[0] += cost_series.window_seconds() / 2  # the first sample is exactly one window old: gone
    assert figure(_MODEL) == pytest.approx(4_000.0)


def test_cs18a_only_a_call_past_the_high_water_with_a_readable_delta_adds_a_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, None)  # unreadable RSS
    assert reranker_mod._ram_cost.fit(_STORE, _MODEL) is None
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, 500.0)  # peak is now 1,000 but nothing was added
    assert reranker_mod._ram_cost.fit(_STORE, _MODEL) is None
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_500, 0.1, -9.0)  # past it, negative delta clamps to 0
    state = reranker_mod._ram_cost.state(_STORE, _MODEL)
    assert state is not None and (state.sums.n, state.sums.sum_x, state.sums.sum_y) == (1, 500, 0.0)


def test_cs18e_recording_a_rerank_writes_nothing_to_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_STORE, "record_cost_samples", lambda *a, **k: calls.append("write"))
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 1_000, 0.1, 1_000_000.0)
    assert calls == [] and cost_series.has_pending(_STORE.db_path)


def test_cs18f_the_ram_sample_is_buffered_before_the_new_high_water_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    real = reranker_mod._ram_cost.record

    def spy(store, key, x, y):
        seen.append(reranker_mod._ram_state.get(key, reranker_mod._RamState()).peak_x)
        real(store, key, x, y)

    monkeypatch.setattr(reranker_mod._ram_cost, "record", spy)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.1, 1_000.0)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 300, 0.3, 2_000.0)
    assert seen == [0, 100], "at each record the high water published is still the previous one"
    assert reranker_mod._ram_state[_MODEL].peak_x == 300


def test_cs18f_the_estimate_reads_the_high_water_before_the_ram_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.1, 1_000.0)
    real_fit = reranker_mod._ram_cost.fit
    landed: list[bool] = []

    def fit(store, key):
        if not landed:  # a writer lands between the reader's two reads
            landed.append(True)
            reranker_mod._record_rerank_cost(_MODEL, _STORE, 300, 0.3, 2_000.0)
        return real_fit(store, key)

    monkeypatch.setattr(reranker_mod._ram_cost, "fit", fit)
    est = reranker_mod.rerank_cost_estimate(_MODEL, _STORE)
    assert est is not None
    assert est.peak_padded_tokens == 100, "the high water was read first"
    assert est.ram_bytes_per_token == pytest.approx(10.0), "the figure already holds the later sample"


# ---------------------------------------------------------------------------
# S178: a too-high saved RAM figure recovers through the window; once the window is empty the
# next measured rerank of a fresh process re-seeds it as on a first-ever startup (A1c).
# S180: a RUNNING process (one that has recorded a measured rerank) keeps the last figure it saw
# when its window empties (K1a-K1c, K3); the lull job re-seeds it (test_rerank_recovery.py, K2).
# ---------------------------------------------------------------------------


class _GrowingProvider(_CostScriptedProvider):
    """The cost-scripted fake whose every `rerank()` also grows the scripted RSS by
    `per_token` bytes per padded token (len(documents) x the longest pair)."""

    def __init__(self, clock: _Clock, rss: list[float], per_token: float) -> None:
        super().__init__(clock, overhead=0.0, rate=1e-4)
        self.rss = rss
        self.per_token = per_token

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        out = super().rerank(query, documents)
        self.rss[0] += self.per_token * len(documents) * max(self.pair_token_lengths(query, documents))
        return out


def test_s178_a_too_high_figure_blocks_until_its_samples_age_out_and_a_fresh_process_re_seeds_it(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    wall = [1_000_000.0]
    rss = [1.0e9]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: rss[0])
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0e6)
    _persist_estimate()
    _persist_ram(bytes_per_token=1.0e9)  # absurdly high: no batch fits 1 MB
    provider = _GrowingProvider(clock, rss, per_token=500.0)
    window = cost_series.window_seconds()

    blocked = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert blocked.reranked is False and blocked.hand_off == "budget", "the F7b situation, so the case is not vacuous"
    assert [c for c in provider.calls if len(c) > 1] == [], "no measured rerank ran"
    assert reranker_mod._ram_cost.state(_STORE, _MODEL).sums.n == 1, "nothing recorded"

    wall[0] += window - 60.0  # the sample is still inside the window: the figure applies as before
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.ram_bytes_per_token == pytest.approx(1.0e9)
    assert rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0).reranked is False

    wall[0] += 120.0  # the window is empty now
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.ram_bytes_per_token is None and est.measured_batches > 0
    seeded = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert seeded.reranked and seeded.width == 5, "the first measured rerank of the process: the width-5 seed"
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.ram_bytes_per_token == pytest.approx(500.0), "re-seeded from this measurement"

    assert cost_series.flush(_STORE) >= 1  # a restart now sizes the first rerank from the new figure
    reranker_mod._reset_rerank_cost_model()
    cost_series._reset_pending()
    fresh = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    est = rerank_cost_estimate(_MODEL)
    assert fresh.width == _reference_width(
        [100] * 50, _anchor_pair_tokens(), est.overhead_seconds, est.seconds_per_token, 4.0,
        ram_per_token=500.0, headroom=1.0e6, peak=0,
    ) > 5


def _growth_for(n: int, peak: int, pair_tokens: int = 100) -> int:
    """Independent: padded tokens past `peak` of a batch of `n` real candidates of `pair_tokens`
    plus its anchors, padded to the longest pair (the width fit's RAM-term input)."""
    k = min(8, n // 2)
    longest = max([pair_tokens, *_anchor_pair_tokens()[:k]])
    return max(0, (n + k) * longest - peak)


def _running_process_with_a_figure(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, wall: list[float], rss: list[float]
) -> tuple[_GrowingProvider, int]:
    """A saved time estimate, no saved RAM figure: the process's first rerank is the width-5 seed
    (it records the 500 B/token sample), so it is now "running" (its RAM state holds the model id).
    Returns the provider and the process's high water."""
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: rss[0])
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    _persist_estimate()
    provider = _GrowingProvider(clock, rss, per_token=500.0)
    first = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert first.width == 5, "no saved figure: the process's first rerank is the seed"
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx(500.0) and est.ram_figure_kept is False
    return provider, reranker_mod._ram_state[_MODEL].peak_x


def test_s180_a_running_process_whose_ram_window_emptied_keeps_its_last_figure_and_sizes_by_it(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1a/K1b (replaces A1g, which pinned the opposite): RAM samples exist only for calls past the
    process's high water, so a plateaued process loses its window 24 h after its last new peak. It
    keeps the last figure: the width fit applies the RAM term, nothing runs unbounded and no seed runs."""
    wall, rss = [1_000_000.0], [1.0e9]
    provider, peak = _running_process_with_a_figure(clock, monkeypatch, wall, rss)

    wall[0] += cost_series.window_seconds() + 60.0  # the process kept running; its only RAM sample aged out
    state = reranker_mod._ram_cost.state(_STORE, _MODEL)
    assert state is not None and state.sums.n == 0, "the window is empty"
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx(500.0) and est.ram_figure_kept is True
    assert est.peak_padded_tokens == peak

    headroom = 500.0 * _growth_for(8, peak) + 1.0  # room for 8 real candidates and not for 9
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    reference = _reference_width(
        [100] * 50, _anchor_pair_tokens(), est.overhead_seconds, est.seconds_per_token, 4.0,
        ram_per_token=500.0, headroom=headroom, peak=peak,
    )
    unbounded = _reference_width([100] * 50, _anchor_pair_tokens(), est.overhead_seconds, est.seconds_per_token, 4.0)
    assert 5 < reference < unbounded, "the RAM term (and only it) limits this width; a seed would give 5"
    second = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert second.width == reference
    batch = provider.calls[-1]
    padded = len(batch) * max(provider.pair_token_lengths(_QUERY, batch))
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_figure_kept is False, "the call past the high water recorded a sample: the window answers again"
    assert est.ram_bytes_per_token == pytest.approx(500.0 * padded / (padded - peak))


def test_s180_a_process_that_has_not_measured_does_not_use_a_kept_figure_and_seeds(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1a: S178 is unchanged for a process whose RAM state lacks the model id (plan review F1)."""
    wall, rss = [1_000_000.0], [1.0e9]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: rss[0])
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0e6)
    _persist_estimate()
    _persist_ram(bytes_per_token=1.0e9)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == pytest.approx(1.0e9), "the figure is in the window; the process saw it"
    assert len(reranker_mod._ram_kept) == 1 and _MODEL not in reranker_mod._ram_state
    wall[0] += cost_series.window_seconds() + 60.0
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token is None and est.ram_figure_kept is False
    provider = _GrowingProvider(clock, rss, per_token=500.0)
    seeded = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert seeded.reranked and seeded.width == 5


def _age_out(wall: list[float]) -> None:
    wall[0] += cost_series.window_seconds() + 60.0
    reranker_mod._rerank_cost.record(_STORE, _MODEL, 700, 0.07)  # keeps the time series answering


def test_s180_the_kept_figure_follows_the_last_figure_seen_and_a_zero_rate_forgets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K1c: a new sample replaces the kept figure; a window that answers a rate of 0 removes it, so
    the older figure does not come back when that window empties (plan review F3)."""
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    reranker_mod._rerank_cost.record(_STORE, _MODEL, 700, 0.07)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.01, 1.0e5)  # 1000 B/token, high water 100
    assert rerank_cost_estimate(_MODEL).ram_bytes_per_token == pytest.approx(1000.0)
    _age_out(wall)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_figure_kept and est.ram_bytes_per_token == pytest.approx(1000.0)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 200, 0.02, 3.0e5)  # 3000 B/token past 100
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_figure_kept is False and est.ram_bytes_per_token == pytest.approx(3000.0)
    _age_out(wall)
    assert rerank_cost_estimate(_MODEL).ram_bytes_per_token == pytest.approx(3000.0), "the LAST figure, not the first"
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 300, 0.03, 0.0)  # past the high water, no growth
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token == 0.0 and est.ram_figure_kept is False
    assert reranker_mod._ram_kept == {}, "a window rate of 0 is no figure and forgets the kept one"
    _age_out(wall)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_token is None and est.ram_figure_kept is False, "the older figure does not resurrect"


def test_s180_the_seed_sample_is_kept_even_if_no_estimate_call_happens_before_it_ages_out(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-6 review F2: the figure is remembered when the process RECORDS the sample, not only when an
    estimate call reads the window. One message after a restart (its seed sample is the only RAM sample),
    then more than a day of quiet: the next message still finds the figure."""
    wall, rss = [1_000_000.0], [1.0e9]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: rss[0])
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)
    _persist_estimate()
    provider = _GrowingProvider(clock, rss, per_token=500.0)
    first = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert first.width == 5
    assert len(reranker_mod._ram_kept) == 1, "remembered at the record"
    wall[0] += cost_series.window_seconds() + 60.0  # no estimate call in between
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_figure_kept is True and est.ram_bytes_per_token == pytest.approx(500.0)


def test_s180_the_kept_figure_is_keyed_by_database_and_model_id_and_shared_by_stores_on_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1c: another model id, or another database, has no kept figure; two store objects on one
    file (the job's per-tick store and the reply path's) share it."""
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    reranker_mod._rerank_cost.record(_STORE, _MODEL, 700, 0.07)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.01, 1.0e5)
    other_model = "cost-other"
    reranker_mod._rerank_cost.record(_STORE, other_model, 700, 0.07)
    reranker_mod._record_rerank_cost(other_model, _STORE, 100, 0.01, None)  # measured, no RAM sample
    assert rerank_cost_estimate(_MODEL).ram_bytes_per_token == pytest.approx(1000.0)
    wall[0] += cost_series.window_seconds() + 60.0
    for model in (_MODEL, other_model):
        reranker_mod._rerank_cost.record(_STORE, model, 700, 0.07)
    assert rerank_cost_estimate(_MODEL).ram_figure_kept is True
    est = rerank_cost_estimate(other_model)
    assert est.ram_bytes_per_token is None and est.ram_figure_kept is False, "another model id: no kept figure"
    again = MemoryStore(_STORE.db_path, integrity_check=False)  # another object on the same file
    try:
        est = reranker_mod.rerank_cost_estimate(_MODEL, again)
        assert est.ram_figure_kept is True and est.ram_bytes_per_token == pytest.approx(1000.0)
    finally:
        again.close()
    elsewhere = MemoryStore(tmp_path / "elsewhere.db", integrity_check=False)  # another database
    try:
        reranker_mod._rerank_cost.record(elsewhere, _MODEL, 700, 0.07)
        est = reranker_mod.rerank_cost_estimate(_MODEL, elsewhere)
        assert est is not None and est.ram_bytes_per_token is None and est.ram_figure_kept is False
    finally:
        elsewhere.close()


def test_s180_a_restart_forgets_the_kept_figure_and_nothing_is_written_for_it(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K1d: the kept figure is process memory: a restart (module state reset, a fresh connection) with
    the window empty has no figure and seeds again (S178), and reading it wrote no row."""
    wall, rss = [1_000_000.0], [1.0e9]
    provider, _peak = _running_process_with_a_figure(clock, monkeypatch, wall, rss)
    assert cost_series.flush(_STORE) >= 1
    wall[0] += cost_series.window_seconds() + 60.0
    statements: list[str] = []
    _STORE._conn.set_trace_callback(statements.append)
    try:
        est = rerank_cost_estimate(_MODEL)
    finally:
        _STORE._conn.set_trace_callback(None)
    assert est.ram_figure_kept is True
    assert not [q for q in statements if q.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE"))]
    reranker_mod._reset_rerank_cost_model()
    cost_series._reset_pending()
    fresh = MemoryStore(_STORE.db_path, integrity_check=False)
    try:
        est = reranker_mod.rerank_cost_estimate(_MODEL, fresh)
        assert est.ram_bytes_per_token is None and est.ram_figure_kept is False
        monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1.0e6)
        seeded = reranker_mod.rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0, store=fresh)
        assert seeded.reranked and seeded.width == 5
    finally:
        fresh.close()


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


def test_s180_the_kept_figure_is_touched_only_under_the_cost_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """K3: estimate calls from several threads (each with its OWN store object on the one file, as the
    job, the tool and the reply path have), a recording thread and a reset never touch the kept dict
    without holding `_cost_lock`. The guard records the owning thread (a bare `locked()` check would
    pass while another thread holds the lock) and the test asserts the readers really reached the dict
    in both branches (window answering, window empty and the figure kept)."""
    wall = [1_000_000.0]
    monkeypatch.setattr(cost_series, "_wall_clock", lambda: wall[0])
    lock = _OwnerLock()
    violations: list[str] = []
    guarded = _GuardedDict(lock, violations)
    monkeypatch.setattr(reranker_mod, "_cost_lock", lock)
    monkeypatch.setattr(reranker_mod, "_ram_kept", guarded)
    reranker_mod._rerank_cost.record(_STORE, _MODEL, 700, 0.07)
    reranker_mod._record_rerank_cost(_MODEL, _STORE, 100, 0.01, 1.0e5)  # running; window answers 1000 B/token
    errors: list[BaseException] = []
    results: list[RerankCostEstimate | None] = []
    reader_ids: set[int] = set()

    def _reader(phase_kept: bool) -> None:
        store = MemoryStore(_STORE.db_path, integrity_check=False)
        try:
            reader_ids.add(threading.get_ident())
            for _ in range(20):
                est = reranker_mod.rerank_cost_estimate(_MODEL, store)
                results.append(est)
                if est is not None:
                    assert est.ram_figure_kept is phase_kept
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)
        finally:
            store.close()

    def _run(phase_kept: bool) -> None:
        threads = [threading.Thread(target=_reader, args=(phase_kept,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

    _run(False)  # the window answers: readers write the figure
    wall[0] += cost_series.window_seconds() + 60.0
    reranker_mod._rerank_cost.record(_STORE, _MODEL, 700, 0.07)
    _run(True)  # the window is empty: readers read the kept figure
    reranker_mod._reset_rerank_cost_model()  # a reset under the same lock
    assert not errors, errors
    answered = [e for e in results if e is not None]
    assert len(answered) == len(results) == 160, "every reader call answered an estimate (the store was usable from its thread)"
    assert reader_ids <= guarded.touched_by, "every reader thread reached the guarded dict"
    assert violations == [], "the kept dict was touched outside the cost lock"
