"""Name-recall fix R1: the per-message rerank width (spec §1; criteria C1a,
C1c, CONC-1, C2c part 1, INV-I13a, ADV-9 in
`changes/name-recall-fix/1.5-criteria.md`).

The hourly width sample (one query's top-5 documents, cached ~1 h,
diagnosis H8) is replaced by a cost model learned from every recall-time
rerank: seconds = overhead + rate * (pairs * longest pair), both fitted by
least squares over running sums, per process and per reranker model id;
each message fits its own width to its own candidates.

All offline and deterministic: timing comes from a scripted clock injected
through `reranker._clock`, advanced by a fake reranker whose per-call cost is
exactly `overhead + rate * len(documents) * longest pair`; RSS comes from a
scripted `_current_rss_bytes`. No model, no network.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import brain.memory.reranker as reranker_mod
from brain.dev_constants import RERANK_MIN_REAL_CANDIDATES
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.reranker import (
    ANCHOR_POOL,
    CrossEncoderProvider,
    FakeRerankerProvider,
    RerankCostEstimate,
    fit_rerank_width,
    rerank_cost_estimate,
    rerank_for_recall,
    surviving_pair_char_lengths,
)

_MODEL = "cost-scripted"
_QUERY = "q"


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
        longest = max(self.pair_char_lengths(query, documents))
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


def _docs(n: int, pair_chars: int, tag: str = "d") -> list[str]:
    """`n` distinct documents whose (query `_QUERY`, doc) pair is exactly
    `pair_chars` characters."""
    body = pair_chars - len(_QUERY)
    return [(f"{tag}{i:03d}" + "x" * body)[:body] for i in range(n)]


def _anchor_pair_chars(query: str = _QUERY) -> list[int]:
    return [len(query) + len(a) for a in ANCHOR_POOL]


def _reference_width(
    cand: list[int],
    anchors: list[int],
    overhead: float,
    rate: float,
    budget: float,
    *,
    ram_per_char: float | None = None,
    headroom: float | None = None,
    cap: int = CANDIDATE_POOL,
) -> int:
    """Independent recomputation of the width (spec §1 / S67 / S23): the
    LARGEST n whose batch of n real + min(8, n // 2) anchors, padded to its
    longest pair, is predicted to fit the budget and the RAM headroom. Scans
    every n (no early stop), so it does not lean on the implementation's
    monotonicity argument."""
    best = 0
    for n in range(1, min(len(cand), cap) + 1):
        k = min(8, n // 2)
        longest = max([*cand[:n], *anchors[:k]])
        padded = (n + k) * longest
        if overhead + rate * padded > budget:
            continue
        if ram_per_char and headroom is not None and ram_per_char * padded > headroom:
            continue
        best = n
    return best


def _sum_model_width(cand: list[int], anchors: list[int], rate: float, budget: float) -> int:
    """The OLD model's shape (a plain per-character sum, no padding, no
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
    sums = reranker_mod._cost_sums[_MODEL]
    assert (sums.n, sums.sum_x, sums.sum_xx) == (1, x, x * x)
    assert sums.sum_y == pytest.approx(y)
    est = rerank_cost_estimate(_MODEL)
    assert est is not None and est.measured_batches == 1
    # One batch: overhead 0, rate = ratio of sums.
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_char == pytest.approx(y / x)


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

    sums = reranker_mod._cost_sums[_MODEL]
    assert sums.n == 2
    # Independent ordinary least squares over the measured (x, y) pairs.
    measured = [c for c in provider.calls if len(c) > 1]
    xs = [len(c) * max(len(_QUERY) + len(d) for d in c) for c in measured]
    ys = [o0 + r0 * x for x in xs]
    assert len(set(xs)) == 2, "fixture precondition: two different padded sizes"
    slope, intercept = np.polyfit(xs, ys, 1)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == pytest.approx(intercept, rel=1e-9) == pytest.approx(o0, rel=1e-9)
    assert est.seconds_per_char == pytest.approx(slope, rel=1e-9) == pytest.approx(r0, rel=1e-9)
    # Able to fail: the plain ratio-of-sums model (no overhead term) is wrong here.
    assert sum(ys) / sum(xs) != pytest.approx(r0, rel=1e-3)


def test_c1a_ii_same_padded_size_is_degenerate_ratio_of_sums() -> None:
    reranker_mod._record_rerank_cost(_MODEL, 700, 1.0, None)
    reranker_mod._record_rerank_cost(_MODEL, 700, 2.0, None)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_char == pytest.approx(3.0 / 1400)


def test_c1a_ii_negative_fitted_overhead_falls_back_to_ratio() -> None:
    # slope 0.002, intercept -0.1 -> negative overhead -> ratio of sums.
    reranker_mod._record_rerank_cost(_MODEL, 100, 0.1, None)
    reranker_mod._record_rerank_cost(_MODEL, 200, 0.3, None)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_char == pytest.approx(0.4 / 300)


def test_c1a_ii_non_positive_fitted_rate_falls_back_to_ratio() -> None:
    reranker_mod._record_rerank_cost(_MODEL, 100, 0.3, None)
    reranker_mod._record_rerank_cost(_MODEL, 200, 0.1, None)
    est = rerank_cost_estimate(_MODEL)
    assert est.overhead_seconds == 0.0
    assert est.seconds_per_char == pytest.approx(0.4 / 300)


def test_no_estimate_before_any_measurement() -> None:
    assert rerank_cost_estimate(_MODEL) is None


# ---------------------------------------------------------------------------
# C1a (iii): the width is the longest prefix whose PADDED batch fits.
# ---------------------------------------------------------------------------


def test_c1a_iii_long_candidate_inside_short_prefix_stops_the_width(clock: _Clock) -> None:
    """Six short candidates, then one long one: padding the batch to the long
    pair makes the 7-candidate prefix cost 10 x 1000 characters, although the
    plain sum of its pair lengths would have fit."""
    reranker_mod._record_rerank_cost(_MODEL, 10_000, 1.0, None)  # overhead 0, rate 1e-4
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-4)
    cand = [250] * 6 + [1000] + [250] * 5
    docs = [_docs(1, n, f"c{i}")[0] for i, n in enumerate(cand)]
    budget = 0.5

    out = rerank_for_recall(provider, _QUERY, docs, budget_seconds=budget)

    anchors = _anchor_pair_chars()
    expected = _reference_width(cand, anchors, 0.0, 1e-4, budget)
    assert out.width == expected == 6
    assert _sum_model_width(cand, anchors, 1e-4, budget) > 6, (
        "fixture discriminates: the old sum-of-pair-lengths model would have taken the long candidate"
    )
    combined = [c for c in provider.calls if len(c) > 1]
    assert combined[-1] == docs[:6] + ANCHOR_POOL[:3]


@pytest.mark.parametrize(
    ("cand", "overhead", "rate", "budget", "ram_per_char", "headroom"),
    [
        ([120] * 50, 0.05, 1e-4, 4.0, None, None),
        ([900, 120, 300, 2000, 80, 80, 80, 1500] + [200] * 30, 0.3, 5e-5, 2.0, None, None),
        ([400] * 50, 0.0, 1e-5, 4.0, 1_000.0, 2_000_000.0),
        ([60] * 12, 1.5, 1e-3, 4.0, None, None),
        ([3000] * 50, 0.1, 1e-4, 4.0, None, None),
    ],
)
def test_c1a_iii_fit_matches_the_independent_reference(
    cand: list[int],
    overhead: float,
    rate: float,
    budget: float,
    ram_per_char: float | None,
    headroom: float | None,
) -> None:
    est = RerankCostEstimate(overhead, rate, ram_per_char, measured_batches=3)
    anchors = _anchor_pair_chars()
    got = fit_rerank_width(cand, anchors, est, budget, headroom)
    assert got == _reference_width(
        cand, anchors, overhead, rate, budget, ram_per_char=ram_per_char, headroom=headroom
    )


def test_fit_is_the_minimum_before_the_first_measurement() -> None:
    assert fit_rerank_width([5000] * 50, _anchor_pair_chars(), None, 0.001, None) == 5
    assert fit_rerank_width([10] * 3, _anchor_pair_chars(), None, 4.0, None) == 3


def test_fit_is_capped_at_fifty_real_and_at_max_real() -> None:
    est = RerankCostEstimate(0.0, 1e-12, None, measured_batches=2)
    assert fit_rerank_width([10] * 80, _anchor_pair_chars(), est, 4.0, None) == CANDIDATE_POOL == 50
    assert fit_rerank_width([10] * 80, _anchor_pair_chars(), est, 4.0, None, max_real=12) == 12


# ---------------------------------------------------------------------------
# C1a (iv): long -> short -> long -> short; each message's own width.
# ---------------------------------------------------------------------------


def test_c1a_iv_short_message_after_a_long_one_gets_a_wider_rerank(clock: _Clock) -> None:
    o0, r0, budget = 0.2, 2e-4, 4.0
    provider = _CostScriptedProvider(clock, overhead=o0, rate=r0)
    long_docs, short_docs = _docs(50, 1200, "L"), _docs(50, 150, "S")
    anchors = _anchor_pair_chars()

    widths = []
    for docs, pair in ((long_docs, 1200), (short_docs, 150), (long_docs, 1200), (short_docs, 150)):
        est = rerank_cost_estimate(_MODEL)
        out = rerank_for_recall(provider, _QUERY, docs, budget_seconds=budget)
        if est is None:
            expected = 5
        else:
            expected = _reference_width(
                [pair] * 50, anchors, est.overhead_seconds, est.seconds_per_char, budget
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
# C1c: pair lengths at their surviving (post-truncation) length (S62).
# ---------------------------------------------------------------------------


def _fake_encoding(query: str, doc: str, doc_keep: int | None) -> SimpleNamespace:
    """A pair encoding shaped like `tokenizers.Encoding`: one token per 10
    characters, special tokens (no sequence id) around each segment,
    document truncated to `doc_keep` characters when given."""
    offsets, seq_ids = [(0, 0)], [None]
    for start in range(0, len(query), 10):
        offsets.append((start, min(start + 10, len(query))))
        seq_ids.append(0)
    offsets.append((0, 0))
    seq_ids.append(None)
    kept = len(doc) if doc_keep is None else doc_keep
    for start in range(0, kept, 10):
        offsets.append((start, min(start + 10, kept)))
        seq_ids.append(1)
    offsets.append((0, 0))
    seq_ids.append(None)
    overflowing = [] if doc_keep is None else [SimpleNamespace()]
    return SimpleNamespace(offsets=offsets, sequence_ids=seq_ids, overflowing=overflowing)


class _FakeTokenizer:
    def __init__(self, limit_chars: int) -> None:
        self.limit_chars = limit_chars
        self.batches: list[list[tuple[str, str]]] = []

    def encode_batch(self, pairs):
        self.batches.append(list(pairs))
        return [
            _fake_encoding(q, d, self.limit_chars if len(d) > self.limit_chars else None)
            for q, d in pairs
        ]


def test_c1c_truncated_document_counts_at_its_surviving_length() -> None:
    query = "what is up"
    long_doc, short_doc = "y" * 5000, "z" * 300
    encodings = [_fake_encoding(query, long_doc, 1200), _fake_encoding(query, short_doc, None)]

    lengths = surviving_pair_char_lengths(query, [long_doc, short_doc], encodings)

    assert lengths[0] == len(query) + 1200, "a plain len(doc) would give 5010"
    assert lengths[1] == len(query) + 300, "a document under the limit counts at its full length"


def _encoder_provider(tokenizer) -> CrossEncoderProvider:
    """A CrossEncoderProvider built without fastembed (no model files): its
    `_model.model` stands in for fastembed's loaded cross-encoder."""
    provider = object.__new__(CrossEncoderProvider)
    provider._model_id = "fake-encoder"
    provider._rerank_lock = threading.Lock()
    inner = SimpleNamespace(model=object(), tokenizer=tokenizer, loads=0)
    provider._model = SimpleNamespace(model=inner)
    return provider


def test_c1c_cross_encoder_provider_uses_its_tokenizer_offsets() -> None:
    tokenizer = _FakeTokenizer(limit_chars=1200)
    provider = _encoder_provider(tokenizer)
    docs = ["y" * 5000, "z" * 300]

    lengths = provider.pair_char_lengths("what is up", docs)

    assert lengths == [10 + 1200, 10 + 300]
    assert tokenizer.batches == [[("what is up", d) for d in docs]], "one encode_batch over the pairs"


def test_c1c_tokenizer_loaded_on_first_use() -> None:
    tokenizer = _FakeTokenizer(limit_chars=1200)
    provider = _encoder_provider(tokenizer)
    inner = provider._model.model
    inner.model = None

    def _load() -> None:
        inner.loads += 1
        inner.model = object()

    inner.load_onnx_model = _load
    provider.pair_char_lengths("q", ["abc"])
    provider.pair_char_lengths("q", ["abc"])
    assert inner.loads == 1


def test_c1c_tokenizer_failure_falls_back_to_full_lengths(caplog: pytest.LogCaptureFixture) -> None:
    class _Boom:
        def encode_batch(self, pairs):
            raise RuntimeError("simulated tokenizer failure")

    provider = _encoder_provider(_Boom())
    with caplog.at_level(logging.WARNING, logger=reranker_mod.__name__):
        lengths = provider.pair_char_lengths("q", ["y" * 5000])
    assert lengths == [1 + 5000], "falls back to the full (over-estimating) length"
    assert any("surviving pair lengths unavailable" in r.message for r in caplog.records)


def test_c1c_width_fit_and_measurement_use_the_surviving_lengths(clock: _Clock) -> None:
    class _Truncating(_CostScriptedProvider):
        def pair_char_lengths(self, query, documents):
            return [len(query) + min(len(d), 1200) for d in documents]

    provider = _Truncating(clock, overhead=0.0, rate=1e-5)
    rerank_for_recall(provider, _QUERY, ["w" * 5000] * 10, budget_seconds=4.0)

    sums = reranker_mod._cost_sums[_MODEL]
    assert sums.sum_x == 7 * (1 + 1200), "x uses the surviving pair length, not 7 * 5001"


# ---------------------------------------------------------------------------
# C2c part 1: every rerank carries >= 5 real + >= 2 anchors and is normalized.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool", [0, 1, 4, 5, 6, 9, 30, 60])
@pytest.mark.parametrize("seed_rate", [None, 1e-7, 1e-3, 1.0])
def test_c2c_every_rerank_has_five_real_and_two_anchors_and_is_normalized(
    clock: _Clock, pool: int, seed_rate: float | None
) -> None:
    if seed_rate is not None:
        reranker_mod._record_rerank_cost(_MODEL, 1_000, seed_rate * 1_000, None)
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
    assert est.ram_bytes_per_char is None

    out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
    assert out.width == 50, "no RAM figure -> width is time-bound only, headroom ignored"


def test_ram_bound_from_rss_delta_per_padded_char(clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
    rss = iter([1_000.0, 1_000.0 + 7 * 400 * 1_000.0])  # +1,000 bytes per padded character
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: next(rss))
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)
    rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_char == pytest.approx(1_000.0)

    headroom = 1_000.0 * 12 * 400  # fits 12 padded documents: 8 real + 4 anchors
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: headroom)
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: None)
    out = rerank_for_recall(provider, _QUERY, _docs(50, 400), budget_seconds=4.0)
    assert out.width == _reference_width(
        [400] * 50, _anchor_pair_chars(), 0.0, est.seconds_per_char, 4.0,
        ram_per_char=1_000.0, headroom=headroom,
    ) == 8


def test_negative_rss_delta_clamps_to_zero_in_the_ram_sum() -> None:
    reranker_mod._record_rerank_cost(_MODEL, 1_000, 0.1, -5_000.0)
    reranker_mod._record_rerank_cost(_MODEL, 2_000, 0.2, 4_000.0)
    est = rerank_cost_estimate(_MODEL)
    assert est.ram_bytes_per_char == pytest.approx(4_000.0 / 3_000)


# ---------------------------------------------------------------------------
# CONC-1: an update injected between another update's read and write is not
# lost; a reader never sees half an update. Fails with the lock removed.
# ---------------------------------------------------------------------------


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_interleaved_updates(monkeypatch: pytest.MonkeyPatch) -> tuple[int, RerankCostEstimate | None]:
    second_started = threading.Event()
    second_done = threading.Event()
    reader_result: list[RerankCostEstimate | None] = []
    fired = []

    def _second_update() -> None:
        second_started.set()
        reranker_mod._record_rerank_cost(_MODEL, 300, 0.3, None)
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
        # Unguarded: it completes inside the first update's read->write gap.
        second_done.wait(timeout=0.3)

    monkeypatch.setattr(reranker_mod, "_cost_update_hook", _hook)
    reranker_mod._record_rerank_cost(_MODEL, 100, 0.1, None)
    for t in threads:
        t.join(timeout=5.0)
    return reranker_mod._cost_sums[_MODEL].n, (reader_result[0] if reader_result else None)


def test_conc1_interleaved_update_is_counted_and_reader_sees_whole_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count, seen = _run_interleaved_updates(monkeypatch)
    sums = reranker_mod._cost_sums[_MODEL]
    assert count == 2, "both updates counted"
    assert sums.sum_x == 400 and sums.sum_y == pytest.approx(0.4)
    # The reader ran while the first update held the lock, so it saw a whole
    # state (one sample or both), never a mix of halves. Both samples lie on
    # y = 0.001 * x, so either whole state fits to that rate.
    assert seen is not None and seen.measured_batches in (1, 2)
    assert seen.seconds_per_char == pytest.approx(0.001)
    assert seen.overhead_seconds == pytest.approx(0.0, abs=1e-12)


def test_conc1_able_to_fail_without_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_cost_lock", _NoLock())
    count, _ = _run_interleaved_updates(monkeypatch)
    assert count == 1, "without the lock the interleaved update is lost (the guard is what CONC-1 tests)"


# ---------------------------------------------------------------------------
# ADV-9 (advisory; owner input for the PARKED Q15 / F10): the absorbing state.
# ---------------------------------------------------------------------------


def test_adv9_inflated_first_sample_is_absorbing_until_restart(clock: _Clock) -> None:
    """Demonstration, not a requirement: one inflated measurement makes every
    later message hand off ("budget") with no rerank, so no new measurement
    is ever taken and the estimate never moves. No recovery rule is built
    (PARKED for the owner); this test changes when he rules."""
    reranker_mod._record_rerank_cost(_MODEL, 100, 1_000.0, None)  # 10 s per character
    before = reranker_mod._cost_sums[_MODEL]
    provider = _CostScriptedProvider(clock, overhead=0.0, rate=1e-9)  # the host is actually fast

    for _ in range(5):
        out = rerank_for_recall(provider, _QUERY, _docs(50, 100), budget_seconds=4.0)
        assert out.hand_off == "budget" and not out.measured

    assert reranker_mod._cost_sums[_MODEL] == before
    assert [c for c in provider.calls if len(c) > 1] == []
