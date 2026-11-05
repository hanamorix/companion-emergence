"""Name-recall fix R6 (spec §3, §4, §7; plan P-7, P-10, P-23, P-26-P-29; S97, S98):
the per-paragraph semantic search and the final result order.

Criteria: C8b (guaranteed slots), C8c (guarantees over the cap, S64), C9a (one
paragraph behaves as before; S61 budget accounting), C9b (whole-message
fallback; S63 first-message rule), C9d (allocation above 50, S58), C10 (the
calibration row's query is the paragraph), CONC-1 (the `T_m` / `T_p` running
averages), CONC-6 (the stacked-matrix cache), P-27 (the lean pool), S86 (family
tails regrouped and de-duplicated), S97/S98 (six result groups in one editable
list, passive recall and the search tool alike).

All offline: an axis embedder (each paragraph text is its own unit axis),
scripted fake rerankers, a scripted clock, an in-tmp store. Synthetic data only.
"""

from __future__ import annotations

import logging
import math
import random
import threading
from pathlib import Path

import numpy as np
import pytest

from brain import dev_constants, tunables
from brain.bridge import model_tier
from brain.memory import embedding_matrix as matrix_mod
from brain.memory import reranker as reranker_mod
from brain.memory import semantic_recall as sr
from brain.memory.embedding_matrix import EmbeddingMatrix, build_embedding_matrix
from brain.memory.embeddings import EmbeddingProvider, cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.reranker import ANCHOR_POOL, RerankerProvider
from brain.memory.semantic_recall import (
    COSINE_FAMILY,
    COSINE_GENUINE,
    COSINE_PATH,
    KEYWORD_FAMILY,
    KEYWORD_GENUINE,
    RERANKED_FAMILY,
    RERANKED_GENUINE,
    RERANKED_PATH,
    GatedRanking,
    ParagraphOutcome,
    ParagraphSearch,
    SemanticHit,
    SemanticRecallResult,
    allocate_widths,
    assemble_paragraph_results,
    build_semantic_candidate_pool,
    genuine_first_coarse_cut,
    merge_with_keyword,
    order_result_groups,
    run_semantic_recall,
    search_paragraphs,
    split_paragraphs,
)
from brain.memory.store import CALIBRATION_SCORE_SCALE, COSINE_SCORE_SCALE, Memory, MemoryStore
from brain.tools.dispatch import dispatch
from tests.thread_support import finish_threads
from tests.unit.brain.chat.test_recall_keyword_merge import _LONG, _active_rows, _render
from tests.unit.brain.memory.test_monologue_last import (
    _EMBEDDER_ID,
    _RERANKER_ID,
    _cosine_floor,
    _no_reranker,
    _rerank_floor,
    _warm,
)

_DIM = 384
# Query texts not in the embedder's axis table (the whole-message query, the
# first message's "rest" query) all embed to this axis.
_OTHER = _DIM - 1
# Every memory's leftover length goes here; no query ever points at it.
_RESIDUAL = _DIM - 2


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Clock:
    """The budget clock (`semantic_recall._clock`), advanced only by the fakes."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _AxisEmbedder(EmbeddingProvider):
    """Each known text embeds to its own unit axis; any other text to
    `_OTHER`. Records every batch; each text of a batch advances the clock by
    `seconds_per_text`."""

    def __init__(
        self,
        axes: dict[str, int],
        clock: _Clock | None = None,
        seconds_per_text: float = 0.0,
        scale: float = 1.0,
    ):
        self.axes = axes
        self.clock = clock
        self.seconds_per_text = seconds_per_text
        self.scale = scale
        self.batches: list[list[str]] = []

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        v[self.axes.get(text, _OTHER)] = self.scale
        return v

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        if self.clock is not None:
            self.clock.advance(self.seconds_per_text * len(texts))
        return [self.embed(t) for t in texts]

    def embedding_dim(self) -> int:
        return _DIM

    def model_id(self) -> str:
        return _EMBEDDER_ID


class _Reranker(RerankerProvider):
    """Scripted scores by document (anchors 0.0, so normalized == raw).
    Records `(query, documents)` per call. A single-document call (a warm-up)
    advances the clock by `warm_seconds`; each pair-size probe by
    `probe_seconds`."""

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        default: float = -1_000.0,
        clock: _Clock | None = None,
        warm_seconds: float = 0.0,
        probe_seconds: float = 0.0,
    ):
        self.scores = {**dict.fromkeys(ANCHOR_POOL, 0.0), **(scores or {})}
        self.default = default
        self.clock = clock
        self.warm_seconds = warm_seconds
        self.probe_seconds = probe_seconds
        self.calls: list[tuple[str, list[str]]] = []
        self.probed: list[str] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if len(documents) == 1 and self.clock is not None:
            self.clock.advance(self.warm_seconds)
        return [self.scores.get(d, self.default) for d in documents]

    def pair_token_lengths(self, query, documents):
        self.probed.append(query)
        if self.clock is not None:
            self.clock.advance(self.probe_seconds)
        return super().pair_token_lengths(query, documents)

    def model_id(self) -> str:
        return _RERANKER_ID

    def scored(self) -> list[tuple[str, list[str]]]:
        return [(q, d) for q, d in self.calls if len(d) > 1]


def _vec(cosines: dict[int, float]) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    for axis, c in cosines.items():
        v[axis] = c
    v[_RESIDUAL] = math.sqrt(max(0.0, 1.0 - sum(c * c for c in cosines.values())))
    return v


def _put_vector(store: MemoryStore, mid: str, vec: np.ndarray) -> None:
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (np.asarray(vec, dtype=np.float32).tobytes(), _EMBEDDER_ID, mid),
    )
    store._conn.commit()  # noqa: SLF001


def _mem(
    store: MemoryStore, content: str, cosines: dict[int, float], memory_type: str = "event"
) -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="d")
    store.create(m)
    _put_vector(store, m.id, _vec(cosines))
    return m


def _install(monkeypatch: pytest.MonkeyPatch, embedder: _AxisEmbedder, reranker=None) -> None:
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: embedder)
    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, _EMBEDDER_ID)
    if reranker is not None:
        monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: reranker)


def _budget(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    real = tunables.get_tunable

    def _get(key, default):
        return seconds if key == "reranker.latency_budget_seconds" else real(key, default)

    monkeypatch.setattr(tunables, "get_tunable", _get)


def _spy_rank_and_gate(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    real = sr.rank_and_gate

    def _spy(store, query, pool, coarse, **kwargs):
        calls.append({"query": query, "coarse": list(coarse), **kwargs})
        return real(store, query, pool, coarse, **kwargs)

    monkeypatch.setattr(sr, "rank_and_gate", _spy)
    return calls


def _ids(memories) -> list[str]:
    return [m.id for m in memories]


def _result_ids(result: SemanticRecallResult) -> list[str]:
    return _ids([*result.full, *result.snippet])


def _cal_queries(store: MemoryStore) -> list[tuple[str, str]]:
    return [
        (row[0], row[1])
        for row in store._conn.execute(  # noqa: SLF001
            "SELECT query, score_scale FROM calibration_log ORDER BY id"
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# Splitting (spec §3, S10, S29, S34, S53)
# ---------------------------------------------------------------------------


def test_split_uses_splitlines_and_drops_blank_lines() -> None:
    text = "red kite\r\n\r\n   \ngrey heron\rblue tit\n\n"
    assert split_paragraphs(text, None) == ["red kite", "grey heron", "blue tit"]


def test_a_message_with_no_line_break_is_one_paragraph_however_many_sentences() -> None:
    text = "The kite flew. The heron waited! Did the tit sing?"
    assert split_paragraphs(text, {"kite", "heron"}) == [text]


def test_keyword_empty_paragraphs_are_dropped_and_a_known_name_paragraph_is_kept() -> None:
    words = {"kite", "harbour", "bo"}
    text = "ok\nlol\nbo\nthe kite over the harbour"
    assert split_paragraphs(text, words) == ["bo", "the kite over the harbour"]


def test_one_non_blank_paragraph_is_kept_even_without_a_keyword_word() -> None:
    assert split_paragraphs("\n\nok\n", {"kite"}) == ["ok"]


def test_the_recall_block_passes_known_name_words_so_a_name_only_paragraph_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tested after name protection (spec §3): "bo" is a 2-letter lower-case
    word the keyword selector drops, but it is a known name here, so its
    paragraph survives the emptiness test."""
    store = MemoryStore(tmp_path / "memories.db")
    captured: list = []

    def _spy(store_, persona_dir, user_input, *, keyword_words=None):
        captured.append(keyword_words)
        return None

    monkeypatch.setattr("brain.chat.prompt.names_in", lambda persona_dir, text: ["bo"])
    monkeypatch.setattr("brain.chat.prompt.run_semantic_recall", _spy)
    from brain.chat.prompt import _build_recall_block

    message = "ok\nbo\nthe kite over the harbour"
    _build_recall_block(store, message, persona_dir=tmp_path)

    (words,) = captured
    assert "bo" in words
    assert split_paragraphs(message, words) == ["bo", "the kite over the harbour"]


# ---------------------------------------------------------------------------
# C9a: one paragraph behaves as before (pool equal to the per-row computation)
# ---------------------------------------------------------------------------


def _random_store(monkeypatch, tmp_path, embedder, n=120, family_every=3, seed=7):
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    rng = random.Random(seed)
    memories = []
    for i in range(n):
        raw = np.array([rng.gauss(0.0, 1.0) for _ in range(_DIM)], dtype=np.float32)
        memory_type = "monologue" if i % family_every == 0 else "event"
        m = Memory.create_new(content=f"r6 random memory {i}", memory_type=memory_type, domain="d")
        store.create(m)
        _put_vector(store, m.id, raw / np.linalg.norm(raw))
        memories.append(m)
    return store, memories


def _base_coarse_cut(store: MemoryStore, embedder: _AxisEmbedder, query: str):
    """Today's per-row computation (the base code's pool, scan and cut)."""
    pool = build_semantic_candidate_pool(store, build_embedding_matrix(store.db_path))
    qv = embedder.embed(query)
    scored = [(mid, cosine_similarity(qv, vec)) for mid, (_, vec) in pool.items()]
    return genuine_first_coarse_cut(scored, pool)


def test_one_paragraph_embeds_once_and_its_pool_equals_the_per_row_computation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = "The kite flew over the harbour. Then it came down."
    embedder = _AxisEmbedder({}, scale=3.0)  # not unit length: the scan must normalize it
    store, memories = _random_store(monkeypatch, tmp_path, embedder)
    rec = _Reranker({m.content: 5.0 for m in memories})
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    calls = _spy_rank_and_gate(monkeypatch)

    result = run_semantic_recall(store, tmp_path, message)

    assert embedder.batches == [[message]], "one embed call with the whole message"
    (call,) = calls
    base = _base_coarse_cut(store, embedder, message)
    assert [mid for mid, _ in call["coarse"]] == [mid for mid, _ in base]
    assert [c for _, c in call["coarse"]] == pytest.approx([c for _, c in base], abs=1e-5)
    assert call["query"] == message
    assert result is not None and result.paragraphs[0].query == message
    assert {q for q, _ in rec.scored()} == {message}, "the rerank query is the whole message"
    assert _cal_queries(store) == [(message, CALIBRATION_SCORE_SCALE)]


@pytest.mark.parametrize("text", ["\r\n  \nkite over the harbour\r\n\n", "\rkite over the harbour\r\r"])
def test_blank_lines_around_one_paragraph_leave_one_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str
) -> None:
    embedder = _AxisEmbedder({"kite over the harbour": 0})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    _mem(store, "a kite memory", {0: 0.9})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, text)

    assert embedder.batches == [["kite over the harbour"]]
    assert result is not None
    assert len(result.paragraphs) == 1 and result.whole_message_fallback is False


# ---------------------------------------------------------------------------
# C9a second half (S61, P-26, P-28): the budget accounting, scripted clock
# ---------------------------------------------------------------------------


def _slow_store_reads(monkeypatch: pytest.MonkeyPatch, clock: _Clock, seconds: float) -> None:
    """The lean pool read is part of `T_m`: it advances the clock."""
    real = MemoryStore.active_state_memory_types

    def _slow(self):
        clock.advance(seconds)
        return real(self)

    monkeypatch.setattr(MemoryStore, "active_state_memory_types", _slow)


def _budget_case(monkeypatch, tmp_path, *, embed_seconds, overhead=0.25, probe=0.125, warm=10.0):
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    embedder = _AxisEmbedder({"kite over the harbour": 0}, clock, embed_seconds)
    store = MemoryStore(tmp_path / "memories.db")
    memories = [_mem(store, f"r6 budget memory {i}", {0: 0.9 - 0.01 * i}) for i in range(6)]
    rec = _Reranker(
        {m.content: 5.0 for m in memories}, clock=clock, warm_seconds=warm, probe_seconds=probe
    )
    _install(monkeypatch, embedder, rec)
    _slow_store_reads(monkeypatch, clock, overhead)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    return store, rec, clock


def test_one_paragraph_pays_its_embed_scan_and_pair_size_time_out_of_the_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S61: no exemption for P = 1. Budget 4 s, `T_m` 0.25 s (the pool read),
    embed 0.5 s (the embed cost series' prediction for this one text, S108),
    scan 0.125 s (the pair-size probe); the 10 s warm-up is one-time and not
    counted. Share = 4 - 0.25 - 0.5 - 0.125."""
    _budget(monkeypatch, 4.0)
    store, rec, clock = _budget_case(monkeypatch, tmp_path, embed_seconds=0.5)
    calls = _spy_rank_and_gate(monkeypatch)

    result = run_semantic_recall(store, tmp_path, "kite over the harbour")

    assert result is not None
    (call,) = calls
    assert call["budget_seconds"] == pytest.approx(3.125)
    assert result.rerank_budget == pytest.approx(3.125)
    assert result.budget == pytest.approx(4.0)
    assert sr._message_overhead.mean(store, "") == pytest.approx(0.25)
    assert sr._scan_time.mean(store, "") == pytest.approx(0.125)
    text = "kite over the harbour"
    assert sr._embed_cost.fit(store, _EMBEDDER_ID).predict(len(text)) == pytest.approx(0.5)


def test_the_rerank_budget_uses_the_running_averages_not_this_messages_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P-28: `B - T_m - P x T_p`, both running averages (ratio of sums). A
    second, slower message (embed 0.75 s after 0.25 s) is budgeted with the
    average `T_p` 0.5 s, not its own 0.75 s."""
    _budget(monkeypatch, 4.0)
    store, rec, clock = _budget_case(monkeypatch, tmp_path, embed_seconds=0.25, probe=0.0)
    calls = _spy_rank_and_gate(monkeypatch)
    run_semantic_recall(store, tmp_path, "kite over the harbour")
    embedder = _AxisEmbedder({"kite over the harbour": 0}, clock, 0.75)
    _install(monkeypatch, embedder)

    run_semantic_recall(store, tmp_path, "kite over the harbour")

    assert [c["budget_seconds"] for c in calls] == pytest.approx([4 - 0.25 - 0.25, 4 - 0.25 - 0.5])


def _threshold_budget(rec: _Reranker, store: MemoryStore, query: str, docs: list[str]) -> float:
    """The smallest rerank share at which 5 real candidates fit the cost model."""
    estimate = reranker_mod.rerank_cost_estimate(_RERANKER_ID, store)
    lengths = rec.pair_token_lengths(query, [*docs, *ANCHOR_POOL])
    cand, anchors = lengths[: len(docs)], lengths[len(docs) :]
    k = reranker_mod.anchor_count(5)
    cost5 = reranker_mod.prefix_cost(estimate, 5 + k, max([*cand[:5], *anchors[:k]]))
    assert reranker_mod.fit_rerank_width(cand, anchors, estimate, cost5, None) >= 5
    assert reranker_mod.fit_rerank_width(cand, anchors, estimate, cost5 - 1e-6, None) < 5
    return cost5


@pytest.mark.parametrize(("embed_seconds", "path"), [(0.0, RERANKED_PATH), (0.5, COSINE_PATH)])
def test_the_embed_and_scan_time_can_push_one_paragraph_onto_the_cosine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, embed_seconds: float, path: str
) -> None:
    """S61: with the embed time subtracted the share no longer fits 5 real
    candidates, so the message takes the cosine path; the same message with
    zero embed time reranks."""
    store, rec, clock = _budget_case(
        monkeypatch, tmp_path, embed_seconds=embed_seconds, probe=0.0, warm=0.0
    )
    reranker_mod._record_rerank_cost(_RERANKER_ID, store, 1_000, 1.0, None)
    docs = [f"r6 budget memory {i}" for i in range(6)]
    cost5 = _threshold_budget(rec, store, "kite over the harbour", docs)
    _budget(monkeypatch, 0.25 + cost5 + 0.25)

    result = run_semantic_recall(store, tmp_path, "kite over the harbour")

    assert result is not None and result.path == path
    reranked = result.paragraphs[0].gated.real_width
    assert (reranked >= 5) if path == RERANKED_PATH else (reranked == 0)


# ---------------------------------------------------------------------------
# C9b: whole-message fallback; the first message of a process (S63)
# ---------------------------------------------------------------------------

_P1, _P2, _P3 = "alpha kite notes", "beta heron notes", "gamma tit notes"
_THREE = f"{_P1}\n{_P2}\n{_P3}"
_REST = f"{_P2}\n{_P3}"


def test_paragraphs_that_do_not_fit_are_searched_as_one_whole_message_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _budget(monkeypatch, 4.0)
    embedder = _AxisEmbedder({_P1: 0, _P2: 1, _P3: 2})
    store = MemoryStore(tmp_path / "memories.db")
    sr._embed_cost.record(store, _EMBEDDER_ID, 100, 0.0)  # warm, embedding costs nothing
    sr._scan_time.record(store, "", 1, 3.0)  # 3 x 3 s > 4 s
    _install(monkeypatch, embedder)
    _mem(store, "a memory of the whole message", {_OTHER: 0.9})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)
    calls = _spy_rank_and_gate(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_THREE]], "exactly one embed, of the whole message"
    assert [c["query"] for c in calls] == [_THREE]
    assert result is not None
    assert result.whole_message_fallback is True and len(result.paragraphs) == 1


def _two_query_store(monkeypatch, tmp_path, clock, seconds_per_text):
    embedder = _AxisEmbedder({_P1: 0, _P2: 1, _P3: 2}, clock, seconds_per_text)
    store = MemoryStore(tmp_path / "memories.db")
    first = [_mem(store, f"r6 first memory {i}", {0: 0.9 - 0.01 * i, _OTHER: 0.3}) for i in range(6)]
    rest = [_mem(store, f"r6 rest memory {i}", {_OTHER: 0.9 - 0.01 * i, 0: 0.3}) for i in range(6)]
    rec = _Reranker({m.content: 5.0 for m in [*first, *rest]})
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    _warm(store)
    return store, embedder, rec, first, rest


_REST_CUT = "beta heron"  # _REST cut to what fits at budget 2.9 s with the first paragraph at 1.5 s


def test_the_first_message_rest_query_is_sized_and_its_probe_is_scan_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S63/S199 path, round-2 review N3: the rest query's token count is probed
    (its embed call becomes an embed-series sample) and the probe's time is the
    rest query's scan-side work. The rest is cut to what the first paragraph's
    seconds per token leave of the budget: budget 2.9 s, P1 (16 chars) 1.5 s,
    so 14 tokens fit (a word-boundary prefix: "beta heron", 10). Unequal per-call
    seconds (P1 1.5 s, the cut 2.5 s)."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 2.9)  # 3 x 1.5 s and 2 x 1.5 s do not fit: the S199 split
    store, embedder, rec, first, rest = _two_query_store(monkeypatch, tmp_path, clock, 0.0)
    seconds = {_P1: 1.5, _REST_CUT: 2.5}

    def embed_batch(texts):
        embedder.batches.append(list(texts))
        clock.advance(sum(seconds.get(t, 0.0) for t in texts))
        return [embedder.embed(t) for t in texts]

    def token_lengths_timed(texts):
        if texts == [_REST]:
            clock.advance(0.05)
        return [len(t) for t in texts], None

    monkeypatch.setattr(embedder, "embed_batch", embed_batch)
    monkeypatch.setattr(embedder, "token_lengths_timed", token_lengths_timed)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert result is not None and [o.query for o in result.paragraphs] == [_P1, _REST_CUT]
    state = sr._embed_cost.state(store, _EMBEDDER_ID)
    assert state.sums.n == 2 and state.sums.sum_x == len(_P1) + len(_REST_CUT)
    assert state.sums.sum_y == pytest.approx(4.0)
    assert sr._scan_time.state(store, "").sums.sum_y == pytest.approx(0.05), (
        "the rest probe is the rest query's scan time"
    )


def test_a_first_message_that_does_not_fit_is_scored_as_its_first_paragraph_and_the_cut_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S63/S115/S199: no measurement yet; the first paragraph is embedded and timed
    alone (1.5 s); 3 x 1.5 s and 2 x 1.5 s (paragraphs 1 and 3) do not fit 2.9 s,
    so the rest is ONE query cut to what fits and the two are scored as two
    paragraphs: own candidates (a memory found by both goes to its better cosine).
    The cut takes the rest of the budget, so neither query has a rerank share and
    both take the cosine path."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 2.9)
    store, embedder, rec, first, rest = _two_query_store(monkeypatch, tmp_path, clock, 1.5)
    calls = _spy_rank_and_gate(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1], [_REST_CUT]]
    assert result is not None
    assert len(result.paragraphs) == 2 and result.whole_message_fallback is False
    assert [o.query for o in result.paragraphs] == [_P1, _REST_CUT]
    by_query = {c["query"]: c for c in calls}
    assert set(by_query) == {_P1, _REST_CUT}
    assert [mid for mid, _ in by_query[_P1]["coarse"]] == _ids(first)
    assert [mid for mid, _ in by_query[_REST_CUT]["coarse"]] == _ids(rest)
    assert result.rerank_budget == pytest.approx(0.0)
    assert rec.scored() == [], "the cut used the budget: the cosine path serves both"


def test_a_first_message_whose_alternate_paragraphs_fit_searches_paragraphs_one_and_three(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S199: every paragraph does not fit (3 x 1.5 s > 4.0 s) but paragraphs 1 and
    3 do (2 x 1.5 s): the first is timed alone, the other alternate follows."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 4.0)
    store, embedder, rec, first, rest = _two_query_store(monkeypatch, tmp_path, clock, 1.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1], [_P3]]
    assert result is not None
    assert [o.query for o in result.paragraphs] == [_P1, _P3]
    assert result.whole_message_fallback is False


def test_a_first_message_whose_rest_fits_is_scored_as_two_paragraphs_each_reranked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S63/S199: the first paragraph is long (16 x 8 characters, 1.5 s), the rest is
    short and fits what the first paragraph's seconds per character leave of the
    budget uncut, and its embed is quick: the two are scored as two paragraphs, each
    reranked against its own text with its own anchors, from equal shares."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 2.9)  # 3 x 1.5 s and 2 x 1.5 s do not fit
    long_first = "alpha kite notes " * 8
    rest_text = "beta heron\ngamma tit"
    message = f"{long_first.strip()}\n{rest_text}"
    embedder = _AxisEmbedder({long_first.strip(): 0}, clock, 0.0)
    store = MemoryStore(tmp_path / "memories.db")
    first = [_mem(store, f"s63 first memory {i}", {0: 0.9 - 0.01 * i, _OTHER: 0.3}) for i in range(6)]
    rest = [_mem(store, f"s63 rest memory {i}", {_OTHER: 0.9 - 0.01 * i, 0: 0.3}) for i in range(6)]
    rec = _Reranker({m.content: 5.0 for m in [*first, *rest]})
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    _warm(store)
    seconds = {long_first.strip(): 1.5, rest_text: 0.05}

    def embed_batch(texts):
        embedder.batches.append(list(texts))
        clock.advance(sum(seconds.get(t, 0.0) for t in texts))
        return [embedder.embed(t) for t in texts]

    monkeypatch.setattr(embedder, "embed_batch", embed_batch)
    calls = _spy_rank_and_gate(monkeypatch)

    result = run_semantic_recall(store, tmp_path, message)

    assert embedder.batches == [[long_first.strip()], [rest_text]]
    assert result is not None
    assert [o.query for o in result.paragraphs] == [long_first.strip(), rest_text]
    assert result.whole_message_fallback is False
    by_query = {c["query"]: c for c in calls}
    assert by_query[long_first.strip()]["budget_seconds"] == pytest.approx(
        by_query[rest_text]["budget_seconds"]
    )
    assert by_query[long_first.strip()]["budget_seconds"] > 0
    scored = rec.scored()
    assert [q for q, _ in scored] == [long_first.strip(), rest_text], "one rerank per query"
    for _, docs in scored:
        assert set(docs) & set(ANCHOR_POOL), "each rerank carries its own anchors"


def test_two_paragraphs_that_do_not_both_fit_search_only_the_first_warm_and_first_ever(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S195 literal reading with two paragraphs: the alternates are [paragraph 1],
    so paragraph 2 is searched only when both fit."""
    store, embedder = _warm_decision_case(monkeypatch, tmp_path, scan=2.0, overhead=0.0, budget=3.5)
    result = run_semantic_recall(store, tmp_path, f"{_P1}\n{_P2}")
    assert embedder.batches == [[_P1]]
    assert result is not None and [o.query for o in result.paragraphs] == [_P1]


def test_a_first_message_where_not_even_a_token_of_the_rest_fits_searches_the_first_paragraph_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S199: the first paragraph alone already uses the budget, so the rest has no
    room: only the first paragraph is searched."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 1.0)
    store, embedder, rec, first, rest = _two_query_store(monkeypatch, tmp_path, clock, 1.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1]]
    assert result is not None and [o.query for o in result.paragraphs] == [_P1]


def test_a_first_message_that_fits_searches_every_paragraph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 4.0)
    store, embedder, rec, first, rest = _two_query_store(monkeypatch, tmp_path, clock, 0.25)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1], [_P2, _P3]], "the first timed alone, the rest in one batch"
    assert result is not None
    assert [o.query for o in result.paragraphs] == [_P1, _P2, _P3]
    assert result.whole_message_fallback is False


def test_a_memory_found_by_two_paragraphs_goes_to_its_better_cosine_keeping_that_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S11/S53. Warm `T_p`, both paragraphs searched; X is 0.6 to P1 and 0.7
    to P2 (in both pools) and is scored once, for P2, at 0.7."""
    embedder = _AxisEmbedder({_P1: 0, _P2: 1})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    x = _mem(store, "shared memory", {0: 0.6, 1: 0.7})
    only1 = _mem(store, "first only", {0: 0.8})
    only2 = _mem(store, "second only", {1: 0.5})
    _cosine_floor(store, 0.1)
    _no_reranker(monkeypatch)
    calls = _spy_rank_and_gate(monkeypatch)

    run_semantic_recall(store, tmp_path, f"{_P1}\n{_P2}")

    by_query = {c["query"]: dict(c["coarse"]) for c in calls}
    assert set(by_query[_P1]) == {only1.id}
    assert set(by_query[_P2]) == {x.id, only2.id}
    assert by_query[_P2][x.id] == pytest.approx(0.7, abs=1e-5)


# ---------------------------------------------------------------------------
# C9d: allocation above 50 (S58)
# ---------------------------------------------------------------------------


def _reference_allocation(fitted: list[tuple[int, float | None]]) -> list[int]:
    """An independent loop over the S58 text: serve by best cosine score
    (ties by index; no candidate last), each takes min(fit, 50 - used) if >= 5."""
    order = sorted(
        range(len(fitted)),
        key=lambda i: (fitted[i][1] is None, -(fitted[i][1] or 0.0), i),
    )
    out = [0] * len(fitted)
    used = 0
    for i in order:
        give = min(fitted[i][0], 50 - used)
        if give >= 5:
            out[i] = give
            used += give
    return out


def test_allocation_worked_case_twelve_paragraphs_fitting_eight_each() -> None:
    bests = [0.31, 0.92, 0.45, 0.77, 0.12, 0.66, 0.58, 0.83, 0.25, 0.71, 0.39, 0.95]
    widths = allocate_widths([(8, b) for b in bests])
    served = sorted(range(12), key=lambda i: -bests[i])[:6]
    assert widths == [8 if i in served else 0 for i in range(12)]
    assert sum(widths) == 48, "2 left is below 5: the rest take the cosine path"
    assert widths == _reference_allocation([(8, b) for b in bests])


def test_allocation_keeps_every_share_when_they_sum_to_fifty_or_less() -> None:
    assert allocate_widths([(20, 0.1), (20, 0.9), (10, 0.5)]) == [20, 20, 10]
    assert allocate_widths([(3, 0.9), (7, 0.1)]) == [0, 7], "below 5 is the cosine path"


def test_allocation_serves_by_best_cosine_not_by_position_ties_by_index() -> None:
    assert allocate_widths([(30, 0.2), (30, 0.8)]) == [20, 30]
    assert allocate_widths([(30, 0.5), (30, 0.5)]) == [30, 20]
    assert allocate_widths([(48, 0.9), (30, 0.8), (30, None)]) == [48, 0, 0]


def test_allocation_matches_the_reference_loop_on_random_inputs() -> None:
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(1, 14)
        fitted = [
            (rng.randint(0, 50), None if rng.random() < 0.1 else round(rng.random(), 2))
            for _ in range(n)
        ]
        widths = allocate_widths(fitted)
        assert widths == _reference_allocation(fitted)
        assert sum(widths) <= 50


def test_the_message_never_reranks_more_than_fifty_real_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """8 paragraphs of 8 candidates each, all of which fit: 64 > 50, so the 6
    with the best cosine get 8 (48) and the other 2 (2 left) take the cosine
    path. Best-cosine order differs from paragraph order."""
    bests = [0.61, 0.95, 0.70, 0.88, 0.52, 0.79, 0.66, 0.90]
    texts = [f"paragraph {chr(97 + i)} words" for i in range(8)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = []
    for i, best in enumerate(bests):
        for j in range(8):
            m = _mem(store, f"r6 para {i} memory {j}", {i: best - 0.001 * j})
            contents.append(m.content)
    rec = _Reranker(dict.fromkeys(contents, 5.0))
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.1)
    _warm(store)

    search = search_paragraphs(store, "\n".join(texts), log_calibration=False)

    assert search is not None
    served = sorted(range(8), key=lambda i: -bests[i])[:6]
    widths = [o.gated.real_width for o in search.outcomes]
    assert widths == [8 if i in served else 0 for i in range(8)]
    assert search.total_width == 48
    assert {q for q, _ in rec.scored()} == {texts[i] for i in served}


# ---------------------------------------------------------------------------
# C8b: guaranteed slots
# ---------------------------------------------------------------------------

_LAKE, _GARDEN, _RIVER = "lake trip words", "garden notes words", "river walk words"


def _guarantee_store(monkeypatch, tmp_path, *, p2_type="event", with_p3=False):
    embedder = _AxisEmbedder({_LAKE: 0, _GARDEN: 1, _RIVER: 2})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    p1 = [_mem(store, f"r6 lake memory {i}", {0: 0.9 - 0.01 * i}) for i in range(12)]
    p2 = [_mem(store, f"r6 garden memory {i}", {1: c}, p2_type) for i, c in enumerate([0.8, 0.7, 0.6])]
    scores = {m.content: 21.0 - i for i, m in enumerate(p1)}
    p3 = []
    if with_p3:
        p3 = [_mem(store, f"r6 river memory {i}", {2: 0.9 - 0.01 * i}) for i in range(6)]
        scores.update({m.content: 2.5 - 0.1 * i for i, m in enumerate(p3)})
    rec = _Reranker(scores)
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.5)
    _warm(store)
    return store, p1, p2, p3


def test_a_cosine_path_paragraphs_best_genuine_memory_heads_the_cosine_genuine_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P1 reranks 12 floor-clearing results; P2 (3 candidates) takes the
    cosine path. The 9 are P1's 8 best, then P2's best genuine memory, at the
    head of the cosine-path genuine group; P2's others are not kept."""
    store, p1, p2, _ = _guarantee_store(monkeypatch, tmp_path)

    result = run_semantic_recall(store, tmp_path, f"{_LAKE}\n{_GARDEN}")

    assert result is not None
    assert _result_ids(result) == _ids(p1[:8]) + [p2[0].id]
    assert [h.path for h in result.hits] == [RERANKED_PATH] * 8 + [COSINE_PATH]
    assert [h.paragraph for h in result.hits] == [0] * 8 + [1]
    assert result.guaranteed_ids == (p1[0].id, p2[0].id)


def test_every_reranked_paragraphs_best_genuine_memory_is_in_the_nine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P3 reranks too, every score below P1's: its best still takes a slot."""
    store, p1, p2, p3 = _guarantee_store(monkeypatch, tmp_path, with_p3=True)

    result = run_semantic_recall(store, tmp_path, f"{_LAKE}\n{_GARDEN}\n{_RIVER}")

    assert result is not None
    assert _result_ids(result) == _ids(p1[:7]) + [p3[0].id, p2[0].id]


def test_a_cosine_path_paragraph_with_only_family_hits_gets_no_guaranteed_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, p1, p2, _ = _guarantee_store(monkeypatch, tmp_path, p2_type="monologue")

    result = run_semantic_recall(store, tmp_path, f"{_LAKE}\n{_GARDEN}")

    assert result is not None
    assert _result_ids(result) == _ids(p1[:9])
    assert result.guaranteed_ids == (p1[0].id,)


# ---------------------------------------------------------------------------
# C8c: guarantees over the cap (S64)
# ---------------------------------------------------------------------------


def test_eleven_paragraphs_over_the_passive_cap_keep_reranked_guarantees_then_cosine_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """6 reranked paragraphs (normalized bests 0.2-0.7) and 5 cosine-path ones
    (cosine bests 0.80-0.90, numerically HIGHER, never compared): the 9 kept
    are the 6 reranked guarantees by normalized score, then the 3 best
    cosine-path guarantees by cosine; the other 2 guarantees are dropped and
    no non-guaranteed result is kept."""
    texts = [f"para {chr(97 + i)} words" for i in range(11)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    scores: dict[str, float] = {}
    reranked_bests, cosine_bests = [], []
    for i in range(6):
        best = 0.2 + 0.1 * i
        members = [_mem(store, f"r6 rr {i} {j}", {i: 0.9 - 0.01 * j}) for j in range(5)]
        for j, m in enumerate(members):
            scores[m.content] = best if j == 0 else best - 0.05 - 0.01 * j
        reranked_bests.append(members[0])
    for k in range(5):
        best = 0.80 + 0.025 * k
        members = [_mem(store, f"r6 cos {k} {j}", {6 + k: best - 0.01 * j}) for j in range(2)]
        cosine_bests.append(members[0])
    rec = _Reranker(scores)
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 0.1)
    _cosine_floor(store, 0.5)
    _warm(store)

    result = run_semantic_recall(store, tmp_path, "\n".join(texts))

    assert result is not None
    expected = _ids(reversed(reranked_bests)) + _ids(list(reversed(cosine_bests))[:3])
    assert _result_ids(result) == expected


def _hand_search(outcomes: list[tuple[str, list[tuple[Memory, float]], float, list[tuple[Memory, float]] | None]]):
    """A `ParagraphSearch` from `(path, [(memory, score)], pass_mark, tail)`
    per paragraph (tail: cosine-scale family results, pass mark 0.5)."""
    pool: dict[str, tuple[Memory, np.ndarray]] = {}
    built = []
    for index, (path, ranked, pass_mark, tail) in enumerate(outcomes):
        for m, _ in [*ranked, *(tail or [])]:
            pool[m.id] = (m, np.zeros(1, dtype=np.float32))
        tail_ranking = (
            None
            if tail is None
            else GatedRanking(COSINE_PATH, COSINE_SCORE_SCALE, [(m.id, s) for m, s in tail], 0.5)
        )
        gated = GatedRanking(
            path,
            CALIBRATION_SCORE_SCALE if path == RERANKED_PATH else COSINE_SCORE_SCALE,
            [(m.id, s) for m, s in ranked],
            pass_mark,
            tail=tail_ranking,
        )
        built.append(ParagraphOutcome(index, f"p{index}", gated, len(ranked), None))
    return ParagraphSearch(tuple(built), pool, False, 0, 4.0, 4.0)


def _m(label: str, memory_type: str = "event") -> Memory:
    return Memory.create_new(content=label, memory_type=memory_type, domain="d")


def test_a_paragraph_whose_best_genuine_memory_misses_its_floor_has_no_guarantee() -> None:
    """Cap 3: three reranked guarantees fill it; a fourth paragraph's genuine
    memory is below its floor (no guarantee, no slot consumed); a cosine-path
    guarantee beyond the cap is dropped, not squeezed in."""
    a, a2, b, c, low, cos = _m("a"), _m("a2"), _m("b"), _m("c"), _m("low"), _m("cos")
    search = _hand_search(
        [
            (RERANKED_PATH, [(a, 0.9), (a2, 0.85)], 0.1, None),
            (RERANKED_PATH, [(b, 0.3)], 0.1, None),
            (RERANKED_PATH, [(low, 0.05)], 0.1, None),
            (RERANKED_PATH, [(c, 0.2)], 0.1, None),
            (COSINE_PATH, [(cos, 0.99)], 0.5, None),
        ]
    )
    assert [h.memory_id for h in assemble_paragraph_results(search, 3)] == [a.id, b.id, c.id]
    assert [h.memory_id for h in assemble_paragraph_results(search, 5)] == [
        a.id,
        a2.id,
        b.id,
        c.id,
        cos.id,
    ]


def test_the_tool_with_limit_three_keeps_the_three_best_cosine_path_guarantees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S64 through `search_memories`: 5 paragraphs, each on the cosine path
    with a floor-clearing best genuine memory; `limit=3` keeps the three
    paragraphs with the best cosine, their bests in cosine order; each
    paragraph's second memory outscores other paragraphs' bests but is not
    guaranteed and is not kept."""
    bests = [0.70, 0.95, 0.80, 0.90, 0.60]
    texts = [f"topic {chr(97 + i)} words" for i in range(5)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    firsts = []
    for i, best in enumerate(bests):
        firsts.append(_mem(store, f"r6 tool {i} first", {i: best}))
        _mem(store, f"r6 tool {i} second", {i: best - 0.01})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    out = dispatch(
        "search_memories",
        {"query": "\n".join(texts), "limit": 3},
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )

    assert out["mode"] == "semantic"
    assert [m["id"] for m in out["memories"]] == [firsts[1].id, firsts[3].id, firsts[2].id]


# ---------------------------------------------------------------------------
# S86: family tails regrouped across paragraphs and de-duplicated
# ---------------------------------------------------------------------------


def test_family_tails_are_regrouped_after_all_cosine_path_genuine_results_and_deduplicated() -> None:
    rg0, rm0, rg1, cg, fam_c = _m("rg0"), _m("rm0", "monologue"), _m("rg1"), _m("cg"), _m("fc", "monologue")
    tail_a, tail_b = _m("ta", "monologue_trace"), _m("tb", "monologue")
    search = _hand_search(
        [
            (RERANKED_PATH, [(rg0, 3.0), (rm0, 2.0)], 1.0, [(tail_a, 0.7), (tail_b, 0.95)]),
            (COSINE_PATH, [(cg, 0.6), (fam_c, 0.8)], 0.5, None),
            (RERANKED_PATH, [(rg1, 4.0)], 1.0, [(tail_b, 0.9)]),
        ]
    )
    hits = assemble_paragraph_results(search, 20)
    assert [h.memory_id for h in hits] == [rg1.id, rg0.id, rm0.id, cg.id, tail_b.id, fam_c.id, tail_a.id]
    assert [h.group for h in hits] == [
        RERANKED_GENUINE,
        RERANKED_GENUINE,
        RERANKED_FAMILY,
        COSINE_GENUINE,
        COSINE_FAMILY,
        COSINE_FAMILY,
        COSINE_FAMILY,
    ]


# ---------------------------------------------------------------------------
# S97 / S98: six result groups, one editable list, passive and tool alike
# ---------------------------------------------------------------------------

_S97_ORDER = (
    RERANKED_GENUINE,
    RERANKED_FAMILY,
    COSINE_GENUINE,
    KEYWORD_GENUINE,
    COSINE_FAMILY,
    KEYWORD_FAMILY,
)


def _six(store: MemoryStore | None = None):
    labels = {
        RERANKED_GENUINE: ("rg", "event"),
        RERANKED_FAMILY: ("rm", "monologue"),
        COSINE_GENUINE: ("cg", "event"),
        COSINE_FAMILY: ("cm", "monologue"),
    }
    semantic = [(_m(lab, t), group) for group, (lab, t) in labels.items()]
    kg, km = _m("kg"), _m("km", "monologue")
    return semantic, [kg, km]


def test_merge_orders_the_six_groups_by_the_list() -> None:
    semantic, keyword = _six()
    merged = merge_with_keyword(semantic, keyword, cap=9)
    assert [m.content for m in merged] == ["rg", "rm", "cg", "kg", "cm", "km"]


def test_editing_the_list_changes_the_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_constants, "RECALL_RESULT_GROUP_ORDER", tuple(reversed(_S97_ORDER)))
    semantic, keyword = _six()
    merged = merge_with_keyword(semantic, keyword, cap=9)
    assert [m.content for m in merged] == ["km", "cm", "kg", "cg", "rm", "rg"]


def test_a_list_that_does_not_name_each_group_once_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_constants, "RECALL_RESULT_GROUP_ORDER", _S97_ORDER[:5] + (RERANKED_GENUINE,))
    with pytest.raises(ValueError, match="RECALL_RESULT_GROUP_ORDER"):
        order_result_groups({}, 9)


def test_a_non_reranked_family_memory_never_takes_a_slot_ahead_of_a_keyword_genuine_hit() -> None:
    rg = [(_m(f"rg{i}"), RERANKED_GENUINE) for i in range(3)]
    cm = [(_m(f"cm{i}", "monologue"), COSINE_FAMILY) for i in range(6)]
    kg = [_m(f"kg{i}") for i in range(6)]
    merged = merge_with_keyword([*rg, *cm], kg, cap=9)
    assert [m.content for m in merged] == [f"rg{i}" for i in range(3)] + [f"kg{i}" for i in range(6)]


def test_guaranteed_semantic_results_are_kept_first_and_a_semantic_duplicate_wins() -> None:
    rg = [(_m(f"rg{i}"), RERANKED_GENUINE) for i in range(9)]
    head = _m("cg-head")
    kg = [_m("kg0"), rg[0][0]]
    merged = merge_with_keyword([*rg, (head, COSINE_GENUINE)], kg, cap=9, guaranteed=[rg[0][0].id, head.id])
    assert [m.content for m in merged] == [f"rg{i}" for i in range(8)] + ["cg-head"]
    merged = merge_with_keyword(rg[:2], kg, cap=9)
    assert [m.content for m in merged] == ["rg0", "rg1", "kg0"], "found by both: shown once, as semantic"


def _s97_passive_case(tmp_path: Path):
    store = MemoryStore(":memory:")
    reranked = Memory.create_new(content="a reranked genuine memory", memory_type="event", domain="us")
    tail = Memory.create_new(content="an unreranked monologue memory", memory_type="monologue", domain="us")
    for m in (reranked, tail):
        store.create(m)
    keyword = Memory.create_new(content=f"{_LONG} (keyword)", memory_type="conversation", domain="us")
    store.create(keyword)
    semantic = SemanticRecallResult(
        full=[reranked, tail],
        snippet=[],
        scores={reranked.id: 5.0},
        hits=[
            SemanticHit(reranked, 5.0, RERANKED_PATH, 0, False),
            SemanticHit(tail, 0.9, COSINE_PATH, 0, True),
        ],
    )
    return store, semantic, reranked, tail, keyword


def test_passive_recall_puts_keyword_genuine_hits_above_non_reranked_family_memories(
    tmp_path: Path,
) -> None:
    store, semantic, reranked, tail, keyword = _s97_passive_case(tmp_path)
    block = _render(store, "quokka photographs", tmp_path, semantic)
    ids = [mid for mid, _ in _active_rows(block)]
    assert ids == [reranked.id, keyword.id, tail.id]


def test_passive_recall_follows_an_edited_group_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        dev_constants,
        "RECALL_RESULT_GROUP_ORDER",
        (COSINE_FAMILY, RERANKED_GENUINE, RERANKED_FAMILY, COSINE_GENUINE, KEYWORD_GENUINE, KEYWORD_FAMILY),
    )
    store, semantic, reranked, tail, keyword = _s97_passive_case(tmp_path)
    block = _render(store, "quokka photographs", tmp_path, semantic)
    ids = [mid for mid, _ in _active_rows(block)]
    assert ids == [tail.id, reranked.id, keyword.id]


def test_a_family_result_moved_past_position_five_renders_as_a_snippet(tmp_path: Path) -> None:
    """Positions 1-5 render full: the non-reranked family memory was full in
    its semantic tier but lands at position 7 behind 5 keyword genuine hits."""
    from brain.chat.prompt import _assemble_active_entries

    store, semantic, reranked, tail, keyword = _s97_passive_case(tmp_path)
    keyword_hits = [_m(f"kw{i}") for i in range(5)]
    entries = _assemble_active_entries(semantic, keyword_hits, limit=8)
    assert [m.id for m, _ in entries] == [reranked.id, *_ids(keyword_hits), tail.id]
    assert [full for _, full in entries] == [True, True, True, True, True, False, False]


def _tool_s97(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[list[str], Memory, Memory, Memory]:
    import brain.tools.impls.search_memories as tool

    store = MemoryStore(":memory:")
    reranked = Memory.create_new(content="a reranked genuine memory", memory_type="event", domain="us")
    tail = Memory.create_new(content="an unreranked monologue memory", memory_type="monologue", domain="us")
    keyword = Memory.create_new(content="the quokka photographs", memory_type="conversation", domain="us")
    for m in (reranked, tail, keyword):
        store.create(m)
    semantic = tool._SemanticTopK(  # noqa: SLF001
        grouped=[(reranked, RERANKED_GENUINE), (tail, COSINE_FAMILY)], guaranteed=[reranked.id]
    )
    monkeypatch.setattr(tool, "_semantic_top_k", lambda *a, **kw: semantic)
    out = dispatch(
        "search_memories",
        {"query": "quokka photographs", "limit": 5},
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )
    assert out["mode"] == "semantic"
    return [m["id"] for m in out["memories"]], reranked, tail, keyword


def test_the_tool_puts_keyword_genuine_hits_above_non_reranked_family_memories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids, reranked, tail, keyword = _tool_s97(monkeypatch, tmp_path)
    assert ids == [reranked.id, keyword.id, tail.id]


def test_the_tool_follows_an_edited_group_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        dev_constants,
        "RECALL_RESULT_GROUP_ORDER",
        (COSINE_FAMILY, RERANKED_GENUINE, RERANKED_FAMILY, COSINE_GENUINE, KEYWORD_GENUINE, KEYWORD_FAMILY),
    )
    ids, reranked, tail, keyword = _tool_s97(monkeypatch, tmp_path)
    assert ids == [tail.id, reranked.id, keyword.id]


# ---------------------------------------------------------------------------
# C10 (R6 part): each paragraph's calibration row carries the paragraph
# ---------------------------------------------------------------------------


def test_each_paragraphs_rerank_logs_the_paragraph_as_its_query_and_the_tool_logs_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embedder = _AxisEmbedder({_P1: 0, _P2: 1})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = [_mem(store, f"r6 cal {axis} {i}", {axis: 0.9 - 0.01 * i}).content for axis in (0, 1) for i in range(6)]
    rec = _Reranker(dict.fromkeys(contents, 5.0))
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _warm(store)

    run_semantic_recall(store, tmp_path, f"{_P1}\n{_P2}")

    assert sorted(_cal_queries(store)) == sorted(
        [(_P1, CALIBRATION_SCORE_SCALE), (_P2, CALIBRATION_SCORE_SCALE)]
    )
    before = len(_cal_queries(store))
    dispatch(
        "search_memories",
        {"query": f"{_P1}\n{_P2}"},
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )
    assert len(_cal_queries(store)) == before


# ---------------------------------------------------------------------------
# P-27: the lean pool
# ---------------------------------------------------------------------------


def test_the_lean_pool_holds_the_same_candidates_as_the_list_active_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embedder = _AxisEmbedder({"kite words": 0})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    kept = [_mem(store, f"r6 lean {i}", {0: 0.5 + 0.01 * i}) for i in range(6)]
    fading = _mem(store, "r6 fading", {0: 0.99})
    store.fade(fading.id, summary="softened")
    gone = _mem(store, "r6 deactivated", {0: 0.98})
    store.deactivate(gone.id)
    unembedded = Memory.create_new(content="r6 no vector", memory_type="event", domain="d")
    store.create(unembedded)
    _cosine_floor(store, 0.1)
    _no_reranker(monkeypatch)
    before = {m.id: store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()[0] for m in kept}  # noqa: SLF001

    search = search_paragraphs(store, "kite words", log_calibration=False)
    base = build_semantic_candidate_pool(store, build_embedding_matrix(store.db_path))

    assert search is not None
    assert set(search.pool) == set(base) == set(_ids(kept))
    for mid, (memory, _) in search.pool.items():
        assert memory.content == base[mid][0].content
    after = {m.id: store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (m.id,)).fetchone()[0] for m in kept}  # noqa: SLF001
    assert after == before, "the pool read never bumps"


# ---------------------------------------------------------------------------
# CONC-6: the stacked-matrix cache; CONC-1: the running averages
# ---------------------------------------------------------------------------


def _unit_row(dim: int, axis: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 2.0
    return v


def test_a_put_while_the_stack_is_built_waits_and_the_next_reader_sees_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `put` from another thread injected while `stacked()` builds its cache
    under the lock: it cannot land inside the build (it waits for the lock),
    so the cache the build stores is never stale, and the next reader sees the
    new row."""
    MemoryStore(tmp_path / "memories.db")
    matrix = EmbeddingMatrix(tmp_path / "memories.db", model_id=_EMBEDDER_ID)
    matrix.ensure_built()
    matrix.put("a", _unit_row(4, 0))
    done_inside: list[bool] = []

    def _hook() -> None:
        writer = threading.Thread(target=matrix.put, args=("b", _unit_row(4, 1)))
        writer.start()
        writer.join(timeout=0.3)
        done_inside.append(not writer.is_alive())
        monkeypatch.setattr(matrix_mod, "_stack_hook", None)
        _hook.writer = writer  # type: ignore[attr-defined]

    monkeypatch.setattr(matrix_mod, "_stack_hook", _hook)
    ids, unit = matrix.stacked()
    finish_threads(_hook.writer)  # type: ignore[attr-defined]

    assert done_inside == [False], "the put waited for the build's lock"
    assert ids == ("a",)
    ids2, unit2 = matrix.stacked()
    assert ids2 == ("a", "b")
    assert unit2[1] == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_a_reader_keeps_its_snapshot_across_a_put_and_evict_and_the_next_reader_rebuilds(
    tmp_path: Path,
) -> None:
    MemoryStore(tmp_path / "memories.db")
    matrix = EmbeddingMatrix(tmp_path / "memories.db", model_id=_EMBEDDER_ID)
    matrix.ensure_built()
    matrix.put("a", _unit_row(4, 0))
    matrix.put("b", _unit_row(4, 1))
    ids, unit = matrix.stacked()
    snapshot = unit.copy()
    assert matrix.stacked()[1] is unit, "cached between mutations: no restack per message"

    matrix.put("a", _unit_row(4, 2))
    matrix.put("c", _unit_row(4, 3))
    matrix.evict("b")

    assert ids == ("a", "b") and np.array_equal(unit, snapshot), "the reader's pair is untouched"
    assert not unit.flags.writeable
    ids2, unit2 = matrix.stacked()
    assert ids2 == ("a", "c")
    assert unit2[0] == pytest.approx([0.0, 0.0, 1.0, 0.0])
    matrix.evict("c")
    assert matrix.stacked()[0] == ("a",), "an evict alone also drops the cached stack"


def test_each_cosine_path_paragraphs_best_heads_the_group_before_other_cosine_genuine_results() -> None:
    """C8b: A's second result (0.85) outscores B's best (0.7), yet B's best
    sits at the head of the cosine-path genuine group with A's best."""
    a1, a2, b1 = _m("a1"), _m("a2"), _m("b1")
    search = _hand_search(
        [(COSINE_PATH, [(a1, 0.9), (a2, 0.85)], 0.5, None), (COSINE_PATH, [(b1, 0.7)], 0.5, None)]
    )
    assert [h.memory_id for h in assemble_paragraph_results(search, 9)] == [a1.id, b1.id, a2.id]


_KEYWORD_FIRST = (
    KEYWORD_GENUINE,
    RERANKED_GENUINE,
    RERANKED_FAMILY,
    COSINE_GENUINE,
    COSINE_FAMILY,
    KEYWORD_FAMILY,
)


def test_passive_recall_keeps_a_guaranteed_slot_ahead_of_keyword_hits_under_any_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With keyword genuine hits ordered first and more of them than the cap,
    the paragraph's guaranteed semantic result still takes a slot."""
    monkeypatch.setattr(dev_constants, "RECALL_RESULT_GROUP_ORDER", _KEYWORD_FIRST)
    store = MemoryStore(":memory:")
    head = Memory.create_new(content="a cosine path genuine memory", memory_type="event", domain="us")
    store.create(head)
    keyword = [
        Memory.create_new(content=f"{_LONG} (keyword {i})", memory_type="conversation", domain="us")
        for i in range(12)
    ]
    for m in keyword:
        store.create(m)
    semantic = SemanticRecallResult(
        full=[head],
        snippet=[],
        scores={head.id: 0.9},
        hits=[SemanticHit(head, 0.9, COSINE_PATH, 0, False)],
        path=COSINE_PATH,
        guaranteed_ids=(head.id,),
    )
    ids = [mid for mid, _ in _active_rows(_render(store, "quokka photographs", tmp_path, semantic))]
    assert len(ids) == 9
    assert ids[-1] == head.id


def test_the_tool_keeps_a_guaranteed_slot_ahead_of_keyword_hits_under_any_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import brain.tools.impls.search_memories as tool

    monkeypatch.setattr(dev_constants, "RECALL_RESULT_GROUP_ORDER", _KEYWORD_FIRST)
    store = MemoryStore(":memory:")
    head = Memory.create_new(content="a cosine path genuine memory", memory_type="event", domain="us")
    store.create(head)
    for i in range(6):
        store.create(
            Memory.create_new(content=f"the quokka photographs {i}", memory_type="conversation", domain="us")
        )
    semantic = tool._SemanticTopK(grouped=[(head, COSINE_GENUINE)], guaranteed=[head.id])  # noqa: SLF001
    monkeypatch.setattr(tool, "_semantic_top_k", lambda *a, **kw: semantic)
    out = dispatch(
        "search_memories",
        {"query": "quokka photographs", "limit": 3},
        store=store,
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )
    ids = [m["id"] for m in out["memories"]]
    assert len(ids) == 3 and ids[-1] == head.id


# ---------------------------------------------------------------------------
# Review round 1 (R6/6-redteam-code.md): F2-F6, F8
# ---------------------------------------------------------------------------


def test_the_paragraph_time_includes_the_scan_and_the_cut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F2: the scan time = matrix product 0.0625 + coarse cut 0.03125 +
    pair-size probe 0.125 (P-28); the embed (0.5) is the embed series' (S108)."""
    _budget(monkeypatch, 4.0)
    store, rec, clock = _budget_case(monkeypatch, tmp_path, embed_seconds=0.5)
    real_rows, real_walk = sr._unit_rows, sr._walk_coarse_cut

    def _rows(vectors):
        clock.advance(0.0625)
        return real_rows(vectors)

    def _walk(ordered, is_family, size):
        clock.advance(0.03125)
        return real_walk(ordered, is_family, size)

    monkeypatch.setattr(sr, "_unit_rows", _rows)
    monkeypatch.setattr(sr, "_walk_coarse_cut", _walk)

    result = run_semantic_recall(store, tmp_path, "kite over the harbour")

    assert result is not None
    assert sr._scan_time.mean(store, "") == pytest.approx(0.0625 + 0.03125 + 0.125)
    assert result.rerank_budget == pytest.approx(4.0 - 0.25 - 0.5 - 0.21875)


def test_a_rerank_that_ran_without_a_gate_still_counts_toward_the_fifty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F3 (S58): no rerank floor and no cosine floor yet (a fresh persona):
    every query reranks and then has nothing to gate with, yet its reranked
    candidates still count, so 11 paragraphs of 5 rerank 10 times (50), not 11."""
    texts = [f"fresh {chr(97 + i)} words" for i in range(11)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = [_mem(store, f"r6 fresh {i} {j}", {i: 0.9 - 0.01 * j}).content for i in range(11) for j in range(5)]
    rec = _Reranker(dict.fromkeys(contents, 5.0))
    _install(monkeypatch, embedder, rec)

    search = search_paragraphs(store, "\n".join(texts), log_calibration=False)

    assert search is not None
    assert all(o.gated is None for o in search.outcomes), "no floor: nothing to gate"
    assert len(rec.scored()) == 10
    assert search.total_width == 50
    assert sum(o.reranked_width for o in search.outcomes) == 50


def test_the_matrix_first_build_is_kept_out_of_the_message_overhead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F4: the one-time lazy matrix load (5 s here) is not a `T_m` sample."""
    _budget(monkeypatch, 4.0)
    store, rec, clock = _budget_case(monkeypatch, tmp_path, embed_seconds=0.5)
    real_build = EmbeddingMatrix.ensure_built

    def _slow_first_build(self):
        if not self._built:
            clock.advance(5.0)
        return real_build(self)

    monkeypatch.setattr(EmbeddingMatrix, "ensure_built", _slow_first_build)

    run_semantic_recall(store, tmp_path, "kite over the harbour")

    assert sr._message_overhead.mean(store, "") == pytest.approx(0.25)


def test_a_reranker_that_cannot_load_is_tried_once_per_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F5: three rerankable paragraphs and a reranker that fails to build:
    one attempt for the message, every paragraph on the cosine path."""
    embedder = _AxisEmbedder({_P1: 0, _P2: 1, _P3: 2})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    for axis in range(3):
        for j in range(6):
            _mem(store, f"r6 down {axis} {j}", {axis: 0.9 - 0.01 * j})
    _cosine_floor(store, 0.5)
    attempts: list[int] = []

    def _boom(**kwargs):
        attempts.append(1)
        raise RuntimeError("simulated reranker construction failure")

    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", _boom)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert len(attempts) == 1
    assert result is not None
    assert [o.gated.path for o in result.paragraphs] == [COSINE_PATH] * 3


def test_a_bad_group_list_fails_at_import_not_silently_per_turn(tmp_path: Path) -> None:
    """F6: a list that does not name each group once stops the import, so a
    mis-edit cannot switch semantic recall off quietly."""
    import subprocess
    import sys

    code = (
        "import brain.dev_constants as d\n"
        "d.RECALL_RESULT_GROUP_ORDER = d.RECALL_RESULT_GROUP_ORDER[:5]\n"
        "import brain.memory.semantic_recall\n"
    )
    root = Path(__file__).resolve().parents[4]
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert run.returncode != 0
    assert "RECALL_RESULT_GROUP_ORDER" in run.stderr


def test_a_rebuild_drops_the_cached_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """F8 / P-23: a build swap invalidates the stack like put and evict."""
    store = MemoryStore(tmp_path / "memories.db")
    first = _mem(store, "r6 first row", {0: 0.5})
    matrix = EmbeddingMatrix(tmp_path / "memories.db", model_id=_EMBEDDER_ID)
    assert matrix.stacked()[0] == (first.id,)
    second = _mem(store, "r6 second row", {1: 0.5})  # written to the DB only

    matrix.rebuild()

    assert set(matrix.stacked()[0]) == {first.id, second.id}


def test_get_active_by_ids_skips_a_row_that_faded_or_was_deactivated(tmp_path: Path) -> None:
    """F8 / P-27: the full-row read after the cosine cut re-checks the state,
    so a memory that faded between the two reads is not a candidate."""
    store = MemoryStore(tmp_path / "memories.db")
    kept, faded, gone = (
        Memory.create_new(content=c, memory_type="event", domain="d") for c in ("k", "f", "g")
    )
    for m in (kept, faded, gone):
        store.create(m)
    store._conn.execute("UPDATE memories SET state = 'fading' WHERE id = ?", (faded.id,))  # noqa: SLF001
    store._conn.execute("UPDATE memories SET active = 0 WHERE id = ?", (gone.id,))  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001

    assert set(store.get_active_by_ids([kept.id, faded.id, gone.id])) == {kept.id}


def test_the_emptiness_test_is_case_insensitive() -> None:
    assert split_paragraphs("ok\nKITE over the Harbour", {"kite"}) == ["KITE over the Harbour"]


def test_a_memory_equally_close_to_two_paragraphs_goes_to_the_earlier_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embedder = _AxisEmbedder({_P1: 0, _P2: 1})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    tie = _mem(store, "equally close", {0: 0.6, 1: 0.6})
    _cosine_floor(store, 0.1)
    _no_reranker(monkeypatch)
    calls = _spy_rank_and_gate(monkeypatch)

    run_semantic_recall(store, tmp_path, f"{_P1}\n{_P2}")

    by_query = {c["query"]: dict(c["coarse"]) for c in calls}
    assert tie.id in by_query[_P1]
    assert _P2 not in by_query or tie.id not in by_query[_P2]


def _warm_decision_case(monkeypatch, tmp_path, *, scan, overhead, budget):
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, budget)
    embedder = _AxisEmbedder({_P1: 0, _P2: 1, _P3: 2})
    store = MemoryStore(tmp_path / "memories.db")
    sr._embed_cost.record(store, _EMBEDDER_ID, 100, 0.0)
    sr._scan_time.record(store, "", 1, scan)
    sr._message_overhead.record(store, "", 1, overhead)
    _install(monkeypatch, embedder)
    _slow_store_reads(monkeypatch, clock, overhead)
    _mem(store, "a memory", {_OTHER: 0.9})
    _mem(store, "a memory of the first", {0: 0.9})
    _mem(store, "a memory of the third", {2: 0.9})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)
    return store, embedder


def test_the_fit_decision_counts_the_message_overhead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """3 x scan 1.0 (embedding free) fits 3.5 s alone, but with `T_m` 0.75 it
    does not; paragraphs 1 and 3 (2 x 1.0 + 0.75) do (S195)."""
    store, embedder = _warm_decision_case(monkeypatch, tmp_path, scan=1.0, overhead=0.75, budget=3.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1, _P3]]
    assert result is not None and result.whole_message_fallback is False
    assert [o.query for o in result.paragraphs] == [_P1, _P3]


def test_when_the_alternate_paragraphs_do_not_fit_either_the_whole_message_is_one_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scan 1.5: 2 x 1.5 + 0.75 > 3.5 as well, so the whole message is ONE query
    (embedding costs nothing here, so the cut is the whole message)."""
    store, embedder = _warm_decision_case(monkeypatch, tmp_path, scan=1.5, overhead=0.75, budget=3.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_THREE]]
    assert result is not None and result.whole_message_fallback is True


def test_the_whole_message_is_cut_to_the_tokens_the_embed_fit_says_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S195: embed fit overhead 0.5 s + 0.01 s per token; budget 4.0, `T_m` 0.25,
    scan 1.0: room 2.75 s, so (2.75 - 0.5) / 0.01 = 225 tokens (characters in this
    test). Three paragraphs of 100 characters do not fit (3 x 1.0 scan + embed),
    nor do paragraphs 1 and 3 (2 x 1.0 + 0.5 + 0.01 x 2 x 100 = 4.5 > 3.75); the
    query is the longest word-boundary prefix of the message of at most 225
    characters."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 4.0)
    words = [f"w{i:03d}" for i in range(60)]
    paragraphs = [" ".join(words[i * 20 : (i + 1) * 20])[:99] + "x" for i in range(3)]
    message = "\n".join(paragraphs)
    embedder = _AxisEmbedder(dict.fromkeys(paragraphs, 0), clock)
    store = MemoryStore(tmp_path / "memories.db")
    sr._embed_cost.record(store, _EMBEDDER_ID, 100, 0.5 + 0.01 * 100)
    sr._embed_cost.record(store, _EMBEDDER_ID, 200, 0.5 + 0.01 * 200)
    sr._scan_time.record(store, "", 1, 1.0)
    sr._message_overhead.record(store, "", 1, 0.25)
    _install(monkeypatch, embedder)
    _slow_store_reads(monkeypatch, clock, 0.25)
    _mem(store, "a memory", {_OTHER: 0.9})
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, message)

    (batch,) = embedder.batches
    (query,) = batch
    assert result is not None and result.whole_message_fallback is True
    assert message.startswith(query) and len(query) <= 225
    # the longest word-boundary prefix that fits: the next word would not
    rest = message[len(query) :]
    next_word = rest.split()[0]
    assert len(query) + (len(rest) - len(rest.lstrip())) + len(next_word) > 224


def test_when_not_even_one_token_fits_the_first_paragraph_alone_is_searched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S195 edge: the setup and one query's scan already use the whole budget."""
    store, embedder = _warm_decision_case(monkeypatch, tmp_path, scan=3.0, overhead=0.75, budget=3.5)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1]]
    assert result is not None and [o.query for o in result.paragraphs] == [_P1]


def test_the_first_message_decision_counts_the_message_overhead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S199 decision: first paragraph timed at 1.0 s; 3 x 1.0 fits 3.5 s alone,
    but with `T_m` 0.75 it does not; paragraphs 1 and 3 (0.75 + 2 x 1.0) do."""
    clock = _Clock()
    monkeypatch.setattr(sr, "_clock", clock)
    _budget(monkeypatch, 3.5)
    store, embedder, rec, first, rest = _two_query_store(monkeypatch, tmp_path, clock, 1.0)
    _slow_store_reads(monkeypatch, clock, 0.75)

    run_semantic_recall(store, tmp_path, _THREE)

    assert embedder.batches == [[_P1], [_P3]]


def test_only_a_query_that_can_rerank_is_probed_for_pair_sizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embedder = _AxisEmbedder({_P1: 0, _P2: 1})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    big = [_mem(store, f"r6 big {i}", {0: 0.9 - 0.01 * i}).content for i in range(6)]
    for i in range(2):
        _mem(store, f"r6 small {i}", {1: 0.8 - 0.01 * i})
    rec = _Reranker(dict.fromkeys(big, 5.0))
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.5)
    _warm(store)

    run_semantic_recall(store, tmp_path, f"{_P1}\n{_P2}")

    assert set(rec.probed) == {_P1}


def test_a_rerank_whose_normalization_fell_back_still_counts_toward_the_fifty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """F3 (S58): a rerank that ran but could not be normalized (P-6) used its
    width: 6 paragraphs whose reranks each score 10 and fall back use the 50
    after 5 of them, so the 6th is not reranked.

    The stub takes any arguments (a later added one cannot turn the call into a
    TypeError that the broad `except` in `rank_and_gate` would swallow as a
    cosine-path turn), and the test asserts it is on the intended path: the stub
    ran for every paragraph with the real call shape, no "reranker failed" warning
    was logged, and each fallback logged the normalization hand-off."""
    caplog.set_level(logging.INFO, logger="brain.memory.semantic_recall")
    texts = [f"norm {chr(97 + i)} words" for i in range(6)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder, _Reranker())
    for i in range(6):
        for j in range(12):
            _mem(store, f"r6 norm {i} {j}", {i: 0.9 - 0.01 * j})
    _cosine_floor(store, 0.5)
    calls: list[tuple[tuple, dict]] = []

    def _fell_back(*args, **kwargs):
        calls.append((args, kwargs))
        return reranker_mod.RecallRerank(
            width=10, reranked=False, hand_off="normalization", normalization=None, measured=False
        )

    monkeypatch.setattr(reranker_mod, "rerank_for_recall", _fell_back)

    search = search_paragraphs(store, "\n".join(texts), log_calibration=False)

    assert search is not None
    # The stub ran, once per rerank attempt, with the call shape production uses.
    assert len(calls) == 5, "the 6th query has 0 left: no rerank call"
    for (args, kwargs), expected in zip(calls, [50, 40, 30, 20, 10], strict=True):
        assert len(args) == 3 and args[1] in texts and len(args[2]) >= 5
        assert kwargs["max_real"] == expected and "budget_seconds" in kwargs and "sizes" in kwargs
    # The path is the normalization hand-off, not a swallowed failure.
    assert not [r for r in caplog.records if "reranker failed" in r.getMessage()]
    handoffs = [r.getMessage() for r in caplog.records if "no rerank (normalization, width 10)" in r.getMessage()]
    assert len(handoffs) == 5
    assert search.total_width == 50
    assert [o.reranked_width for o in search.outcomes] == [10, 10, 10, 10, 10, 0]
    assert {o.gated.path for o in search.outcomes} == {COSINE_PATH}


# ---------------------------------------------------------------------------
# Review round 2 (R6/6-redteam-code.round2.md): G1-G3
# ---------------------------------------------------------------------------


def test_a_pool_below_five_never_builds_the_reranker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1: fewer than 5 real candidates exist, so no rerank can run and the
    reranker (a model load on first use) is not even built."""
    embedder = _AxisEmbedder({"kite words": 0})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    for j in range(3):
        _mem(store, f"r6 few {j}", {0: 0.9 - 0.01 * j})
    _cosine_floor(store, 0.5)
    built: list[int] = []

    def _build(**kwargs):
        built.append(1)
        return _Reranker()

    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", _build)

    result = run_semantic_recall(store, tmp_path, "kite words")

    assert result is not None and result.path == COSINE_PATH
    assert built == []


class _WarmUpFails(_Reranker):
    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        raise RuntimeError("simulated warm-up failure")


def test_a_reranker_whose_warm_up_fails_is_tried_once_per_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G2: the provider builds but its first rerank (the warm-up) raises: one
    attempt for the message, no pair-size probe, every paragraph on the
    cosine path."""
    embedder = _AxisEmbedder({_P1: 0, _P2: 1, _P3: 2})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    for axis in range(3):
        for j in range(6):
            _mem(store, f"r6 warm {axis} {j}", {axis: 0.9 - 0.01 * j})
    _cosine_floor(store, 0.5)
    rec = _WarmUpFails()
    _install(monkeypatch, embedder, rec)

    result = run_semantic_recall(store, tmp_path, _THREE)

    assert len(rec.calls) == 1
    assert rec.probed == []
    assert result is not None
    assert [o.gated.path for o in result.paragraphs] == [COSINE_PATH] * 3


class _ProbeFailsFor(_Reranker):
    def __init__(self, failing_query: str, **kwargs):
        super().__init__(**kwargs)
        self.failing_query = failing_query

    def pair_token_lengths(self, query, documents):
        if query == self.failing_query:
            self.probed.append(query)
            raise RuntimeError("simulated probe failure for one query")
        return super().pair_token_lengths(query, documents)


def test_one_querys_probe_failure_keeps_the_other_queries_sizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G2: a pair-size probe that fails for P2 only: P1 still reranks with the
    sizes computed ahead; P2 gets none (its own rank_and_gate retries)."""
    embedder = _AxisEmbedder({_P1: 0, _P2: 1})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = [
        _mem(store, f"r6 probe {axis} {j}", {axis: 0.9 - 0.01 * j}).content
        for axis in (0, 1)
        for j in range(6)
    ]
    rec = _ProbeFailsFor(_P2, scores=dict.fromkeys(contents, 5.0))
    _install(monkeypatch, embedder, rec)
    _rerank_floor(store, 1.0)
    _cosine_floor(store, 0.5)
    _warm(store)
    calls = _spy_rank_and_gate(monkeypatch)

    run_semantic_recall(store, tmp_path, f"{_P1}\n{_P2}")

    sizes = {c["query"]: c["sizes"] for c in calls}
    assert sizes[_P1] is not None and sizes[_P1].lengths is not None
    assert sizes[_P2] is None


def test_a_rerank_whose_floor_read_failed_still_counts_toward_the_fifty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G3 (S58): each query reranks 12 and then the rerank floor cannot be
    read: the 12 still count, so 4 queries rerank (48) and the rest do not."""
    texts = [f"floor {chr(97 + i)} words" for i in range(6)]
    embedder = _AxisEmbedder({t: i for i, t in enumerate(texts)})
    store = MemoryStore(tmp_path / "memories.db")
    _install(monkeypatch, embedder)
    contents = [
        _mem(store, f"r6 floor {i} {j}", {i: 0.9 - 0.01 * j}).content
        for i in range(6)
        for j in range(12)
    ]
    rec = _Reranker(dict.fromkeys(contents, 5.0))
    _install(monkeypatch, embedder, rec)
    _cosine_floor(store, 0.5)
    _warm(store)

    def _broken(self, model_id):
        raise RuntimeError("simulated floor read failure")

    monkeypatch.setattr(MemoryStore, "get_reranker_floor", _broken)

    search = search_paragraphs(store, "\n".join(texts), log_calibration=False)

    assert search is not None
    assert len(rec.scored()) == 4
    assert search.total_width == 48
