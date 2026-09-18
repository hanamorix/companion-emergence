"""Tests for brain.memory.relevance_judge — the F2a #250 inc6 local
relevance judge + Haiku tie-break (spec Section 6).

All OFFLINE (FakeRelevanceJudgeProvider / a scripted Haiku stub) — no real
model download, no torch import, no network. A real-model validation test
lives in tests/unit/brain/memory/test_relevance_judge_real_model.py, marked
BOTH @pytest.mark.requires_network AND @pytest.mark.integration, mirroring
test_reranker_real_model.py's convention.
"""

from __future__ import annotations

import json
import math

import pytest

import brain.memory.relevance_judge as rj_mod
from brain.memory.relevance_judge import (
    AMBIGUOUS_BAND_HALF_WIDTH,
    FakeRelevanceJudgeProvider,
    RelevanceJudgeProvider,
    _make_haiku_tiebreak,
    _reset_judge_provider_cache,
    build_judge_provider,
    label_calibration_sample,
    label_for_score,
)
from brain.memory.store import Memory, MemoryStore


def _mem(content: str = "x", **kw: object) -> Memory:
    defaults = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    return Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# FakeRelevanceJudgeProvider — scriptable, offline
# ---------------------------------------------------------------------------


def test_fake_judge_returns_scripted_score_for_exact_pair() -> None:
    provider = FakeRelevanceJudgeProvider(scores={("q", "d"): 7.5})
    assert provider.score("q", "d") == 7.5


def test_fake_judge_unscripted_pair_gets_the_default() -> None:
    provider = FakeRelevanceJudgeProvider(scores={})
    assert provider.score("q", "unscripted") == FakeRelevanceJudgeProvider._DEFAULT_UNSCORED


def test_fake_judge_custom_default_is_honored() -> None:
    provider = FakeRelevanceJudgeProvider(scores={}, default=3.0)
    assert provider.score("q", "d") == 3.0


def test_fake_judge_model_id_is_stable_and_distinct() -> None:
    assert FakeRelevanceJudgeProvider().model_id() == "fake-relevance-judge"


def test_fake_judge_is_a_relevance_judge_provider() -> None:
    assert isinstance(FakeRelevanceJudgeProvider(), RelevanceJudgeProvider)


# ---------------------------------------------------------------------------
# label_for_score — sigmoid + self-contained ambiguous band (spec Section 6)
# ---------------------------------------------------------------------------


def test_label_for_score_large_positive_logit_is_relevant_and_clear() -> None:
    label, ambiguous = label_for_score(10.0)
    assert label == "relevant"
    assert ambiguous is False


def test_label_for_score_large_negative_logit_is_irrelevant_and_clear() -> None:
    label, ambiguous = label_for_score(-10.0)
    assert label == "irrelevant"
    assert ambiguous is False


def test_label_for_score_zero_logit_sigmoid_is_exactly_the_boundary_and_ambiguous() -> None:
    """sigmoid(0) == 0.5 exactly — dead center of the ambiguous band
    regardless of band width (as long as the band is non-zero)."""
    label, ambiguous = label_for_score(0.0)
    assert ambiguous is True
    assert label == "relevant", "the >= 0.5 tie-break rule: p==0.5 counts as relevant"


def test_label_for_score_just_inside_the_band_is_ambiguous() -> None:
    half_width = 0.05
    # Solve for the raw logit whose sigmoid sits just inside (0.5 - half_width).
    p = 0.5 - half_width + 0.001
    x = math.log(p / (1 - p))
    _, ambiguous = label_for_score(x, band_half_width=half_width)
    assert ambiguous is True


def test_label_for_score_just_outside_the_band_is_clear() -> None:
    half_width = 0.05
    p = 0.5 - half_width - 0.001
    x = math.log(p / (1 - p))
    label, ambiguous = label_for_score(x, band_half_width=half_width)
    assert ambiguous is False
    assert label == "irrelevant"


def test_label_for_score_defaults_to_the_live_tunable_band_width() -> None:
    """No band_half_width passed -> reads the live
    calibration.judge_ambiguous_band_half_width tunable (default
    AMBIGUOUS_BAND_HALF_WIDTH), mirroring reranker.py's LATENCY_BUDGET_SECONDS
    override pattern."""
    p_inside = 0.5 - AMBIGUOUS_BAND_HALF_WIDTH + 0.001
    x_inside = math.log(p_inside / (1 - p_inside))
    _, ambiguous = label_for_score(x_inside)
    assert ambiguous is True

    p_outside = 0.5 - AMBIGUOUS_BAND_HALF_WIDTH - 0.001
    x_outside = math.log(p_outside / (1 - p_outside))
    _, ambiguous = label_for_score(x_outside)
    assert ambiguous is False


# ---------------------------------------------------------------------------
# build_judge_provider — process-wide cache (mirrors test_reranker.py's
# coverage of build_reranker_provider's own cache).
# ---------------------------------------------------------------------------


def test_build_judge_provider_is_process_cached_by_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_judge_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRelevanceJudgeProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(rj_mod, "TorchCrossEncoderJudge", _CountingFake)
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "fake-judge-id")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")

    p1 = build_judge_provider()
    p2 = build_judge_provider()
    assert p1 is p2, "same model_id must return the SAME cached provider instance"
    assert calls["n"] == 1, "construction must happen exactly once, not per call"
    _reset_judge_provider_cache()


def test_reset_judge_provider_cache_forces_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_judge_provider_cache()
    calls = {"n": 0}

    class _CountingFake(FakeRelevanceJudgeProvider):
        def __init__(self, model_id: str, cache_dir) -> None:
            super().__init__()
            calls["n"] += 1

    monkeypatch.setattr(rj_mod, "TorchCrossEncoderJudge", _CountingFake)
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "fake-judge-id")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")

    build_judge_provider()
    _reset_judge_provider_cache()
    build_judge_provider()
    assert calls["n"] == 2, "a reset must force the next call to construct again"
    _reset_judge_provider_cache()


def test_importing_this_module_never_pulls_torch_into_sys_modules() -> None:
    """The load-bearing proof for I6/the owner ruling: torch must only be
    imported inside TorchCrossEncoderJudge.__init__, never at this module's
    top level. If torch happens to already be present in sys.modules from
    an EARLIER test in this same process (e.g. the real-model test file),
    this assertion would be meaningless — so this test constructs its own
    fresh subprocess-free proof by checking the plain import graph instead:
    relevance_judge's own __init__ source never references torch at module
    scope. Belt-and-braces alongside the conftest-level FakeRelevanceJudge
    default and the dedicated import test in test_semantic_recall-adjacent
    hot-path modules."""
    import ast
    import inspect

    source = inspect.getsource(rj_mod)
    tree = ast.parse(source)
    module_level_names: set[str] = set()
    for node in tree.body:  # only TOP-LEVEL statements, not inside functions/classes
        if isinstance(node, ast.Import):
            module_level_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level_names.add(node.module)
    assert not any(name.startswith("torch") or name.startswith("sentence_transformers") for name in module_level_names), (
        f"torch/sentence_transformers must only be imported inside a function body, "
        f"found at module scope: {module_level_names}"
    )


# ---------------------------------------------------------------------------
# _make_haiku_tiebreak — mirrors consolidation._make_haiku_classifier's
# fail-soft-on-any-failure shape.
# ---------------------------------------------------------------------------


class _ScriptedHaikuProvider:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self._reply


class _RaisingHaikuProvider:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("boom")


def test_haiku_tiebreak_parses_relevant_label() -> None:
    provider = _ScriptedHaikuProvider('{"label": "relevant"}')
    tiebreak = _make_haiku_tiebreak(provider)
    assert tiebreak("query", "doc") == "relevant"


def test_haiku_tiebreak_parses_irrelevant_label() -> None:
    provider = _ScriptedHaikuProvider('{"label": "irrelevant"}')
    tiebreak = _make_haiku_tiebreak(provider)
    assert tiebreak("query", "doc") == "irrelevant"


def test_haiku_tiebreak_returns_none_on_unrecognized_label() -> None:
    provider = _ScriptedHaikuProvider('{"label": "maybe"}')
    tiebreak = _make_haiku_tiebreak(provider)
    assert tiebreak("query", "doc") is None


def test_haiku_tiebreak_returns_none_on_malformed_json() -> None:
    provider = _ScriptedHaikuProvider("not json at all")
    tiebreak = _make_haiku_tiebreak(provider)
    assert tiebreak("query", "doc") is None


def test_haiku_tiebreak_returns_none_on_provider_exception() -> None:
    tiebreak = _make_haiku_tiebreak(_RaisingHaikuProvider())
    assert tiebreak("query", "doc") is None, "a provider failure must be caught, not propagated"


def test_haiku_tiebreak_prompt_and_query_reach_the_provider() -> None:
    provider = _ScriptedHaikuProvider('{"label": "relevant"}')
    tiebreak = _make_haiku_tiebreak(provider)
    tiebreak("what does bob like", "bob likes coffee")
    (prompt, system) = provider.calls[0]
    assert "what does bob like" in prompt
    assert "bob likes coffee" in prompt
    assert system == rj_mod._HAIKU_TIEBREAK_PROMPT


# ---------------------------------------------------------------------------
# label_calibration_sample — the orchestration entry point
# _run_calibration_tick calls (spec Section 6/7, acceptance #7).
# ---------------------------------------------------------------------------


def test_label_calibration_sample_returns_zero_when_nothing_to_label(store: MemoryStore) -> None:
    assert label_calibration_sample(store, judge=FakeRelevanceJudgeProvider()) == 0


def test_label_calibration_sample_clear_case_never_calls_haiku(store: MemoryStore) -> None:
    """AC#7: a stubbed judge test confirms clear-case pairs are labeled
    locally and only ambiguous-band pairs trigger a (stubbed) Haiku call —
    no Haiku call on clear cases."""
    mem = _mem("deep breathing helps when anxious")
    store.create(mem)
    store.log_calibration_sample(
        query="how do I calm down", candidate_ids=[mem.id], reranker_scores=[5.0],
        reranker_model_id="m",
    )
    judge = FakeRelevanceJudgeProvider(scores={("how do I calm down", mem.content): 10.0})  # far clear-relevant
    haiku = _ScriptedHaikuProvider('{"label": "irrelevant"}')

    labeled = label_calibration_sample(store, provider=haiku, judge=judge)

    assert labeled == 1
    assert haiku.calls == [], "no Haiku call on a clear-case pair"
    row = store._conn.execute(
        "SELECT local_judge_label, haiku_label FROM calibration_log"
    ).fetchone()
    assert json.loads(row["local_judge_label"]) == ["relevant"]
    assert json.loads(row["haiku_label"]) == [None]


def test_label_calibration_sample_ambiguous_case_calls_haiku_and_writes_its_label(
    store: MemoryStore,
) -> None:
    mem = _mem("something in between")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[0.0], reranker_model_id="m"
    )
    # raw_score=0.0 -> sigmoid(0.0)==0.5 -> dead center of the ambiguous band.
    judge = FakeRelevanceJudgeProvider(scores={("q", mem.content): 0.0})
    haiku = _ScriptedHaikuProvider('{"label": "irrelevant"}')

    labeled = label_calibration_sample(store, provider=haiku, judge=judge)

    assert labeled == 1
    assert len(haiku.calls) == 1, "exactly one Haiku call for the one ambiguous candidate"
    row = store._conn.execute(
        "SELECT local_judge_label, haiku_label FROM calibration_log"
    ).fetchone()
    assert json.loads(row["local_judge_label"]) == ["relevant"], "local judge's own provisional call still recorded"
    assert json.loads(row["haiku_label"]) == ["irrelevant"], "Haiku's override recorded separately"


def test_label_calibration_sample_ambiguous_with_no_provider_degrades_to_local_only(
    store: MemoryStore,
) -> None:
    """No provider injected (mirrors consolidation's degrade-to-promote-all
    when no provider is available) -> ambiguous candidates get no Haiku
    call at all; haiku_label stays None, local_judge_label is the only
    signal recorded."""
    mem = _mem("ambiguous content")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[0.0], reranker_model_id="m"
    )
    judge = FakeRelevanceJudgeProvider(scores={("q", mem.content): 0.0})

    labeled = label_calibration_sample(store, provider=None, judge=judge)

    assert labeled == 1
    row = store._conn.execute(
        "SELECT local_judge_label, haiku_label FROM calibration_log"
    ).fetchone()
    assert json.loads(row["haiku_label"]) == [None]


def test_label_calibration_sample_missing_candidate_gets_unknown_label(store: MemoryStore) -> None:
    store.log_calibration_sample(
        query="q", candidate_ids=["does-not-exist"], reranker_scores=[1.0], reranker_model_id="m"
    )
    judge = FakeRelevanceJudgeProvider()

    labeled = label_calibration_sample(store, judge=judge)

    assert labeled == 1
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    assert json.loads(row["local_judge_label"]) == ["unknown"]


def test_label_calibration_sample_judge_construction_failure_labels_nothing(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault isolation: a judge-construction failure (torch missing, model
    download failed, ...) must not crash — returns 0, no rows touched."""
    mem = _mem("x")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
    )

    def _raise() -> RelevanceJudgeProvider:
        raise RuntimeError("torch not installed")

    monkeypatch.setattr(rj_mod, "build_judge_provider", _raise)

    labeled = label_calibration_sample(store, judge=None)

    assert labeled == 0
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    assert row["local_judge_label"] is None, "row must stay unlabeled, eligible for a later tick"


def test_label_calibration_sample_one_candidate_failure_does_not_sink_the_row(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault isolation at candidate granularity: judge.score() raising for
    ONE candidate in a multi-candidate row must not lose the other
    candidates' labels."""
    mem_good = _mem("fine content")
    mem_bad = _mem("bad content")
    store.create(mem_good)
    store.create(mem_bad)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem_good.id, mem_bad.id], reranker_scores=[1.0, 2.0],
        reranker_model_id="m",
    )

    class _FlakyJudge(FakeRelevanceJudgeProvider):
        def score(self, query: str, document: str) -> float:
            if document == "bad content":
                raise RuntimeError("boom")
            return 10.0

    labeled = label_calibration_sample(store, judge=_FlakyJudge())

    assert labeled == 1
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    labels = json.loads(row["local_judge_label"])
    assert labels[0] == "relevant"
    assert labels[1] == "error"


def test_label_calibration_sample_one_row_failure_does_not_sink_other_rows(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault isolation at row granularity: write_calibration_labels raising
    for one row must not stop the pass from labeling the rest."""
    mem_a = _mem("a content")
    mem_b = _mem("b content")
    store.create(mem_a)
    store.create(mem_b)
    store.log_calibration_sample(
        query="q1", candidate_ids=[mem_a.id], reranker_scores=[1.0], reranker_model_id="m"
    )
    store.log_calibration_sample(
        query="q2", candidate_ids=[mem_b.id], reranker_scores=[1.0], reranker_model_id="m"
    )

    original_write = store.write_calibration_labels
    calls = {"n": 0}

    def _flaky_write(row_id, local_labels, haiku_labels):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return original_write(row_id, local_labels, haiku_labels)

    monkeypatch.setattr(store, "write_calibration_labels", _flaky_write)

    labeled = label_calibration_sample(store, judge=FakeRelevanceJudgeProvider())

    assert labeled == 1, "one row failed and one row succeeded"


def test_label_calibration_sample_respects_sample_rows_limit(store: MemoryStore) -> None:
    for i in range(5):
        store.log_calibration_sample(
            query=f"q{i}", candidate_ids=[], reranker_scores=[], reranker_model_id="m"
        )
    labeled = label_calibration_sample(store, judge=FakeRelevanceJudgeProvider(), sample_rows=2)
    assert labeled == 2
