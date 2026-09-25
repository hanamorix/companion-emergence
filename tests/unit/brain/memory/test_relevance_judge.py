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
import subprocess
import sys
import textwrap
from pathlib import Path

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
# label_for_score — slope/intercept (F2c inc3, spec §5): ABSENT-SAFE Platt
# calibration params. Both None (the default, and every live call site
# until F2c inc4) must reproduce today's exact fixed behavior.
# ---------------------------------------------------------------------------


def test_label_for_score_slope_intercept_absent_matches_fixed_default() -> None:
    """No slope/intercept passed -> byte-for-byte the pre-inc3 fixed
    sigmoid-0.5 behavior (absent-safe fallback, spec §5)."""
    for raw_score in (-10.0, -0.3, 0.0, 0.3, 10.0):
        assert label_for_score(raw_score) == label_for_score(raw_score, slope=None, intercept=None)


def test_label_for_score_slope_1_intercept_0_is_identical_to_fixed_default() -> None:
    """Explicit identity params (slope=1.0, intercept=0.0) must reproduce
    the fixed default exactly — this is the mapping absent params are
    documented as equivalent to."""
    for raw_score in (-10.0, -0.3, 0.0, 0.3, 10.0):
        assert label_for_score(raw_score, slope=1.0, intercept=0.0) == label_for_score(raw_score)


def test_label_for_score_only_slope_given_intercept_falls_back_to_zero() -> None:
    """Either argument being None uses the default for just the missing
    one, not an all-or-nothing requirement."""
    assert label_for_score(0.0, slope=2.0) == label_for_score(0.0, slope=2.0, intercept=0.0)


def test_label_for_score_only_intercept_given_slope_falls_back_to_one() -> None:
    assert label_for_score(0.0, intercept=1.0) == label_for_score(0.0, slope=1.0, intercept=1.0)


def test_label_for_score_fitted_intercept_shifts_the_decision_boundary() -> None:
    """BITE: a raw score that is "irrelevant" under the fixed default (its
    sigmoid sits below 0.5) flips to "relevant" once a fitted intercept
    shifts the boundary past it — proving the params path actually changes
    the label, not merely accepted and ignored."""
    raw_score = -0.5
    fixed_label, _ = label_for_score(raw_score)
    assert fixed_label == "irrelevant", "sanity: -0.5 is below the fixed 0.5 boundary"

    fitted_label, _ = label_for_score(raw_score, slope=1.0, intercept=1.0)
    assert fitted_label == "relevant", "shifted boundary (z = -0.5 + 1.0 = 0.5 > 0) now covers this score"


def test_label_for_score_fitted_params_also_shift_the_ambiguous_band() -> None:
    """The ambiguous band is defined on the (possibly recalibrated)
    probability, so a fitted mapping shifts where the band sits too, not
    just the pass/fail label."""
    # Under the fixed default, raw_score=1.0 is comfortably clear
    # (sigmoid(1.0) ~= 0.73, outside the default 0.05 half-width band).
    _, fixed_ambiguous = label_for_score(1.0)
    assert fixed_ambiguous is False
    # A fitted slope that compresses the score toward 0 moves it back
    # inside the band around the new boundary.
    _, fitted_ambiguous = label_for_score(1.0, slope=0.01, intercept=0.0)
    assert fitted_ambiguous is True


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
# TorchCrossEncoderJudge.score() — the double-sigmoid regression guard
# (`activation_fn=lambda x: x`, see the class's own docstring). This is the
# ONLY offline/default-gate coverage of that line: the sole prior test that
# would catch a revert to `activation_fn=None` lived in
# test_relevance_judge_real_model.py, which is requires_network + integration
# and deselected from the default gate/CI.
# ---------------------------------------------------------------------------


class _FakeSentenceTransformersCrossEncoder:
    """Stand-in for `sentence_transformers.CrossEncoder`, monkeypatched onto
    the REAL `sentence_transformers` module so `TorchCrossEncoderJudge`'s
    lazy `from sentence_transformers import CrossEncoder` (inside its own
    `__init__`) picks it up — mirrors real `.predict()`'s documented
    activation_fn contract just enough to prove the production code forces
    an identity pass-through: if `activation_fn` is None, applies
    sentence-transformers' OWN documented default for a num_labels=1 model
    (Sigmoid) — exactly the double-sigmoid trap
    `TorchCrossEncoderJudge.score()`'s comment describes; otherwise applies
    whatever callable it was given."""

    RAW_LOGIT = 2.7

    def __init__(self, model_id: str, cache_folder: str | None = None) -> None:
        self.model_id = model_id
        self.cache_folder = cache_folder
        self.predict_calls: list[object] = []

    def predict(self, pairs, activation_fn=None):
        self.predict_calls.append(activation_fn)
        fn = activation_fn if activation_fn is not None else (lambda x: 1.0 / (1.0 + math.exp(-x)))
        return [fn(self.RAW_LOGIT)]


def test_torch_cross_encoder_judge_score_forces_identity_activation_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructs a REAL `TorchCrossEncoderJudge` (bypassing
    `build_judge_provider`, which the autouse `_fake_relevance_judge_
    provider_by_default` fixture fakes) with `sentence_transformers.
    CrossEncoder` itself monkeypatched to `_FakeSentenceTransformersCrossEncoder`.
    Asserts BOTH that the `activation_fn` the fake's `.predict()` received is
    an identity callable, AND that `score()`'s return value is the untouched
    raw logit rather than a sigmoid-squashed probability — either assertion
    alone would miss a regression that broke only one half of the contract."""
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeSentenceTransformersCrossEncoder)

    judge = rj_mod.TorchCrossEncoderJudge(model_id="fake-judge-model", cache_dir="/tmp/fake-cache-dir")
    score = judge.score("q", "d")

    assert score == pytest.approx(_FakeSentenceTransformersCrossEncoder.RAW_LOGIT), (
        f"score() must return the RAW logit ({_FakeSentenceTransformersCrossEncoder.RAW_LOGIT}), "
        f"not a pre-sigmoided probability — got {score!r}"
    )
    fake_model = judge._model  # noqa: SLF001 — test-only reach into the fake we just installed
    assert len(fake_model.predict_calls) == 1
    received_activation_fn = fake_model.predict_calls[0]
    assert received_activation_fn is not None, (
        "activation_fn must never be None — None triggers sentence-transformers' own default "
        "Sigmoid for a num_labels=1 model, double-sigmoiding against label_for_score's own sigmoid"
    )
    assert received_activation_fn(2.7) == 2.7, "activation_fn must be an identity pass-through"


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

    def _flaky_write(row_id, local_labels, haiku_labels, local_judge_raw_score=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return original_write(
            row_id, local_labels, haiku_labels, local_judge_raw_score=local_judge_raw_score
        )

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


# ---------------------------------------------------------------------------
# F2c inc1 (data foundation only, spec §3 Addition A): the judge's RAW
# score/logit is accumulated alongside the derived label and persisted via
# `write_calibration_labels`.
# ---------------------------------------------------------------------------


def test_label_calibration_sample_persists_raw_scores_not_derived_labels(
    store: MemoryStore,
) -> None:
    """BITE: `local_judge_raw_score` holds the RAW logit the judge actually
    returned (e.g. 12.5, far outside [0,1]) — not the sigmoid-derived
    relevant/irrelevant label, and positionally aligned with
    `candidate_ids`/`local_judge_label`."""
    mem_a = _mem("clearly relevant content")
    mem_b = _mem("clearly irrelevant content")
    store.create(mem_a)
    store.create(mem_b)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem_a.id, mem_b.id], reranker_scores=[1.0, 2.0],
        reranker_model_id="m",
    )
    judge = FakeRelevanceJudgeProvider(
        scores={("q", mem_a.content): 12.5, ("q", mem_b.content): -8.25}
    )

    labeled = label_calibration_sample(store, judge=judge)

    assert labeled == 1
    row = store._conn.execute(
        "SELECT local_judge_label, local_judge_raw_score FROM calibration_log"
    ).fetchone()
    labels = json.loads(row["local_judge_label"])
    raw_scores = json.loads(row["local_judge_raw_score"])
    assert labels == ["relevant", "irrelevant"]
    assert raw_scores == [12.5, -8.25], "the RAW logits, not the derived labels"
    assert raw_scores != labels


def test_label_calibration_sample_raw_score_is_null_for_unknown_candidate(
    store: MemoryStore,
) -> None:
    """A candidate the judge never scored (deleted since logging -> the
    "unknown" label sentinel) gets `None` at that position in
    `local_judge_raw_score`, never a fabricated 0.0 that could be mistaken
    for a real score."""
    store.log_calibration_sample(
        query="q", candidate_ids=["does-not-exist"], reranker_scores=[1.0], reranker_model_id="m"
    )
    judge = FakeRelevanceJudgeProvider()

    labeled = label_calibration_sample(store, judge=judge)

    assert labeled == 1
    row = store._conn.execute(
        "SELECT local_judge_label, local_judge_raw_score FROM calibration_log"
    ).fetchone()
    assert json.loads(row["local_judge_label"]) == ["unknown"]
    assert json.loads(row["local_judge_raw_score"]) == [None]


def test_label_calibration_sample_raw_score_is_null_for_error_candidate(
    store: MemoryStore,
) -> None:
    """A candidate whose judge.score() call raises (the "error" label
    sentinel) also gets `None` at that position, not a fabricated score."""
    mem_bad = _mem("bad content")
    store.create(mem_bad)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem_bad.id], reranker_scores=[1.0], reranker_model_id="m"
    )

    class _FlakyJudge(FakeRelevanceJudgeProvider):
        def score(self, query: str, document: str) -> float:
            raise RuntimeError("boom")

    labeled = label_calibration_sample(store, judge=_FlakyJudge())

    assert labeled == 1
    row = store._conn.execute(
        "SELECT local_judge_label, local_judge_raw_score FROM calibration_log"
    ).fetchone()
    assert json.loads(row["local_judge_label"]) == ["error"]
    assert json.loads(row["local_judge_raw_score"]) == [None]


# ---------------------------------------------------------------------------
# F2c inc1 (spec §6): the durable in-code Haiku-oracle note must actually be
# present at the F2a judge/label site — grep/lint check (acceptance #9).
# ---------------------------------------------------------------------------


def test_haiku_oracle_note_is_present_in_relevance_judge_source() -> None:
    """A plain code-comment presence check — not a behavior test — proving
    the required durable note (spec §6: Haiku is the effective relevance
    ORACLE the judge converges toward) actually exists at this module,
    immediately above `label_calibration_sample` (the F2a judge/label
    site), so a later relevance-quality problem has a documented place to
    look. Scoped to just that note block (not the whole module, which uses
    em-dashes freely elsewhere in ordinary docstrings) since the "no
    em-dash" requirement applies to this specific durable note, not to
    every comment in the file."""
    import inspect

    source = inspect.getsource(rj_mod)
    marker = "# F2c (durable note, spec"
    assert marker in source, "the durable Haiku-oracle note must precede label_calibration_sample"
    note_start = source.index(marker)
    note_end = source.index("def label_calibration_sample", note_start)
    note = source[note_start:note_end]
    assert "oracle" in note.lower()
    assert "haiku" in note.lower()
    assert "—" not in note, "no em-dashes in this durable note (plain code comment, no LLM-tells)"


# ---------------------------------------------------------------------------
# F2c inc4a (spec §5 "where the tuned judge loads from") — the LOAD side:
# `label_calibration_sample` now reads THIS PERSONA's persisted
# `judge_knob_calibration` row (via `store.get_judge_knob_calibration`) and
# threads it into every `label_for_score` call the pass makes.
# ---------------------------------------------------------------------------


def test_label_calibration_sample_applies_this_personas_persisted_knob(store: MemoryStore) -> None:
    """BITE: raw_score=-0.5 is "irrelevant" under the fixed sigmoid-0.5
    default (same sanity fact `test_label_for_score_fitted_intercept_
    shifts_the_decision_boundary` establishes directly against
    `label_for_score`), but flips to "relevant" once this persona's
    persisted knob (slope=1.0, intercept=1.0) shifts the boundary past it
    — proving the LOAD path actually reaches the live judge-labeling call,
    not just that `label_for_score` is capable of applying params (inc3)."""
    mem = _mem("borderline content")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
    )
    judge = FakeRelevanceJudgeProvider(scores={("q", mem.content): -0.5})
    store.write_judge_knob_calibration(judge.model_id(), slope=1.0, intercept=1.0)

    labeled = label_calibration_sample(store, judge=judge)

    assert labeled == 1
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    assert json.loads(row["local_judge_label"]) == ["relevant"], (
        "the persona's fitted knob (slope=1.0, intercept=1.0) must shift the boundary past "
        "raw_score=-0.5, which is 'irrelevant' under the fixed sigmoid-0.5 default"
    )


def test_label_calibration_sample_absent_knob_is_byte_identical_to_fixed_default(
    store: MemoryStore,
) -> None:
    """ABSENT-SAFE: no persisted `judge_knob_calibration` row for this
    judge's model id -> labeling of the same borderline raw_score=-0.5
    stays "irrelevant" (the fixed sigmoid-0.5 default), byte-identical to
    pre-inc4a behavior — not silently "relevant" from some fabricated
    default."""
    mem = _mem("borderline content")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
    )
    judge = FakeRelevanceJudgeProvider(scores={("q", mem.content): -0.5})
    assert store.get_judge_knob_calibration(judge.model_id()) is None, "sanity: no persisted knob"

    labeled = label_calibration_sample(store, judge=judge)

    assert labeled == 1
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    assert json.loads(row["local_judge_label"]) == ["irrelevant"], (
        "an absent knob must fall back to the fixed sigmoid-0.5 default, unchanged from pre-inc4a"
    )


def test_label_calibration_sample_persona_isolation_no_cross_persona_bleed() -> None:
    """AC11 (load side): persona A's persisted knob must never affect
    persona B's judge labeling. Both personas' judges share the SAME
    `model_id()` string ("fake-relevance-judge") — deliberately, so the
    isolation this proves comes from each persona's own `MemoryStore`/db
    file (I1), not from any persona-scoping column keying the row."""
    store_a = MemoryStore(db_path=":memory:")
    store_b = MemoryStore(db_path=":memory:")

    mem_a = _mem("borderline content a")
    store_a.create(mem_a)
    store_a.log_calibration_sample(
        query="q", candidate_ids=[mem_a.id], reranker_scores=[1.0], reranker_model_id="m"
    )
    judge_a = FakeRelevanceJudgeProvider(scores={("q", mem_a.content): -0.5})
    store_a.write_judge_knob_calibration(judge_a.model_id(), slope=1.0, intercept=1.0)

    mem_b = _mem("borderline content b")
    store_b.create(mem_b)
    store_b.log_calibration_sample(
        query="q", candidate_ids=[mem_b.id], reranker_scores=[1.0], reranker_model_id="m"
    )
    judge_b = FakeRelevanceJudgeProvider(scores={("q", mem_b.content): -0.5})
    # store_b (persona B) never gets write_judge_knob_calibration called on it.

    assert label_calibration_sample(store_a, judge=judge_a) == 1
    assert label_calibration_sample(store_b, judge=judge_b) == 1

    row_a = store_a._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    row_b = store_b._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    assert json.loads(row_a["local_judge_label"]) == ["relevant"], "persona A's own fitted knob applies"
    assert json.loads(row_b["local_judge_label"]) == ["irrelevant"], (
        "persona B has no persisted knob of its own and must not inherit persona A's, "
        "even though both judges share the same model_id() string"
    )


def test_label_calibration_sample_with_persona_knob_does_not_import_torch() -> None:
    """AC8 / I6: loading a persona's persisted knob (F2c inc4a) at the
    judge-label call site is a plain SQLite read (`MemoryStore.get_judge_
    knob_calibration`) — it must not pull torch/sentence_transformers into
    `sys.modules`. Fresh subprocess (mirrors test_judge_selftune.py's
    `test_tick_does_not_import_torch_or_sentence_transformers`) so an
    earlier test's own torch import in this same process can't make an
    in-process `sys.modules` check meaningless.
    """
    script = textwrap.dedent(
        """
        import json
        import sys

        from brain.memory.relevance_judge import FakeRelevanceJudgeProvider, label_calibration_sample
        from brain.memory.store import Memory, MemoryStore

        store = MemoryStore(db_path=":memory:")
        mem = Memory.create_new(content="borderline content", memory_type="conversation", domain="us")
        store.create(mem)
        store.log_calibration_sample(
            query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
        )
        judge = FakeRelevanceJudgeProvider(scores={("q", mem.content): -0.5})
        store.write_judge_knob_calibration(judge.model_id(), slope=1.0, intercept=1.0)

        labeled = label_calibration_sample(store, judge=judge)
        assert labeled == 1
        row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
        assert json.loads(row["local_judge_label"]) == ["relevant"], "the fitted knob must still apply"

        assert "torch" not in sys.modules, sorted(sys.modules)
        assert "sentence_transformers" not in sys.modules, sorted(sys.modules)
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


# ---------------------------------------------------------------------------
# F2c inc6/inc7 — FullModelJudge serves the persona's ONE tuned plain
# checkpoint. FullModelJudge's real load is torch; tests monkeypatch
# judge_full_ft.load_full_scorer so no model is built.
# ---------------------------------------------------------------------------


def test_build_judge_provider_full_model_dir_returns_full_model_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # C11: full_model_dir -> FullModelJudge; model_id is the BASE id (so the
    # per-persona knob lookup stays correct); a plain scorer produces a score.
    from brain.memory import judge_full_ft
    from brain.memory.relevance_judge import FullModelJudge

    _reset_judge_provider_cache()
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "BASE-JUDGE-ID")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")
    monkeypatch.setattr(judge_full_ft, "load_full_scorer", lambda d, **kw: (lambda item: 2.5))

    judge = build_judge_provider(full_model_dir="/tmp/full-x")
    assert isinstance(judge, FullModelJudge)
    assert judge.model_id() == "BASE-JUDGE-ID"
    assert judge.score("q", "d") == pytest.approx(2.5)


def test_build_judge_provider_has_no_adapter_argument_and_no_adapter_judge() -> None:
    # V1 (F2c inc7, ruling Q3): serving never loads an adapter — the provider
    # takes only `full_model_dir`, and the adapter judge class is gone.
    # Bite: at d25e1002 the signature is (adapter_dir, full_model_dir) and
    # LoraAdapterJudge exists.
    import inspect

    assert list(inspect.signature(build_judge_provider).parameters) == ["full_model_dir"]
    assert not hasattr(rj_mod, "LoraAdapterJudge")
    assert list(inspect.signature(label_calibration_sample).parameters)[-1] == "full_model_dir"
    assert "adapter_dir" not in inspect.signature(label_calibration_sample).parameters


def test_full_model_judge_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    # C13: full-model judges (a new accepted model each week) must not
    # accumulate in the process-wide provider cache.
    from brain.memory import judge_full_ft
    from brain.memory.relevance_judge import FullModelJudge, _provider_cache

    _reset_judge_provider_cache()
    monkeypatch.setattr("brain.bridge.model_tier.model_for_tier", lambda tier: "BASE-JUDGE-ID")
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: "/tmp/fake-cache-dir")
    monkeypatch.setattr(judge_full_ft, "load_full_scorer", lambda d, **kw: (lambda item: 0.0))

    for i in range(5):
        judge = build_judge_provider(full_model_dir=f"/tmp/full-{i}")
        assert isinstance(judge, FullModelJudge)
    assert len(_provider_cache) == 0, "full-model judges must not be cached"
    _reset_judge_provider_cache()


def test_default_suite_judge_stub_matches_the_real_signature(
    request: pytest.FixtureRequest,
) -> None:
    # V3 (F2c inc7): the autouse conftest stub must accept exactly what the
    # real provider accepts; a mismatch raises inside the stub and is
    # swallowed by label_calibration_sample's fault isolation (bite: at
    # d25e1002 the stub was `lambda adapter_dir=None` while the real call
    # passed full_model_dir).
    import inspect

    stub = rj_mod.build_judge_provider  # patched per-test by the autouse fixture
    assert stub is not build_judge_provider, "autouse stub must be active in this test"
    assert list(inspect.signature(stub).parameters) == list(inspect.signature(build_judge_provider).parameters)


def test_labeling_pass_on_the_default_stub_labels_rows(store: MemoryStore) -> None:
    # V3: no injected judge -> the default stub is built and rows get labeled.
    mem = _mem("x")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
    )
    assert label_calibration_sample(store, judge=None) == 1


def test_judge_construction_failure_is_logged_at_warning_or_above(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # V4: a judge-construction failure returns 0 AND is logged at >= WARNING
    # (relevance_judge.label_calibration_sample's except branch), never silent.
    mem = _mem("x")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
    )

    def _raise(full_model_dir=None):
        raise TypeError("unexpected keyword argument")

    monkeypatch.setattr(rj_mod, "build_judge_provider", _raise)
    with caplog.at_level("DEBUG", logger="brain.memory.relevance_judge"):
        assert label_calibration_sample(store, judge=None) == 0
    assert any(
        r.levelno >= 30 and "failed to construct the local judge provider" in r.getMessage()
        for r in caplog.records
    )


def test_judge_construction_failure_oracle_can_fail(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # V4 self-test (ST1.5f): with the module logger's `exception` silenced, the
    # same observation finds no >= WARNING record — so the V4 check can fail.
    mem = _mem("x")
    store.create(mem)
    store.log_calibration_sample(
        query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m"
    )

    def _raise(full_model_dir=None):
        raise TypeError("unexpected keyword argument")

    monkeypatch.setattr(rj_mod, "build_judge_provider", _raise)
    monkeypatch.setattr(rj_mod.logger, "exception", lambda *a, **k: None)
    with caplog.at_level("DEBUG", logger="brain.memory.relevance_judge"):
        assert label_calibration_sample(store, judge=None) == 0
    assert not any(
        r.levelno >= 30 and "failed to construct the local judge provider" in r.getMessage()
        for r in caplog.records
    )
