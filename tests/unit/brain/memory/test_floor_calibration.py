"""Tests for brain.memory.floor_calibration — the F2a #250 inc7 reranker
abstention-floor derivation (spec Section 7): threshold fit, EMA smoothing,
the bootstrap stability gate, cold-start entry/exit, and per-persona
persistence via MemoryStore.

All OFFLINE — no real model download, no network. Cold-start tests inject a
FakeRerankerProvider (mirrors reranker.py's own test convention) rather than
constructing the real jina provider.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.memory.floor_calibration import (
    FLOOR_FIT_MIN_LABELED_PAIRS,
    RETENTION_WINDOW_DAYS_DEFAULT,
    derive_and_persist_floor,
    derive_retention_window_days,
    ema_update,
    fit_threshold_fbeta,
    stability_gate_accepts,
)
from brain.memory.reranker import FakeRerankerProvider
from brain.memory.store import MemoryStore

MODEL_ID = "fake-reranker-for-floor-tests"


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


def _seed_labeled_row(
    store: MemoryStore,
    scores: list[float],
    labels: list[str],
    *,
    reranker_model_id: str = MODEL_ID,
    haiku_labels: list[str | None] | None = None,
) -> None:
    """Insert one already-labeled calibration_log row directly (bypassing
    log_calibration_sample + write_calibration_labels' two-step API, since
    these tests want the row fully labeled in one shot)."""
    import json

    if haiku_labels is None:
        haiku_labels = [None] * len(labels)
    store._conn.execute(
        "INSERT INTO calibration_log "
        "(query, candidate_ids, reranker_scores, reranker_model_id, local_judge_label, haiku_label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "some query",
            json.dumps([f"m{i}" for i in range(len(scores))]),
            json.dumps(scores),
            reranker_model_id,
            json.dumps(labels),
            json.dumps(haiku_labels),
        ),
    )
    store._conn.commit()


# ---------------------------------------------------------------------------
# fit_threshold_fbeta — Youden's-J / F-beta cutoff fit.
# ---------------------------------------------------------------------------


def test_fit_threshold_separates_two_well_clustered_classes() -> None:
    """A clean-separation case: relevant scores cluster high, irrelevant
    low. The fitted threshold must land strictly between the two clusters,
    not at any hardcoded value."""
    pairs = [(s, "relevant") for s in [8.0, 9.0, 10.0, 8.5, 9.5]] + [
        (s, "irrelevant") for s in [-8.0, -9.0, -10.0, -8.5, -9.5]
    ]
    threshold = fit_threshold_fbeta(pairs, beta=2.0)
    assert -8.0 < threshold < 8.0, f"threshold {threshold} must separate the two clean clusters"


def test_fit_threshold_is_recall_leaning_on_overlapping_distributions() -> None:
    """Acceptance #8: when the two distributions OVERLAP, the beta>1
    (recall-leaning) fit must sit on the IRRELEVANT side of the naive
    midpoint between class means — i.e. more inclusive/permissive than a
    beta=1 (balanced) fit would be, biasing toward not missing a true
    positive over not admitting a false one."""
    relevant = [0.0, 1.0, 2.0, 3.0, 4.0]
    irrelevant = [-4.0, -3.0, -2.0, -1.0, 0.0]  # overlaps relevant at 0.0
    pairs = [(s, "relevant") for s in relevant] + [(s, "irrelevant") for s in irrelevant]

    naive_midpoint = (np.mean(relevant) + np.mean(irrelevant)) / 2.0
    recall_leaning = fit_threshold_fbeta(pairs, beta=2.0)
    balanced = fit_threshold_fbeta(pairs, beta=1.0)

    assert recall_leaning <= balanced, (
        "a higher beta (more recall-leaning) must never sit ABOVE the balanced (beta=1) threshold"
    )
    assert recall_leaning < naive_midpoint, (
        "recall-leaning threshold must sit on the irrelevant side of the naive class-mean midpoint"
    )


def test_fit_threshold_all_relevant_serves_everything() -> None:
    pairs = [(1.0, "relevant"), (2.0, "relevant"), (0.5, "relevant")]
    threshold = fit_threshold_fbeta(pairs, beta=2.0)
    assert threshold < min(s for s, _ in pairs)


def test_fit_threshold_all_irrelevant_serves_nothing() -> None:
    pairs = [(1.0, "irrelevant"), (2.0, "irrelevant"), (0.5, "irrelevant")]
    threshold = fit_threshold_fbeta(pairs, beta=2.0)
    assert threshold > max(s for s, _ in pairs)


def test_fit_threshold_requires_at_least_one_pair() -> None:
    with pytest.raises(ValueError):
        fit_threshold_fbeta([], beta=2.0)


# ---------------------------------------------------------------------------
# ema_update
# ---------------------------------------------------------------------------


def test_ema_update_with_no_prior_returns_raw_floor_unchanged() -> None:
    assert ema_update(None, 5.0, window_days=7.0) == 5.0


def test_ema_update_moves_partway_toward_the_new_raw_floor() -> None:
    updated = ema_update(0.0, 10.0, window_days=7.0)
    alpha = 2.0 / 8.0
    assert updated == pytest.approx(alpha * 10.0)
    assert 0.0 < updated < 10.0, "EMA must move PARTWAY, never jump straight to the raw value"


def test_ema_update_with_matching_prior_and_raw_is_a_no_op() -> None:
    assert ema_update(5.0, 5.0, window_days=7.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# stability_gate_accepts — bootstrap-CI vs previous EMA, both directions.
# ---------------------------------------------------------------------------


def test_stability_gate_accepts_when_no_prior_floor_exists() -> None:
    pairs = [(1.0, "relevant"), (-1.0, "irrelevant")]
    rng = np.random.default_rng(0)
    assert stability_gate_accepts(None, pairs, beta=2.0, ci=0.95, iterations=50, rng=rng) is True


def test_stability_gate_accepts_when_prior_floor_is_consistent_with_todays_data() -> None:
    """A day whose labeled sample is CONSISTENT with the established floor
    must clear the gate (accept direction)."""
    rng = np.random.default_rng(1)
    relevant = [4.0 + i * 0.1 for i in range(30)]
    irrelevant = [-4.0 - i * 0.1 for i in range(30)]
    pairs = [(s, "relevant") for s in relevant] + [(s, "irrelevant") for s in irrelevant]
    prior_floor = fit_threshold_fbeta(pairs, beta=2.0)  # squarely inside this sample's own CI

    accepted = stability_gate_accepts(prior_floor, pairs, beta=2.0, ci=0.95, iterations=200, rng=rng)
    assert accepted is True


def test_stability_gate_trips_on_a_wildly_inconsistent_noisy_day() -> None:
    """Acceptance #9 (reject direction): a prior floor established far away
    from today's tiny, tightly-clustered, wildly different distribution
    must NOT fall inside today's bootstrap CI -> the gate trips (holds)."""
    rng = np.random.default_rng(2)
    # Today's noisy day: a small, tightly clustered sample far from the
    # established prior.
    pairs = [(100.0, "relevant"), (100.1, "relevant"), (99.9, "irrelevant"), (100.05, "irrelevant")]
    prior_floor = -50.0  # nowhere near today's tightly-clustered ~100 range

    accepted = stability_gate_accepts(prior_floor, pairs, beta=2.0, ci=0.95, iterations=200, rng=rng)
    assert accepted is False, "a wildly inconsistent noisy day must trip the gate, not swing the floor"


# ---------------------------------------------------------------------------
# derive_retention_window_days — Part C finalization (acceptance 5b's
# formula, not the pruning mechanics themselves — see test_store.py /
# test_calibration_cadence.py for the prune behavior).
# ---------------------------------------------------------------------------


def test_derive_retention_window_days_is_max_of_the_two_bounds() -> None:
    assert derive_retention_window_days(
        min_labeled_pairs=200, ema_window_days=7.0, worst_case_daily_yield=100
    ) == 7.0  # ceil(200/100)=2 < 7.0
    assert derive_retention_window_days(
        min_labeled_pairs=1000, ema_window_days=7.0, worst_case_daily_yield=50
    ) == 20.0  # ceil(1000/50)=20 > 7.0


def test_derive_retention_window_days_not_hardcoded_14() -> None:
    """The whole point of inc7's Part C: this must be a computed value tied
    to the registered tunables (traceable back to `derive_retention_window_
    days()`'s own formula), not a re-typed 14.0 magic number. Changing
    EITHER input changes the output — proof it is a live derivation, not a
    constant that merely happens to be computed once at import time."""
    assert RETENTION_WINDOW_DAYS_DEFAULT == derive_retention_window_days()
    changed = derive_retention_window_days(
        min_labeled_pairs=FLOOR_FIT_MIN_LABELED_PAIRS * 100, ema_window_days=7.0, worst_case_daily_yield=100
    )
    assert changed != RETENTION_WINDOW_DAYS_DEFAULT


# ---------------------------------------------------------------------------
# derive_and_persist_floor — full orchestration: cold-start, real fit,
# EMA, stability gate, persistence round-trip.
# ---------------------------------------------------------------------------


def test_cold_start_when_no_real_labeled_data_exists(store: MemoryStore) -> None:
    # First 3 pairs of reranker._FP16_GATE_PAIRS are "genuine" (labeled
    # relevant), next 3 are "decoy" (labeled irrelevant) —
    # FakeRerankerProvider scores by DOCUMENT text only, so key on the doc
    # half of each bundled (query, doc) pair.
    from brain.memory.reranker import _FP16_GATE_PAIRS

    scores_by_doc = {doc: (10.0 if i < 3 else -10.0) for i, (_query, doc) in enumerate(_FP16_GATE_PAIRS[:6])}
    provider = FakeRerankerProvider(scores=scores_by_doc)

    outcome = derive_and_persist_floor(store, MODEL_ID, reranker_provider=provider)

    assert outcome.accepted is True
    assert outcome.is_cold_start is True
    assert outcome.sample_pairs == 6
    assert -10.0 < outcome.floor < 10.0

    persisted = store.get_reranker_floor(MODEL_ID)
    assert persisted is not None
    assert persisted["is_cold_start"] is True
    assert persisted["floor"] == pytest.approx(outcome.floor)


def test_cold_start_persists_even_with_zero_calibration_log_rows(store: MemoryStore) -> None:
    """Acceptance #10's inc7-scoped slice: a totally fresh install (zero
    calibration_log rows at all) must still get a servable bootstrap floor,
    never a crash or an unset floor."""
    from brain.memory.reranker import _FP16_GATE_PAIRS

    scores_by_doc = {doc: (5.0 if i < 3 else -5.0) for i, (_q, doc) in enumerate(_FP16_GATE_PAIRS[:6])}
    provider = FakeRerankerProvider(scores=scores_by_doc)

    assert store.labeled_calibration_pairs(MODEL_ID) == []
    outcome = derive_and_persist_floor(store, MODEL_ID, reranker_provider=provider)
    assert outcome.accepted is True
    assert store.get_reranker_floor(MODEL_ID) is not None


def test_cold_start_exit_is_outcome_based_not_day_counted(store: MemoryStore) -> None:
    """Feed exactly (min_labeled_pairs - 1) usable real pairs -> still cold
    start; feed exactly min_labeled_pairs -> a REAL fit runs. No dates or
    day counts are involved anywhere in this test — purely a row-count
    outcome, proving the exit condition is NOT day-based."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    from brain.memory.reranker import _FP16_GATE_PAIRS

    scores_by_doc = {doc: (5.0 if i < 3 else -5.0) for i, (_q, doc) in enumerate(_FP16_GATE_PAIRS[:6])}
    provider = FakeRerankerProvider(scores=scores_by_doc)

    # One short of the threshold — every row is a single-candidate row so
    # row count == pair count.
    for i in range(min_pairs - 1):
        label = "relevant" if i % 2 == 0 else "irrelevant"
        _seed_labeled_row(store, [float(i)], [label])
    outcome = derive_and_persist_floor(store, MODEL_ID, reranker_provider=provider)
    assert outcome.is_cold_start is True, "one pair short of the threshold must still be cold-start"

    # Cross the threshold with one more labeled pair.
    _seed_labeled_row(store, [999.0], ["relevant"])
    outcome2 = derive_and_persist_floor(store, MODEL_ID, reranker_provider=provider)
    assert outcome2.is_cold_start is False, "crossing the threshold must exit cold-start immediately"
    assert outcome2.sample_pairs == min_pairs


def test_real_fit_first_derivation_has_no_prior_history_and_trivially_accepts(
    store: MemoryStore,
) -> None:
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    for i in range(min_pairs):
        if i < min_pairs // 2:
            _seed_labeled_row(store, [10.0 + i * 0.01], ["relevant"])
        else:
            _seed_labeled_row(store, [-10.0 - i * 0.01], ["irrelevant"])

    outcome = derive_and_persist_floor(store, MODEL_ID, rng=np.random.default_rng(3))
    assert outcome.accepted is True
    assert outcome.is_cold_start is False
    assert outcome.held_for_stability is False
    assert outcome.floor == pytest.approx(outcome.raw_fit_floor), (
        "first-ever real derivation has no EMA history -> raw fit applied unchanged"
    )

    persisted = store.get_reranker_floor(MODEL_ID)
    assert persisted is not None
    assert persisted["is_cold_start"] is False
    assert persisted["floor"] == pytest.approx(outcome.floor)


def _seed_noisy_labeled_pairs(
    store: MemoryStore, rng: np.random.Generator, *, n: int, relevant_loc: float, irrelevant_loc: float
) -> None:
    """Seed `n` labeled rows drawn from two OVERLAPPING gaussian clusters
    (real within-class variance, not a razor-thin perfectly-separated gap)
    so the bootstrap gate has genuine width to work with, mirroring a real
    day's noisy score distribution rather than a synthetic single-candidate
    boundary."""
    half = n // 2
    relevant_scores = rng.normal(loc=relevant_loc, scale=1.5, size=half)
    irrelevant_scores = rng.normal(loc=irrelevant_loc, scale=1.5, size=n - half)
    for s in relevant_scores:
        _seed_labeled_row(store, [float(s)], ["relevant"])
    for s in irrelevant_scores:
        _seed_labeled_row(store, [float(s)], ["irrelevant"])


def test_real_fit_second_consistent_day_ema_smooths_partway(store: MemoryStore) -> None:
    """Accept direction of acceptance #9: given a SEEDED prior floor that a
    fresh day's noisy-but-consistent labeled sample's bootstrap CI comfortably
    contains, the update is accepted and EMA-BLENDED — never equal to either
    the untouched prior or the day's own raw fit outright."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    rng_seed = np.random.default_rng(42)
    _seed_noisy_labeled_pairs(store, rng_seed, n=min_pairs, relevant_loc=5.0, irrelevant_loc=-5.0)

    # Seed a prior EMA floor squarely inside where this data's own fit lands
    # (both clusters centered +/-5 with scale 1.5 -> a threshold near 0.0 is
    # well inside a 95% bootstrap CI of the resulting fit).
    store.write_reranker_floor(
        MODEL_ID, floor=0.0, raw_fit_floor=0.0, sample_pairs=min_pairs, is_cold_start=False
    )

    outcome = derive_and_persist_floor(store, MODEL_ID, rng=np.random.default_rng(43))

    assert outcome.accepted is True
    assert outcome.held_for_stability is False
    assert outcome.floor != pytest.approx(outcome.raw_fit_floor), (
        "an accepted update against an existing prior must be EMA-blended, not equal to its own raw fit"
    )
    assert outcome.floor != pytest.approx(0.0), (
        "an accepted update must actually MOVE from the untouched prior, not leave it exactly in place"
    )
    lo, hi = sorted([0.0, outcome.raw_fit_floor])
    assert lo <= outcome.floor <= hi, "EMA must land strictly between the prior and the raw fit"

    persisted = store.get_reranker_floor(MODEL_ID)
    assert persisted["floor"] == pytest.approx(outcome.floor)


def test_real_fit_noisy_day_holds_and_leaves_persisted_floor_unchanged(store: MemoryStore) -> None:
    """Reject direction of acceptance #9: a SEEDED prior floor that sits
    WAY outside a fresh day's own bootstrap CI (the noisy/outlier-day
    scenario) must HOLD — the persisted floor stays byte-for-byte what was
    seeded, and `accepted` is False."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    rng_seed = np.random.default_rng(44)
    # Today's data fits tightly around 0.0 (clusters at +/-5, scale 1.5).
    _seed_noisy_labeled_pairs(store, rng_seed, n=min_pairs, relevant_loc=5.0, irrelevant_loc=-5.0)

    # A prior floor established WAY outside where today's data could
    # plausibly land — the "one wildly different day" scenario, just
    # expressed as "the established history is now wildly far from today."
    store.write_reranker_floor(
        MODEL_ID, floor=500.0, raw_fit_floor=500.0, sample_pairs=min_pairs, is_cold_start=False
    )
    persisted_before = store.get_reranker_floor(MODEL_ID)

    outcome = derive_and_persist_floor(store, MODEL_ID, rng=np.random.default_rng(45))

    assert outcome.accepted is False
    assert outcome.held_for_stability is True
    assert outcome.floor == pytest.approx(500.0), "a held cycle reports the still-standing prior floor"
    persisted_after = store.get_reranker_floor(MODEL_ID)
    assert persisted_after == persisted_before, (
        "a held cycle must leave the persisted floor row byte-for-byte unchanged"
    )


def test_floor_derivation_per_persona_scoped_by_being_a_fresh_store(store: MemoryStore) -> None:
    """'Per-persona' persistence (I1) means: lives in THIS persona's own
    memories.db, keyed by reranker_model_id within it — not a lookup across
    personas. A second, independent MemoryStore never sees the first's
    PERSISTED floor (F2a inc8: it still gets a floor — the transient
    bootstrap, computed fresh for itself — but never the first store's
    ACTUAL persisted/derived value; this is the distinction that still
    proves per-persona scoping under the bootstrap ruling)."""
    from brain.memory.reranker import _FP16_GATE_PAIRS

    scores_by_doc = {doc: (5.0 if i < 3 else -5.0) for i, (_q, doc) in enumerate(_FP16_GATE_PAIRS[:6])}
    provider = FakeRerankerProvider(scores=scores_by_doc)
    derive_and_persist_floor(store, MODEL_ID, reranker_provider=provider)
    first_persisted = store.get_reranker_floor(MODEL_ID)
    assert first_persisted is not None
    assert first_persisted["is_cold_start"] is True

    other_persona_store = MemoryStore(db_path=":memory:")
    other_result = other_persona_store.get_reranker_floor(MODEL_ID)
    assert other_result is not None, "F2a inc8: a fresh store still gets a bootstrap floor, not None"
    assert other_result["updated_at"] is None, (
        "the second store must get the TRANSIENT bootstrap, never a persisted row — proving it "
        "never read the first persona's actual persisted/derived floor"
    )
    count = other_persona_store._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) AS n FROM reranker_floor_calibration WHERE reranker_model_id = ?",
        (MODEL_ID,),
    ).fetchone()["n"]
    assert count == 0, "the second persona's OWN table must never gain a row just from reading the bootstrap"


# ---------------------------------------------------------------------------
# get_bootstrap_floor — F2a inc8 (#250 §7 UPDATED, Roy 2026-09-18): the
# hot-path DEFAULT `get_reranker_floor` serves instead of None when no
# persisted row exists yet. All offline via a scripted FakeRerankerProvider
# monkeypatched onto `reranker._bootstrap_reranker_provider` directly (the
# autouse conftest fixture already defaults this to an UNSCRIPTED
# FakeRerankerProvider for the rest of the suite — these tests override it
# per-test to control the exact scores the fit sees).
# ---------------------------------------------------------------------------


def _install_bootstrap_provider(
    monkeypatch: pytest.MonkeyPatch, scores_by_doc: dict[str, float]
) -> None:
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(
        "brain.memory.reranker._bootstrap_reranker_provider",
        lambda model_id: FakeRerankerProvider(scores=scores_by_doc),
    )


def _bundled_scores(relevant_score: float, irrelevant_score: float) -> dict[str, float]:
    from brain.memory.reranker import _FP16_GATE_PAIRS

    return {
        doc: (relevant_score if i < 3 else irrelevant_score)
        for i, (_query, doc) in enumerate(_FP16_GATE_PAIRS[:6])
    }


def test_bootstrap_floor_lands_on_a_reachable_jina_scale_value_not_the_dead_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap must be a REAL fit result on a reachable jina-scale
    logit (jina's raw scores run roughly -4..+4, per the spec) — never the
    outgoing dead `-9.25` constant, and never an arbitrary hardcoded pin."""
    from brain.memory import floor_calibration

    floor_calibration._reset_bootstrap_floor_cache()
    _install_bootstrap_provider(monkeypatch, _bundled_scores(2.5, -3.0))

    result = floor_calibration.get_bootstrap_floor("bootstrap-jina-scale-test-model")

    assert result is not None
    assert result["floor"] != pytest.approx(-9.25), "must not be the dead ported MiniLM constant"
    assert -3.0 < result["floor"] < 2.5, (
        "the fitted floor must sit strictly between the relevant/irrelevant clusters it was fit "
        "from — a reachable value, not an extreme/placeholder"
    )
    assert result["raw_fit_floor"] == pytest.approx(result["floor"]), (
        "the bootstrap has no EMA history to smooth against — raw fit applied unchanged"
    )
    assert result["is_cold_start"] is True
    assert result["sample_pairs"] == 6


def test_bootstrap_floor_changes_when_the_bundled_pair_scores_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I3 proof: the bootstrap comes from a REAL fit over the bundled pairs,
    not a hardcoded number — changing what the reranker provider scores
    those pairs must change the resulting floor."""
    from brain.memory import floor_calibration

    model_id = "bootstrap-derivation-proof-model"

    floor_calibration._reset_bootstrap_floor_cache()
    _install_bootstrap_provider(monkeypatch, _bundled_scores(10.0, -10.0))
    floor_a = floor_calibration.get_bootstrap_floor(model_id)["floor"]

    floor_calibration._reset_bootstrap_floor_cache()
    _install_bootstrap_provider(monkeypatch, _bundled_scores(1.0, 0.5))
    floor_b = floor_calibration.get_bootstrap_floor(model_id)["floor"]

    assert floor_a != pytest.approx(floor_b), (
        "a differently-scored bundled set must fit to a DIFFERENT floor — proof this is a live "
        "derivation, not a constant that merely happens to be computed once"
    )


def test_bootstrap_floor_is_computed_once_and_cached_not_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I6: the fit must run ONCE per model_id and be served from cache on
    every later no-row `get_reranker_floor` call — never recomputed per
    recall. Patches `fit_threshold_fbeta` itself (bare-name call inside
    `get_bootstrap_floor`) to count invocations while still delegating to
    the real implementation, so the assertion is on CALL COUNT, not on the
    result changing."""
    from brain.memory import floor_calibration

    floor_calibration._reset_bootstrap_floor_cache()
    _install_bootstrap_provider(monkeypatch, _bundled_scores(5.0, -5.0))

    call_count = {"n": 0}
    real_fit = floor_calibration.fit_threshold_fbeta

    def _counting_fit(pairs, *, beta):
        call_count["n"] += 1
        return real_fit(pairs, beta=beta)

    monkeypatch.setattr(floor_calibration, "fit_threshold_fbeta", _counting_fit)

    store = MemoryStore(db_path=":memory:")
    results = [store.get_reranker_floor("bootstrap-cache-once-test-model") for _ in range(5)]

    assert all(r is not None for r in results)
    assert all(r["floor"] == pytest.approx(results[0]["floor"]) for r in results), (
        "every call must serve the SAME cached value"
    )
    assert call_count["n"] == 1, "the fit must run exactly once across 5 repeated no-row calls"


def test_bootstrap_floor_never_touches_torch_or_the_relevance_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2a inc8 COMPUTE CONSTRAINT (spec Section 7, load-bearing): only jina
    (ONNX, via `reranker.CrossEncoderProvider`/`_bootstrap_reranker_
    provider`) may score the bundled pairs — the §6 torch-backed relevance
    judge must NEVER be invoked, and this computation must never newly
    import torch (a bootstrap that dragged torch onto the recall path would
    regress the 'torch scoped to the offline judge only' decision).

    Two independent proofs: (1) `relevance_judge.build_judge_provider` is
    poisoned to raise if called at all — a call-count of zero is the only
    way this test passes; (2) `sys.modules`'s torch membership is compared
    BEFORE vs AFTER (not asserted absent outright — some earlier test in
    the same process may have already imported torch for unrelated reasons,
    so only a CHANGE caused by this call would indicate a regression)."""
    import sys

    from brain.memory import floor_calibration, relevance_judge

    floor_calibration._reset_bootstrap_floor_cache()
    _install_bootstrap_provider(monkeypatch, _bundled_scores(5.0, -5.0))

    def _poisoned_judge_provider():
        raise AssertionError("the bootstrap path must never build/invoke the offline relevance judge")

    monkeypatch.setattr(relevance_judge, "build_judge_provider", _poisoned_judge_provider)

    torch_loaded_before = "torch" in sys.modules

    result = floor_calibration.get_bootstrap_floor("bootstrap-torch-guard-test-model")

    assert result is not None
    assert ("torch" in sys.modules) == torch_loaded_before, (
        "the bootstrap computation must never newly import torch"
    )


def test_bootstrap_floor_fails_soft_caches_nothing_and_retries_on_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft contract (docstring: "FAIL-SOFT ... any failure ... is
    caught, logged, and returns None"), the part left untested: if computing
    the bootstrap raises (provider build or fit failure), `get_bootstrap_
    floor` must (a) catch the exception and never let it propagate, (b)
    return None, (c) cache NOTHING for that model_id (a poisoned/empty cache
    entry must not linger), and (d) a SUBSEQUENT call must RE-ATTEMPT the
    computation rather than short-circuit on a cached failure — this is the
    hot-path safety contract: a fresh-install day-0 recall must degrade to
    lexical, never crash, and must not get PERMANENTLY stuck degraded if a
    transient failure (e.g. a momentary provider-build hiccup) clears up.

    Patches `reranker._bootstrap_reranker_provider` itself (the same seam
    `_install_bootstrap_provider` above patches) to raise, with a call
    counter proving BOTH that the failing path is actually exercised and
    that a second call genuinely retries rather than serving a cached
    result."""
    from brain.memory import floor_calibration

    model_id = "bootstrap-fail-soft-retry-test-model"
    floor_calibration._reset_bootstrap_floor_cache()

    call_count = {"n": 0}

    def _raising_bootstrap_provider(model_id: str):
        call_count["n"] += 1
        raise RuntimeError("simulated bootstrap provider build failure")

    monkeypatch.setattr(
        "brain.memory.reranker._bootstrap_reranker_provider",
        _raising_bootstrap_provider,
    )

    result = floor_calibration.get_bootstrap_floor(model_id)
    assert result is None, "a raising bootstrap computation must fail soft to None, never propagate"
    assert call_count["n"] == 1
    assert model_id not in floor_calibration._bootstrap_floor_cache, (
        "a failed computation must cache NOTHING for this model_id"
    )

    result2 = floor_calibration.get_bootstrap_floor(model_id)
    assert result2 is None, "a second call after a failure must still fail soft to None"
    assert call_count["n"] == 2, (
        "a subsequent call must RE-ATTEMPT (recompute) the bootstrap, not serve a poisoned cache "
        "entry from the prior failure"
    )
    assert model_id not in floor_calibration._bootstrap_floor_cache, (
        "the cache must still hold no entry for this model_id after a second failed attempt"
    )
