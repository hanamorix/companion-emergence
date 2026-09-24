"""Tests for brain.memory.floor_calibration — the F2a #250 inc7 reranker
abstention-floor derivation (spec Section 7): threshold fit, cold-start
bootstrap, and per-persona persistence via MemoryStore.

Pre-flip revision Change 1 ("nimble floor", 2026-09-23) drops the EMA
smoothing and bootstrap-CI stability gate that used to sit on top of the
raw fit, and re-points the daily fit from the full multi-day retention pool
to a single day (the most recently completed one). This file's coverage
was rewritten accordingly: `ema_update`/`stability_gate_accepts`/
`_bootstrap_ci` no longer exist (see the dead-code test below), and
`derive_and_persist_floor`'s own cold-start branch (bundled-pair fit,
unconditionally persisted) is gone, replaced by a data-starvation backstop.
`get_bootstrap_floor` (the SEPARATE, still-standing hot-path bootstrap) and
`fit_threshold_fbeta` (the raw fit itself) are UNCHANGED by Change 1 and
keep their existing coverage below.

All OFFLINE — no real model download, no network. Cold-start/bootstrap
tests inject a FakeRerankerProvider (mirrors reranker.py's own test
convention) rather than constructing the real jina provider.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from brain.memory.floor_calibration import (
    FLOOR_FIT_BETA,
    FLOOR_FIT_MIN_LABELED_PAIRS,
    FLOOR_RETENTION_SAFETY_BUFFER_DAYS,
    RETENTION_WINDOW_DAYS_DEFAULT,
    derive_and_persist_floor,
    derive_retention_window_days,
    fit_threshold_fbeta,
)
from brain.memory.reranker import FakeRerankerProvider
from brain.memory.store import MemoryStore

MODEL_ID = "fake-reranker-for-floor-tests"

def _bundled_scores(relevant_score: float, irrelevant_score: float) -> dict[str, float]:
    """`reranker._FP16_GATE_PAIRS[:6]`'s 3 relevant / 3 irrelevant docs
    scored at the given values — the shared fixture every cold-start/bootstrap
    test in this file builds its `FakeRerankerProvider` from."""
    from brain.memory.reranker import _FP16_GATE_PAIRS

    return {
        doc: (relevant_score if i < 3 else irrelevant_score)
        for i, (_query, doc) in enumerate(_FP16_GATE_PAIRS[:6])
    }


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
    day_bucket: str | None = None,
) -> None:
    """Insert one already-labeled calibration_log row directly (bypassing
    log_calibration_sample + write_calibration_labels' two-step API, since
    these tests want the row fully labeled in one shot).

    `day_bucket` (pre-flip revision Change 1): `None` (the default) leaves
    the column's own `strftime('%Y-%m-%d','now')` default in place — every
    row in a test that never passes this lands on the SAME (today's) day,
    so Change 1's new day-scoping is a no-op for any test that doesn't
    care about it. Tests that DO exercise the day filter pass an explicit
    value.
    """
    import json

    if haiku_labels is None:
        haiku_labels = [None] * len(labels)
    if day_bucket is None:
        store._conn.execute(
            "INSERT INTO calibration_log "
            "(query, candidate_ids, reranker_scores, reranker_model_id, local_judge_label, "
            "haiku_label) "
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
    else:
        store._conn.execute(
            "INSERT INTO calibration_log "
            "(day_bucket, query, candidate_ids, reranker_scores, reranker_model_id, "
            "local_judge_label, haiku_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                day_bucket,
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
# fit_threshold_fbeta — Youden's-J / F-beta cutoff fit. UNCHANGED by Change 1.
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
# derive_retention_window_days — Change 1's SHRUNK derivation: today +
# yesterday (structural) + a safety buffer (tunable), decoupled from the
# removed EMA window and from FLOOR_FIT_MIN_LABELED_PAIRS' worst-case
# multi-day accumulation (neither input applies anymore — the fit reads
# only one day, never pools).
# ---------------------------------------------------------------------------


def test_derive_retention_window_days_is_today_yesterday_plus_buffer() -> None:
    assert derive_retention_window_days(safety_buffer_days=1.0) == 3.0
    assert derive_retention_window_days(safety_buffer_days=0.0) == 2.0, (
        "even a zero buffer must never collapse below the structural today+yesterday minimum"
    )
    assert derive_retention_window_days(safety_buffer_days=5.0) == 7.0


def test_derive_retention_window_days_never_collapses_below_two_days() -> None:
    """Hard rule (Change 1's Open Reconfirmation): retention must retain at
    least today+yesterday+buffer, NOT collapse to <2 days — proven here
    against the registered tunable's own (non-negative) default, not just
    a hand-picked safe input."""
    assert derive_retention_window_days() >= 2.0


def test_derive_retention_window_days_default_is_live_not_a_re_hardcoded_14() -> None:
    """This must be a computed value tied to the registered tunables
    (traceable back to `derive_retention_window_days()`'s own formula), not
    a re-typed magic number (neither the old 14.0 placeholder nor a new
    hardcoded replacement). Changing the buffer input changes the output —
    proof it is a live derivation."""
    assert RETENTION_WINDOW_DAYS_DEFAULT == derive_retention_window_days()
    changed = derive_retention_window_days(safety_buffer_days=FLOOR_RETENTION_SAFETY_BUFFER_DAYS + 10.0)
    assert changed != RETENTION_WINDOW_DAYS_DEFAULT
    assert RETENTION_WINDOW_DAYS_DEFAULT != 14.0


def test_retention_window_derivation_decoupled_from_removed_ema_symbols() -> None:
    """I3/I7 coupling note (Change 1's Open Reconfirmation G): the old
    formula's inputs (`FLOOR_FIT_MIN_LABELED_PAIRS`'s worst-case daily
    yield, `FLOOR_EMA_WINDOW_DAYS`) no longer feed this derivation at all —
    `derive_retention_window_days` takes only `safety_buffer_days` now."""
    import inspect

    sig = inspect.signature(derive_retention_window_days)
    assert list(sig.parameters) == ["safety_buffer_days"]


# ---------------------------------------------------------------------------
# Dead code (acceptance criterion 6): the EMA layer, the bootstrap-CI
# stability gate, and their now-unused constants must be genuinely GONE
# from the module, not merely unused.
# ---------------------------------------------------------------------------


def test_ema_and_stability_gate_symbols_no_longer_exist() -> None:
    from brain.memory import floor_calibration as fc_mod

    for removed_symbol in (
        "ema_update",
        "stability_gate_accepts",
        "_bootstrap_ci",
        "FLOOR_EMA_WINDOW_DAYS",
        "FLOOR_STABILITY_CI",
        "FLOOR_STABILITY_BOOTSTRAP_ITERATIONS",
        "_worst_case_daily_labeled_pairs",
    ):
        assert not hasattr(fc_mod, removed_symbol), (
            f"{removed_symbol} must be removed entirely (Change 1), not just unused"
        )


def test_derive_and_persist_floor_no_longer_takes_provider_or_rng() -> None:
    """The removed cold-start branch was the only thing here that ever
    needed a live reranker provider; the removed stability gate was the
    only thing that ever needed a bootstrap RNG. Neither parameter should
    exist on the signature anymore."""
    import inspect

    sig = inspect.signature(derive_and_persist_floor)
    assert list(sig.parameters) == ["store", "reranker_model_id"]


# ---------------------------------------------------------------------------
# derive_and_persist_floor — Change 1's nimble fit + data-starvation
# backstop. Acceptance criteria 3, 4, 5(a)(b)(c) below.
# ---------------------------------------------------------------------------


def _seed_min_pairs(store: MemoryStore, *, n: int, relevant_loc: float, irrelevant_loc: float,
                     scale: float = 1.5, rng: np.random.Generator, day_bucket: str | None = None,
                     reranker_model_id: str = MODEL_ID) -> None:
    """Seed `n` labeled rows drawn from two overlapping gaussian clusters
    (real within-class variance, mirroring a genuine day's noisy score
    distribution) for `reranker_model_id`, optionally on a specific
    `day_bucket`."""
    half = n // 2
    relevant_scores = rng.normal(loc=relevant_loc, scale=scale, size=half)
    irrelevant_scores = rng.normal(loc=irrelevant_loc, scale=scale, size=n - half)
    for s in relevant_scores:
        _seed_labeled_row(
            store, [float(s)], ["relevant"], day_bucket=day_bucket, reranker_model_id=reranker_model_id
        )
    for s in irrelevant_scores:
        _seed_labeled_row(
            store, [float(s)], ["irrelevant"], day_bucket=day_bucket, reranker_model_id=reranker_model_id
        )


def test_real_fit_lands_exactly_at_the_raw_value_no_ema(store: MemoryStore) -> None:
    """Acceptance #3 ("No EMA"): a fresh day's fit, when it refits, lands
    EXACTLY at the raw Youden's-J/F-beta value for that day's distribution
    — float-identical, not partway between old and new. Seeds a PRIOR
    persisted floor far away first, so a pre-Change-1 EMA blend would have
    visibly landed somewhere between the two; the fix must land exactly on
    the raw fit regardless."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    rng = np.random.default_rng(1)

    store.write_reranker_floor(
        MODEL_ID, floor=-500.0, raw_fit_floor=-500.0, sample_pairs=min_pairs, is_cold_start=False
    )
    _seed_min_pairs(store, n=min_pairs, relevant_loc=5.0, irrelevant_loc=-5.0, rng=rng)

    real_pairs = store.labeled_calibration_pairs(MODEL_ID)
    expected_raw = fit_threshold_fbeta(real_pairs, beta=FLOOR_FIT_BETA)

    outcome = derive_and_persist_floor(store, MODEL_ID)

    assert outcome.accepted is True
    assert outcome.held_for_data_starvation is False
    assert outcome.floor == pytest.approx(expected_raw)
    assert outcome.raw_fit_floor == pytest.approx(expected_raw)
    assert outcome.floor == outcome.raw_fit_floor, "an accepted cycle's floor IS its raw fit, always"
    assert outcome.floor != pytest.approx(-500.0), (
        "must NOT be anywhere near the seeded prior — no EMA blend toward it"
    )

    persisted = store.get_persisted_reranker_floor(MODEL_ID)
    assert persisted["floor"] == pytest.approx(expected_raw)


def test_noisy_but_sufficient_day_still_refits_directly_no_gate(store: MemoryStore) -> None:
    """Acceptance #4 ("No gate"): a single deliberately noisy day (a
    synthetic outlier distribution, >= FLOOR_FIT_MIN_LABELED_PAIRS pairs)
    still refits DIRECTLY to that day's raw fit — no held/blocked case,
    even against a wildly distant seeded prior that the OLD stability gate
    would have refused to accept (see
    test_real_fit_noisy_day_holds_and_leaves_persisted_floor_unchanged in
    this file's git history for the pre-Change-1 reject-direction test this
    replaces)."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS

    # A prior floor established WAY outside where today's tightly-clustered
    # noisy data could plausibly land — exactly the scenario the removed
    # stability gate existed to hold against.
    store.write_reranker_floor(
        MODEL_ID, floor=500.0, raw_fit_floor=500.0, sample_pairs=min_pairs, is_cold_start=False
    )
    # Today's noisy day: small tight cluster far from the seeded prior.
    pairs = (
        [(100.0 + i * 0.01, "relevant") for i in range(min_pairs // 2)]
        + [(99.0 - i * 0.01, "irrelevant") for i in range(min_pairs - min_pairs // 2)]
    )
    for score, label in pairs:
        _seed_labeled_row(store, [score], [label])

    real_pairs = store.labeled_calibration_pairs(MODEL_ID)
    expected_raw = fit_threshold_fbeta(real_pairs, beta=FLOOR_FIT_BETA)

    outcome = derive_and_persist_floor(store, MODEL_ID)

    assert outcome.accepted is True, "a noisy-but-sufficient day must NOT be held — no gate anymore"
    assert outcome.held_for_data_starvation is False
    assert outcome.floor == pytest.approx(expected_raw)
    assert outcome.floor != pytest.approx(500.0), "must have actually moved off the stale seeded prior"

    persisted = store.get_persisted_reranker_floor(MODEL_ID)
    assert persisted["floor"] == pytest.approx(expected_raw)


def test_backstop_holds_prior_byte_identical_when_starved(store: MemoryStore) -> None:
    """Acceptance #5(a): a day with < FLOOR_FIT_MIN_LABELED_PAIRS pairs AND
    an existing prior persisted floor leaves the persisted floor
    byte-identical to the prior value — no refit attempted at all."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    store.write_reranker_floor(
        MODEL_ID, floor=1.2345, raw_fit_floor=1.2345, sample_pairs=min_pairs, is_cold_start=False
    )
    persisted_before = store.get_persisted_reranker_floor(MODEL_ID)

    # Starved: only a handful of labeled pairs, nowhere near the threshold.
    for i in range(min_pairs - 1):
        label = "relevant" if i % 2 == 0 else "irrelevant"
        _seed_labeled_row(store, [float(i)], [label])

    outcome = derive_and_persist_floor(store, MODEL_ID)

    assert outcome.accepted is False
    assert outcome.held_for_data_starvation is True
    assert outcome.floor == pytest.approx(1.2345)
    assert outcome.raw_fit_floor == pytest.approx(1.2345)

    persisted_after = store.get_persisted_reranker_floor(MODEL_ID)
    assert persisted_after == persisted_before, (
        "a held cycle must leave the persisted floor row byte-for-byte unchanged"
    )


def test_backstop_holds_carry_no_memory_across_multiple_starved_days(store: MemoryStore) -> None:
    """Acceptance #5(b): several consecutive starved (<200-pair) days
    followed by one sufficient (>=200-pair) day — the floor on that day
    refits to THAT day's raw fit, with NO damping or partial-move from
    however many holds preceded it. Proves holds carry no memory, unlike
    the removed stability gate (which compared every later day against the
    same never-updated stale anchor once it first held)."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    rng = np.random.default_rng(3)

    store.write_reranker_floor(
        MODEL_ID, floor=-77.0, raw_fit_floor=-77.0, sample_pairs=min_pairs, is_cold_start=False
    )

    # Three consecutive starved days, each on its own day_bucket, each held.
    for day_index in range(3):
        day_bucket = f"2026-03-{day_index + 1:02d}"
        for i in range(5):
            label = "relevant" if i % 2 == 0 else "irrelevant"
            _seed_labeled_row(store, [float(i)], [label], day_bucket=day_bucket)
        held_outcome = derive_and_persist_floor(store, MODEL_ID)
        assert held_outcome.accepted is False
        assert held_outcome.held_for_data_starvation is True
        assert held_outcome.floor == pytest.approx(-77.0), "every hold must still report the untouched prior"

    # A fourth, sufficient day — must fit FRESH from just this day's data,
    # with no trace of the -77.0 prior or the three holds that preceded it.
    sufficient_day = "2026-03-04"
    _seed_min_pairs(
        store, n=min_pairs, relevant_loc=20.0, irrelevant_loc=10.0, rng=rng, day_bucket=sufficient_day,
    )
    fresh_pairs = store.labeled_calibration_pairs(MODEL_ID)
    expected_raw = fit_threshold_fbeta(fresh_pairs, beta=FLOOR_FIT_BETA)

    outcome = derive_and_persist_floor(store, MODEL_ID)
    assert outcome.accepted is True
    assert outcome.floor == pytest.approx(expected_raw)
    assert outcome.floor != pytest.approx(-77.0)
    assert not (-77.0 < outcome.floor < -70.0), "must not be damped toward the old prior in any way"


def test_no_prior_row_and_starved_writes_nothing_no_crash(store: MemoryStore) -> None:
    """Acceptance #5(c), first half: a fresh-deploy state (no persisted
    floor row at all) with < FLOOR_FIT_MIN_LABELED_PAIRS pairs writes
    NOTHING — no row is created, no crash on a None prior — and a read
    through the hot-path get_bootstrap_floor in the SAME state still
    returns a served floor (recall stays served through the bootstrap
    while the tick stays silent)."""
    from brain.memory import reranker as reranker_mod

    reranker_mod._bootstrap_reranker_provider = lambda model_id: FakeRerankerProvider(
        scores=_bundled_scores(5.0, -5.0)
    )
    try:
        assert store.get_persisted_reranker_floor(MODEL_ID) is None

        for i in range(FLOOR_FIT_MIN_LABELED_PAIRS - 1):
            label = "relevant" if i % 2 == 0 else "irrelevant"
            _seed_labeled_row(store, [float(i)], [label])

        outcome = derive_and_persist_floor(store, MODEL_ID)

        assert outcome.accepted is False
        assert outcome.held_for_data_starvation is True
        assert outcome.floor is None, "nothing is in effect from this call — no floor to report"
        assert outcome.raw_fit_floor is None
        assert store.get_persisted_reranker_floor(MODEL_ID) is None, "must write NOTHING, not a placeholder row"

        bootstrap_served = store.get_reranker_floor(MODEL_ID)
        assert bootstrap_served is not None, "recall must still get a served floor via the bootstrap hot path"
        assert bootstrap_served["is_cold_start"] is True
    finally:
        del reranker_mod._bootstrap_reranker_provider


def test_no_prior_row_then_a_sufficient_day_fits_and_persists_normally(store: MemoryStore) -> None:
    """Acceptance #5(c), second half: once a day accumulates >=
    FLOOR_FIT_MIN_LABELED_PAIRS real pairs, the tick fits and persists
    normally from that point on — exactly as (a)/(b) describe once a prior
    row exists."""
    min_pairs = FLOOR_FIT_MIN_LABELED_PAIRS
    rng = np.random.default_rng(4)
    assert store.get_persisted_reranker_floor(MODEL_ID) is None

    _seed_min_pairs(store, n=min_pairs, relevant_loc=3.0, irrelevant_loc=-3.0, rng=rng)
    real_pairs = store.labeled_calibration_pairs(MODEL_ID)
    expected_raw = fit_threshold_fbeta(real_pairs, beta=FLOOR_FIT_BETA)

    outcome = derive_and_persist_floor(store, MODEL_ID)

    assert outcome.accepted is True
    assert outcome.floor == pytest.approx(expected_raw)
    persisted = store.get_persisted_reranker_floor(MODEL_ID)
    assert persisted is not None
    assert persisted["floor"] == pytest.approx(expected_raw)
    assert persisted["is_cold_start"] is False


def test_day_only_read_fit_matches_day_alone_not_the_full_pool(store: MemoryStore) -> None:
    """Acceptance #2 ("Day-only read"): 7 days of accumulated labeled pairs
    where only the most recent day's pairs are distinguishable (days 1-6
    all cluster around 0.0 with high noise/overlap, day 7 is CLEANLY
    separated far away) — the fitted floor must match a fit computed on
    day-7-alone and must NOT match a fit computed on the full 7-day pool,
    proving the day filter actually replaced the pooled read."""
    rng = np.random.default_rng(5)
    per_day = FLOOR_FIT_MIN_LABELED_PAIRS  # each day clears the threshold on its own

    for day_index in range(6):
        day_bucket = f"2026-04-{day_index + 1:02d}"
        _seed_min_pairs(
            store, n=per_day, relevant_loc=0.5, irrelevant_loc=-0.5, scale=2.0, rng=rng, day_bucket=day_bucket,
        )
    day7_bucket = "2026-04-07"
    _seed_min_pairs(
        store, n=per_day, relevant_loc=50.0, irrelevant_loc=40.0, scale=1.0, rng=rng, day_bucket=day7_bucket,
    )

    day7_only_pairs = [
        (score, label)
        for score, label in store._conn.execute(  # noqa: SLF001
            "SELECT reranker_scores, local_judge_label FROM calibration_log "
            "WHERE day_bucket = ? AND reranker_model_id = ?",
            (day7_bucket, MODEL_ID),
        ).fetchall()
        for score, label in zip(json.loads(score), json.loads(label), strict=True)
    ]
    expected_day7_floor = fit_threshold_fbeta(day7_only_pairs, beta=FLOOR_FIT_BETA)

    all_rows = store._conn.execute(  # noqa: SLF001
        "SELECT reranker_scores, local_judge_label FROM calibration_log WHERE reranker_model_id = ?",
        (MODEL_ID,),
    ).fetchall()
    pooled_pairs = [
        (score, label)
        for raw_scores, raw_labels in all_rows
        for score, label in zip(json.loads(raw_scores), json.loads(raw_labels), strict=True)
    ]
    pooled_floor = fit_threshold_fbeta(pooled_pairs, beta=FLOOR_FIT_BETA)

    outcome = derive_and_persist_floor(store, MODEL_ID)

    assert outcome.accepted is True
    assert outcome.sample_pairs == per_day, "must read ONLY day 7's pairs, not all 7 days pooled"
    assert outcome.floor == pytest.approx(expected_day7_floor)
    assert outcome.floor != pytest.approx(pooled_floor), (
        "must NOT match the full 7-day pooled fit — the day filter actually replaced the pooled read"
    )


# ---------------------------------------------------------------------------
# Acceptance #1: lock repro, REVERSED. Seeds the T7 Part A scenario (>=200
# pairs/day x 10 sim days, a shift=3.0 clean series and a shift=3.0 noisy
# series) through the real nimble fit; asserts the floor TRACKS the shift
# within 1-2 days on BOTH series — the same repro that proved the OLD
# mechanism's permanent lock (an independent code verify confirmed the
# removed stability gate had no recovery path: once it held, every later
# day was compared against the same never-updated stale anchor) must now
# prove that lock's absence.
# ---------------------------------------------------------------------------


def _run_sim_days(
    store: MemoryStore,
    model_id: str,
    *,
    n_days: int,
    shift_at_day: int,
    shift: float,
    scale: float,
    seed: int,
) -> list[float | None]:
    """Simulate `n_days` daily ticks for `model_id`: days before
    `shift_at_day` (0-indexed) draw from a baseline distribution
    (relevant~+5, irrelevant~-5); `shift_at_day` onward draw from the SAME
    shape with BOTH clusters shifted up by `shift` together (relevant~
    +5+shift, irrelevant~-5+shift — separation preserved, only the
    position moves) — a sustained corpus/precision/model-swap-style shift,
    per the spec's own framing. `scale` controls noise (clean vs noisy
    series). Returns the
    persisted floor after each day's tick (>= FLOOR_FIT_MIN_LABELED_PAIRS
    every day, so the backstop never fires and every entry is a real,
    accepted fit)."""
    rng = np.random.default_rng(seed)
    floors: list[float | None] = []
    for day_index in range(n_days):
        day_bucket = f"2026-05-{day_index + 1:02d}"
        # A "shift" moves the WHOLE score distribution up by a constant
        # (both clusters together, preserving their separation) — a model
        # swap / precision flip / corpus change moves where scores sit on
        # the number line, it does not change how well-separated the two
        # classes are. relevant/irrelevant each start 5.0 apart from 0.0
        # and both add `shift` once the shift hits, so the fitted
        # threshold (which tracks the midpoint-ish region between the two
        # clusters) should move by roughly `shift` too, not stay put.
        offset = 0.0 if day_index < shift_at_day else shift
        _seed_min_pairs(
            store,
            n=FLOOR_FIT_MIN_LABELED_PAIRS,
            relevant_loc=5.0 + offset,
            irrelevant_loc=-5.0 + offset,
            scale=scale,
            rng=rng,
            day_bucket=day_bucket,
            reranker_model_id=model_id,
        )
        outcome = derive_and_persist_floor(store, model_id)
        assert outcome.accepted is True, f"day {day_index} must be a real accepted fit, not a hold"
        floors.append(outcome.floor)
    return floors


@pytest.mark.parametrize(
    "scale,label,seed",
    [(1.0, "clean", 101), (3.0, "noisy", 202)],
    # Fixed, literal seeds — NOT `hash(label)`: Python's string hash is
    # randomized per-process (PEP 456) unless PYTHONHASHSEED is pinned, so
    # a seed derived from `hash(...)` would make this test's outcome
    # non-reproducible run to run, exactly the kind of hidden nondeterminism
    # that would make a real failure look like flakiness instead of a bug.
)
def test_lock_repro_reversed_floor_tracks_a_sustained_shift_within_two_days(
    scale: float, label: str, seed: int
) -> None:
    """Acceptance #1. Runs BOTH the clean (tight clusters) and noisy
    (wide, overlapping clusters) series via pytest parametrization — both
    must show the SAME qualitative behavior: stable pre-shift, then
    tracking the shift within 1-2 days post-shift, never locked."""
    store = MemoryStore(db_path=":memory:")
    model_id = f"lock-repro-{label}"
    n_days = 10
    shift_at_day = 6  # 0-indexed: days 0-5 baseline, days 6-9 shifted
    shift = 3.0

    floors = _run_sim_days(
        store, model_id, n_days=n_days, shift_at_day=shift_at_day, shift=shift, scale=scale, seed=seed,
    )

    pre_shift_floors = floors[:shift_at_day]
    post_shift_floors = floors[shift_at_day:]

    # Pre-shift: the floor should stay in a stable, tight band around the
    # baseline's own natural fit (same distribution every day) — it must
    # NOT already be drifting toward the post-shift value before the shift
    # even happens.
    pre_shift_spread = max(pre_shift_floors) - min(pre_shift_floors)
    assert pre_shift_spread < shift, (
        f"[{label}] pre-shift floor must stay stable (spread={pre_shift_spread}), not already moving"
    )

    # Post-shift: within 1-2 days of the shift's onset (index 0 or 1 of
    # post_shift_floors), the floor must have moved to track the shift —
    # i.e. sit closer to the NEW baseline's own natural fit than to the
    # OLD (pre-shift) floor level. Using the shift amount itself as the
    # yardstick: the floor must have moved by at least half the shift
    # within that window, which a genuinely LOCKED floor (frozen at its
    # pre-shift value, the defect this reverses) would never do.
    pre_shift_level = pre_shift_floors[-1]
    tracked_within_two_days = any(
        abs(post_shift_floors[i] - pre_shift_level) >= (shift / 2.0) for i in range(min(2, len(post_shift_floors)))
    )
    assert tracked_within_two_days, (
        f"[{label}] floor must track the shift within 1-2 days — got pre-shift={pre_shift_level}, "
        f"first two post-shift days={post_shift_floors[:2]}"
    )

    # Never locked: by the LAST simulated day, the floor must sit near the
    # NEW distribution's own raw fit (immediate tracking, since Change 1
    # has zero smoothing) — not anywhere near the stale pre-shift level,
    # which is exactly what the removed stability gate would have produced
    # forever once it first held.
    assert abs(floors[-1] - pre_shift_level) >= (shift / 2.0), (
        f"[{label}] the final day's floor must not still be anchored near the pre-shift level "
        f"(permanent-lock defect) — pre_shift={pre_shift_level}, final={floors[-1]}"
    )
    store.close()


# ---------------------------------------------------------------------------
# get_bootstrap_floor — F2a inc8 (#250 §7 UPDATED, Roy 2026-09-18): the
# hot-path DEFAULT `get_reranker_floor` serves instead of None when no
# persisted row exists yet. UNCHANGED by Change 1. All offline via a
# scripted FakeRerankerProvider monkeypatched onto
# `reranker._bootstrap_reranker_provider` directly.
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
