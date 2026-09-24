"""F2c inc5b-2 SLICE 2 — the tick LoRA/full-FT weight-retrain lifecycle.

Drives `_run_judge_selftune_tick`'s mid/beefy dispatch with SCRIPTED, no-model
fakes (monkeypatched `build_lora_retrain_fn` / `run_champion_challenger` /
`load_lora_scorer` / `build_judge_provider`) + a real in-memory `MemoryStore`
and the real filesystem champion-pointer helpers. NEVER loads a real model
(the real persist->reload is proven with a tiny model in test_judge_lora.py).

Covers criteria: C1 (accept/revert dispatch), C2 (Haiku-only triples fed +
split), C4 (accept knob = fresh tuned scores, not logged), C5 (revert reuses
logged scores), C7 (consume-at-end once + fault leaves re-eligible), C13
(ordering + post-swap-fault pointer rollback), C16 (2B consume semantics),
C17 (no provider-cache growth).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE
from brain.memory import judge_eval, judge_full_ft, judge_lora, judge_selftune, relevance_judge
from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
from brain.memory.store import MemoryStore

# Captured at import (before the autouse `build_judge_provider` stub patches
# it per-test) so C17 can exercise the REAL provider-cache behavior.
_REAL_BUILD_JUDGE_PROVIDER = relevance_judge.build_judge_provider


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


def _seed_row(
    store: MemoryStore,
    *,
    query: str,
    docs: list[str] | None,
    local_labels: list[str],
    haiku_labels: list[str | None],
    raw_scores: list[float | None],
) -> int:
    n = len(local_labels)
    store.log_calibration_sample(
        query=query,
        candidate_ids=[f"{query}-m{i}" for i in range(n)],
        reranker_scores=[1.0] * n,
        reranker_model_id="jina",
        candidate_docs=docs,
    )
    row_id = store._conn.execute(
        "SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]
    store.write_calibration_labels(row_id, local_labels, haiku_labels, raw_scores)
    return row_id


def _seed_scenario(store: MemoryStore) -> dict[str, int]:
    """3 doc-having rows (contribute Haiku triples) + 1 legacy doc-absent row
    (candidate_docs NULL, a Haiku position but no doc)."""
    r1 = _seed_row(
        store, query="q1", docs=["da", "db"],
        local_labels=["relevant", "irrelevant"], haiku_labels=[None, "relevant"],
        raw_scores=[0.5, -0.5],
    )
    r2 = _seed_row(
        store, query="q2", docs=["dc", "dd"],
        local_labels=["irrelevant", "relevant"], haiku_labels=["irrelevant", None],
        raw_scores=[-0.3, 0.7],
    )
    r3 = _seed_row(
        store, query="q3", docs=["de"],
        local_labels=["relevant"], haiku_labels=["relevant"], raw_scores=[0.9],
    )
    r4_docabsent = _seed_row(
        store, query="q4", docs=None,
        local_labels=["relevant", "irrelevant"], haiku_labels=[None, "relevant"],
        raw_scores=[0.4, -0.6],
    )
    return {"r1": r1, "r2": r2, "r3": r3, "r4_docabsent": r4_docabsent}


def _consumed(store: MemoryStore, row_id: int) -> bool:
    row = store._conn.execute(
        "SELECT selftune_consumed_at FROM calibration_log WHERE id = ?", (row_id,)
    ).fetchone()
    return row["selftune_consumed_at"] is not None


def _force_midbeefy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scripted RAM so tier-detect selects a weight-retrain tier with generous
    OOM headroom (no downgrade)."""
    monkeypatch.setattr(
        judge_selftune, "_read_total_ram_bytes",
        lambda: judge_selftune.JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES + 1.0,
    )
    monkeypatch.setattr(
        judge_selftune, "_available_ram_headroom_bytes",
        lambda: judge_selftune.JUDGE_TUNE_FOOTPRINT_LORA_BYTES + 1.0,
    )


def _small_tunables(monkeypatch: pytest.MonkeyPatch, *, gate: int = 2, min_n: int = 2) -> None:
    real = judge_selftune.tunables.get_tunable

    def fake(key: str, default: object) -> object:
        if key == "judge_selftune.gate_handful_decisions":
            return gate
        if key == "judge_selftune.eval_min_test_n":
            return min_n
        return real(key, default)

    monkeypatch.setattr(judge_selftune.tunables, "get_tunable", fake)


def _fake_retrain_factory(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """`build_lora_retrain_fn(..., save_adapter_dir=X)` -> a retrain_fn that
    writes a dummy adapter to X (so the staged dir exists) and returns a fake
    label fn. Captures the staged dir + train items."""

    def fake_build(base, *, target_modules, modules_to_save, save_adapter_dir=None, **kw):
        captured["save_adapter_dir"] = save_adapter_dir

        def retrain_fn(train_items):
            captured["train_items"] = list(train_items)
            Path(save_adapter_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_adapter_dir) / "adapter_model.txt").write_text("x", encoding="utf-8")
            return lambda item: "relevant"

        return retrain_fn

    monkeypatch.setattr(judge_lora, "build_lora_retrain_fn", fake_build)


def _fake_cc(monkeypatch: pytest.MonkeyPatch, *, accepted: bool, captured: dict) -> None:
    """Scripted `run_champion_challenger`: calls retrain_fn (to write the
    staged dir, matching the real contract), captures train/test, returns the
    scripted accept/revert decision."""

    def fake(*, champion, retrain_fn, train_items, test_items, rollback, **kw):
        captured["cc_train"] = list(train_items)
        captured["cc_test"] = list(test_items)
        challenger = retrain_fn(train_items)
        return judge_eval.ChampionChallengerResult(
            accepted=accepted,
            judge=challenger if accepted else champion,
            b=0, c=0, p_value=1.0, n_test=len(test_items),
            champion_correct=[], challenger_correct=[],
        )

    monkeypatch.setattr(judge_eval, "run_champion_challenger", fake)


def _fake_base_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        relevance_judge, "build_judge_provider",
        lambda adapter_dir=None: FakeRelevanceJudgeProvider(default=0.0),
    )


def _fresh_scorer(monkeypatch: pytest.MonkeyPatch, score_map: dict[tuple[str, str], float]) -> None:
    """`load_lora_scorer(base, adapter)` -> a scorer returning FRESH scores
    (distinct from the logged raw scores) so C4 can prove the knob is fit on
    fresh, not logged."""
    monkeypatch.setattr(
        judge_lora, "load_lora_scorer",
        lambda base, adapter, **kw: (lambda item: score_map.get((item[0], item[1]), 3.14)),
    )


# ---------------------------------------------------------------------------
# C1 / C2 / C4 / C16 — ACCEPT path.
# ---------------------------------------------------------------------------


def test_accept_persists_adapter_rescore_knob_and_consumes_doc_having_only(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_midbeefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    # Fresh tuned scores DISTINCT from the logged raw scores.
    _fresh_scorer(monkeypatch, {})  # default 3.14 for every (q,d) -> a constant fresh score

    now = datetime.now(UTC)
    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=now, persona_dir=tmp_path
    )

    assert result["fired"] is True and result["accepted"] is True
    assert result["adapter_persisted"] is True
    assert result["error"] is None

    # C3: the champion pointer resolves to the persisted adapter.
    champ = judge_lora.resolve_champion_adapter(judge_lora.champion_dir(tmp_path))
    assert champ is not None and (champ / "adapter_model.txt").exists()

    # C2: only Haiku triples were fed. train items are (query, doc, label)
    # 3-tuples; test items are run_champion_challenger's ((query, doc), label)
    # shape. Split is disjoint (no leakage), and the doc-absent row q4 never
    # appears (no doc snapshot to forward-pass / train on).
    assert all(len(item) == 3 for item in captured["cc_train"]), "(query, doc, label) triples"
    train_qd = {(q, d) for (q, d, _l) in captured["cc_train"]}
    test_qd = {(qd[0], qd[1]) for (qd, _l) in captured["cc_test"]}
    assert all(q != "q4" for (q, _d) in train_qd | test_qd), "doc-absent row excluded"
    assert train_qd.isdisjoint(test_qd), "no train/test leakage"

    # C4: the persisted knob equals fit on FRESH scores (all 3.14 -> single
    # score value -> fit_platt_knob returns the identity for a degenerate
    # single-score fit, i.e. NOT the logged-score fit which would separate).
    knob = store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE)
    assert knob is not None
    rescore_items = store.judge_knob_refit_rescore_items(
        [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    )
    fresh_pairs = [(3.14, label) for (_q, _d, label) in rescore_items]
    expected_slope, expected_intercept = judge_selftune.fit_platt_knob(fresh_pairs)
    assert knob["slope"] == pytest.approx(expected_slope)
    assert knob["intercept"] == pytest.approx(expected_intercept)
    logged_pairs = store.judge_knob_refit_pairs(
        [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    )
    logged_slope, logged_intercept = judge_selftune.fit_platt_knob(logged_pairs)
    assert (knob["slope"], knob["intercept"]) != (logged_slope, logged_intercept), (
        "knob must be fit on fresh tuned scores, NOT the logged scores"
    )

    # C16 (accept): doc-having rows consumed; the legacy doc-absent row is NOT
    # (it was not trained here — 2B).
    assert _consumed(store, ids["r1"]) and _consumed(store, ids["r2"]) and _consumed(store, ids["r3"])
    assert not _consumed(store, ids["r4_docabsent"]), "doc-absent row stays re-eligible (2B)"


# ---------------------------------------------------------------------------
# C1 / C5 / C16 — REVERT path.
# ---------------------------------------------------------------------------


def test_revert_keeps_champion_uses_logged_knob_and_consumes_full_set(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_midbeefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=False, captured=captured)
    _fake_base_judge(monkeypatch)
    # If load_lora_scorer were called on revert this would blow up (it must
    # NOT be — revert reuses logged scores).
    monkeypatch.setattr(
        judge_lora, "load_lora_scorer",
        lambda *a, **k: pytest.fail("load_lora_scorer must not run on REVERT (C5)"),
    )

    now = datetime.now(UTC)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=now, persona_dir=tmp_path)

    assert result["fired"] is True and result["accepted"] is False
    assert result["adapter_persisted"] is False
    # C5: champion pointer never swapped (no adapter live).
    assert judge_lora.resolve_champion_adapter(judge_lora.champion_dir(tmp_path)) is None
    # knob fit on the LOGGED pairs (inc3 path).
    all_ids = [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    logged_slope, logged_intercept = judge_selftune.fit_platt_knob(
        store.judge_knob_refit_pairs(all_ids)
    )
    knob = store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE)
    assert (knob["slope"], knob["intercept"]) == pytest.approx((logged_slope, logged_intercept))
    # C16 (revert): the doc-agnostic logged fit trained on every row -> consume
    # the FULL set incl. the doc-absent row.
    assert all(_consumed(store, rid) for rid in all_ids)


# ---------------------------------------------------------------------------
# C1 (degenerate floor) — too few Haiku triples -> weak knob-refit, keep champ.
# ---------------------------------------------------------------------------


def test_thin_haiku_data_falls_through_to_weak_knob_refit(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_midbeefy(monkeypatch)
    _small_tunables(monkeypatch, min_n=99)  # min_n far above available triples
    monkeypatch.setattr(
        judge_eval, "run_champion_challenger",
        lambda **k: pytest.fail("champion/challenger must not run below min_n (degenerate floor)"),
    )
    now = datetime.now(UTC)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=now, persona_dir=tmp_path)
    assert result["fired"] is True and result["accepted"] is None
    # weak floor consumes the FULL set (logged knob trained on all).
    assert all(
        _consumed(store, rid) for rid in [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    )
    assert judge_lora.resolve_champion_adapter(judge_lora.champion_dir(tmp_path)) is None


# ---------------------------------------------------------------------------
# C7 / C13 — fault before end-consume leaves rows unconsumed; post-swap fault
# rolls the pointer back.
# ---------------------------------------------------------------------------


def test_fault_before_consume_leaves_rows_unconsumed(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_midbeefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    _fresh_scorer(monkeypatch, {})
    # Inject a fault at the knob-write step (AFTER the pointer swap).
    orig_write = store.write_judge_knob_calibration

    def boom(*a, **k):
        raise RuntimeError("injected knob-write fault")

    monkeypatch.setattr(store, "write_judge_knob_calibration", boom)

    now = datetime.now(UTC)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=now, persona_dir=tmp_path)

    assert result["fired"] is False and result["error"] is not None
    # C7: nothing consumed (consume is LAST, after the failed knob-write).
    assert not any(
        _consumed(store, rid) for rid in [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    )
    # C13: this was a FIRST-EVER tune, so the post-swap-fault rollback CLEARS
    # the pointer -> resolves to base (I9), never a half-committed champion.
    assert judge_lora.resolve_champion_adapter(judge_lora.champion_dir(tmp_path)) is None
    del orig_write


def test_post_swap_fault_rolls_pointer_back_to_prior_champion(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pre-existing champion adapter for this persona.
    root = judge_lora.champion_dir(tmp_path)
    prior = judge_lora.staged_adapter_path(root)
    prior.mkdir(parents=True)
    (prior / "adapter_model.txt").write_text("prior", encoding="utf-8")
    judge_lora.swap_champion_pointer(root, prior)

    _seed_scenario(store)
    _force_midbeefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    # champion here is the tuned (prior) adapter -> load_lora_scorer is called
    # for BOTH the champion and the re-score; return a scorer either way.
    monkeypatch.setattr(judge_lora, "load_lora_scorer", lambda *a, **k: (lambda item: 1.0))
    monkeypatch.setattr(store, "write_judge_knob_calibration", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["error"] is not None
    # C13: pointer rolled back to the PRIOR champion (last-known-good).
    assert judge_lora.resolve_champion_adapter(root) == prior


# ---------------------------------------------------------------------------
# C11 — the weak-tier path imports no torch (extends the existing AC8 guard to
# the new dispatch). C17 — adapter judges are not cached.
# ---------------------------------------------------------------------------


def test_provider_cache_does_not_grow_across_adapter_builds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # C17: build_judge_provider(adapter_dir=...) never caches the adapter judge.
    # Restore the REAL build_judge_provider (the autouse fixture stubs it).
    monkeypatch.setattr(relevance_judge, "build_judge_provider", _REAL_BUILD_JUDGE_PROVIDER)
    relevance_judge._reset_judge_provider_cache()
    monkeypatch.setattr(judge_lora, "load_lora_scorer", lambda *a, **k: (lambda item: 0.0))
    for i in range(5):
        judge = relevance_judge.build_judge_provider(adapter_dir=f"/tmp/adapter-{i}")
        assert isinstance(judge, relevance_judge.LoraAdapterJudge)
    assert len(relevance_judge._provider_cache) == 0, "adapter judges must not be cached"


# ===========================================================================
# F2c inc6 — the BEEFY (full-FT) tier dispatch, mirroring the LoRA lifecycle
# above on the FULL store (`judge_lora.full_champion_dir`). Scripted RAM forces
# FULL_FT; the full-FT mechanism is monkeypatched (no real model), exactly as
# the LoRA tests monkeypatch the LoRA mechanism.
# ===========================================================================


def _force_beefy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scripted RAM so tier-detect selects the BEEFY (full-FT) tier with
    generous OOM headroom (no downgrade)."""
    monkeypatch.setattr(
        judge_selftune, "_read_total_ram_bytes",
        lambda: judge_selftune.JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES + 1.0,
    )
    monkeypatch.setattr(
        judge_selftune, "_available_ram_headroom_bytes",
        lambda: judge_selftune.JUDGE_TUNE_FOOTPRINT_FULL_FT_BYTES + 1.0,
    )


def _fake_full_retrain_factory(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """`build_full_ft_retrain_fn(..., save_full_dir=X)` -> a retrain_fn that
    writes a dummy full model to X and returns a fake label fn."""

    def fake_build(base, *, save_full_dir=None, **kw):
        captured["save_full_dir"] = save_full_dir

        def retrain_fn(train_items):
            captured["full_train_items"] = list(train_items)
            Path(save_full_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_full_dir) / "model.safetensors.txt").write_text("x", encoding="utf-8")
            return lambda item: "relevant"

        return retrain_fn

    monkeypatch.setattr(judge_full_ft, "build_full_ft_retrain_fn", fake_build)


def _fresh_full_scorer(monkeypatch: pytest.MonkeyPatch, score_map: dict[tuple[str, str], float]) -> None:
    monkeypatch.setattr(
        judge_full_ft, "load_full_scorer",
        lambda d, **kw: (lambda item: score_map.get((item[0], item[1]), 3.14)),
    )


def _full_root(tmp_path: Path) -> Path:
    return judge_lora.full_champion_dir(tmp_path)


# --- C7: BEEFY dispatches to the full-FT mechanism + writes the FULL store ---


def test_beefy_tier_dispatches_full_ft_and_writes_full_store(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_full_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    _fresh_full_scorer(monkeypatch, {})
    # The LoRA mechanism must NOT be used for the FULL_FT tier (bite against the
    # pre-inc6 placeholder, which routed FULL_FT through build_lora_retrain_fn).
    monkeypatch.setattr(
        judge_lora, "build_lora_retrain_fn",
        lambda *a, **k: pytest.fail("FULL_FT tier must use build_full_ft_retrain_fn, not LoRA"),
    )

    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["fired"] is True and result["accepted"] is True
    assert result["tune_grade"] == judge_selftune.TUNE_GRADE_FULL_FT
    # The FULL store's champion pointer resolves; the LoRA store is untouched.
    assert judge_lora.resolve_champion_adapter(_full_root(tmp_path)) is not None
    assert judge_lora.resolve_champion_adapter(judge_lora.champion_dir(tmp_path)) is None
    assert captured["save_full_dir"] is not None


# --- C8: full-FT ACCEPT reuses the slice-2 lifecycle (re-score on tuned model,
#         2B consume) on the FULL store ---


def test_full_ft_accept_rescore_knob_and_consumes_doc_having_only(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_full_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    _fresh_full_scorer(monkeypatch, {})  # constant fresh 3.14

    now = datetime.now(UTC)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=now, persona_dir=tmp_path)
    assert result["accepted"] is True and result["adapter_persisted"] is True

    champ = judge_lora.resolve_champion_adapter(_full_root(tmp_path))
    assert champ is not None and (champ / "model.safetensors.txt").exists()

    # only Haiku triples fed; doc-absent row q4 excluded; no leakage.
    train_qd = {(q, d) for (q, d, _l) in captured["cc_train"]}
    test_qd = {(qd[0], qd[1]) for (qd, _l) in captured["cc_test"]}
    assert all(q != "q4" for (q, _d) in train_qd | test_qd)
    assert train_qd.isdisjoint(test_qd)

    # knob = fit on FRESH tuned scores (all 3.14), NOT the logged scores.
    knob = store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE)
    all_ids = [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    rescore_items = store.judge_knob_refit_rescore_items(all_ids)
    fresh_pairs = [(3.14, label) for (_q, _d, label) in rescore_items]
    exp_slope, exp_intercept = judge_selftune.fit_platt_knob(fresh_pairs)
    assert knob["slope"] == pytest.approx(exp_slope) and knob["intercept"] == pytest.approx(exp_intercept)
    logged_slope, logged_intercept = judge_selftune.fit_platt_knob(store.judge_knob_refit_pairs(all_ids))
    assert (knob["slope"], knob["intercept"]) != (logged_slope, logged_intercept)

    # 2B consume: doc-having consumed, doc-absent stays re-eligible.
    assert _consumed(store, ids["r1"]) and _consumed(store, ids["r2"]) and _consumed(store, ids["r3"])
    assert not _consumed(store, ids["r4_docabsent"])


# --- C9: full-FT REVERT keeps champ + logged knob + full-set consume; and the
#         degenerate floor falls through to the weak knob-refit ---


def test_full_ft_revert_keeps_champion_and_consumes_full_set(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_full_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=False, captured=captured)
    _fake_base_judge(monkeypatch)
    monkeypatch.setattr(
        judge_full_ft, "load_full_scorer",
        lambda *a, **k: pytest.fail("load_full_scorer must not run on REVERT (logged knob)"),
    )

    now = datetime.now(UTC)
    result = judge_selftune._run_judge_selftune_tick(store=store, now=now, persona_dir=tmp_path)
    assert result["accepted"] is False and result["adapter_persisted"] is False
    assert judge_lora.resolve_champion_adapter(_full_root(tmp_path)) is None
    all_ids = [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    logged = judge_selftune.fit_platt_knob(store.judge_knob_refit_pairs(all_ids))
    knob = store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE)
    assert (knob["slope"], knob["intercept"]) == pytest.approx(logged)
    assert all(_consumed(store, rid) for rid in all_ids)


def test_full_ft_thin_haiku_data_falls_through_to_weak_knob_refit(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch, min_n=99)
    monkeypatch.setattr(
        judge_eval, "run_champion_challenger",
        lambda **k: pytest.fail("champion/challenger must not run below min_n (degenerate floor)"),
    )
    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["fired"] is True and result["accepted"] is None
    assert all(_consumed(store, rid) for rid in [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]])
    assert judge_lora.resolve_champion_adapter(_full_root(tmp_path)) is None


# --- C10: consume-at-end / crash-safety preserved for the full tier ---


def test_full_ft_fault_before_consume_leaves_rows_unconsumed(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_full_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    _fresh_full_scorer(monkeypatch, {})
    monkeypatch.setattr(
        store, "write_judge_knob_calibration",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected knob-write fault")),
    )

    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["fired"] is False and result["error"] is not None
    assert not any(
        _consumed(store, rid) for rid in [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]
    )
    # first-ever tune -> post-swap-fault rollback CLEARS the full pointer.
    assert judge_lora.resolve_champion_adapter(_full_root(tmp_path)) is None


def test_full_ft_post_swap_fault_rolls_pointer_back_to_prior_full_champion(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _full_root(tmp_path)
    prior = judge_lora.staged_adapter_path(root)
    prior.mkdir(parents=True)
    (prior / "model.safetensors.txt").write_text("prior", encoding="utf-8")
    judge_lora.swap_champion_pointer(root, prior)

    _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_full_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    # champion here is the prior full model -> load_full_scorer called for both
    # champion and re-score; return a scorer either way.
    monkeypatch.setattr(judge_full_ft, "load_full_scorer", lambda *a, **k: (lambda item: 1.0))
    monkeypatch.setattr(
        store, "write_judge_knob_calibration",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["error"] is not None
    assert judge_lora.resolve_champion_adapter(root) == prior


# --- C17: sibling-clear on ACCEPT keeps exactly one tuned judge live ---


def test_full_ft_accept_clears_a_pre_existing_lora_champion(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pre-existing LoRA champion (from an earlier mid-tier week).
    lora_root = judge_lora.champion_dir(tmp_path)
    lora_prior = judge_lora.staged_adapter_path(lora_root)
    lora_prior.mkdir(parents=True)
    (lora_prior / "adapter_model.txt").write_text("lora", encoding="utf-8")
    judge_lora.swap_champion_pointer(lora_root, lora_prior)
    assert judge_lora.resolve_champion_adapter(lora_root) is not None

    _seed_scenario(store)
    _force_beefy(monkeypatch)
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_full_retrain_factory(monkeypatch, captured)
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    _fresh_full_scorer(monkeypatch, {})

    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["accepted"] is True
    # The FULL champion is live; the sibling LoRA pointer was CLEARED (spec §1
    # "alternatives, never both"), so resolve_serving_tuned_judge -> ("full", …).
    assert judge_lora.resolve_champion_adapter(_full_root(tmp_path)) is not None
    assert judge_lora.resolve_champion_adapter(lora_root) is None
    kind, _dir = judge_lora.resolve_serving_tuned_judge(tmp_path)
    assert kind == "full"


def test_lora_accept_clears_a_pre_existing_full_champion(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Symmetric: a mid-tier LoRA accept clears a pre-existing FULL champion.
    full_root = _full_root(tmp_path)
    full_prior = judge_lora.staged_adapter_path(full_root)
    full_prior.mkdir(parents=True)
    (full_prior / "model.safetensors.txt").write_text("full", encoding="utf-8")
    judge_lora.swap_champion_pointer(full_root, full_prior)

    _seed_scenario(store)
    _force_midbeefy(monkeypatch)  # LoRA tier
    _small_tunables(monkeypatch)
    captured: dict = {}
    _fake_retrain_factory(monkeypatch, captured)  # LoRA retrain fake
    _fake_cc(monkeypatch, accepted=True, captured=captured)
    _fake_base_judge(monkeypatch)
    _fresh_scorer(monkeypatch, {})

    result = judge_selftune._run_judge_selftune_tick(
        store=store, now=datetime.now(UTC), persona_dir=tmp_path
    )
    assert result["accepted"] is True
    assert judge_lora.resolve_champion_adapter(judge_lora.champion_dir(tmp_path)) is not None
    assert judge_lora.resolve_champion_adapter(full_root) is None
