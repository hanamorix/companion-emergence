"""F2c weekly self-tune tick — the LoRA/full-FT weight-retrain lifecycle, ONE
model lineage per persona (inc5b-2 slice 2, inc6, reworked in inc7).

Drives `_run_judge_selftune_tick`'s mid/beefy dispatch with SCRIPTED, no-model
fakes (monkeypatched `build_lora_retrain_fn` / `build_full_ft_retrain_fn` /
`run_champion_challenger` / `load_full_scorer` / `build_judge_provider`) + a
real in-memory `MemoryStore` and the real filesystem store helpers. NEVER
loads a real model (the real train/merge/save/reload is proven with a tiny
model in test_judge_lora.py / test_judge_full_ft.py).

inc7 criteria covered here (changes/f2c-inc7-one-lineage/1.5-criteria.md):
L1-L7 (each week's update applied to the persisting model; revert; dispatch
matrix; champion = serving model; AC13 tier drops; knob follows the served
model), S2-S5 (pointer only on accept; delete-after-swap with retry; commit
point; fail-soft resolve), H4 (observability), plus the carried inc5b-2/inc6
lifecycle contracts (Haiku-only triples, no leakage, re-score on accept, logged
knob on revert, 2B consume, degenerate floor, consume-last).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE
from brain.memory import judge_eval, judge_full_ft, judge_lora, judge_selftune, relevance_judge
from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
from brain.memory.store import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# Seeding + scripted environment
# ---------------------------------------------------------------------------


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
    row_id = store._conn.execute("SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1").fetchone()["id"]
    store.write_calibration_labels(row_id, local_labels, haiku_labels, raw_scores)
    return row_id


def _seed_scenario(store: MemoryStore, tag: str = "w1") -> dict[str, int]:
    """3 doc-having rows (contribute Haiku triples) + 1 legacy doc-absent row
    (candidate_docs NULL, a Haiku position but no doc). Queries carry `tag`
    so a later week's rows are distinguishable from an earlier week's."""
    r1 = _seed_row(
        store, query=f"{tag}q1", docs=[f"{tag}da", f"{tag}db"],
        local_labels=["relevant", "irrelevant"], haiku_labels=[None, "relevant"],
        raw_scores=[0.5, -0.5],
    )
    r2 = _seed_row(
        store, query=f"{tag}q2", docs=[f"{tag}dc", f"{tag}dd"],
        local_labels=["irrelevant", "relevant"], haiku_labels=["irrelevant", None],
        raw_scores=[-0.3, 0.7],
    )
    r3 = _seed_row(
        store, query=f"{tag}q3", docs=[f"{tag}de"],
        local_labels=["relevant"], haiku_labels=["relevant"], raw_scores=[0.9],
    )
    r4_docabsent = _seed_row(
        store, query=f"{tag}q4", docs=None,
        local_labels=["relevant", "irrelevant"], haiku_labels=[None, "relevant"],
        raw_scores=[0.4, -0.6],
    )
    return {"r1": r1, "r2": r2, "r3": r3, "r4_docabsent": r4_docabsent}


def _all_ids(ids: dict[str, int]) -> list[int]:
    return [ids["r1"], ids["r2"], ids["r3"], ids["r4_docabsent"]]


def _consumed(store: MemoryStore, row_id: int) -> bool:
    row = store._conn.execute(
        "SELECT selftune_consumed_at FROM calibration_log WHERE id = ?", (row_id,)
    ).fetchone()
    return row["selftune_consumed_at"] is not None


def _set_tier(monkeypatch: pytest.MonkeyPatch, grade: str) -> None:
    """Scripted RAM + headroom so tier-detect selects `grade` (no downgrade)."""
    if grade == judge_selftune.TUNE_GRADE_FULL_FT:
        ram = judge_selftune.JUDGE_TUNE_RAM_TIER_FULL_FT_MIN_BYTES + 1.0
        head = judge_selftune.JUDGE_TUNE_FOOTPRINT_FULL_FT_BYTES + 1.0
    elif grade == judge_selftune.TUNE_GRADE_LORA:
        ram = judge_selftune.JUDGE_TUNE_RAM_TIER_LORA_MIN_BYTES + 1.0
        head = judge_selftune.JUDGE_TUNE_FOOTPRINT_LORA_BYTES + 1.0
    else:
        ram = 1.0
        head = judge_selftune.JUDGE_TUNE_FOOTPRINT_FULL_FT_BYTES + 1.0
    monkeypatch.setattr(judge_selftune, "_read_total_ram_bytes", lambda: ram)
    monkeypatch.setattr(judge_selftune, "_available_ram_headroom_bytes", lambda: head)


def _small_tunables(monkeypatch: pytest.MonkeyPatch, *, gate: int = 2, min_n: int = 2) -> None:
    real = judge_selftune.tunables.get_tunable

    def fake(key: str, default: object) -> object:
        if key == "judge_selftune.gate_handful_decisions":
            return gate
        if key == "judge_selftune.eval_min_test_n":
            return min_n
        return real(key, default)

    monkeypatch.setattr(judge_selftune.tunables, "get_tunable", fake)


def _write_plain_checkpoint(d: Path, marker: str = "x") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors.txt").write_text(marker, encoding="utf-8")


class Env:
    """Scripted builders/eval/scorers; records every call the tick makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mp = monkeypatch
        self.builds: list[dict] = []  # one per week: method, start, staged, train_items
        self.scorer_dirs: list[str] = []  # every load_full_scorer(dir) call, in order
        self.champion_labels: list[str] = []
        self.accept: bool = True

        def fake_lora(start, *, target_modules, modules_to_save, save_dir=None, **kw):
            return self._retrain("lora", start, save_dir)

        def fake_full(start, *, save_full_dir=None, **kw):
            return self._retrain("full_ft", start, save_full_dir)

        def fake_cc(*, champion, retrain_fn, train_items, test_items, rollback, **kw):
            self.cc_train = list(train_items)
            self.cc_test = list(test_items)
            for item, _label in test_items:
                self.champion_labels.append(champion(item))
            challenger = retrain_fn(train_items)
            return judge_eval.ChampionChallengerResult(
                accepted=self.accept,
                judge=challenger if self.accept else champion,
                b=0, c=0, p_value=1.0, n_test=len(test_items),
                champion_correct=[], challenger_correct=[],
            )

        def fake_scorer(d, **kw):
            self.scorer_dirs.append(str(d))
            return lambda item: 3.14

        monkeypatch.setattr(judge_lora, "build_lora_retrain_fn", fake_lora)
        monkeypatch.setattr(judge_full_ft, "build_full_ft_retrain_fn", fake_full)
        monkeypatch.setattr(judge_eval, "run_champion_challenger", fake_cc)
        monkeypatch.setattr(judge_full_ft, "load_full_scorer", fake_scorer)
        monkeypatch.setattr(
            relevance_judge, "build_judge_provider",
            lambda full_model_dir=None: FakeRelevanceJudgeProvider(default=0.0),
        )

    def _retrain(self, method: str, start, staged):
        rec = {"method": method, "start": str(start), "staged": staged}
        self.builds.append(rec)

        def retrain_fn(train_items):
            rec["train_items"] = list(train_items)
            _write_plain_checkpoint(Path(staged), marker=f"{method}-{len(self.builds)}")
            return lambda item: "relevant"

        return retrain_fn


def _tick(store: MemoryStore, persona_dir: Path) -> dict:
    return judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC), persona_dir=persona_dir)


def _root(persona_dir: Path) -> Path:
    return judge_lora.champion_dir(persona_dir)


def _pointer_bytes(persona_dir: Path) -> bytes | None:
    p = _root(persona_dir) / "current"
    return p.read_bytes() if p.exists() else None


def _stored_dirs(persona_dir: Path) -> set[str]:
    root = _root(persona_dir)
    if not root.is_dir():
        return set()
    return {e.name for e in root.iterdir() if e.is_dir() and e.name.startswith("adapter-")}


def _current(persona_dir: Path) -> Path | None:
    return judge_lora.resolve_current_checkpoint(persona_dir)


def _knob(store: MemoryStore, checkpoint: Path | None) -> dict | None:
    """The knob row bound to `checkpoint` (F2c inc9 key), or the base judge's."""
    return store.get_judge_knob_calibration(relevance_judge.judge_knob_key(MODEL_RELEVANCE_JUDGE, checkpoint))


def _place_checkpoint(persona_dir: Path) -> Path:
    root = _root(persona_dir)
    d = judge_lora.staged_adapter_path(root)
    _write_plain_checkpoint(d, "prior")
    judge_lora.swap_champion_pointer(root, d)
    return d


GRADES = [judge_selftune.TUNE_GRADE_LORA, judge_selftune.TUNE_GRADE_FULL_FT]


# ---------------------------------------------------------------------------
# ACCEPT path (carried C1/C2/C4/C16; L7; S1 plain checkpoint; H4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_accept_swaps_in_plain_checkpoint_rescores_knob_and_consumes_doc_having_only(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    real_replace = os.replace
    replaced: list[tuple[str, str]] = []

    def spy_replace(src, dst):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(judge_lora.os, "replace", spy_replace)

    result = _tick(store, tmp_path)

    assert result["fired"] is True and result["accepted"] is True and result["error"] is None
    assert result["adapter_persisted"] is True and result["method"] == grade
    # One store, one pointer, naming the staged plain checkpoint (S1, S2).
    cur = _current(tmp_path)
    assert cur is not None and cur == Path(env.builds[0]["staged"])
    assert not (cur / "adapter_config.json").exists()
    assert not (_root(tmp_path) / "full").exists(), "no per-tier sub-store"
    assert any(dst.endswith(os.sep + "current") for (_s, dst) in replaced), "pointer swapped via os.replace"
    # Haiku-only triples, doc-absent row excluded, no leakage (carried C2).
    train_qd = {(q, d) for (q, d, _l) in env.cc_train}
    test_qd = {(qd[0], qd[1]) for (qd, _l) in env.cc_test}
    assert all(not q.endswith("q4") for (q, _d) in train_qd | test_qd)
    assert train_qd.isdisjoint(test_qd)
    # L7: the knob is fit on scores re-scored through the NEW checkpoint (the
    # serve loader), not the logged scores.
    assert env.scorer_dirs[-1] == str(cur)
    knob = _knob(store, cur)
    fresh = [(3.14, label) for (_q, _d, label) in store.judge_knob_refit_rescore_items(_all_ids(ids))]
    assert (knob["slope"], knob["intercept"]) == pytest.approx(judge_selftune.fit_platt_knob(fresh))
    logged = judge_selftune.fit_platt_knob(store.judge_knob_refit_pairs(_all_ids(ids)))
    assert (knob["slope"], knob["intercept"]) != pytest.approx(logged)
    # 2B consume.
    assert _consumed(store, ids["r1"]) and _consumed(store, ids["r2"]) and _consumed(store, ids["r3"])
    assert not _consumed(store, ids["r4_docabsent"])
    # H4: observability.
    assert result["current"] == "base" and result["start"] == "base"


# ---------------------------------------------------------------------------
# REVERT path (carried C5/C16; S2; H4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_revert_keeps_champion_uses_logged_knob_and_consumes_full_set(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    env.accept = False
    before = _pointer_bytes(tmp_path)

    result = _tick(store, tmp_path)

    assert result["fired"] is True and result["accepted"] is False and result["adapter_persisted"] is False
    assert _pointer_bytes(tmp_path) == before, "pointer untouched on revert (S2)"
    assert _stored_dirs(tmp_path) == {prior.name}, "rejected staged dir reaped; champion kept (S3b)"
    # Re-score never ran: the only load was the champion (the current model).
    assert env.scorer_dirs == [str(prior)]
    logged = judge_selftune.fit_platt_knob(store.judge_knob_refit_pairs(_all_ids(ids)))
    knob = _knob(store, prior)
    assert (knob["slope"], knob["intercept"]) == pytest.approx(logged)
    assert all(_consumed(store, rid) for rid in _all_ids(ids))
    # H4 on a revert: current and start both name the unchanged champion.
    assert result["current"] == prior.name and result["start"] == prior.name


@pytest.mark.parametrize("grade", GRADES)
def test_thin_haiku_data_falls_through_to_weak_knob_refit(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch, min_n=99)
    monkeypatch.setattr(
        judge_eval, "run_champion_challenger",
        lambda **k: pytest.fail("champion/challenger must not run below min_n (degenerate floor)"),
    )
    before = _pointer_bytes(tmp_path)
    result = _tick(store, tmp_path)
    assert result["fired"] is True and result["accepted"] is None
    assert all(_consumed(store, rid) for rid in _all_ids(ids))
    assert _pointer_bytes(tmp_path) == before and _stored_dirs(tmp_path) == {prior.name}
    assert _knob(store, prior) is not None, "the floor's knob is bound to the serving checkpoint"


# ---------------------------------------------------------------------------
# S4 — crash-safety with a commit point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_fault_before_swap_leaves_pointer_reaps_staged_and_rows_unconsumed(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    Env(monkeypatch)
    monkeypatch.setattr(
        store, "judge_knob_refit_rescore_items",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected re-score fault")),
    )
    before = _pointer_bytes(tmp_path)
    result = _tick(store, tmp_path)
    assert result["fired"] is False and result["error"] is not None
    assert _pointer_bytes(tmp_path) == before
    assert _stored_dirs(tmp_path) == {prior.name}, "staged dir reaped"
    assert not any(_consumed(store, rid) for rid in _all_ids(ids))


@pytest.mark.parametrize("grade", GRADES)
def test_knob_write_fault_leaves_prior_serving(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    Env(monkeypatch)
    monkeypatch.setattr(
        store, "write_judge_knob_calibration",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected knob-write fault")),
    )
    result = _tick(store, tmp_path)
    assert result["error"] is not None
    # F2c inc9 knob-first: the knob write precedes the swap, so the pointer
    # is never moved (no rollback step exists).
    assert _current(tmp_path) == prior, "the prior checkpoint keeps serving"
    assert _stored_dirs(tmp_path) == {prior.name}
    assert not any(_consumed(store, rid) for rid in _all_ids(ids))


@pytest.mark.parametrize("grade", GRADES)
def test_first_ever_tune_knob_write_fault_leaves_base_serving(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    Env(monkeypatch)
    monkeypatch.setattr(
        store, "write_judge_knob_calibration",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected knob-write fault")),
    )
    result = _tick(store, tmp_path)
    assert result["fired"] is False and result["error"] is not None
    assert _current(tmp_path) is None, "first-ever tune: no pointer was ever written -> base (I9)"
    assert not any(_consumed(store, rid) for rid in _all_ids(ids))


@pytest.mark.parametrize("grade", GRADES)
def test_fault_after_commit_keeps_new_checkpoint_with_its_knob(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S4: a fault after the commit (the swap) keeps the new checkpoint with
    its own knob (inc7 bite: the as-built code rolled the pointer back even
    after the new knob was written)."""
    _place_checkpoint(tmp_path)
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    monkeypatch.setattr(
        store, "mark_selftune_consumed",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected consume fault")),
    )
    result = _tick(store, tmp_path)
    assert result["error"] is not None
    new = Path(env.builds[0]["staged"])
    assert _current(tmp_path) == new, "committed: the new checkpoint keeps serving"
    knob = _knob(store, new)
    fresh = [(3.14, label) for (_q, _d, label) in store.judge_knob_refit_rescore_items(_all_ids(ids))]
    assert (knob["slope"], knob["intercept"]) == pytest.approx(judge_selftune.fit_platt_knob(fresh))
    assert not any(_consumed(store, rid) for rid in _all_ids(ids))


# ---------------------------------------------------------------------------
# L1 / L2 — each week's update is applied to the persisting model (AC12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("week1", "week2"),
    [
        (judge_selftune.TUNE_GRADE_LORA, judge_selftune.TUNE_GRADE_LORA),
        (judge_selftune.TUNE_GRADE_FULL_FT, judge_selftune.TUNE_GRADE_FULL_FT),
        (judge_selftune.TUNE_GRADE_LORA, judge_selftune.TUNE_GRADE_FULL_FT),
        (judge_selftune.TUNE_GRADE_FULL_FT, judge_selftune.TUNE_GRADE_LORA),
    ],
)
def test_week_two_starts_from_week_one_champion_and_trains_on_its_own_rows(
    week1: str, week2: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, week1)
    r1 = _tick(store, tmp_path)
    assert r1["accepted"] is True
    c1 = _current(tmp_path)
    assert env.builds[0]["start"] == MODEL_RELEVANCE_JUDGE, "a never-tuned persona starts from base"

    _seed_scenario(store, "w2")
    _set_tier(monkeypatch, week2)
    r2 = _tick(store, tmp_path)
    assert r2["accepted"] is True
    assert env.builds[1]["method"] == week2
    assert env.builds[1]["start"] == str(c1), "week 2 is applied ON TOP OF week 1's champion"
    queries = {q for (q, _d, _l) in env.builds[1]["train_items"]}
    assert queries and all(q.startswith("w2") for q in queries), "trains on this tick's labels only"
    assert r2["start"] == c1.name and r2["current"] == c1.name
    assert _current(tmp_path) == Path(env.builds[1]["staged"])


@pytest.mark.parametrize("grade", GRADES)
def test_week_after_a_revert_starts_from_the_unchanged_champion(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    _small_tunables(monkeypatch)
    _set_tier(monkeypatch, grade)
    env = Env(monkeypatch)
    env.accept = False
    _seed_scenario(store, "w1")
    assert _tick(store, tmp_path)["accepted"] is False
    rejected = Path(env.builds[0]["staged"])
    assert not rejected.exists(), "the rejected attempt is gone"

    env.accept = True
    _seed_scenario(store, "w2")
    assert _tick(store, tmp_path)["accepted"] is True
    assert env.builds[1]["start"] == str(prior), "trains the model that never got the rejected adjustment"


# ---------------------------------------------------------------------------
# L3 — dispatch matrix (6 cells) / L4 — champion is the serving model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_checkpoint", [False, True])
@pytest.mark.parametrize(
    "grade",
    [judge_selftune.TUNE_GRADE_KNOB_REFIT, judge_selftune.TUNE_GRADE_LORA, judge_selftune.TUNE_GRADE_FULL_FT],
)
def test_dispatch_matrix(
    grade: str, has_checkpoint: bool, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path) if has_checkpoint else None
    _seed_scenario(store)
    _set_tier(monkeypatch, grade)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    before = _pointer_bytes(tmp_path)
    result = _tick(store, tmp_path)
    assert result["fired"] is True and result["method"] == grade
    if grade == judge_selftune.TUNE_GRADE_KNOB_REFIT:
        assert env.builds == [], "knob week trains no weights"
        assert _pointer_bytes(tmp_path) == before
        # H4 on a knob week: current and start both name the serving model.
        expected = prior.name if prior is not None else "base"
        assert result["current"] == expected and result["start"] == expected
        return
    expected_start = str(prior) if prior is not None else MODEL_RELEVANCE_JUDGE
    assert [b["method"] for b in env.builds] == [grade]
    assert env.builds[0]["start"] == expected_start
    assert _current(tmp_path) == Path(env.builds[0]["staged"])
    # L4: the champion evaluated was the serving model.
    if prior is not None:
        assert env.scorer_dirs[0] == str(prior), "champion loaded from the current checkpoint"
    else:
        assert env.scorer_dirs == [str(_current(tmp_path))], "base champion: only the re-score loaded"


# ---------------------------------------------------------------------------
# L5 / L6 — a RAM tier change never swaps in a different model (AC13)
# ---------------------------------------------------------------------------


def test_drop_to_weak_tier_only_refits_the_knob_on_the_same_model(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_FULL_FT)
    assert _tick(store, tmp_path)["accepted"] is True
    before_ptr, before_dirs = _pointer_bytes(tmp_path), _stored_dirs(tmp_path)

    ids = _seed_scenario(store, "w2")
    # The week-1 doc-absent row stays unconsumed after an accept (2B), so the
    # weak week's logged fit covers every row the gate scanned this tick.
    _count, tick_rows = store.count_new_haiku_decisions()
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_KNOB_REFIT)
    result = _tick(store, tmp_path)
    assert result["fired"] is True and result["method"] == judge_selftune.TUNE_GRADE_KNOB_REFIT
    served = _current(tmp_path).name
    assert result["current"] == served and result["start"] == served
    assert _pointer_bytes(tmp_path) == before_ptr and _stored_dirs(tmp_path) == before_dirs
    assert len(env.builds) == 1, "no weight training on the weak week"
    logged = judge_selftune.fit_platt_knob(store.judge_knob_refit_pairs(tick_rows))
    knob = _knob(store, _current(tmp_path))
    assert (knob["slope"], knob["intercept"]) == pytest.approx(logged)
    assert all(_consumed(store, rid) for rid in tick_rows) and set(_all_ids(ids)) <= set(tick_rows)


@pytest.mark.parametrize("accept_week2", [True, False])
def test_beefy_to_mid_drop_runs_lora_on_the_same_model(
    accept_week2: bool, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_FULL_FT)
    assert _tick(store, tmp_path)["accepted"] is True
    c1 = _current(tmp_path)
    before_ptr = _pointer_bytes(tmp_path)

    env.accept = accept_week2
    _seed_scenario(store, "w2")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_LORA)
    result = _tick(store, tmp_path)
    assert env.builds[1]["method"] == judge_selftune.TUNE_GRADE_LORA
    assert env.builds[1]["start"] == str(c1), "the LoRA trains on the fine-tuned model, not the base"
    if accept_week2:
        assert result["accepted"] is True
        assert _current(tmp_path) == Path(env.builds[1]["staged"])
        assert _stored_dirs(tmp_path) == {Path(env.builds[1]["staged"]).name}, "previous deleted (Q1)"
    else:
        assert result["accepted"] is False
        assert _pointer_bytes(tmp_path) == before_ptr and c1.exists()


# ---------------------------------------------------------------------------
# S3 — delete-after-swap (ruling Q1), with retry on a later tick
# ---------------------------------------------------------------------------


def test_stored_dir_set_across_a_week_sequence(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)

    def week(tag: str, grade: str, accept: bool) -> dict:
        env.accept = accept
        _seed_scenario(store, tag)
        _set_tier(monkeypatch, grade)
        return _tick(store, tmp_path)

    assert week("a", judge_selftune.TUNE_GRADE_FULL_FT, True)["accepted"] is True
    c1 = Path(env.builds[-1]["staged"]).name
    assert _stored_dirs(tmp_path) == {c1}

    assert week("b", judge_selftune.TUNE_GRADE_LORA, True)["accepted"] is True
    c2 = Path(env.builds[-1]["staged"]).name
    assert _stored_dirs(tmp_path) == {c2}, "C1 deleted in the same tick as the swap"

    assert week("c", judge_selftune.TUNE_GRADE_KNOB_REFIT, True)["method"] == judge_selftune.TUNE_GRADE_KNOB_REFIT
    assert _stored_dirs(tmp_path) == {c2}

    assert week("d", judge_selftune.TUNE_GRADE_LORA, False)["accepted"] is False
    assert _stored_dirs(tmp_path) == {c2}

    assert week("e", judge_selftune.TUNE_GRADE_FULL_FT, True)["accepted"] is True
    c3 = Path(env.builds[-1]["staged"]).name
    assert _stored_dirs(tmp_path) == {c3}
    assert _current(tmp_path).name == c3


def test_refused_delete_is_not_raised_and_a_later_tick_reaps_it(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    ids = _seed_scenario(store)
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_LORA)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    real_rmtree = judge_lora.shutil.rmtree
    refuse = {"on": True}

    def rmtree(path, *a, **k):
        if refuse["on"] and Path(path) == prior:
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(judge_lora.shutil, "rmtree", rmtree)

    result = _tick(store, tmp_path)
    assert result["accepted"] is True and result["error"] is None, "a refused delete is not raised"
    new = Path(env.builds[0]["staged"])
    assert _current(tmp_path) == new
    assert all(_consumed(store, rid) for rid in [ids["r1"], ids["r2"], ids["r3"]])
    assert prior.exists(), "leftover stays until a later tick"

    # Next tick: nothing new to train on (gate does not fire), the OS now
    # allows the delete, and the leftover is reaped; the served one is kept.
    refuse["on"] = False
    r2 = _tick(store, tmp_path)
    assert r2["fired"] is False
    assert not prior.exists()
    assert _current(tmp_path) == new and _stored_dirs(tmp_path) == {new.name}


def test_pointer_to_a_legacy_adapter_dir_trains_from_base_and_is_not_reaped_early(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S5 at tick level: an inc5/inc6 adapter dir named by the pointer is not a
    plain checkpoint -> the week trains from base; the reap never deletes the
    dir the pointer names before the swap."""
    root = _root(tmp_path)
    legacy = judge_lora.staged_adapter_path(root)
    legacy.mkdir(parents=True)
    (legacy / "adapter_config.json").write_text("{}", encoding="utf-8")
    judge_lora.swap_champion_pointer(root, legacy)
    _seed_scenario(store)
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_LORA)
    _small_tunables(monkeypatch)
    env = Env(monkeypatch)
    env.accept = False
    result = _tick(store, tmp_path)
    assert result["accepted"] is False
    assert env.builds[0]["start"] == MODEL_RELEVANCE_JUDGE
    assert legacy.exists(), "the pointer-named dir is never reaped while the pointer names it"


# ---------------------------------------------------------------------------
# AC11 — per-persona isolation at the tick
# ---------------------------------------------------------------------------


def test_one_personas_accept_does_not_touch_another_personas_store(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a, b = tmp_path / "A", tmp_path / "B"
    b_ckpt = _place_checkpoint(b)
    b_before = (_pointer_bytes(b), _stored_dirs(b))
    _seed_scenario(store)
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_FULL_FT)
    _small_tunables(monkeypatch)
    Env(monkeypatch)
    assert _tick(store, a)["accepted"] is True
    assert (_pointer_bytes(b), _stored_dirs(b)) == b_before and b_ckpt.exists()
