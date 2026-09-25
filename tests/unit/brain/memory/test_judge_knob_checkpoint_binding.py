"""F2c inc9 — the Platt knob is bound to the checkpoint it was fit on (spec §5
knob-first order): the knob row is written FIRST under
`<base id>@<checkpoint>`, the pointer swap is the single commit point, then the
previous checkpoint and its knob row are deleted; a knob row of a checkpoint
the pointer does not name is an orphan reaped on the next tick.

Drives the real `_run_judge_selftune_tick` with the lifecycle suite's scripted
no-model fakes, and checks what the DAILY serve path applies through the real
`relevance_judge.label_calibration_sample` (the judge built the way
`supervisor._run_calibration_tick` builds it: `full_model_dir` =
`resolve_current_checkpoint`, a `FullModelJudge` with a scripted scorer).
Synthetic data only, never a live persona (I11). Criteria ids (K1..K11)
refer to `changes/f2c-inc9-open-findings/1.5-criteria.md`.

"P's knob" is slope 1.0 / intercept 1.0: at raw score -0.5 it labels
"relevant", while the fixed sigmoid-0.5 mapping labels "irrelevant".
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE
from brain.memory import judge_full_ft, judge_lora, judge_selftune, relevance_judge
from brain.memory.relevance_judge import FakeRelevanceJudgeProvider, label_for_score
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.memory.test_judge_selftune_lifecycle import (
    GRADES,
    Env,
    _consumed,
    _current,
    _place_checkpoint,
    _seed_scenario,
    _set_tier,
    _small_tunables,
    _tick,
)

RAW = -0.5


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


class _BaseJudge(FakeRelevanceJudgeProvider):
    """The base judge as production builds it: `model_id()` is the real id."""

    def model_id(self) -> str:
        return MODEL_RELEVANCE_JUDGE


def _key(checkpoint: Path | str | None) -> str:
    from brain.memory.relevance_judge import judge_knob_key

    return judge_knob_key(MODEL_RELEVANCE_JUDGE, checkpoint)


def _serve_label(store: MemoryStore, persona_dir: Path, monkeypatch: pytest.MonkeyPatch, raw: float = RAW) -> str:
    """Label one fresh calibration row through the daily serve path and return
    the label the persona's served judge + knob produce for `raw`."""
    monkeypatch.setattr(judge_full_ft, "load_full_scorer", lambda d, **kw: (lambda item: raw))

    def build(full_model_dir=None):
        if full_model_dir is not None:
            return relevance_judge.FullModelJudge(MODEL_RELEVANCE_JUDGE, full_model_dir)
        return _BaseJudge(default=raw)

    monkeypatch.setattr(relevance_judge, "build_judge_provider", build)
    mem = Memory.create_new(content=f"serve-{uuid.uuid4().hex}", memory_type="conversation", domain="us")
    store.create(mem)
    store.log_calibration_sample(query="serve-q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m")
    row_id = store._conn.execute("SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1").fetchone()["id"]
    current = judge_lora.resolve_current_checkpoint(persona_dir)
    labeled = relevance_judge.label_calibration_sample(
        store, full_model_dir=str(current) if current is not None else None
    )
    assert labeled >= 1
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log WHERE id = ?", (row_id,)).fetchone()
    return json.loads(row["local_judge_label"])[0]


def _seed_serving_knob(store: MemoryStore, monkeypatch: pytest.MonkeyPatch, persona_dir: Path) -> None:
    """Commit-agnostic: a knob week on whatever serves writes the single knob
    row (under whatever key this commit uses), then its params are set to
    1.0/1.0 (P's knob)."""
    _small_tunables(monkeypatch)
    _seed_scenario(store, "w0")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_KNOB_REFIT)
    result = _tick(store, persona_dir)
    assert result["fired"] is True and result["error"] is None
    n = store._conn.execute("SELECT COUNT(*) AS n FROM judge_knob_calibration").fetchone()["n"]
    assert n == 1
    store._conn.execute("UPDATE judge_knob_calibration SET slope = 1.0, intercept = 1.0")
    store._conn.commit()


def _raise(*a, **k):
    raise RuntimeError("injected fault")


def _gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _small_tunables(monkeypatch, gate=10_000)


def _discriminating_raw(
    slope: float, intercept: float, other: tuple[float | None, float | None] = (None, None)
) -> float:
    """A raw score where the knob (slope, intercept) and `other` (default: the
    fixed mapping) give different provisional labels (the label the serve path
    stores when no Haiku tie-break provider is given, ambiguous or not)."""
    for i in range(-400, 401):
        r = i / 20.0
        a, _ = label_for_score(r, slope=slope, intercept=intercept)
        b, _ = label_for_score(r, slope=other[0], intercept=other[1])
        if a != b:
            return r
    raise AssertionError(f"no raw score separates knob ({slope}, {intercept}) from {other}")


# ---------------------------------------------------------------------------
# K1 — the inc7 F2 double fault cannot serve a stale knob (bite at 4eb4d199)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_k1_double_fault_never_serves_a_stale_knob(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    Env(monkeypatch)
    _seed_serving_knob(store, monkeypatch, tmp_path)
    assert _serve_label(store, tmp_path, monkeypatch) == "relevant", "sanity: P's knob applies to P"

    env = Env(monkeypatch)
    ids = _seed_scenario(store, "w1")
    _set_tier(monkeypatch, grade)
    monkeypatch.setattr(store, "write_judge_knob_calibration", _raise)
    real_swap = judge_lora.swap_champion_pointer
    calls: list[int] = []

    def swap(root, staged):
        calls.append(1)
        if len(calls) >= 2:  # the old post-swap rollback (only reached at 4eb4d199)
            raise RuntimeError("injected rollback fault")
        return real_swap(root, staged)

    monkeypatch.setattr(judge_lora, "swap_champion_pointer", swap)
    result = _tick(store, tmp_path)

    assert result["error"] is not None
    assert not any(_consumed(store, rid) for rid in ids.values())
    assert env.builds, "the weight-retrain ran"
    assert _current(tmp_path) == prior, "the previous checkpoint still serves (never a new one with P's knob)"
    assert _serve_label(store, tmp_path, monkeypatch) == "relevant", "P serves with its own knob"


# ---------------------------------------------------------------------------
# K2 — normal ACCEPT: knob first, swap = commit, then P and its row go
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_k2_accept_writes_knob_first_then_swaps_then_deletes_the_previous(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    Env(monkeypatch)
    _seed_serving_knob(store, monkeypatch, tmp_path)
    env = Env(monkeypatch)
    ids = _seed_scenario(store, "w1")
    _set_tier(monkeypatch, grade)

    events: list[tuple[str, str]] = []
    real_write, real_delete, real_consume = (
        store.write_judge_knob_calibration,
        store.delete_judge_knob_calibration,
        store.mark_selftune_consumed,
    )
    real_replace = judge_lora.os.replace
    real_rmtree = judge_lora.shutil.rmtree

    def write(key, **kw):
        events.append(("knob_write", key))
        return real_write(key, **kw)

    def delete(key):
        events.append(("knob_delete", key))
        return real_delete(key)

    def consume(row_ids, **kw):
        events.append(("consume", ""))
        return real_consume(row_ids, **kw)

    def replace(src, dst):
        if str(dst).endswith("current"):
            events.append(("swap", Path(src).read_text(encoding="utf-8")))
        return real_replace(src, dst)

    def rmtree(p, *a, **k):
        events.append(("rmtree", Path(p).name))
        return real_rmtree(p, *a, **k)

    monkeypatch.setattr(store, "write_judge_knob_calibration", write)
    monkeypatch.setattr(store, "delete_judge_knob_calibration", delete)
    monkeypatch.setattr(store, "mark_selftune_consumed", consume)
    monkeypatch.setattr(judge_lora.os, "replace", replace)
    monkeypatch.setattr(judge_lora.shutil, "rmtree", rmtree)

    result = _tick(store, tmp_path)
    assert result["accepted"] is True and result["error"] is None
    new = Path(env.builds[-1]["staged"])
    assert _current(tmp_path) == new

    i_write = events.index(("knob_write", _key(new)))
    i_swap = events.index(("swap", new.name))
    i_rm = events.index(("rmtree", prior.name))
    i_del = events.index(("knob_delete", _key(prior)))
    i_consume = events.index(("consume", ""))
    assert i_write < i_swap < i_rm < i_del < i_consume, events

    fresh = [(3.14, label) for (_q, _d, label) in store.judge_knob_refit_rescore_items(list(ids.values()))]
    slope, intercept = judge_selftune.fit_platt_knob(fresh)
    knob = store.get_judge_knob_calibration(_key(new))
    assert (knob["slope"], knob["intercept"]) == pytest.approx((slope, intercept)), "Option A re-score"
    assert store.get_judge_knob_calibration(_key(prior)) is None, "P's knob row is gone"
    raw = _discriminating_raw(slope, intercept, other=(1.0, 1.0))  # new knob vs P's
    assert _serve_label(store, tmp_path, monkeypatch, raw) == label_for_score(raw, slope=slope, intercept=intercept)[0]


# ---------------------------------------------------------------------------
# K3 / K3b — crash between the knob write and the swap; the next tick reaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_k3_crash_between_knob_write_and_swap_keeps_prior_and_next_tick_reaps_orphan(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    Env(monkeypatch)
    _seed_serving_knob(store, monkeypatch, tmp_path)
    store.write_judge_knob_calibration(MODEL_RELEVANCE_JUDGE, slope=5.0, intercept=5.0)  # a base row
    env = Env(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, grade)
    real_swap = judge_lora.swap_champion_pointer
    monkeypatch.setattr(judge_lora, "swap_champion_pointer", _raise)

    result = _tick(store, tmp_path)
    assert result["error"] is not None
    staged = Path(env.builds[-1]["staged"])
    assert _current(tmp_path) == prior
    assert _serve_label(store, tmp_path, monkeypatch) == "relevant", "P serves with its own knob"
    assert store.get_judge_knob_calibration(_key(staged)) is not None, "precondition: the orphan row exists"

    monkeypatch.setattr(judge_lora, "swap_champion_pointer", real_swap)
    _gate_off(monkeypatch)
    nxt = _tick(store, tmp_path)
    assert nxt["fired"] is False
    assert store.get_judge_knob_calibration(_key(staged)) is None, "the orphan is reaped next tick"
    assert store.get_judge_knob_calibration(_key(prior)) is not None, "P's own row is kept"
    assert store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE) is not None, "the base row is kept"
    assert not staged.exists()


def test_k3b_first_ever_crash_before_swap_keeps_base_and_reaps_orphan(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Env(monkeypatch)
    _seed_serving_knob(store, monkeypatch, tmp_path)  # base judge's knob week
    env = Env(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_LORA)
    real_swap = judge_lora.swap_champion_pointer
    monkeypatch.setattr(judge_lora, "swap_champion_pointer", _raise)

    result = _tick(store, tmp_path)
    assert result["error"] is not None
    staged = Path(env.builds[-1]["staged"])
    assert _current(tmp_path) is None, "the base judge still serves"
    assert _serve_label(store, tmp_path, monkeypatch) == "relevant", "with the base judge's own knob"
    assert store.get_judge_knob_calibration(_key(staged)) is not None, "precondition: the orphan row exists"

    monkeypatch.setattr(judge_lora, "swap_champion_pointer", real_swap)
    _gate_off(monkeypatch)
    _tick(store, tmp_path)
    assert store.get_judge_knob_calibration(_key(staged)) is None
    assert store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE) is not None


# ---------------------------------------------------------------------------
# K5 — the keyed knob load stays torch-free
# ---------------------------------------------------------------------------


def test_k5_keyed_knob_load_is_torch_free() -> None:
    script = textwrap.dedent(
        """
        import json, sys
        from brain.memory import judge_full_ft
        from brain.memory.relevance_judge import FullModelJudge, judge_knob_key, label_calibration_sample
        from brain.memory.store import Memory, MemoryStore

        judge_full_ft.load_full_scorer = lambda d, **kw: (lambda item: -0.5)
        store = MemoryStore(db_path=":memory:")
        mem = Memory.create_new(content="borderline", memory_type="conversation", domain="us")
        store.create(mem)
        store.log_calibration_sample(query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m")
        judge = FullModelJudge("base-id", "/nowhere/adapter-abc")
        store.write_judge_knob_calibration(judge_knob_key("base-id", "/nowhere/adapter-abc"), slope=1.0, intercept=1.0)
        assert label_calibration_sample(store, judge=judge) == 1
        row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
        assert json.loads(row["local_judge_label"]) == ["relevant"], "the checkpoint's own knob applies"
        assert "torch" not in sys.modules
        assert "sentence_transformers" not in sys.modules
        print("SUBPROCESS_OK")
        """
    )
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root), capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout


# ---------------------------------------------------------------------------
# K7b — the pre-inc9 residue (a checkpoint whose knob sits under the base id)
# ---------------------------------------------------------------------------


def test_k7b_pre_inc9_base_keyed_knob_is_not_applied_and_heals(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    store.write_judge_knob_calibration(MODEL_RELEVANCE_JUDGE, slope=1.0, intercept=1.0)
    assert _serve_label(store, tmp_path, monkeypatch) == "irrelevant", "a base-id knob never applies to a checkpoint"

    Env(monkeypatch)
    _small_tunables(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_KNOB_REFIT)
    assert _tick(store, tmp_path)["fired"] is True
    knob = store.get_judge_knob_calibration(_key(prior))
    assert knob is not None, "the knob week binds a knob to the serving checkpoint"
    raw = _discriminating_raw(knob["slope"], knob["intercept"])
    assert _serve_label(store, tmp_path, monkeypatch, raw) == label_for_score(
        raw, slope=knob["slope"], intercept=knob["intercept"]
    )[0]
    assert store.get_judge_knob_calibration(MODEL_RELEVANCE_JUDGE) is not None, "the base row is never reaped"


# ---------------------------------------------------------------------------
# K8 — the knob looked up is bound to the judge object that scores (bite)
# ---------------------------------------------------------------------------


def test_k8_injected_full_model_judge_never_reads_the_base_knob(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_full_ft, "load_full_scorer", lambda d, **kw: (lambda item: RAW))
    judge = relevance_judge.FullModelJudge(MODEL_RELEVANCE_JUDGE, "/nowhere/adapter-x")
    store.write_judge_knob_calibration(MODEL_RELEVANCE_JUDGE, slope=1.0, intercept=1.0)
    mem = Memory.create_new(content="borderline", memory_type="conversation", domain="us")
    store.create(mem)
    store.log_calibration_sample(query="q", candidate_ids=[mem.id], reranker_scores=[1.0], reranker_model_id="m")

    assert relevance_judge.label_calibration_sample(store, judge=judge) == 1  # no full_model_dir passed
    row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
    assert json.loads(row["local_judge_label"]) == ["irrelevant"], "the base judge's knob is not this checkpoint's"


# ---------------------------------------------------------------------------
# K9 — a rejected challenger leaves no knob row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grade", GRADES)
def test_k9_rejected_challenger_leaves_no_knob_row(
    grade: str, store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    env = Env(monkeypatch)
    env.accept = False
    _small_tunables(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, grade)
    result = _tick(store, tmp_path)
    assert result["accepted"] is False and result["error"] is None
    staged = Path(env.builds[-1]["staged"])
    tuned_keys = [k for k in store.list_judge_knob_keys() if k.startswith(MODEL_RELEVANCE_JUDGE + "@")]
    assert tuned_keys == [_key(prior)], "only the serving checkpoint's knob; none for the rejected one"
    assert store.get_judge_knob_calibration(_key(staged)) is None

    _gate_off(monkeypatch)
    _tick(store, tmp_path)
    assert sorted(store.list_judge_knob_keys()) == sorted(tuned_keys)


# ---------------------------------------------------------------------------
# K10 — the orphan reap never deletes the served knob or the base row
# ---------------------------------------------------------------------------


def test_k10a_reap_keeps_the_served_checkpoints_row_after_an_accept(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = Env(monkeypatch)
    _small_tunables(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_LORA)
    assert _tick(store, tmp_path)["accepted"] is True
    new = Path(env.builds[-1]["staged"])
    _gate_off(monkeypatch)
    _tick(store, tmp_path)
    assert store.get_judge_knob_calibration(_key(new)) is not None


def test_k10b_empty_pointer_reaps_nothing(store: MemoryStore, tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    root.mkdir(parents=True)
    (root / "current").write_text("", encoding="utf-8")
    store.write_judge_knob_calibration(_key("adapter-x"), slope=1.0, intercept=0.0)
    store.write_judge_knob_calibration(_key("adapter-y"), slope=1.0, intercept=0.0)
    assert judge_selftune._reap_orphan_knob_rows(store, tmp_path) == 0
    assert len(store.list_judge_knob_keys()) == 2


def test_k10c_absent_pointer_reaps_tuned_rows_only(store: MemoryStore, tmp_path: Path) -> None:
    store.write_judge_knob_calibration(MODEL_RELEVANCE_JUDGE, slope=1.0, intercept=0.0)
    store.write_judge_knob_calibration("other-model@adapter-x", slope=1.0, intercept=0.0)
    store.write_judge_knob_calibration(_key("adapter-x"), slope=1.0, intercept=0.0)
    assert judge_selftune._reap_orphan_knob_rows(store, tmp_path) == 1
    assert sorted(store.list_judge_knob_keys()) == sorted([MODEL_RELEVANCE_JUDGE, "other-model@adapter-x"])


# ---------------------------------------------------------------------------
# K11 — a faulted post-commit delete of P's row is swept by the next tick
# ---------------------------------------------------------------------------


def test_k11_faulted_post_commit_row_delete_is_swept_next_tick(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior = _place_checkpoint(tmp_path)
    Env(monkeypatch)
    _seed_serving_knob(store, monkeypatch, tmp_path)
    env = Env(monkeypatch)
    _seed_scenario(store, "w1")
    _set_tier(monkeypatch, judge_selftune.TUNE_GRADE_FULL_FT)
    real_delete = store.delete_judge_knob_calibration
    monkeypatch.setattr(store, "delete_judge_knob_calibration", _raise)

    result = _tick(store, tmp_path)
    assert result["accepted"] is True and result["error"] is None, "a best-effort delete never fails the tune"
    new = Path(env.builds[-1]["staged"])
    assert _current(tmp_path) == new
    assert store.get_judge_knob_calibration(_key(new)) is not None
    assert store.get_judge_knob_calibration(_key(prior)) is not None, "precondition: P's row survived the tick"

    monkeypatch.setattr(store, "delete_judge_knob_calibration", real_delete)
    _gate_off(monkeypatch)
    _tick(store, tmp_path)
    assert store.get_judge_knob_calibration(_key(prior)) is None
    assert store.get_judge_knob_calibration(_key(new)) is not None
