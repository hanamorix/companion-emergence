"""F2c inc5b-2/inc6, reworked in inc7 — the daily calibration tick serves a
persona's ONE tuned judge (always a plain checkpoint since inc7: a full
fine-tune or a merged LoRA week) when its weekly self-tune has accepted one,
else the base judge, with no cross-persona bleed (AC11, criterion V2).

Exercises `supervisor._run_calibration_tick`'s resolution
(`judge_lora.resolve_current_checkpoint`) + the `full_model_dir` threading into
`label_calibration_sample.build_judge_provider`. No real model:
`build_judge_provider` is monkeypatched to capture the dir it is asked for.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from brain.bridge.supervisor import _run_calibration_tick
from brain.memory import judge_lora, relevance_judge
from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
from brain.memory.store import MemoryStore


def _seed_unlabeled_row(pd: Path) -> None:
    store = MemoryStore(pd / "memories.db", integrity_check=False)
    store.log_calibration_sample(
        query="q", candidate_ids=["a"], reranker_scores=[1.0], reranker_model_id="jina",
    )
    store.close()


def _place_checkpoint(pd: Path, *, adapter_layout: bool = False) -> Path:
    """Place a stored dir and point the persona's one pointer at it. With
    `adapter_layout` the dir holds an inc5/inc6-style `adapter_config.json`."""
    root = judge_lora.champion_dir(pd)
    staged = judge_lora.staged_adapter_path(root)
    staged.mkdir(parents=True)
    if adapter_layout:
        (staged / "adapter_config.json").write_text("{}", encoding="utf-8")
    else:
        (staged / "model.safetensors.txt").write_text("x", encoding="utf-8")
    judge_lora.swap_champion_pointer(root, staged)
    return staged


def _capture_full_model_dir(monkeypatch) -> list[str | None]:
    seen: list[str | None] = []

    def fake(full_model_dir=None):
        seen.append(full_model_dir)
        return FakeRelevanceJudgeProvider()

    monkeypatch.setattr(relevance_judge, "build_judge_provider", fake)
    return seen


def test_tick_serves_the_persona_checkpoint_when_present(monkeypatch) -> None:
    seen = _capture_full_model_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        ckpt = _place_checkpoint(pd)
        _run_calibration_tick(pd)
        assert seen == [str(ckpt)], f"expected the pointer's checkpoint dir, got {seen}"


def test_tick_serves_base_judge_when_no_checkpoint(monkeypatch) -> None:
    seen = _capture_full_model_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        _run_calibration_tick(pd)
        assert seen == [None], f"expected base judge (full_model_dir=None), got {seen}"


def test_tick_serves_base_judge_for_a_legacy_adapter_dir(monkeypatch) -> None:
    # S5: a pointer naming an inc5/inc6 adapter dir is not a plain checkpoint;
    # the plain loader is never handed it.
    seen = _capture_full_model_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        _place_checkpoint(pd, adapter_layout=True)
        _run_calibration_tick(pd)
        assert seen == [None], f"adapter dir must resolve to the base judge, got {seen}"


def test_tick_ignores_a_legacy_full_sub_store(monkeypatch) -> None:
    # S1/S5: the removed per-tier `full/` sub-store is never consulted.
    seen = _capture_full_model_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        old_full = judge_lora.champion_dir(pd) / "full"
        staged = judge_lora.staged_adapter_path(old_full)
        staged.mkdir(parents=True)
        judge_lora.swap_champion_pointer(old_full, staged)
        _run_calibration_tick(pd)
        assert seen == [None], f"a legacy full/ sub-store must not be served, got {seen}"


def test_no_cross_persona_bleed(monkeypatch) -> None:
    seen = _capture_full_model_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        pd_a, pd_b = Path(da), Path(db)
        _seed_unlabeled_row(pd_a)
        _seed_unlabeled_row(pd_b)
        ckpt_a = _place_checkpoint(pd_a)

        _run_calibration_tick(pd_a)
        _run_calibration_tick(pd_b)

        assert seen[0] == str(ckpt_a)
        assert str(pd_a) in seen[0] and str(pd_b) not in seen[0]
        assert seen[1] is None, "persona B must not see A's checkpoint"


def test_injected_judge_is_not_clobbered_by_a_disk_checkpoint(monkeypatch) -> None:
    seen = _capture_full_model_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        _place_checkpoint(pd)
        injected = FakeRelevanceJudgeProvider()
        _run_calibration_tick(pd, judge=injected)
        assert seen == [], f"injected judge must be used, build_judge_provider not called: {seen}"
