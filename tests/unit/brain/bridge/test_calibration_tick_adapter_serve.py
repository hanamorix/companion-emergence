"""F2c inc5b-2 (C8/C9) + inc6 — the daily calibration tick serves a persona's
TUNED judge when its weekly self-tune has accepted one (a LoRA adapter OR, on
the beefy tier, a FULL fine-tune), else the base judge, with no cross-persona
bleed.

Exercises `supervisor._run_calibration_tick`'s tuned-judge resolution
(`judge_lora.resolve_serving_tuned_judge`, full>lora>base) + the
`adapter_dir`/`full_model_dir` threading into
`label_calibration_sample.build_judge_provider`. No real model:
`build_judge_provider` is monkeypatched to capture the dirs it is asked for.
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


def _place_champion_adapter(pd: Path) -> Path:
    root = judge_lora.champion_dir(pd)
    staged = judge_lora.staged_adapter_path(root)
    staged.mkdir(parents=True)
    (staged / "adapter_model.txt").write_text("x", encoding="utf-8")
    judge_lora.swap_champion_pointer(root, staged)
    return staged


def _place_champion_full_model(pd: Path) -> Path:
    root = judge_lora.full_champion_dir(pd)
    staged = judge_lora.staged_adapter_path(root)
    staged.mkdir(parents=True)
    (staged / "model.safetensors.txt").write_text("x", encoding="utf-8")
    judge_lora.swap_champion_pointer(root, staged)
    return staged


def _capture_adapter_dir(monkeypatch) -> list[str | None]:
    seen: list[str | None] = []

    # inc6: build_judge_provider now takes full_model_dir too (additive). The
    # tick passes exactly one of adapter_dir / full_model_dir (or neither);
    # capture whichever is non-None (else None = base).
    def fake(adapter_dir=None, full_model_dir=None):
        seen.append(full_model_dir if full_model_dir is not None else adapter_dir)
        return FakeRelevanceJudgeProvider()

    monkeypatch.setattr(relevance_judge, "build_judge_provider", fake)
    return seen


def test_tick_serves_the_persona_tuned_adapter_when_present(monkeypatch) -> None:
    # C8: an accepted champion adapter → the tick labels via
    # build_judge_provider(adapter_dir=<the resolved adapter>).
    seen = _capture_adapter_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        staged = _place_champion_adapter(pd)
        _run_calibration_tick(pd)
        assert seen == [str(staged)], f"expected the resolved adapter dir, got {seen}"


def test_tick_serves_base_judge_when_no_adapter(monkeypatch) -> None:
    # C8: absent champion adapter → base judge (adapter_dir=None), byte-identical
    # to pre-inc5b2.
    seen = _capture_adapter_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        _run_calibration_tick(pd)
        assert seen == [None], f"expected base judge (adapter_dir=None), got {seen}"


def test_tick_serves_the_full_model_when_present(monkeypatch) -> None:
    # inc6 (C11/C12): a beefy-tier FULL champion → the tick labels via
    # build_judge_provider(full_model_dir=<the resolved full dir>), NOT adapter_dir.
    seen: list[tuple[str | None, str | None]] = []

    def fake(adapter_dir=None, full_model_dir=None):
        seen.append((adapter_dir, full_model_dir))
        return FakeRelevanceJudgeProvider()

    monkeypatch.setattr(relevance_judge, "build_judge_provider", fake)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        full = _place_champion_full_model(pd)
        _run_calibration_tick(pd)
        assert seen == [(None, str(full))], f"expected full_model_dir set, adapter_dir None: {seen}"


def test_tick_full_model_takes_precedence_over_a_stale_lora_adapter(monkeypatch) -> None:
    # inc6 (C12 precedence): if BOTH stores are populated (a cross-tier window),
    # the tick serves the FULL model (full>lora).
    seen: list[tuple[str | None, str | None]] = []

    def fake(adapter_dir=None, full_model_dir=None):
        seen.append((adapter_dir, full_model_dir))
        return FakeRelevanceJudgeProvider()

    monkeypatch.setattr(relevance_judge, "build_judge_provider", fake)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        _place_champion_adapter(pd)
        full = _place_champion_full_model(pd)
        _run_calibration_tick(pd)
        assert seen == [(None, str(full))], f"full must win over a stale lora adapter: {seen}"


def test_no_cross_persona_adapter_bleed(monkeypatch) -> None:
    # C9: persona A has an adapter, B has none. A's tick resolves A's adapter
    # (under A's dir); B's tick resolves nothing — no bleed.
    seen = _capture_adapter_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        pd_a, pd_b = Path(da), Path(db)
        _seed_unlabeled_row(pd_a)
        _seed_unlabeled_row(pd_b)
        staged_a = _place_champion_adapter(pd_a)

        _run_calibration_tick(pd_a)
        _run_calibration_tick(pd_b)

        assert seen[0] == str(staged_a)
        assert str(pd_a) in seen[0] and str(pd_b) not in seen[0]
        assert seen[1] is None, "persona B must not see A's adapter"


def test_injected_judge_is_not_clobbered_by_a_disk_adapter(monkeypatch) -> None:
    # Finding 6: an explicitly injected judge takes precedence; the tick does
    # NOT resolve/serve a disk adapter over it.
    seen = _capture_adapter_dir(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        _seed_unlabeled_row(pd)
        _place_champion_adapter(pd)
        injected = FakeRelevanceJudgeProvider()
        _run_calibration_tick(pd, judge=injected)
        # build_judge_provider is never called (the injected judge is used).
        assert seen == [], f"injected judge must be used, build_judge_provider not called: {seen}"
