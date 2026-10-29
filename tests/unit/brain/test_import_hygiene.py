"""The judge self-tune modules and their knob-refit path never import the heavy training stack (I6/AC8).

One fresh subprocess (never inheriting this process's own `sys.modules`, which earlier tests may already have filled
with torch) imports the four judge modules and runs the firing self-tune tick, a champion/challenger pass and the Platt
knob fit. After each step none of torch, sentence_transformers, peft or datasets may be in `sys.modules`, and the failing
step is named. The training tiers (`judge_lora`, `judge_full_ft`) import those packages lazily, inside the functions that
train; only the real training runs may pull them in.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys
    from datetime import UTC, datetime

    HEAVY = ("torch", "sentence_transformers", "peft", "datasets")


    def check(step):
        for heavy in HEAVY:
            assert heavy not in sys.modules, f"{heavy} was imported by: {step}"
        print("ok:", step, flush=True)


    from brain.memory import judge_selftune
    check("import judge_selftune")
    from brain.memory import judge_eval
    check("import judge_eval")
    from brain.memory import judge_lora  # noqa: F401
    check("import judge_lora")
    from brain.memory import judge_full_ft  # noqa: F401
    check("import judge_full_ft")

    # The firing self-tune tick: enough labelled calibration rows (with raw judge scores) to pass the >handful gate,
    # so the knob refit really trains and persists.
    from brain import dev_constants
    from brain.memory.store import MemoryStore

    store = MemoryStore(db_path=":memory:")
    handful = dev_constants.JUDGE_SELFTUNE_GATE_HANDFUL_DECISIONS
    for i in range(handful + 1):
        store.log_calibration_sample(
            query=f"q{i}", candidate_ids=["m"], reranker_scores=[1.0], reranker_model_id="m"
        )
        row_id = store._conn.execute("SELECT id FROM calibration_log ORDER BY id DESC LIMIT 1").fetchone()["id"]
        store.write_calibration_labels(row_id, ["relevant"], ["relevant"], local_judge_raw_score=[2.0])
    result = judge_selftune._run_judge_selftune_tick(store=store, now=datetime.now(UTC))
    assert result["fired"] is True, result
    assert store.get_judge_knob_calibration(judge_selftune.MODEL_RELEVANCE_JUDGE) is not None
    check("the firing self-tune tick")

    # A full champion/challenger pass with a challenger REJECTION (which also runs rollback.restore).
    test_items = [(f"t{i}", "relevant") for i in range(8)]
    labels = {f"t{i}": "irrelevant" for i in range(8)}
    verdict = judge_eval.run_champion_challenger(
        champion=lambda item: "relevant",
        retrain_fn=lambda train_items: (lambda item: labels[item]),
        train_items=["x"],
        test_items=test_items,
        rollback=judge_eval.RollbackHandle(),
        alpha=0.05,
        min_n=1,
    )
    assert verdict.accepted is False, verdict
    check("the champion/challenger pass with a rejection")

    # The Platt fit on a two-class sample (the tick above seeds one class, which returns before the fit loop).
    slope, intercept = judge_selftune.fit_platt_knob([(-3.0, "irrelevant"), (3.0, "relevant")])
    assert isinstance(slope, float) and isinstance(intercept, float)
    check("fit_platt_knob")
    print("SUBPROCESS_OK")
    """
)


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_judge_modules_stay_torch_free() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SUBPROCESS_OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
