"""Name-recall fix R7 (spec section 6, S48; plan P-22; criterion R7-T1): the daily
calibration tick prunes the recall diagnostics log on the calibration log's
retention window, right after the calibration log and before the judge-labeling
step, and never from a recall path.

Offline: a real `_run_calibration_tick` over a tmp persona directory with a stub
labeling step; the log is written by hand with timestamps either side of the cutoff.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants, tunables
from brain.bridge.supervisor import _run_calibration_tick
from brain.memory import recall_diagnostics as rd
from brain.memory import relevance_judge as rj_mod
from brain.memory.store import CALIBRATION_LOG_RETENTION_WINDOW_DAYS, MemoryStore

_LOG = dev_constants.RECALL_DIAGNOSTICS_LOG_FILENAME


def _line(ts: datetime, tag: str) -> str:
    return json.dumps({"ts": ts.isoformat(), "source": "passive", "tag": tag})


def _seed_log(persona: Path, days_old: dict[str, float]) -> None:
    now = datetime.now(UTC)
    lines = [_line(now - timedelta(days=d), tag) for tag, d in days_old.items()]
    (persona / _LOG).write_text("\n".join(lines) + "\n")


def _tags(persona: Path) -> set[str]:
    return {
        json.loads(line)["tag"]
        for line in (persona / _LOG).read_text().splitlines()
        if line.strip()
    }


def _stub_labeling(monkeypatch: pytest.MonkeyPatch, *, pause: bool = False, raises: bool = False):
    calls: list[str] = []

    def _label(store, **kwargs):
        calls.append("label")
        if raises:
            raise RuntimeError("judge blew up")
        if pause:
            kwargs["progress_out"]["paused"] = True
        return 0

    monkeypatch.setattr(rj_mod, "label_calibration_sample", _label)
    return calls


def _window() -> float:
    return float(
        tunables.get_tunable("calibration.retention_window_days", CALIBRATION_LOG_RETENTION_WINDOW_DAYS)
    )


def test_the_tick_removes_old_records_and_keeps_newer_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_labeling(monkeypatch)
    w = _window()
    _seed_log(tmp_path, {"old": w + 5, "older": w + 40, "recent": w - 1, "today": 0.0})

    assert _run_calibration_tick(tmp_path) is True

    assert _tags(tmp_path) == {"recent", "today"}


def test_the_prune_runs_after_the_calibration_log_and_before_the_labeling_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    real_cal_prune = MemoryStore.prune_calibration_log
    real_prune = rd.prune

    def _cal_prune(self, *a, **k):
        order.append("calibration_prune")
        return real_cal_prune(self, *a, **k)

    def _prune(path, **k):
        order.append("diagnostics_prune")
        return real_prune(path, **k)

    monkeypatch.setattr(MemoryStore, "prune_calibration_log", _cal_prune)
    monkeypatch.setattr(rd, "prune", _prune)
    calls = _stub_labeling(monkeypatch)
    monkeypatch.setattr(
        rj_mod,
        "label_calibration_sample",
        lambda *a, **k: order.append("label") or calls.append("label") or 0,
    )

    _run_calibration_tick(tmp_path)

    assert order == ["calibration_prune", "diagnostics_prune", "label"]


def test_the_prune_still_runs_when_the_labeling_step_pauses_and_the_tick_reports_paused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_labeling(monkeypatch, pause=True)
    w = _window()
    _seed_log(tmp_path, {"old": w + 5, "today": 0.0})

    assert _run_calibration_tick(tmp_path) is None, "paused: the cadence must not advance"

    assert _tags(tmp_path) == {"today"}


def test_the_prune_still_runs_when_the_labeling_step_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_labeling(monkeypatch, raises=True)
    w = _window()
    _seed_log(tmp_path, {"old": w + 5, "today": 0.0})

    assert _run_calibration_tick(tmp_path) is True

    assert _tags(tmp_path) == {"today"}


@pytest.mark.parametrize("breakage", ["prune_raises", "path_raises"])
def test_a_prune_failure_is_caught_and_the_rest_of_the_tick_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, breakage: str
) -> None:
    labeling = _stub_labeling(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated diagnostics failure")

    monkeypatch.setattr(rd, "prune" if breakage == "prune_raises" else "diagnostics_path", _boom)

    assert _run_calibration_tick(tmp_path) is True

    assert labeling == ["label"], "the labeling step still ran"


def test_a_tick_that_defers_for_a_busy_session_does_not_prune(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.ingest.buffer import ingest_turn

    _stub_labeling(monkeypatch)
    w = _window()
    _seed_log(tmp_path, {"old": w + 5, "today": 0.0})
    ingest_turn(tmp_path, {"session_id": "sess_busy", "speaker": "user", "text": "hi"})

    assert _run_calibration_tick(tmp_path, is_session_busy=lambda s: True) is False

    assert _tags(tmp_path) == {"old", "today"}


@pytest.mark.slow(reason='1.1 s of CPU in its call phase on a 4-core dev VM, 1.1 s wall (serial run for #540, J12)')
def test_the_window_is_the_calibration_retention_tunable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R7-T1 (f): a record 2.5 days old is KEPT under the default window (3 days) and
    REMOVED when the tunable is 2 days, so a prune that ignored the tunable (a constant
    window) fails one of the two runs."""
    _stub_labeling(monkeypatch)
    assert _window() == pytest.approx(3.0), "the default this test's first half relies on"
    _seed_log(tmp_path, {"two_and_a_half_days": 2.5, "one_day": 1.0})
    _run_calibration_tick(tmp_path)
    assert _tags(tmp_path) == {"two_and_a_half_days", "one_day"}

    real = tunables.get_tunable
    monkeypatch.setattr(
        tunables,
        "get_tunable",
        lambda key, default: 2.0 if key == "calibration.retention_window_days" else real(key, default),
    )
    _run_calibration_tick(tmp_path)

    assert _tags(tmp_path) == {"one_day"}


def _prune_call_sites(root: Path) -> set[str]:
    """Files under `root` that call the diagnostics module's `prune` by any import
    spelling: `recall_diagnostics.prune(...)`, an alias of the module, or `prune`
    imported by name (with or without `as`)."""
    sites: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name == "recall_diagnostics.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: set[str] = set()
        functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "recall_diagnostics":
                        modules.add(alias.asname or alias.name)
                    if node.module and node.module.endswith("recall_diagnostics") and alias.name == "prune":
                        functions.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("recall_diagnostics") and alias.asname:
                        modules.add(alias.asname)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "prune":
                if isinstance(func.value, ast.Name) and func.value.id in modules:
                    sites.add(path.relative_to(root).as_posix())
            elif isinstance(func, ast.Name) and func.id in functions:
                sites.add(path.relative_to(root).as_posix())
    return sites


@pytest.mark.slow(reason='1.7 s of CPU in its call phase on a 4-core dev VM, 1.7 s wall (serial run for #540, J12)')
def test_the_prune_has_no_call_site_outside_the_supervisor() -> None:
    """The diagnostics prune is a tick job, never a recall-path call (the calibration
    prune's own rule: off the hot path, I6). The scan sees every import spelling."""
    root = Path(rd.__file__).resolve().parents[1]
    assert _prune_call_sites(root) == {"bridge/supervisor.py"}


def test_the_call_site_scan_sees_aliased_and_from_imported_calls(tmp_path: Path) -> None:
    """The scan itself can fail: planted calls in three spellings are all found."""
    (tmp_path / "a.py").write_text("from brain.memory import recall_diagnostics as rd\nrd.prune(1)\n")
    (tmp_path / "b.py").write_text("from brain.memory.recall_diagnostics import prune as p\np(1)\n")
    (tmp_path / "c.py").write_text("from brain.memory.recall_diagnostics import prune\nprune(1)\n")
    (tmp_path / "d.py").write_text("def prune(x):\n    return x\nprune(1)\n")
    assert _prune_call_sites(tmp_path) == {"a.py", "b.py", "c.py"}
