"""TEMPORARY CI diagnostic (reverted right after one manual run). Not part of the PR."""
from __future__ import annotations

import collections
import sys
import threading
import time
import traceback
from unittest.mock import patch

import pytest

from tests.unit.brain.bridge.test_deploy_recalibration import _persona_dir, _run_folded_in_thread


def test_diag_first_pass_where_does_time_go(tmp_path) -> None:
    persona_dir = _persona_dir(tmp_path)
    stop = threading.Event()
    probes: list[float] = []
    t0 = time.monotonic()

    def _not_due(_pd):
        probes.append(time.monotonic() - t0)
        return False

    counts: collections.Counter = collections.Counter()
    with (
        patch("brain.bridge.supervisor._deploy_recalibration_due", side_effect=_not_due),
        patch("brain.bridge.supervisor._run_deploy_recalibration"),
    ):
        t = _run_folded_in_thread(persona_dir, stop)
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and len(probes) < 2:
            frame = sys._current_frames().get(t.ident)
            if frame is not None:
                stack = traceback.extract_stack(frame)[-7:]
                counts[" <- ".join(f"{s.filename.split('companion-emergence')[-1][-40:]}:{s.lineno}:{s.name}" for s in reversed(stack))] += 1
            time.sleep(0.05)
        stop.set()
        t.join(timeout=10.0)
    lines = [f"probes at: {[round(p, 2) for p in probes]}"] + [f"{n:4d} {k}" for k, n in counts.most_common(12)]
    pytest.fail("\n".join(lines))
