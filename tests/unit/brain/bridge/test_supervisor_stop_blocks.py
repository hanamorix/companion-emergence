"""Nothing new starts once a stop is requested (C1, D2).

``run_folded`` read ``stop_event`` only at the top of its loop and in the inter-tick wait, so a stop requested during
one iteration still ran every later non-gated block of that iteration (soul review, log rotation, vocab repair, pending
reconcile, voice reflection, maker, notes, kindled link), each starting new work and advancing its cadence. These tests
run the real loop with every block due; the work just before block k requests the stop, and block k must not run and
its cadence file must stay byte-identical.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import brain.bridge.provider as provider_mod
import brain.bridge.supervisor as supervisor_mod
from brain.bridge import cli_throttle
from brain.bridge.events import EventBus
from brain.paths import cadence_state_path

# (block, work function name in supervisor.py, cadence file or None when the block has no supervisor-owned cadence)
_BLOCKS = [
    ("soul_review", "_run_soul_review_tick", "soul_review_state.json"),
    ("log_rotation", "_run_log_rotation_tick", "log_rotation_cadence.json"),
    ("vocab_repair", "_run_vocab_repair_tick", "vocab_repair_cadence.json"),
    ("pending_reconcile", "_run_pending_reconcile_tick", "pending_reconcile_cadence.json"),
    ("voice_reflection", "_run_voice_reflection_tick", "voice_reflection_cadence.json"),
    ("maker", "_maybe_run_maker_tick", None),
    ("notes", "_maybe_run_notes_tick", None),
    ("kindled_link", "_maybe_run_kindled_link_tick", None),
]


@pytest.fixture(autouse=True)
def _clean_throttle():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


def _kwargs(persona_dir: Path, stop_event: threading.Event) -> dict:
    return {
        "stop_event": stop_event,
        "persona_dir": persona_dir,
        "provider": provider_mod.FakeProvider(),
        "event_bus": EventBus(),
        "tick_interval_s": 0.01,
        "heartbeat_interval_s": None,
        "soul_review_interval_s": 3600.0,
        "finalize_interval_s": None,
        "log_rotation_interval_s": 3600.0,
        "initiate_review_interval_s": None,
        "voice_reflection_interval_s": 3600.0,
        "self_model_interval_s": None,
        "compaction_interval_s": None,
        "calibration_interval_s": None,
        "interest_sweep_interval_s": None,
        "judge_selftune_interval_s": None,
        "clustering_interval_s": None,
        "vocab_repair_interval_s": 3600.0,
        "pending_reconcile_interval_s": 3600.0,
        "maker_enabled": True,
        "notes_enabled": True,
        "kindled_link_enabled": True,
    }


def _snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


@pytest.mark.parametrize("index", range(len(_BLOCKS)), ids=[b[0] for b in _BLOCKS])
def test_a_stop_requested_before_a_block_skips_it(tmp_path, monkeypatch, index):
    persona_dir = tmp_path / "persona"
    (persona_dir / "active_conversations").mkdir(parents=True)
    stop_event = threading.Event()
    in_loop = {"on": False}
    called: list[str] = []
    name, _, cadence_file = _BLOCKS[index]

    def _requests_stop_if_previous(block: str) -> None:
        # The work just before the block under test requests the stop.
        previous = _BLOCKS[index - 1][0] if index > 0 else "central_pass"
        if block == previous:
            stop_event.set()

    def _central_pass(*_a, **_kw):
        in_loop["on"] = True
        _requests_stop_if_previous("central_pass")
        return []

    monkeypatch.setattr(supervisor_mod.central_cadence, "run_central_pass", _central_pass)
    monkeypatch.setattr(supervisor_mod, "build_tier_provider", lambda *_a, **_kw: provider_mod.FakeProvider())

    def _spy(block: str, result=None):
        def run(*_a, **_kw):
            if in_loop["on"]:  # vocab repair and pending reconcile also run once at startup
                called.append(block)
                _requests_stop_if_previous(block)
            return result

        return run

    for block, fn, _ in _BLOCKS:
        monkeypatch.setattr(supervisor_mod, fn, _spy(block, (0, 0) if block == "soul_review" else None))

    before = _snapshot(cadence_state_path(persona_dir, cadence_file)) if cadence_file else None
    t = threading.Thread(target=lambda: supervisor_mod.run_folded(**_kwargs(persona_dir, stop_event)), daemon=True)
    t.start()
    t.join(timeout=30)
    stop_event.set()
    t.join(timeout=10)
    assert not t.is_alive()

    assert name not in called, f"{name} ran after the stop: {called}"
    if index > 0:
        assert _BLOCKS[index - 1][0] in called  # the stop really came from the block before
    if cadence_file:
        assert _snapshot(cadence_state_path(persona_dir, cadence_file)) == before
