"""Rollback canary (#286 §7): every persisted-state reader keeps the fields it
knows and ignores fields a newer brain added. Each test writes through the
module's real writer (or its real on-disk shape), adds one unknown field, and
reads back through the real reader. See brain/state_compat.py for the rule."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

FUTURE = {"_future_field": "written by a newer brain"}


def _add_future_field(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(FUTURE)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bridge_state(tmp_path):
    from brain.bridge import state_file
    from brain.bridge.state_file import BridgeState

    st = BridgeState(persona="p", pid=123, port=4567, started_at="2026-09-26T00:00:00Z",
                     stopped_at=None, shutdown_clean=False, client_origin="tests",
                     auth_token="tok", drain_errors=0)
    state_file.write(tmp_path, st)
    _add_future_field(tmp_path / state_file.STATE_FILENAME)
    assert state_file.read(tmp_path) == st


def test_current_read(tmp_path):
    from brain.attunement import store
    from brain.attunement.schemas import CurrentRead

    cr = CurrentRead(ts="2026-09-26T00:00:00Z", source_turn_id="t1", tone_label="warm",
                     tone_justification="j", cadence_label="slow", cadence_justification="j",
                     mood_valence=0.2, mood_intensity=0.4, predicted_arc_shape="steady",
                     schema_version="0.0.29")
    store.write_current_read(tmp_path, cr)
    _add_future_field(store._current_read_path(tmp_path))
    assert store.read_current_read(tmp_path) == cr


def test_learned_pattern(tmp_path):
    from brain.attunement import store
    from brain.attunement.schemas import LearnedPattern

    lp = LearnedPattern(id="id-1", category="tone", canonical_key="k", description="d",
                        evidence_count=3, maturity="forming", first_seen_at="2026-09-01T00:00:00Z",
                        last_confirmed_at="2026-09-20T00:00:00Z", last_addressed_at=None,
                        crystallised_at=None, falsified_at=None, examples=["e"],
                        schema_version="0.0.29")
    path = store._learned_patterns_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**asdict(lp), **FUTURE}) + "\n", encoding="utf-8")
    assert store.read_learned_patterns(tmp_path) == [lp]


def test_attunement_backfill_state(tmp_path):
    from brain.attunement import backfill
    from brain.attunement.schemas import BackfillState

    bs = BackfillState(started_at="2026-09-26T00:00:00Z", total_windows=10, sampled_windows=5,
                       processed_windows=2, patterns_emitted=1, status="running",
                       last_cursor="c", schema_version="0.0.29")
    backfill._save_state(tmp_path, bs)
    _add_future_field(backfill._state_path(tmp_path))
    assert backfill._load_state(tmp_path) == bs


def test_felt_time_anchors_and_pressure(tmp_path):
    from brain.felt_time.state import (STATE_FILENAME, Anchor, FeltTimeState,
                                       PressureCounters, load_or_recover, persist)

    persist(FeltTimeState.cold_start(), tmp_path)
    path = tmp_path / STATE_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    anchor = Anchor(type="dream", ts="2026-09-26T00:00:00+00:00", label="a dream",
                    source_ref="dreams.log.jsonl:1")
    arc = Anchor(type="arc", ts="2026-09-25T00:00:00+00:00", label="an arc",
                 source_ref="growth.log.jsonl:2", event_type="arc_opened")
    counters = PressureCounters(heartbeats=2, chat_turns=3, reflex_firings=1, wall_clock_s=9.0)
    data["anchors"] = {"dream": {**asdict(anchor), **FUTURE}}
    data["arc_anchors"] = [{**asdict(arc), **FUTURE}]
    data["pressure"] = {**asdict(counters), **FUTURE}
    data["horizon_pressure"] = {"day": {"counters": {**asdict(counters), **FUTURE},
                                        "prev_counters": {**asdict(counters), **FUTURE},
                                        "period_start_ts": "2026-09-26T00:00:00+00:00"}}
    path.write_text(json.dumps(data), encoding="utf-8")

    state, _ = load_or_recover(tmp_path)
    assert state.anchors["dream"] == anchor
    assert state.arc_anchors == [arc]
    assert state.pressure == counters
    assert state.horizon_pressure["day"].counters == counters
    assert state.horizon_pressure["day"].prev_counters == counters
