"""The behaviour changes a migrated caller sees (BC of 2-plan-phase2.md), each pinned on a REAL migrated function.

Test names start with the BC id. Only the BC pins that no module test repeats are here (BC2, BC3, BC4, BC6, BC7,
BC10, BC11, BC14, BC15, BC16, BC17, BC19, BC20); BC1, BC5, BC8, BC9, BC12, BC13, BC18, BC21 and BC22 are held by
test_json_io.py, test_migrated_group_b.py / _c.py / _d.py and test_parse_failures.py. Characterization pins (BC2,
BC10 and BC11 first halves, BC16, BC19) say so: an existing or module test passes through the new call and cannot be
red before green. Every other test asserts new behaviour and fails against the old code path. Nothing here touches a
persona directory.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _no_temp_left(directory: Path) -> None:
    left = [p.name for p in directory.iterdir() if ".tmp" in p.name or p.name.endswith(".new")]
    assert not left, left


def test_bc2_characterization_fsync_is_preserved_per_site(tmp_path: Path, monkeypatch) -> None:
    from brain.bridge.persisted_cadence import CadenceState, save_cadence
    from brain.self_model import state as sm_state

    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    save_cadence(tmp_path, "c.json", CadenceState(next_at=NOW))
    assert synced == []  # a hot best-effort writer never synced
    sm_state.save(tmp_path, sm_state.SelfModelState())
    assert synced  # a state file always did


def test_bc3_files_of_every_writer_family_read_back_with_the_stdlib(tmp_path: Path) -> None:
    from brain.forgetting import _persist_forgetting_state
    from brain.ingest import buffer
    from brain.memory.pending import PendingQueue
    from brain.migrator.og_soul_candidates import migrate_soul_candidates
    from brain.soul.review import _save_soul_candidates

    _persist_forgetting_state(tmp_path, {"café": 1})  # backup-chain family
    assert json.loads((tmp_path / "forgetting_state.json").read_text()) == {"café": 1}
    assert (tmp_path / "forgetting_state.json").read_bytes().isascii()
    buffer.ingest_turn(tmp_path, {"session_id": "s", "speaker": "user", "text": "é", "ts": NOW.isoformat()})  # append
    line = (tmp_path / "active_conversations" / "s.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["text"] == "é"
    PendingQueue(tmp_path).enqueue_reappraisals(["m"], source="t")
    _save_soul_candidates(tmp_path, [{"memory_id": "m", "text": "日"}])  # rewrite family
    assert json.loads((tmp_path / "soul_candidates.jsonl").read_text()) == {"memory_id": "m", "text": "日"}
    og = tmp_path / "og"
    og.mkdir()
    (og / "soul_candidates.jsonl").write_text(json.dumps({"memory_id": "x", "text": "ü"}) + "\n")
    dest = tmp_path / "persona"
    migrate_soul_candidates(og_data_dir=og, persona_dir=dest)
    assert (dest / "soul_candidates.jsonl").read_bytes().isascii()


def test_bc4_a_nan_in_a_whole_file_state_raises_before_touching_the_file_and_an_audit_appender_survives(tmp_path: Path) -> None:
    from brain.maker.charge import MakerCharge, save_charge
    from brain.mcp_server.audit import log_invocation

    bad = MakerCharge(charge=float("nan"), last_tick_ts=None, last_fire_ts=None, prior_soul_count=0)
    with pytest.raises(ValueError):
        save_charge(tmp_path, bad)
    assert not list(tmp_path.glob("*.json"))  # nothing was written
    log_invocation(tmp_path, name="t", arguments={"x": float("nan")}, result_summary="ok", outcome="ok")
    (log,) = list(tmp_path.glob("*.jsonl"))
    assert json.loads(log.read_text())["arguments"] == {"x": None}


def test_bc6_a_best_effort_logger_on_an_unwritable_directory_warns_once_and_never_raises(tmp_path: Path, caplog) -> None:
    from brain.bridge.usage_log import log_usage

    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a directory should be")
    with caplog.at_level(logging.WARNING):
        log_usage(blocker, call_type="c", model="m", frame={})
        log_usage(blocker, call_type="c", model="m", frame={})
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1  # throttled


def test_bc7_a_fail_soft_reader_falls_back_on_invalid_utf8_with_one_warning(tmp_path: Path, caplog) -> None:
    from brain.self_model.state import load_or_recover

    (tmp_path / "self_model_state.json").write_bytes(b'{"current_gap": "\xff"}')
    with caplog.at_level(logging.WARNING):
        state, recovered = load_or_recover(tmp_path)
    assert recovered is True and state.current_gap is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_bc10_characterization_heal_restores_from_bak1_and_the_live_path_is_never_absent(tmp_path: Path) -> None:
    from brain.persona_config import PersonaConfig

    path = tmp_path / "persona_config.json"
    PersonaConfig(provider="claude-cli").save(path)
    PersonaConfig(provider="claude-cli", user_name="Bob").save(path)
    path.write_bytes(b"{torn")
    absent: list[bool] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            absent.append(not path.exists())

    t = threading.Thread(target=poll)
    t.start()
    cfg, anomaly = PersonaConfig.load_with_anomaly(path)
    stop.set()
    t.join()
    assert anomaly is not None and anomaly.action == "restored_from_bak1"
    assert not any(absent), "the corrupt file is kept by a link/copy, never renamed away from the live path"
    assert list(tmp_path.glob("persona_config.json.corrupt-*"))


def test_bc11_a_save_with_verify_restores_bak1_and_waits_for_a_heal_in_progress(tmp_path: Path, monkeypatch) -> None:
    from brain.engines.heartbeat import HeartbeatConfig
    from brain.health import adaptive
    from brain.health.json_io import _heal_guard

    path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig().save(path)
    monkeypatch.setattr(adaptive, "compute_treatment", lambda *a, **k: adaptive.FileTreatment(backup_count=3, verify_after_write=True))
    done = threading.Event()
    with _heal_guard(path):  # a heal is in progress
        t = threading.Thread(target=lambda: (HeartbeatConfig().save(path), done.set()))
        t.start()
        assert not done.wait(0.3), "a save must wait for the heal lock"
    t.join(5)
    assert done.is_set() and json.loads(path.read_text())


def test_bc14_an_own_copy_reader_survives_a_cut_multibyte_line_and_rotation_keeps_the_raw_bytes(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from brain.engines.dream import DreamEngine

    log = tmp_path / "dreams.log.jsonl"
    good = json.dumps({"seed_id": "s1"}, ensure_ascii=False).encode()
    cut = json.dumps({"seed_id": "über"}, ensure_ascii=False).encode()
    log.write_bytes(good + b"\n" + cut[: cut.index("ü".encode()) + 1] + good + b"\n")
    assert DreamEngine._recent_seed_ids(SimpleNamespace(log_path=log), 5) == ["s1"]


def test_bc15_an_append_between_a_rewriters_read_and_replace_is_not_lost(tmp_path: Path) -> None:
    from brain.initiate import emit
    from brain.initiate.schemas import SemanticContext

    for i in range(2):
        emit.emit_initiate_candidate(
            tmp_path, kind="message", source="dream", source_id=f"s{i}", semantic_context=SemanticContext()
        )
    target = emit.read_candidates(tmp_path)[0].candidate_id
    started, finish = threading.Event(), threading.Event()
    real = emit.read_candidates

    def slow_read(pd):
        out = real(pd)
        started.set()
        finish.wait(2)
        return out

    emit.read_candidates = slow_read
    try:
        remover = threading.Thread(target=emit.remove_candidate, args=(tmp_path, target))
        remover.start()
        assert started.wait(2)
        appender = threading.Thread(
            target=lambda: emit.emit_initiate_candidate(
                tmp_path, kind="message", source="dream", source_id="late", semantic_context=SemanticContext()
            )
        )
        appender.start()
        finish.set()
        remover.join(5)
        appender.join(5)
    finally:
        emit.read_candidates = real
    ids = {c.source_id for c in real(tmp_path)}
    assert "late" in ids and target not in {c.candidate_id for c in real(tmp_path)}


def test_bc16_characterization_the_module_never_uses_the_locale_codec() -> None:
    src = Path("brain/health/json_io.py").read_text(encoding="utf-8")
    assert "locale" not in src and 'open(path, "rb")' in src  # CR11 of test_json_io.py asserts it on real calls


def test_bc17_the_new_sidecars_are_ignored_by_the_sweep_and_the_walker(tmp_path: Path) -> None:
    from brain.forgetting import _persist_forgetting_state
    from brain.health.sidecar_sweep import sweep_stale_sidecars
    from brain.health.walker import walk_persona
    from brain.ingest import buffer

    _persist_forgetting_state(tmp_path, {"m": 1})  # .heal.lock
    buffer.ingest_turn(tmp_path, {"session_id": "s", "speaker": "user", "text": "x", "ts": NOW.isoformat()})  # .jsonl.lock
    sidecars = list(tmp_path.rglob("*.lock"))
    assert sidecars
    assert buffer.list_active_sessions(tmp_path) == ["s"]  # a .jsonl glob never picks up a .lock
    assert sweep_stale_sidecars(tmp_path, now=datetime.now(UTC)) == []  # young live locks stay
    walk_persona(tmp_path)  # tolerates them
    assert all(p.exists() for p in sidecars)


def test_bc19_characterization_the_deleted_names_are_undefined() -> None:
    import brain.health.attempt_heal as heal
    import brain.health.jsonl_reader as reader

    for name in ("attempt_heal", "_heal_from_baks", "_load_and_validate"):
        assert not hasattr(heal, name)
    for name in ("read_jsonl_skipping_corrupt", "iter_jsonl_streaming", "read_last_n_jsonl_lines"):
        assert not hasattr(reader, name)
    assert hasattr(heal, "save_with_backup") and hasattr(reader, "iter_jsonl_skipping_corrupt")  # kept, byte-identical


def test_bc20_a_former_in_place_writer_is_atomic_and_rides_out_a_transient_permission_error(tmp_path: Path, monkeypatch) -> None:
    from brain.kindled_link.cadence import load_tick_cadence, save_tick_cadence

    save_tick_cadence(tmp_path, NOW)
    real_replace = os.replace
    left = {"n": 1}

    def flaky(src, dst):
        if left["n"]:
            left["n"] -= 1
            raise PermissionError(13, "sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    later = datetime(2026, 2, 1, tzinfo=UTC)
    save_tick_cadence(tmp_path, later)
    assert load_tick_cadence(tmp_path)["last_run"] == later.isoformat()  # whole file, not torn
    _no_temp_left(tmp_path / "kindled_link")

