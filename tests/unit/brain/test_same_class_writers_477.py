"""#477: the lower-stakes writers of the same class.

* the voice-change accept in the bridge wrote `voice.md.tmp` (fixed name, no fsync, no retry);
* `felt_time.chat_log._rewrite_log` read, trimmed and rewrote the tick log through a fixed `.tmp` with no lock, next to a plain appender,
  so a row appended in between was lost;
* `engines.research_notes.append_session_notes` read the whole notes file and rewrote it with no lock and no retry, so two sessions
  finishing together dropped one block.
All three now go through the shared atomic text writer under the file's lock.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

import brain.felt_time.chat_log as chat_log
import brain.health.attempt_heal as heal
from brain.bridge.server import build_app
from brain.engines.research_notes import append_session_notes, notes_path
from brain.felt_time.chat_log import (
    CHAT_TURNS_LOG_FILENAME,
    append_chat_tick,
    load_recent_samples,
)
from brain.initiate.audit import append_audit_row, iter_initiate_audit_full
from brain.initiate.schemas import AuditRow


def _flaky_replace(monkeypatch, suffix: str, failures: int = 2):
    real = os.replace
    state = {"n": failures}

    def flaky(src, dst):
        if state["n"] > 0 and str(dst).endswith(suffix):
            state["n"] -= 1
            raise PermissionError(32, "sharing violation")
        return real(src, dst)

    monkeypatch.setattr(heal.os, "replace", flaky)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    return state


# --- research notes -------------------------------------------------------------------------------


def test_concurrent_note_appends_keep_every_block(tmp_path, monkeypatch):
    real_read = Path.read_text

    def slow_read(self, *a, **k):
        text = real_read(self, *a, **k)
        if self.suffix == ".md":
            time.sleep(0.05)
        return text

    monkeypatch.setattr(Path, "read_text", slow_read)
    threads = [
        threading.Thread(
            target=append_session_notes,
            args=(tmp_path, "abc", f"note-{i}"),
            kwargs={"now": datetime(2026, 7, 13, tzinfo=UTC)},
        )
        for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    monkeypatch.undo()
    text = notes_path(tmp_path, "abc").read_text(encoding="utf-8")
    assert all(f"note-{i}" in text for i in range(5))


def test_a_note_append_retries_a_transient_permission_error(tmp_path, monkeypatch):
    state = _flaky_replace(monkeypatch, "abc.md")
    append_session_notes(tmp_path, "abc", "kept", now=datetime(2026, 7, 13, tzinfo=UTC))
    assert state["n"] == 0
    assert "kept" in notes_path(tmp_path, "abc").read_text(encoding="utf-8")


# --- chat tick log --------------------------------------------------------------------------------


def _seed_log(persona: Path, old: int, recent: int) -> None:
    now = datetime.now(UTC)
    for i in range(old):
        append_chat_tick(persona, ts=now - timedelta(days=40, minutes=i), turns=1)
    for i in range(recent):
        append_chat_tick(persona, ts=now - timedelta(hours=i + 1), turns=2)


def test_a_tick_appended_while_the_log_is_being_trimmed_is_not_lost(tmp_path, monkeypatch):
    _seed_log(tmp_path, old=15, recent=5)
    real_iter = chat_log.iter_jsonl_skipping_corrupt
    appended = {"done": False}
    late = datetime.now(UTC) - timedelta(minutes=1)

    def iter_then_append(path):
        rows = list(real_iter(path))
        if not appended["done"]:
            appended["done"] = True
            append_chat_tick(tmp_path, ts=late, turns=7)
        return iter(rows)

    monkeypatch.setattr(chat_log, "iter_jsonl_skipping_corrupt", iter_then_append)
    load_recent_samples(tmp_path)
    monkeypatch.undo()
    text = (tmp_path / CHAT_TURNS_LOG_FILENAME).read_text(encoding="utf-8")
    assert '"turns": 7' in text
    assert text.count("\n") == 6  # 5 recent + the late one, the 15 old trimmed


def test_the_tick_log_trim_retries_a_transient_permission_error(tmp_path, monkeypatch):
    _seed_log(tmp_path, old=15, recent=5)
    state = _flaky_replace(monkeypatch, CHAT_TURNS_LOG_FILENAME)
    load_recent_samples(tmp_path)
    assert state["n"] == 0
    assert (tmp_path / CHAT_TURNS_LOG_FILENAME).read_text(encoding="utf-8").count("\n") == 5


# --- voice change accept --------------------------------------------------------------------------


def _seed_voice_edit(persona: Path, audit_id: str, old: str, new: str) -> None:
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts="2026-05-11T14:47:09+00:00",
        kind="voice_edit_proposal",
        subject="a small edit to my voice",
        tone_rendered="proposing a change",
        decision="send_quiet",
        decision_reasoning="seen three times",
        gate_check={"allowed": True, "reason": None},
        delivery={
            "delivered_at": "2026-05-11T14:47:09+00:00",
            "state_transitions": [{"to": "delivered", "at": "2026-05-11T14:47:09+00:00"}],
            "current_state": "delivered",
        },
        diff=f"- {old}\n+ {new}\n",
    )
    append_audit_row(persona, row)


def test_voice_accept_retries_a_transient_permission_error(tmp_path, monkeypatch):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "voice.md").write_text("line A\nold line\nline C\n")
    _seed_voice_edit(persona, "ia_ve_477", "old line", "new line")
    state = _flaky_replace(monkeypatch, "voice.md")
    app = build_app(persona_dir=persona, client_origin="tests")
    with TestClient(app) as client:
        r = client.post("/initiate/voice-edit/accept", json={"audit_id": "ia_ve_477", "with_edits": None})
    assert r.status_code == 200, r.text
    assert state["n"] == 0
    assert "new line" in (persona / "voice.md").read_text()
    assert not [p.name for p in persona.iterdir() if p.name.startswith("voice.md.")]
    row = next(r for r in iter_initiate_audit_full(persona) if r.audit_id == "ia_ve_477")
    assert row.delivery["current_state"] == "replied_explicit"


def test_the_tick_appender_waits_for_the_logs_lock(tmp_path):
    """Code review 3a: removing the lock from `append_chat_tick` was caught by no test. A trim holds the log's lock while it rewrites;
    an appender that does not wait for it appends to the file the trim is about to replace, and the row is lost."""
    from brain.utils.file_lock import file_lock

    log_path = tmp_path / CHAT_TURNS_LOG_FILENAME
    done = threading.Event()
    with file_lock(log_path):
        t = threading.Thread(
            target=lambda: (append_chat_tick(tmp_path, ts=datetime.now(UTC), turns=3), done.set()), daemon=True
        )
        t.start()
        assert not done.wait(timeout=0.4), "the append did not wait for the log's lock"
    assert done.wait(timeout=5)
    t.join(timeout=5)
    assert log_path.read_text(encoding="utf-8").count("\n") == 1
