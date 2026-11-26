"""#736: the self-model audit counters (self_model_audit.json) are read, changed and saved under the file's own lock.

They are bumped by the supervisor's self-model tick and by the reconcile tool during a chat turn; without a lock, one
increment that reads before the other saves is lost. The race is made deterministic: the first thread pauses between its
read and its save until the second has finished (or a short timeout passes, so a locked run cannot deadlock on it).
"""
from __future__ import annotations

import threading

from brain.self_model import resolve


def test_an_increment_whose_lock_does_not_come_drops_that_count_and_does_not_raise(tmp_path, monkeypatch, caplog):
    # A counter must never hang or fail a chat turn's reconcile: another holder keeps the counter's lock past the wait.
    from brain import dev_constants
    from brain.utils.file_lock import file_lock

    monkeypatch.setattr(dev_constants, "SELF_MODEL_LOCK_WAIT_S", 0.05)
    with file_lock(resolve._audit_path(tmp_path)):
        resolve.increment_gaps_surfaced(tmp_path)
    assert resolve.load_audit(tmp_path)["gaps_surfaced"] == 0
    assert "not counted" in caplog.text
    resolve.increment_gaps_surfaced(tmp_path)  # the lock is free again: counts as before
    assert resolve.load_audit(tmp_path)["gaps_surfaced"] == 1


def test_two_increments_that_interleave_lose_no_count(tmp_path, monkeypatch):
    real_load = resolve.load_audit
    first_loaded = threading.Event()
    second_done = threading.Event()
    first = {}

    def load_then_pause(persona_dir):
        audit = real_load(persona_dir)
        if threading.current_thread() is first.get("thread"):
            first_loaded.set()
            second_done.wait(0.5)  # unlocked: the second increment saves in here and is then overwritten
        return audit

    monkeypatch.setattr(resolve, "load_audit", load_then_pause)

    def second():
        resolve.increment_reconciles(tmp_path)
        second_done.set()

    a = threading.Thread(target=resolve.increment_reconciles, args=(tmp_path,))
    first["thread"] = a
    a.start()
    assert first_loaded.wait(5)
    b = threading.Thread(target=second)
    b.start()
    a.join(10)
    b.join(10)
    assert not a.is_alive() and not b.is_alive()
    assert real_load(tmp_path)["reconciles_called"] == 2
