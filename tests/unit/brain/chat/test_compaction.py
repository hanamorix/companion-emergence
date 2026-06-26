"""Conformance tests for timed-conversation-compaction.

Maps to changes/timed-conversation-compaction/1.5-criteria.md:
  C1a  byte-stable prefix between compactions (+ backstop fires once)
  C2   summary block byte-stability across turns
  C3   summary renders as a head system msg; wrong-order hoisted
  C4   fold path: archive-before-overwrite, buffer shape, gen+1
  C5   tool append path: existing summary kept verbatim
  C6   atomicity: archive write fails -> buffer untouched
  C7   cross-readers never re-ingest / mis-count a summary row
  C8   cursor guard: ts<=cursor removable; None cursor -> no-op
  C12  stale lock reaped; fresh lock skips
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.chat import ChatMessage
from brain.chat.budget import apply_budget
from brain.chat.compaction import compact_conversation
from brain.chat.engine import _buffer_turns_to_messages
from brain.ingest.buffer import (
    _compacting_lock_path,
    acquire_compaction_lock,
    ingest_turn,
    read_archive,
    read_session,
    write_cursor,
)


class _StubProvider:
    """Deterministic provider stub. compaction/budget only call generate()."""

    def __init__(self, response: str = "FADED") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, *, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _ExplodingProvider:
    def generate(self, *, prompt: str) -> str:  # noqa: D401
        raise RuntimeError("provider down")


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _seed(persona_dir: Path, sid: str, n_turns: int, *, base: datetime) -> list[str]:
    """Write n_turns user/assistant rows, oldest first, 1 minute apart.
    Returns the list of ts written (for cursor positioning)."""
    tss: list[str] = []
    for i in range(n_turns):
        ts = _iso(base + timedelta(minutes=i))
        tss.append(ts)
        speaker = "user" if i % 2 == 0 else "assistant"
        ingest_turn(persona_dir, {"session_id": sid, "speaker": speaker,
                                  "text": f"turn {i}", "ts": ts})
    return tss


def _summary_rows(turns: list[dict]) -> list[dict]:
    return [t for t in turns if t.get("speaker") == "summary"]


# --------------------------------------------------------------------------- C8
def test_c8_none_cursor_is_noop(tmp_path: Path) -> None:
    sid = "sess_c8none"
    base = datetime.now(UTC) - timedelta(hours=10)
    _seed(tmp_path, sid, 60, base=base)
    # No cursor written -> nothing provably extracted -> no-op.
    res = compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                               fold_existing_summary=True, provider=_StubProvider())
    assert res.compacted is False
    assert res.reason == "cursor_none"
    assert _summary_rows(read_session(tmp_path, sid)) == []


def test_c8_only_extracted_turns_removed(tmp_path: Path) -> None:
    sid = "sess_c8cur"
    base = datetime.now(UTC) - timedelta(hours=10)
    tss = _seed(tmp_path, sid, 60, base=base)
    # Cursor at turn 20 -> only turns 0..20 are "extracted" and removable; the
    # min_keep_tail also protects the last 40, so removable = turns 0..19 minus
    # any in the protected tail (none here, 60 turns, tail is 20..59).
    write_cursor(tmp_path, sid, tss[20])
    res = compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                               fold_existing_summary=True, provider=_StubProvider(),
                               min_keep_tail=40)
    assert res.compacted is True
    after = read_session(tmp_path, sid)
    assert after[0]["speaker"] == "summary"
    # The C8 invariant: nothing REMOVED (archived) may be past the cursor — an
    # un-extracted turn is never compacted away. (Retained turns may include
    # at-or-before-cursor turns that fall in the protected min_keep_tail window.)
    cur = datetime.fromisoformat(tss[20])
    archived_raw = [t for t in read_archive(tmp_path, sid) if t.get("speaker") != "summary"]
    assert archived_raw, "expected some extracted turns to be archived"
    for t in archived_raw:
        assert datetime.fromisoformat(t["ts"]) <= cur


def test_c8_nothing_aged_is_noop(tmp_path: Path) -> None:
    sid = "sess_c8young"
    base = datetime.now(UTC) - timedelta(minutes=5)
    tss = _seed(tmp_path, sid, 10, base=base)
    write_cursor(tmp_path, sid, tss[-1])
    # older_than=24h but all turns are 5 min old -> nothing aged -> no-op.
    res = compact_conversation(tmp_path, sid, older_than=timedelta(hours=24),
                               fold_existing_summary=True, provider=_StubProvider())
    assert res.compacted is False
    assert res.reason == "nothing_aged"


# --------------------------------------------------------------------------- C4
def test_c4_fold_archives_before_overwrite_and_shape(tmp_path: Path) -> None:
    sid = "sess_c4"
    base = datetime.now(UTC) - timedelta(hours=10)
    tss = _seed(tmp_path, sid, 100, base=base)
    write_cursor(tmp_path, sid, tss[-1])  # all extracted
    res = compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                               fold_existing_summary=True, provider=_StubProvider("S1"),
                               min_keep_tail=40)
    assert res.compacted is True
    after = read_session(tmp_path, sid)
    summaries = _summary_rows(after)
    assert len(summaries) == 1
    assert after[0]["speaker"] == "summary"          # at the head
    assert summaries[0]["compaction"]["gen"] == 1
    retained_raw = [t for t in after if t.get("speaker") != "summary"]
    assert len(retained_raw) == 40                    # min_keep_tail preserved
    # Archive holds exactly the removed raw turns (100 - 40 = 60).
    archived = read_archive(tmp_path, sid)
    archived_raw = [t for t in archived if t.get("speaker") != "summary"]
    assert len(archived_raw) == 60
    assert res.compacted_n == 60


def test_c4_second_fold_archives_old_summary_and_bumps_gen(tmp_path: Path) -> None:
    sid = "sess_c4b"
    base = datetime.now(UTC) - timedelta(hours=20)
    tss = _seed(tmp_path, sid, 100, base=base)
    write_cursor(tmp_path, sid, tss[-1])
    compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                         fold_existing_summary=True, provider=_StubProvider("GEN1"),
                         min_keep_tail=40)
    # Add more aged turns and re-fold.
    base2 = datetime.now(UTC) - timedelta(hours=5)
    tss2 = _seed(tmp_path, sid, 50, base=base2)
    write_cursor(tmp_path, sid, tss2[-1])
    res2 = compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                                fold_existing_summary=True, provider=_StubProvider("GEN2"),
                                min_keep_tail=40)
    assert res2.compacted is True
    after = read_session(tmp_path, sid)
    assert _summary_rows(after)[0]["compaction"]["gen"] == 2
    # The gen-1 summary is now in the archive (faded, not lost).
    archived_summaries = [t for t in read_archive(tmp_path, sid)
                          if t.get("speaker") == "summary"]
    assert any(s.get("text") == "GEN1" for s in archived_summaries)


# --------------------------------------------------------------------------- C5
def test_c5_tool_append_keeps_existing_summary_verbatim(tmp_path: Path) -> None:
    sid = "sess_c5"
    base = datetime.now(UTC) - timedelta(hours=20)
    tss = _seed(tmp_path, sid, 100, base=base)
    write_cursor(tmp_path, sid, tss[-1])
    # First, create a summary via a fold.
    compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                         fold_existing_summary=True, provider=_StubProvider("ORIGINAL"),
                         min_keep_tail=40)
    # More aged turns, then an APPEND (fold=False, the tool path).
    base2 = datetime.now(UTC) - timedelta(hours=5)
    tss2 = _seed(tmp_path, sid, 50, base=base2)
    write_cursor(tmp_path, sid, tss2[-1])
    res = compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                               fold_existing_summary=False, provider=_StubProvider("APPENDED"),
                               min_keep_tail=40)
    assert res.compacted is True
    summary = _summary_rows(read_session(tmp_path, sid))[0]
    # Existing text kept verbatim, new part appended after it.
    assert summary["text"].startswith("ORIGINAL")
    assert "APPENDED" in summary["text"]


# --------------------------------------------------------------------------- C6
def test_c6_archive_failure_leaves_buffer_untouched(tmp_path: Path, monkeypatch) -> None:
    sid = "sess_c6"
    base = datetime.now(UTC) - timedelta(hours=10)
    tss = _seed(tmp_path, sid, 100, base=base)
    write_cursor(tmp_path, sid, tss[-1])
    before = Path(tmp_path / "active_conversations" / f"{sid}.jsonl").read_bytes()

    import brain.chat.compaction as comp

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(comp, "append_archive", _boom)
    res = compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                               fold_existing_summary=True, provider=_StubProvider())
    assert res.compacted is False
    assert res.reason == "archive_failed"
    after = Path(tmp_path / "active_conversations" / f"{sid}.jsonl").read_bytes()
    assert after == before  # buffer byte-identical: no data loss, no torn buffer


# --------------------------------------------------------------------------- C3
def test_c3_summary_renders_as_head_system_message(tmp_path: Path) -> None:
    turns = [
        {"speaker": "summary", "text": "the gist", "ts": _iso(datetime.now(UTC))},
        {"speaker": "user", "text": "hello", "ts": _iso(datetime.now(UTC))},
        {"speaker": "assistant", "text": "hi", "ts": _iso(datetime.now(UTC))},
    ]
    msgs = _buffer_turns_to_messages(tmp_path, turns)
    assert msgs[0].role == "system"
    assert "the gist" in msgs[0].content_text()
    assert msgs[1].role == "user"
    assert msgs[2].role == "assistant"


def test_c3_wrong_order_summary_hoisted_to_head(tmp_path: Path) -> None:
    # A torn/legacy write puts the summary AFTER raw turns; it must still be
    # hoisted to the head, never emitted mid-history.
    turns = [
        {"speaker": "user", "text": "hello", "ts": _iso(datetime.now(UTC))},
        {"speaker": "assistant", "text": "hi", "ts": _iso(datetime.now(UTC))},
        {"speaker": "summary", "text": "the gist", "ts": _iso(datetime.now(UTC))},
    ]
    msgs = _buffer_turns_to_messages(tmp_path, turns)
    assert msgs[0].role == "system"
    assert "the gist" in msgs[0].content_text()
    # Exactly one system message (no mid-history system row).
    assert sum(1 for m in msgs if m.role == "system") == 1


# --------------------------------------------------------------------------- C2
def test_c2_summary_byte_stable_across_turns(tmp_path: Path) -> None:
    sid = "sess_c2"
    base = datetime.now(UTC) - timedelta(hours=10)
    tss = _seed(tmp_path, sid, 100, base=base)
    write_cursor(tmp_path, sid, tss[-1])
    compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                         fold_existing_summary=True, provider=_StubProvider("STABLE"),
                         min_keep_tail=40)
    renders = []
    for i in range(3):
        # Append a fresh turn each "turn" — no compaction event between.
        ingest_turn(tmp_path, {"session_id": sid, "speaker": "user",
                               "text": f"new {i}", "ts": _iso(datetime.now(UTC))})
        msgs = _buffer_turns_to_messages(tmp_path, read_session(tmp_path, sid))
        renders.append(msgs[0].content_text())  # the summary system msg
    assert renders[0] == renders[1] == renders[2]


# -------------------------------------------------------------------------- C1a
def test_c1a_prefix_byte_stable_between_compactions(tmp_path: Path) -> None:
    """Steady state: with the window deleted, the history prefix only grows at
    the tail — prefix[N] is a byte-exact prefix of prefix[N+1]."""
    sid = "sess_c1a"
    base = datetime.now(UTC) - timedelta(hours=10)
    tss = _seed(tmp_path, sid, 100, base=base)
    write_cursor(tmp_path, sid, tss[-1])
    compact_conversation(tmp_path, sid, older_than=timedelta(hours=1),
                         fold_existing_summary=True, provider=_StubProvider("STABLE"),
                         min_keep_tail=40)

    def assemble_prefix() -> str:
        history = _buffer_turns_to_messages(tmp_path, read_session(tmp_path, sid))
        msgs = [ChatMessage(role="system", content="SYS"), *history]
        # apply_budget is a passthrough while under cap (steady state); pass the
        # buffer context so its signature matches production.
        msgs = apply_budget(msgs, max_tokens=10_000_000, preserve_tail_msgs=40,
                            provider=_StubProvider(), persona_dir=tmp_path, session_id=sid)
        return "\n".join(m.content_text() for m in msgs)

    prefixes = []
    for i in range(3):
        prefixes.append(assemble_prefix())
        ingest_turn(tmp_path, {"session_id": sid, "speaker": "user",
                               "text": f"append {i}", "ts": _iso(datetime.now(UTC))})
    assert prefixes[1].startswith(prefixes[0])
    assert prefixes[2].startswith(prefixes[1])


def test_c1a_over_cap_backstop_fires_once_not_per_turn(tmp_path: Path) -> None:
    """Over-cap path (the criterion's required case): apply_budget must NOT insert
    a fresh LLM summary into the prefix every turn (the old per-turn re-summary
    defect). It fires the persisted compaction ONCE (the accepted single cache
    write), the buffer shrinks under cap, and subsequent turns are passthrough —
    so provider.generate is called once across many over-cap turns, and the
    current-turn in-prompt floor is the deterministic note, not an LLM string."""
    sid = "sess_c1a_overcap"
    base = datetime.now(UTC) - timedelta(hours=10)
    # 10 big turns (~4000 chars each) → well over a small cap; all extracted.
    tss: list[str] = []
    for i in range(10):
        ts = _iso(base + timedelta(minutes=i))
        tss.append(ts)
        ingest_turn(tmp_path, {"session_id": sid,
                               "speaker": "user" if i % 2 == 0 else "assistant",
                               "text": "x" * 4000, "ts": ts})
    write_cursor(tmp_path, sid, tss[-1])  # all extracted
    provider = _StubProvider("SUM")

    def one_turn() -> list[ChatMessage]:
        history = _buffer_turns_to_messages(tmp_path, read_session(tmp_path, sid))
        msgs = [ChatMessage(role="system", content="SYS"), *history]
        # Small cap so 10 big turns are over, but [summary + 2 tail] is under.
        return apply_budget(msgs, max_tokens=5000, preserve_tail_msgs=2,
                            provider=provider, persona_dir=tmp_path, session_id=sid)

    out1 = one_turn()
    out2 = one_turn()
    out3 = one_turn()
    # Compaction fired exactly once (one accepted cache write), NOT per turn.
    assert len(provider.calls) == 1
    # The buffer is now [summary, *2-tail] and byte-stable across turns 2 & 3.
    after = read_session(tmp_path, sid)
    assert _summary_rows(after)[0]["text"] == "SUM"
    # The current-turn over-cap floor (turn 1) is the deterministic note, not an
    # LLM summary string.
    note = out1[1].content_text() if len(out1) > 1 else ""
    assert "truncated" in note.lower()
    # Turns 2 & 3 are under cap now → passthrough → identical assembled prefix.
    assert [m.content_text() for m in out2] == [m.content_text() for m in out3]


# -------------------------------------------------------------------------- C12
def test_c12_fresh_lock_blocks_then_stale_lock_reaped(tmp_path: Path) -> None:
    sid = "sess_c12"
    # A live lock (this process, fresh mtime) blocks a second acquire.
    assert acquire_compaction_lock(tmp_path, sid) is True
    assert acquire_compaction_lock(tmp_path, sid) is False  # re-entrancy guard

    # Simulate a crashed predecessor: a lock from a dead pid.
    lock = _compacting_lock_path(tmp_path, sid)
    lock.write_text(json.dumps({"pid": 999_999_999, "ts": "2000-01-01T00:00:00+00:00"}))
    # Dead pid -> reaped -> re-acquired.
    assert acquire_compaction_lock(tmp_path, sid) is True


def test_c12_stale_mtime_reaped(tmp_path: Path) -> None:
    sid = "sess_c12b"
    lock = _compacting_lock_path(tmp_path, sid)
    # A live pid (ours) but an ancient mtime -> reaped via the mtime backstop.
    lock.write_text(json.dumps({"pid": os.getpid(), "ts": "2000-01-01T00:00:00+00:00"}))
    old = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(lock, (old, old))
    assert acquire_compaction_lock(tmp_path, sid, stale_s=600.0) is True


# --------------------------------------------------------------------------- C7
def test_c7_count_chat_turns_excludes_summary(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import count_chat_turns_since

    sid = "sess_c7count"
    base = datetime.now(UTC) - timedelta(minutes=10)
    _seed(tmp_path, sid, 4, base=base)
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "summary",
                           "text": "gist", "ts": _iso(datetime.now(UTC))})
    n = count_chat_turns_since(tmp_path, _iso(base - timedelta(minutes=1)))
    assert n == 4  # the summary row is not a turn


def test_c7_session_hours_skips_summary(tmp_path: Path) -> None:
    from brain.body.session_hours import _entry_timestamps

    sid = "sess_c7hours"
    base = datetime.now(UTC) - timedelta(hours=2)
    _seed(tmp_path, sid, 3, base=base)
    # A summary row whose ts is "now" must not extend session age.
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "summary",
                           "text": "gist", "ts": _iso(datetime.now(UTC))})
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    stamps = _entry_timestamps(buf)
    assert len(stamps) == 3  # only the 3 real turns counted
