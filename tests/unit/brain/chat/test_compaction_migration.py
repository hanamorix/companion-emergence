"""Conformance tests for the backwards-compat backlog migration.

Maps to changes/compaction-backlog-migration/1.5-criteria.md (C-M1..C-M12).

The headline fixture is a real 328-turn chatlog copy at
``~/Downloads/30cd3047-3d47-45b9-a48f-e27f9f2a9bee.jsonl`` (buffer format). It is
skipped-with-reason when absent so CI without the fixture still runs the edge cases.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.chat import compaction_migration as mig
from brain.chat.compaction import CompactionResult, compact_conversation
from brain.chat.compaction_migration import _marker_path, run_backlog_migration
from brain.ingest.buffer import read_archive, read_session, write_cursor

FIXTURE = Path.home() / "Downloads" / "30cd3047-3d47-45b9-a48f-e27f9f2a9bee.jsonl"
FIXTURE_SID = "30cd3047-3d47-45b9-a48f-e27f9f2a9bee"


class _StubProvider:
    """Deterministic provider stub; records each generate() prompt."""

    def __init__(self, response: str = "FADED-SUMMARY") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, *, prompt: str, system: str | None = None, **kw) -> str:
        self.calls.append(prompt)
        return self.response


def _identity(t: dict) -> tuple:
    return (t.get("ts"), t.get("speaker"), t.get("text"))


def _raw(turns: list[dict]) -> list[dict]:
    return [t for t in turns if t.get("speaker") != "summary"]


def _load_fixture(persona_dir: Path) -> list[dict]:
    """Copy the real chatlog into active_conversations/, cursor at the last ts."""
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    dest_dir = persona_dir / "active_conversations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"{FIXTURE_SID}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    write_cursor(persona_dir, FIXTURE_SID, rows[-1]["ts"])  # everything extracted
    return rows


def _count_transcript_raw_turns(prompt: str) -> int:
    """Count user:/assistant: (raw) lines in a recorded fold transcript.

    The transcript is "\\n".join("<speaker>: <text>") over [summary?, *removable];
    a leading 'summary:' line (the re-fed, re-compressed head) is excluded so we
    bound the RAW turns per call (the cold-fold protection)."""
    n = 0
    for line in prompt.splitlines():
        head = line.split(":", 1)[0].strip()
        if head in ("user", "assistant"):
            n += 1
    return n


fixture_required = pytest.mark.skipif(
    not FIXTURE.exists(), reason=f"real chatlog fixture not present: {FIXTURE}"
)


# ---------------------------------------------------------------- C-M2/4/7/10
@fixture_required
def test_cm2_cm7_backlog_drained(tmp_path: Path) -> None:
    now = datetime(2026, 6, 30, tzinfo=UTC)  # fixed → deterministic vs the fixture ts
    rows = _load_fixture(tmp_path)
    original_raw = _raw(rows)
    prov = _StubProvider()

    res = run_backlog_migration(tmp_path, provider=prov, now=now)

    # C-M2: head is one summary, only the protected tail (<=40) remains live.
    session = read_session(tmp_path, FIXTURE_SID)
    assert session[0].get("speaker") == "summary"
    live_raw = _raw(session)
    assert len(live_raw) <= 40
    assert _marker_path(tmp_path).exists()
    assert res.marker_written is True

    # Folded everything aged past the tail, across MULTIPLE 24h time-steps (not one
    # enormous call, and not fixed-size count batches — see the time-stepping test).
    assert res.total_compacted == len(original_raw) - len(live_raw)
    assert 1 < len(prov.calls) < len(original_raw)

    # C-M7: drained, no ceiling hit.
    assert res.sessions_drained == 1 and not res.undrained_sessions


# ------------------------------------------------- time-stepping (24h increments)
def test_time_stepping_one_fold_per_24h_cohort(tmp_path: Path) -> None:
    """The migration REPLAYS the daily cadence: one fold per 24h cohort, oldest
    first — not fixed-size message-count batches. 3 day-cohorts → 3 folds (a count
    batcher would do ceil(260/40)=7)."""
    now = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)
    sid, days, per_day = "sess-days", 3, 100
    rows: list[dict] = []
    for d in range(days):  # d=0 oldest; each day sits cleanly inside one 24h window
        base = now - timedelta(hours=24 * (days - d) + 2)
        for i in range(per_day):
            ts = (base + timedelta(minutes=i)).isoformat(timespec="seconds")
            rows.append({"session_id": sid,
                         "speaker": "user" if len(rows) % 2 == 0 else "assistant",
                         "text": f"day{d} turn{i}", "ts": ts})
    ac = tmp_path / "active_conversations"
    ac.mkdir(parents=True)
    (ac / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_cursor(tmp_path, sid, rows[-1]["ts"])
    prov = _StubProvider()

    res = run_backlog_migration(tmp_path, provider=prov, now=now)

    # One fold per 24h cohort, oldest-first; NOT count batches.
    assert len(prov.calls) == days
    assert res.total_passes == days
    # 100 + 100 + (100 - 40 protected tail) = 260 folded.
    assert res.total_compacted == days * per_day - 40
    archived = [t for t in read_archive(tmp_path, sid) if t.get("speaker") != "summary"]
    assert [t["ts"] for t in archived] == sorted(t["ts"] for t in archived)  # oldest-first
    assert archived[0]["text"] == "day0 turn0"  # oldest cohort folded first
    session = read_session(tmp_path, sid)
    assert session[0]["speaker"] == "summary" and len(_raw(session)) == 40


@fixture_required
def test_cm4_lossless_multiset(tmp_path: Path) -> None:
    rows = _load_fixture(tmp_path)
    original = Counter(_identity(t) for t in _raw(rows))
    run_backlog_migration(tmp_path, provider=_StubProvider())

    archived = Counter(_identity(t) for t in read_archive(tmp_path, FIXTURE_SID)
                       if t.get("speaker") != "summary")
    retained = Counter(_identity(t) for t in _raw(read_session(tmp_path, FIXTURE_SID)))
    # Union preserves every raw turn exactly once; nothing lost or duplicated.
    assert archived + retained == original
    assert all(archived[k] <= original[k] for k in archived)  # no dup in archive


@fixture_required
def test_cm10_archive_oldest_first_contiguous(tmp_path: Path) -> None:
    rows = _load_fixture(tmp_path)
    original_raw = _raw(rows)
    run_backlog_migration(tmp_path, provider=_StubProvider())

    archived_raw = [t for t in read_archive(tmp_path, FIXTURE_SID)
                    if t.get("speaker") != "summary"]
    # Oldest-first contiguous: archived == the original raw prefix of that length.
    n = len(archived_raw)
    assert [_identity(t) for t in archived_raw] == [_identity(t) for t in original_raw[:n]]
    tss = [t.get("ts") for t in archived_raw]
    assert tss == sorted(tss)  # non-decreasing ts


# ---------------------------------------------------------------------- C-M1
def _seed_small_backlog(persona_dir: Path, sid: str, n: int = 100) -> list[dict]:
    base = datetime(2026, 6, 16, tzinfo=UTC)
    rows = [
        {"session_id": sid, "speaker": "user" if i % 2 == 0 else "assistant",
         "text": f"turn {i}", "ts": (base + timedelta(minutes=i)).isoformat(timespec="seconds")}
        for i in range(n)
    ]
    d = persona_dir / "active_conversations"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_cursor(persona_dir, sid, rows[-1]["ts"])
    return rows


def test_cm1_marker_gate_is_noop(tmp_path: Path) -> None:
    _seed_small_backlog(tmp_path, "sid-a")
    before = (tmp_path / "active_conversations" / "sid-a.jsonl").read_bytes()
    # Pre-create the marker.
    mp = _marker_path(tmp_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text("{}")
    prov = _StubProvider()

    res = run_backlog_migration(tmp_path, provider=prov)

    assert res.already_migrated is True
    assert prov.calls == []
    after = (tmp_path / "active_conversations" / "sid-a.jsonl").read_bytes()
    assert before == after


# --------------------------------------------------------------------- C-M5
def test_cm5_idempotent_rerun_and_drained(tmp_path: Path) -> None:
    _seed_small_backlog(tmp_path, "sid-a")
    run_backlog_migration(tmp_path, provider=_StubProvider())
    assert _marker_path(tmp_path).exists()

    # (a) re-run with marker present → no provider calls.
    prov2 = _StubProvider()
    res2 = run_backlog_migration(tmp_path, provider=prov2)
    assert res2.already_migrated is True and prov2.calls == []

    # (b) one more direct core call → genuinely drained (nothing_aged).
    res3 = compact_conversation(
        tmp_path, "sid-a", older_than=timedelta(hours=24),
        fold_existing_summary=True, provider=_StubProvider(), max_compact_turns=40)
    assert res3.compacted is False and res3.reason == "nothing_aged"


# --------------------------------------------------------------------- C-M6
def test_cm6_cursor_none_is_noop_but_marked(tmp_path: Path) -> None:
    # Seed a backlog but write NO cursor → cursor_none (drained: nothing extractable).
    base = datetime(2026, 6, 16, tzinfo=UTC)
    rows = [{"session_id": "sid-a", "speaker": "user", "text": f"t{i}",
             "ts": (base + timedelta(minutes=i)).isoformat(timespec="seconds")}
            for i in range(50)]
    d = tmp_path / "active_conversations"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sid-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = (d / "sid-a.jsonl").read_bytes()
    prov = _StubProvider()

    res = run_backlog_migration(tmp_path, provider=prov)

    assert prov.calls == []                       # nothing folded
    assert (d / "sid-a.jsonl").read_bytes() == before
    assert _marker_path(tmp_path).exists()        # cursor_none is drained → marked
    assert res.marker_written is True


def test_cm6_mid_cursor_only_folds_extracted(tmp_path: Path) -> None:
    rows = _seed_small_backlog(tmp_path, "sid-a", n=100)
    mid_ts = rows[50]["ts"]
    write_cursor(tmp_path, "sid-a", mid_ts)  # only turns <= rows[50] are extracted

    run_backlog_migration(tmp_path, provider=_StubProvider())

    for t in read_archive(tmp_path, "sid-a"):
        if t.get("speaker") == "summary":
            continue
        assert t["ts"] <= mid_ts  # never folded an un-extracted turn


# --------------------------------------------------------------------- C-M8
def test_cm8_fault_isolation_one_session_raises(tmp_path: Path, monkeypatch) -> None:
    _seed_small_backlog(tmp_path, "sid-a")
    _seed_small_backlog(tmp_path, "sid-b")

    real = compact_conversation
    seen: list[str] = []

    def flaky(persona_dir, session_id, **kw):
        seen.append(session_id)
        if session_id == "sid-a":
            raise RuntimeError("boom")
        return real(persona_dir, session_id, **kw)

    monkeypatch.setattr(mig, "compact_conversation", flaky)

    # Must NOT raise out of the entry point.
    res = run_backlog_migration(tmp_path, provider=_StubProvider())

    assert "sid-b" in seen                         # other session still processed
    assert "sid-a" in res.undrained_sessions
    assert not _marker_path(tmp_path).exists()     # a failure withholds the marker
    assert res.marker_written is False


# -------------------------------------------------------------------- C-M12
@pytest.mark.parametrize("transient_reason", ["locked", "archive_failed"])
def test_cm12_transient_noop_withholds_marker(tmp_path: Path, monkeypatch, transient_reason) -> None:
    _seed_small_backlog(tmp_path, "sid-a")
    real = compact_conversation

    def transient(persona_dir, session_id, **kw):
        return CompactionResult(False, 0, 0, False, False, reason=transient_reason)

    monkeypatch.setattr(mig, "compact_conversation", transient)
    res1 = run_backlog_migration(tmp_path, provider=_StubProvider())
    assert not _marker_path(tmp_path).exists()     # transient miss → no marker
    assert "sid-a" in res1.undrained_sessions and res1.marker_written is False

    # Restore the real core → it drains and the marker is now written (retry path).
    monkeypatch.setattr(mig, "compact_conversation", real)
    res2 = run_backlog_migration(tmp_path, provider=_StubProvider())
    assert _marker_path(tmp_path).exists() and res2.marker_written is True
