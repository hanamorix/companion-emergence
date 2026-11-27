"""#781: the writers that opened their file in append mode themselves now go through the shared durable appenders.

Each test appends, then simulates an unclean machine stop (#618): every byte past the last fsync becomes NUL. A writer
that skips the fsync loses its record; one that goes through ``append_jsonl`` or ``append_text`` keeps it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def _crash_after(monkeypatch, path: Path) -> Callable[[], None]:
    """Record the file size at every fsync; ``crash()`` NULs every byte past the largest synced size."""
    real_fsync = os.fsync
    synced = [0]

    def spy(fd: int) -> None:
        synced[0] = max(synced[0], os.fstat(fd).st_size)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)

    def crash() -> None:
        data = path.read_bytes()
        keep = data[: synced[0]]
        path.write_bytes(keep + b"\0" * (len(data) - len(keep)))

    return crash


# --- append_text --------------------------------------------------------------------------------------------------


def test_append_text_survives_an_unclean_stop(tmp_path: Path, monkeypatch) -> None:
    from brain.health.json_io import append_text

    p = tmp_path / "notes.md"
    crash = _crash_after(monkeypatch, p)
    append_text(p, "## one\n")
    append_text(p, "## two\n")
    crash()
    assert p.read_text(encoding="utf-8") == "## one\n## two\n"



def test_append_text_starts_a_new_line_after_a_torn_tail_and_writes_utf8(tmp_path: Path) -> None:
    from brain.health.json_io import append_text

    p = tmp_path / "notes.md"
    p.write_bytes(b"half a line")
    append_text(p, "smart \u2014 dash\n")
    assert p.read_bytes() == "half a line\nsmart \u2014 dash\n".encode()


def test_append_text_raises_by_default_and_logs_on_request(tmp_path: Path) -> None:
    import pytest

    from brain.health.json_io import append_text

    d = tmp_path / "a-directory"
    d.mkdir()
    with pytest.raises(OSError):
        append_text(d, "x\n")
    assert append_text(d, "x\n", on_error="log") is None

# --- the migrated writers -----------------------------------------------------------------------------------------


def test_the_chat_turns_log_survives_an_unclean_stop(tmp_path: Path, monkeypatch) -> None:
    from datetime import UTC, datetime

    from brain.felt_time.chat_log import CHAT_TURNS_LOG_FILENAME, append_chat_tick
    from brain.health.json_io import iter_jsonl

    p = tmp_path / CHAT_TURNS_LOG_FILENAME
    crash = _crash_after(monkeypatch, p)
    append_chat_tick(tmp_path, ts=datetime(2026, 11, 27, 12, 0, tzinfo=UTC), turns=2)
    crash()
    assert list(iter_jsonl(p)) == [{"ts": "2026-11-27T12:00:00+00:00", "turns": 2}]


def test_a_draft_fragment_survives_an_unclean_stop(tmp_path: Path, monkeypatch) -> None:
    from brain.initiate.draft import append_draft_fragment

    p = tmp_path / "draft_space.md"
    crash = _crash_after(monkeypatch, p)
    append_draft_fragment(tmp_path, timestamp="2026-11-27T12:00:00+00:00", source="dream", body="a fragment")
    crash()
    assert p.read_text(encoding="utf-8") == "\n\n## 2026-11-27 12:00 (dream)\n\na fragment\n"


def test_a_draft_fragments_duplicate_check_runs_under_the_files_lock(tmp_path: Path, monkeypatch) -> None:
    """Two writers with the same header must not both pass the check and both append: the file is read under its lock."""
    import contextlib

    import brain.initiate.draft as draft

    held = [False]
    reads: list[bool] = []

    @contextlib.contextmanager
    def lock(path, blocking=True):
        held[0] = True
        try:
            yield True
        finally:
            held[0] = False

    real_read = Path.read_text

    def read(self, *a, **k):
        if self.name == "draft_space.md":
            reads.append(held[0])
        return real_read(self, *a, **k)

    monkeypatch.setattr(draft, "file_lock", lock, raising=False)
    monkeypatch.setattr(Path, "read_text", read)
    kw = {"timestamp": "2026-11-27T12:00:00+00:00", "source": "dream", "body": "a fragment"}
    draft.append_draft_fragment(tmp_path, **kw)
    draft.append_draft_fragment(tmp_path, **kw)
    assert reads == [True]  # the second call reads the file, under the lock, and finds the header
    assert real_read(tmp_path / "draft_space.md", encoding="utf-8").count("## 2026-11-27 12:00 (dream)") == 1


def test_a_demoted_candidate_survives_an_unclean_stop(tmp_path: Path, monkeypatch) -> None:
    from datetime import UTC, datetime

    from brain.initiate.reflection import DDecision, demote_to_draft_space
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
    candidate = InitiateCandidate(
        candidate_id="ic_a", ts=now.isoformat(), kind="message", source="dream", source_id="d1",
        semantic_context=SemanticContext(),
    )
    decision = DDecision(candidate_index=1, decision="filter", reason="not now", confidence="high")
    p = tmp_path / "draft_space.md"
    crash = _crash_after(monkeypatch, p)
    demote_to_draft_space(tmp_path, candidate=candidate, decision=decision, now=now)
    crash()
    text = p.read_text(encoding="utf-8")
    assert "\0" not in text
    assert "candidate_id: ic_a" in text
    assert text.endswith("subject-extraction skipped because D filtered)\n\n")


def test_the_cli_crash_log_survives_an_unclean_stop(tmp_path: Path, monkeypatch) -> None:
    import brain.cli as cli

    monkeypatch.setattr(cli, "get_log_dir", lambda: tmp_path)
    p = tmp_path / "cli-crash.log"
    crash = _crash_after(monkeypatch, p)
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        cli._log_crash(exc, ["nell", "chat"])
    crash()
    text = p.read_text(encoding="utf-8")
    assert "\0" not in text
    assert text.startswith("\n=== crash argv=['nell', 'chat'] ===\n")
    assert "RuntimeError: boom" in text
