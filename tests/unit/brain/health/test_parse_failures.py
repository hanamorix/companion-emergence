"""#364: a model reply that fails to parse leaves its raw text behind."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health import parse_failures as pf


def _rows(persona_dir: Path) -> list[dict]:
    path = persona_dir / pf.PARSE_FAILURE_LOG
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_site_error_and_raw_reply(tmp_path: Path) -> None:
    pf.record_parse_failure(tmp_path, site="extractor", raw="```json\n{oops", error="Expecting value")
    (row,) = _rows(tmp_path)
    assert row["site"] == "extractor"
    assert row["error"] == "Expecting value"
    assert row["raw"] == "```json\n{oops"
    assert row["raw_len"] == len("```json\n{oops")
    assert datetime.fromisoformat(row["ts"]).tzinfo is not None


def test_long_replies_are_truncated_but_their_length_kept(tmp_path: Path) -> None:
    raw = "x" * (pf.RAW_CHARS + 500)
    pf.record_parse_failure(tmp_path, site="s", raw=raw, error="e")
    (row,) = _rows(tmp_path)
    assert len(row["raw"]) == pf.RAW_CHARS
    assert row["raw_len"] == pf.RAW_CHARS + 500


def test_the_log_is_size_capped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pf, "MAX_BYTES", 2_000)
    for _ in range(50):
        pf.record_parse_failure(tmp_path, site="s", raw="y" * 200, error="e")
    assert (tmp_path / pf.PARSE_FAILURE_LOG).stat().st_size < 2_000 + 400
    archives = sorted(tmp_path.glob(pf.PARSE_FAILURE_LOG + ".*.gz"))
    assert 1 <= len(archives) <= pf.ARCHIVE_KEEP


def test_never_raises_when_the_write_fails(tmp_path: Path) -> None:
    missing = tmp_path / "no" / "such" / "dir"
    pf.record_parse_failure(missing, site="s", raw="r", error="e")  # must not raise


def test_no_persona_dir_still_does_not_raise() -> None:
    pf.record_parse_failure(None, site="s", raw="r", error="e")


def test_count_recent_groups_by_site_inside_the_window(tmp_path: Path) -> None:
    now = datetime(2026, 10, 16, 12, tzinfo=UTC)
    path = tmp_path / pf.PARSE_FAILURE_LOG
    rows = [
        {"ts": (now - timedelta(hours=1)).isoformat(), "site": "extractor"},
        {"ts": (now - timedelta(hours=2)).isoformat(), "site": "extractor"},
        {"ts": (now - timedelta(hours=3)).isoformat(), "site": "attunement"},
        {"ts": (now - timedelta(hours=30)).isoformat(), "site": "extractor"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n", encoding="utf-8")
    assert pf.count_recent(tmp_path, hours=24, now=now) == {"extractor": 2, "attunement": 1}


def test_count_recent_is_empty_without_a_log(tmp_path: Path) -> None:
    assert pf.count_recent(tmp_path) == {}


def test_count_recent_survives_bad_bytes_and_naive_timestamps(tmp_path: Path) -> None:
    """/health reads this; a corrupt or tz-less row must never make it raise."""
    now = datetime(2026, 10, 16, 12, tzinfo=UTC)
    path = tmp_path / pf.PARSE_FAILURE_LOG
    good = json.dumps({"ts": (now - timedelta(hours=1)).isoformat(), "site": "extractor"})
    naive = json.dumps({"ts": "2026-10-16T11:00:00", "site": "naive"})
    path.write_bytes(good.encode() + b"\n" + b"\xff\xfe broken\n" + naive.encode() + b"\n")
    assert pf.count_recent(tmp_path, now=now) == {"extractor": 1}


def test_keep_raw_false_stores_only_the_length(tmp_path: Path) -> None:
    """Kindled sites: a peer's text never lands in the persona-root log."""
    pf.record_parse_failure(tmp_path, site="privacy_gate", raw="peer secret text", error="e", keep_raw=False)
    (row,) = _rows(tmp_path)
    assert row["raw"] == "" and row["raw_len"] == len("peer secret text")


def test_the_runtime_log_line_carries_no_reply_text(tmp_path: Path, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        pf.record_parse_failure(tmp_path, site="s", raw="SECRET-REPLY-TEXT", error="bad")
    assert "SECRET-REPLY-TEXT" not in caplog.text and "s" in caplog.text


def test_concurrent_writers_lose_no_rows(tmp_path: Path, monkeypatch) -> None:
    import threading

    monkeypatch.setattr(pf, "MAX_BYTES", 4_000)
    monkeypatch.setattr(pf, "ARCHIVE_KEEP", 50)

    def write(n: int) -> None:
        for i in range(40):
            pf.record_parse_failure(tmp_path, site=f"t{n}", raw="z" * 100, error=str(i))

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    import gzip

    total = sum(1 for _ in (tmp_path / pf.PARSE_FAILURE_LOG).read_text(encoding="utf-8").splitlines())
    for gz in tmp_path.glob(pf.PARSE_FAILURE_LOG + ".*.gz"):
        total += sum(1 for line in gzip.decompress(gz.read_bytes()).decode("utf-8").splitlines() if line)
    assert total == 160
