"""#523: chunked memory extraction. The whole since-cursor span reaches the extractor, oldest first, in chunks no larger
than ``ingest.extraction_chunk_chars``, and the cursor advances per committed chunk.

Synthetic persona only (persona Kestrel, user Bob, made-up words). Criteria: ``1.5-criteria.md`` of the #523 run (C1-C12).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from brain.ingest import ingest_utils
from brain.ingest.extract import format_transcript, format_turn_line
from brain.ingest.ingest_utils import TurnChunk, chunk_turns, extraction_chunk_chars

_T0 = datetime(2026, 10, 1, 20, 0, 0, tzinfo=UTC)


def _ts(seconds: float) -> str:
    return (_T0 + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _row(speaker: str, text: str, seconds: float | None) -> dict:
    row = {"session_id": "sess_kestrel", "speaker": speaker, "text": text}
    if seconds is not None:
        row["ts"] = _ts(seconds)
    return row


# ---------------------------------------------------------------------------
# Oracles (each is shown to fail on a named broken chunker variant below)
# ---------------------------------------------------------------------------


def _boundary_ts_violations(chunks: list[TurnChunk]) -> list[int]:
    """C3(a): indexes k where a row of chunk k shares (or exceeds) a ts of a row in a later chunk."""
    bad = []
    for k in range(len(chunks) - 1):
        before = [ingest_utils.parse_turn_ts(t.get("ts")) for c in chunks[: k + 1] for t in c.turns]
        after = [ingest_utils.parse_turn_ts(t.get("ts")) for c in chunks[k + 1 :] for t in c.turns]
        before = [b for b in before if b is not None]
        after = [a for a in after if a is not None]
        if before and after and max(before) >= min(after):
            bad.append(k)
    return bad


def _reply_openers(chunks: list[TurnChunk]) -> list[int]:
    """C3(b): indexes of chunks after the first that open with an assistant or image row."""
    return [k for k, c in enumerate(chunks) if k > 0 and c.turns[0].get("speaker") in ("assistant", "image")]


def _chunk_text_problems(chunks: list[TurnChunk], max_chars: int) -> list[str]:
    """C2 at the chunker: each chunk's text is exactly its whole rows formatted, within the limit unless oversize."""
    problems = []
    for k, c in enumerate(chunks):
        expected = "\n".join(format_turn_line(t, "Bob", "Kestrel") for t in c.turns)
        if c.text != expected or c.text != format_transcript(list(c.turns), user_name="Bob", assistant_name="Kestrel"):
            problems.append(f"chunk {k}: text is not its whole rows")
        if len(c.text) > max_chars and not c.oversize:
            problems.append(f"chunk {k}: {len(c.text)} > {max_chars} and not oversize")
    return problems


def _same_ts_pairs_history() -> list[dict]:
    """Exchanges where the user row and its reply share one second, and a group of four rows in one second."""
    rows = []
    t = 0
    for i in range(6):
        rows.append(_row("user", f"zorbel question {i} " + "x" * 40, t))
        rows.append(_row("assistant", f"quindle answer {i} " + "y" * 40, t))
        t += 30
    # four rows in one second: no cut may fall inside it even before its second user row
    rows += [
        _row("user", "plimsy one " + "a" * 40, t),
        _row("assistant", "plimsy two " + "b" * 40, t),
        _row("user", "plimsy three " + "c" * 40, t),
        _row("assistant", "plimsy four " + "d" * 40, t),
    ]
    return rows


def _straddling_pairs_history() -> list[dict]:
    """Exchanges whose reply is stamped a second (or minutes) after its user row."""
    rows = []
    t = 0
    for i in range(8):
        rows.append(_row("user", f"vantric ask {i} " + "u" * 50, t))
        rows.append(_row("assistant", f"vantric reply {i} " + "r" * 50, t + 1 + (i % 3) * 60))
        t += 300
    return rows


def _any_boundary(speaker, ts, prefix_max, suffix_min):  # noqa: ANN001 - variant of ingest_utils._cut_allowed
    return True


def _ts_rule_only(speaker, ts, prefix_max, suffix_min):  # noqa: ANN001 - variant: no exchange rule
    return ts is not None and prefix_max is not None and suffix_min is not None and prefix_max < suffix_min


# ---------------------------------------------------------------------------
# The chunker (C2, C3 a/b at the chunker; the pipeline halves are below)
# ---------------------------------------------------------------------------


def test_chunk_text_is_whole_rows_within_the_limit() -> None:
    rows = _straddling_pairs_history() + _same_ts_pairs_history()
    for max_chars in (150, 300, 700, 5000):
        chunks = chunk_turns(rows, max_chars=max_chars, user_name="Bob", assistant_name="Kestrel")
        assert [t for c in chunks for t in c.turns] == rows  # every row once, in order
        assert _chunk_text_problems(chunks, max_chars) == []


def test_same_ts_rows_are_never_split() -> None:
    chunks = chunk_turns(_same_ts_pairs_history(), max_chars=120, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) > 3
    assert _boundary_ts_violations(chunks) == []
    # the four rows of one second stay in one chunk even though the group holds a second user row
    group = [c for c in chunks if any("plimsy" in t["text"] for t in c.turns)]
    assert len(group) == 1 and len(group[0].turns) == 4 and group[0].oversize


def test_no_chunk_opens_with_a_reply_even_when_the_pair_straddles_a_second() -> None:
    chunks = chunk_turns(_straddling_pairs_history(), max_chars=80, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) == 8  # each exchange is its own (oversize) chunk: a user row is never cut from its reply
    assert _reply_openers(chunks) == []
    assert all(len(c.turns) == 2 for c in chunks)


@pytest.mark.parametrize(
    ("variant", "history", "oracle"),
    [
        (_any_boundary, _same_ts_pairs_history, _boundary_ts_violations),
        (_any_boundary, _straddling_pairs_history, _reply_openers),
        (_ts_rule_only, _straddling_pairs_history, _reply_openers),
    ],
    ids=["any-boundary-splits-same-ts", "any-boundary-opens-with-reply", "no-exchange-rule-opens-with-reply"],
)
def test_oracles_catch_broken_chunker_variants(monkeypatch, variant, history, oracle) -> None:
    """Fail-first kept executable (C3): the oracles above fail on a chunker that cuts at any row boundary, and on one
    without the exchange rule."""
    monkeypatch.setattr(ingest_utils, "_cut_allowed", variant)
    chunks = chunk_turns(history(), max_chars=80, user_name="Bob", assistant_name="Kestrel")
    assert oracle(chunks) != []


def test_cursor_ts_is_the_newest_row_up_to_the_chunk_end() -> None:
    rows = _straddling_pairs_history()
    chunks = chunk_turns(rows, max_chars=300, user_name="Bob", assistant_name="Kestrel")
    seen: list[dict] = []
    for c in chunks:
        seen += list(c.turns)
        newest = max(seen, key=lambda t: ingest_utils.parse_turn_ts(t["ts"]))
        assert c.cursor_ts == newest["ts"]


def test_a_unit_larger_than_the_limit_is_sent_whole() -> None:
    rows = [_row("user", "short " * 3, 0), _row("assistant", "grumvex " * 400, 1), _row("user", "after", 60)]
    chunks = chunk_turns(rows, max_chars=500, user_name="Bob", assistant_name="Kestrel")
    assert [len(c.turns) for c in chunks] == [2, 1]
    assert chunks[0].oversize and len(chunks[0].text) > 500
    assert "grumvex " * 400 in chunks[0].text  # whole, never cut mid-row
    assert not chunks[1].oversize


def test_rows_without_a_ts_join_the_row_before_and_are_logged(caplog) -> None:
    rows = [_row("user", "a" * 60, 0), _row("assistant", "b" * 60, 1), _row("user", "c" * 60, None), _row("user", "d" * 60, 90)]
    with caplog.at_level(logging.WARNING, logger="brain.ingest.ingest_utils"):
        chunks = chunk_turns(rows, max_chars=70, user_name="Bob", assistant_name="Kestrel")
    assert [[t["text"][0] for t in c.turns] for c in chunks] == [["a", "b", "c"], ["d"]]
    assert any("conversation_extract_ts_missing" in r.message for r in caplog.records)


def test_summary_rows_are_dropped() -> None:
    rows = [_row("summary", "faded recap", 999), _row("user", "hello", 0), _row("assistant", "hi", 1)]
    chunks = chunk_turns(rows, max_chars=1000, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) == 1 and [t["speaker"] for t in chunks[0].turns] == ["user", "assistant"]
    assert chunks[0].cursor_ts == _ts(1)


# ---------------------------------------------------------------------------
# C12: the named limit, validated
# ---------------------------------------------------------------------------


@pytest.fixture()
def tunables_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t
    from brain.utils import log_throttle

    t._reset_for_tests()
    log_throttle._last.clear()
    yield tmp_path
    t._reset_for_tests()
    log_throttle._last.clear()


def _write_tunables(home, overrides) -> None:
    (home / "tunables.json").write_text(json.dumps({"defaults": {}, "overrides": overrides}), encoding="utf-8")


def test_chunk_chars_default_is_registered_with_its_value(tunables_home) -> None:
    import brain.tunables as t

    assert extraction_chunk_chars() == 24_000
    assert t._registry["ingest.extraction_chunk_chars"] == 24_000


@pytest.mark.parametrize("bad", [0, -1, "x", True, False])
def test_a_bad_override_is_ignored_with_one_warning(tunables_home, caplog, bad) -> None:
    _write_tunables(tunables_home, {"ingest.extraction_chunk_chars": bad})
    with caplog.at_level(logging.WARNING):
        assert extraction_chunk_chars() == 24_000
        assert extraction_chunk_chars() == 24_000
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "ingest.extraction_chunk_chars" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_positive_override_changes_the_chunk_size(tunables_home) -> None:
    _write_tunables(tunables_home, {"ingest.extraction_chunk_chars": 120})
    assert extraction_chunk_chars() == 120
