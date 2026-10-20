"""Tests for brain.health.jsonl_reader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt


def read_jsonl_skipping_corrupt(path: Path) -> list[dict]:
    """The list form of the kept streaming reader (the list wrapper itself was deleted in phase 2)."""
    return list(iter_jsonl_skipping_corrupt(path))


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_jsonl_skipping_corrupt(tmp_path / "missing.jsonl") == []


def test_well_formed_lines_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    out = read_jsonl_skipping_corrupt(p)
    assert out == [{"a": 1}, {"a": 2}]


def test_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    assert read_jsonl_skipping_corrupt(p) == [{"a": 1}, {"a": 2}]


def test_skips_corrupt_lines_and_warns(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"good": 1}) + "\n{not valid\n" + json.dumps({"good": 2}) + "\n",
        encoding="utf-8",
    )
    out = read_jsonl_skipping_corrupt(p)
    assert out == [{"good": 1}, {"good": 2}]

    bad = [r for r in caplog.records if "malformed jsonl line" in r.getMessage()]
    assert len(bad) == 1
    msg = bad[0].getMessage()
    assert "line 2" in msg
    assert "{not valid" in msg


def test_warning_includes_path_and_truncates_long_content(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    p = tmp_path / "log.jsonl"
    long_corrupt = "{" + ("x" * 500)
    p.write_text(long_corrupt + "\n", encoding="utf-8")
    read_jsonl_skipping_corrupt(p)
    msg = next(r.getMessage() for r in caplog.records if "malformed jsonl line" in r.getMessage())
    assert str(p) in msg
    assert "x" * 200 in msg  # 200-char preview
    assert "x" * 500 not in msg  # truncated


# ---------------------------------------------------------------------------
# iter_jsonl_skipping_corrupt — streaming variant (audit 2026-05-07 P3)
# ---------------------------------------------------------------------------


def test_iter_yields_dicts_one_at_a_time(tmp_path: Path) -> None:
    """The generator yields lazily; consumer can break out early."""
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    g = iter_jsonl_skipping_corrupt(p)
    first = next(g)
    second = next(g)
    assert first == {"i": 0}
    assert second == {"i": 1}
    # The remaining 8 lines have not been consumed yet.


def test_iter_missing_file_yields_nothing(tmp_path: Path) -> None:
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    assert list(iter_jsonl_skipping_corrupt(tmp_path / "missing.jsonl")) == []


def test_iter_skips_corrupt_lines_without_breaking_the_stream(tmp_path: Path, caplog) -> None:
    """A bad line in the middle doesn't stop the iterator from yielding the rest."""
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"a": 1}) + "\n" + "{not valid json}\n" + json.dumps({"a": 3}) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        out = list(iter_jsonl_skipping_corrupt(p))
    assert out == [{"a": 1}, {"a": 3}]
    assert any("malformed jsonl line 2" in r.message for r in caplog.records)


def test_iter_skips_non_dict_json(tmp_path: Path) -> None:
    """Lists / scalars / null aren't dicts — silently skipped per JSONL contract."""
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"ok": True})
        + "\n"
        + json.dumps([1, 2, 3])
        + "\n"
        + json.dumps("just a string")
        + "\n"
        + json.dumps(None)
        + "\n"
        + json.dumps({"also": "ok"})
        + "\n",
        encoding="utf-8",
    )
    out = list(iter_jsonl_skipping_corrupt(p))
    assert out == [{"ok": True}, {"also": "ok"}]


def test_streaming_does_not_load_whole_file_into_memory(tmp_path: Path) -> None:
    """A 5 MB log can be processed without spiking memory.

    The streaming guarantee is hard to assert directly without psutil,
    so we settle for: the iterator works on a file too big for the
    test runner to want to keep multiple copies of, AND consumers can
    bail out after the first record without having parsed the rest.
    """
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    p = tmp_path / "big.jsonl"
    # ~5 MB of "filler" content per record, ~1000 records.
    filler = "x" * 5000
    with open(p, "w", encoding="utf-8") as fh:
        for i in range(1000):
            fh.write(json.dumps({"i": i, "filler": filler}) + "\n")
    # Iterator should produce records lazily — consume only first 3.
    g = iter_jsonl_skipping_corrupt(p)
    out = [next(g), next(g), next(g)]
    assert [r["i"] for r in out] == [0, 1, 2]


def test_read_jsonl_is_iter_jsonl_materialised(tmp_path: Path) -> None:
    """The list wrapper yields exactly the same dicts the generator does."""
    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    assert read_jsonl_skipping_corrupt(p) == list(iter_jsonl_skipping_corrupt(p))


def test_a_line_with_invalid_utf8_is_skipped_not_raised(tmp_path, caplog):
    """#433: a crash mid-append can cut a multi-byte character; the next append
    lands on the same line. That merged line is skipped with a warning, like a
    JSON error (the row appended onto it is lost with it, as for any torn
    line), and the read goes on; raising instead would stop a fail-closed
    caller (#392) on every read until a manual repair."""
    import json
    import logging

    good = json.dumps({"text": "über"}, ensure_ascii=False).encode("utf-8")
    torn = good[: good.index("ü".encode()) + 1]  # cut inside the 2-byte ü
    path = tmp_path / "log.jsonl"
    path.write_bytes(good + b"\n" + torn + good + b"\n" + good + b"\n")

    with caplog.at_level(logging.WARNING, logger="brain.health.jsonl_reader"):
        rows = read_jsonl_skipping_corrupt(path)

    assert rows == [{"text": "über"}, {"text": "über"}]
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "line 2" in warning.getMessage() and str(path) in warning.getMessage()


def _torn_utf8_bytes() -> bytes:
    import json

    good = json.dumps({"text": "über"}, ensure_ascii=False).encode("utf-8")
    torn = good[: good.index("ü".encode()) + 1]
    return good + b"\n" + torn + good + b"\n" + good + b"\n"


def test_streaming_reader_skips_invalid_utf8_plain_and_gzip(tmp_path, caplog):
    """#433: the audit readers over active + .gz archives (now json_io.iter_jsonl, which replaced the deleted
    iter_jsonl_streaming) skip a torn UTF-8 line instead of raising out of the loop."""
    import gzip
    import logging

    from brain.health.json_io import iter_jsonl

    plain = tmp_path / "audit.jsonl"
    plain.write_bytes(_torn_utf8_bytes())
    gz = tmp_path / "audit.jsonl.1.gz"
    gz.write_bytes(gzip.compress(_torn_utf8_bytes()))
    for path in (plain, gz):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="brain.health.jsonl_reader"):
            rows = list(iter_jsonl(path))
        assert rows == [{"text": "über"}, {"text": "über"}], path
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1, path
