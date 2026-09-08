"""Tests for brain.health.jsonl_reader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt


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
    from brain.health.jsonl_reader import (
        iter_jsonl_skipping_corrupt,
        read_jsonl_skipping_corrupt,
    )

    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    assert read_jsonl_skipping_corrupt(p) == list(iter_jsonl_skipping_corrupt(p))


# ---------------------------------------------------------------------------
# read_last_n_jsonl_lines — bounded backward-seek tail read (#225, C21)
# ---------------------------------------------------------------------------


def test_read_last_n_missing_file_returns_empty(tmp_path: Path) -> None:
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    assert read_last_n_jsonl_lines(tmp_path / "missing.jsonl", 5) == []


def test_read_last_n_zero_or_negative_returns_empty(tmp_path: Path) -> None:
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"i": 0}) + "\n", encoding="utf-8")
    assert read_last_n_jsonl_lines(p, 0) == []
    assert read_last_n_jsonl_lines(p, -3) == []


def test_read_last_n_returns_the_final_n_lines_in_order(tmp_path: Path) -> None:
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(20)) + "\n", encoding="utf-8"
    )
    out = read_last_n_jsonl_lines(p, 3)
    assert [json.loads(line)["i"] for line in out] == [17, 18, 19]


def test_read_last_n_fewer_than_n_total_lines_returns_all_including_first(
    tmp_path: Path,
) -> None:
    """C21(a): a file with fewer than n total lines returns ALL lines,
    including the first — a version that unconditionally drops the leading
    line (round-6's ambiguous "always drop first" prose) must fail this."""
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(3)) + "\n", encoding="utf-8"
    )
    out = read_last_n_jsonl_lines(p, 100)
    assert [json.loads(line)["i"] for line in out] == [0, 1, 2]


def test_read_last_n_fewer_than_n_lines_no_trailing_newline(tmp_path: Path) -> None:
    """Same BOF case, but the file has no trailing newline after the last line."""
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(3)), encoding="utf-8"
    )  # no trailing \n
    out = read_last_n_jsonl_lines(p, 100)
    assert [json.loads(line)["i"] for line in out] == [0, 1, 2]


def test_read_last_n_self_test_naive_drop_first_line_fails_boundary_case(
    tmp_path: Path,
) -> None:
    """ST1.5f: a version that unconditionally drops the leading line
    (conflating "enough newlines found" with "BOF reached", round-6's bug)
    WOULD fail the fewer-than-n-lines case above — demonstrated directly so
    the assertion is shown to be discriminating, not vacuously true."""
    p = tmp_path / "log.jsonl"
    p.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(3)) + "\n", encoding="utf-8"
    )
    # Simulate the naive "always drop the first accumulated line" version by
    # replicating just that (buggy) tail end of the algorithm inline.
    with p.open("rb") as f:
        block = f.read()
    text = block.decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines.pop(0)  # the bug: always drops, even though BOF was reached
    assert [json.loads(line)["i"] for line in lines] != [0, 1, 2]  # first line lost


def test_read_last_n_multibyte_utf8_across_chunk_boundary_round_trips(
    tmp_path: Path,
) -> None:
    """C21(b): a multi-byte UTF-8 character positioned to straddle a
    chunk_size boundary must round-trip correctly, not be corrupted or raise.

    Byte-accumulate-then-decode-once is what makes this safe: decoding each
    backward-read chunk in isolation (the bug this guards against) would
    split the multi-byte sequence and either raise or mangle it.
    """
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    # Build a file where a 3-byte UTF-8 character (e.g. "の", U+306E) sits
    # exactly on a chunk boundary when read backward in small chunks.
    padding = "a" * 5  # arbitrary filler so the multibyte char lands mid-chunk
    row1 = json.dumps({"subject": padding + "の" + padding}, ensure_ascii=False)
    row2 = json.dumps({"subject": "second row"}, ensure_ascii=False)
    content = row1 + "\n" + row2 + "\n"
    p = tmp_path / "log.jsonl"
    p.write_text(content, encoding="utf-8")

    # chunk_size chosen so the backward seek must split mid-character at
    # some byte offset within row1's encoded bytes.
    row1_bytes = row1.encode("utf-8")
    split_point = len(row1_bytes) - 3  # lands inside/near the multibyte char
    assert 0 < split_point < len(row1_bytes)

    out = read_last_n_jsonl_lines(p, 5, chunk_size=max(split_point, 1))
    parsed = [json.loads(line) for line in out]
    assert parsed[0]["subject"] == padding + "の" + padding
    assert parsed[1]["subject"] == "second row"


def test_read_last_n_bounded_io_does_not_read_whole_large_file(tmp_path: Path) -> None:
    """Supports C19: the read cost scales with n, not file size — a large
    file (10k+ rows) with a small n must not require reading from byte 0."""
    from brain.health.jsonl_reader import read_last_n_jsonl_lines

    p = tmp_path / "big.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(10_000):
            fh.write(json.dumps({"i": i}) + "\n")

    file_size = p.stat().st_size

    real_open = Path.open
    seek_positions: list[int] = []

    def spying_open(self, *args, **kwargs):
        fh = real_open(self, *args, **kwargs)
        if self == p and "b" in (args[0] if args else kwargs.get("mode", "r")):
            original_seek = fh.seek

            def spying_seek(pos, *a, **k):
                # Only track SEEK_SET (absolute-from-start) seeks — the
                # initial f.seek(0, os.SEEK_END) passes pos=0 too, but that
                # means "0 offset from EOF," not byte 0 of the file.
                whence = a[0] if a else k.get("whence", 0)
                if whence == 0:
                    seek_positions.append(pos)
                return original_seek(pos, *a, **k)

            fh.seek = spying_seek
        return fh

    import unittest.mock as mock

    with mock.patch.object(Path, "open", spying_open):
        out = read_last_n_jsonl_lines(p, 5)

    assert [json.loads(line)["i"] for line in out] == [9995, 9996, 9997, 9998, 9999]
    # The furthest-back seek position reached must stay far from byte 0 —
    # proof the read never walked back anywhere near the whole file.
    assert min(seek_positions) > file_size * 0.9
