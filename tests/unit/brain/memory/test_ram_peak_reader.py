"""S220/S222: the per-call peak-growth reader (`brain.memory.peak_rss`). Scripted /proc files
for parsing and failures; a real allocation on this kernel for the reading itself."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from brain.memory import peak_rss

_linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux peak counter")


@pytest.fixture(autouse=True)
def _fresh():
    peak_rss._reset_for_tests()  # noqa: SLF001
    yield
    peak_rss._reset_for_tests()  # noqa: SLF001


def _fake_proc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str) -> Path:
    status_file, clear_file = tmp_path / "status", tmp_path / "clear_refs"
    status_file.write_text(status)
    clear_file.write_text("")
    monkeypatch.setattr(peak_rss, "_PROC_STATUS", str(status_file))
    monkeypatch.setattr(peak_rss, "_CLEAR_REFS", str(clear_file))
    monkeypatch.setattr(peak_rss, "_is_linux", True)
    return clear_file


_STATUS = "Name:\tx\nVmHWM:\t  3000 kB\nVmRSS:\t  1000 kB\nThreads:\t1\n"


def test_begin_resets_the_peak_and_reads_the_rss(monkeypatch, tmp_path) -> None:
    clear_file = _fake_proc(monkeypatch, tmp_path, _STATUS)
    assert peak_rss.begin() == 1000 * 1024.0
    assert clear_file.read_text() == "5", "the peak counter is reset with 5"


def test_growth_is_the_peak_minus_the_rss_before_and_never_negative(monkeypatch, tmp_path) -> None:
    _fake_proc(monkeypatch, tmp_path, _STATUS)
    assert peak_rss.growth_since(1000 * 1024.0) == 2000 * 1024.0
    assert peak_rss.growth_since(5000 * 1024.0) == 0.0


def test_availability_probe_has_no_side_effect(monkeypatch, tmp_path) -> None:
    clear_file = _fake_proc(monkeypatch, tmp_path, _STATUS)
    assert peak_rss.available() is True
    assert clear_file.read_text() == "", "available() must not reset the peak"


def test_off_linux_nothing_is_available(monkeypatch, tmp_path) -> None:
    _fake_proc(monkeypatch, tmp_path, _STATUS)
    monkeypatch.setattr(peak_rss, "_is_linux", False)
    assert peak_rss.available() is False
    assert peak_rss.begin() is None
    assert peak_rss.growth_since(None) is None


def test_growth_without_a_begin_value_is_none(monkeypatch, tmp_path) -> None:
    _fake_proc(monkeypatch, tmp_path, _STATUS)
    assert peak_rss.growth_since(None) is None


def test_a_reset_that_cannot_be_written_latches_the_reader_off(
    monkeypatch, tmp_path, caplog
) -> None:
    _fake_proc(monkeypatch, tmp_path, _STATUS)
    monkeypatch.setattr(peak_rss, "_CLEAR_REFS", str(tmp_path))  # a directory: open() fails
    monkeypatch.setattr(peak_rss.os, "access", lambda *a, **k: True)
    with caplog.at_level("WARNING"):
        assert peak_rss.begin() is None
    assert any("could not be reset" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(peak_rss, "_CLEAR_REFS", str(tmp_path / "clear_refs"))
    assert peak_rss.available() is False, "latched for the process"
    assert peak_rss.begin() is None


def test_an_unreadable_rss_latches_the_reader_off(monkeypatch, tmp_path) -> None:
    _fake_proc(monkeypatch, tmp_path, "Name:\tx\nVmHWM:\t3000 kB\n")  # no VmRSS
    assert peak_rss.begin() is None
    assert peak_rss.available() is False


def test_an_unreadable_peak_latches_the_reader_off(monkeypatch, tmp_path) -> None:
    _fake_proc(monkeypatch, tmp_path, "Name:\tx\nVmRSS:\t1000 kB\n")  # no VmHWM
    before = peak_rss.begin()
    assert before is not None
    assert peak_rss.growth_since(before) is None
    assert peak_rss.available() is False


def test_the_failure_is_warned_once(monkeypatch, tmp_path, caplog) -> None:
    _fake_proc(monkeypatch, tmp_path, "Name:\tx\n")
    with caplog.at_level("DEBUG", logger=peak_rss.log.name):
        peak_rss.begin()
        peak_rss._latched_off = False  # noqa: SLF001
        peak_rss.begin()
    levels = [
        r.levelname for r in caplog.records if "RAM growth cannot be measured" in r.getMessage()
    ]
    assert levels == ["WARNING", "DEBUG"]


# ---------------------------------------------------------------------------
# The real kernel
# ---------------------------------------------------------------------------


def _touch(megabytes: int):
    import numpy as np

    array = np.ones(megabytes * 1024 * 1024 // 8, dtype=np.float64)  # written: pages resident
    return array


@_linux_only
def test_a_real_allocation_inside_a_call_shows_as_peak_growth_and_not_as_rss_difference() -> None:
    """The old reader (RSS after minus before) reads ~0 for a call that returns its memory:
    this call allocates and frees 200 MB, like the reranker with the memory pool off."""
    import psutil

    if not peak_rss.available():
        pytest.skip("the peak counter cannot be reset here")
    process = psutil.Process()
    before = peak_rss.begin()
    assert before is not None
    rss_before = process.memory_info().rss
    block = _touch(200)
    del block
    rss_after = process.memory_info().rss
    growth = peak_rss.growth_since(before)
    assert growth is not None
    assert growth >= 150 * 1024 * 1024, growth
    assert rss_after - rss_before < 50 * 1024 * 1024, "the old reading would have said ~0"


@_linux_only
def test_a_stale_peak_is_reset_by_begin() -> None:
    """A process that once peaked at 300 MB must still read a later 50 MB call as ~50 MB."""
    if not peak_rss.available():
        pytest.skip("the peak counter cannot be reset here")
    old = _touch(300)
    del old
    before = peak_rss.begin()
    assert before is not None
    small = _touch(50)
    growth = peak_rss.growth_since(before)
    del small
    assert growth is not None
    assert 40 * 1024 * 1024 <= growth < 150 * 1024 * 1024, growth
