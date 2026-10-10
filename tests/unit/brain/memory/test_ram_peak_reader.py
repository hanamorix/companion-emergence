"""S220/S222: the per-call peak-growth reader (`brain.memory.peak_rss`). Scripted /proc files
for parsing and failures; a real allocation on this kernel for the reading itself."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from brain.memory import peak_rss

_linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux peak counter")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    # The suite's autouse fixture switches the reader off (fake rerankers); these tests are
    # about the reader itself, so it is back to the platform's real answer.
    monkeypatch.setattr(peak_rss, "_is_linux", sys.platform.startswith("linux"))
    monkeypatch.setattr(peak_rss, "_disabled", False)
    monkeypatch.setattr(peak_rss, "_force_sampler", False)
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


def test_off_linux_the_sampler_reads_the_peak_and_a_disabled_reader_reads_nothing(monkeypatch, tmp_path) -> None:
    """S248 (was: off Linux nothing was available). Off Linux the RSS sampler is the reader; the suite's switch
    turns every reader off."""
    _fake_proc(monkeypatch, tmp_path, _STATUS)
    monkeypatch.setattr(peak_rss, "_is_linux", False)
    monkeypatch.setattr(peak_rss, "_rss_now", lambda: 1000.0)
    assert peak_rss.available() is True
    monkeypatch.setattr(peak_rss, "_disabled", True)
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


# --- S248: the RSS sampling reader (macOS/Windows), forced here on Linux ---

import threading  # noqa: E402


def _samplers() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "peak-rss-sampler" and t.is_alive()]


class _Rss:
    """The scripted resident set: a variable the test sets (state-driven, no timing-dependent value list)."""

    def __init__(self, value: float) -> None:
        self.value: float | None = value

    def __call__(self) -> float | None:
        return self.value


@pytest.fixture
def forced(monkeypatch: pytest.MonkeyPatch) -> _Rss:
    rss = _Rss(100.0)
    monkeypatch.setattr(peak_rss, "_force_sampler", True)
    monkeypatch.setattr(peak_rss, "_rss_now", rss)
    monkeypatch.setattr(peak_rss, "_SAMPLE_INTERVAL_S", 0.001)
    yield rss
    peak_rss.cancel()


def _wait_for(predicate, seconds: float = 5.0) -> bool:
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def test_the_sampler_reports_the_largest_resident_set_during_the_call_raw(forced) -> None:
    before = peak_rss.begin()
    assert before == 100.0 and len(_samplers()) == 1
    forced.value = 700.0
    assert _wait_for(lambda: peak_rss._sampler is not None and peak_rss._sampler.largest == 700.0)  # noqa: SLF001
    forced.value = 150.0  # the memory goes back before the call ends
    assert peak_rss.growth_since(before) == 600.0, "the peak (700) minus the baseline (100), not the final difference (50)"
    assert _samplers() == [], "the thread is stopped"


def test_the_sampler_takes_one_last_reading_and_never_reports_a_negative_growth(forced, monkeypatch) -> None:
    monkeypatch.setattr(peak_rss, "_SAMPLE_INTERVAL_S", 3600.0)  # no periodic reading happens at all
    before = peak_rss.begin()
    forced.value = 900.0  # the final reading is the only one that sees it
    assert peak_rss.growth_since(before) == 800.0
    forced.value = 100.0
    before = peak_rss.begin()
    forced.value = 40.0
    assert peak_rss.growth_since(before) == 0.0


def test_a_second_begin_and_cancel_stop_an_earlier_sampler_and_a_stray_growth_is_none(forced) -> None:
    first = peak_rss.begin()
    assert len(_samplers()) == 1
    second = peak_rss.begin()
    assert _wait_for(lambda: len(_samplers()) == 1), "one sampler at a time"
    assert peak_rss.growth_since(second) is not None
    assert peak_rss.growth_since(first) is None, "no active sampler: no sample, no exception"
    peak_rss.begin()
    peak_rss.cancel()
    peak_rss.cancel()  # idempotent
    assert _samplers() == []


def test_an_unreadable_rss_latches_the_sampler_reader_off_with_one_warning(forced, caplog) -> None:
    forced.value = None
    with caplog.at_level("DEBUG"):
        assert peak_rss.available() is False
        assert peak_rss.begin() is None
    forced.value = 100.0
    before = peak_rss.begin()
    forced.value = None  # the final reading fails
    with caplog.at_level("DEBUG"):
        assert peak_rss.growth_since(before) is None
    assert peak_rss.available() is False, "latched off for the process"
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_thread_that_cannot_start_latches_the_reader_off(forced, monkeypatch) -> None:
    def _no_thread(self) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(peak_rss._Sampler, "start", _no_thread)  # noqa: SLF001
    assert peak_rss.begin() is None
    assert peak_rss.available() is False


def test_the_sampler_availability_probe_has_no_side_effect_and_follows_the_switches(forced, monkeypatch) -> None:
    assert peak_rss.available() is True and _samplers() == []
    monkeypatch.setattr(peak_rss, "_disabled", True)
    assert peak_rss.available() is False
    monkeypatch.setattr(peak_rss, "_disabled", False)
    forced.value = None
    assert peak_rss.available() is False


def test_a_latch_or_a_warning_while_a_sampler_runs_does_not_deadlock(forced) -> None:
    before = peak_rss.begin()
    done = threading.Event()

    def _go() -> None:
        peak_rss._warn_once("probe")  # noqa: SLF001
        peak_rss.growth_since(before)
        done.set()

    worker = threading.Thread(target=_go, daemon=True)
    worker.start()
    assert done.wait(5.0), "stop() joins outside the lock; the sampler never takes it"


def test_without_the_force_linux_keeps_the_exact_reading(monkeypatch, tmp_path) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux path")
    clear_file = _fake_proc(monkeypatch, tmp_path, _STATUS)
    monkeypatch.setattr(peak_rss, "_rss_now", lambda: pytest.fail("the sampler must not be used on Linux"))
    assert peak_rss.begin() == 1000 * 1024.0 and clear_file.read_text() == "5"
    assert peak_rss.growth_since(1000 * 1024.0) == 2000 * 1024.0
    assert _samplers() == []


def test_a_call_through_rerank_timed_reports_the_sampled_peak_and_a_raising_call_leaves_no_thread(forced) -> None:
    from brain.memory.reranker import RerankerProvider

    class _Provider(RerankerProvider):
        fail = False

        def rerank(self, query, documents):
            if self.fail:
                raise RuntimeError("boom")
            forced.value = 400.0
            assert _wait_for(lambda: peak_rss._sampler is not None and peak_rss._sampler.largest == 400.0)  # noqa: SLF001
            forced.value = 120.0
            return [0.0] * len(documents)

        def model_id(self) -> str:
            return "sampled"

    provider = _Provider()
    scores, _seconds, growth = provider.rerank_timed("q", ["d"])
    assert growth == 300.0 and _samplers() == []
    provider.fail = True
    with pytest.raises(RuntimeError):
        provider.rerank_timed("q", ["d"])
    assert _samplers() == [], "the call raised: the sampler was cancelled"


def test_the_ram_bound_applies_under_the_forced_sampler(forced, monkeypatch) -> None:
    from brain.memory import reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    assert reranker_mod._ram_bound_applies() is True  # noqa: SLF001
