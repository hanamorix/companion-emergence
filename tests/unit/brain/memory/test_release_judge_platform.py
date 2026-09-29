"""`release_judge()`'s per-platform allocator-relief step (ram-spike-fix, S27).

Linux gets `malloc_trim(0)`; macOS gets libSystem's
`malloc_zone_pressure_relief(NULL, 0)` (its allocator keeps freed pages
resident otherwise); Windows gets neither. These tests fake `sys.platform` and
`ctypes.CDLL` so every branch is checkable from any host. Whether the macOS
call actually lowers RSS is measured only by the macOS leg of
`judge-release-rss.yml`.
"""

from __future__ import annotations

import ctypes
import sys
from unittest.mock import MagicMock

import pytest

from brain.memory import relevance_judge

_LIBSYSTEM = "/usr/lib/libSystem.B.dylib"


def _install_fake_cdll(monkeypatch: pytest.MonkeyPatch, lib: MagicMock) -> MagicMock:
    cdll = MagicMock(return_value=lib)
    monkeypatch.setattr(ctypes, "CDLL", cdll)
    return cdll


def test_darwin_calls_malloc_zone_pressure_relief_with_null_zone_and_zero_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    lib = MagicMock()
    lib.malloc_zone_pressure_relief.return_value = 4096
    cdll = _install_fake_cdll(monkeypatch, lib)

    relevance_judge.release_judge()

    cdll.assert_called_once_with(_LIBSYSTEM)
    lib.malloc_zone_pressure_relief.assert_called_once_with(None, 0)
    lib.malloc_trim.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        OSError("libSystem not found"),
        AttributeError("no malloc_zone_pressure_relief"),
        ctypes.ArgumentError("bad argument"),
    ],
    ids=["cdll-oserror", "symbol-missing", "call-argument-error"],
)
def test_darwin_relief_never_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    if isinstance(failure, OSError):
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(side_effect=failure))
    elif isinstance(failure, AttributeError):
        lib = MagicMock(spec=[])  # no attributes at all -> AttributeError on lookup
        _install_fake_cdll(monkeypatch, lib)
    else:
        lib = MagicMock()
        lib.malloc_zone_pressure_relief.side_effect = failure
        _install_fake_cdll(monkeypatch, lib)

    relevance_judge.release_judge()  # must not raise


def test_darwin_release_still_clears_the_provider_cache_when_relief_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "CDLL", MagicMock(side_effect=OSError("nope")))
    relevance_judge._provider_cache["sentinel"] = object()  # type: ignore[assignment]
    try:
        relevance_judge.release_judge()
        assert relevance_judge._provider_cache == {}
    finally:
        relevance_judge._reset_judge_provider_cache()


def test_linux_uses_malloc_trim_and_not_the_macos_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    lib = MagicMock()
    cdll = _install_fake_cdll(monkeypatch, lib)

    relevance_judge.release_judge()

    cdll.assert_called_once_with("libc.so.6")
    lib.malloc_trim.assert_called_once_with(0)
    lib.malloc_zone_pressure_relief.assert_not_called()


def test_windows_attempts_no_allocator_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    cdll = _install_fake_cdll(monkeypatch, MagicMock())

    relevance_judge.release_judge()

    cdll.assert_not_called()
