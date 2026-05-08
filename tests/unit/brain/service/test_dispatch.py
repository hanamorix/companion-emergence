"""Tests for the OS-service dispatcher in :mod:`brain.service`.

Linux + Windows backends are stubs today; the dispatcher's job is to
return the *right* module for the active OS so the CLI can surface a
clean ``UnsupportedPlatformError`` message instead of a launchctl /
SCM stack trace deeper in.
"""

from __future__ import annotations

import sys

import pytest

from brain.service import UnsupportedPlatformError, current_backend


def test_current_backend_returns_launchd_on_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = current_backend()
    assert backend.__name__ == "brain.service.launchd"


def test_current_backend_returns_systemd_stub_on_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    backend = current_backend()
    assert backend.__name__ == "brain.service.systemd"
    # Stub: every entry point raises UnsupportedPlatformError.
    with pytest.raises(UnsupportedPlatformError):
        backend.install_service(persona="x", nell_path="/nell")
    with pytest.raises(UnsupportedPlatformError):
        backend.doctor_checks(persona="x")


def test_current_backend_returns_windows_stub_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    backend = current_backend()
    assert backend.__name__ == "brain.service.windows_service"
    with pytest.raises(UnsupportedPlatformError):
        backend.install_service(persona="x", nell_path="C:\\nell.exe")


def test_current_backend_unknown_platform_raises(monkeypatch) -> None:
    """Anything outside darwin/linux/win* raises with a hint."""
    monkeypatch.setattr(sys, "platform", "freebsd13")
    with pytest.raises(UnsupportedPlatformError) as exc:
        current_backend()
    assert "freebsd13" in str(exc.value)
    assert "plan" in str(exc.value).lower()
