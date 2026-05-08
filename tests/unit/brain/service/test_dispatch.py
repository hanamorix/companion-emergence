"""Tests for the OS-service dispatcher in :mod:`brain.service`.

The dispatcher returns the right backend module per platform.
macOS + Linux backends are real; Windows is still a stub. The
exact-equality assertions on module name pin the dispatcher's
contract regardless of what each backend's surface looks like.
"""

from __future__ import annotations

import sys

import pytest

from brain.service import UnsupportedPlatformError, current_backend


def test_current_backend_returns_launchd_on_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = current_backend()
    assert backend.__name__ == "brain.service.launchd"


def test_current_backend_returns_systemd_on_linux(monkeypatch) -> None:
    """Linux gets the real systemd-user backend (graduated from stub
    on 2026-05-08 — unit-file generation + systemctl wrappers)."""
    monkeypatch.setattr(sys, "platform", "linux")
    backend = current_backend()
    assert backend.__name__ == "brain.service.systemd"
    # Surface symbols the dispatcher relies on are present.
    assert hasattr(backend, "install_service")
    assert hasattr(backend, "uninstall_service")
    assert hasattr(backend, "service_status")
    assert hasattr(backend, "doctor_checks")
    assert hasattr(backend, "build_launchd_plist_xml")  # alias for unified CLI


def test_current_backend_returns_windows_stub_on_windows(monkeypatch) -> None:
    """Windows is still a stub — every entry point raises
    UnsupportedPlatformError until the SCM / Task Scheduler backend
    lands."""
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
