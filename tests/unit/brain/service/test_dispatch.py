"""Tests for the OS-service dispatcher in :mod:`brain.service`.

The dispatcher returns the right backend module per platform. All
three backends (launchd / systemd / windows_service) are real
implementations as of v0.0.1. The exact-equality assertions on
module name pin the dispatcher's contract regardless of what each
backend's surface looks like.
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
    assert hasattr(backend, "install_service")
    assert hasattr(backend, "uninstall_service")
    assert hasattr(backend, "service_status")
    assert hasattr(backend, "doctor_checks")
    assert hasattr(backend, "build_launchd_plist_xml")


def test_current_backend_returns_windows_service_on_windows(monkeypatch) -> None:
    """Windows gets the real Task Scheduler backend (graduated from
    stub on 2026-05-08 — task XML generation + schtasks wrappers)."""
    monkeypatch.setattr(sys, "platform", "win32")
    backend = current_backend()
    assert backend.__name__ == "brain.service.windows_service"
    assert hasattr(backend, "install_service")
    assert hasattr(backend, "uninstall_service")
    assert hasattr(backend, "service_status")
    assert hasattr(backend, "doctor_checks")
    assert hasattr(backend, "build_launchd_plist_xml")


def test_current_backend_unknown_platform_raises(monkeypatch) -> None:
    """Anything outside darwin/linux/win* raises with a hint."""
    monkeypatch.setattr(sys, "platform", "freebsd13")
    with pytest.raises(UnsupportedPlatformError) as exc:
        current_backend()
    assert "freebsd13" in str(exc.value)
    assert "plan" in str(exc.value).lower()
