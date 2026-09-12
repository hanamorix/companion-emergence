"""#246 — process-wide auth-expiry state: classifier, debounce, probe, transition logging."""

from __future__ import annotations

import logging
import threading

import pytest

from brain.bridge import provider_auth

LIVE = "Failed to authenticate: OAuth session expired and could not be refreshed"
WEAK = "Not logged in · Please run /login"


@pytest.fixture(autouse=True)
def _reset():
    provider_auth.reset()
    yield
    provider_auth.reset()


def _clock(monkeypatch, start: float = 1000.0):
    t = {"now": start}
    monkeypatch.setattr(provider_auth, "_monotonic", lambda: t["now"])
    return t


# C1 — classifier
@pytest.mark.parametrize("text", [LIVE, WEAK, "Failed to authenticate", "OAuth session expired and could not be refreshed"])
def test_is_auth_failure_true_for_real_strings(text):
    assert provider_auth.is_auth_failure(text) is True


@pytest.mark.parametrize("text", ["api_error_status=429; is_error=True; You've hit your session limit", "error_max_budget_usd", ""])
def test_is_auth_failure_false_for_non_auth(text):
    assert provider_auth.is_auth_failure(text) is False


def test_strong_classifier_independent_of_weak_match():
    assert provider_auth.is_strong_auth_failure(LIVE) is True
    assert provider_auth.is_strong_auth_failure(WEAK) is False


# C16 — debounce
def test_single_weak_hit_does_not_flip(monkeypatch):
    _clock(monkeypatch)
    provider_auth.note_cli_failure(WEAK)
    assert provider_auth.state()["status"] == "ok"


def test_two_weak_hits_60s_apart_flip(monkeypatch):
    t = _clock(monkeypatch)
    provider_auth.note_cli_failure(WEAK)
    t["now"] += 61
    provider_auth.note_cli_failure(WEAK)
    assert provider_auth.state()["status"] == "expired"


def test_two_weak_hits_close_together_do_not_flip(monkeypatch):
    t = _clock(monkeypatch)
    provider_auth.note_cli_failure(WEAK)
    t["now"] += 5
    provider_auth.note_cli_failure(WEAK)
    assert provider_auth.state()["status"] == "ok"


def test_strong_live_text_flips_immediately(monkeypatch):
    _clock(monkeypatch)
    provider_auth.note_cli_failure(LIVE)
    s = provider_auth.state()
    assert s["status"] == "expired"
    assert s["since"] is not None
    assert "OAuth session expired" in s["detail"]


def test_success_between_weak_hits_resets_pending(monkeypatch):
    t = _clock(monkeypatch)
    provider_auth.note_cli_failure(WEAK)
    provider_auth.note_cli_success()
    t["now"] += 61
    provider_auth.note_cli_failure(WEAK)
    assert provider_auth.state()["status"] == "ok"


def test_success_clears_expired(monkeypatch):
    _clock(monkeypatch)
    provider_auth.note_cli_failure(LIVE)
    provider_auth.note_cli_success()
    s = provider_auth.state()
    assert s["status"] == "ok" and s["since"] is None and s["failures"] == 0


def test_non_auth_failure_is_a_noop(monkeypatch):
    _clock(monkeypatch)
    provider_auth.note_cli_failure("api_error_status=429; session limit")
    assert provider_auth.state()["status"] == "ok"


# C4 (probe half) — one grant per interval; transition stamps the probe
def test_probe_one_grant_per_interval(monkeypatch):
    t = _clock(monkeypatch)
    provider_auth.note_cli_failure(LIVE)
    # transition stamped the probe: the very next call is deferred
    assert provider_auth.should_skip_background() is True
    t["now"] += provider_auth.REPROBE_S + 1
    assert provider_auth.should_skip_background() is False  # the probe
    assert all(provider_auth.should_skip_background() for _ in range(9))
    t["now"] += provider_auth.REPROBE_S + 1
    assert provider_auth.should_skip_background() is False


def test_should_skip_is_false_when_ok(monkeypatch):
    _clock(monkeypatch)
    assert provider_auth.should_skip_background() is False


# C6 — one log line per transition
def test_logs_once_per_transition(monkeypatch, caplog):
    _clock(monkeypatch)
    with caplog.at_level(logging.INFO, logger="brain.bridge.provider_auth"):
        for _ in range(20):
            provider_auth.note_cli_failure(LIVE)
        provider_auth.note_cli_success()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(warns) == 1
    assert len(infos) == 1
    assert provider_auth.state()["failures"] == 0


# C7 — executed interleaving under one lock
def test_threaded_interleaving_keeps_state_consistent(monkeypatch):
    _clock(monkeypatch)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for _ in range(50):
                provider_auth.note_cli_failure(LIVE)
                provider_auth.note_cli_success()
                provider_auth.should_skip_background()
                provider_auth.state()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    s = provider_auth.state()
    assert errors == []
    assert s["status"] in ("ok", "expired")
    assert s["failures"] >= 0
