"""#367: a fail-soft reader warns with its traceback, at most once per key per
interval, so a persistent failure doesn't flood and a later re-break still warns."""

from __future__ import annotations

import logging

import pytest

from brain.utils import log_throttle

LOG = logging.getLogger("tests.log_throttle")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(log_throttle, "_last", {})


def _levels(caplog):
    return [r.levelno for r in caplog.records if r.name == LOG.name]


def test_first_failure_warns_then_debug_inside_the_interval(monkeypatch, caplog):
    t = [1000.0]
    monkeypatch.setattr(log_throttle.time, "monotonic", lambda: t[0])
    with caplog.at_level(logging.DEBUG, logger=LOG.name):
        log_throttle.warn_throttled(LOG, "k", "thing %s failed", "x")
        t[0] += 60
        log_throttle.warn_throttled(LOG, "k", "thing %s failed", "x")
        t[0] += log_throttle.INTERVAL_S
        log_throttle.warn_throttled(LOG, "k", "thing %s failed", "x")
    assert _levels(caplog) == [logging.WARNING, logging.DEBUG, logging.WARNING]


def test_keys_are_independent(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOG.name):
        log_throttle.warn_throttled(LOG, "a", "a failed")
        log_throttle.warn_throttled(LOG, "b", "b failed")
    assert _levels(caplog) == [logging.WARNING, logging.WARNING]


def test_the_warning_carries_the_active_traceback(caplog):
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        try:
            raise RuntimeError("root cause")
        except RuntimeError:
            log_throttle.warn_throttled(LOG, "k", "failed")
    (rec,) = caplog.records
    assert rec.exc_info and rec.exc_info[0] is RuntimeError
