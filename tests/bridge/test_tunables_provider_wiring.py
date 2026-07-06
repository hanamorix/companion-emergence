"""Spec §5 canary 5: an override actually changes the streaming loop's
timeout — through the provider path, not the getter in isolation."""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tunables_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield tmp_path
    t._reset_for_tests()


def test_stream_timeouts_honor_overrides(tunables_home):
    (tunables_home / "tunables.json").write_text(
        json.dumps({"defaults": {}, "overrides": {
            "provider.stream_per_event_idle_seconds": 240.0,
            "provider.stream_first_event_seconds": 300.0,
        }}),
        encoding="utf-8",
    )
    from brain.bridge import provider as p

    assert p._stream_per_event_idle_seconds() == 240.0
    assert p._stream_first_event_seconds() == 300.0


def test_stream_timeouts_default_without_file(tunables_home):
    from brain.bridge import provider as p

    assert p._stream_per_event_idle_seconds() == 60.0
    assert p._stream_first_event_seconds() == 120.0
