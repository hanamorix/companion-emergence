"""G8 — the ModelConfig toggle threads through DumbBob's argv + the watchdog ping (no tokens).

Positive per-site assertion: a non-default ``ModelConfig`` appears in the composed argv. DumbBob's
argv is built WITHOUT spawning (``build_argv``) so no ``claude -p`` runs.
"""

from __future__ import annotations

from pathlib import Path

from tests.harness import DumbBob, ModelConfig, Watchdog, real_ping_fn, watchdog_ping_argv
from tests.harness.config import DEFAULT_TIMEOUTS


def test_dumbbob_argv_uses_configured_model() -> None:
    models = ModelConfig(canary="opus", bob="opus", watchdog="haiku")
    bob = DumbBob("/fake/claude", mood="testing", models=models)
    argv = bob.build_argv("hello")
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "opus"
    # default (sonnet) is NOT present — the toggle actually replaced it.
    assert "sonnet" not in argv


def test_dumbbob_default_model_is_sonnet() -> None:
    bob = DumbBob("/fake/claude", mood="testing")
    argv = bob.build_argv("hi")
    assert argv[argv.index("--model") + 1] == "sonnet"


def test_watchdog_ping_uses_configured_model(tmp_path: Path) -> None:
    """F1: assert against the REAL argv the watchdog ping builds — not a hand-rebuilt copy.

    This can fail if the model were hardcoded, because ``watchdog_ping_argv`` is the same function
    ``real_ping_fn`` uses to construct its subprocess argv.
    """
    models = ModelConfig(watchdog="opus")
    argv = watchdog_ping_argv("/fake/claude", models)
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "opus"
    assert "haiku" not in argv  # the default was actually replaced
    # And the watchdog carries the configured models through.
    fn = real_ping_fn("/fake/claude", models, DEFAULT_TIMEOUTS)
    wd = Watchdog(fn, tmp_path / "marker", models=models)
    assert wd.models.watchdog == "opus"
