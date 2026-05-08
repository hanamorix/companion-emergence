"""Runtime log rotation for the bridge runner.

launchd's StandardOut/Error and the legacy spawn_detached log file
both capture Python stdout/stderr without rotation. The runner's own
RotatingFileHandler gives us a primary log we control, capped at
~15MB total per persona (5MB × 3 backups).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from brain.bridge.runner import _setup_runtime_logging


def test_setup_runtime_logging_attaches_rotating_handler(
    tmp_path: Path, monkeypatch
) -> None:
    """Calling _setup_runtime_logging adds a RotatingFileHandler that
    points at <log_dir>/runtime-<persona>.log."""
    monkeypatch.setenv(
        "NELLBRAIN_HOME",
        str(tmp_path / "Application Support" / "companion-emergence"),
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    root = logging.getLogger()
    initial = list(root.handlers)
    try:
        _setup_runtime_logging(persona_dir)
        new_handlers = [h for h in root.handlers if h not in initial]
        rotating = [h for h in new_handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        handler = rotating[0]
        # Path lands on the persona name with the runtime- prefix.
        assert Path(handler.baseFilename).name == "runtime-nell.log"
        # 5 MB max, 3 backups (matches the constants in runner.py).
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 3
    finally:
        # Clean up: remove the handler we added so we don't leak file
        # handles or affect the rest of the test session's logging.
        for h in list(root.handlers):
            if h not in initial:
                root.removeHandler(h)
                h.close()


def test_setup_runtime_logging_swallows_failures(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Best-effort: a failed setup must not raise, only breadcrumb to stderr.

    Forces failure by pointing the log dir at a path that can't be created
    (a regular file already exists at the dir position).
    """
    blocker = tmp_path / "Application Support"
    blocker.write_text("not a directory")
    monkeypatch.setenv("NELLBRAIN_HOME", str(blocker / "companion-emergence"))
    monkeypatch.setenv("HOME", str(tmp_path))

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    # Should NOT raise.
    _setup_runtime_logging(persona_dir)
    err = capsys.readouterr().err
    assert "runtime logging setup failed" in err
