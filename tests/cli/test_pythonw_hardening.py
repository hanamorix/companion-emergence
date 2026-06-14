"""Bug 1 (v0.0.36): under pythonw.exe (Windows Task Scheduler, no console)
sys.stdout/stderr are None; a bare print() in the bridge boot path raised and
killed the process before the server bound — the task could never start the
bridge. main() now hardens the std streams + logs any crash to a file."""
from __future__ import annotations

import sys

import brain.cli as cli


def test_harden_replaces_none_stdout_stderr(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    cli._harden_std_streams()
    # both are now writable — a bare print no longer raises
    assert sys.stdout is not None and sys.stderr is not None
    sys.stdout.write("x")  # must not raise
    sys.stderr.write("y")
    print("boot status line")  # the exact failure mode — must be a no-op now


def test_harden_leaves_real_streams_untouched(monkeypatch):
    import io

    real = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real)
    cli._harden_std_streams()
    assert sys.stdout is real  # don't clobber a valid console


def test_main_logs_crash_to_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "get_log_dir", lambda: tmp_path)

    def _boom(argv=None):
        raise RuntimeError("boot exploded")

    monkeypatch.setattr(cli, "_run_main", _boom)
    try:
        cli.main(["supervisor", "run", "--persona", "x"])
    except RuntimeError:
        pass  # re-raised after logging, by design
    crash = tmp_path / "cli-crash.log"
    assert crash.exists()
    assert "boot exploded" in crash.read_text()
