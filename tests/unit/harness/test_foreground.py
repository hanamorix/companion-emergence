"""Offline unit tests for the run-orchestration anti-stall helper (tests.harness.foreground).

Covers G1 (real boot+confirm path returns a populated ArmSession in-turn), G2 (a dead/failed
boot raises ForegroundBootError in-turn, never hangs), G3 (the real verification paths reject a
bad build / a broken tool roster in-turn), G4 (preserve_artifacts copies + raises on a missing
REQUIRED source), G5 (the banner's exact headline + required content), G6 (the shipped contract
doc names both surfaces + the one rule), and G7 (every new public symbol is exported from
tests.harness and listed in __all__). Mechanical facts only: no live bridge, no tokens, no
network -- runs under the default (non-live) marker set.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from tests.harness import foreground as fg

RUN_ORCHESTRATION_MD = Path(__file__).resolve().parents[3] / "tests" / "harness" / "RUN-ORCHESTRATION.md"


def _write_ready(run_dir: Path, payload: dict) -> None:
    (run_dir / "READY").write_text(json.dumps(payload), encoding="utf-8")


# --- G1: real boot+confirm path returns a populated ArmSession in-turn ----------------------------


def test_boot_and_verify_real_boot_cmd_returns_populated_session(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ready_path = run_dir / "READY"
    boot_cmd = [
        sys.executable,
        "-c",
        f"import json; open({str(ready_path)!r}, 'w').write(json.dumps({{'brain_repo': 'x'}}))",
    ]
    spec = fg.ArmBootSpec(arm="A", port=8931, run_dir=run_dir, ready_timeout=15.0)

    session = fg.boot_and_verify(spec, boot_cmd=boot_cmd)

    assert isinstance(session, fg.ArmSession)
    assert session.arm == "A"
    assert session.port == 8931
    assert session.run_dir == run_dir
    assert session.ready_payload == {"brain_repo": "x"}
    assert session.brain_repo == "x"


def test_boot_and_verify_poll_mode_picks_up_fresh_ready(tmp_path: Path) -> None:
    """boot_cmd=None: bounded poll for a READY written by an already-foregrounded external boot.

    The READY is written from a background thread AFTER the poll's own start timestamp is
    captured (a short delay), mirroring an external ``run_arm.sh``-style boot that writes READY
    while the orchestrator is already polling -- the case the stale-READY guard must NOT reject.
    """
    import threading

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _delayed_write() -> None:
        time.sleep(0.3)
        _write_ready(run_dir, {"brain_repo": "y"})

    threading.Thread(target=_delayed_write, daemon=True).start()
    spec = fg.ArmBootSpec(arm="B", port=8932, run_dir=run_dir, ready_timeout=5.0)

    session = fg.boot_and_verify(spec, boot_cmd=None)

    assert session.ready_payload == {"brain_repo": "y"}


def test_boot_and_verify_poll_mode_ignores_stale_ready(tmp_path: Path) -> None:
    """N3: a READY predating the poll start must be ignored, not read as a false confirmation."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_ready(run_dir, {"brain_repo": "stale"})
    # Backdate the READY file's mtime so it predates the poll's captured start timestamp.
    stale_mtime = time.time() - 3600
    import os

    os.utime(run_dir / "READY", (stale_mtime, stale_mtime))

    spec = fg.ArmBootSpec(arm="C", port=8933, run_dir=run_dir, ready_timeout=0.5)
    with pytest.raises(fg.ForegroundBootError, match="no READY"):
        fg.boot_and_verify(spec, boot_cmd=None)


# --- G2: a dead/failed boot raises in-turn, does not hang -----------------------------------------


def test_boot_and_verify_nonzero_exit_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    boot_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
    spec = fg.ArmBootSpec(arm="A", port=1, run_dir=run_dir, ready_timeout=10.0)

    with pytest.raises(fg.ForegroundBootError, match="exited 1"):
        fg.boot_and_verify(spec, boot_cmd=boot_cmd)


def test_boot_and_verify_error_marker_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    error_path = run_dir / "ERROR"
    boot_cmd = [
        sys.executable,
        "-c",
        f"open({str(error_path)!r}, 'w').write('boom')",
    ]
    spec = fg.ArmBootSpec(arm="A", port=1, run_dir=run_dir, ready_timeout=10.0)

    with pytest.raises(fg.ForegroundBootError, match="error marker"):
        fg.boot_and_verify(spec, boot_cmd=boot_cmd)


def test_boot_and_verify_no_ready_times_out_quickly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = fg.ArmBootSpec(arm="A", port=1, run_dir=run_dir, ready_timeout=0.5)

    start = time.monotonic()
    with pytest.raises(fg.ForegroundBootError, match="no READY"):
        fg.boot_and_verify(spec, boot_cmd=None)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"boot_and_verify hung: took {elapsed}s for a 0.5s ready_timeout"


def test_boot_and_verify_timeout_expired_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    boot_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    spec = fg.ArmBootSpec(arm="A", port=1, run_dir=run_dir, ready_timeout=0.5)

    start = time.monotonic()
    with pytest.raises(fg.ForegroundBootError, match="did not complete"):
        fg.boot_and_verify(spec, boot_cmd=boot_cmd)
    elapsed = time.monotonic() - start

    assert elapsed < 10.0, f"boot_and_verify hung past its own bound: {elapsed}s"


# --- G3(a): expect_brain_repo_under cross-check rejects a bad build -------------------------------


def _boot_cmd_writing_ready(ready_path: Path, payload: dict) -> list[str]:
    """A real, foreground boot_cmd that writes a READY payload -- the G1 real path, reused here
    so the brain_repo cross-check is exercised via the actual boot_cmd branch (no poll-mode
    staleness concerns, since boot_cmd mode does not apply the poll freshness guard)."""
    return [
        sys.executable,
        "-c",
        f"import json; open({str(ready_path)!r}, 'w').write(json.dumps({payload!r}))",
    ]


def test_boot_and_verify_brain_repo_outside_expected_root_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    other_repo = tmp_path / "other-repo"
    other_repo.mkdir()

    expected_root = tmp_path / "expected-repo"
    expected_root.mkdir()
    spec = fg.ArmBootSpec(
        arm="A", port=1, run_dir=run_dir, ready_timeout=5.0, expect_brain_repo_under=expected_root
    )
    boot_cmd = _boot_cmd_writing_ready(run_dir / "READY", {"brain_repo": str(other_repo)})

    with pytest.raises(fg.ForegroundBootError, match="NOT under"):
        fg.boot_and_verify(spec, boot_cmd=boot_cmd)


def test_boot_and_verify_brain_repo_under_expected_root_passes(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected-repo"
    expected_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nested = expected_root / "checkout"
    nested.mkdir()

    spec = fg.ArmBootSpec(
        arm="A", port=1, run_dir=run_dir, ready_timeout=5.0, expect_brain_repo_under=expected_root
    )
    boot_cmd = _boot_cmd_writing_ready(run_dir / "READY", {"brain_repo": str(nested)})

    session = fg.boot_and_verify(spec, boot_cmd=boot_cmd)
    assert session.brain_repo == str(nested)


# --- G3(b): assert_tool_callable rejects an absent tool, passes a present one ---------------------


def _turn_diag_row(system_text: str) -> str:
    return json.dumps({"sent": {"system": system_text}})


def test_assert_tool_callable_raises_when_tool_absent(tmp_path: Path) -> None:
    diag = tmp_path / "turn_diag.jsonl"
    diag.write_text(_turn_diag_row("Available tools: search_memories, list_works") + "\n")

    with pytest.raises(fg.ToolSideBroken, match="read_full_memory"):
        fg.assert_tool_callable(diag, "read_full_memory")


def test_assert_tool_callable_passes_when_tool_present(tmp_path: Path) -> None:
    diag = tmp_path / "turn_diag.jsonl"
    diag.write_text(
        _turn_diag_row("Available tools: search_memories, read_full_memory, list_works") + "\n"
    )

    fg.assert_tool_callable(diag, "read_full_memory")  # must not raise


def test_assert_tool_callable_accepts_a_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text(
        _turn_diag_row("Available tools: read_full_memory") + "\n"
    )

    fg.assert_tool_callable(run_dir, "read_full_memory")  # must not raise


def test_assert_tool_callable_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(fg.ToolSideBroken):
        fg.assert_tool_callable(tmp_path / "does-not-exist.jsonl", "read_full_memory")


# --- G4: preserve_artifacts copies, reports missing, raises on a missing REQUIRED source ----------


def test_preserve_artifacts_copies_files_and_subproc_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")
    (run_dir / "transcript.jsonl").write_text("{}\n")
    subproc = run_dir / "mcp-child"
    subproc.mkdir()
    (subproc / "log.txt").write_text("hello\n")

    result = fg.preserve_artifacts(run_dir, subproc_dirs=("mcp-child",))

    assert isinstance(result, fg.PreserveResult)
    assert result.dest == run_dir / "valid-run"
    assert (result.dest / "turn_diag.jsonl").exists()
    assert (result.dest / "transcript.jsonl").exists()
    assert (result.dest / "mcp-child" / "log.txt").read_text() == "hello\n"
    assert result.missing == []
    assert len(result.copied) == 3


def test_preserve_artifacts_reports_nonrequired_missing_without_raising(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")
    # transcript.jsonl deliberately absent, and NOT required.

    result = fg.preserve_artifacts(run_dir)

    assert (result.dest / "turn_diag.jsonl").exists()
    assert run_dir / "transcript.jsonl" in result.missing
    assert (result.dest / "transcript.jsonl") not in result.copied


def test_preserve_artifacts_missing_required_source_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")
    # transcript.jsonl absent AND required this time.

    with pytest.raises(fg.PreservationIncomplete, match="transcript.jsonl"):
        fg.preserve_artifacts(run_dir, require=("transcript.jsonl",))


def test_preserve_artifacts_missing_required_subproc_dir_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")
    (run_dir / "transcript.jsonl").write_text("{}\n")

    with pytest.raises(fg.PreservationIncomplete, match="mcp-child"):
        fg.preserve_artifacts(run_dir, subproc_dirs=("mcp-child",), require=("mcp-child",))


def test_preserve_artifacts_require_entry_absent_from_names_still_raises(tmp_path: Path) -> None:
    """A `require=` entry that is NOT also listed in `names=`/`subproc_dirs=` must still be
    enforced -- it must never be silently skipped just because it was never attempted. This
    guards against the enforcement gap where `require` was only checked INSIDE the copy loops:
    `transcript.jsonl` exists on disk but is deliberately left out of `names=`, so the pre-fix
    code would return green with `missing=[]` instead of raising."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")
    (run_dir / "transcript.jsonl").write_text("{}\n")  # present on disk, NOT in names=

    with pytest.raises(fg.PreservationIncomplete, match="transcript.jsonl"):
        fg.preserve_artifacts(
            run_dir,
            names=("turn_diag.jsonl",),
            require=("transcript.jsonl",),
        )


def test_preserve_artifacts_require_entry_in_names_and_present_copies_cleanly(
    tmp_path: Path,
) -> None:
    """Complementary case: a `require=` entry that IS listed in `names=` and present on disk
    copies cleanly (no raise) -- the enforcement fix must not break the normal path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")

    result = fg.preserve_artifacts(
        run_dir,
        names=("turn_diag.jsonl",),
        require=("turn_diag.jsonl",),
    )

    assert (result.dest / "turn_diag.jsonl").exists()
    assert result.dest / "turn_diag.jsonl" in result.copied
    assert result.missing == []


def test_preserve_artifacts_custom_dest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "turn_diag.jsonl").write_text("{}\n")
    dest = tmp_path / "elsewhere"

    result = fg.preserve_artifacts(run_dir, dest=dest, names=("turn_diag.jsonl",))

    assert result.dest == dest
    assert (dest / "turn_diag.jsonl").exists()


# --- G5: the banner's exact headline + required content -------------------------------------------


def test_drive_now_banner_first_line_is_exact_headline(tmp_path: Path) -> None:
    banner = fg.drive_now_banner(arm="A", port=8931, run_dir=tmp_path / "run")
    first_line = banner.splitlines()[0]
    assert first_line == fg.DRIVE_NOW_BANNER_HEADLINE
    assert first_line == "▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to wait"


def test_drive_now_banner_names_both_surfaces_and_the_rule() -> None:
    banner = fg.drive_now_banner(arm="A", port=8931, run_dir="/tmp/run")
    assert "run_in_background" in banner
    assert "background sub-agent" in banner
    assert "one arm = one turn" in banner
    assert "foreground" in banner.lower()


def test_drive_now_banner_includes_run_context_and_roster_ok() -> None:
    banner = fg.drive_now_banner(arm="my-arm", port=9999, run_dir="/tmp/xyz", roster_ok=True)
    assert "my-arm" in banner
    assert "9999" in banner
    assert "/tmp/xyz" in banner
    assert "roster_ok=True" in banner


def test_drive_now_banner_extra_lines_appended() -> None:
    banner = fg.drive_now_banner(
        arm="A", port=1, run_dir="/tmp/run", extra_lines=["custom note here"]
    )
    assert "custom note here" in banner


# --- G6: the shipped contract doc names both surfaces + the one rule ------------------------------


def test_run_orchestration_doc_exists() -> None:
    assert RUN_ORCHESTRATION_MD.exists(), f"expected {RUN_ORCHESTRATION_MD} to exist"


def test_run_orchestration_doc_names_both_surfaces_and_the_rule() -> None:
    text = RUN_ORCHESTRATION_MD.read_text(encoding="utf-8")
    assert "run_in_background" in text
    assert "background sub-agent" in text
    assert "one arm = one" in text.lower() or "one arm = one agent = one turn" in text
    assert "FOREGROUND" in text


# --- G7: every new public symbol is exported from tests.harness and listed in __all__ -------------


def test_all_new_symbols_importable_and_in_all() -> None:
    import tests.harness as harness

    names = [
        "ArmBootSpec",
        "ArmSession",
        "PreserveResult",
        "ForegroundBootError",
        "ToolSideBroken",
        "PreservationIncomplete",
        "boot_and_verify",
        "assert_tool_callable",
        "preserve_artifacts",
        "drive_now_banner",
        "DRIVE_NOW_BANNER_HEADLINE",
    ]
    for name in names:
        assert hasattr(harness, name), f"tests.harness has no attribute {name!r}"
        assert name in harness.__all__, f"{name!r} missing from tests.harness.__all__"
