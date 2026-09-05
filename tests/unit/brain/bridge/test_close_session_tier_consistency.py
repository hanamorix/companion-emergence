"""C9 (#154): close_session's extraction must resolve to the SAME tier as the
idle-sweep snapshot/finalize ticks — closing the stage-3 iteration-1 MAJOR finding
that leaving `close_session`'s caller unrouted would create a trigger-dependent
quality inconsistency in the identical underlying extraction operation
(`extract_items_with_status`, shared by `close_session` and
`extract_session_snapshot`).

Full end-to-end driving of the FastAPI `/sessions/close` route and the
supervisor's background tick loop is heavy relative to what's being asserted here
(which model TIER each call site names) — this test instead verifies, by source
inspection of the actual shipped code, that all three call sites
(`server.py`'s `/sessions/close` handler, `supervisor.py`'s session-snapshot tick,
`supervisor.py`'s finalize tick) name the identical tier constant,
`TIER_BACKGROUND_HOUSEKEEPING`. Combined with
`test_model_tier.py::test_build_tier_provider_constructs_claude_cli_provider_with_the_tier_model`
(which proves that constant resolves to a real Haiku `ClaudeCliProvider`), this
closes the loop from "same tier named" to "same model actually constructed."
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(relpath: str) -> str:
    path = _REPO_ROOT / "brain" / relpath
    assert path.exists(), f"expected file missing: {path}"
    return path.read_text(encoding="utf-8")


def _tier_passed_to_close_session_blocking(server_src: str) -> str:
    """Extract the tier constant used in the /sessions/close route's
    build_tier_provider(...) call feeding _close_session_blocking."""
    m = re.search(
        r"_close_session_blocking,\s*\n\s*s\.persona_dir,\s*\n\s*sid,\s*\n\s*"
        r"build_tier_provider\(s\.persona_dir,\s*([A-Z_]+)\)",
        server_src,
    )
    assert m is not None, "could not find the /sessions/close route's build_tier_provider(...) call"
    return m.group(1)


def _tiers_passed_to_snapshot_and_finalize(supervisor_src: str) -> tuple[str, str]:
    snapshot_m = re.search(
        r"reports = snapshot_stale_sessions\(\s*\n"
        r"\s*persona_dir,\s*\n"
        r"\s*silence_minutes=silence_minutes,\s*\n"
        r"\s*store=store,\s*\n"
        r"\s*hebbian=hebbian,\s*\n"
        r"\s*provider=build_tier_provider\(persona_dir,\s*([A-Z_]+)\)",
        supervisor_src,
    )
    assert snapshot_m is not None, "could not find snapshot_stale_sessions' provider= call"

    finalize_m = re.search(
        r"_run_finalize_tick\(\s*\n"
        r"\s*persona_dir,\s*\n"
        r"\s*build_tier_provider\(persona_dir,\s*([A-Z_]+)\)",
        supervisor_src,
    )
    assert finalize_m is not None, "could not find _run_finalize_tick's build_tier_provider(...) call"

    return snapshot_m.group(1), finalize_m.group(1)


def test_close_session_and_idle_sweep_ticks_all_name_the_same_tier():
    server_src = _read("bridge/server.py")
    supervisor_src = _read("bridge/supervisor.py")

    close_session_tier = _tier_passed_to_close_session_blocking(server_src)
    snapshot_tier, finalize_tier = _tiers_passed_to_snapshot_and_finalize(supervisor_src)

    assert close_session_tier == snapshot_tier == finalize_tier == "TIER_BACKGROUND_HOUSEKEEPING"


def test_daemon_recovery_also_names_the_same_tier():
    """The dirty-shutdown recovery path (daemon.py) independently constructs its
    own provider (not threaded from supervisor/server) but feeds the identical
    snapshot_stale_sessions extraction — must also resolve to housekeeping."""
    daemon_src = _read("bridge/daemon.py")
    m = re.search(r"provider = build_tier_provider\(persona_dir,\s*([A-Z_]+)\)", daemon_src)
    assert m is not None, "could not find daemon.py's build_tier_provider(...) call"
    assert m.group(1) == "TIER_BACKGROUND_HOUSEKEEPING"
