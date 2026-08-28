"""Offline unit tests for the run-level roster preflight (tests.harness.roster_preflight).

Covers C3 (passes on a live child + B-1 self-check + in-process ordering), C4 (hard-fail on a
dead/unresponsive child — the F5 catch), C5 (hard-fail on a roster mismatch), C6 (declared roster
from the copy's registry, no hardcoded tool name), C7 (child-argv mirrors the shared helper), and
C9 (a tools/call under -P dispatches). All offline / network-free: a minimal seeded persona and a
real `python -P -m brain.mcp_server` child (brain + mcp are already installed in the worktree
venv), mirroring tests/unit/brain/mcp_server/test_server.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import brain
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools import NELL_TOOL_NAMES
from brain.tools.schemas import build_schemas
from tests.harness import roster_preflight as rp
from tests.harness.dropin import DropinBuild, DropinMismatch

# The worktree repo root (contains brain/) — the copy root for an in-place run.
REPO_ROOT = Path(brain.__file__).resolve().parent.parent
RP_SRC = Path(rp.__file__).read_text(encoding="utf-8")


def _worktree_build() -> DropinBuild:
    return DropinBuild(repo=REPO_ROOT, python=Path(sys.executable), source=REPO_ROOT)


def _seed_persona(tmp_path: Path) -> Path:
    d = tmp_path / "persona"
    d.mkdir()
    MemoryStore(db_path=d / "memories.db").close()
    HebbianMatrix(db_path=d / "hebbian.db").close()
    return d


# --- C6: declared roster from the copy's registry, no hardcoded tool name ------------------------


def test_declared_roster_matches_server_rule(tmp_path: Path) -> None:
    persona = _seed_persona(tmp_path)
    schemas = build_schemas(persona.name)
    expected = {n for n in NELL_TOOL_NAMES if n in schemas}
    assert rp._declared_roster(persona) == expected


def test_module_imports_the_registry_symbols() -> None:
    assert "from brain.tools import NELL_TOOL_NAMES" in RP_SRC
    assert "build_schemas" in RP_SRC


def test_no_hardcoded_tool_name_in_source() -> None:
    for name in NELL_TOOL_NAMES:
        assert f'"{name}"' not in RP_SRC and f"'{name}'" not in RP_SRC, (
            f"tool name {name!r} is hardcoded in the preflight — it must be read from the copy"
        )


def test_absence_sweep_oracle_can_fail() -> None:
    """Self-test (ST1.5f): the sweep fires on a seeded literal."""
    seeded = 'x = "search_memories"\n'
    assert any(f'"{n}"' in seeded for n in NELL_TOOL_NAMES)


# --- C7: child argv mirrors the shared helper ----------------------------------------------------


def test_child_argv_mirrors_provider_helper(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    from brain.bridge.provider import brain_tools_mcp_entry

    entry = brain_tools_mcp_entry(persona)
    assert rp._child_argv(persona) == [entry["command"], *entry["args"]]


def test_child_argv_mirror_oracle_can_fail(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    bogus = [sys.executable, "-m", "brain.mcp_server", "--persona-dir", str(persona)]  # no -P
    assert rp._child_argv(persona) != bogus


# --- B-1: the preflight self-enforces its containment precondition --------------------------------


def test_declared_authoritative_passes_for_worktree() -> None:
    rp._assert_declared_authoritative(_worktree_build())  # in-process brain IS under the repo root


def test_declared_authoritative_raises_for_bogus_copy(tmp_path: Path) -> None:
    bogus = DropinBuild(repo=tmp_path / "not-the-copy", python=Path(sys.executable), source=tmp_path)
    with pytest.raises(DropinMismatch, match="precondition failed"):
        rp._assert_declared_authoritative(bogus)


# --- C4: hard-fail on a dead / unresponsive child (the F5 catch) ----------------------------------


def test_live_roster_raises_on_exited_child() -> None:
    dead = [sys.executable, "-P", "-c", "import sys; sys.exit(70)"]
    with pytest.raises(DropinMismatch, match="did not serve tools/list"):
        rp._live_roster(dead, timeout=10.0)


def test_live_roster_raises_on_nonhandshaking_child() -> None:
    hang = [sys.executable, "-P", "-c", "import time; time.sleep(60)"]
    with pytest.raises(DropinMismatch, match="did not serve tools/list"):
        rp._live_roster(hang, timeout=2.0)


# --- C3: passes on a correctly-wired live child; ordering exercised in-process --------------------


def test_preflight_passes_on_live_child(tmp_path: Path) -> None:
    persona = _seed_persona(tmp_path)
    build = _worktree_build()
    # No raise: runs _assert_declared_authoritative THEN the live spawn (the ordering the
    # un-applied call-site would own — here exercised in-process).
    rp.assert_brain_tools_roster(build, persona, timeout=30.0)
    # Evidence: the served roster equals the declared roster.
    assert rp._live_roster(rp._child_argv(persona), timeout=30.0) == rp._declared_roster(persona)


# --- C5: hard-fail on a served-vs-declared roster mismatch ----------------------------------------


def test_preflight_raises_on_roster_mismatch(tmp_path: Path, monkeypatch) -> None:
    persona = _seed_persona(tmp_path)
    build = _worktree_build()
    declared = rp._declared_roster(persona)
    dropped = sorted(declared)[0]
    served = declared - {dropped}
    monkeypatch.setattr(rp, "_live_roster", lambda *a, **k: served)
    with pytest.raises(DropinMismatch, match="served roster != declared roster"):
        rp.assert_brain_tools_roster(build, persona)


# --- C9: a tools/call under -P dispatches (production-hardening claim, not just tools/list) --------


def _call_tool(argv: list[str], name: str, arguments: dict, *, cwd: str | None = None):
    import os

    async def _go():
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=argv[0], args=list(argv[1:]), env=dict(os.environ), cwd=cwd
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    return asyncio.run(asyncio.wait_for(_go(), 30.0))


def test_tools_call_under_safe_path_dispatches(tmp_path: Path) -> None:
    persona = _seed_persona(tmp_path)
    argv = rp._child_argv(persona)  # carries -P
    assert argv[1] == "-P"
    result = _call_tool(argv, "list_works", {})
    # A working dispatch: not an MCP error frame ("no such tool" / import failure would set this).
    assert result.isError is not True, getattr(result, "content", result)
    assert result.content, "expected a tool result payload"


def test_tools_call_under_broken_shadow_fails(tmp_path: Path) -> None:
    """C9 discrimination (ST1.5f): a tools/call whose child resolves a BROKEN shadowed `brain`
    fails (no result), so the positive assertion above distinguishes a working dispatch from a
    broken one — not just an always-passing call.

    Construction mirrors the real bug: from a cwd holding a fake top-level `brain/` that raises on
    import, launch the child WITHOUT `-P` (the pre-#138 argv), so `-m brain.mcp_server` imports the
    broken shadow, the child dies at startup, and the MCP handshake / tools/call gets no result.
    The `-P` in the positive test's argv is exactly what prevents this — the pair is the oracle
    shown able to fail.
    """
    persona = _seed_persona(tmp_path)
    shadow = tmp_path / "shadow"
    (shadow / "brain").mkdir(parents=True)
    (shadow / "brain" / "__init__.py").write_text(
        "raise ImportError('shadowed broken brain')\n", encoding="utf-8"
    )
    bad_argv = [sys.executable, "-m", "brain.mcp_server", "--persona-dir", str(persona)]  # no -P
    with pytest.raises(Exception):  # noqa: B017,PT011 — child dies at import; any failure == no dispatch
        _call_tool(bad_argv, "list_works", {}, cwd=str(shadow))
