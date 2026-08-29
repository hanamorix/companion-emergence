"""Run-level roster preflight: the tool-side half of the split-brain guard.

The drop-in guard (:mod:`tests.harness.dropin`) closes the *prompt-side* split-brain: it asserts
the bridge process resolves ``brain`` under the drop-in copy, and installs a venv hook that
``os._exit(70)``s any process (including the ``brain-tools`` MCP child) whose ``brain`` resolves
outside the copy. But that hook can only *kill* a mis-wired child — it cannot *fail the run*,
because the thing that spawns and then ignores a dead ``brain-tools`` child is the **external
``claude`` CLI**, which (verified live: "F5") proceeds **tool-less and silent** when the child
dies. So a cwd-shadowed run passes the prompt-side guard AND completes normally while every tool
call silently degrades to "no such tool."

This preflight is the **run-level** gate that catches that. Call it once, after
:func:`tests.harness.dropin.assert_brain_under_dropin` and BEFORE any token spend. It:

1. re-asserts (from ``build``) that the in-process ``brain`` resolves under the copy, so the
   in-process tool registry it reads next is authoritative (it does not delegate this precondition
   to the call-site);
2. spawns the ``brain-tools`` child reproducing the argv/config provider builds
   (:func:`brain.bridge.provider.brain_tools_mcp_entry`) and performs a **real MCP ``tools/list``
   handshake over stdio** — the ground-truth roster the child actually serves, reflecting its real
   ``brain`` resolution (not an in-process import, which would not reflect a separate child's
   cwd-shadow);
3. **hard-fails the run** (raises :class:`tests.harness.dropin.DropinMismatch`) on **either** a
   dead / exited / unresponsive child **or** a served-vs-declared roster mismatch.

**Mirror scope.** The child argv/config is a *config-level* mirror of provider's, not a
*process-level* one: a real run's child is spawned by the ``claude`` CLI (``bridge → CLI →
child``); this one by the bridge directly (``bridge → child``). They share cwd/env only under the
assumption that the CLI spawns the MCP child with the bridge's cwd/env — true today because the
``mcp.json`` provider writes carries no ``cwd`` key. That assumption bounds the *regression-gate*
value (reproducing a future ``-P``-drop cwd-shadow) only; the *primary* venv/``sys.path`` catch
holds regardless, because that shadow trips the venv guard hook irrespective of cwd.

**Roster comparison is by tool NAME.** Two ``brain`` versions with an identical tool-name set
produce an identical roster. In the drop-in harness this is not a gap: a same-name foreign shadow
makes the child ``os._exit(70)`` at import (the venv guard hook), so the preflight catches it as a
*dead child*, not a roster mismatch. The name comparison is the secondary net for a genuinely
divergent roster; the primary catch is child-liveness. This preflight is a harness gate and is not
wired into production (production's #138 protection is the ``-P`` prevention in provider).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from brain.bridge.provider import brain_tools_mcp_entry
from tests.harness.dropin import (
    DropinBuild,
    DropinMismatch,
    _is_under,
    _resolve_module_origin,
)

_DEFAULT_TIMEOUT = 15.0


def _assert_declared_authoritative(build: DropinBuild) -> None:
    """Re-assert the in-process ``brain`` resolves under the copy BEFORE trusting its registry.

    Self-enforces the preflight's precondition rather than delegating it to the (un-applied)
    call-site: if the bridge process's ``brain`` is not under ``build.repo``, the in-process
    ``NELL_TOOL_NAMES`` the declared roster is built from is not the copy's, so the comparison
    would be meaningless — hard-fail instead.
    """
    origin = _resolve_module_origin("brain")
    if origin is None or not _is_under(origin, build.repo):
        raise DropinMismatch(
            f"roster-preflight precondition failed: in-process 'brain' resolved to {origin}, NOT "
            f"under the drop-in copy {build.repo.resolve()} — the declared roster cannot be "
            "trusted. Run assert_brain_under_dropin first / launch under the drop-in venv."
        )


def _declared_roster(persona_dir: Path) -> set[str]:
    """The roster the copy declares, computed the SAME way the server's ``_list_tools`` does.

    Mirrors ``brain/mcp_server/tools.py`` (``companion_name = persona_dir.name``; served =
    ``[n for n in NELL_TOOL_NAMES if n in build_schemas(companion_name)]``). Read from the copy at
    runtime — no hardcoded tool name. Trustworthy only after :func:`_assert_declared_authoritative`.
    """
    from brain.tools import NELL_TOOL_NAMES
    from brain.tools.schemas import build_schemas

    schemas = build_schemas(persona_dir.name)
    return {name for name in NELL_TOOL_NAMES if name in schemas}


def _child_argv(persona_dir: Path) -> list[str]:
    """The child argv provider's sites build — the config-level mirror (same shared entry)."""
    entry = brain_tools_mcp_entry(persona_dir)
    return [entry["command"], *entry["args"]]


def _live_roster(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> set[str]:
    """Spawn the child at ``argv`` and return the tool names it serves via a real MCP ``tools/list``.

    Raises :class:`DropinMismatch` if the child dies / never handshakes / serves nothing (the F5
    catch). ``env`` defaults to a copy of ``os.environ`` so the child inherits the run's
    environment (mirroring the CLI-spawned child), not the mcp SDK's minimal default env.
    """
    child_env = dict(os.environ) if env is None else env

    async def _go() -> set[str]:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=argv[0],
            args=list(argv[1:]),
            env=child_env,
            cwd=str(cwd) if cwd is not None else None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return {tool.name for tool in result.tools}

    try:
        return asyncio.run(asyncio.wait_for(_go(), timeout))
    except DropinMismatch:
        raise
    except Exception as exc:  # noqa: BLE001 — any spawn/handshake failure or exit == dead child
        raise DropinMismatch(
            "brain-tools MCP child did not serve tools/list (dead / exited / unresponsive): "
            f"{type(exc).__name__}: {exc}. argv={argv}. This is the F5 catch — the claude CLI "
            "would proceed tool-less and silent; the run is invalid and is hard-failed here."
        ) from exc


def assert_brain_tools_roster(
    build: DropinBuild,
    persona_dir: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> None:
    """Run-level preflight. Call AFTER ``assert_brain_under_dropin``, BEFORE token spend.

    Takes ``build`` so it can self-enforce its containment precondition. Hard-fails
    (:class:`DropinMismatch`) on a dead/exited child OR a served-vs-declared roster mismatch.
    """
    _assert_declared_authoritative(build)
    declared = _declared_roster(persona_dir)
    live = _live_roster(_child_argv(persona_dir), env=env, cwd=cwd, timeout=timeout)
    if live != declared:
        raise DropinMismatch(
            "brain-tools served roster != declared roster: "
            f"served-only={sorted(live - declared)}, declared-only={sorted(declared - live)}. "
            "The MCP child resolved a different 'brain' than the drop-in copy (tool-side "
            "split-brain)."
        )
