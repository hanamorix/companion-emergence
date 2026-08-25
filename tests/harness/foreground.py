"""Run-orchestration hardening: the anti-stall foreground helper (item B of #139).

Live-arm runs repeatedly dead-stalled for ~30 minutes. Root cause (issue #139, live-verified
by Testing): an orchestrator agent **backgrounds a step and ends its turn to wait for it**. A
backgrounded child -- either a ``run_in_background: true`` Bash task OR a background sub-agent
(e.g. AgentBob) -- routes its completion notification to **main**, never back to the waiting
parent. This is structurally guaranteed by the agent messaging topology, not a flaky bug: a
completed child cannot wake the specific parent that parked on it.

The fix pattern (Testing, owner-endorsed): **one arm = one agent = one turn, everything
FOREGROUND.** Boot, verify, drive all turns, pause, preserve, and report within a single turn,
every step foreground. A foreground call blocks and returns in-turn, so the orchestrator can
never park. This module encapsulates the setup/teardown steps that CAN be encapsulated as
blocking, bounded, in-turn calls with no await surface to park on:

- :func:`boot_and_verify` -- boot the arm (bounded) and confirm a clean READY.
- :func:`assert_tool_callable` -- the orchestrator-side tool-side check (a broken MCP child
  fails SILENTLY; the served roster in the transcript is the only real signal).
- :func:`preserve_artifacts` -- copy the raw run artifacts before any teardown.
- :func:`drive_now_banner` -- the loud structural nudge at the READY decision point.

The **drive turns cannot be encapsulated in code** -- that is the orchestrator's own
turn-by-turn reasoning (improvising Bob), or a synchronous AgentBob call. See
``tests/harness/RUN-ORCHESTRATION.md`` for the full one-turn contract this module supports.

**Design note -- why `boot_and_verify` does NOT call the item-A process-self asserts**
(``assert_brain_under_dropin`` / ``assert_brain_tools_roster``). Those check the *calling*
process's own venv/``brain`` resolution and are only meaningful inside ``live_server.py``
(where item A already wires them). ``boot_and_verify`` runs in the **orchestrator** process,
which runs under a different venv -- calling those asserts there would raise on a *healthy*
run. The correct orchestrator-side verification is READY-confirmation (their effect, written
by the in-process guard) plus the transcript-based tool-side check, not re-running the asserts
in the wrong process.

This module hardcodes no gitignored path: ``boot_cmd``, ``source``/turn_diag locations, and
``subproc_dirs`` are all caller-supplied. It imports only the standard library.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

DRIVE_NOW_BANNER_HEADLINE = "▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to wait"

_POLL_INTERVAL = 0.1
_READY_NAME = "READY"


@dataclass(frozen=True)
class ArmBootSpec:
    """What to boot and how to bound the wait for a clean READY.

    ``ready_timeout`` bounds the whole boot: a dead boot RAISES within it, never hangs.
    ``expect_brain_repo_under``, if set, is an optional orchestrator-side cross-check that the
    READY payload's ``brain_repo`` resolves under a given drop-in repo root.
    """

    arm: str
    port: int
    run_dir: Path
    ready_timeout: float = 60.0
    expect_brain_repo_under: Path | None = None


@dataclass(frozen=True)
class ArmSession:
    """A confirmed, booted arm: the parsed READY payload plus the spec fields it answers."""

    arm: str
    port: int
    run_dir: Path
    ready_payload: dict
    brain_repo: str | None


@dataclass(frozen=True)
class PreserveResult:
    """What :func:`preserve_artifacts` copied, and what non-required source was missing."""

    dest: Path
    copied: list[Path] = field(default_factory=list)
    missing: list[Path] = field(default_factory=list)


class ForegroundBootError(RuntimeError):
    """Raised by :func:`boot_and_verify` on a non-zero boot / error marker / timeout / bad READY.

    Always raised in-turn, within ``spec.ready_timeout`` -- ``boot_and_verify`` never hangs.
    """


class ToolSideBroken(RuntimeError):  # noqa: N818 — exact public API name (spec/criteria)
    """Raised by :func:`assert_tool_callable` when a version-unique tool is absent from the
    served MCP roster -- the orchestrator-side catch for a silently tool-less run."""


class PreservationIncomplete(RuntimeError):  # noqa: N818 — exact public API name (spec/criteria)
    """Raised by :func:`preserve_artifacts` when a source named in ``require=`` is missing.

    The raw run artifacts are load-bearing (the VM has crashed mid-run before); a required
    source going missing must never report as a silently-green preserve.
    """


def boot_and_verify(
    spec: ArmBootSpec,
    *,
    boot_cmd: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    error_markers: Sequence[str] = ("ERROR",),
) -> ArmSession:
    """FOREGROUND, blocking, bounded. Boot -> confirm clean READY -> (optional) cross-check -> return.

    MUST be called in the foreground, in the same turn you will drive the arm in. NEVER wrap
    this in a ``run_in_background`` task or a background sub-agent call: a backgrounded
    child's completion routes to **main**, not back to you, and you will dead-stall. This call
    blocks and returns in-turn; a dead boot RAISES within ``spec.ready_timeout``, it does not
    hang.

    Orchestrator-side verification is CONFIRMATION, not re-execution of the in-process
    drop-in guard: ``live_server.py`` writes ``READY`` only after its in-process guard passes,
    and writes an error marker on hard-fail -- so READY-present-without-error IS the legitimate
    orchestrator-side confirmation that the guard/roster passed.

    Behavior:

    1. If ``boot_cmd`` is given: a start timestamp is captured BEFORE ``subprocess.run``, then it
       is run via ``subprocess.run(boot_cmd, timeout=ready_timeout)`` -- FOREGROUND, blocking,
       bounded. A non-zero exit or a timeout raises :class:`ForegroundBootError` in-turn; it
       never hangs. After a successful (exit-0) boot, the same stale-READY guard as the poll
       branch applies: ``<run_dir>/READY`` must exist AND its mtime must be ``>=`` the captured
       start, else :class:`ForegroundBootError` is raised -- a leftover ``READY`` from a reused
       run dir must not be read as a false confirmation just because ``boot_cmd`` happened to
       exit 0. If ``boot_cmd`` is ``None``: a bounded poll loop waits (up to ``ready_timeout``)
       for ``<run_dir>/READY`` or an error marker to appear, for an already-foregrounded external
       boot (e.g. Testing's ``run_arm.sh``, which itself blocks until READY). A start timestamp
       is captured before polling begins and a ``READY`` whose mtime predates it is ignored (a
       stale READY left over in a reused run dir must not be read as a false confirmation).

       **Pitfall:** a ``boot_cmd`` that leaves a surviving process holding the captured
       stdout/stderr pipe (e.g. an un-detached background server launched by the command) will
       make ``subprocess.run(capture_output=True)`` block until ``ready_timeout`` even on an
       otherwise-successful boot, since the pipe never reaches EOF while a child still holds it
       open. This is bounded (it still raises ``ForegroundBootError`` at ``ready_timeout``
       rather than hanging forever), but it is a usage pitfall, not a correctness bug in this
       function. For booting a long-lived server, prefer poll mode (``boot_cmd=None``) after
       foregrounding the boot separately (see ``tests/harness/RUN-ORCHESTRATION.md``'s
       one-turn sequence), or ensure ``boot_cmd`` fully detaches its children's stdio.
    2. If any ``<run_dir>/<error_marker>`` exists, raise :class:`ForegroundBootError` with its
       contents. Otherwise read and JSON-parse ``<run_dir>/READY``; absent or unparseable
       raises :class:`ForegroundBootError`.
    3. If ``spec.expect_brain_repo_under`` is set, assert the READY payload's ``brain_repo``
       resolves under it, else raise :class:`ForegroundBootError` -- an orchestrator-side file
       read, not a process-self check. This confirms the booted server loaded the drop-in copy.
    4. Return a populated :class:`ArmSession`.

    ``error_markers`` defaults to ``("ERROR",)`` only -- ``PAUSED_ON_LEAK.json`` is a teardown
    pause-for-adjudication state, not a boot failure, and is never written in the boot window.
    """
    run_dir = Path(spec.run_dir)

    boot_start: float | None = None
    if boot_cmd is not None:
        boot_start = time.time()
        try:
            proc = subprocess.run(
                list(boot_cmd),
                cwd=cwd,
                env=env,
                timeout=spec.ready_timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ForegroundBootError(
                f"boot_cmd {list(boot_cmd)!r} did not complete within "
                f"{spec.ready_timeout}s (timed out, not hung -- this raised in-turn)."
            ) from exc
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-2000:]
            raise ForegroundBootError(
                f"boot_cmd {list(boot_cmd)!r} exited {proc.returncode}.\nstderr (tail):\n"
                f"{stderr_tail}"
            )
    else:
        _poll_for_ready_or_error(run_dir, spec.ready_timeout, error_markers)

    _raise_on_error_marker(run_dir, error_markers)

    if boot_start is not None:
        _require_fresh_ready(run_dir, boot_start)

    ready_payload = _read_ready_payload(run_dir)

    brain_repo = ready_payload.get("brain_repo") if isinstance(ready_payload, dict) else None
    if spec.expect_brain_repo_under is not None:
        expected_root = Path(spec.expect_brain_repo_under).resolve()
        if not brain_repo:
            raise ForegroundBootError(
                "expect_brain_repo_under was set but the READY payload carries no 'brain_repo' "
                f"field: {ready_payload!r}"
            )
        resolved_brain_repo = Path(brain_repo).resolve()
        if resolved_brain_repo != expected_root and expected_root not in resolved_brain_repo.parents:
            raise ForegroundBootError(
                f"READY payload brain_repo {resolved_brain_repo} is NOT under the expected "
                f"drop-in root {expected_root} -- the booted server did not load the drop-in "
                "copy."
            )

    return ArmSession(
        arm=spec.arm,
        port=spec.port,
        run_dir=run_dir,
        ready_payload=ready_payload,
        brain_repo=brain_repo,
    )


def _poll_for_ready_or_error(
    run_dir: Path, ready_timeout: float, error_markers: Sequence[str]
) -> None:
    """Bounded poll waiting for ``<run_dir>/READY`` (fresh) or an error marker. Never hangs."""
    start = time.time()
    deadline = start + ready_timeout
    ready_path = run_dir / _READY_NAME
    while time.time() < deadline:
        for marker in error_markers:
            if (run_dir / marker).exists():
                return
        if ready_path.exists() and ready_path.stat().st_mtime >= start:
            return
        time.sleep(_POLL_INTERVAL)
    raise ForegroundBootError(
        f"no READY (or error marker) appeared under {run_dir} within {ready_timeout}s "
        "(bounded poll timed out, did not hang)."
    )


def _require_fresh_ready(run_dir: Path, start: float) -> None:
    """boot_cmd-branch counterpart to the poll branch's stale-READY guard.

    A successful (exit-0) ``boot_cmd`` is not itself proof the run booted -- a leftover
    ``READY`` file from a reused ``run_dir`` would otherwise be misread as this boot's own
    confirmation. Require ``<run_dir>/READY`` to exist AND have an mtime ``>= start``.
    """
    ready_path = run_dir / _READY_NAME
    if not ready_path.exists() or ready_path.stat().st_mtime < start:
        raise ForegroundBootError(
            f"boot_cmd exited 0 but {ready_path} is missing or stale (its mtime predates the "
            "boot's own start) -- refusing to treat a leftover READY in a reused run_dir as a "
            "false confirmation."
        )


def _raise_on_error_marker(run_dir: Path, error_markers: Sequence[str]) -> None:
    for marker in error_markers:
        marker_path = run_dir / marker
        if marker_path.exists():
            contents = marker_path.read_text(encoding="utf-8", errors="replace")
            raise ForegroundBootError(f"boot wrote error marker {marker!r}:\n{contents}")


def _read_ready_payload(run_dir: Path) -> dict:
    ready_path = run_dir / _READY_NAME
    if not ready_path.exists():
        raise ForegroundBootError(f"no READY file at {ready_path} after boot.")
    raw = ready_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ForegroundBootError(f"READY at {ready_path} is not valid JSON: {raw!r}") from exc
    if not isinstance(payload, dict):
        raise ForegroundBootError(f"READY at {ready_path} did not parse to an object: {payload!r}")
    return payload


def assert_tool_callable(
    source: Path,
    tool_name: str,
    *,
    roster_field: str = "sent.system",
) -> None:
    """FOREGROUND. Confirm ``tool_name`` is actually in the served MCP roster (RUN-INSTRUCTIONS
    step 4, after turn 1).

    A broken MCP child fails SILENTLY -- the ``claude`` CLI proceeds tool-less, no abort -- so
    this is the real, load-bearing tool-side verify: it can only be caught from the actual
    served roster in the transcript, orchestrator-side. Raises :class:`ToolSideBroken` if
    ``tool_name`` is absent from the served roster found in ``source``.

    ``source`` may be a ``turn_diag.jsonl`` / ``transcript.jsonl`` file path directly, or a run
    dir containing one of those files (``turn_diag.jsonl`` is preferred if both exist).

    N1: the served roster appears under ``sent.system`` per RUN-INSTRUCTIONS step 4, but the
    exact turn_diag serialization lives in the gitignored live-run lane and is UNVERIFIABLE
    from tracked code. Extraction is therefore tolerant: this scans the ``roster_field`` /
    general system text of every JSON row in ``source`` for ``tool_name`` as a substring, rather
    than assuming a rigid schema. ``roster_field`` documents the expected field but is not
    strictly required to be present -- rows without it are still scanned as a whole. No live
    bridge; this only reads a file.
    """
    path = _resolve_diag_source(source)
    text = path.read_text(encoding="utf-8", errors="replace")

    if _tool_name_in_served_text(text, tool_name, roster_field):
        return

    raise ToolSideBroken(
        f"tool {tool_name!r} was NOT found in the served roster (field {roster_field!r}) of "
        f"{path} -- the MCP tool child is silently broken (the run proceeded tool-less)."
    )


def _resolve_diag_source(source: Path) -> Path:
    source = Path(source)
    if source.is_file():
        return source
    if source.is_dir():
        for name in ("turn_diag.jsonl", "transcript.jsonl"):
            candidate = source / name
            if candidate.exists():
                return candidate
        raise ToolSideBroken(
            f"{source} is a directory but contains neither turn_diag.jsonl nor "
            "transcript.jsonl to scan for the served roster."
        )
    raise ToolSideBroken(f"{source} does not exist (no turn_diag/transcript to scan).")


def _tool_name_in_served_text(text: str, tool_name: str, roster_field: str) -> bool:
    """Tolerant scan (N1): look for ``tool_name`` inside any row's roster/system text.

    Tries, per non-empty line: parse as JSON and walk for a value at (or containing)
    ``roster_field``'s dotted path, falling back to scanning the WHOLE row's raw text if the
    field is absent or unparseable -- so this survives an unknown/differently-shaped
    serialization rather than hard-assuming a rigid schema.
    """
    field_parts = roster_field.split(".")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if tool_name in line:
                return True
            continue
        value = _dig(row, field_parts)
        haystack = json.dumps(value) if value is not None else json.dumps(row)
        if tool_name in haystack:
            return True
    return False


def _dig(obj: object, parts: list[str]) -> object | None:
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def preserve_artifacts(
    run_dir: Path,
    *,
    dest: Path | None = None,
    names: Sequence[str] = ("turn_diag.jsonl", "transcript.jsonl"),
    subproc_dirs: Sequence[str] = (),
    require: Sequence[str] = (),
) -> PreserveResult:
    """FOREGROUND, blocking. Copy raw run artifacts into ``dest`` BEFORE any teardown.

    Copies each of ``names`` (files) and ``subproc_dirs`` (directories) found directly under
    ``run_dir`` into ``dest`` (default ``<run_dir>/valid-run``). Preserves as much as exists --
    the VM has crashed mid-run before, so a missing non-required source is reported in
    ``PreserveResult.missing`` rather than aborting the rest.

    The guarantee: if a name is listed in ``require=``, then after a clean (non-raising) return
    that source WAS copied into ``dest``. This is enforced globally, not just inside the
    ``names``/``subproc_dirs`` copy loops -- a ``require=`` entry that isn't ALSO listed in
    ``names=`` or ``subproc_dirs=`` (so it was never even attempted) raises
    :class:`PreservationIncomplete` just the same as one that was attempted and found missing on
    disk. A required entry never silently no-ops just because it was left out of the copy lists.
    Idempotent (re-running overwrites the same dest cleanly); no network.
    """
    import shutil

    run_dir = Path(run_dir)
    dest_dir = Path(dest) if dest is not None else run_dir / "valid-run"
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    missing: list[Path] = []
    required = set(require)
    confirmed: set[str] = set()

    for name in names:
        src = run_dir / name
        dst = dest_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(dst)
            if name in required:
                confirmed.add(name)
        else:
            missing.append(src)

    for dirname in subproc_dirs:
        src_dir = run_dir / dirname
        dst_dir = dest_dir / dirname
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            copied.append(dst_dir)
            if dirname in required:
                confirmed.add(dirname)
        else:
            missing.append(src_dir)

    unconfirmed = sorted(required - confirmed)
    if unconfirmed:
        raise PreservationIncomplete(
            f"required source(s) {unconfirmed!r} were not confirmed copied into {dest_dir} -- "
            "either missing on disk, or never listed in names=/subproc_dirs= at all -- "
            "refusing a silently-green preserve."
        )

    return PreserveResult(dest=dest_dir, copied=copied, missing=missing)


def drive_now_banner(
    *,
    arm: str,
    port: int,
    run_dir: Path | str,
    roster_ok: bool | None = None,
    extra_lines: Sequence[str] | None = None,
) -> str:
    """Return the operator-facing READY banner. Pure function; no I/O.

    First line is exactly :data:`DRIVE_NOW_BANNER_HEADLINE`. The body states the run context
    (arm/port/run_dir, and roster status if given), names BOTH parkable surfaces (a
    ``run_in_background: true`` Bash task AND a background sub-agent -- both route their
    completion to main, not back to you), and restates the one rule: one arm = one turn,
    everything foreground -- drive inline or via a synchronous AgentBob in THIS turn.

    Operator-facing text (printed to the human/agent running the harness); it is never
    injected into any companion/persona context, so the em-dash in the headline is intentional
    and correct here (see the spec's note on the no-claudisms rule).
    """
    lines = [DRIVE_NOW_BANNER_HEADLINE, ""]
    lines.append(f"arm={arm} port={port} run_dir={run_dir}")
    if roster_ok is not None:
        lines.append(f"roster_ok={roster_ok}")
    lines.append("")
    lines.append(
        "Both a `run_in_background: true` Bash task AND a background sub-agent are PARKABLE "
        "surfaces: each one's completion routes to main, not back to you, so ending your turn "
        "to wait on either one dead-stalls you."
    )
    lines.append(
        "THE ONE RULE: one arm = one turn, everything foreground -- drive inline or via a "
        "synchronous AgentBob, in THIS SAME turn."
    )
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines)
