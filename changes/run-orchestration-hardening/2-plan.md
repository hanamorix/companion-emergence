# 2 — Plan: run-orchestration hardening

## Baseline / regression

**No stage-0 baseline.** This is a greenfield capability (a new module + a new doc + a
banner). There is no prior version of these artifacts and no measurable runtime behavior to
snapshot. Stage 8 runs **conformance-only** against the 1.5 criteria; the project's standing
regression metrics (cost/cache/turns from `chat_usage.jsonl`) are untouched — this change
does not run chat turns and adds no production code path.

## Instrumentation

**None required.** The criteria are mechanical facts about pure/blocking Python functions and
a shipped doc; they are verified directly by `pytest` + `ruff` + file inspection. No new
logging, no telemetry, no production instrumentation. (This is expected for a
test-harness-only change; the config's `metrics`/`check` are for production behavior changes
and stay as-is.)

## Design detail (the shapes the build implements)

### New module `tests/harness/foreground.py`

Note the stage-3 blocker F1 fix: `boot_and_verify` does **not** call the item-A process-self
asserts (`assert_brain_under_dropin`, `assert_brain_tools_roster`) — those check the *calling*
process's venv/`brain` and are only meaningful inside `live_server.py` (where item A already
wires the guard). The orchestrator-side verify is READY-confirmation + an optional READY-payload
cross-check + a transcript-based tool-side check. All deterministic, all real-path unit-testable.

```python
DRIVE_NOW_BANNER_HEADLINE = "▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to wait"

@dataclass(frozen=True)
class ArmBootSpec:
    arm: str
    port: int
    run_dir: Path
    ready_timeout: float = 60.0            # bounds boot; a dead boot RAISES, never hangs
    expect_brain_repo_under: Path | None = None   # optional READY-payload cross-check

@dataclass(frozen=True)
class ArmSession:
    arm: str
    port: int
    run_dir: Path
    ready_payload: dict                    # parsed <run_dir>/READY
    brain_repo: str | None                 # from the READY payload (informational + cross-check)

@dataclass(frozen=True)
class PreserveResult:
    dest: Path
    copied: list[Path]                     # what landed in dest
    missing: list[Path]                    # requested (non-required) sources that did not exist

class ForegroundBootError(RuntimeError): ...        # non-zero boot / ERROR|leak marker / no READY in time
class ToolSideBroken(RuntimeError): ...             # version-unique tool absent from the served MCP roster
class PreservationIncomplete(RuntimeError): ...     # a REQUIRED source was missing at preserve time


def boot_and_verify(
    spec: ArmBootSpec,
    *,
    boot_cmd: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    error_markers: Sequence[str] = ("ERROR",),
) -> ArmSession:
    """FOREGROUND, blocking, bounded. Boot -> confirm clean READY -> (optional) cross-check -> return.

    MUST be called in the foreground, in the same turn you will drive the arm in.
    NEVER wrap this in a run_in_background task: a backgrounded child's completion
    routes to main, not back to you, and you will dead-stall. This call blocks and
    returns in-turn; a dead boot RAISES within ready_timeout, it does not hang.

    Orchestrator-side verification is CONFIRMATION, not re-execution of the in-process
    drop-in guard: live_server.py writes READY only after its guard passes and writes an
    error marker on hard-fail, so READY-present-without-error IS the confirmation.
    """
```

Behavior of `boot_and_verify`:
1. If `boot_cmd` given: `subprocess.run(boot_cmd, cwd=cwd, env=env, timeout=spec.ready_timeout,
   capture_output=True)`. FOREGROUND, blocking, bounded. Non-zero exit → raise
   `ForegroundBootError` (include stderr tail). `subprocess.TimeoutExpired` → raise
   `ForegroundBootError` — either way it raises in-turn, does not hang. If `boot_cmd` is None:
   bounded poll loop (`time.sleep(0.1)` up to `ready_timeout`) waiting for `<run_dir>/READY` or
   an error marker to appear; timeout → raise `ForegroundBootError`. **Stale-READY guard (N3):**
   in poll mode, capture a start timestamp before polling and ignore a `READY` whose mtime
   predates it, so a leftover `READY` in a reused run dir cannot be read as a false confirmation
   (belt-and-suspenders with the clean-slate rule; the `boot_cmd`-given path has no such race).
   `error_markers` defaults to `("ERROR",)` only — `PAUSED_ON_LEAK.json` is a teardown
   pause-for-adjudication state, not a boot failure (N4), and is never written in the boot window.
2. If any `<run_dir>/<error_marker>` exists → raise `ForegroundBootError` with its contents.
   Else read and JSON-parse `<run_dir>/READY`; absent/unparseable → raise `ForegroundBootError`.
3. If `spec.expect_brain_repo_under` is set, assert the READY payload's `brain_repo` resolves
   under it → else raise `ForegroundBootError` (an orchestrator-side file read, not a
   process-self check). This is the confirmation that the booted server loaded the drop-in copy.
4. Return populated `ArmSession`.

```python
def assert_tool_callable(
    source: Path,                    # a turn_diag.jsonl / transcript.jsonl / a run dir to search
    tool_name: str,                  # a tool unique to the version under test, e.g. "read_full_memory"
    *,
    roster_field: str = "sent.system",
) -> None:
    """FOREGROUND. RUN-INSTRUCTIONS step 4: after turn 1, confirm the version-unique tool is in
    the served MCP roster (a broken MCP child fails SILENTLY / tool-less). Raises ToolSideBroken
    if the tool is absent from the served roster in `source`. Reads a file; no live bridge.

    N1: the served roster appears under `sent.system` per RUN-INSTRUCTIONS step 4, but the exact
    turn_diag serialization lives in the gitignored live-run lane and is UNVERIFIABLE from tracked
    code. Extraction is therefore tolerant (scan the sent.system / system text of the turn rows
    for the tool name as a served tool), and the handoff note tells Testing to confirm the
    roster-field assumption against a real captured turn_diag before relying on it live."""

def preserve_artifacts(
    run_dir: Path,
    *,
    dest: Path | None = None,           # default <run_dir>/valid-run
    names: Sequence[str] = ("turn_diag.jsonl", "transcript.jsonl"),
    subproc_dirs: Sequence[str] = (),   # caller-named; the gitignored lane owns the exact name
    require: Sequence[str] = (),        # names/dirs whose absence RAISES PreservationIncomplete
) -> PreserveResult:
    """FOREGROUND, blocking. Copy raw artifacts into dest BEFORE any teardown. A source named in
    `require` that is missing RAISES PreservationIncomplete (the raw artifacts are load-bearing);
    a non-required missing source is reported in PreserveResult.missing. Preserve as much as
    exists (the VM has crashed mid-run). `subproc_dirs` is caller-supplied because the exact
    subprocess-transcript dir name lives in the gitignored live-run lane, not this tracked code."""
```

```python
def drive_now_banner(
    *,
    arm: str,
    port: int,
    run_dir: Path | str,
    roster_ok: bool | None = None,
    extra_lines: Sequence[str] | None = None,
) -> str:
    """Return the operator-facing READY banner. First line is exactly
    DRIVE_NOW_BANNER_HEADLINE. Names both parkable surfaces + the one rule."""
```

The banner body (after the headline) includes: the run context (arm/port/run_dir, and roster
status if given), a line naming the two parkable surfaces ("a run_in_background Bash task AND a
background sub-agent both route completion to main, not back to you"), and the one rule ("one
arm = one turn, everything foreground — drive inline or via a synchronous AgentBob in THIS
turn"). Operator-facing text; the em-dash is intentional and permitted (see spec note).

### `tests/harness/__init__.py`
Add `from .foreground import (ArmBootSpec, ArmSession, PreserveResult, ForegroundBootError,
ToolSideBroken, PreservationIncomplete, boot_and_verify, assert_tool_callable,
preserve_artifacts, drive_now_banner, DRIVE_NOW_BANNER_HEADLINE)` and extend `__all__` with
those names (mirror how `assert_brain_tools_roster` was added in item A).

### `tests/harness/RUN-ORCHESTRATION.md` (tracked doc — the single canonical anti-stall contract)
Adapt the CONTENT of Testing's draft (`RUN-INSTRUCTIONS.md`, read-only input) into a tracked
contract doc. Must: state the one rule prominently; name BOTH parkable surfaces and why each
stalls (completion routes to main); give the one-turn sequence (clean slate → boot+confirm via
`boot_and_verify` foreground → tool-side check via `assert_tool_callable` after turn 1 → drive
inline OR synchronous AgentBob → `preserve_artifacts` foreground → report); loudly warn never to
wrap the helper calls in `run_in_background`; state that on a `ForegroundBootError` the
orchestrator must clean-slate before retry (F3 — a killed boot can orphan a `nohup`'d server);
and note the structural option for the synchronous-AgentBob path (wrap boot→drive→preserve with
no orchestrator-visible seam) while being explicit that the *inline* drive is reasoning that
cannot be encapsulated (so F5's single-callable wrapper is not shipped — it would only fit a
programmatic drive, not the live-arm inline/AgentBob drive; see Thresholds). Reference the helper
and banner as the mechanical support. Keep it focused (not a copy of the whole README).

### `tests/harness/README.md`
The README already carries an "### The orchestration protocol" section (~lines 194-233,
BridgeServer-based). To avoid two divergent copies of the run contract (stage-3 F4), do NOT
restate the anti-stall rule there: add a prominent pointer at the top of that section naming
`RUN-ORCHESTRATION.md` as the canonical anti-stall contract to read first. Minimal edit; the
one-rule text lives in exactly one file.

### `changes/run-orchestration-hardening/9-handoff-drive-banner-callsite.md`
The thin call-site handoff for Testing: (1) the exact one-line edit to `run_arm.sh`'s READY
summary line (currently `echo "READY arm=$ARM …"` at run_arm.sh:60) to additionally emit
`drive_now_banner(...)`'s headline — e.g. a shell `echo` of the literal headline, or a
`python -c "from tests.harness import drive_now_banner; print(drive_now_banner(...))"` call;
(2) a short worked example showing `boot_and_verify` + `assert_tool_callable` (after turn 1) +
drive + `preserve_artifacts` all foreground in one turn; (3) the load-bearing note that these
calls must never be backgrounded, and that on a `ForegroundBootError` the orchestrator must
clean-slate before retry.

## Measurement (how stage 8 verifies each gating criterion)

| Criterion | Measurement |
|---|---|
| G1 | `pytest tests/unit/harness/test_foreground.py` — real boot+confirm path: `boot_cmd` writes a valid READY → populated `ArmSession` returned in-turn |
| G2 | `pytest` — non-zero `boot_cmd`; `boot_cmd` writes an error marker; no READY within a short `ready_timeout` → each raises `ForegroundBootError` quickly, no hang |
| G3 | `pytest` — real paths: READY `brain_repo` outside `expect_brain_repo_under` → `ForegroundBootError`; `assert_tool_callable` over a crafted roster without/with the tool → `ToolSideBroken` / clean |
| G4 | `pytest` — `preserve_artifacts` over a fake run dir: copied + `missing` reported; a `require`d absent source → `PreservationIncomplete` |
| G5 | `pytest` — assert `drive_now_banner(...)` first line equals the literal; both surfaces + rule present |
| G6 | `pytest` reads `tests/harness/RUN-ORCHESTRATION.md` and asserts both surfaces + the rule present; plus inspection |
| G7 | import all new symbols from `tests.harness` in the test; inspect `__all__` |
| G8 | `uv run ruff check .`; `uv run pytest -m "not live and not requires_claude_cli and not integration" -q` (minus the two known out-of-scope failures) |

## Thresholds (what bounces the loop)

- Any **gating** criterion failing at stage 8 → bounce to build (stage 5), or to spec if the
  criterion itself is wrong.
- **Blocker** (e.g. the helper CAN hang, or the banner text is wrong, or a production change
  turns out to be required) → stop, restart the loop / ask the orchestrator.
- **Major** at stage 8 → stop for human.
- **Minor** (naming, docstring wording, a missing advisory) → fix-and-proceed.
- **Harness-effort cap:** if the `test_foreground.py` mechanism bounces more than twice while
  trying to prove a mechanical fact, STOP building the test — that is the signal the fact is
  not unit-provable this way, not a signal to build a bigger harness. Re-scope to inspection +
  a simpler assertion and record the call.
- **F5 (single-callable `run_arm(spec, drive=…)` wrapper) — considered and DEFERRED, not a
  bounce.** It would remove the residual seam between `boot_and_verify` and `preserve_artifacts`
  only for a *programmatic* drive callable; the actual live-arm drives are inline LLM reasoning
  (not a Python callable) or a synchronous AgentBob (an Agent-tool call, not a Python callable),
  neither of which a Python wrapper can invoke. It would therefore add surface + test burden
  without covering the real drive mechanisms — against the effort cap. The one-turn guarantee for
  the drive stays contractual (doc + banner), which is the honest boundary. Recorded so the
  stage-6 review sees this as a deliberate scope call, not an oversight.

## Cold-review context (paths, priority order) for stages 3 and 6

1. `changes/run-orchestration-hardening/{1-spec,1.5-criteria,2-plan}.md` — the artifacts under review.
2. `tests/harness/roster_preflight.py` — item A's shipped pattern this mirrors (signatures, raise/return contract).
3. `tests/harness/dropin.py` — `DropinBuild`, `assert_brain_under_dropin`, `DropinMismatch` (verify types/signatures the plan reuses).
4. `tests/harness/__init__.py` — current `__all__` (the export edit).
5. `hunts/symptom-cluster-rootcause/live-run/RUN-INSTRUCTIONS.md` — Testing's draft (content source for the doc; read-only, must not be edited).
6. `changes/harness-drop-in-code-ingest/9-handoff-live-run-callsite-edits.md` — item A's handoff-note precedent.
7. (stage 6 only) `tests/harness/foreground.py`, `tests/harness/RUN-ORCHESTRATION.md`, `tests/unit/harness/test_foreground.py`, and the `git diff`.

All paths validated at run start (see `decisions.md`).
