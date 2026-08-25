# DROP-IN-RUNBOOK.md — operating the drop-in harness for a live arm

An operator guide for running a live arm against a specific **version-under-test** of `brain`
using the drop-in ingestion + split-brain guards (`tests/harness/dropin.py`,
`tests/harness/roster_preflight.py`). This is the "how do I actually run it" doc; the design
rationale lives in `dropin.py`'s module docstring and the `changes/harness-drop-in-code-ingest/`,
`changes/toolside-splitbrain-itemA/`, and `changes/run-orchestration-hardening/` audit trails —
read those if you need to know *why*, not just *how*.

For driving the turns themselves once the arm is up, this doc hands off to
`tests/harness/RUN-ORCHESTRATION.md` — the one-arm-one-turn-foreground contract. Do not duplicate
that contract here; read it separately.

## 1. What this solves

Without drop-in, a live arm resolves `brain` from two different places in the same run: the
prompt/bridge process picks up whatever code the launcher's venv happens to point at, while the
`brain-tools` MCP child is a fresh `sys.executable` subprocess that does **not** inherit the
parent's in-memory `sys.path` and can resolve a completely different `brain` install. The prior
incident was exactly this: the prompt ran the new code under test, the tool child quietly served
the old one, and nothing detected it. Drop-in makes "test this version" literal — copy the
version-under-test into one sandbox location, build one dedicated venv on that copy, and launch
**every** process (prompt and the transitively-spawned tool child) under that one interpreter, so
there is only one `brain` to resolve, by construction.

## 2. Prerequisites

- The build under test **must** already contain the #138 fix (`brain_tools_mcp_entry` helper in
  `brain/bridge/provider.py`, which spawns the `brain-tools` MCP child with `-P` at all three
  call sites). **Read this loudly:** `tests/harness/roster_preflight.py` imports
  `brain.bridge.provider.brain_tools_mcp_entry` directly from the copied build in order to
  reproduce its actual child-spawn argv. A pre-#138 build has no such helper, so the roster
  preflight's import fails with an `ImportError` at the gate — before any token is spent, but
  also before you get a useful diagnosis unless you know to check for this. If you are testing an
  older `brain` snapshot deliberately (not the harness's own regression), you cannot use the
  roster preflight as shipped; that is expected, not a bug to chase.
- `uv` on PATH is preferred for the venv build (`ingest_version(..., deps="auto")` shells out to
  `uv venv` / `uv pip install -e`); it falls back to the stdlib `venv` module + `pip` if `uv` is
  absent. `deps="auto"` needs network — run it outside any network-sandboxed shell.
- Only one harness server against a given `dest` at a time (see the serial-use precondition in
  section 3a) — this matches the standing "one harness server at a time" operating rule.

## 3. The sequence

All of steps (a)-(c) happen once, before any token is spent. Step (d) is the arm itself
(hands off to `RUN-ORCHESTRATION.md`). Step (e) happens immediately after driving, before
teardown.

### (a) Ingest the version under test

```python
from pathlib import Path
from tests.harness import ingest_version

build = ingest_version(
    source=Path("<version-under-test repo root, contains brain/>"),
    dest=Path("<stable sandbox copy dir>"),   # e.g. /home/zero/.dropin/symptom-cluster
    on_existing="archive",   # or "delete"; NEVER "prompt" in an autonomous run
    deps="auto",             # uv pip install -e on the copy — NEEDS NETWORK
)
# build.repo       -> the copied code root (contains brain/); the containment root the guards check
# build.python     -> the dedicated venv interpreter EVERY run process must launch under
# build.venv_root  -> build.python.parent.parent, the venv prefix (== <dest>/.dropin-venv)
```

`ingest_version(source, dest, *, on_existing="prompt", deps="auto", install_guard=True) ->
DropinBuild` copies `source` into `dest` (excluding VCS metadata, caches, any pre-existing venv,
the drop-in venv itself, and the `changes/` guarded-change stage folder), builds a dedicated venv
on that copy, and — with `install_guard=True` (the default) — installs a `.pth` + startup-guard
module into the venv's `site-packages` that hard-exits (`os._exit(70)`) any process under this
venv whose `brain` resolves from outside the copy. That hook is what defends the MCP child, which
has no `DropinBuild` handle of its own to check against.

**The prior-copy prompt, and the non-interactive flag.** If `dest` already exists,
`on_existing` controls what happens to it:
- `"prompt"` (the default) asks interactively — archive to a zip, delete, or abort — on a real
  tty. **With no tty it raises `DropinMismatch` instead of blocking.** An autonomous/scripted run
  must pass `on_existing="archive"` or `"delete"` explicitly; never leave it at `"prompt"` for a
  non-interactive invocation.
- `"archive"` zips the prior copy (excluding the drop-in venv) to `<dest>.<UTC-timestamp>.zip`
  beside `dest`, then removes the directory.
- `"delete"` removes it outright, no archive.

**Serial-use precondition (CP7).** `dest` is a stable, cross-process shared location. Do not
re-ingest into, or archive/delete, a `dest` while a run against it is still live — `on_existing`
in `{"delete", "archive"}` does an `rmtree(dest)`, and pulling the module tree out from under a
running MCP child is a teardown race, not a clean rebuild.

### (b) Launch every process under `build.python`

The prompt/bridge process (`live_server.py` or equivalent) and, transitively, the `brain-tools`
MCP child it spawns must **both** run under `build.python` — not `uv run python`, not whatever
interpreter the launching shell happens to have on PATH. This is the whole mechanism, not a
nicety: the MCP child is launched as `sys.executable -m brain.mcp_server ...`
(`brain_tools_mcp_entry`, `brain/bridge/provider.py:270`), so `sys.executable` of the *parent*
prompt process is what determines which `brain` the child resolves. Launch the prompt process
itself with `build.python <script> ...` (not `uv run python <script> ...`, which would resolve a
venv from the invoking cwd instead) and the correct interpreter propagates to the child for free.

### (c) The startup guards, in order

Two guards run in sequence, in two different processes, at two different points:

1. **`assert_brain_under_dropin(build)`** — process-self check, called inside the prompt process
   (e.g. `live_server.py`) after the persona dir has been resolved inside the sandbox. It asserts,
   in order: (i) `Path(sys.prefix).resolve() == build.venv_root.resolve()` — this *is* the
   prompt process actually running under the drop-in venv, not just resolving `brain` from it by
   accident; and (ii) `brain`'s resolved origin is under `build.repo`. Raises `DropinMismatch` on
   either failure, aborting the run at prompt startup, before any child is spawned. Pass the
   `build` you got back from `ingest_version` — do not reconstruct a `DropinBuild` from
   `sys.executable` inside the prompt process, since that would compare the running venv to
   itself and always pass, defeating the check.

2. **`assert_brain_tools_roster(build, persona_dir)`** — run-level, tool-side preflight, called
   **after `assert_brain_under_dropin(build)`**, after `build_persona()` has resolved
   `persona_dir`, and **before `BridgeServer.start()`** (before any token spend).
   `persona_dir` is only resolvable once `build_persona()` has run inside the sandbox, which is
   why this cannot go any earlier. It re-asserts the in-process `brain` is under `build.repo`
   (so the in-process tool registry it reads is trustworthy), then spawns its own `brain-tools`
   child reproducing `brain_tools_mcp_entry`'s real argv/config, does a real MCP `tools/list`
   handshake, and hard-fails (`DropinMismatch`) on either a dead/unresponsive child or a
   served-vs-declared roster mismatch.

```python
from tests.harness import assert_brain_under_dropin, assert_brain_tools_roster

# Inside the prompt process, after persona_dir resolves inside the sandbox, before token spend:
assert_brain_under_dropin(build)
assert_brain_tools_roster(build, persona_dir)
```

Do not swap the order and do not skip either one just because the other passed — they check
different processes (prompt vs. the tool child) and only the second one can catch a dead/mis-wired
tool child (see gotchas below).

### (d) Drive the arm, one-turn-foreground

Once the arm is booted and both guards have passed, drive it per
`tests/harness/RUN-ORCHESTRATION.md`: **one arm = one agent = one turn, everything foreground.**
Boot via `tests.harness.foreground.boot_and_verify`, confirm the tool side via
`assert_tool_callable` after turn 1, drive all remaining turns inline or via a synchronous
(`run_in_background: false`) AgentBob, then pause. Never background a boot or a drive step and end
your turn to wait on it — that is the dead-stall this contract exists to prevent. Full detail,
including the exact `boot_and_verify` / `assert_tool_callable` call shapes, lives in
`RUN-ORCHESTRATION.md`; this runbook does not repeat it.

### (e) Preserve artifacts

Immediately after driving, before any teardown, copy the raw run artifacts out with
`tests.harness.foreground.preserve_artifacts`:

```python
from tests.harness.foreground import preserve_artifacts

result = preserve_artifacts(
    run_dir,
    subproc_dirs=("<your subprocess-transcript dir name>",),
    require=("turn_diag.jsonl", "transcript.jsonl", "<subproc dir>"),
)
```

Pass `require=` for every artifact that is load-bearing for adjudication. A required source that
is missing on disk, or that was never listed in `names=`/`subproc_dirs=` at all, raises
`PreservationIncomplete` rather than returning a silently-green result — the VM has crashed
mid-run before, and a required artifact silently absent from the preserved copy is worse than a
loud failure here.

## 4. Gotchas — why this fails loud, not silent

- **The `-P` cwd-shadow (#138).** The `brain-tools` MCP child is spawned with the `-P`
  (`PYTHONSAFEPATH`) flag, which drops the implicit launch-cwd entry from the child's `sys.path`
  so a foreign top-level `brain/` sitting in the launch directory (for example, the drop-in copy
  root itself, which contains one) cannot shadow the venv/editable `brain`. This is what makes
  `sys.executable` of the parent process the single thing that determines which `brain` the child
  gets — no `cwd=` trick is used or needed. If the build under test predates #138, this flag isn't
  applied and the roster preflight can't even import the helper it needs to reproduce the child
  spawn (see Prerequisites).
- **`DropinMismatch` from the roster preflight means a dead or mis-wired tool child — take it
  seriously, don't retry blind.** The reason `assert_brain_tools_roster` exists at all: a
  cwd-shadowed or wrong-venv `brain-tools` child that dies at import (the venv guard hook's
  `os._exit(70)`) is invisible to the external `claude` CLI, which proceeds **tool-less and
  silent** rather than aborting (this is "F5" in the audit trail). Without the roster preflight, a
  split-brain run completes normally and *looks* fine while every tool call silently degrades to
  "no such tool." A `DropinMismatch` here is the run telling you the child never came up right —
  re-check the venv/interpreter wiring (step (b)) before re-running, don't just retry.
- **Placement constraints are not cosmetic.** `assert_brain_under_dropin` must run before any
  child is spawned (it's the check that would have caught the original incident: a wrong-venv
  prompt process spawning a stale, unguarded child). `assert_brain_tools_roster` must run after
  `build_persona()` (it needs a resolvable `persona_dir`) and before `BridgeServer.start()` (it
  must abort before any token is spent). Moving either guard earlier than its precondition allows,
  or later than "before token spend," reopens the window the guard exists to close.
- **Pass `build` into the roster preflight, not just `persona_dir`.** It self-enforces its own
  precondition (re-checking `brain` resolves under `build.repo`) rather than trusting the caller
  to have run `assert_brain_under_dropin` first; passing only the persona dir would silently drop
  that self-check.
- **One `dest` at a time.** Re-ingesting or archiving/deleting a `dest` (via `ingest_version`)
  while a run against it is still live races the running MCP child's own module tree. Tear down
  the run before touching `dest` again.

## See also

- `tests/harness/dropin.py` — the copy-in + venv-build + guard-hook implementation (module
  docstring has the full design rationale).
- `tests/harness/roster_preflight.py` — the tool-side roster preflight implementation.
- `tests/harness/RUN-ORCHESTRATION.md` — the one-arm-one-turn-foreground driving contract.
- `tests/harness/README.md` — the harness's general API reference.
