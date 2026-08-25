# Handoff — thin call-site edits for the Testing lane (NOT applied here)

**Audience:** the owner of the gitignored `hunts/symptom-cluster-rootcause/live-run/` files
(`run_arm.sh`, `live_server.py`), which live only in the main clone
(`/home/zero/Desktop/companion-emergence`). This note is documentation only — the tracked
capability is on branch `ThinkerOfThoughts/harness-drop-in-code-ingest`, module
`tests/harness/foreground.py`, exported from `tests.harness`.

**The contract you must follow when orchestrating a live arm** is
`tests/harness/RUN-ORCHESTRATION.md` (tracked). One rule: **one arm = one agent = one turn,
everything foreground.** Never background a step and end your turn — a backgrounded child's
completion routes to main, not back to you, and you dead-stall (issue #139).

## What the harness now gives you (all foreground, all in-turn)

```python
from tests.harness import (
    ArmBootSpec, boot_and_verify, assert_tool_callable, preserve_artifacts, drive_now_banner,
)
```

- `boot_and_verify(spec, *, boot_cmd=None, cwd=None, env=None) -> ArmSession` — foreground,
  bounded by `spec.ready_timeout`. Confirms the arm reached a clean READY (raises
  `ForegroundBootError` on the ERROR marker / a stale-or-absent READY / timeout) and, if
  `spec.expect_brain_repo_under` is set, cross-checks the READY payload's `brain_repo` is under
  it. It does NOT re-run the in-process drop-in self-asserts (`assert_brain_under_dropin` /
  `assert_brain_tools_roster`) — those are process-self checks that belong inside
  `live_server.py` (where item A wires the guard); READY-present-without-ERROR is the
  orchestrator-side confirmation that they passed.
- `assert_tool_callable(source, "read_full_memory") -> None` — after turn 1, confirm a
  version-unique tool is in the served MCP roster (raises `ToolSideBroken` if not). `source` is
  the turn_diag/transcript file or the run dir. **N1 caveat:** extraction scans the served-roster
  / `sent.system` text for the tool name; the exact `turn_diag.jsonl` serialization lives in
  YOUR lane, so confirm this field assumption against one real captured turn_diag before relying
  on it live (adjust `roster_field=` if your served roster sits under a different key).
- `preserve_artifacts(run_dir, *, dest=None, names=(...), subproc_dirs=(...), require=(...))
  -> PreserveResult` — foreground copy of raw artifacts into `<run_dir>/valid-run`. **Pass
  `require=` for every load-bearing artifact** (turn_diag + transcript + your subprocess
  transcript dir): a required source that is absent (or not listed in `names`/`subproc_dirs`)
  RAISES `PreservationIncomplete` instead of returning a silently-green result.
- `drive_now_banner(*, arm, port, run_dir, roster_ok=None) -> str` — the operator-facing READY
  banner; first line is the literal `▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to
  wait`, followed by both parkable surfaces + the one rule.

## Worked example (the whole arm in ONE foreground turn)

```python
from pathlib import Path
from tests.harness import ArmBootSpec, boot_and_verify, assert_tool_callable, preserve_artifacts

spec = ArmBootSpec(arm=ARM, port=PORT, run_dir=Path(RUN),
                   expect_brain_repo_under=Path(CE_DROPIN_REPO))   # optional cross-check
# 1. BOOT + CONFIRM (foreground; run_arm.sh already blocks to READY, so pass it as boot_cmd —
#    OR foreground `bash run_arm.sh ...` yourself and call boot_and_verify with boot_cmd=None).
session = boot_and_verify(spec, boot_cmd=["bash", "run_arm.sh", ARM, PORT, RUN, CE_DROPIN_REPO, CE_DROPIN_PY])
# 2. DRIVE turn 1 (inline via agent_send.py, OR a synchronous AgentBob — NEVER run_in_background).
# 3. TOOL-SIDE CHECK after turn 1:
assert_tool_callable(Path(RUN) / "turn_diag.jsonl", "read_full_memory")
# 4. DRIVE remaining turns (still foreground, same turn).
# 5. PAUSE + PRESERVE immediately (before any teardown):
result = preserve_artifacts(Path(RUN),
                            subproc_dirs=("<your subprocess-transcript dir name>",),
                            require=("turn_diag.jsonl", "transcript.jsonl", "<subproc dir>"))
# 6. report; hold. On a ForegroundBootError, clean-slate (teardown) BEFORE any retry.
```

**Load-bearing:** none of `boot_and_verify` / `assert_tool_callable` / `preserve_artifacts` may
be wrapped in a `run_in_background` task, and the AgentBob drive must be spawned
`run_in_background: false` (synchronous). That is the entire anti-stall guarantee.

## Edit — emit the DRIVE-NOW banner at `run_arm.sh`'s READY summary (line ~60)

Current final summary line in `run_arm.sh` (verbatim):

```bash
echo "READY arm=$ARM sandbox=$SB seed_loaded_ok=$OK max_turns=$MT brain_gated=$GB bob_prompt=$RUN/bob_prompt.txt"
```

Add ONE line immediately after it so the orchestrator sees the loud nudge at the exact decision
point (choose either form):

```bash
# Option A — call the harness function (single source of the banner text):
python -c "from tests.harness import drive_now_banner; print(drive_now_banner(arm='$ARM', port=$PORT, run_dir='$RUN'))"
```
```bash
# Option B — if you don't want a python call in the hot path, echo the literal headline:
printf '%s\n' "▶ DRIVE NOW, IN THIS SAME TURN — do not end your turn to wait"
```

Option A is preferred (the banner text stays defined in one tracked place,
`tests.harness.foreground.DRIVE_NOW_BANNER_HEADLINE`, and carries the surfaces + rule body).
This is the only `run_arm.sh` edit; it is additive and does not change boot behavior.

## Why this handoff is not load-bearing for correctness
If the banner edit is skipped, runs still work — the banner is a nudge, not a gate. The anti-stall
guarantee is delivered by the foreground shape of the three helpers + the `RUN-ORCHESTRATION.md`
contract, which ship in the tracked harness regardless of whether `run_arm.sh` is edited.
