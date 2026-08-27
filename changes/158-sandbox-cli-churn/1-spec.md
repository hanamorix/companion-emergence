# 1 — Spec: SandboxLeak guard false-positives on the orchestrator CLI's own `~/.claude` churn (#158)

## Problem

`tests/harness/sandbox.py` fingerprints a set of "guarded roots" (including the real
`~/.claude`) before a run and re-checks after; any change to a guarded root is a leak signal.
The `~/.claude` root is **content-hashed** (`_hash_critical` → `True`), so each entry is
`(size, mtime_ns, sha256)`.

During every orchestrator-driven run, the `claude` CLI (the process *driving* the harness, and
its own subprocess invocations) rewrites two top-level files under the developer's real
`~/.claude`:

- `policy-limits.json` — Anthropic-server-pushed enforcement/policy config.
- `remote-settings.json` — server-pushed remote settings (observed content `{}`).

These are **CLI/server-managed config**, not companion state. They are rewritten with a bumped
`mtime` even when their bytes are unchanged (see Evidence). Because `mtime_ns` is part of the
fingerprint tuple, the mtime bump alone flips the `~/.claude` fingerprint and the guard reacts.

This is a benign false positive: the sandboxed companion's own `claude` state is redirected to
the tempdir via `CLAUDE_CONFIG_DIR` (`_subprocess_env`, `brain/bridge/provider.py`; the
respecting guard `if "CLAUDE_CONFIG_DIR" not in env:` is at `provider.py:175`), so
`~/.claude/policy-limits.json` and `~/.claude/remote-settings.json` are the
**orchestrator's** real config, not something the companion escaped to. It surfaced across all 8
P0-2 baseline arms.

### How it currently manifests (interaction with the option-(c) downgrade)

This branch already carries #104's option-(c) downgrade (commit `8c11516b`): a diff **confined
to `~/.claude` alone** is downgraded from a fatal `SandboxLeak` to a `RuntimeWarning`; any diff
also touching another guarded root still raises. When only these two files change, the diff is
`~/.claude`-only, so today it emits a **spurious `RuntimeWarning`** every run — loud, repeated
noise that trains reviewers to ignore the one signal that is supposed to be rare. (On a build
predating the downgrade it would instead be a fatal raise; the fix removes the source either
way.)

## Evidence (empirical, this box, 2026-08-27)

Snapshot `~/.claude` top-level file `(size, mtime)`, run one `claude -p --model haiku "pong"`,
re-snapshot. Exactly two files changed, both mtime-only (size identical):

```
policy-limits.json    214 bytes  mtime 1787748261 -> 1787852570
remote-settings.json    2 bytes  mtime 1787748261 -> 1787852570
```

No other top-level `~/.claude` file changed (`.credentials.json`, `settings.json`,
`settings.local.json`, `CLAUDE.md`, `history.jsonl`, `.last-cleanup`, etc. were untouched;
`.last-cleanup` / `history.jsonl` are already on the F4 session-log exclude list anyway). This
confirms the churn set is exactly these two files and is complete for the observed CLI version.

## Fix (what to build) — CONTENT-AWARE mtime-insensitive fingerprint (amended after stage-3)

**Amendment (orchestrator decision, post stage-3 red-team, 2026-08-27):** do NOT fully name-prune
these two files. The stage-3 reviewer verified (against `brain/bridge/provider.py`) that the
sandboxed Canary CAN write an absolute `~/.claude/...` path — `generate`/`chat`/`chat_stream`
run `--dangerously-skip-permissions` with no tool disallow-list — so a *full* prune would drop
these two files from the option-(c) "warn" floor to **silent** on a genuine content write, below
the floor the owner ratified as "warn." That is a detection loss the orchestrator declined to
take.

Instead, make these two files' fingerprint entries **mtime-insensitive but content-sensitive**:
record `(size, sha256)` with the `mtime_ns` slot nulled, so

- a benign **mtime-only bump with identical bytes** (the measured false positive) leaves the
  entry unchanged → **silent** (bug fixed), while
- a **genuine content change** to either file still flips `sha256` → the `~/.claude`-only diff →
  the **same option-(c) `RuntimeWarning`** every other `~/.claude` path gets (detection floor
  preserved; raise-vs-warn semantics untouched).

This is a **detection-scope narrowing limited to the mtime axis of exactly two top-level paths** —
not a blanket `~/.claude` ignore, and not a change to what the guard *does* with a diff. Because
these files are **server-pushed CLI config** (a distinct concept from the F4 "session-runtime
logs"), they get their **own** dedicated constant and a helper that returns their absolute paths,
passed as a new `mtime_insensitive=` argument to `_fingerprint` for the `~/.claude` root only.
`_CLAUDE_SESSION_LOG_DIRS/FILES` are left unchanged (the F4 "Canary-can't-write" mechanism and
`test_af2_exclusion_set_unchanged` stay accurate).

Top-level only: the match is by exact resolved path, so a same-basename file in a `~/.claude`
subtree (e.g. a hypothetical `~/.claude/plugins/policy-limits.json`) keeps its normal
mtime-sensitive entry and stays fully guarded.

## Out of scope / non-goals

- **Raise-vs-warn semantics are unchanged.** The option-(c) downgrade stays exactly as-is. This
  change only removes a false-positive *source*; it does not re-litigate what a genuine
  `~/.claude` diff does. If the red-team argues the semantics must change, STOP and escalate to
  the owner (`to: "main"`) — not a runner decision.
- No blanket `~/.claude` ignore.
- No production (`brain/**`) change. Harness-only: `tests/harness/sandbox.py` + its unit tests.
- No edits to gitignored `hunts/.../live-run/` files or the main clone.

## Expected touched files

- `tests/harness/sandbox.py` — add the CLI-housekeeping-file constant + union it into
  `_claude_session_log_excludes()` (and a short justification comment).
- `tests/unit/harness/test_sandbox_isolation.py` — add the positive/negative discrimination
  tests (and any assertion pinning the new constant).

## Rejected alternative (documented)

**Full name-pruning** (exclude both files from the fingerprint entirely, like `history.jsonl`) —
was the original stage-1/2 draft. Rejected at stage-3: it drops these two paths from the
option-(c) "warn" floor to fully silent on a genuine content write, and the reviewer verified the
Canary *can* make such a write. Content-aware (above) fixes the same false positive at the cost
of ~6 extra lines while keeping the "warn" floor, and does not touch raise-vs-warn semantics — so
it needs no owner semantics-escalation.
