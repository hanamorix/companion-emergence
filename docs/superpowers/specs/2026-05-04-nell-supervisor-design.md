# `nell supervisor` — canonical bridge lifecycle command

**Date:** 2026-05-04
**Status:** Design — pending implementation plan
**Owner:** Hana
**Replaces / supersedes:** `nell bridge` (kept as deprecating alias until v0.1)
**Roadmap link:** §2.1 of `docs/roadmap.md` — "Replace stubs with useful CLI commands one at a time"

## Why

Two facts collide today:

1. The source spec (`docs/source-spec/2026-04-21-framework-rebuild-design.md` lines 196 and 621) names `nell supervisor` as *the* canonical lifecycle command for the per-persona bridge daemon. The fork-and-customise flow described for new users runs `nell supervisor start --persona their_name`.
2. `nell supervisor` is currently a stub at `brain/cli.py:52-68` that exits non-zero with "not implemented yet". Meanwhile, SP-7 shipped `nell bridge start|stop|status|tail-events --persona X` which already wires `brain/bridge/daemon.py:cmd_start/cmd_stop/cmd_status/cmd_tail`.

The April-30 audit (`docs/audits/2026-04-30-full-code-audit.md` lines 177–195) flagged this surface as a P2 issue: *"`status` and `supervisor` are operationally important names; stubbing them as success is risky UX."* `status` was implemented on May 3. `supervisor` is the remaining half.

This design closes the audit finding *and* aligns the operator-facing CLI with the spec, with a planned cleanup of the `nell bridge` alias before v0.1 ships publicly.

## What ships

A canonical `nell supervisor` subcommand with six actions: `start`, `stop`, `status`, `restart`, `tail-events`, `tail-log`. Same args and exit codes as the existing `nell bridge` actions, plus two new behaviours (`restart` is atomic stop+start; `tail-log` reads the bridge log file cross-platform).

`nell bridge` is kept as a deprecating alias that still works but prints a one-line warning to stderr. Removed in v0.1.

## CLI surface

```
nell supervisor start    --persona NAME [--idle-shutdown MIN] [--client-origin {cli,tauri,tests}]
nell supervisor stop     --persona NAME [--timeout SECONDS]
nell supervisor status   --persona NAME
nell supervisor restart  --persona NAME [--idle-shutdown MIN] [--client-origin {cli,tauri,tests}] [--timeout SECONDS]
nell supervisor tail-events  --persona NAME
nell supervisor tail-log     --persona NAME [-n LINES] [-f]
```

**Defaults preserved from `nell bridge`:**

- `--idle-shutdown 30` minutes (0 = never)
- `--client-origin cli` (choices: `cli`, `tauri`, `tests`)
- `--timeout 180.0` for `stop`

### `restart` semantics

1. Print `stopping bridge...` to stdout.
2. Call `cmd_stop` to completion. Full timeout, full clean shutdown — never kill mid-ingest. The non-daemon supervisor thread (`brain/bridge/supervisor.py:run_folded`) waits on graceful exit by design.
3. If stop returned 0 (clean stop OR no bridge was running — `cmd_stop` collapses both into 0), proceed to start.
4. If stop returned anything non-zero (today the only case is exit 1: bridge ignored SIGTERM and timed out — i.e. wedged), print `restart aborted: stop failed (exit N)` to stderr and bail with stop's exit code. Do NOT spawn a second bridge while the first is wedged.
5. Print `starting bridge...` to stdout.
6. Call `cmd_start`. Restart's exit code is `cmd_start`'s return code (whether 0, 1, or 2).

Two-phase progress output is intentional — operators want to see where in the cycle a hang lives.

### `tail-log` semantics

- Reads `get_log_dir() / f"bridge-{persona}.log"`.
- `-n` defaults to 50 (last N lines). Must be a non-negative integer. `-n 0` means print zero historical lines — useful only when paired with `-f` to watch only newly-written lines. Negative values rejected with exit 2 (argparse-level validation).
- `-f` opens follow mode: block, emit new lines as written, exit 0 on `KeyboardInterrupt`.
- If the log file does not exist: exit 1, print `bridge log not found at <path> — has the supervisor ever started?` to stderr.
- Cross-platform implementation: pure Python with `pathlib` + a small loop. No `subprocess.run("tail")` — Windows CI does not have `tail`.

### `tail-events` semantics

Identical to current `nell bridge tail-events`. WebSocket subscriber to `/events`, prints JSON lines until ctrl-c.

## Architecture

Single-file change to `brain/cli.py`. No new modules. The lifecycle logic lives where it already lives (`brain/bridge/daemon.py`); the new surface binds a second argparse subparser tree to the same handlers.

| File | Change |
|---|---|
| `brain/cli.py` | Add `supervisor` subparser with 6 actions; add deprecation alias for `bridge`; remove `"supervisor"` from the stub list at line 52 |
| `brain/bridge/daemon.py` | Add `cmd_restart` and `cmd_tail_log`; both functions live next to existing `cmd_start/stop/status/tail` |
| `tests/unit/brain/test_cli_supervisor.py` (new) | Argparse wiring tests for all six actions |
| `tests/unit/brain/bridge/test_daemon.py` | Extend with `cmd_restart` and `cmd_tail_log` behaviour tests |
| `tests/unit/brain/test_cli.py` | Extend with deprecation-alias behaviour tests for `nell bridge` |
| `CHANGELOG.md` | Added + Deprecated entries (text below) |
| `docs/roadmap.md` | Strike through `nell supervisor` in §2; add v0.1 removal item to §3; prepend "Done recently" entry |
| Code comments in `brain/cli.py` | Cross-references between deprecation helper and `_chat_handler` |

**Data flow:** unchanged. Same `state_file.BridgeState`, same `/health` readiness check, same dirty-shutdown recovery via `run_recovery_if_needed`, same `acquire_lock` pattern, same per-persona spawning. The supervisor command is a *naming surface*, not new behaviour — except `restart` (sequential stop-then-start, gated on stop success) and `tail-log` (file tail of the existing log).

**Boundary discipline:** `cmd_restart` and `cmd_tail_log` go in `brain/bridge/daemon.py` next to their siblings. The CLI layer stays thin (argparse → handler dispatch only). `daemon.py` is currently 272 lines; adding two more handlers does not push it past the 500-line threshold.

## Deprecation alias for `nell bridge`

Keep the existing `nell bridge` subparser tree exactly as it is, but route every action through a thin wrapper that prints a one-line warning to stderr before calling the real handler.

```python
# Deprecation alias for `nell bridge` — to be removed in v0.1.
# Note: `nell chat` auto-spawn (see `_chat_handler`) uses brain.bridge.daemon
# internals directly, not this CLI surface, so removing the alias does
# not break chat. See docs/roadmap.md §3.
def _deprecated_bridge(real_handler):
    def wrapped(args):
        print(
            "warning: 'nell bridge' is deprecated; use 'nell supervisor' instead. "
            "This alias will be removed in v0.1.",
            file=sys.stderr,
        )
        return real_handler(args)
    wrapped.__name__ = f"deprecated_{real_handler.__name__}"
    return wrapped
```

Wired with `b_start.set_defaults(func=_deprecated_bridge(cmd_start))`, etc.

**What stays identical:**

- Args, flags, defaults, exit codes — bit-for-bit.
- Help text in `nell --help` still lists `bridge` (intentional — discoverable for muscle-memory operators; the warning teaches them).
- Same `func` chain, same daemon handlers, same observable behaviour minus the stderr line.

**What does NOT change in this PR:**

- `nell bridge` is not hidden, not silenced, not suppressed.
- No deprecation timer or env var to silence the warning. One release of nagging is the whole point.

**Removal in v0.1 (separate PR, tracked in roadmap §3):**

- Delete the bridge subparser block in `brain/cli.py`.
- Delete `_deprecated_bridge` helper.
- CHANGELOG under "Removed" with a `nell bridge → nell supervisor` migration line.

## Error handling and exit codes

Inherit the existing `daemon.py` exit-code contract. Do not invent new codes.

**Existing `cmd_start` codes (verified at `brain/bridge/daemon.py:136-198`):**

- `0` — bridge started, `/health` responded within 5s.
- `1` — persona dir missing, OR `/health` did not respond in 5s (orphan child killed, inspection path printed).
- `2` — bridge already running, OR lockfile already held.

**`cmd_stop`, `cmd_status`, `cmd_tail`:** keep their current contracts unchanged.

**`cmd_restart` (new):**

- `0` — stop returned 0 (clean stop or nothing-was-running, both folded into 0 by `cmd_stop`) AND start succeeded.
- Stop's exit code — stop returned non-zero (today: 1 for SIGTERM-timeout / wedged bridge); bail before start; print `restart aborted: stop failed (exit N)` to stderr; restart returns whatever stop returned. Refusing to start over a wedged bridge is the whole point.
- Start's exit code — stop returned 0 but start failed (e.g. start returns 1 for `/health` timeout, or 2 for a racing already-running bridge); surface start's stderr unchanged; restart returns whatever start returned.

**`cmd_tail_log` (new):**

- `0` — printed N lines successfully, or follow mode exited cleanly via ctrl-c.
- `1` — log file not found at expected path; print helpful message to stderr.
- `1` — IOError reading log; print error path + reason to stderr.
- Follow mode catches `KeyboardInterrupt` and returns 0.

**Persona-not-found:** uniform across all six actions. Same error and exit code (1) as `cmd_start` produces today. `cmd_restart` and `cmd_tail_log` validate `persona_dir.exists()` before doing anything destructive.

**Deprecation warning:** writes to stderr, never stdout. Exit code is unaffected — the wrapping handler returns the real handler's code unchanged. Important so any script piping `nell bridge status` output through `jq` or `grep` does not get a warning line in its data stream.

## Testing approach

| Layer | File | Tests |
|---|---|---|
| CLI argparse wiring | `tests/unit/brain/test_cli_supervisor.py` (new) | All 6 actions parse correctly with required + optional args; missing `--persona` errors out; help text lists all six actions |
| `cmd_restart` logic | `tests/unit/brain/bridge/test_daemon.py` (extend) | Stop succeeds → start runs; stop fails → start NOT called, exit 1 propagates; nothing-running → start runs cleanly; both-phase progress lines emitted |
| `cmd_tail_log` logic | `tests/unit/brain/bridge/test_daemon.py` (extend) | Reads last N lines from a fixture log; `-f` follow mode receives newly-written lines; missing log file → exit 1 with helpful message; IOError surfaces; KeyboardInterrupt → exit 0 |
| Deprecation alias | `tests/unit/brain/test_cli.py` (extend) | `nell bridge start` still parses + dispatches identically; warning text appears on stderr; warning text does NOT appear on stdout; exit code matches the underlying handler |
| Cross-platform `tail-log` | `tests/unit/brain/bridge/test_daemon.py` | Works on macOS, Linux, Windows runners — pure Python implementation, no shell `tail` |

**Mocking strategy:** follow the established pattern. Patch `brain.bridge.daemon.spawn_detached` for start tests, patch `state_file.is_running` + `state_file.read` for status tests, use a real `tmpdir` log file for `tail-log` tests (no mock — too lossy for line-handling).

**Test budget:** ~15-20 new tests. Existing CLI tests average 8-12 per command surface.

**TDD ordering for the implementation plan:** tests for new behaviour (`cmd_restart` two-phase logic, `cmd_tail_log` edge cases, deprecation warning placement on stderr) get written first. Argparse wiring tests can come after handlers exist since argparse is declarative.

**Explicitly NOT included:** integration tests that spawn a real bridge subprocess. The existing daemon tests already cover spawn-and-handshake at the integration layer. `nell supervisor` adds no new integration risk over `nell bridge` — same daemon code path.

## Documentation deltas

### `CHANGELOG.md` (under existing `## 0.0.1 - Unreleased`)

Under `### Added`:

> - `nell supervisor` lifecycle command — canonical operator surface for the per-persona bridge daemon. Actions: `start`, `stop`, `status`, `restart`, `tail-events`, `tail-log`. Wraps the existing bridge daemon implementation; same args, same exit codes, plus sequential `restart` (stop-then-start, gated on stop success) and cross-platform `tail-log`.

Under a new `### Deprecated` subsection:

> - `nell bridge` — use `nell supervisor` instead. The alias still works and forwards to the new command, but prints a deprecation warning to stderr. Will be removed in v0.1.

### `docs/roadmap.md`

§2 "Replace stubs": cross out `nell supervisor` from the suggested-order list with a `(shipped 2026-05-04)` trailing note. `nell rest` and `nell works` move to positions 1 and 2.

§3 "Firm up packaging" — add a new bullet under "Before public/tagged release":

> - Remove the deprecated `nell bridge` alias. Removing it does not affect `nell chat` auto-spawn — chat uses `brain.bridge.daemon` internals directly (inside `_chat_handler`), not the CLI surface.

"Done recently" — prepend:

> - Implemented `nell supervisor` as the canonical bridge lifecycle command, with `nell bridge` kept as a deprecating alias until v0.1.

### `brain/cli.py` code comments

Above the `_deprecated_bridge` helper:

```python
# Deprecation alias for `nell bridge` — to be removed in v0.1.
# Note: `nell chat` auto-spawn (see `_chat_handler`) uses brain.bridge.daemon
# internals directly, not this CLI surface, so removing the alias does
# not break chat. See docs/roadmap.md §3.
```

Above the chat auto-spawn block inside `_chat_handler` in `brain/cli.py`:

```python
# Note: spawns the bridge by importing daemon directly, not by shelling
# out to `nell bridge`/`nell supervisor`. The deprecated bridge alias
# (removed in v0.1) does not affect this path.
```

### `README.md`

No change in this PR. README already says "bridge daemon and HTTP/WebSocket API" without naming the command — stays accurate. Updating README belongs with the v0.1 removal PR when the alias actually goes away.

### Source spec

`docs/source-spec/2026-04-21-framework-rebuild-design.md` already calls the command `nell supervisor`. No spec change — we are catching up to the spec, not amending it.

## Non-impact notes

These are facts the author of the v0.1 removal PR will need so they do not panic:

- **`nell chat` auto-spawn (`_chat_handler` in `brain/cli.py`)** imports `brain.bridge.daemon` and calls `cmd_start` / `state_file.read` directly. It does NOT shell out to `nell bridge` or `nell supervisor`. Removing the deprecated `nell bridge` subparser block has zero effect on chat behaviour.
- The supervisor *thread* (`brain/bridge/supervisor.py:run_folded`) is unrelated to this naming change — it lives inside the bridge process and is started by the daemon, not by the CLI. The CLI surface is `nell supervisor` (process lifecycle); the in-process supervisor thread is unchanged.
- No persona data, no on-disk state, no MCP tool surface, no event schema is touched.

## Out of scope for this PR

- Multi-persona "start them all" mode. Spec line 196 is single-persona; YAGNI for now.
- Pause / resume the supervisor thread (`run_folded`) without killing the bridge. No real user story; would expose internal mechanism.
- Removing `nell bridge`. Tracked separately for v0.1.
- Updating the README's command-naming surface. Belongs with the v0.1 removal.
- Implementation of `nell rest` and `nell works` — those are next on the roadmap and need their own brainstorming pass first (the roadmap explicitly says "clarify before implementing").
