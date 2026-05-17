# Promote Nell supervisor to a macOS launchd agent

Date: 2026-05-08
Status: **Implemented and live-validated** (macOS arm64, Hana's box, 2026-05-08)
Scope: macOS first. Linux `systemd --user` and Windows service/task support come later behind the same `nell service` abstraction.

## Status

The brain runs as `com.companion-emergence.supervisor.<persona>` under
the user's LaunchAgents. The .app is now a thin viewer that reads
`bridge.json` and connects. Validated end-to-end: `nell service install`
brings the supervisor up, the .app launches without spawning a
duplicate, and killing the .app does not affect the supervisor.

Surfaces shipped:

- `brain/service/launchd.py` — plist build / install / uninstall /
  status / doctor.
- `nell service {print-plist | install | uninstall | status | doctor}`
  CLI subcommands.
- `nell supervisor run` foreground entry point with `--client-origin
  launchd`.
- Tauri command `install_supervisor_service(persona)` →
  `installSupervisorService` (frontend) → wired into both the wizard
  (auto-install on first run) and ConnectionPanel (one-click migration
  for existing personas).

## Surprises hit during live validation

Three things were not obvious from the design and would have wasted
time on a second machine without this record.

**1. `claude` lives in `~/.local/bin`, not on the system path.**
The doctor's `claude_cli` check failed even after the plist was correct
because `DEFAULT_LAUNCHD_PATH` was a static string covering only
Homebrew + macOS system dirs. Anthropic's installer puts the binary at
`~/Users/<you>/.local/bin/claude`, which launchd doesn't inherit (it
strips the user's shell PATH). Fix: `DEFAULT_LAUNCHD_PATH` is now
computed at plist-build time and includes `~/.local/bin`.

**2. CLI's `--env-path` default silently shadowed the library default.**
`brain/cli.py:_add_service_common` had its own hardcoded copy of the
same path string as the library's `DEFAULT_LAUNCHD_PATH`. When the
library was updated, `nell service` subcommands kept using the old
value because argparse fixed the default at parser-build time. Symptom:
`nell service doctor` and `nell service install` produced different
plist contents than `build_launchd_plist` did directly. Fix: lazy-import
`DEFAULT_LAUNCHD_PATH` inside `_add_service_common` so the CLI tracks
the library.

**3. Source edits don't reach a running supervisor.** Python caches
imported modules in `sys.modules`. Editing `brain/engines/daemon_state.py`
to bump `_SUMMARY_MAX_CHARS` from 250 → 1500 had zero effect on the
running supervisor — the value was captured at import time. The fix
is `launchctl kickstart -k gui/$(id -u)/<label>`. Add this to the
runbook for any code change that needs to take effect on a live
LaunchAgent.

Existing daemon_state.json entries also need a refresh when a cap or
format changes, since they were written under the old rules. The
cleanest path is a `nell heartbeat refresh-state --persona <name>`
subcommand that reads the most recent reflex/dream/research memory
and overwrites the corresponding `last_*` field. (Hand-patched via
inline Python on 2026-05-08 — codify this next.)


## Goal

Make the brain a real user-level OS service and make the desktop app a thin viewer.

Today, the app starts the bridge on demand through Tauri by invoking:

```text
nell supervisor start --persona <name>
```

That command spawns a detached Python bridge process. The bridge process owns a folded supervisor thread that closes stale sessions and runs heartbeat/growth/research cadences. This works, but the brain lifecycle is still coupled to the app boot path. If the app never opens, the brain does not start. If the app moves toward a thin viewer, it should not own Python runtime startup at all.

Target state:

- A user LaunchAgent starts Nell at login and keeps it alive.
- The service owns the bridge and folded supervisor lifecycle.
- The app reads `bridge.json`, connects to localhost, and renders state/chat only.
- `nell init`, `nell service install`, and `nell upgrade` are the official lifecycle and upgrade surfaces.
- Deleting or upgrading the app does not silently kill the brain service.

## Current architecture to preserve

Important current surfaces:

- `brain/bridge/daemon.py` implements `cmd_start`, `cmd_stop`, `cmd_status`, `cmd_restart`, `cmd_tail`, and `cmd_tail_log`.
- `cmd_start` creates a detached child with `spawn_detached(...)`, then waits for `/health`.
- `brain/bridge/runner.py` is the actual long-lived bridge process. It writes `bridge.json` with pid, port, auth token, and dirty/clean shutdown state, then runs uvicorn.
- `brain/bridge/server.py` starts `run_folded(...)` as a non-daemon supervisor thread inside the bridge process.
- `brain/paths.py` centralizes state, cache, and log dirs using `platformdirs`, with `NELLBRAIN_HOME` override.
- `app/src-tauri/src/lib.rs` currently invokes `nell supervisor start` from `ensure_bridge_running`.
- `app/src/appConfig.ts` and `app/src/App.tsx` await `ensureBridgeRunning(...)` during boot.
- The app reads per-persona `bridge.json` to discover port and bearer token, then uses HTTP/WS on `127.0.0.1`.

Keep these invariants:

- `bridge.json` remains the local discovery and auth-token handoff file.
- The bridge binds only to `127.0.0.1`.
- Bearer token stays ephemeral per service start and is never embedded in launchd plist.
- Persona data remains under `get_home()/personas/<name>`.
- Logs remain under `get_log_dir()` unless a launchd-specific stdout/stderr path is explicitly chosen.
- Clean shutdown still drains sessions and runs close-trigger heartbeat.

## Key design decision: launchd must own a foreground process

Do **not** point launchd at `nell supervisor start`.

`supervisor start` is a launcher. It forks/detaches `brain.bridge.runner` and exits. If launchd owns that command, launchd will see the job exit immediately and will either think the job is done or KeepAlive-loop the launcher while orphan bridge children keep running.

Add a foreground command instead:

```text
nell supervisor run --persona NAME [--client-origin launchd] [--idle-shutdown 0]
```

Semantics:

- Does not detach.
- Acquires the same per-persona startup lock.
- Runs dirty-shutdown recovery before serving.
- Calls the same foreground bridge runner path currently used by `brain.bridge.runner`.
- Writes and protects `bridge.json` exactly as today.
- Exits only when the bridge exits.
- Converts SIGTERM from launchd into the existing graceful shutdown path.

Then LaunchAgent `ProgramArguments` can point at this foreground command and launchd supervises the actual service process.

## Proposed CLI surface

Keep `nell supervisor` for process-level bridge semantics. Add `nell service` for OS-managed lifecycle.

```text
nell supervisor run --persona NAME [--client-origin launchd] [--idle-shutdown 0]

nell service install   --persona NAME [--force] [--runtime current|PATH] [--claude-path PATH]
nell service uninstall --persona NAME [--keep-logs] [--purge-bridge-state]
nell service start     --persona NAME
nell service stop      --persona NAME
nell service restart   --persona NAME
nell service status    --persona NAME [--json]
nell service logs      --persona NAME [-n LINES] [-f]
nell service doctor    --persona NAME
nell service print-plist --persona NAME
nell upgrade           [--channel stable|nightly] [--dry-run] [--rollback]
```

Why separate `service` from `supervisor`:

- `supervisor` remains portable and developer-friendly: start/stop a bridge process now.
- `service` is OS integration: write plist, bootstrap/bootout, status through launchctl, validate environment.
- Later Linux and Windows can reuse `nell service` with different backends.

Exit-code posture:

- `0`: success or already in requested state.
- `1`: runtime/service failure.
- `2`: invalid args or invalid local environment.
- Machine-readable details available through `--json` where useful.

## LaunchAgent model

Use a per-user LaunchAgent, not a root LaunchDaemon.

Reasons:

- No root installer or privileged helper required.
- Runs inside the logged-in user's context and can access user data, keychain, and Claude CLI auth.
- Fits local-first privacy expectations.
- Easier uninstall: remove one plist from `~/Library/LaunchAgents` and bootout the user job.

One LaunchAgent per persona:

```text
~/Library/LaunchAgents/com.companion-emergence.supervisor.<persona>.plist
```

Recommended label:

```text
com.companion-emergence.supervisor.<persona>
```

Persona names are already constrained to a small safe grammar in setup/Tauri. The service backend should still validate and escape defensively before using a persona in a launchd label or filename.

Skeleton plist:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.companion-emergence.supervisor.nell</string>

  <key>ProgramArguments</key>
  <array>
    <string>/ABSOLUTE/PATH/TO/nell</string>
    <string>supervisor</string>
    <string>run</string>
    <string>--persona</string>
    <string>nell</string>
    <string>--client-origin</string>
    <string>launchd</string>
    <string>--idle-shutdown</string>
    <string>0</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <dict>
    <key>Crashed</key>
    <true/>
    <key>SuccessfulExit</key>
    <false/>
  </dict>

  <key>WorkingDirectory</key>
  <string>/Users/HANA</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>

  <key>StandardOutPath</key>
  <string>/Users/HANA/Library/Logs/companion-emergence/supervisor-nell.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/HANA/Library/Logs/companion-emergence/supervisor-nell.err.log</string>
</dict>
</plist>
```

Notes:

- Do not include bearer tokens in the plist.
- Persist `NELLBRAIN_HOME` in `EnvironmentVariables` only when the user has explicitly configured it.
- Either pin `PATH` at install time or write a `CLAUDE_PATH`/provider config if provider code grows support for an absolute Claude CLI path. LaunchAgents do not inherit interactive shell startup files.
- Use `RunAtLoad=true` so login starts the brain.
- Use KeepAlive on crash, but not on intentional clean exit. Service `stop` should bootout or disable the job rather than merely SIGTERM a still-enabled crashing job.
- Set `--idle-shutdown 0`; an OS service should not self-exit just because the app is closed.

## launchctl operations

Implement in a macOS backend module, for example `brain/service/launchd.py`.

Commands:

```text
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/<label>.plist
launchctl bootout   gui/$UID/<label>
launchctl kickstart -k gui/$UID/<label>
launchctl print     gui/$UID/<label>
```

Install flow:

1. Validate macOS and user session.
2. Validate persona exists.
3. Validate `nell` runtime path is absolute and executable.
4. Validate Claude CLI can be found, or warn clearly that chat will fail until fixed.
5. Stop any legacy detached bridge for the persona with `nell supervisor stop`.
6. Render plist to a temp file and atomically replace the target plist.
7. `chmod 0644` plist, create log dir.
8. `launchctl bootout` old label if loaded, ignoring not-loaded errors.
9. `launchctl bootstrap gui/$UID <plist>`.
10. `launchctl kickstart -k gui/$UID/<label>`.
11. Wait for `bridge.json` and `/health` with bearer token.
12. Print status and log paths.

Uninstall flow:

1. `launchctl bootout gui/$UID/<label>`, ignoring not-loaded.
2. Remove plist.
3. Optionally remove stale `bridge.json` with `--purge-bridge-state`; never delete persona data by default.
4. Leave logs by default; remove only with an explicit flag.

Status flow:

- Combine launchd status and bridge status.
- Report:
  - plist installed/not installed
  - launchd loaded/not loaded
  - pid from launchd
  - `bridge.json` pid/port/auth state
  - `/health` reachable/unreachable
  - current runtime path
  - log paths
  - environment warnings such as missing Claude CLI

Doctor flow:

- Validate plist XML round-trip.
- Validate ProgramArguments executable exists.
- Validate `NELLBRAIN_HOME` path if set.
- Validate persona files exist.
- Validate log dir permissions.
- Validate `claude` can run under the plist PATH.
- Validate bridge health.
- Suggest exact repair command.

## Runtime and upgrade model

The long-term clean model is a stable user runtime outside the app bundle:

```text
~/Library/Application Support/companion-emergence/
  runtimes/
    0.0.1/
      bin/nell
      python-runtime/...
    0.0.2/
      bin/nell
      python-runtime/...
  current -> runtimes/0.0.2
  personas/
  app_config.json
```

The LaunchAgent should point at:

```text
~/Library/Application Support/companion-emergence/current/bin/nell
```

or at a small stable shim that execs current:

```text
~/Library/Application Support/companion-emergence/bin/nell
```

The shim approach is safer because the plist does not need rewriting on every upgrade.

`nell upgrade` flow:

1. Download or build the target runtime into `runtimes/<version>.partial`.
2. Verify checksums/signature/provenance.
3. Install wheel and bundled Python runtime.
4. Run smoke checks in the new runtime:
   - `nell --version`
   - `nell status --persona <name>`
   - `nell supervisor run --persona <temp> --dry-run` or equivalent service smoke if added
5. Atomically move to `runtimes/<version>`.
6. Update `current` pointer or shim metadata atomically.
7. Restart installed services one at a time.
8. Health-check each restarted persona.
9. If any health check fails, roll back `current`, restart the old runtime, and keep failure logs.

This lets the app become truly thin. The app bundle no longer needs to embed Python as the authoritative brain runtime. It can be upgraded, moved, or deleted without changing the service runtime.

Transitional option:

- Phase 1 can point launchd at the bundled app runtime path under `/Applications/Companion Emergence.app/.../python-runtime/bin/nell` to prove the launchd model.
- Do not stop there. That couples service survival to the app bundle path. The stable runtime + shim is the real target.

## Thin app behavior

Target app boot:

1. Read `app_config.json` for selected persona.
2. Read `personas/<persona>/bridge.json`.
3. Call `/health` using bearer token.
4. If healthy, render viewer/chat.
5. If not healthy, show a service status screen:
   - "Nell service is not installed"
   - "Nell service is installed but stopped"
   - "Nell service is starting"
   - "Nell service is unhealthy"
   - Include exact CLI repair commands and log path.

The app should not spawn Python as the normal path once service mode is stable.

First-run choices:

- **Strict thin viewer:** app cannot create personas. User runs `nell init --persona NAME` and `nell service install --persona NAME`, then opens app.
- **Assisted viewer:** app can call an already installed system `nell` CLI to run init/install, but does not ship or own the runtime.
- **Transitional app:** app keeps bundled runtime for wizard/init only until `nell upgrade` and service install are fully polished.

Recommendation: use transitional app first, then move to assisted/strict thin viewer once `nell upgrade` is reliable.

## Files and modules to add/change

Phase 1 foreground runner:

- `brain/bridge/daemon.py`
  - Add `cmd_run(args)` or helper for foreground serving.
  - Reuse recovery, lock, state-file, and readiness semantics.
- `brain/bridge/runner.py`
  - Accept `--persona` in addition to `--persona-dir`, or expose a callable `run_bridge_foreground(...)` that `cmd_run` can call.
  - Accept `--client-origin launchd` or a generic `service` value.
  - Accept `--idle-shutdown-seconds 0` as disabled.
- `brain/cli.py`
  - Add `nell supervisor run`.
- Tests:
  - Unit tests for argparse wiring.
  - Unit tests that `cmd_run` does not call `spawn_detached`.
  - Signal/shutdown tests where practical.

Phase 2 launchd backend:

- `brain/service/__init__.py`
- `brain/service/launchd.py`
  - label generation
  - plist rendering via `plistlib`
  - bootstrap/bootout/kickstart/print wrappers
  - health wait helpers
  - doctor checks
- `brain/cli.py`
  - Add `nell service ...` commands.
- Tests:
  - Plist generation golden tests.
  - Persona label sanitization tests.
  - `launchctl` command wrapper tests with subprocess mocked.
  - Install/uninstall dry-run tests.

Phase 3 app transition:

- `app/src-tauri/src/lib.rs`
  - Replace normal `ensure_bridge_running` spawning with service/health detection.
  - Optionally add `get_service_status` command.
  - Keep spawn fallback behind a development flag only.
- `app/src/appConfig.ts` and `app/src/App.tsx`
  - Show service status screen instead of indefinite bridge startup.
- Frontend tests:
  - Installed/healthy path.
  - Not installed path.
  - Installed but unhealthy path.

Phase 4 upgrade path:

- `brain/upgrade.py` or `brain/service/upgrade.py`
  - Runtime install/download/verify.
  - Current pointer/shim management.
  - Service restart/rollback.
- `brain/cli.py`
  - Add `nell upgrade`.
- Tests:
  - Atomic current-pointer update.
  - Rollback on failed health.
  - Keeps existing personas untouched.

Phase 5 docs/release:

- `INSTALL.md`
  - Add `nell init`, `nell service install`, `nell service status`, `nell service uninstall` flow.
- `docs/release-checklist.md`
  - Add launchd install/uninstall smoke tests.
- `docs/roadmap.md`
  - Track macOS service as shipped and Linux/Windows service as future.
- Release notes:
  - Make clear that dragging the app to trash does not uninstall the background service. Users must run `nell service uninstall --persona NAME`.

## Logging plan

Current detached bridge logs go through `get_log_dir() / f"bridge-{persona}.log"`.

Launchd has `StandardOutPath` and `StandardErrorPath`. Use one of two strategies:

1. **Separate launchd stdout/stderr logs:**
   - `supervisor-<persona>.out.log`
   - `supervisor-<persona>.err.log`
   - Keep bridge application logging unchanged.

2. **Unified log file:**
   - Point both stdout and stderr at `bridge-<persona>.log`.
   - Simpler operator story, but mixed output can be noisy.

Recommendation: separate launchd stdout/stderr initially, while keeping `nell service logs` smart enough to show both launchd and bridge logs.

Add log rotation before public release. launchd does not rotate these files automatically. Minimum viable rotation:

- Rotate if a service log exceeds 5 MB.
- Keep 3 backups.
- Rotate on service start and during `nell service doctor`.

## Signing and distribution implications

LaunchAgent avoids privileged install and avoids SMAppService entitlements for now. It still changes distribution assumptions:

- If the plist points inside the `.app`, moving/deleting/upgrading the app can break the service.
- If the plist points at a user runtime, the runtime becomes the signed/trusted artifact that matters.
- Open-source unsigned builds can install a user LaunchAgent without root, but macOS may still warn on the app itself.
- A future Developer ID path can optionally switch app-managed login items to SMAppService, but that is not required for the CLI-driven service model.

For the open-source path, prefer:

- user LaunchAgent
- stable user runtime
- checksum-verified runtime downloads
- explicit CLI install/uninstall
- no background privileged helper

## Migration plan for existing users

No forced migration at first.

Stage A, opt-in:

```text
nell service install --persona nell
```

The installer stops any legacy detached bridge first, installs launchd service, starts it, waits for health, and prints viewer instructions.

Stage B, app detects service:

- If service installed, app never spawns `supervisor start`.
- If service not installed, app can keep the old spawn path behind a compatibility flag.

Stage C, service default:

- New install docs tell users to run `nell init` then `nell service install` before opening the app.
- App no longer starts Python by default.

Stage D, thin viewer:

- App bundle stops shipping the full Python runtime.
- `nell upgrade` owns brain runtime updates.

Uninstall warning:

- Dragging the app to trash does not remove the service or persona data.
- Official uninstall is:

```text
nell service uninstall --persona nell
# optional, destructive, separate:
rm -rf "$HOME/Library/Application Support/companion-emergence/personas/nell"
```

## Test and validation matrix

Automated tests:

- Unit: `supervisor run` argparse and no-detach behavior.
- Unit: plist generation, label validation, env rendering, log path rendering.
- Unit: launchctl wrapper commands with subprocess mocked.
- Unit: service install/uninstall dry-run behavior.
- Unit: app service-status command if added.
- Existing Python suite must remain green.
- Frontend tests for service status screen.

Mac manual smoke tests:

1. Fresh `NELLBRAIN_HOME=$(mktemp -d)`.
2. `nell init --persona launchd_test --fresh --voice-template skip`.
3. `nell service install --persona launchd_test`.
4. Verify `launchctl print gui/$UID/com.companion-emergence.supervisor.launchd_test`.
5. Verify `nell service status --persona launchd_test` shows launchd loaded and bridge healthy.
6. Quit the app, wait 2 minutes, verify service remains healthy.
7. Reboot or log out/in, verify service restarts.
8. Open app, verify it connects without spawning Python.
9. `nell service restart --persona launchd_test`, verify app reconnects or shows transient starting state.
10. `nell service uninstall --persona launchd_test`, verify launchd unloaded, plist removed, persona data preserved.
11. Run `nell service install` with a missing Claude CLI PATH and verify doctor gives a clear warning.
12. Run upgrade dry-run and rollback smoke once `nell upgrade` exists.

CI posture:

- Plist generation and CLI tests can run everywhere.
- launchctl integration should be macOS-only and probably manual or nightly at first. GitHub macOS runners may not provide a normal GUI launchd session, so do not block all PRs on bootstrap until proven stable.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| LaunchAgent lacks shell PATH, so `claude` is missing | Installer validates `claude`; plist includes explicit PATH; doctor reports exact fix. |
| launchd crash loop | KeepAlive only on crash; add backoff if needed; logs/doctor expose repeated exits. |
| Service points into moved/deleted app bundle | Transitional only; real target is stable user runtime/shim. |
| Multiple personas consume resources | One agent per persona, explicit install per persona, status shows active agents. Later add `nell service list`. |
| Logs grow forever | Add rotation before public release. |
| App cannot first-run without Python runtime | Transitional app keeps init/install path until CLI runtime install is smooth. |
| launchctl behavior differs across macOS versions | Use `bootstrap gui/$UID`, `bootout`, `kickstart`, `print`; add manual matrix for current supported macOS versions. |
| Dirty shutdown during upgrade | Upgrade restarts one service at a time, waits for clean stop, health-checks, and rolls back on failure. |
| Tokens leak through plist/logs | Never write token into plist; avoid query params; sanitize logs; rely on existing owner-only `bridge.json` permissions. |

## Open decisions

1. Should the service label use `com.companion-emergence.supervisor.<persona>` or the Tauri bundle identifier prefix? Pick one before implementation and keep it forever.
2. Should `client_origin` gain a new `launchd` value, or should service use `cli`? New value is clearer but touches type choices and tests.
3. Does the first service release keep the app-bundled runtime as a transitional source, or do we build the stable user runtime first?
4. Should install capture the current PATH, write a conservative default PATH, or store an absolute Claude CLI path in persona config/provider config?
5. Should the app have buttons that invoke `nell service start/install`, or should it remain a strict viewer and show CLI instructions only?
6. How many old runtimes should `nell upgrade` keep for rollback?

## Recommended implementation order

1. Add `nell supervisor run` foreground mode and tests.
2. Add `brain/service/launchd.py` plist generation and mocked launchctl tests.
3. Add `nell service print-plist`, `install --dry-run`, and `doctor` before real bootstrap.
4. Add real install/start/stop/status/uninstall commands.
5. Manual macOS smoke-test with temp `NELLBRAIN_HOME`.
6. Teach app to detect service status and stop spawning when service is installed.
7. Add docs for opt-in service install.
8. Build stable user runtime and `nell upgrade` with rollback.
9. Make service the default install path.
10. Remove Python runtime responsibility from the app bundle.
