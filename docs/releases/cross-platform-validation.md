# Cross-platform validation — living checklist

macOS arm64 is the primary target and the only platform with a dev host + CI
that compiles the `#[cfg(target_os="windows")]` arm. Windows and Linux changes
therefore ship **test-verified-on-macOS-only** and sit as open VALIDATION GAPS
until a designated user confirms post-ship. This is structural (no runner), so
it can't be closed in CI — but the gaps must not be *forgotten*. `release-review`
reconciles this list each release.

## Standing gates (run before every tag)

- `bash scripts/release_preflight.sh <tag>` — the CI validate gate locally.
- `python scripts/lint_windows_pitfalls.py` — static check for the known Windows
  subprocess/kill footguns. Review each finding against the gotchas below.

## Open validation gaps (from the deferred ledger)

Each row: what shipped test-verified-only, which platform confirms it, and the
deferred-ledger item. Move a row to "Confirmed" (with the version + validator)
once a real machine validates it.

| Area | Platform | Ships as | Ledger |
|---|---|---|---|
| Windows bridge shutdown (pythonw / schtasks / `--force`) | Windows | test-verified on macOS; a Windows user confirms post-ship | 39 |
| True Windows SCM host (vs Task Scheduler) | Windows | deferred | 38 |
| Linux x86_64 real-machine click-through (systemd `--user` install, install-shape) | Linux (Kubuntu 26.04 validator) | code shipped v0.0.15-alpha.3; awaiting manual pass | 7 |
| Kindled-link cross-machine (peer-to-peer over a real relay) | any 2 machines | EXPERIMENTAL, unvalidated cross-machine | 51 |
| Brain clean-login spawn/stdin flow (`start_brain_login` 40-line URL scan) | Windows/Linux | macOS-live only; 40-line cap may miss a longer banner | 65 |

## Windows gotchas the linter guards (don't reintroduce)

- **Bridge listens on `127.0.0.1`, not `tauri.localhost`** (the latter broke CORS
  on Windows, v0.0.12-alpha.3).
- **Rust `nellbrain_home()` must equal Python `platformdirs`** exactly (v0.0.12-alpha.2).
- **Force UTF-8 encoding on every subprocess** — Windows default isn't UTF-8
  (v0.0.12-alpha.4). (Linter check 1.)
- **Heavy payloads via `--system-prompt-file` + stdin, never argv** — Windows
  `CreateProcess` caps the joined command line at 32,767 chars (WinError 206,
  v0.0.12-alpha.5).
- **`os.kill(pid, SIGTERM)` is TerminateProcess on Windows** — no cleanup, never
  writes `shutdown_clean`. Route through `BridgeShutdownController` (v0.0.33).
  (Linter check 2.)
- **Bare `print()` under pythonw** (Task Scheduler, no console) — `sys.stdout` is
  None → the write raises before logging is up → silent exit 1 (v0.0.37).
  Neutralised globally by `cli._harden_std_streams`; keep that hardening.
- **Windows Task Scheduler launches `pythonw.exe` windowlessly** + the provider
  spawns with `CREATE_NO_WINDOW` (no console flash).
