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
| Memory-free `claude` working directory (#122) — on Windows both candidates (`%LOCALAPPDATA%\Temp`, `%LOCALAPPDATA%\hanamorix\companion-emergence`) sit under `%USERPROFILE%`, so a profile-level `.claude\CLAUDE.md` still reaches the companion; the bridge WARNs at start. Validator probe: from the directory the bridge logs, run `claude -p --model haiku "If any text in your context contains <a string unique to your global CLAUDE.md>, quote that line; else reply NONE"` and report. | Windows | fix shipped for macOS/Linux; Windows tracked in #252 | 122 |
| Windows bridge shutdown (pythonw / schtasks / `--force`) | Windows | test-verified on macOS; a Windows user confirms post-ship | 39 |
| In-app brain update — Check → Update → Use the release brain (#286). Never run live yet: the publisher is live (`brain-main` and a kept `brain-main-dryrun`, both signed and verified, 2026-09-28), so the only blocker left is an app at `min_bundle_version` 0.0.43 (an app at 0.0.42 is never offered one). Steps in #309. | macOS, Windows, Linux | unit/contract/CI-tested (overlay e2e on all three runners; minisign proof on the #307 PR run); **run at the 0.0.43 release review** | #309 |
| True Windows SCM host (vs Task Scheduler) | Windows | deferred | 38 |
| Updater `scripts/update.sh` (#179) — bash; the bundled Windows runtime ships no bash | Windows | bundled Windows installs update in the app (Check for updates → brain row, #286 — live validation pending) or with `nell update`, both into the overlay (proved by `scripts/overlay_e2e.sh` on windows-2022); `scripts/update.sh` stays bash for source/dev installs. #255 closed by #286 slice 4 | — |
| Linux x86_64 real-machine click-through (systemd `--user` install, install-shape) | Linux (Kubuntu 26.04 validator) | code shipped v0.0.15-alpha.3; awaiting manual pass | 7 |
| systemd install test `LoadState=loaded` + `UnitFileState=enabled` assertions (#160 / #164) | Linux with a live `systemd --user` | reasoned, not executed: self-skips on macOS and on GitHub ubuntu runners; first executes on the Kubuntu validator's run | 164 |
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
