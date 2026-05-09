# Changelog

Notable user-facing changes per release. The framework is pre-1.0 —
breaking changes can land in any release, and the runtime ships
unsigned binaries until the project is stable enough to justify code
signing costs. See [`docs/roadmap.md`](docs/roadmap.md) for what's on
deck and [`docs/release-checklist.md`](docs/release-checklist.md) for
what each release has to clear.

## 0.0.3-alpha — 2026-05-09

Windows-only emergency fix.

- **`uv trampoline failed to canonicalize script path` on first
  launch.** The bundled `Scripts/nell.exe` was a uv trampoline
  launcher with the GitHub runner's absolute path to `python.exe`
  baked in. Replaced with a relocatable `Scripts/nell.bat` that
  resolves the bundled python via `%~dp0..\` (path-of-the-bat). No
  changes to macOS or Linux behaviour. Windows users on `0.0.2-alpha`
  should re-download `0.0.3-alpha`.

## 0.0.2-alpha — 2026-05-09

First public release.

- Same framework runtime as 0.0.1 — the bump marks the transition
  from private development to OSS distribution, not a behavioural
  change.
- Pre-built bundles for macOS arm64, Linux x86_64, and Windows
  x86_64 attached to the GitHub release. Intel macOS users build
  from source (`pnpm tauri build` from `app/`) until a reliable
  Intel runner is available.
- The wizard's `nell-example` voice template is now a generic
  Nell archetype intended to be edited; the canonical Nell that
  the framework was developed against lives in private and isn't
  shipped.

## 0.0.1 — private alpha

Iteration window before public release. The project's design
substrate, brain, bridge, chat / session flow, soul module, dream
engine, heartbeat orchestrator, reflex engine, research engine,
memory store, Hebbian edges, creative voice fingerprint, OG
NellBrain migrator, and NellFace desktop app all landed during
this period.

Highlights from the iteration:

- **Plan C launchd / systemd-user / Task-Scheduler service
  backends.** First-launch installs a user-scoped supervisor so
  the brain survives `.app` quit / relaunch cycles. macOS arm64
  is the most-tested path; Linux + Windows backends are
  unit-tested and bundled but pre-live-host validation.
- **Cross-platform Phase 7 release pipeline.** GitHub Actions
  matrix builds the desktop bundle on three runners (macos-14,
  ubuntu-22.04, windows-2022) using a portable
  python-build-standalone runtime, attaches the bundles to the
  release directly from each platform job, and computes
  `SHA256SUMS-<platform>.txt` for verification.
- **Soul module + crystallization workflow.** Soul candidates are
  proposed by the daemon, surfaced for review, and crystallize as
  load-bearing permanent memories that the persona's voice
  template can reference.
- **emergence-kit auto-importer.** `nell migrate --source
  emergence-kit --install-as <name>` reads a kit's
  `memories_v2.json` + `soul_template.json` + `personality.json`
  and seeds a fresh persona, no manual JSON wrangling.
- **Bridge + provider hardening.** Bearer-subprotocol auth on
  WebSocket; constant-time token compare; redacted MCP audit
  logs; bridge state files chmod 0700 on POSIX; explicit
  `ws.close(code=1000)` after the streaming `done` frame.
- **Image support end-to-end.** Typed `ContentBlock` union
  replacing flat `content: str`; sha-addressable
  `<persona_dir>/images/`; `claude-cli --image` passthrough;
  `/upload` endpoint separate from `/chat`.

For the full pre-public iteration log including audit-cycle
findings and per-PR breakdown, see the project's git history.
