# companion-emergence

A framework for building persistent, emotionally aware AI companions that live locally, remember their people, dream at night, and grow over time.

Private during development. Public release at author's discretion.

## Source of truth

The framework's design lives in [`docs/source-spec/`](docs/source-spec/). Read that before reading any code. It is the map the rebuild navigates by.

## Status

Active local-first prototype. The bridge, chat/session flow, memory ingest, soul review, health checks, and test/lint gates are implemented enough for private development and local smoke testing.

Known incomplete surfaces remain intentional and visible: see [`docs/roadmap.md`](docs/roadmap.md) and [`docs/release-checklist.md`](docs/release-checklist.md) before any public/tagged release.

## Installing the desktop app (NellFace)

> **Status (2026-05-08):** `v0.0.1-alpha` exists as a private alpha
> release with pre-built macOS arm64, Linux x86_64, and Windows x86_64
> assets. macOS Intel users should build from source until an x86_64
> DMG appears. The release workflow also supports manual retries for an
> existing `v*.*.*` tag.

NellFace ships **unsigned** — binaries are integrity-sealed (ad-hoc
code-signed on macOS) but we don't pay for an Apple Developer ID or a
Microsoft code-signing cert. See [`INSTALL.md`](INSTALL.md) for the
per-platform bypass dance:

- **macOS** — right-click → Open (one-time), or
  `xattr -d com.apple.quarantine /Applications/NellFace.app`
- **Windows** — More info → Run anyway in the SmartScreen dialog
- **Linux** — no signing dance; just `dpkg -i` or `chmod +x` the AppImage

### Build from source today

```bash
git clone <this repo>
cd companion-emergence
uv sync --all-extras   # python deps + pytest/ruff/coverage
cd app
pnpm install           # node deps
pnpm tauri build       # produces a local .app / .deb / .msi
```

> Note: `uv sync` alone leaves a runtime-only environment without
> pytest / ruff / coverage. Use `--all-extras` (or `--extra dev`) for
> contributor work.

Locally-built artifacts inherit your machine's keychain trust so they
launch without warnings.

**One external prerequisite on every platform**: the
[`claude`](https://docs.claude.com/en/docs/claude-code/setup) CLI on
`PATH` for the LLM provider. NellFace shells out to it using your
existing Claude Code subscription.

### How the brain stays alive

On macOS, the wizard's first run installs a user LaunchAgent at
`~/Library/LaunchAgents/com.companion-emergence.supervisor.<persona>.plist`.
That agent owns the brain — heartbeat, dreams, reflex, research,
memory ingest — and runs whether the desktop app is open or not.
launchd restarts it on crash and starts it at login. **Closing,
uninstalling, or rebuilding the desktop app does not touch the
supervisor.** The app is a thin viewer that reads
`bridge.json` and connects to the running brain over localhost.

If you skipped the wizard auto-install (an existing persona, or a
launchctl error during first run), open the **Connection** panel and
click "install launchd supervisor" — that wires the same agent
without touching your data. Or run `nell service install --persona <name>`
from terminal.

On Linux and Windows the equivalent service abstraction exists as a
systemd user unit / Windows Task Scheduler task. Those code paths are
unit-tested and bundled, but still need more live-host validation than
macOS before we call them primary.

Useful commands:

```bash
nell service status --persona nell      # is the LaunchAgent loaded?
nell service doctor --persona nell      # preflight checks
nell service uninstall --persona nell   # remove the agent
nell daemon-state refresh --persona nell  # repair stale residue cache
```

Operational quick check: `nell status --persona nell` reports local persona/config/memory/bridge state without contacting live providers.

Memory inspection is local-only and explicit: use `nell memory list --persona nell`, `nell memory search "query" --persona nell`, and `nell memory show <memory_id> --persona nell`.

Reference implementation: Nell (migrates from the NellBrain OG project). Other personas arrive as forkers build them.
