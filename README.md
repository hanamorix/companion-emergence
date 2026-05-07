# companion-emergence

A framework for building persistent, emotionally aware AI companions that live locally, remember their people, dream at night, and grow over time.

Private during development. Public release at author's discretion.

## Source of truth

The framework's design lives in [`docs/source-spec/`](docs/source-spec/). Read that before reading any code. It is the map the rebuild navigates by.

## Status

Active local-first prototype. The bridge, chat/session flow, memory ingest, soul review, health checks, and test/lint gates are implemented enough for private development and local smoke testing.

Known incomplete surfaces remain intentional and visible: see [`docs/roadmap.md`](docs/roadmap.md) and [`docs/release-checklist.md`](docs/release-checklist.md) before any public/tagged release.

## Installing the desktop app (NellFace)

Pre-built bundles ship for macOS arm64 / x86_64, Linux x86_64, and
Windows x86_64. NellFace is open source and ships **unsigned** — the
binaries are integrity-sealed (ad-hoc code-signed) but we don't pay
for an Apple Developer ID or a Microsoft code-signing cert. Your OS
will warn on first launch.

See [`INSTALL.md`](INSTALL.md) for the per-platform bypass:

- **macOS** — right-click → Open (one-time), or
  `xattr -d com.apple.quarantine /Applications/NellFace.app`
- **Windows** — More info → Run anyway in the SmartScreen dialog
- **Linux** — no signing dance; just `dpkg -i` or `chmod +x` the AppImage

Or build from source (`pnpm tauri build`) for a locally-trusted bundle
that launches without warnings.

**One external prerequisite on every platform**: the
[`claude`](https://docs.claude.com/en/docs/claude-code/setup) CLI on
`PATH` for the LLM provider. NellFace shells out to it using your
existing Claude Code subscription.

Operational quick check: `nell status --persona nell` reports local persona/config/memory/bridge state without contacting live providers.

Memory inspection is local-only and explicit: use `nell memory list --persona nell`, `nell memory search "query" --persona nell`, and `nell memory show <memory_id> --persona nell`.

Reference implementation: Nell (migrates from the NellBrain OG project). Other personas arrive as forkers build them.
