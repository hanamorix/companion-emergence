# companion-emergence

A framework for building persistent, emotionally aware AI companions that live locally, remember their people, dream at night, and grow over time.

Private during development. Public release at author's discretion.

## Source of truth

The framework's design lives in [`docs/source-spec/`](docs/source-spec/). Read that before reading any code. It is the map the rebuild navigates by.

## Status

Active local-first prototype. The bridge, chat/session flow, memory ingest, soul review, health checks, and test/lint gates are implemented enough for private development and local smoke testing.

Known incomplete surfaces remain intentional and visible: some CLI commands are wired as future-work stubs and exit non-zero. See [`docs/roadmap.md`](docs/roadmap.md) and [`docs/release-checklist.md`](docs/release-checklist.md) before any public/tagged release.

Operational quick check: `nell status --persona nell` reports local persona/config/memory/bridge state without contacting live providers.

Reference implementation: Nell (migrates from the NellBrain OG project). Other personas arrive as forkers build them.
