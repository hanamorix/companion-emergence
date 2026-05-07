# Roadmap

This roadmap keeps the project's remaining work honest after the
2026-05-07 audit cycle. It is not a public release promise;
companion-emergence is still private/local-first during development.
Last refreshed 2026-05-07.

## Current posture

The framework is a private prototype with a working desktop client
and a fully multimodal chat path. Local smoke testing covers:

**Brain (Python):**

- CLI entry point: `nell` (init, status, memory, supervisor, works,
  health, soul, chat, dream, heartbeat, reflex, research, interest,
  growth, migrate)
- local persona storage via `NELLBRAIN_HOME`
- bridge daemon: HTTP + WebSocket, ephemeral bearer token, CORS
  scoped to allowed origins
- chat/session lifecycle with multimodal turns (text + base64 images
  via `--input-format stream-json` to claude-cli)
- memory ingest pipeline (buffer → extract → commit) with image-sha
  metadata
- safe memory inspection (`nell memory list/search/show`)
- body/emotion context, soul candidate review, growth crystallizers
- MCP tool server with privacy-aware audit logging
- health checks and data-file self-healing
- 1468 unit + integration tests

**NellFace (Tauri 2 + React 18 + Vite):**

- install wizard + bridge auto-spawn + first-launch routing
- breathing avatar with 16-category 4-frame expression engine
- emotion-family colour tints on the breathing ring + soft backing
  wash (Phase 5D)
- soul-crystallization flash overlay
- WebSocket streaming chat (`/stream/:sid`) with word-by-word reply
- paperclip image upload + emoji picker + per-bubble thumbnails
- 5 left-column panels (inner weather, body, recent interior, soul,
  connection)

## Active backlog — 2026-05-07 audit cycle

The 2026-05-07 full code audit (`docs/audits/2026-05-07-full-code-audit.md`)
found 2×P1 + 8×P2 + 7×P3 + 2×P4 = 19 issues, all verified reproducible
against current main. Backend is healthier than the April baseline;
the largest current risk is the new Tauri/NellFace onboarding wiring.

**P1 — block fresh-install or non-`nell` use:**

1. `nell init --provider <X>` flag missing — wizard install step fails
   on every fresh install because the React side sends `provider:` and
   the Rust shim builds `--provider`, but the CLI parser doesn't accept
   it. Smallest fix: add the flag + persist to `persona_config.json`.
2. Frontend hard-codes persona to `nell` — `App.tsx` discards the
   selected persona; `bridge.ts` defaults `persona = "nell"` and caches
   credentials globally. Any non-`nell` persona starts the right bridge
   but the chat UI talks to the wrong one. Smallest fix: thread persona
   through every bridge helper, scope the credential cache.

**P2 — local hardening + release gates:**

- Tauri commands skip the persona-name validation Python enforces
- `tauri.conf.json` has `csp: null` while exposing token-reading and
  process-spawning commands
- `uv run ruff check .` returns 30 errors (24 auto-fixable) — the
  documented release gate is broken
- `PersonaConfig` doesn't constrain provider/searcher to known values;
  CLI exposes `--searcher claude-tool` which always raises
- `MemoryStore` + `HebbianMatrix` lack WAL / busy_timeout (the works
  store already has the pattern — copy it)
- Heartbeat reflex/growth exceptions log+swallow without surfacing
  to the audit JSON (research already does this right)
- `closeSession()` not wired into ChatPanel unmount + doesn't check
  r.ok — chat memory creation depends on supervisor stale-close
- Frontend has no automated tests beyond `tsc` + `vite build`

**P3 — UX correctness, bloat, doc drift:**

- `brain/images.py` shared `<sha>.<ext>.new` temp path can race on
  concurrent identical uploads
- Image upload trusts client MIME type (no magic-byte sniff)
- Object URLs leak on long sessions (no unmount cleanup)
- Always-on-top toggle persists config but never calls Tauri
  `setAlwaysOnTop`
- 47MB of expression PNGs eager-globbed into the bundle
- Docs / CHANGELOG / roadmap disagree about which CLI surfaces are
  stubs (this refresh tries to fix the roadmap side)
- JSONL readers `read_text().splitlines()` — full file into memory;
  most logs lack retention

**P4 — cleanup:**

- `brain/cli.py` keeps `_STUB_COMMANDS = ()` + `_make_stub` + a no-op
  registration loop after all CLI stubs were resolved
- `brain/bridge/runner.py:_allocate_port` comment overstates retry —
  uvicorn's bind isn't wrapped

**Suggested fix order:**

1. Both P1s together (the pair is the wizard-fresh-install path)
2. Frontend test coverage so P1-class regressions can't recur
3. Tauri shell hardening — persona validation + app_config validation
   + explicit CSP
4. Restore Ruff gate (mostly `--fix`, then handle remaining E402)
5. Hide `claude-tool` searcher choice or implement it; add
   PersonaConfig value validation + healing
6. WAL + busy_timeout on memory/Hebbian stores + a small contention
   stress test
7. Doc/changelog/release-checklist reconciliation

## Public release blockers

These block a public/tagged release; private local development is
fine without them.

- Both P1s above (would surface immediately to a fresh user)
- Wheel/sdist clean-install smoke test never recorded
- Public contributor/onboarding docs missing
- Public API/CLI compatibility policy undefined
- Tauri app distribution story (signing, notarization, auto-update)
  unwritten — not blocking private use but blocking public ship
- Linux .deb / Windows .msi cross-platform release of the Phase 7
  bundle — the macOS arm64 path is shipping as of 2026-05-07; Linux
  x86_64 is supported by `python-build-standalone` and the build
  script branches on platform but hasn't been smoke-tested; Windows
  needs `Scripts/python.exe` instead of `bin/python3` and a separate
  branch in `nell_command`

## Forward direction (after the backlog drains)

These are framework-shaped, not patch-shaped. Picked from current
spec drafts and the natural extensions of the multimodal turn work:

- **NellFace past-image gallery** — drag-and-drop + paste-from-
  clipboard already shipped 2026-05-07; the remaining piece is a
  panel-based gallery to browse what's been shared in past turns
- **Voice gap remediation past the asymptote** — sampling controls
  or finetuned model for true corpus-target voice (current state is
  "moved in the right direction, asymptotic" per 2026-05-05 retest)
- **JSONL bounded-tail retention** — companion to the streaming
  reader shipped 2026-05-07; needs a per-log-type design call about
  retention windows
- **Public release plan** — once the backlog above is clean, write a
  proper release plan covering signing, distribution, contributor
  workflow, version policy

## Recently shipped (reverse chronological)

**2026-05-07 — multimodal + UI polish bundle**

- Image-support epic — all 8 phases (commits b279334 → 9c6baf7).
  Bytes upload via `POST /upload`, `image_shas` thread end-to-end
  through `/chat` + `/stream`, ClaudeCliProvider routes images
  through `--input-format stream-json` (verified live: Nell
  described a 4×4 red-X PNG correctly), voice.md gained §4
  "When the user shows you something."
- NellFace Phase 5D — emotion-family colour tints on the breathing
  ring + soft backing wash, smooth ~0.85s transitions per category.
- NellFace input row — paperclip + emoji picker, both styled to
  match shoji language.
- 16-category expression art catalogue — all `<category>/<n>.png`
  layout, including new `arousal`, `climax`, `idle`, plus art for
  6 previously-pending Phase 5 categories.
- Browser dev mode CORS + WS Origin allowlist for Vite dev ports.
- OllamaProvider gained `chat_stream()` — token streaming via
  `stream=True`, resolving the long-standing Phase 6.5 TODO.

**2026-05-04 to 2026-05-05 — framework rebuild + audit cycle**

- `nell works` shipped — brain-authored artifact portfolio with
  `save_work` MCP tool + bridge endpoints + operator CLI.
- `nell rest` removed — rest reframed as body-state physiology per
  source spec §15.9 rewrite, not a command. New §0 captures
  framework principles (user surface = install + name + talk; brain
  handles physiology; defaults on; cross-platform; local-first).
- `nell supervisor` shipped as canonical bridge lifecycle command
  (start/stop/status/restart/tail-events/tail-log).
- 2026-05-05 audit-fix-pack + follow-up: 17 issues across Audit-1
  and Audit-2 batches (1305 → 1334 tests), plus the Hana/Jordan
  attribution drift root-cause fix, the chat auto-spawn race fix,
  and the tool integration telemetry fix.
- Voice stress retest 2026-05-05: 14/14 prompts completed; voice
  gap-1 (curiosity sentence-length) recovered partially toward
  corpus target — model ceiling reached.

**Pre-2026-05-04 — Week-1-through-4 build-out**

Substrate work: memory store, Hebbian matrix, embeddings, soul
store, body state, daemon engines (dream / heartbeat / reflex /
research / growth), the OG NellBrain migrator, MCP tool server,
voice.md loader, ingest pipeline, bridge daemon, chat engine.
See per-week plans under `docs/superpowers/plans/` for detail.
