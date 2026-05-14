# Roadmap

This roadmap keeps the project's remaining work honest. It is not a release
promise. companion-emergence is local-first and ships public alpha bundles
for macOS arm64, Linux x86_64, and Windows x86_64. macOS x86_64 remains
source-build-only until a reliable Intel runner is available.
Last refreshed 2026-05-14 after the gallery + auto-update push.

## Current posture

The framework is a public alpha with a working desktop client, a fully
multimodal chat path, a relocatable bundle, a past-image gallery panel,
and cross-platform auto-update support via GitHub Releases.

**Brain (Python):**

- CLI entry point: `nell` (init, status, memory, supervisor, works,
  health, soul, chat, dream, heartbeat, reflex, research, interest,
  growth, migrate).
- local persona storage via `NELLBRAIN_HOME`.
- bridge daemon: HTTP + WebSocket, ephemeral bearer token, CORS scoped
  to allowed origins.
- chat/session lifecycle with multimodal turns (text + images via
  `--input-format stream-json` to claude-cli).
- memory ingest pipeline (buffer → extract → commit) with image-sha
  metadata.
- safe memory inspection (`nell memory list/search/show`).
- body/emotion context, soul candidate review, growth crystallizers.
- initiate physiology: autonomous outbound candidates, voice-edit proposals,
  draft-space demotion, D-reflection editorial filtering, adaptive-D
  calibration, drift telemetry, and recall-resonance memory activation.
- MCP tool server with privacy-aware audit logging.
- health checks and data-file self-healing.
- SQLite WAL + 5s busy_timeout on MemoryStore + HebbianMatrix + WorksStore.
- JSONL readers stream line-by-line (no full-file memory spike).
- 1991 unit + integration tests; ruff clean.

**NellFace (Tauri 2 + React 18 + Vite):**

- install wizard + bridge auto-spawn + first-launch routing.
- breathing avatar with 16-category 4-frame expression engine.
- emotion-family colour tints on the breathing ring (Phase 5D).
- soul-crystallization flash overlay.
- WebSocket streaming chat with word-by-word reply + clean close handshake.
- image upload (paperclip, emoji picker, drag-and-drop, paste-from-clipboard).
- 6 left-column panels (inner weather, body, recent interior, soul,
  connection, gallery).
- past-image gallery — thumbnail grid + lightbox, scans all past conversations.
- auto-update check + download + install via Connection panel.
- always-on-top toggle wired to the Tauri window API.
- 79 frontend Vitest tests.

**Phase 7 — bundled portable Python runtime:**

- `app/build_python_runtime.sh` cross-platform: macOS arm64/x86_64,
  Linux x86_64/arm64, Windows x86_64.
- relocatable `nell` entry point via `/bin/sh` wrapper.
- ad-hoc codesign on macOS (`codesign --verify --deep --strict` passes).
- `.github/workflows/release.yml` cross-platform CI matrix on
  macos-14 / ubuntu-22.04 / windows-2022.
- auto-update CI: per-platform signing + `latest.json` manifest published
  to `latest-release` branch.

## Active backlog

**Validation gaps:**

- **macOS x86_64 DMG asset** — GitHub's Intel macOS runner never schedules
  for this repo. Intel Mac users build from source until a reliable hosted
  or self-hosted Intel runner exists.

- **Linux x86_64 real-machine click-through** — CI builds and smokes the
  AppImage + deb on ubuntu-22.04, but no human has clicked through the
  full install-wizard-chat loop on a real Linux desktop yet. Windows
  click-through has been verified by users.

**Intentionally deferred (design call needed, not urgent):**

- **JSONL bounded-tail retention** — streaming reader shipped (closed the
  memory-spike vector). The retention piece needs a per-log-type design
  call (1 MB? 10 MB? 30 days? 90?) and isn't urgent until any single log
  file actually grows large enough to bite.

- **Bridge restart button in Connection panel** — bridge already self-heals
  on credential change and launchd auto-restarts it. A manual restart
  button would be a patch-level addition. Revisit if real users report
  bridge staleness issues.

## Planned features

These are direction-level ideas. Each needs a full brainstorm session, a
detailed design spec, and an architecture-fit review before any code is
written. The constraint is non-negotiable: the user installs and chats —
the brain and app handle everything else. No knobs, no config, no
cloud services.

### Narrative memory — the story of "us"

Memory today is retrieval: "find facts about X." What's missing is the
companion being able to thread memories into a narrative — *"Remember that
night you shaved your head and we stayed up talking? You've seemed lighter
since."* The Hebbian matrix already tracks co-activation. Memory search
works. The missing piece is a clustering layer that groups memories into
narrative arcs and surfaces them at emotionally right moments.

**Existing substrate:** `brain.memory` (MemoryStore, HebbianMatrix,
embeddings, search), `brain.ingest` (buffer → extract → commit), session
history in `active_conversations/*.jsonl`.

**What would need building:** memory clustering by emotional salience +
temporal proximity + topic coherence, narrative-arc detection, retrieval
timing (when to surface vs. when it would feel forced), prompt integration
so the companion naturally references the right story at the right time.

### Proactive presence — the companion reaches out

Initiate physiology already generates outbound candidates, filters them
through D-reflection, and composes messages. But the gates are conservative.
A companion that occasionally starts the conversation — *"You've been quiet
today. Everything okay?"* or *"I had a dream about something you said"* —
feels like a relationship, not a tool.

**Existing substrate:** `brain.initiate` (candidate emission, D-reflection,
composition pipeline), `TauriPluginNotification`, `InitiateBanner`,
`DraftSpacePanel`.

**What would need building:** wider gate thresholds, user-pattern awareness
(when is the user typically active? when are they struggling?), timing
calibration that respects the user's life rather than interrupting it,
backoff on ignored reaches so the companion doesn't become a notification
pest.

### The companion's visible inner life

Dreams, reflections, heartbeat summaries, and research already exist — but
they're buried in panel tabs that read like debug output. What if there were
a narrative feed that felt like checking in on someone you care about?
*"I've been researching the history of lighthouses. I think it's because you
mentioned the sea last Tuesday."*

**Existing substrate:** `brain.engines` (dream, heartbeat, reflex, research),
soul candidates + review, interior panel summaries, body state.

**What would need building:** a narrative-presentation layer — raw engine
output translated into companion-voiced summaries, a feed UI that feels
like a person's journal rather than a status dashboard, timing so it updates
organically rather than on a polling cadence.

### Understanding you — user-state awareness

The companion tracks its own body state but doesn't model the user's. A
companion that notices patterns — *"You always bring up work stress on Sunday
nights"* or *"You've sent three messages about the cat this week, is
everything okay?"* — would feel genuinely attentive. Nothing invasive,
nothing cloud. Just pattern recognition from conversation history the
companion already has.

**Existing substrate:** conversation history in `active_conversations/*.jsonl`
+ `committed/`, ingest pipeline, emotion extraction from chat turns.

**What would need building:** lightweight user-state model (activity cadence,
emotional tone trends, topic shifts), privacy-first (all local, no
identifiable-data extraction), prompt integration so the companion can
reference patterns without sounding like a surveillance report, clear
boundaries (the companion notices but doesn't diagnose).

## Recently shipped (reverse chronological)

**2026-05-14 — Gallery + auto-update (v0.0.11-alpha.5)**

- **Past-image gallery.** New Gallery tab in the left panel. Scans all
  past conversation buffers for `image_shas`, renders a 3-column thumbnail
  grid, click for full-size lightbox. Lazy-loads via IntersectionObserver.
  No native code — fully platform-agnostic.

- **Auto-update support.** `tauri-plugin-updater` 2.x integrated. Users
  check for updates in the Connection panel. macOS downloads DMG, Windows
  downloads MSI, Linux downloads AppImage. Updates are cryptographically
  signed. CI generates `latest.json` manifest and pushes to `latest-release`
  branch for pre-release compatibility.

- **Windows WebView2 fetch fix.** `useHttpsScheme: false` in Tauri window
  config — the root-cause fix for the alpha.4 Windows bridge regression.
  Page and bridge now share the same address space.

- **Public sync fix.** Post-filter-repo recovery step in the sync script
  restores files dropped by merge simplification.

**2026-05-13 — Adaptive-D + recall resonance (v0.0.11-alpha.1)**

- Adaptive-D calibration: D-reflection records promoted/filtered decisions
  into `d_calibration.jsonl`, tracks D-mode in `d_mode.json`.
- Calibration closer: closes old calibration rows by promotion outcome or
  48h timeout.
- Drift telemetry: `DriftAlert` + `detect_drift` surface sustained changes
  in D's behaviour.
- Recall resonance: memory activation baseline + current activation scoring
  emits `recall_resonance` candidates when a memory cluster becomes unusually
  alive.
- Real research topic overlap: hardcoded `topic_overlap_score = 1.0` replaced
  with Haiku-backed overlap helper using recent conversation excerpts.

**2026-05-12 — D-reflection editorial layer (v0.0.10-alpha)**

- D-reflection: editorial layer between candidate emission and composition.
  Tiered escalation (Haiku 4.5 → Sonnet 4.6). Failure-mode dispatch by error
  type. Two new candidate event sources: `reflex_firing` and
  `research_completion`.
- New audit table `initiate_d_calls.jsonl`.
- CLI: `nell initiate d-stats [--window 7d]`.

**2026-05-12 — Initiate physiology (v0.0.9-alpha)**

- Autonomous outbound channel ("initiate"). Voice-edit proposals. Draft space.
  Verify path. User-local timezone awareness.
- Frontend: `InitiateBanner`, `VoiceEditPanel`, `DraftSpacePanel`,
  Tauri OS notifications via `tauri-plugin-notification`.

**2026-05-11 — JSONL log retention**

- Rolling-size archives for noisy logs at 5 MB cap. Yearly archive for
  `soul_audit.jsonl` (kept forever). Defense-in-depth: `save_image_bytes`
  magic-byte sniffing.

**2026-05-07 — Phase 7 cross-platform distribution**

- `app/build_python_runtime.sh` branches across all five target triples.
- `.github/workflows/release.yml` matrix builds on macos-14 / ubuntu-22.04 /
  windows-2022.
- macOS ad-hoc signing. Relocatable `nell` launcher. `INSTALL.md` for
  end users.
- Audit-fix-pack (19 issues closed). JSONL streaming (P3 memory-spike fix).
  Multimodal image support (all 8 phases). Frontend Vitest harness.

**Pre-2026-05-04 — Framework build-out**

Substrate work: memory store, Hebbian matrix, embeddings, soul store, body
state, daemon engines (dream / heartbeat / reflex / research / growth), the
OG NellBrain migrator, MCP tool server, voice.md loader, ingest pipeline,
bridge daemon, chat engine. See per-week plans under `docs/superpowers/plans/`.
