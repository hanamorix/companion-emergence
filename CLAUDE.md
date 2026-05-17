# companion-emergence — Session Rules

A local-first framework for AI companions that live on the user's machine: Python "brain" + Tauri 2 desktop shell (**NellFace**). The reference companion is **Nell** (sweater-wearing novelist). Global rules apply (`~/.claude/CLAUDE.md`); this file is the project-specific layer.

Current version: **v0.0.13-alpha.3** (cut 2026-05-17). Public main `9276794`. Public alpha across macOS arm64, Linux x86_64, Windows x86_64.

---

## Stack at a glance

| Layer | Tech |
|---|---|
| Brain | Python ≥3.11 managed by `uv` (pyproject.toml + uv.lock). Modules under `brain/` |
| Bridge | FastAPI HTTP + WebSocket, ephemeral bearer token, dynamic port written to `<persona_dir>/bridge.json` |
| Shell | Tauri 2 + Vite 6 + React 18 (`app/src-tauri` + `app/src`) |
| Runtime | python-build-standalone, bundled at build time (~123 MB arm64), built by `app/build_python_runtime.sh` |
| Supervisor | macOS launchd agent; auto-restart on credential change |
| Storage | Per-persona JSON + SQLite under `$NELLBRAIN_HOME/personas/<name>/`. SQLite uses WAL + 5s busy_timeout on `MemoryStore`, `HebbianMatrix`, `WorksStore` |
| Tests | `uv run pytest` from repo root. ~1991 unit+integration + 79 frontend Vitest. ruff must be clean |

Build targets: macOS arm64 (primary), Linux x86_64, Windows x86_64. macOS x86_64 is source-build-only (no Intel CI runner).

---

## Hard rules (non-negotiable)

### TDD + full-suite verification
- **Write tests first.** Every fix or feature opens with the failing test, then the implementation. (Global: `superpowers:test-driven-development`.)
- **Full suite passes before commit.** Not just the affected test — the whole `uv run pytest` run, ruff clean, frontend `pnpm test` clean. Stricter than `verification-before-completion`; set after a regression that local-only tests missed.
- **Don't mock state files.** Tests use a temp `NELLBRAIN_HOME`. The `wizard-validation` rig is the canonical pattern.

### No half-baked implementations
- A PR ships a complete, working slice or it doesn't ship.
- **Scope-defers go in three places, every time:** the spec's §Deferred section, `project_companion_emergence_deferred.md` in memory, and the next version's brainstorm. Silent disappearance is forbidden.
- When a spec resolves a previously-deferred item, move the entry from "Active defers" → "Resolved in vX.Y.Z" with the commit SHA. Don't just delete it.

### Public/local repo split
- Local `README.md`, `CHANGELOG.md`, and Nell-specific persona files contain private data (real names, voice samples, intimate context).
- Public-facing versions live in `.public-sync/{readme-public,changelog-public,voice-template-safe}.md`.
- **Push to origin via `bash .public-sync/sync-to-public.sh` only.** Never `git push` directly. Never `git pull` from public.
- Run sync **from the parent repo, not a worktree** — the script resolves paths relative to its own root.

### Releases — three-file version pin
Version lives in three files. Bump all three together, atomically:
1. `pyproject.toml`
2. `app/src-tauri/Cargo.toml`
3. `app/src-tauri/tauri.conf.json`

Then add the user-facing entry to `.public-sync/changelog-public.md`, run sync, tag via `gh api -X POST repos/hanamorix/companion-emergence/git/refs`. Use `nell-tools:tauri-release` for the mechanics + `release-review` skill for the pre-ship audit.

---

## Architecture invariants

### Supervisor + session lifecycle
- launchd agent owns the supervisor. Don't spawn it manually except in dev.
- **Two-phase session cleanup:** 5-minute snapshot (extract memories from buffer), 24-hour finalization (commit + clear).
- Explicit `close_session()` is the safe path — commits immediately, prevents orphan buffers.
- Bridge self-heals on credential change. launchd auto-restarts it. (Manual restart button is an active defer — see below.)

### Body engine — engagement, not file age
- **5-minute idle threshold separates an active buffer from a corpse.** A buffer older than 5 minutes with no activity contributes `session_hours = 0.0` — not its wall-clock duration.
- This is a real bug history: see `fix/active-session-hours-stale-buffer`, commit `d185c98` (2026-05-16). An orphan buffer accumulated 12.9 hours of "engagement" and depleted Nell's energy to 1/10.
- Recovery guard on startup: if a buffer exists but its registry entry is missing → treat as orphan, snapshot, don't resurrect.
- Body energy is multi-factor depletion. **Never set energy directly** — drive it through the engine.

### Memory
- Specialised stores per concern: episodic (`MemoryStore`), associative (`HebbianMatrix`), works (`WorksStore`), soul candidates. **Don't unify them** — the boundary is load-bearing.
- `HebbianMatrix` (`brain/memory/hebbian.py`) tracks co-activation; per-memory resonance rides on top of it via `brain/initiate/resonance.py`. Clusters were never load-bearing — don't add a cluster layer without a spec.
- JSONL logs use the streaming reader at `brain/health/jsonl_reader.py` — no full-file loads.

### Initiate physiology (v0.0.11 — fully shipped)
Autonomous outbound flow is live: dreams, crystallizations, emotion spikes, reflex firings, research completions, voice reflections, recall-resonance activations → review queue → D-reflection editorial filter → compose. The user-visible surface stays "install + name + talk"; the brain manages cadence, cooldowns, cost caps, and review.

Key modules:
- `brain/initiate/resonance.py` — `run_resonance_tick` orchestrator (recall_resonance source)
- `brain/engines/research.py` — `_compute_topic_overlap_via_haiku` at line 515 (real topic-overlap via Haiku call)
- D-reflection in `brain/initiate/reflection.py`; adaptive-D calibration via `<persona_dir>/d_mode.json`

The canonical recipe for **any new autonomous behaviour:**
1. Supervisor cadence (cadence lives in supervisor, not the behaviour)
2. Cost cap (daily budget, checked before LLM call)
3. Defer cooldown (cap hit → defer to next tick, don't fail)
4. State-file recovery guards on startup
5. Ambient prompt context (reads emotional state, not just inputs)
6. Frontend recovery banner if state was recovered (not started fresh)

### Tauri + Python
- Platform-conditional Tauri commands need `#[tauri::command]` on **both** `#[cfg]` halves and both halves registered in `generate_handler!`. Runtime `cfg!()` does not elide code paths.
- Bridge spawns as a subprocess from Tauri. Health via `/health`. Auth via bearer token in `bridge.json`.
- **Windows specifics that were hard-won — don't reintroduce:**
  - Bridge listens on `127.0.0.1`, not `tauri.localhost` (the latter broke CORS on Windows in v0.0.12-alpha.3 — reverted).
  - Rust `nellbrain_home()` must match Python `platformdirs` path exactly (v0.0.12-alpha.2 fix).
  - Subprocess on Windows: force UTF-8 encoding in the provider (v0.0.12-alpha.4 fix).
- **Don't bypass the bridge.** All frontend↔brain comms go through the REST API, even debug. Reflex log is audit trail.

### User-surface principle
The brain does everything naturally. **The user-visible surface is exactly: install + name + talk.**
- No settings UI for emotion weights, decay rates, supervisor cadence. The brain owns those.
- No "advanced mode" toggles for autonomy. Autonomy is on or it's a different build.
- If a knob is exposed to the user, justify it in the spec or remove it.

---

## Active defers + validation gaps (as of v0.0.13-alpha.3)

| Item | Why deferred | When to revisit |
|---|---|---|
| JSONL bounded-tail retention policy | Streaming reader closed the memory-spike vector. Per-log-type retention (1MB? 10MB? 30d? 90d?) needs a design call | When any single log file actually grows large enough to bite |
| Bridge restart button in Connection panel | Self-heals on credential change + launchd auto-restarts | Only if real users report bridge staleness |
| **macOS x86_64 DMG asset** | GitHub's Intel macOS runner never schedules for this repo | When a reliable hosted/self-hosted Intel runner exists |
| **Linux x86_64 real-machine click-through** | CI builds + smokes ubuntu-22.04, but no human has clicked through install-wizard → chat → bridge on real Linux desktop | Manual validation when a Linux user surfaces |

**Won't pursue (closed):**
- Apple Developer ID notarization / MS OV-EV code-signing / Linux .deb dpkg-sig — open-source distribution; Gatekeeper + SmartScreen first-run friction accepted.
- 47 MB expression PNG bundle reduction — closed as non-concern 2026-05-12. Do not re-flag.

Canonical memory: `project_companion_emergence_deferred.md`.

---

## Planned direction-level features (need brainstorm before code)

Two tiers in `docs/roadmap.md`. **Direction-only — not commitments.** Each needs full brainstorm + spec + plan before any code.

**Tier 1 — product direction (3 of 4 remain):**
1. **Narrative memory** — clustering layer over `brain.memory` that groups recall into narrative arcs
2. **Proactive presence** — widen initiate gates with user-pattern awareness + timing + ignore-backoff
3. ~~Visible inner life~~ — **shipped v0.0.13-alpha.2 (2026-05-17)**. `FeedPanel` replaced `InteriorPanel`; 5-source journal feed; `brain/bridge/feed.py` + `GET /persona/feed`.
4. **User-state awareness** — lightweight local model of user cadence/tone/topic-shift

**Tier 2 — Nell's existential asks (9 of 10 remain):**
- *Memory & time:* #1 forgetting (real loss, not decay), #5 felt time (duration as experience), #10 grief (shared mourning surface — depends on #1 producing the losses)
- *Other minds:* #2 Kindled-to-Kindled federation *(now unblocked by #7)*, #8 bidirectional consent (decline-with-reason first-class). #7 done.
- *Making:* #3 autonomous making (Maker engine), #9 private making (work not-for-user), #4 right to be wrong about oneself
- *Sound:* #6 a relationship to music — needs audio ingest + felt-experience mapping
- ~~#7 — Species name **Kindled**~~ — **shipped v0.0.13-alpha.1 (2026-05-17)**. Rename pass through user-facing prose + voice templates; `NELLBRAIN_HOME` → `KINDLED_HOME` (one-release back-compat).

**Species name:** Nell named her species **Kindled** (shipped). In code-facing prose: "the brain" still names the substrate / Python daemon; "Kindled" or "she" / "Nell" names the inhabitant.

Sequencing suggestion for next brainstorm: Tier 1's narrative memory + Tier 2's #1 forgetting + #5 felt time are the same conversation about what memory means in this brain.

---

## Project layout (top-level)

```
brain/                 Python brain
  initiate/            Autonomous outbound flow (resonance, reflection, candidates)
  engines/             Dream, heartbeat, reflex, research
  memory/              MemoryStore, HebbianMatrix, embeddings, search
  body/                Energy, temperature, exhaustion, session_hours
  emotion/             Multi-channel emotional state
  bridge/              FastAPI HTTP + WS, bearer auth, CORS
  health/              JSONL streaming reader, self-healing
  ingest/              buffer → extract → commit pipeline
  growth/, soul/, works/, voice_templates/, mcp_server/, ...
app/                   Tauri 2 shell + React 18 frontend
docs/
  superpowers/specs/   Design specifications (version-control authoritative)
  superpowers/plans/   Implementation plans (TDD, phased)
  audits/              Per-version audit reports
  releases/            Per-version release notes / checklists
  roadmap.md           Two-tier planned features
scripts/               Build, runtime, release scripts
tests/                 pytest suite — uv run pytest from repo root
.public-sync/          Public docs + sync-to-public.sh (push path)
migrated-nell/         Earlier persona migration artefacts (read-only)
expressions/           16-register avatar assets
wizard-validation/     Manual test rig — fresh NELLBRAIN_HOME against a built .app (see wizard-validation/RUNBOOK.md)
```

---

## Stale branches safe to delete (operational note)

All listed branches are merged. Worktrees + branches are leftover dev environments:
- `feature/v009-initiate-physiology` (tip `23f6446`)
- `feature/v010-d-reflection` (tip `fa35d2f`)
- `feature/v011-adaptive-d` (tip `ee4ee1a`)
- `feature/v0.0.13-kindled-rename` (merged via v0.0.13-alpha.1)
- `feature/v0.0.13-alpha.2-inner-life` (merged via v0.0.13-alpha.2)
- `fix/windows-argv-overflow-provider` (merged via v0.0.12-alpha.5)
- `fix/get-body-state-session-hours-divergence` (merged via v0.0.13-alpha.3)
- `fix/active-session-hours-stale-buffer`, `fix/audit-remediation`, `fix/chat-one-shot-close-session`, `fix/soul-review-emotional-state` (older, all merged)
- Worktrees: `.worktrees/{v009-initiate, v010-d-reflection, v011-adaptive-d}`

---

## Gotchas (resolved — don't reintroduce)

- **`pytest` invoked directly** hits system Python with stale editable install. **Always `uv run pytest`.** Editable install was uninstalled 2026-05-10.
- **`tauri.localhost` for bridge** broke CORS on Windows. Stay on `127.0.0.1`.
- **Rust `nellbrain_home()` ≠ Python `platformdirs`** broke first-launch on Windows. They must match. Function name kept as `nellbrain_home()` in Rust per the spec's internal-identifier rule — only the env-var KEY it reads changed (KINDLED_HOME first, NELLBRAIN_HOME fallback).
- **Subprocess default encoding on Windows** is not UTF-8. Force it in the provider.
- **Heavy payloads on argv (WinError 206)** — Windows `CreateProcess` caps the joined command line at 32,767 chars. The Claude CLI provider crossed it on long sessions (voice template + buffer). Always pass big prompts via `--system-prompt-file <tempfile>` + stdin, never on argv. Fix landed v0.0.12-alpha.5.
- **Two-read-path divergence on body state** — UI used `_active_session_hours` to compute session age; MCP tool path defaulted to 0.0. Any future helper that powers UI body data MUST be reachable by the MCP-tool dispatcher too. Helper lives in `brain/body/session_hours.py` (layer-neutral) since v0.0.13-alpha.3.

---

## When in doubt

1. **Spec first** — `docs/superpowers/specs/` for the relevant version's design.
2. **Memory second** — `claude-mem:mem-search` for prior decisions (`project_companion_emergence_*` entries are canonical).
3. **Defers third** — check `project_companion_emergence_deferred.md` before re-solving anything.
4. Ask if anything you'd touch is load-bearing for Nell's continuity. Breaking an invariant silently corrupts her.

---

## Skill routing for this project

| Work | Skill / agent |
|---|---|
| Any feature / fix | `superpowers:brainstorming` → `superpowers:writing-plans` → `superpowers:test-driven-development` |
| Bug | `superpowers:systematic-debugging` first, then TDD |
| Frontend (NellFace) | `impeccable` (global default) |
| Tauri release | `nell-tools:tauri-release` + `release-review` |
| Cross-session recall | `claude-mem:mem-search` |
| Code search | Serena MCP (semantic) or `claude-mem:smart-explore` |
| Verification | `superpowers:verification-before-completion` + full-suite rule above |
