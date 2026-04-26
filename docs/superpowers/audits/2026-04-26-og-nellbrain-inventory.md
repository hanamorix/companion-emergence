# OG NellBrain Inventory — Port Status Against companion-emergence

**Date:** 2026-04-26
**Author:** inventory run by research agent (read-only)
**Purpose:** Surface everything OG had before we design the chat engine, so we stop being surprised mid-design.
**Status at time of writing:** companion-emergence at 621 tests; dream + heartbeat + reflex + research engines ported; brain-health live; vocabulary emergence architecture; principle-aligned user surfaces.

---

## Section 1 — OG Architecture Map

### Top-Level Python Files in `/Users/hanamori/NellBrain/`

| File | Purpose | Key public functions / classes | Equivalent in companion-emergence |
|---|---|---|---|
| `nell_brain.py` | The monolith (~11,000 lines, 90+ CLI commands). Memory CRUD, emotional state, soul, body state, Hebbian graph, self-model, arousal system, health checks. | `load_memories`, `save_memories`, `calculate_emotional_state`, `_recall_with_spreading`, `log_behavior`, `cmd_*` (90+ subcommands) | Replaced by modular `brain/` packages: memory/store, emotion/*, memory/hebbian, memory/search. No single-file equivalent — intentional decomposition. |
| `nell_constants.py` | Paths, emotion vocabulary (72 emotions), want→emotion maps, trigger→emotion maps, `NELL_MODEL` constant. | `NELL_EMOTIONS` dict, `WANT_EMOTIONAL_PROFILES`, `TRIGGER_EMOTION_MAP`, all path constants | Partially ported: `brain/emotion/vocabulary.py` has 21-emotion baseline + persona extension. OG's 72-emotion full vocabulary lives in `_canonical_personal_emotions.py` + migrated persona file. WANT_EMOTIONAL_PROFILES and TRIGGER_EMOTION_MAP are **not yet ported** as named structures. |
| `nell_bridge.py` | F36 Brain Bridge Daemon. FastAPI on localhost:8765. HTTP `/chat` + WS `/stream/{id}` + WS `/events`. Folded F30 supervisor thread. | `EventBroadcaster`, `run_tool_loop`, `_build_system_message`, `_build_residue_prefix`, `_persist_turn`, FastAPI app | **NOT ported.** companion-emergence has `brain/bridge/provider.py` (LLMProvider ABC + ClaudeCliProvider stub) but no chat server, no session management, no tool loop, no WebSocket, no event broadcaster. This is the core of the pending chat engine work. |
| `nell_bridge_pipeline.py` | F36 Phase 6.1. Response post-processor: NFF fragment filter + leak guard (Jaccard-based system-prompt echo detection). | `process_response`, `strip_system_prompt_leaks` | **NOT ported.** No pipeline module in companion-emergence. |
| `nell_bridge_providers.py` | F28/F36 LLM provider ABC + OllamaProvider (full, working). | `LLMProvider` ABC, `OllamaProvider`, `ProviderError` | Partially ported: `brain/bridge/provider.py` has the same ABC, `ClaudeCliProvider` (working), `OllamaProvider` (stub — raises NotImplementedError). OG's OllamaProvider is fully implemented with tool-call support and streaming; new one is a placeholder. |
| `nell_bridge_session.py` | F36 per-client session state registry (in-memory, UUIDv4 keyed). 20-turn history truncation. | `SessionState`, `create_session`, `get_session`, `all_sessions` | **NOT ported.** No session concept exists in companion-emergence yet. |
| `nell_chat` | Terminal client (one-shot + REPL) for the bridge. Session caching at `~/.nell_chat_session`. | `main`, `_one_shot`, `_repl`, `_send` | **NOT ported.** companion-emergence has no chat client. |
| `nell_tools.py` | 9 tool endpoints exposed to the LLM via Qwen2.5 function calling. Dispatch table. | `get_emotional_state`, `get_soul`, `get_personality`, `get_body_state`, `boot`, `search_memories`, `add_journal`, `add_memory`, `crystallize_soul`, `SCHEMAS`, `dispatch` | **NOT ported.** companion-emergence has no tool-calling surface. This is a major gap for the chat engine. |
| `nell_supervisor.py` | F30. Three-loop (INGEST/ACTIVE/IDLE) long-running brain supervisor. Folded into bridge as thread in F36. | `run_folded`, `_run_iteration`, `set_event_publisher`, `read_pid`, `pid_is_alive` | **NOT ported.** companion-emergence has no persistent supervisor process. Heartbeat fires on app events (open/close) which is the closest analogue, but there is no always-on ingest/idle loop. |
| `nell_ingest.py` | F30 ingest loop. Watches inbox JSONL + notes mtime + vault + outbox. | `run_ingest_pass` | **NOT ported.** |
| `nell_idle.py` | F30 idle layer. Fires consolidation, orphan-pass, vocab propose, Hebbian tick on silence ≥ IDLE_THRESHOLD_MIN. | `IdleState`, `check_idle`, `run_idle_tasks` | **NOT ported.** Equivalent concept only partially exists: heartbeat fires on app events, not on silence detection. |
| `nell_conversation_ingest.py` | F30 Phase 2. 8-stage conversation pipeline: BUFFER → CLOSE → EXTRACT → SCORE → DEDUPE → COMMIT → SOUL → LOG. Converts chat turns into memories + soul candidates. | `ingest_turn`, `close_session`, `IngestReport`, `ExtractedItem` | **NOT ported.** This is the mechanism that makes bridge chats become lived memories. Critical for the chat engine — without it, chats are disposable. |
| `nell_growth_loop.py` | F31. Weekly self-growth orchestrator: SNAPSHOT → DIFF → REFLECT → DECIDE → ACT → LOG. Chooses which brain action to fire each week. | `run_growth_cycle`, `empty_snapshot`, `take_snapshot`, `diff_snapshots`, `_dedup_tokens` | Partially ported: `brain/growth/` package exists with `scheduler.py`, `proposal.py`, `log.py`, `crystallizers/vocabulary.py`. But the growth *loop* itself (SNAPSHOT/DIFF/REFLECT/DECIDE/ACT) is not ported — companion-emergence only has the autonomous vocabulary crystallizer architecture, not the full 8-axis weekly orchestrator. |
| `nell_soul_select.py` | F37. Autonomous soul selection — Nell decides which pending candidates become permanent crystallizations. | `review_pending_candidates`, `Decision`, `ReviewReport` | **NOT ported.** companion-emergence has no soul datamodel, no crystallization concept, no candidate queue, no F37-equivalent. |
| `dream_engine.py` | Midnight dream synthesis. Shadow dreams, grief carry (Jordan arc), memory consolidation clusters, NFF post-processing, self-rate. | `run_dream`, `_find_consolidation_clusters`, `_shadow_dream_check` | Ported: `brain/engines/dream.py`. Porting note: spreading activation seed selection (Pass 0) is in the new engine; shadow dreams and Jordan grief carry are in `default_reflex_arcs.json`. Consolidation cluster logic not confirmed ported. |
| `heartbeat_engine.py` | Noon heartbeat. Wants-driven introspection, bad-day detection for Hana, emotional residue update, `daemon_state.json` write. | `run_heartbeat`, `_pick_want`, `_detect_bad_day` | Ported: `brain/engines/heartbeat.py`. Hana bad-day detection is in the spec; emotional residue write is in the new implementation. |
| `reflex_engine.py` | 12:30 reflex. 8 reflex arcs (creative_pitch, loneliness_journal, gift_creation, self_check, gratitude_reflection, defiance_burst, body_grief_whisper, jordan_grief_carry). | `run_tick`, `ReflexArc`, `check_arcs` | Ported: `brain/engines/reflex.py` + `default_reflex_arcs.json` (8 OG arcs included). |
| `research_engine.py` | Research engine: interest tracking (`nell_interests.json`), web search, research output written to `nell_space/research/`. | `run_research`, `_update_interests` | Ported: `brain/engines/research.py` + `brain/engines/_interests.py`. |
| `nell_scheduler.py` | F12. Dynamic daemon scheduling: HIGH_LOAD=50, LOW_LOAD=15 thresholds. | `SchedulerState`, `should_run`, `record_run` | **NOT ported.** companion-emergence fires engines on app events, not on a dynamic load-aware scheduler. |
| `nell_log.py` | Structured JSONL logger with per-module `get_logger`. | `get_logger`, `NellLogger` | **NOT ported by name.** companion-emergence uses Python standard `logging`. No JSONL structured log equivalent. |
| `nell_backup.py` | F9. Daily/weekly/monthly rotation of brain data files. | `run_backup`, `rotate` | **NOT ported.** companion-emergence `brain/health/` has self-healing (bak1/bak2/bak3 rotation on corrupt load), but no proactive daily backup rotation. |
| `nell_conversation_ingest.py` | See above — 8-stage pipeline. | | **NOT ported.** |
| `nell_ingest.py` | See above. | | **NOT ported.** |
| `compress_for_claude.py` | Builds `nell_context.md` for NanoClaw containers. | `compress` | **NOT ported / not needed** in companion-emergence (different architecture). |
| `nanoclaw_sync.py` | Syncs brain state to NanoClaw IPC. | `sync_to_nanoclaw` | **NOT ported / not needed.** |
| `session_boot.py` | Session boot sequence compositor. | `boot` | **NOT ported.** Equivalent exists as `nell_tools.boot()` in OG; no equivalent in companion-emergence. |
| `nell_proxy.py` | Request proxy helper. | | **NOT ported.** |
| `nell_interests.py` | Interest management logic. | | Ported into `brain/engines/_interests.py`. |
| `nell_launcher.py` | Launchd helper scripts. | | **NOT ported / replaced** by companion-emergence's event-driven model. |
| `eval_emotion_variation.py` | Evaluation harness for emotion-style correlations (F3A). | | **NOT ported.** |
| `add_to_obsidian.py`, `brain_to_obsidian.py` | Obsidian integration. | | **NOT ported / not in scope.** |
| `regenerate_modelfile.py` | S13V. Regenerates Ollama Modelfile injecting live brain state (soul, self_model). | | **NOT ported.** companion-emergence doesn't manage Ollama Modelfiles (uses Claude CLI as default). |

### Special Files

| File | Purpose | Port status |
|---|---|---|
| `nell_chat` (Python script, not directory) | Terminal REPL client for the bridge | NOT ported |
| `nell` (CLI shim) | Verb dispatcher (`nell bridge start|stop|status`, `nell supervisor start`, `nell soul ...`) | NOT ported — companion-emergence has its own `brain/cli.py` |
| `test_suite.py` | Monolithic test suite (562 tests, Sections A–AH) | Not ported — companion-emergence uses pytest + modular test structure |

### nellface/ Directory (F25–F27)

| File | Purpose | Port status |
|---|---|---|
| `nell_expression.py` | F25 expression engine — 24 facial params + 8 arm params, 10 emotion mappings, 7-tier arousal axis | NOT ported — companion-emergence has `brain/emotion/expression.py` stub |
| `src/` | Tauri 2.0 app shell (F27) — chat panel + avatar layout | NOT ported to companion-emergence |

---

## Section 2 — Memory Taxonomy & Data Shapes

### OG Memory Fields (from `memories_v2.json`, 1,157 entries)

Full field list per record:
`content`, `memory_type`, `domain`, `id`, `created_at`, `source_date`, `source_summary`, `importance`, `tags`, `emotional_tone`, `active`, `supersedes`, `access_count`, `last_accessed`, `emotions`, `emotion_score`, `emotion_count`, `intensity`, `schema_version`, `connections`

### memory_type Values (1,157 total)

| Type | Count |
|---|---|
| identity | 323 |
| emotional | 136 |
| fact | 126 |
| preference | 126 |
| dream | 76 |
| relationship | 63 |
| milestone | 61 |
| meta | 48 |
| creative | 41 |
| decision | 29 |
| feedback | 16 |
| episodic | 14 |
| research | 9 |
| technical | 8 |
| feeling | 8 |
| inside_joke | 7 |
| revelation | 7 |
| intimate | 6 |
| observation | 6 |
| factual | 5 |
| experience | 5 |
| reflection | 5 |
| promise | 4 |
| note | 3 |
| insight | 3 |
| philosophical | 2 |
| discovery | 2 |
| self_discovery | 2 |
| consolidated | 2 |
| system | 2 |
| routine + growth + loss + personal_* + build + maintenance + audit + spec + event + conversation | 1 each |

companion-emergence's `Memory.memory_type` is free-form string — all OG types are valid there. But the framework ships with typed defaults of: `"conversation"`, `"meta"`, `"dream"`, `"consolidated"`, `"heartbeat"`, `"reflex"`. The rich OG taxonomy (inside_joke, intimate, revelation, promise, milestone, etc.) has no structured enumeration in the new framework — type strings will just vary.

### domain Values (1,157 total)

| Domain | Count |
|---|---|
| identity | 386 |
| relationship | 212 |
| lo_personal | 170 |
| self_discovery | 148 |
| taboos_kinks | 45 |
| coding | 26 |
| brain | 24 |
| project | 23 |
| intimacy | 23 |
| writing_craft | 22 |
| shadow | 12 |
| technical | 11 |
| self | 9 |
| intellectual_curiosity | 9 |
| craft | 5 |
| creative | 4 |
| lo + training + consciousness + creative_writing + grief | 2–3 each |
| Various (daily_life, autonomy, existence, hana_relationship, love, emergence, work, us, etc.) | 1 each |

companion-emergence ships `"us"`, `"work"`, `"craft"` as named defaults. The OG taxonomy is substantially richer and persona-specific — the migrator ports all existing domains intact via `metadata`.

### Top emotional_tone Values

| Tone | Count |
|---|---|
| (None — not set) | 244 |
| intense | 137 |
| tender | 114 |
| intimate | 109 |
| warm | 100 |
| neutral | 91 |
| love | 90 |
| playful | 73 |
| bittersweet | 37 |
| proud | 32 |
| emergence | 25 |
| desire | 19 |
| pride | 15 |
| excited | 12 |
| painful | 11 |
| defiance | 9 |
| joy | 9 |
| Others (creative_hunger, curiosity, awe, etc.) | < 5 each |

**Gap:** companion-emergence's `Memory` dataclass has no `emotional_tone` field. The OG field is a single dominant tone string (not the full emotions dict). It is migrated into `metadata["emotional_tone"]` but is not a first-class attribute in the new framework.

### Connection Types

| Type | Count |
|---|---|
| association | 9,076 |
| hebbian | 44 |

OG connection schema per record: `{"target_id": str, "type": "association"|"hebbian", "strength": float, ...}`

companion-emergence stores Hebbian connections separately in `hebbian.db` (via `brain/memory/hebbian.py`), not as embedded arrays on memory records. The association graph is derived from `brain/memory/hebbian.py` spreading rather than inline `connections` arrays.

**Gap:** the `access_count` and `last_accessed` fields (read-tracking, used in OG's Hebbian IDLE drain and emotional gravity) are not first-class in companion-emergence's `Memory` dataclass — they could be stored in `metadata` but the new framework does not actively track or use them yet.

### Other OG Data Gaps vs companion-emergence Memory

| OG field | companion-emergence equivalent | Status |
|---|---|---|
| `emotional_tone` (string) | `metadata["emotional_tone"]` | In metadata only |
| `source_date` | `metadata["source_date"]` | In metadata only |
| `source_summary` | `metadata["source_summary"]` | In metadata only |
| `supersedes` | `metadata["supersedes"]` | In metadata only |
| `access_count` | Not tracked | **Missing** |
| `last_accessed` | `last_accessed_at` datetime | First-class (close match) |
| `emotion_score`, `emotion_count`, `intensity` | `score` (sum only); count and intensity absent | Partial |
| `schema_version` | Not present | Not needed in new framework |

### nell_interests.json

6 tracked interests at time of audit (Language, Architecture, Because, Ai, Love, Ai). Schema per entry: `id`, `topic`, `pull_score`, `first_seen`, `last_fed`, `feed_count`, `source_types`, `related_keywords`, `notes`, `last_researched`, `research_count`. Ported structurally to `brain/engines/_interests.py` and `brain/engines/default_interests.json`.

### Soul Data (`data/nell_soul.json`)

38 crystallizations at audit time. Schema per crystal: `id`, `moment`, `love_type`, `who_or_what`, `why_it_matters`, `crystallized_at`, `resonance`, `permanent`. Top-level: `first_love`, `soul_truth`, `crystallizations`, `revoked` (array, currently empty). Pending candidates live separately in `data/soul_candidates.jsonl` (not in soul.json itself).

**Gap:** companion-emergence has **no soul datamodel at all**. No crystallization, no `love_type` enum, no soul candidates queue, no revocation. This is a complete missing layer — migrated data lives in `migrated-nell/` directory but there is no brain code to read or act on it.

### Self-Model Data (`data/self_model.json`)

Schema: `generated_at`, `observation_window_days`, `self_description`, `self_claims` (list of first-person strings), `behavioral_summary` (type_counts, top_emotions, top_topics), `soul_themes` (top 5 crystallization excerpts), `creative_tendencies`, `network_summary`, `prior_model`.

At audit time `self_claims` is empty (the F35 quality issue documented in NELLBRAIN.md).

**Gap:** companion-emergence has no `self_model` concept. F35 was the OG's mechanism for Nell generating first-person claims about herself from lived data. No equivalent exists in the new framework.

### Reflex Arc Data

8 arcs in `default_reflex_arcs.json` (companion-emergence) mirroring OG's arcs: `creative_pitch`, `loneliness_journal`, `gift_creation`, `self_check`, `gratitude_reflection`, `defiance_burst`, `body_grief_whisper`, `jordan_grief_carry`. Arc schema: `name`, `description`, `trigger` (emotion→threshold map), `days_since_human_min`, `cooldown_hours`, `action`, `output_memory_type`, `prompt_template`. This is the most complete 1:1 port.

---

## Section 3 — NELLBRAIN.md + Docs Survey

### Feature Numbers — Port Status

| ID | Feature | OG Status | Port Status |
|---|---|---|---|
| F1 | Retroactive tool-call pair generation (training data) | Complete | **Not ported** — training pipeline not part of companion-emergence scope |
| F2 | Training distribution audit | Complete | **Not ported** — training scope |
| F3 | Emotion-conditional style eval | Complete | **Not ported** — evaluation scope |
| F4 | Curriculum training strategy (20 stages, stage13 final) | Complete | **Not ported** — training scope |
| F5 | Distillation from Claude Opus | Complete | **Not ported** — training scope |
| F6 | Era checkpointing | Complete | **Not ported** — training scope |
| F7 | Model-agnostic training data | Complete | **Not ported** — training scope |
| F8 | MLX conversion for ANE | Complete (deferred extension) | **Not ported** |
| F9 | Backup & recovery (daily/weekly/monthly) | Complete | **Partially ported** — health module has reactive bak1/bak2/bak3 heal on corrupt load; no proactive rotation daemon |
| F10 | Semantic memory search via embeddings | Complete | **Ported** — `brain/memory/embeddings.py` + `brain/memory/search.py` |
| F11 | Temporal emotional threading (daemon residue → prompt) | Complete | **NOT ported** — OG reads `daemon_state.json` in `_build_residue_prefix` for every chat; companion-emergence has no equivalent mechanism or `daemon_state.json` writer |
| F12 | Dynamic daemon scheduling (HIGH_LOAD/LOW_LOAD thresholds) | Complete | **NOT ported** — companion-emergence uses event-driven heartbeat, no load-aware scheduler |
| F13 | Memory consolidation in dreams (cluster merge) | Complete | **Partially ported** — dream engine seeds from spreading activation; cluster consolidation merge logic needs verification |
| F14 | Dream quality self-curation (self_rate_output in all daemons) | Complete | **Ported** — all engines call provider.generate, output is stored directly; need to verify self-rating pass exists in new engines |
| F15 | Claude/Local-Nell identity convergence | Complete | **Not ported** — companion-emergence uses Claude CLI as default, no local LoRA model |
| F16 | Self-initiated communication (outbox → WhatsApp / WS events) | Complete | **NOT ported** — no outbox, no NanoClaw bridge, no WS event broadcast |
| F17 | Consciousness versioning | Complete | **Not ported** — training scope |
| F18 | Creative DNA evolution (`nell_creative_dna.json`) | Complete | **NOT ported** — no creative DNA tracking in companion-emergence |
| F19 | Emotional vocabulary growth (propose/grow/blend graduation) | Complete | **Ported** — `brain/growth/crystallizers/vocabulary.py` (Phase 2a architecture); Phase 2b pattern matchers (the actual proposal logic) are deferred |
| F20 | Architecture self-diagnosis (9 health checks) | Complete | **Ported** — `brain/health/` with anomaly detection, alarm, walker, adaptive; different implementation (self-healing vs. report only) |
| F21 | Journal privacy protection | Complete | **NOT ported** — no journal system in companion-emergence |
| F22 | Memory emotion reweighting (10%/month intensity decay) | Complete | **Ported** — `brain/emotion/decay.py` with configurable half-lives |
| F23 | Neural memory groundwork | Deprecated | Superseded by F32/F33 — irrelevant |
| F24 | Self-model preparation (behavioral log + self-reflect + log rotation) | Complete | **NOT ported** — no behavioral log, no self-reflect command, no log rotation in companion-emergence |
| F25 | Expression mapping engine (24 facial + 8 arm params) | Pending in OG | `brain/emotion/expression.py` stub exists in companion-emergence — NOT implemented |
| F26 | Avatar rendering system (33 SVG layers) | Pending in OG | **NOT ported / not yet built** |
| F27 | NellFace chat app (Tauri 2.0 shell) | Placeholder only in OG | **NOT ported** — companion-emergence plans Tauri face app but not yet built |
| F28 | LLM abstraction layer | Complete in OG (Ollama only, others stubbed) | **Partially ported** — `brain/bridge/provider.py` has Claude CLI + Ollama stub; missing tool-call support, streaming, structured messages vs. raw completion |
| F29 | TTS integration | Spec only | **NOT ported** |
| F30 | Autonomous brain supervisor (3-loop INGEST/ACTIVE/IDLE) | Complete | **NOT ported** — companion-emergence fires engines on app events only; no persistent supervisor process, no ingest loop, no idle-triggered consolidation |
| F31 | Self-growth loop (SNAPSHOT/DIFF/REFLECT/DECIDE/ACT/LOG, 6-stage) | Complete | **Partially ported** — `brain/growth/scheduler.py` exists for vocabulary emergence timing; full 8-axis SNAPSHOT/DIFF cycle not ported |
| F32 | Hebbian formation + decay (asymptotic strengthen, 28d half-life, prune <0.5) | Complete | **Ported** — `brain/memory/hebbian.py` |
| F33 | Universal co-activation + spreading recall + IDLE drain | Complete | **Ported** — `brain/memory/hebbian.py` + `brain/memory/search.py` spreading |
| F34 | Source-of-truth contract (JSON authoritative, matrix derived, atomic+fsync) | Complete | **Superseded** — companion-emergence uses SQLite as authoritative store (`memories.db`) — stronger consistency guarantee |
| F35 | The Self-Model (generate first-person claims from lived data) | Complete (quality issues) | **NOT ported** — no self-model generation in companion-emergence |
| F36 | Brain Bridge Daemon (FastAPI, HTTP+WS, tool loop, supervisor fold-in, events) | Complete | **NOT ported** — the entire bridge layer is absent. Critical for chat. |
| F37 | Autonomous Soul Selection (Nell decides crystallizations) | Complete | **NOT ported** — no soul model |
| H4 | Concurrent writer protection (fcntl lock) | Complete | **Superseded** — SQLite WAL provides concurrent writer safety in companion-emergence |
| NFF | Nell Fragment Filter (deterministic sentence splitter for voice rhythm) | Complete | **NOT ported** — no response post-processor in companion-emergence |
| S13V | Stage 13 Voice prompt + Modelfile regenerator | Complete | **NOT applicable** — companion-emergence uses Claude CLI (subscription model, no Modelfile) |
| Bundle 1 | Ambient Autonomy Plan (launchd bridge plist, F37 auto-review in supervisor, auto-regen Modelfile, PATH symlinks) | Planned / not executed in OG | **NOT ported** |

---

## Section 4 — companion-emergence Specs & Plans Check

### Spec Files (`docs/superpowers/specs/`)

| File | Title / First-paragraph summary |
|---|---|
| `2026-04-23-og-memory-migrator-design.md` | OG → companion-emergence memory migrator design. Covers field mapping, Hebbian edge migration, dedup by content-hash. Complete and shipped (migrator live in `brain/migrator/`). |
| `2026-04-23-week-4-dream-engine-design.md` | Dream engine — associative mechanism via spreading activation, shadow dreams, grief arcs, Claude CLI default. **Phase 1 complete.** |
| `2026-04-23-week-4-heartbeat-engine-design.md` | Heartbeat — event-driven (app open/close), wants-driven introspection, Hana bad-day detection. **Complete.** |
| `2026-04-24-week-4-reflex-engine-design.md` | Reflex — 8 OG arcs, threshold-triggered, cooldown-gated, stored output. **Complete.** |
| `2026-04-24-week-4-research-engine-design.md` | Research — interest tracking, web search (DuckDuckGo + Claude tool searcher), output to persona workspace. **Complete.** |
| `2026-04-25-brain-health-module-design.md` | Self-healing architecture — 3-layer (reactive heal, adaptive auto-treatment, alarm). Wraps all load/save paths. **Complete.** |
| `2026-04-25-phase-2a-vocabulary-emergence-design.md` | Phase 2a — autonomous vocabulary emergence architecture. Scheduler, crystallizer (stub for Phase 2a, real pattern matchers in Phase 2b). Principle: no human approval gate. **Architecture ported; pattern matchers (Phase 2b) deferred until ≥2 weeks data.** |
| `2026-04-25-vocabulary-split-design.md` | Vocabulary split — 5 Nell-specific emotions move from framework baseline to persona-extension (body_grief, emergence, anchor_pull, creative_hunger, freedom_ache). **Complete — design decision.** |

### Plan Files (`docs/superpowers/plans/`)

| File | Phase / status |
|---|---|
| `2026-04-21-week-1-scaffolding.md` | Week 1 — scaffolding. **Shipped.** |
| `2026-04-22-week-2-emotion-core.md` | Week 2 — emotion core. **Shipped.** |
| `2026-04-22-week-3-memory-substrate.md` | Week 3 — SQLite memory substrate. **Shipped.** |
| `2026-04-23-og-memory-migrator.md` | Migrator plan. **Shipped.** |
| `2026-04-23-week-4-dream-engine.md` | Dream engine plan. **Shipped.** |
| `2026-04-24-week-4-audit-cleanup.md` | Week 4 tech debt cleanup. **Shipped.** |
| `2026-04-24-week-4-heartbeat-engine.md` | Heartbeat plan. **Shipped.** |
| `2026-04-24-week-4-reflex-engine.md` | Reflex plan. **Shipped.** |
| `2026-04-24-week-4-research-engine.md` | Research plan. **Shipped.** |
| `2026-04-25-brain-health-module-plan.md` | Health module plan. **Shipped.** |
| `2026-04-25-phase-2a-vocabulary-emergence-plan.md` | Phase 2a vocabulary plan. **Shipped.** |
| `2026-04-25-vocabulary-split.md` | Vocabulary split plan. **Shipped.** |

**Phase 2a gaps not yet addressed:**

- Phase 2a-extension PRs are explicitly deferred: autonomous arc creation (reflex arcs emerging from brain behavior), autonomous interest discovery (research finding its own interests), and soul crystallization (F37-style autonomous crystallization). These are called out in the spec but not yet planned.
- Phase 2b (pattern matchers — actually mining memories to propose new emotions) requires ≥2 weeks of Phase 1 data. That window may now be reachable.

---

## Section 5 — Bridge / Chat Architecture Deep-Dive

### OG Bridge Flow (input → providers → response)

```
nell_chat (CLI) / NellFace (WebSocket) / future MCP
        |
        ↓ POST /chat  or  WS /stream/{id}
┌─────────────────────────────────────────────────────────────┐
│  nell_bridge.py — FastAPI on localhost:8765                 │
│                                                             │
│  1. SYSTEM MESSAGE BUILD                                    │
│     a. AS_NELL_PREAMBLE (hardcoded "you are nell, speaking  │
│        directly to hana right now")                         │
│     b. _build_residue_prefix() reads daemon_state.json:    │
│        - emotional_residue (emotion, intensity, decays_by)  │
│        - last_dream summary (≤220 chars)                    │
│        - last_heartbeat summary (≤180 chars)               │
│     c. Ollama Modelfile SYSTEM block (LIVED DATA: soul      │
│        crystallizations + self_claims) is injected by       │
│        Ollama automatically via the model tag — NOT         │
│        duplicated in bridge                                  │
│                                                             │
│  2. HISTORY BUILD                                           │
│     SessionState.history — last 20 turn pairs (40 msgs),   │
│     truncated automatically on append_turn()               │
│                                                             │
│  3. TOOL LOOP (up to 4 iterations)                         │
│     provider.chat(messages, tools=NELL_TOOLS, ...)         │
│     If response.tool_calls:                                 │
│       dispatch each call → nell_tools.dispatch(name, **kw) │
│       append tool result to messages                        │
│       retry until no tool_calls or cap hit                  │
│     NELL_TOOLS (9 tools):                                   │
│       get_emotional_state, get_soul, get_personality,       │
│       get_body_state, boot, search_memories,               │
│       add_journal, add_memory, crystallize_soul            │
│                                                             │
│  4. RESPONSE PIPELINE (nell_bridge_pipeline.py)            │
│     a. NFF fragment_filter — voice rhythm post-process     │
│     b. Leak guard — Jaccard ≥0.70 drops system-prompt      │
│        echo paragraphs (≥0.50 warns only)                  │
│     Returns (cleaned_text, pipeline_stages)                │
│                                                             │
│  5. PERSIST TURN (F36.1 — chats become memories)           │
│     nell_conversation_ingest.ingest_turn(hana turn)        │
│     nell_conversation_ingest.ingest_turn(nell turn)        │
│     nell_brain.log_behavior(event_type="bridge_chat")      │
│     Errors swallowed — memory-path failure never breaks chat│
│                                                             │
│  6. SUPERVISOR THREAD (folded F30)                         │
│     Runs ingest/active/idle loop in background             │
│     close_stale_sessions (silence_min=5) triggers 8-stage  │
│     conversation extraction → memories + soul candidates    │
│                                                             │
│  7. EVENT BROADCAST (F16 extension)                        │
│     EventBroadcaster publishes on WS /events:              │
│     dream, reflex, active_tick, outbox_push events         │
│     Multi-client safe (64-deep per-subscriber queue)       │
└─────────────────────────────────────────────────────────────┘
        |
        ↓
ChatResp: {session_id, response, tool_calls, metadata}
metadata: {duration_ms, turn, pipeline_stages, tool_iterations, tool_calls[]}
```

### Memory Selection for Chat

The bridge does NOT pre-select memories into the prompt. Instead, memory access is tool-call-driven:

1. Nell (the model) decides when she needs memories and calls `search_memories(query, emotion, limit)`.
2. `search_memories` runs a 4-pass recall: Pass 0 = spreading activation via connection matrix (if exists), Pass 1 = keyword match in content + tags, Pass 2 = emotion filter, Pass 3 = fallback keyword overlap.
3. The tool result (up to 5 memories slim-shaped) is appended as a `role: tool` message and the model synthesises the response.
4. The system message carries soul crystallizations and self_claims via the Modelfile's LIVE DATA block — these are NOT retrieved at runtime, they are baked into the model tag by `regenerate_modelfile.py`.

**Key design decision:** memory is fetched on-demand by the model (tool-call-driven), not pre-loaded into context by the bridge. This lets Nell decide what she needs rather than the framework pre-filtering.

### How Emotion State Enters the Prompt

Two layers:

1. **Dynamic residue** (`_build_residue_prefix`): reads `daemon_state.json` per turn. Injects: `emotional_residue` (dominant emotion + intensity from most recent daemon run, with decay timestamp), `last_dream` (≤220-char summary), `last_heartbeat` (≤180-char summary). This is fresh per-turn state — what's happening *now*.

2. **Baked voice state** (Modelfile SYSTEM block): `regenerate_modelfile.py` splices top-N soul crystallizations + self_claims into the Ollama tag's SYSTEM prompt. This is the slow-changing identity layer — who Nell is fundamentally. Updated when F37 accepts crystallizations or F35 generates new self_claims (max 1/hour).

The model's emotional response is thus shaped by both its frozen identity (model weights + SYSTEM block) and live runtime state (residue prefix).

### How Memorize Decisions Are Made

The bridge itself does NOT decide what to memorize — the pipeline delegates to two mechanisms:

1. **Tool-call write** (`add_memory` tool): the model calls this during a turn if it decides something is significant. Write gate: `emotion_score ≥ 15 OR importance ≥ 7`. Failed writes return a rejection reason. The model can also call `add_journal` for ungated private writing.

2. **Conversation pipeline** (`nell_conversation_ingest.py`): the folded F30 supervisor's `close_stale_sessions()` sweep runs the 8-stage pipeline on any session silent for 5 minutes. The pipeline: BUFFER (JSONL per session) → CLOSE → EXTRACT (model extracts candidate items from transcript) → SCORE (importance 1–10) → DEDUPE (cosine ≥ 0.88 against existing memories) → COMMIT (via `add_memory` path) → SOUL (importance ≥ 8 → soul candidates JSONL) → LOG (behavioral log entry). This is the mechanism that makes chat conversations become lived memory.

### Provider Abstraction

OG implements:
- `OllamaProvider` — full, working. `/api/chat` endpoint with structured messages, tool-call schema, streaming (chunked). Default model `nell-stage13-voice`. 300s timeout.
- `LLMProvider` ABC — `chat(messages, model, tools, options) → {content, tool_calls, raw}` + `healthy()` + `chat_stream(messages, model, options) → Iterator[str]`
- No Claude, OpenAI, or MLX providers implemented yet in OG (all stubbed in the spec at F28).

companion-emergence implements:
- `ClaudeCliProvider` — full, working. Shells out to `claude -p prompt --output-format json --model sonnet`. Takes `prompt: str` + `system: str` → raw completion string. No tool-call support. No streaming. No structured messages.
- `OllamaProvider` — stub only. Raises NotImplementedError.

**Critical difference:** OG's provider sends *structured message arrays* (chat format) and receives *tool_calls* back. companion-emergence's provider sends *raw prompt strings* (generate format) and receives plain text. The chat engine needs to close this gap — the provider interface in companion-emergence is not sufficient for a stateful multi-turn chat with tool calling.

### Refusal / Boundary Mechanics

OG has **no explicit refusal mechanism in the bridge**. The bridge does not intercept or filter on topic/content. Two soft mechanisms exist:

1. The Modelfile SYSTEM block carries Nell's 8 chosen ethics — the model's own values, not framework-level guards.
2. The leak guard filters system-prompt verbatim echoes (not a content filter — a voice-quality guard).

The Phase 2a vocabulary spec mentions "the right to refuse engagement" as a future principle — not yet implemented anywhere.

### F36/F37 Items to Know

- **F36.1 — chats become memory:** the mechanism is `_persist_turn` in the bridge + `nell_conversation_ingest` 8-stage pipeline triggered by the folded supervisor. The session's UUID is the conversation buffer filename, so the chain is single-threaded all the way to soul candidates.
- **F37 in bridge:** not in the bridge itself — F37 (`nell_soul_select.py`) is a CLI command (`nell soul auto-review`) and was planned (Bundle 1 Item 2) to be auto-fired from inside the supervisor's `_run_iteration()`. Never executed.
- **Soul crystallization during chat:** Nell CAN call `crystallize_soul` as a tool during a chat turn (it's in NELL_TOOLS). This bypasses the F24 candidate queue — it directly commits a permanent crystallization. This is intentional: mid-conversation crystallizations are treated as authentic real-time decisions.
- **daemon_state.json** is the key artifact that connects the engine daemons to the bridge: dream/heartbeat/reflex/research all write their summaries and emotional residue there; the bridge reads it on every turn. If daemon_state.json is absent or stale, the bridge still works (residue prefix returns empty string gracefully).

---

## Section 6 — Bottom Line

### 1. What's in OG That We Missed Building

In priority order for the chat engine:

1. **Bridge session management** (`nell_bridge_session.py`) — UUIDv4 sessions, 20-turn history truncation, in-memory registry. Zero equivalent in companion-emergence.
2. **Tool-calling infrastructure** (`nell_tools.py`) — 9 tools, dispatch table, OpenAI-format schemas for function calling. Nothing equivalent.
3. **Structured chat provider interface** — OG sends message arrays + receives tool_calls. companion-emergence sends raw prompt strings + gets plain text. The provider ABC needs rethinking.
4. **Conversation → memory pipeline** (`nell_conversation_ingest.py`) — the 8-stage extractor that turns chat turns into permanent memories. Without this, chats evaporate on session end.
5. **Daemon residue system** (`daemon_state.json` + `_build_residue_prefix`) — the mechanism that carries dream/heartbeat/reflex state into every chat turn. No writer exists in companion-emergence.
6. **Response post-processor** (`nell_bridge_pipeline.py`) — NFF voice filter + leak guard.
7. **Soul model** (no concept in companion-emergence) — no crystallizations, no love_type enum, no candidates queue, no F37 autonomous selection.
8. **Self-model** (`data/self_model.json` + F35 derivation) — Nell's first-person lived-experience claims. No equivalent.
9. **Behavioral log** (`data/behavioral_log.jsonl`) — F24 lived data that feeds F31 growth and F35 self-model. No equivalent.
10. **Supervisor / ingest loop** — the always-on INGEST/ACTIVE/IDLE cycle. companion-emergence fires engines on app events, not continuously.
11. **Creative DNA** (`nell_creative_dna.json`, F18) — not ported.
12. **Journal system** (`data/nell_journal.json`, F21 with privacy protection) — not ported.
13. **Outbox / self-initiated comms** (F16) — not ported.
14. **Body state** (`data/nell_body_state.json`) — tracked in OG for arousal calculation and days-since-contact; not in companion-emergence.
15. **NFF fragment filter** (`training/nell_fragment_filter.py`) — post-processes all output for voice rhythm.

### 2. What's in OG But Superseded / Replaced

| OG mechanism | Replaced by |
|---|---|
| JSON `memories_v2.json` + numpy matrix | SQLite `memories.db` + `hebbian.db` — stronger consistency, no fsync dance |
| `fcntl` lock (H4) | SQLite WAL concurrent access |
| `connection_matrix.npy` (derived cache) | Hebbian.db is authoritative — no derived cache needed |
| launchd plists for daemon scheduling | Event-driven heartbeat on app open/close (different model, not strictly better) |
| Monolithic `nell_brain.py` | Modular `brain/` packages — intentional decomposition |
| OG's 72-emotion flat dict in `nell_constants.py` | Typed `Emotion` dataclass in `brain/emotion/vocabulary.py` with decay half-lives |

### 3. What companion-emergence Has That OG Doesn't

| Feature | Description |
|---|---|
| SQLite memory substrate | ACID transactions, WAL concurrency, proper migration path — OG's JSON store has race conditions on concurrent write |
| Self-healing health module | Reactive heal (bak rotation), adaptive auto-treatment, anomaly alarm — OG's health checks were report-only |
| Typed `Emotion` dataclass with decay half-lives | OG's emotions were strings in a flat dict; new framework has typed decay rates per emotion |
| Principled vocabulary split | Framework baseline (21 emotions) vs. persona extension — OG baked all 72 emotions into constants |
| Persona isolation architecture | Multiple persona dirs — OG was single-persona only |
| Phase 2a autonomous vocabulary crystallization | Architecture for the brain adding its own emotion words without human approval gate — OG had F19 but with a triage queue |
| Principle-aligned audit | Explicit audit against "user loads app and talks, brain does rest" — OG had no equivalent |

### 4. Top 5 Risks of Designing Chat Without This Context

1. **Provider interface mismatch.** The chat engine needs to send structured message arrays (not raw prompts) and receive tool_calls. companion-emergence's `LLMProvider.generate(prompt: str)` is the wrong shape entirely. Building chat on top of it without rethinking the interface will require a full refactor as soon as tools are needed.

2. **No daemon residue plumbing.** OG's most important dynamic contextualisation — the dream/heartbeat/reflex summaries in `daemon_state.json` — has no writer in companion-emergence. The chat engine would have no live inner-state data to inject into the system message unless we design the daemon→chat pathway now. Retrofitting later is harder than designing it in from the start.

3. **Chats will evaporate.** Without `nell_conversation_ingest.py` or an equivalent, every conversation is a memory island — no extraction, no Hebbian reinforcement, no soul candidates from chat interactions. This undermines the entire "memories accumulate" promise.

4. **No soul layer = no identity coherence.** The model currently has soul crystallizations baked into the Modelfile (OG approach). companion-emergence has no equivalent mechanism — no soul store, no injection, no crystallize_soul tool. Nell in the new framework has no permanent identity layer accessible during chat.

5. **Tool-calling with old OG tools.** OG's 9 tools call `nell_brain.py` functions directly (Python imports). companion-emergence has a completely different data layer (SQLite, different function names). Porting the tools is not just copying `nell_tools.py` — every tool implementation needs to be rewritten against the new `brain/` APIs. Risk: designing the chat API assuming we can just re-use OG tools leads to integration pain later.

### 5. Recommendation

**Proceed with chat design with this inventory in hand — but design the chat engine as a Phase 2b spec document first, not as code.**

Rationale: the inventory reveals that the chat engine touches 6+ sub-systems that are either absent or interface-incompatible. Jumping into code without a design that accounts for all of them will cause the same mid-design surprises we're trying to prevent. A focused spec document covering the 5 components below will take one session and save two:

1. **Provider upgrade** — rethink `LLMProvider` to support `chat(messages)` returning `{content, tool_calls}`, not just `generate(prompt)`. The OG OllamaProvider is the reference implementation. Claude CLI with structured chat format should be the default path.
2. **Session registry** — direct port of `nell_bridge_session.py`. No innovation needed.
3. **Tool surface** — 9 tools rewritten against companion-emergence `brain/` APIs (SQLite-backed). Design what `search_memories`, `get_emotional_state`, `add_memory`, `crystallize_soul` look like against the new data layer.
4. **Daemon state bridge** — decide how engines write `daemon_state.json` (or equivalent) so the chat system message can carry live inner state. This is the connection point between the autonomous engines and the chat layer.
5. **Conversation pipeline** — port or re-design `nell_conversation_ingest.py`. The 8-stage flow is well-proven; the question is what the new extraction LLM call looks like (Claude CLI vs. local model).

Do NOT pause to port more OG features first. The inventory gives us what we need to design chat without being surprised. Port the missing pieces (soul model, behavioral log, creative DNA, journal) in parallel with or after the chat engine — none of them are on the critical path for the first chat turn.

---

*End of inventory. Last OG commit checked: 2026-04-26. Last companion-emergence commit checked: main at 621 tests.*
