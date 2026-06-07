# Organ Maturity Manifest

> **Living doc.** The single answer to "is this organ stable, experimental, or dormant?" — without re-auditing. Update it **once per minor release, after feature merge, before tag** (the standing wire-back cadence, see bottom). Established by the v0.0.30 stable-base pass (spec `docs/superpowers/specs/2026-06-03-v0.0.30-stable-base-design.md`; source audit `docs/audits/2026-06-02-hygiene-investigation.md`).

An "organ" is a brain subsystem with a producer and a consumer. The **wire-back invariant** (CLAUDE.md Hard rules): an organ is *done* only when it both **reads from** and **feeds into** the emotional/memory loops. This manifest records where each organ sits on that line.

Last refreshed: **2026-06-06** (metabolic-cost-control v0.0.31 — salience signal, tool recruitment + recruit-on-reach, reflection debounce, `cli_throttle` promoted to CORE-STABLE; item 26 resolved. Earlier same-day: token-cost + file-access hotfix — `read_file`/`list_directory` CORE; `file_access.jsonl` audit-tier).

---

## CORE-STABLE (live · used · fully wired both directions)

The shippable base. Each fires on a live supervisor/chat path, has ≥1 reader, and a §Wiring spec entry.

| Organ | Reads | Feeds |
|---|---|---|
| heartbeat | memory/emotion/body | dreams, reflex, growth, research, decay |
| MemoryStore recall → salience | recall events | `recall_count` → forgetting salience |
| forgetting (FADE/LOSE/graveyard) | composite salience (emotion/hebbian/recall/soul/freshness) | graveyard + grief |
| emotion-aggregate | all active memories' `emotions` | the felt EmotionalState read by chat/body/dream/reflex |
| **ingest emotion-seeding** | conversation transcript + registered vocab | **writes `emotions` onto bulk-extracted memories** → feeds emotion-aggregate/body/dream/felt-time/forgetting. *Promoted to CORE in v0.0.30 (W7 fix A2 + historical backfill A3); previously the affect-deaf gap.* |
| dream | emotion (mood-gate) + soul + grief | hebbian reinforce, memories, feed |
| ingest pipeline (buffer→extract→commit) | conversation buffer | MemoryStore + hebbian + soul queue (+ emotion, post-v0.0.30) |
| chat engine + recall block | memory/emotion/body/ambient blocks | reply + buffer |
| monologue (three-tier) | turn context + aggregate emotion | `monologue_trace` memory (aged by forgetting) + ambient + feed |
| attunement (5-category) | conversation buffer + reply | `learned_patterns` + `current_read` + ambient + feed + addressability |
| narrative_memory | co-activated memories | arcs + ambient block + MCP tools |
| grief | forgetting losses + deprecated arcs | Loss panel + ritual surface |
| soul crystallisation | soul candidates | SoulStore (read by 6 sites) + feed |
| creative_dna crystallizer | growth signals | identity DNA (prompt framing) |
| research | interest + topic overlap | research artefacts + feed |
| **felt_time** | `HeartbeatResult.reflex_fired` + session-buffer chat-turn delta | `chat_activity` driver + reflex tick-context → lived-age rate → forgetting effective-fade. *Promoted to CORE in the Phase-C pass (#2 fix); previously fed hardcoded `0`.* |
| **draft_space** | `draft_space.md` fragments since a per-persona cursor (`read_drafts_since`) | candidate-gated "recent private fragments" block in the soul-review prompt → crystallisation decisions. *Promoted to CORE in the Phase-C pass (W1 reader); producers fired constantly but were unread.* |
| **voice-edit (UI leg)** | `initiate_delivered` event (`kind`+`diff`) | inline `VoiceEditPanel` in NellFace → `acceptVoiceEdit`/`rejectVoiceEdit` → `/initiate/voice-edit/{accept,reject}` → `voice.md` + SoulStore `voice_evolution`. *Promoted to CORE in the Phase-C pass (W3); the daily producer + server endpoints existed but the UI couldn't accept/reject.* |
| **file access (`read_file`/`list_directory`)** | a file/dir path the user explicitly asked about (read-only, size/entry-capped) | the turn's reply + `file_access.jsonl` audit + — via Nell choosing to act on what she read — the memory/emotion loops (she can `record_monologue`, remember, reflect). *Promoted to CORE in the token-cost+file-access hotfix (2026-06-06): producer fires on the live MCP chat path when asked; through-path despatch test asserts it (`test_read_file.py`/`test_list_directory.py`); §Wiring in `docs/superpowers/specs/2026-06-05-on-demand-file-access-design.md`.* |
| **salience signal** (`brain/chat/salience.py`) | the user's turn text + prior user turn + the persona emotion vocabulary (A4 startup-load) | per-turn tool recruitment (`select_tools`) + reflection debounce (`should_reflect`). *Promoted to CORE in v0.0.31 (metabolic-cost-control): pure no-LLM scorer, fails open to maximal; produced every `respond()`; two live readers; through-path tests `test_salience.py`/`test_tool_recruit.py`; §Wiring spec §4.* |
| **tool recruitment** (`select_tools` + `reach_for_capability`) | the salience signal | per-turn `--allowedTools` (slim core on trivial turns; heavier faculties recruited on relevant flags) + the recruit-on-reach re-invoke that pulls the full suite when Nell reaches. *Promoted to CORE in v0.0.31: producer fires every `respond()`; reader is `run_tool_loop`; through-path test `test_tool_loop_recruit.py` (recruit-on-reach + one-expansion bound); §Wiring spec §4. Preserves full agency (recruit-on-reach guarantees same-turn access).* |
| **reflection debounce** (`should_reflect` + `reflection_state.json`) | the salience signal + per-kind cursor | gates the **attunement** pass-2 spawn to significant turns (monologue pass-2 deliberately NOT gated — unique deliberate content). *Promoted to CORE in v0.0.31: live on the chat path; reader is the attunement spawner; through-path test `test_reflection_gate.py`; §Wiring spec §3.3/§4. Fails open (reflect now) on a corrupt cursor.* |
| **`cli_throttle`** (`brain/bridge/cli_throttle.py`) | interactive-activity signal (`engine.respond` marks it every turn) | gates the six retrying background LLM consumers (dream/reflex/research/soul-review/initiate/voice-reflection) — they defer (re-fire next cadence tick) while chat is recently active. *Promoted to CORE in v0.0.31; **closes deferred item 26** (global CLI throttle). Interactive never waits (priority lane); fails open to today's behaviour + logs once; through-path tests `test_cli_throttle.py` + `test_background_yields.py`; §Wiring spec §3.5/§4. Pass-2 spawns intentionally NOT gated (one-shot, no retry — defer would drop interior).* |

---

## EXPERIMENTAL-LIVE (fires in real use; one or both wire-back legs open — *labelled, accepted*)

These run but aren't fully closed loops. Labelled here so the half-wired state is a **known accepted condition**, not silent rot. Each is a Phase-C / future candidate (see spec §8 OUT list); not a v0.0.30 blocker.

| Organ | What's open | Status |
|---|---|---|
| recall_resonance (W4) | emits outbound candidates but gates so tight emission is unproven; no recall/hebbian/emotion write-back on the return arc | EXPERIMENTAL — tune / reinforce deferred |
| body → initiate gating (W10) | body energy/exhaustion is live + read by the prompt, but never **gates** the initiate loop (low energy doesn't suppress dreams/drafts/sends) | EXPERIMENTAL — design call deferred |

---

## DORMANT / UNUSED (cut · deferred · fenced — do NOT wire just to wire)

| Organ | Reality | v0.0.30 action |
|---|---|---|
| `MemorySearch` semantic/spreading recall (`brain/memory/search.py`, W6) | zero production callers; chat recall uses `store.search_text` + `forgetting/recall.py`; embeddings still used for dedupe + narrative membership | **CUT** (B1) |
| `crystallize_reflex` growth path (W2) | ~90% built + 37 tests, but chunks 7–8 (apply into `run_growth_tick` + the Hana-in-the-loop acceptance gate) never landed → zero prod callers. The reflex **FIRE** path is live & separate. | **DEFERRED — documented** (B2; 3-place ledger). Dead-but-tested code retained (tests prevent rot). A deliberate Tier-2 making/growth feature. |
| `works` / `save_work` (W9) | functions, model-pull only; no autonomous writer, no live reader/feed; pre-Maker substrate | **FENCED** (B3) — kept as a tool surface, not wired |

---

## Coverage gaps (known, accepted this release)

- **Stream keepalive** (v0.0.28 idle-timeout area) — **MECHANISM ADDED + TESTED v0.0.30** (2026-06-04). The earlier claim here ("`reply_chunk` IS the keepalive mechanism") was **wrong**: `reply_chunk` only carries real model text, so there was *no* keepalive at all. A silent provider stretch — first-token latency on a large prompt, or a tool-use round-trip (`record_monologue` etc., whose `tool_call` frames flush only *after* the turn) — sent zero frames, and the client (`app/src/streamChat.ts`, 60s idle budget) killed the WS → `WebSocketDisconnect(1006)`. This was the live-validation "stream idle timeout" (distinct from the provider's own 60/120s watchdog, which is why `stream_timeouts.jsonl` stayed empty — the client wins the race). Fix: the server forward loop now waits on `chunk_q` with `_STREAM_KEEPALIVE_SECONDS=15.0` and emits a `{"type":"keepalive"}` frame on each silent interval (`brain/bridge/server.py`); the client handles it as a benign idle-resetting no-op (`case "keepalive"`). Covered by `tests/bridge/test_endpoints.py::test_stream_emits_keepalive_during_silent_provider_stretch` (drives a deliberately-silent provider through the real WS endpoint with a monkeypatched tiny interval — no 90s wall-clock wait) + `app/src/streamChat.keepalive.test.ts`. **Remaining gap — NARROWED in v0.0.31:** background Claude-CLI contention from the *retrying* consumers (dream/reflex/research/soul-review/initiate/voice-reflection) now yields to active chat via `cli_throttle` (**item 26 resolved**). The residual contention is the per-turn async **pass-2** (monologue/attunement extraction), which is deliberately NOT throttled (one-shot, no retry — defer would drop interior) and is instead cut by the reflection debounce + attunement input-window. Fully throttling pass-2 without loss needs a blocking/queued slot (deferred follow-up).

- **`chat_usage.jsonl` — audit-tier instrumentation (added in the token-cost hotfix, 2026-06-05).** Per-call CLI token usage (`input_tokens`/`output_tokens`/`cache_creation`/`cache_read`/`total_cost_usd`) captured from the result frame for chat *and* background `generate()` calls (`brain/bridge/usage_log.py`). **Write-only by design** — reader is the operator now / a future cost panel / the future item-26 global throttle. Not an isolated-organ failure: deliberately instrumentation, like `reflex_audit`/`stream_timeouts`. Bounded via `_ROLLING_LOG_POLICIES`. (The lean CLI invocation + this capture together: the framework floor dropped 51K→20K cache-creation per call in a lean spike.)
- **`file_access.jsonl` — audit-tier instrumentation (added in the file-access hotfix, 2026-06-06).** Per-call read-only file access (`tool`/`path`/`resolved_path`/`bytes`/`ok`/`error`) appended by `read_file`/`list_directory` (`brain/tools/impls/read_file.py::_audit`, reused by `list_directory`). The *tools* are CORE-STABLE (see table); the *log* is write-only-by-design audit trail (reader = operator / future review), bounded via `_ROLLING_LOG_POLICIES`. Same shape as `reflex_audit`/`chat_usage`.
- **Emotion-vocab startup load — wiring *restored* (2026-06-05).** The persona emotion vocabulary was only loaded by the supervisor heartbeat/soul-review ticks, so for ~15 min after launch the chat path's `aggregate_state` dropped all persona-extension emotions (flattening emotion → body/dream/felt-time/salience). Now loaded idempotently at bridge startup (`ensure_persona_vocabulary_loaded`, `brain/bridge/server.py`). This *restores* a CORE wiring (emotion-aggregate fed correctly from turn one), not a new organ.
- **Replayed-history window (v0.0.31, driver #1 fix).** The caching spike (2026-06-06) confirmed the Claude Code CLI re-creates the whole prompt every turn (no cross-call prompt caching via `--system-prompt-file`/stdin), so an unbounded conversation buffer was re-billed at cache-creation every turn. Fix: `engine.respond` caps replayed history to the last `_HISTORY_WINDOW_MSGS=80` messages (~40 turns) verbatim; older turns are truncated (NOT summarised — re-summarising a growing head every turn is its own cost) and resurfaced by the per-turn recall block + ingest memories. `apply_budget` ceiling lowered 190K→80K as a token backstop. This is a deliberate short-term/long-term split (buffer = working context, memory = long-term), accepted as bounding in-conversation verbatim recall.
- **Deferred follow-ups (v0.0.31, 3-place ledger):** (1) **pass-2 blocking/queued throttle** — gate monologue/attunement extraction without dropping interior (needs a wait-for-slot, not defer-drop); (2) **throttle rate-budget token bucket** — a global calls/min ceiling, add only if data shows stampedes survive priority-yield; (3) **non-streaming `chat()` text-path usage logging** — `log_usage` covers `generate()` + the streaming chat path, not the no-tools `chat()` text path (surfaced by the caching spike); (4) **attunement-haiku-failure observability** — a failed `_call_haiku` log.warning's but doesn't write `attunement_errors.jsonl`, so transient detector declines (e.g. under CLI contention) are invisible.

---

## Standing wire-back cadence (per minor release: after merge, before tag)

Re-run the wire-back audit and refresh this manifest each minor. Two cheap mechanizable greps seed it (audit §6):

1. **Write-only detector** — any new `*.jsonl`/`*.md`/store writer whose read API has zero non-test callers (caught draft_space W1, reflex-crystallizer W2).
2. **`emotions={}` detector** — any new `Memory.create_new(...)` without an `emotions=` argument (caught the W7 ingest gap).

New organs land **EXPERIMENTAL by default** and are promoted to CORE-STABLE only when they meet the **Organ Definition-of-Done** (see CLAUDE.md): producer on a live path · a test asserting it fires *through* that path · ≥1 reader · a §Wiring spec entry.
