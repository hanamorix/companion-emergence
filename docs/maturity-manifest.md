# Organ Maturity Manifest

> **Living doc.** The single answer to "is this organ stable, experimental, or dormant?" — without re-auditing. Update it **once per minor release, after feature merge, before tag** (the standing wire-back cadence, see bottom). Established by the v0.0.30 stable-base pass (spec `docs/superpowers/specs/2026-06-03-v0.0.30-stable-base-design.md`; source audit `docs/audits/2026-06-02-hygiene-investigation.md`).

An "organ" is a brain subsystem with a producer and a consumer. The **wire-back invariant** (CLAUDE.md Hard rules): an organ is *done* only when it both **reads from** and **feeds into** the emotional/memory loops. This manifest records where each organ sits on that line.

Last refreshed: **2026-06-05** (Phase-C wire-backs — felt_time, draft_space, voice-edit promoted to CORE-STABLE).

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

- **Stream keepalive** (v0.0.28 idle-timeout area) — **MECHANISM ADDED + TESTED v0.0.30** (2026-06-04). The earlier claim here ("`reply_chunk` IS the keepalive mechanism") was **wrong**: `reply_chunk` only carries real model text, so there was *no* keepalive at all. A silent provider stretch — first-token latency on a large prompt, or a tool-use round-trip (`record_monologue` etc., whose `tool_call` frames flush only *after* the turn) — sent zero frames, and the client (`app/src/streamChat.ts`, 60s idle budget) killed the WS → `WebSocketDisconnect(1006)`. This was the live-validation "stream idle timeout" (distinct from the provider's own 60/120s watchdog, which is why `stream_timeouts.jsonl` stayed empty — the client wins the race). Fix: the server forward loop now waits on `chunk_q` with `_STREAM_KEEPALIVE_SECONDS=15.0` and emits a `{"type":"keepalive"}` frame on each silent interval (`brain/bridge/server.py`); the client handles it as a benign idle-resetting no-op (`case "keepalive"`). Covered by `tests/bridge/test_endpoints.py::test_stream_emits_keepalive_during_silent_provider_stretch` (drives a deliberately-silent provider through the real WS endpoint with a monkeypatched tiny interval — no 90s wall-clock wait) + `app/src/streamChat.keepalive.test.ts`. **Remaining gap:** background Claude-CLI contention (soul-review startup burst, backfill, prior-turn async pass-2) still *lengthens* silent stretches — the keepalive masks it from the user, but the principled fix is the global CLI throttle (deferred ledger item 26 / roadmap Tier-1 follow-up).

---

## Standing wire-back cadence (per minor release: after merge, before tag)

Re-run the wire-back audit and refresh this manifest each minor. Two cheap mechanizable greps seed it (audit §6):

1. **Write-only detector** — any new `*.jsonl`/`*.md`/store writer whose read API has zero non-test callers (caught draft_space W1, reflex-crystallizer W2).
2. **`emotions={}` detector** — any new `Memory.create_new(...)` without an `emotions=` argument (caught the W7 ingest gap).

New organs land **EXPERIMENTAL by default** and are promoted to CORE-STABLE only when they meet the **Organ Definition-of-Done** (see CLAUDE.md): producer on a live path · a test asserting it fires *through* that path · ≥1 reader · a §Wiring spec entry.
