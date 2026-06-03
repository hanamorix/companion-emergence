# Organ Maturity Manifest

> **Living doc.** The single answer to "is this organ stable, experimental, or dormant?" — without re-auditing. Update it **once per minor release, after feature merge, before tag** (the standing wire-back cadence, see bottom). Established by the v0.0.30 stable-base pass (spec `docs/superpowers/specs/2026-06-03-v0.0.30-stable-base-design.md`; source audit `docs/audits/2026-06-02-hygiene-investigation.md`).

An "organ" is a brain subsystem with a producer and a consumer. The **wire-back invariant** (CLAUDE.md Hard rules): an organ is *done* only when it both **reads from** and **feeds into** the emotional/memory loops. This manifest records where each organ sits on that line.

Last refreshed: **2026-06-03** (v0.0.30 stable-base).

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

---

## EXPERIMENTAL-LIVE (fires in real use; one or both wire-back legs open — *labelled, accepted*)

These run but aren't fully closed loops. Labelled here so the half-wired state is a **known accepted condition**, not silent rot. Each is a Phase-C / future candidate (see spec §8 OUT list); not a v0.0.30 blocker.

| Organ | What's open | Status |
|---|---|---|
| voice-edit UI (W3) | producer fires daily; the NellFace UI can't accept/reject (server endpoints exist) | EXPERIMENTAL — wire-back deferred to Phase C |
| draft_space (W1) | producers (D-reflection demote, heartbeat wobble) fire constantly; **no reader**. v0.0.30 amended the false re-ingestion promise in `reflection.py` (B6) so the prompt no longer lies; the reader itself is still unbuilt | EXPERIMENTAL — reader deferred |
| felt_time chat/reflex drivers (#2) | the tick is live but fed hardcoded `0` for `chat_activity`/`reflex_firings` | EXPERIMENTAL — supervisor-glue fix deferred |
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

- Stream-keepalive integration test (v0.0.28 idle-timeout area) — see H4 outcome; logged here if the harness was deferred rather than built.

---

## Standing wire-back cadence (per minor release: after merge, before tag)

Re-run the wire-back audit and refresh this manifest each minor. Two cheap mechanizable greps seed it (audit §6):

1. **Write-only detector** — any new `*.jsonl`/`*.md`/store writer whose read API has zero non-test callers (caught draft_space W1, reflex-crystallizer W2).
2. **`emotions={}` detector** — any new `Memory.create_new(...)` without an `emotions=` argument (caught the W7 ingest gap).

New organs land **EXPERIMENTAL by default** and are promoted to CORE-STABLE only when they meet the **Organ Definition-of-Done** (see CLAUDE.md): producer on a live path · a test asserting it fires *through* that path · ≥1 reader · a §Wiring spec entry.
