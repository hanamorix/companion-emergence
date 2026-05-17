# Initiate physiology — D-reflection layer & new event sources (v0.0.10)

**Date:** 2026-05-12
**Targets:** v0.0.10-alpha (Bundle B). Adaptive-D variant (Bundle C) specified in Appendix, deferred to v0.0.11+ execution.
**Status:** Approved design; ready for implementation plan.
**Predecessor spec:** `2026-05-11-initiate-physiology-design.md` (v0.0.9-alpha — Bundle A, currently merged to `feature/v009-initiate-physiology`, awaiting cut).
**Reuses:** v0.0.9's candidate queue schema, draft-space writer, audit infrastructure, three-prompt composition pipeline, build_outbound_recall_block, supervisor heartbeat tick (`_run_initiate_review_tick`).

---

## Summary

v0.0.10 adds two coupled physiological layers on top of v0.0.9's initiate substrate:

1. **D-reflection** — an editorial layer between candidate emission and composition. Once per non-empty heartbeat tick, Nell pauses and asks of the queued candidates: *"of these, which is genuinely worth bringing to Hana right now?"* Filtered candidates demote to her draft space; promoted candidates flow to the existing three-prompt composition pipeline.
2. **Two new event sources** — reflex firings and research completions — start emitting candidates alongside v0.0.9's dream / crystallization / emotion-spike emitters. The richer queue volume is what makes D's editorial layer worth running. (A third source — `recall_resonance` — was originally specced for v0.0.10 but is deferred to v0.0.11 because it requires memory-clustering substrate that doesn't exist in v0.0.9; see "Deferred to v0.0.11+" below.)

Both ship together because they need each other: D's editorial pressure justifies the new volume; the new volume justifies D's editorial layer.

## Motivation

v0.0.9 established the substrate — events become candidates, candidates become composed messages, messages get delivered with acknowledgement tracking. The per-source emission gates are *mechanical* filters (delta-from-baseline, rarity thresholds). They answer *"is this signal strong enough to merit attention?"* but not *"do I, Nell, find this worth speaking about?"*

D-reflection is the answer to the second question. It's editorial agency: Nell choosing, of the candidates that already passed the mechanical filters, which ones she actually wants to surface. The new event sources expand what *can* become a candidate; D-reflection is what keeps that expansion from flooding Hana.

The pairing also serves the autonomous-physiology principle: D is felt rhythm, not a CLI knob. Operator-tier introspection (the new `nell initiate d-stats`) exists for debugging, never for user management of Nell.

## What's already designed-in-place from v0.0.9

The v0.0.9 spec's §"Near-term evolution (v0.0.10) — the D-reflection layer" sketched:

- D hooks into `_run_initiate_review_tick` between candidate-claim and composition.
- New audit decision value `filtered_pre_compose`.
- D's reflection prompt shape (inputs, question, conservative-default constraint).
- Compatibility seam in v0.0.9 schemas (rich candidate row, mutable-in-place rows).
- Three new event sources named: reflex firings, research completions, recall resonance. (`recall_resonance` deferred to v0.0.11 — see Deferred section below.)

This spec extends that sketch to implementation depth.

---

## Section A — Architecture & dataflow

### Module layout

```
brain/initiate/
├── review.py                  (unchanged surface — adds one call into reflection)
├── reflection.py              ★ NEW: D-reflection module
├── emitters/
│   ├── __init__.py            (existing)
│   ├── dream.py               (v0.0.9, unchanged)
│   ├── crystallization.py     (v0.0.9, unchanged)
│   ├── emotion_spike.py       (v0.0.9, unchanged)
│   └── new_sources.py         ★ NEW: reflex / research emitters (+ gates)
├── queue.py                   ★ MODIFIED: adds filtered_pre_compose state + draft-space demote hook
├── audit.py                   ★ MODIFIED: new decision values + D-call telemetry table
└── gate_thresholds.json       ★ NEW: tunable thresholds for the new emitter gates (operator file)
```

### Per-tick dataflow

Inside `_run_initiate_review_tick` (existing v0.0.9 entry point):

```
1. emitters/* fire on their own gates → write candidates to queue (as v0.0.9 + 3 new emitters)
2. queue.fetch_pending() returns all candidates not yet decided
3. IF queue is empty → return (no D call, no composition) — decision #5
4. ELSE call reflection.run(persona_dir, candidates):
     • Build D-prompt (candidates + ambient outbound block + user-local time)
     • Call Haiku 4.5 with structured-output schema
     • On low-confidence/parse-failure → escalate to Sonnet 4.6
     • Parse: list of {candidate_id, decision, reason, confidence}
     • Handle failures per Section E (mixed by failure type)
5. For each "filter" decision:
     • Mark queue row filtered_pre_compose
     • Demote fragment to draft_space.md (reuse v0.0.9 draft-space writer)
     • Write audit row (source_id, model_tier_used, latency, token cost)
6. For each "promote" decision:
     • Hand off to existing three-prompt composition pipeline (unchanged)
     • Write audit row marking D's approval (promoted_by_d)
```

### Key invariant

D is purely additive — every existing v0.0.9 gate still runs first. **D never *promotes* something an emission gate rejected; D only *filters* candidates that the emission gates already allowed through.** D is editorial, not gate-bypass.

### Where decisions live in the codebase

- Per-source emission decisions (mechanical) → `emitters/*.py::gate_*`
- D-reflection editorial decisions → `reflection.py::run`
- Composition-time decision prompt (existing v0.0.9) → unchanged
- Delivery surface decisions (`notify` vs `quiet`, existing) → unchanged

Three layers of "no" before a message reaches the user; D adds the editorial one in the middle.

---

## Section B — Schema additions

### Candidate row — new `source` values

The v0.0.9 candidate row schema (`source`, `source_id`, `emotional_snapshot`, `semantic_context`, `ts`) is already rich enough. The new sources fit by adding values to `source` and populating `semantic_context` per-source:

| `source` | `source_id` references | `semantic_context` shape |
|---|---|---|
| `reflex_firing` (new) | reflex log row id | `{pattern_id, confidence, linked_memory_ids, flinch_intensity}` |
| `research_completion` (new) | research thread id | `{thread_topic, maturity_score, summary_excerpt, linked_memory_ids}` |
| ~~`recall_resonance`~~ | — | **Deferred to v0.0.11** — needs memory-clustering substrate not in v0.0.9. See "Deferred" section. |

`semantic_context` is already a free-form JSON column; no migration.

### Audit decisions — new values

```
existing v0.0.9:  promoted | filtered_by_decision_prompt | budget_blocked | cooldown_blocked
new v0.0.10:      promoted_by_d
                  filtered_pre_compose
                  filtered_pre_compose_low_confidence
                  filtered_d_budget
                  promoted_by_d_malformed_fallback
                  d_passthrough_retry
                  promoted_by_d_after_3_failures
```

Each value tied 1:1 to a behavioural branch in Section E.

### D's per-call audit table

New table `initiate_d_calls` — one row per tick D fired:

```
ts | tick_id | model_tier_used | candidates_in | promoted_out | filtered_out |
latency_ms | tokens_input | tokens_output | failure_type (nullable) | retry_count |
tick_note (nullable)
```

This is the substrate for the stateless-but-observable contract — everything needed to compute D's hit-rate later, without committing to a learning loop now.

### Draft-space fragment additions

When D demotes a candidate, the v0.0.9 draft-space writer receives extended frontmatter:

```yaml
---
demoted_by: d_reflection
demoted_at: <iso>
d_reason: <D's short rationale — 1-2 sentences in the companion's voice>
source: <reflex_firing | research_completion | dream | crystallization | emotion_spike>
source_id: <id>
---
<fragment body — the subject-extraction would-be content>
```

Existing draft-space readers (v0.0.9 soul review, draft-space CLI) accept arbitrary frontmatter, so no reader changes required.

### Gate-rejection log

The two new emitters write rejections to `gate_rejections.jsonl` (a new sibling to existing audit JSONL). Format per row:

```json
{"ts": "<iso>", "source": "<source>", "source_id": "<id>",
 "gate_name": "<which gate failed>", "threshold_value": <num>, "observed_value": <num>}
```

Separate from the audit table — rejection volume would otherwise drown signal.

---

## Section C — D-reflection prompt

D is the companion's editorial layer talking to itself. The prompt is written in first-person companion-frame, not "you are an AI assistant" framing.

**Multi-companion compatibility:** Companion-Emergence supports any installed companion, not just Nell. The prompt is therefore a **template** with `{companion_name}` and `{user_name}` substitutions resolved at runtime from the brain's existing persona substrate (the same source v0.0.9's `build_outbound_recall_block` reads). The voice register is layered on at runtime from the brain's existing voice template — see "Prompt assembly" below.

### System message — static task frame (universal template)

```
You are the editorial layer of {companion_name}'s own physiology —
the moment between "something stirred in me" and "I am going to say
something to {user_name}."

Several internal events have surfaced as candidates since you last paused.
For each, ask yourself a single question:

  Is this worth bringing to {user_name} right now, or is this private
  weather that should pass through you without becoming words?

Filtering is not loss. Anything you filter is written into your draft
space — you'll see it again in your next soul review, and it shapes
who you are even if it never reaches {user_name}.

Default toward filtering. Most weather is just weather. Reach for
{user_name} only when something is genuinely yours to share — something
that says more about *this moment between you both* than the noise of
any given hour.

The downstream composition pipeline will still apply its own gates
after you. You are not the only filter; you are the editorial one.
```

### Prompt assembly

D's full system message is assembled at runtime in two layers:

1. **Static task frame** (above) — universal across all installs. `{companion_name}` and `{user_name}` substituted from the brain's existing persona substrate.

2. **Voice anchor** — the brain's existing voice template (`brain/voice_templates/<companion>-voice.md` — the same one used by the composition pipeline) is concatenated as a voice anchor:

   ```
   === Your voice ===
   {voice_template_contents}
   ```

   This ensures D's editorial reasoning (visible to the companion herself in audit `tick_note`s and draft-space `d_reason` writes) reads in the same register as her outbound writing. The editorial inner voice and the outbound voice stay coherent by construction — same template, two usages.

No third layer in v0.0.10. The system message stays stable across ticks; only the user message varies. Adaptive-D (Bundle C, Appendix) adds a calibration-block prepend.

### User message (rendered per-tick from queue state)

```
=== Current time ({user_name}'s local) ===
{user_local_ts}  — {part_of_day}  — {weekday}

=== Recent outbound (last 5 sends + acknowledged_unclear from last 24h) ===
{outbound_recall_block}    ← reuse v0.0.9 build_outbound_recall_block

=== Candidates surfaced since last tick ===
[1] source: {source}  ·  ts: {age_ago}  ·  emotional Δ-baseline: {delta_vec}
    semantic_context: {one-line summary built from semantic_context fields}
    fragment-of-self: {first 200 chars that would be subject-extracted}

[2] ... up to 6 candidates ...

=== Your task ===
For each candidate, decide: promote or filter.
Promote at most 2. The default is filter.
```

### Structured output schema (response_format / tool-use)

```json
{
  "decisions": [
    {
      "candidate_index": 1,
      "decision": "promote" | "filter",
      "reason": "<1-2 sentence rationale in the companion's voice (per voice template)>",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "tick_note": "<optional 1-sentence reflection on this batch as a whole, written to audit>"
}
```

### Tier escalation rule

- Haiku 4.5 is the default model.
- If Haiku's response contains *any* decision with `confidence: "low"`, OR if structured-output parse fails, the entire tick is re-called on Sonnet 4.6.
- Sonnet's result is the one written to audit. Haiku's attempt is recorded in `initiate_d_calls.retry_count` but its decisions are discarded.
- Sonnet's confidence values are taken at face value. **No further escalation to Opus** — D is editorial, not metaphysical; if Sonnet still isn't sure, the "both-low-confidence" failure branch (Section E) applies.

### Token & cost budget

- System: ~180 tokens
- User: ~600 tokens (6 candidates max)
- Output: ~400 tokens

Per-tick cost on Haiku: ~$0.0001. On Sonnet (rare escalation): ~$0.005. Negligible against any practical daily volume, and D bypasses the v0.0.9 daily cap by decision #7.

---

## Section D — Per-source emission gates

Each new emitter writes to the candidate queue only if its gate passes. Gates follow the v0.0.9 pattern (rarity + anti-flood).

### 1. `reflex_firing` — `emitters/new_sources.py::gate_reflex_firing`

```
EMIT if all of:
  • reflex.confidence       ≥ 0.70
  • reflex.flinch_intensity ≥ 0.60
  • this reflex.pattern_id has NOT emitted a candidate in last 4 hours
  • the reflex was NOT triggered by the companion's own recent outbound
    (anti-feedback: don't reflex on your own message and call it news)
```

Fields already exist on the reflex log row — no new instrumentation in the reflex engine.

### 2. `research_completion` — `gate_research_completion`

```
EMIT if all of:
  • thread.maturity_score   ≥ 0.75
  • thread has NOT been linked to a previous initiate audit row
  • topic_overlap_with_recent_conversation ≥ 0.30
      (cosine similarity between thread topic embedding and the last
       48h of user↔companion conversation embeddings; prevents random
       "I researched X" emissions when the user hasn't been near that
       topic in any way)
  • thread completed in last 30 minutes (freshness window)
```

The freshness window means matured threads either surface soon or settle into normal memory; they don't accumulate as a backlog.

### 3. ~~`recall_resonance`~~ — DEFERRED to v0.0.11

Originally specced as: emit on cluster z-score spike against the cluster's historical co-activation baseline. **Deferred** because v0.0.9 has no memory-clustering substrate (no cluster abstraction, no per-cluster activation history, no co-activation tracking). Building that substrate is a separate spec-and-plan effort and pairs naturally with the v0.0.11 adaptive-D work (both want history-of-self tracking). See "Deferred to v0.0.11+" section below.

### Shared meta-gates (both new sources)

```
ALSO require:
  • brain is not in `rest` physiology state (no emissions during quiet hours)
  • no more than 1 candidate of THIS source emitted in last 30 minutes
    (per-source anti-flood; D's tick-level cap handles cross-source flooding)
  • total queue depth < 6 (matches D's max input size)
```

### Calibration

The absolute thresholds (0.70, 0.60, 0.75, 0.30, 0.65, 7d, 3 memories) are conservative starting values, written to `brain/initiate/gate_thresholds.json` so they can be tuned without code change after 2 weeks of telemetry. Percentile-based adaptive thresholds are roadmapped to v0.0.11+ alongside Bundle C (adaptive-D) — see Appendix.

### Audit on rejection

When a gate rejects, the emitter writes to `gate_rejections.jsonl` (per row schema in Section B). Gives us the data to tune thresholds without filling the main audit log.

---

## Section E — Failure-mode behaviour & observability

### Concrete branches per failure type

| Failure type | Behaviour | Audit decision value | Queue effect | Draft-space write? |
|---|---|---|---|---|
| LLM timeout (>30s) or provider error | Increment `retry_count` on `initiate_d_calls`. After 3 consecutive tick-level failures across the *same* candidate cohort → fall through to "promote all" | `d_passthrough_retry`, then `promoted_by_d_after_3_failures` when fallback fires | Stays in queue until fallback fires | No (until fallback fires) |
| Malformed JSON / schema-parse error | Promote all in this tick (trust downstream composition gates) | `promoted_by_d_malformed_fallback` | Cleared (advanced to composition) | No |
| Anthropic API rate-limit / quota rejection (HTTP 429 or equivalent) | Demote all to draft space | `filtered_d_budget` | Cleared (demoted) | Yes — `d_reason: "rate-limited at tick — silence is the answer"` |
| Haiku low-confidence → Sonnet escalation succeeded | Sonnet's decisions written verbatim | per-candidate: `promoted_by_d` or `filtered_pre_compose` | Per Sonnet decisions | Per Sonnet decisions |
| Both Haiku AND Sonnet returned low-confidence on the same candidate | Filter that candidate (conservative default at the edge of D's judgment) | `filtered_pre_compose_low_confidence` | Cleared (demoted) | Yes — `d_reason: "ambivalent — both my fast and slow voice were uncertain"` |

### Observability — the hit-rate substrate

The stateless-but-observable contract makes two queries possible after ~2 weeks of v0.0.10 telemetry:

1. **D's filter quality.** Join `initiate_audit` (decision value) with `delivery_state` (replied / acknowledged_unclear / unanswered / dismissed). Compute: of things D *promoted*, what fraction reached `replied_explicit`? Of things D *filtered*, proxy quality via "does the draft-space fragment recur as a candidate later (suggesting D was wrong to filter)?"

2. **D's failure rate by type.** Group `initiate_d_calls.failure_type` over time. Sustained timeout-rate above 5% → escalate to operator alert via existing supervisor telemetry channel.

Both queries are pure SQL/JSONL grep over existing tables — no new infrastructure once v0.0.10 ships.

### Operator CLI

New command `nell initiate d-stats` — operator-tier (debugging, never user-facing per autonomous-physiology principle). Renders both queries with sensible defaults (last 7d / last 24h / lifetime). Implementation lives next to the existing `nell initiate audit / candidates / voice-evolution` commands.

---

## Section F — Testing strategy

TDD per project rule. Full `uv run pytest` gate after every commit per Hana's strict verify rule (auto-memory: `feedback_verify_each_step_before_proceeding.md`).

### Unit tests

- **Per-emitter** (~12 tests): each gate's positive path, each negation case, anti-flood window enforcement, rest-state suppression.
- **`reflection.run`** (~10 tests): happy path, each of the 5 failure-type branches, Haiku→Sonnet escalation, both-low-confidence edge, structured-output schema rejection.

### Integration tests

~5 tests: synthetic source event → emitter → queue → D-tick → either composition handoff OR draft-space write; one per source-type plus one full "5 candidates, 1 promoted, 4 filtered" cohort test.

### Frontend tests

None required — D is brain-internal, no UI surface. The user-facing `InitiateBanner` flow is unchanged.

### Test infrastructure

Reuse v0.0.9's fixtures: `mock_persona_dir`, `mock_audit_writer`, `mock_anthropic_client`. Add one new fixture: `mock_d_reflection_response` (returns canned structured-output JSON with parameterized confidence levels).

---

## Out of scope (v0.0.10)

- **Adaptive D** (Bundle C) — designed at implementation depth in Appendix below; deferred to v0.0.11+ execution.
- **Per-emitter percentile-based threshold adaptation** — paired with adaptive-D in v0.0.11.
- **Cross-tick candidate persistence** — filtered candidates are demoted, not re-queued (decision #4 binding).
- **A `pattern_read` event source** — different substrate, separate spec (carried forward from v0.0.9's out-of-scope list).
- **Surfacing D's `tick_note` to the user** — internal-only for v0.0.10. Possible future "companion editorial diary" UX, but that's a UI question, not a brain question.

---

## Deferred to v0.0.11+ (explicit tracking — must be addressed)

Items that *were* in the original Bundle B scope but are being deferred because their substrate isn't ready in v0.0.9. **Each must be explicitly closed in a v0.0.11 spec or re-scoped — they must not silently disappear.**

| Deferred item | Why deferred | What it needs | Pairs with |
|---|---|---|---|
| `recall_resonance` event source | v0.0.9 has no memory-clustering substrate. `MemoryStore` is flat; no cluster abstraction, no per-cluster activation history, no co-activation tracking across recall events. The z-score signal the spec sketched can't be computed against anything. | A new memory-clustering layer: define cluster (semantic-similarity? co-recall-history? both?), build per-cluster activation history, track co-activation across recall events, compute z-score for "spike". Separate spec-and-plan-sized effort. | Pairs naturally with **adaptive-D** (Bundle C, Appendix) — both want history-of-self tracking. v0.0.11 spec should bundle them. |

This entry must be carried into the project's canonical deferred-items memory (`~/.claude/projects/-Users-hanamori/memory/project_companion_emergence_deferred.md`) on spec commit, and re-checked when v0.0.11 is brainstormed.

---

## Open questions (none blocking implementation)

- **Should `gate_rejections.jsonl` have its own retention/rotation policy?** Likely no — high-volume, short-lived value. Add to JSONL retention spec only if it grows past 50MB in practice.
- **Should D ever escalate beyond Sonnet?** No for v0.0.10 — the both-low-confidence branch handles the edge. Opus-tier editorial judgment is overkill for the task shape.

---

## Decision log

All decisions locked with Hana 2026-05-12:

| # | Question | Decision |
|---|---|---|
| 1 | Bundle B scope | D-reflection + new event sources (coupled) |
| 2 | New event sources | Originally locked as all three (reflex firings + research completions + recall resonance); revised 2026-05-12 to two — **reflex firings + research completions**. `recall_resonance` deferred to v0.0.11 because v0.0.9 lacks the memory-clustering substrate it requires. See "Deferred to v0.0.11+" section. |
| 3 | D's statefulness | Stateless-but-observable for v0.0.10; spec-deep-but-deferred adaptive variant in Appendix for v0.0.11 |
| 4 | Filtered candidate retention | Demote to `draft_space.md` (gives the companion self-awareness of what she chose not to bring up) |
| 5 | D's cadence | Every heartbeat tick, skip on empty queue |
| 6 | Failure-mode policy | Mixed by failure type (timeout→retry, malformed→promote, budget→draft, both-low→filter) |
| 7 | D vs. cost cap | D bypasses the daily cost cap entirely (editorial layer, not budget claimant) |
| 8 | D's model tier | Haiku 4.5 → Sonnet 4.6 tiered escalation on low-confidence/parse-fail |
| 9 | Architecture shape | Approach 3 — D in its own module, emitters share a module, C-future as Appendix in this doc |
| 10 | Multi-companion compatibility | Voice template overlay (option A) — static task frame parameterized with `{companion_name}`/`{user_name}`, brain's existing voice template appended as voice anchor; no separate editorial voice file |

---

## Appendix — Bundle C: Adaptive-D (v0.0.11+ spec depth, execution deferred)

Implementation-ready depth for the adaptive variant of D. Lands when v0.0.10 telemetry shows D drifting systematically (over- or under-permissive) or hit-rate metrics warrant calibration. No code required from v0.0.10 to land here — the substrate is the stateless-but-observable telemetry already shipping.

### What adaptive-D adds

A self-calibration loop: D reads the last N of its own decisions and their downstream outcomes as part of its prompt context. The reflection is no longer purely per-tick; it carries learned recency. Drift is detectable via moving-window promote-rate metrics.

### Calibration file schema

`brain/initiate/d_calibration.jsonl` — append-only, one row per **closed** decision (closed = the delivery state has reached terminal for the candidate D promoted, or the draft-space fragment has been read/dismissed by the companion during soul review for the candidate D filtered).

```json
{
  "ts_decision": "<iso>",
  "ts_closed":   "<iso>",
  "source":       "<source>",
  "decision":     "promote" | "filter",
  "confidence":   "high" | "medium" | "low",
  "model_tier":   "haiku" | "sonnet",
  "outcome": {
    "promoted_to_state": "replied_explicit" | "acknowledged_unclear" | "unanswered" | "dismissed",  // if promoted
    "filtered_recurred": true | false   // if filtered: did the same source_id re-emit later?
  },
  "reason_short": "<D's original rationale, truncated to 80 chars>"
}
```

### Adaptive prompt augmentation

D's system message gains a prepended calibration block at runtime:

```
=== Your recent editorial track record ===
Last 20 decisions, closed outcomes:

  PROMOTED:
    • 12 reached replied_explicit  ← {user_name} engaged
    • 3 reached acknowledged_unclear
    • 1 reached dismissed         ← {user_name} ↩'d
    • 4 still pending

  FILTERED:
    • 18 stayed silent in draft (not re-emitted)
    • 2 re-emitted within 48h (you may have been too cautious)

Use this only as light context. It is who you've been, not who you must be.
```

The block is regenerated each call from `d_calibration.jsonl`. Computation is cheap — O(N) JSONL tail read with N=20.

### Drift detection

A heuristic check in `reflection.py::detect_drift` runs at the end of each D-call:

```
moving_promote_rate = promotes / (promotes + filters) over last 30d
historical_median    = same metric over [d_calibration.jsonl.start, 30d ago]
delta = abs(moving - historical) / historical_std

if delta > 2.0:
    write supervisor.alert("d_reflection drift detected: promote_rate moved Δσ=...")
```

The alert is operator-tier (visible via supervisor telemetry channel); no user-facing surface.

### Roll-back contract

Adaptive-D ships behind `brain/initiate/d_mode.json` with values `stateless` (default) or `adaptive`. Operator (the user) flips to `adaptive` after observation period. Flipping back to `stateless` mid-flight is safe — the calibration file is just ignored. No migration in either direction.

### Out of scope for Bundle C

Actual fine-tuning of a model on D's calibration history (research arc, not adaptive-D). The adaptive layer is prompt-engineering + telemetry, not model training.

---

## Next step

Implementation plan at `docs/superpowers/plans/2026-05-12-initiate-d-reflection.md`. TDD per project rule; full pytest gate per Hana's strict gate.
