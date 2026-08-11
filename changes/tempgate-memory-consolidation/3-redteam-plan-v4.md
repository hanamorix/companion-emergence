# 3 — Cold review of the settled v2 (separate-queue) spec — verbatim record

## Provenance
- **Reviewer:** cold subagent, `general-purpose`, model **opus** (the single cold review Roy
  requested on the settled architecture). Full charter (five lenses + discipline + firing
  position/concurrency lenses + CH8/CH9/CH10 + CH11/CH12 ratification audit), quoted verbatim in the
  prompt, with rounds 1–3 records carried as context.
- **Context (closed set):** the three v2 artifacts + rounds-1/2/3 records + priority-ordered source.
  Reviewer-reported sha256: 1-spec `df1231ec963825674501179ee31b631fcc940d035100ccfb86cbb25f9cc74d4b`;
  1.5-criteria `005333f2dc3230f607fb2977f929c45db8bdf0a161089773cc045a29ab51658a`; 2-plan
  `3ea118e6bcd0546d13e3cd159351cb3f292e32d5582add8626f1607284c4158d`. (Note: these hashes predate the
  round-4 fixes below, which were applied after the review.)

## Verbatim reviewer output (key content)

**Bottom line: MAJOR → the architecture is fundamentally SOUND and does NOT need a respec; resolve
these before build.** Focus verdicts: (1) **reject-fate axis genuinely DISSOLVED** — no path writes a
candidate to `memories.db` before/during the gate; the only candidate→DB write is PROMOTE's
`store.create`; grief/forgetting/hebbian/graveyard operate only on DB rows. (2) Interior-continuity
sound with two minor gaps. (3) Consumer-migration INCOMPLETE. (4) **Queue concurrency SOUND** —
enqueue+drain share the `file_lock`; a candidate is never lost; `read_recent` lock-free tolerates a
partial line (`jsonl_reader.py:77`). (5) Merge/gate-lock sound except one uncovered case. (6) plumbing
feasible; `to_dict/from_dict→create` faithful in practice; `set_edge_weight` necessary (strengthen
ADDS, can't hit exactly 5.0). Ratification audit: R1/R2 honestly hedged ("relayed coordinator
message" ≠ transcript quote), no RAT2 inflation, routes correctly to owner sign-off.

Findings:
- **[MAJOR] reflex/journal straddle.** `engines/reflex.py:452` emits variable `arc.output_memory_type`;
  one configured value is `journal_entry` (`reflex.py:468`), a **deliberate** type consumed by
  `chat/prompt.py:1079` (weekly self-narrative from `list_by_type("journal_entry")`). The site-level
  "reflex → enqueue" rule would gate journal entries and strip them from that block. The
  automatic/deliberate partition can't be site-level.
- **[MINOR] ingest/commit.py:69 variable `item.label`** — per-type consumer reasoning has a blind spot
  for variable-typed sites.
- **[MINOR] recall.py:42 keep-sharp `store.get` bump no-ops** for a queue trace (id not a DB row) —
  harmless, unaddressed.
- **[MINOR] drain truncates before processing** — a crash mid-Pass loses the drained batch (differs
  from the soul precedent, which rewrites survivors last). Low impact (regenerable, stopgap).
- **[MINOR] merge-vs-concurrent-`fade` lost-update** — the KeyError guard covers LOSE/hard_delete, not
  `fade` (rewrites `content`→summary between the gate's read and update). Non-corrupting, unenumerated.
- **[MINOR] C17 over-defers the initiate double-send** (a build-resolvable bug) behind the feed
  product decision — split it. Initiate double-send is REAL (`initiate/memory.py:105` dedup can't see
  queued outbounds).
- **[ADVISORY] research.py:518/522 self-dedup** sees promoted-only → at most one wasted cycle.
- **[ADVISORY] to_dict lossy** but harmless (create recomputes peak from emotions; rest take defaults).
- CLEAN + earned: position/cache-prefix; queue concurrency; reject-fate dissolution; the
  ratification records.

## Author disposition (round-4 fixes applied — see decisions.md)
Architecture accepted (no respec). Fixes applied to the artifacts:
1. **MAJOR → type-based routing.** Write routing keys on `memory_type ∈ GATED_TYPES`, not call site
   (`route_write` helper); variable-typed sites (reflex, ingest) self-classify. `journal_entry`
   excluded → `prompt.py:1079` unaffected. `GATED_TYPES` membership is the scope knob for sign-off.
2. **Initiate double-send RESOLVED by default** (`initiate_outbound` excluded from `GATED_TYPES` →
   direct). C17 split: **C17a** (initiate dedup, gating, testable now) + **C17b** (feed product
   decision, stage-4).
3. Merge: re-read + **skip-if-fading/missing** before edit (closes the fade window) + KeyError guard.
4. Drain durability gap named-accepted for the stopgap (option to rewrite-survivors noted).
5. recall.py keep-sharp no-op named-accepted; research self-dedup advisory noted; C5 gains a
   `to_dict→from_dict→create` round-trip assertion; C10 wording corrected.

Route: architecture sound + round-4 fixes applied → **STOP at stage-4 → owner sign-off** (relay the
scope/`GATED_TYPES` + feed-behavior decisions). A confirmation review of the fixes was not separately
run (Roy scoped one review); stage-6 code review is the backstop.
