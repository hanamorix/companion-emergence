# 3 — Red-team of {1-spec, 1.5-criteria, 2-plan} (verbatim record) — ROUND 1

## Provenance
- **Reviewer:** cold subagent, agent type `general-purpose`, model **opus** (chosen: adversarial
  design review against core memory-store invariants — recall seam, concurrency, migration —
  where a subtle miss defeats the change's purpose; strongest reasoning warranted).
- **Charter given (verbatim):** the red-team charter core from
  `~/.claude/skills/guarded-change/stages/charter.md` (five lenses + discipline + the conditional
  position-sensitivity and shared-state concurrency lenses, both declared as firing) PLUS the
  stage-3 additions CH8 (coverage challenge, required section), CH9/CH10 (label audit, earned
  per-criterion). Quoted in full in the reviewer prompt; magnitude/threshold deferral declared
  out-of-scope for findings.
- **Context (closed set):** the three stage artifacts (1-spec, 1.5-criteria, 2-plan) + priority-
  ordered source under the worktree `brain/` + the config's optional Phoebe/token-trace paths
  (declared optional — no telemetry replay workload for this behavioral change). No supplementary
  author-authored context.
- **Reviewer-reported sha256 (context files):**
  - `1-spec.md` `3f3ac332b977f6a5e9bc5aa5306f3b767ec483054c843ceb95cdfce4bea91e44`
  - `1.5-criteria.md` `7dab873cad6ccf06c2f57ba7fa222fc6aca7aa259e38a41051c001914781f011`
  - `2-plan.md` `76e30276434082dcb76eec60f5473bd07b77c931b56cff57c2b3d0077080643c`
  - `brain/memory/store.py` `d627605961e96b4a10f0ba69d1c1d25833dd741aa1db68a812f619a54e8e71b4`
  - `brain/chat/prompt.py` `d904f423cb39df55bc6fb529df4de6963144fe650c4dab2090358bfc499e5916`
  - `brain/forgetting/recall.py` `b096b9f417b181417a3fc6a337dc84abe500a3f75a26f8f9a5bbc441c5396021`
  - `brain/tools/impls/search_memories.py` `9b31524bed57383b78f20db08af3498b40354c23de58589d5ec189f9b819adfa`
  - `brain/monologue/ambient.py` `74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e`
  - `brain/engines/heartbeat.py` `248cf2e63b6709ee1f5f8a107316c80cc0eee75ec7023f15fd465a7fcdae37f5`
  - `brain/memory/hebbian.py` `50776e0063bfbb5a565525592055880c3a326ab804a1b1adddcc0444671d2079`
  - `brain/chat/compaction.py` `cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb`
  - `brain/monologue/trace.py` `02b0bbeaf3144356105a834fffc39a7ad7b7040af95788bd367a479cf7a9a66a`
  - `brain/forgetting/graveyard.py` `40756ecff061610a4563bb0f57d97c945c6f19f48ee3a860db67379c87e25e00`
  - `brain/bridge/server.py` `f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e`

## Verbatim reviewer output

Worst severity: **MAJOR** (route: replan). Findings:

- **F-1 [MAJOR] salience floor mischaracterized + destructive.** `SALIENCE_FLOOR=0.0` with
  `importance <= 0.0` drop-via-`hard_delete`: but `importance==0.0` is the `create_new` default
  (`store.py:126`), and content-bearing writers `reflex.py:450-462` and `heartbeat.py:931-943`
  produce rows at exactly `0.0`. The built mechanism would hard-delete every reflex output and
  heartbeat reflection each idle tick. (monologue_trace escapes: `trace.py` sets importance 0.3.)
- **A-1 [MAJOR] "single-threaded heartbeat tick" is FALSE; overlapping ticks double-mutate committed
  rows.** `run_tick` is invoked from ≥3 unsynchronized threads each with its own MemoryStore
  connection on the same DB: background supervisor (`supervisor.py:1051`, ~15 min), session-close
  daemon worker (`server.py:626-651`, its docstring: "opens its own per-call stores inside the
  worker thread"), CLI. Only close-vs-close is debounced (`server.py:611`). A background tick and a
  close tick overlap → two gate runs with overlapping snapshots → both merge X→Y: double
  `store.update(Y)` (content drift the "surgical" rule exists to prevent) + second `hard_delete(X)`
  → KeyError (swallowed by fault-isolation, but Y already re-mutated). The plan's guard ("per-id on
  read ID set") defends the *append-during-window* accessor, NOT overlapping gate runs. Accessor
  table row 2 (`2-plan.md:142-147`) is factually wrong.
- **A-2 [MAJOR] gate-drops via grief graveyard resurface as "lost" → grief.** Dropped candidates →
  `graveyard.append`; recall (`prompt.py:842-878`, `seen_lost`→`handle_recall_touch`) surfaces
  graveyard hits as a *lost* bucket driving grief touches. A gate-dropped exact-dup has identical
  content to its surviving twin, so `graveyard.search` can surface "you lost this memory" grief for
  content still committed. No criterion observes this.
- **F-2 [MINOR] fidelity partial-proxy on weight-5 seeding.** `ensure_edge` (`hebbian.py:162-181`)
  inserts weight 5 ONLY if no edge exists. If a pre-gate linker (`ingest/commit.py:81`, candidates
  stay visible to it) already made a ~0.5 edge for the same pair, `ensure_edge` no-ops and the
  "strong association" intent isn't applied. C8's fresh pair can't catch this.
- **L-1 [MINOR] graveyard.append is not free plumbing** — requires salience_at_drop, SalienceInputs,
  lived_age_hours the gate must synthesize.
- **CH8-1 [ADVISORY→MAJOR if traces vanish] the gate deletes/merges the very monologue_trace rows
  interior-continuity reads.** monologue_trace is candidate (`trace.py:31`); Pass-1 exact-drop /
  Pass-2 drop `hard_delete`s them; `ambient.py:34` reads recent traces. C4 proves only the read
  filter doesn't hide candidate traces — nothing observes the gate PRUNING the source pool.
- **CH8-5 recall-counter churn** — the gate's `search_text` dedup scans bump `recall_count`/
  `last_accessed_at` on committed rows (`store.py:552`), perturbing later salience/forgetting.
- **A-3 [ADVISORY] within-session recall change** (conversation auto-ingest `ingest/commit.py:69`,
  extractor `extractor.py`) — owner-pinned intent, but no criterion names the affected consumers.
- **M-1 [ADVISORY] cheaper concurrency close:** atomic claim-update
  (`UPDATE ... SET state='processing' WHERE id IN (…) AND state='candidate'`, act on claimed rows)
  or a persona `file_lock` (already used in `graveyard.py`).

**CLEAN, earned:** the spec's crux (separate `consolidation_state` column, single choke-point at
`create()`, recall-seam via `committed_only`, cache-prefix isolation) is sound and accurately
grounded — spot-checked: `create()` INSERT omits state (`store.py:314-341`), sole
`INSERT INTO memories`=`store.py:316`, `recall.py:48-49` partition, `search_memories.py:94`,
`ambient.py:34` uses `list_by_type`, `compaction.py:376-381`, `hebbian.py:29/162-181`. Position
lens (b) cache-prefix isolation CLEAN with evidence (recall only from `build_system_message:192`,
static prefix is `build_static_system_message:318`). Label audit: C7/C8 stub-classifier is a
LEGITIMATE mechanism-check (not a dodge — governed path is the dispatch on the verdict); C16 & the
gate-log advisory labels are legitimate. C10 is a *partial proxy* — tests append-during-window, not
the real overlapping-tick race (A-1).

**Citation spot-check:** all artifact code citations verified TRUE; the two plan CLAIMS
"single-threaded tick" and "SALIENCE_FLOOR=0.0 = only zero-salience filler" verified **FALSE**.

## Author disposition (see decisions.md gate-3 entry)
Round-1 verdict MAJOR → replan (stage 2). Fixes adopted in the v2 artifacts, re-reviewed in
`3-redteam-plan-v2.md`.
