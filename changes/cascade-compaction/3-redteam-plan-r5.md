# 3-redteam-plan-r5.md — Cascade Compaction, stage-3 plan red-team ROUND 5

Cold, independent review. No shared context with the author beyond the artifacts and source cited below.
Model: sonnet (per owner directive, decisions.md round-1 gate-4 entry). This is the review requested
post-owner-tie-break: verify the artifacts CONFORM to the pinned design, and hunt specifically for new
edge cases the terminal-tier-3 ruling could have introduced.

---

## Provenance

Reviewed 2026-08-13 (UTC, per `date -u` at review time: Thu Aug 13 19:06:54 UTC 2026). Repo:
`/home/zero/Desktop/companion-emergence/.claude/worktrees/cascade-compaction`, branch
`ThinkerOfThoughts/cascade-compaction`, HEAD `cd29bc611a2d00aabe12c56a2dc47a5eeeaa914e`.

Artifacts (sha256):
```
71a4478c381c130a99459cc278b79bbf542a27f772aab738cef5247878f52e22  changes/cascade-compaction/1-spec.md
ccbf90b81dd24d7e7b11a0ddac73a760f811fc3b33c96ae44be11e12369a962a  changes/cascade-compaction/1.5-criteria.md
fb4b167dcaabf3e535b4943a46de64a77ca8eb9087b0dc9f54cae457bbd00efd  changes/cascade-compaction/2-plan.md
f9c629d2e413c450afd0fdc911a69050974747d95541ac1f3a861f5a43908210  changes/cascade-compaction/decisions.md
```

Source files read in full (sha256):
```
cc9b6e22ac3cf05aa1109e84abb5d8217619153d85cf172ae1b84644b1aad7fb  brain/chat/compaction.py
773f1e0b0ae3dad2cbdc4f316e16e65194d09ab047665ff740259eace1a4dc34  brain/chat/session.py
f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e  brain/bridge/server.py
cefb079963884fbafea3a0d8125c74bdc3a9e889894731329f482f49f93da56b  brain/ingest/buffer.py
cfe8b63b3d642dabe998f52b52a087ccc4c0acbd8bab5b32bba177f0b309d331  brain/ingest/pipeline.py (§140-240, §300-630 read directly; rest scanned via grep)
ca6eeba8070959cf502e76177a3a635832ecc0377e12e43773b3cf9629c116b2  brain/bridge/supervisor.py (§600-670, §1610-1700 read directly; rest scanned via grep)
d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260  brain/chat/engine.py (§220-449 read directly)
b71ed34d8f0d50108739fad682a3698989cfb5bd1278f0e4a3c7a011d873c7ca  brain/chat/budget.py
7d32a50b29f47b34645bd48c04b6cae3bb62e2dd15e031712b2b752fd59e1d77  brain/chat/compaction_migration.py
74dd5ba03614872c430fd2f3d2e40f23d4f70349a69816309ea0a4e7b8ee054e  brain/monologue/ambient.py
ddfbec5d20d3b4655afe7c14ce0c2d24564f09fa3e8f78f79d6560eb67051d90  brain/health/attempt_heal.py (§240-271 read directly)
```
Also grepped repo-wide for all `get_or_hydrate_session(` call sites, all `in_flight_locks` accessors, all
`remove_session(` call sites, and searched spec/plan/criteria text for `/state`, `apply_budget`,
`_fold_into_section`, "two"/"multiple"/"concat" to check for gaps.

Design authority consulted: `decisions.md` OWNER DESIGN RULINGS block (2026-08-13) only — did not re-read
`~/.claude/plans/memory-dream-rework-p1-cascade-brief.md` or `New_mem_system.md` directly (out of scope per
charter: "the design is owner-pinned... verify the artifacts CONFORM to it," and decisions.md already
records the ruling verbatim, which is what conformance is checked against).

---

## Carried-forward findings (round 4 → round 5) — resolution table

| # | Round-4 item | Resolved in artifacts? | Evidence |
|---|---|---|---|
| 1 | Owner design conformance — TERMINAL tier3, oldest-edge graduation, human labels in plan §1.3/spec §2b/§2a table | **YES** | spec §2a table row "tier3 — TERMINAL"; §2b "material graduates raw→tier1→tier2→tier3... then STAYS in tier 3"; plan §1.3 step 4 explicit terminal paragraph; labels "yesterday"/"day before yesterday"/"a few days ago" in spec §2d, plan §1.2, criteria C5 |
| 2 | C14 asserts graduation-then-persistence, non-tautological, long-inactivity-only-tier3 | **YES, with a new gap found — see F1 below** | criteria C14 text + plan §1.3 "Graduation + terminal persistence" paragraph both correctly state the terminal assertion, marker-preserving fake provider, non-`covers_until_ts` basis. **However**: neither specifies that the fixture injects *continuous* new raw material across passes, which is exactly the condition under which the NEW edge case (F1) below arises. Flagged, not fully resolved. |
| 3 | Tier3 hard cap `_SECTION_72H_CHAR_CAP = 0.20 × _SECTION_24H_CHAR_CAP`, enforced like tier1 | **YES** | plan §0 Q8 second paragraph: "`_SECTION_72H_CHAR_CAP = 0.20 × _SECTION_24H_CHAR_CAP` (= 2 400 chars...) enforced the same way (sentence-boundary truncation + validator re-request)"; criteria C3 asserts both caps. Minor arithmetic inconsistency noted below (N1). |
| 4 | Labels static, byte-stable, owner strings | **YES** | plan §1.2: "a static tier label (NOT a computed date)"; criteria C5 asserts exact strings |
| 5 | G1 (Major) — `/sessions/close` cleanup uses resolved sid at `server.py:2835-2836` | **YES, design-level; see F2 for a sibling gap the fix doesn't cover** | plan §0 "Successor pointer" section + §5 table row explicitly fixes 2835/2836 to resolved `sess.session_id`; criteria C20. Verified current (pre-change) code at those exact lines uses raw `req.session_id` — `server.py:2835` `remove_session(req.session_id)`, `:2836` `s.in_flight_locks.pop(req.session_id, None)` — confirming G1 is real pre-change and the plan's fix targets the right lines. **But** the same class of bug exists at a 5th, unenumerated call site (`/state/{session_id}`, `server.py:1255-1272`) that no criterion touches — see F2. |
| 6 | M1 — `supervisor.py:1686` finalize `remove_session` enumerated as harmless | **YES** | plan §5 table, dedicated row; verified `supervisor.py:1686` `remove_session(r.session_id)` under extraction-only finalize — matches |
| 7 | M2 — same-tick ordering cascade-then-rollover | **PARTIALLY — mechanism gap, see F3 (minor)** | plan §2.2 states the ordering explicitly, but see F3 for an unstated assumption about lock continuity |
| 8 | Round-3 carried (F1 4-site redirect, F2 lock-key, atomic write C17, C11 crash-durability, C18 seed cursor) | **YES, still holds** | plan §0/§8 unchanged from round 4 on these; re-verified server.py line numbers (2365/2424/2696/2753) match spec/plan claims exactly |

**Bottom line on carried findings: 5 of 7 substantive items cleanly resolved; 2 (C14/#2, G1's sibling/#5) surface
a genuinely new gap under closer scrutiny of the terminal ruling — reported below as F1/F2, not re-opening the
original carried finding (which IS fixed as scoped).**

---

## Per-lens findings

### Lens 1 — Factual (claims vs. code)

**F2 (Major).** Spec §2e, plan §0/§5/§8, and criteria C16/C19/C20 all assert, repeatedly and in bold/caps,
that `get_or_hydrate_session` has **"FOUR"** call sites needing the resolved-sid fix: `/chat` (`server.py:2365`),
`/stream` (`:2424`), `/sessions/snapshot` (`:2696`), `/sessions/close` (`:2753`). This is factually incomplete.
`grep -n "get_or_hydrate_session(" brain/bridge/server.py` (excluding the import line) returns **five** matches:
the four listed, plus `server.py:1261` inside `GET /state/{session_id}` (`server.py:1255-1272`):

```python
@app.get("/state/{session_id}", dependencies=[Depends(require_http_auth)])
def state_endpoint(session_id: str) -> dict[str, Any]:
    ...
    sess = get_or_hydrate_session(s.persona_dir, s.persona, session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    in_flight = session_id in s.in_flight_locks and s.in_flight_locks[session_id].locked()
    return {"session_id": sess.session_id, ..., "in_flight": in_flight}
```

Consequences once the redirect lands in `get_or_hydrate_session` (per plan §0) and `in_flight_locks` is
re-keyed by resolved sid at "the four handlers" (C19) but NOT here:
1. `/state/{old_sid}` will resolve via the (correctly fixed) shared function to the successor's
   `SessionState`, so `sess.session_id` in the response body becomes the **successor's** id — silently
   different from the id the client asked about, and inconsistent with `/chat`'s explicit contract of
   echoing back `req.session_id` unchanged (`server.py:2398`, `"session_id": req.session_id`) that plan §0
   holds up as the no-client-contract-change guarantee ("client keeps its sid... transparent").
2. `in_flight` (line 1264) is computed by looking up the **raw path-param `session_id`** in
   `s.in_flight_locks` — the old sid. Once C19 lands, nothing is ever keyed by the old sid anymore (all
   four fixed handlers key by the *resolved* sid), so `session_id in s.in_flight_locks` is permanently
   `False` for any client still holding the old sid — the exact population trigger B targets
   (continuously-attached clients, spec §2e). `/state/{old_sid}` will report `in_flight: false` forever,
   even while the successor session is actively mid-turn.

This is not a hypothetical proxy concern: `/state/{sid}` is a real, tested, auth-gated production endpoint
(`tests/bridge/test_endpoints.py:86-105,151,212,679`), and criteria C20's OWN oracle uses
`/state/{successor}` as its verification instrument ("`/state/{successor}` reports it not-live") without
ever noticing that the same function has an unpatched 404-adjacent sibling at the *old*-sid path. No
criterion (C16/C19/C20 or any other) exercises `/state/{old_sid}` after a swap. Severity: **Major** — this
sits squarely in the concurrency lens's own mandate (enumerate every `in_flight_locks` accessor; plan §5's
own accessor table is the thing that's incomplete), the "ALL FOUR handlers" claim is asserted as an
exhaustive completeness guarantee in three separate artifacts, and the failure mode (permanently wrong
status readout for the exact client population the whole redirect mechanism exists to serve transparently)
is a real, live-observable regression, not a proxy/theoretical one.

**N1 (Nitpick).** Plan §0 Q8 contains two mutually stale numbers for tier-3's worst-case size in the same
subsection. First paragraph: "Worst-case full head then ≈ 12 000 (24h) + 4 800 (48h = 40%) + **960** (72h =
20%) ≈ 17.8 k chars" — this computes tier-3 as 20% of the (already-capped) tier-2 figure (4800 × 0.20 = 960),
i.e. a *cascaded-fraction* proxy. The very next paragraph (the owner's actual ruling) defines the real cap
as `_SECTION_72H_CHAR_CAP = 0.20 × _SECTION_24H_CHAR_CAP` = 0.20 × 12000 = **2400** (not derived from tier2 at
all), and a third paragraph recomputes the worst-case total correctly as "≈ 19.2k chars" using 2400. The
first paragraph's "17.8k" / "960" figures are superseded but left in the text, so a reader skimming only the
opening sentence carries away the wrong number. Cosmetic — the enforced constant (2400) and the final total
(19.2k, well inside the 80k budget) are both correct; only the intermediate throwaway estimate is stale.

**No other factual claim checked against source was found inaccurate.** Spot-checked and confirmed accurate:
all four handler line numbers (2365/2424/2696/2753); `in_flight_locks` decl at `server.py:816`; G1's exact
lines (2835/2836) and their current raw-`req.session_id` usage; `_ATTACH_MAX_AGE_HOURS=24.0` at
`server.py:1224`; `/sessions/active` body at 1203-1253; `finalize_stale_sessions` empty-branch delete at
`pipeline.py:150-154` (session-level) and `:556-560` (the module's own finalize, confirmed both branches
currently delete, matching spec's claim that BOTH need the delete removed); `_run_compaction_tick` at
`supervisor.py:1619-1646` calling `compact_conversation(..., older_than=timedelta(hours=24),
fold_existing_summary=True)` exactly as spec §6 states; finalize `remove_session(r.session_id)` at
`supervisor.py:1686`; `rewrite_session_atomic` fsync-then-replace at `buffer.py:265-281`;
`append_archive` fsync + byte-count return at `buffer.py:290-307`; `compaction.py`'s `.strip()`-only store at
`:338` and summary-row shape (no `sections` key) at `:364-374`; `attempt_heal.py:250-266`'s posix-guarded
directory-fsync pattern (cited by plan §3 as the reuse target) exists as described; `ambient.py`'s
`monologue_trace` read is a `MemoryStore` call entirely separate from the compaction buffer, confirming C13
"preserved by construction."

### Lens 2 — Logical

**F1 (Major) — the terminal-tier design structurally requires a multi-prior-section fold every cycle under
continuous use, and neither the plan's own worked example nor any criterion accounts for it.**

Trace the section-age arithmetic the plan itself specifies (§1.3 steps 2-3: `bucket_of(sec)` = 24h if
`age(covers_from_ts) ≤ 48h`, else 48h if `≤72h`, else 72h) forward one full daily cycle under steady,
continuous daily use (new raw arriving every day — the population the owner's whole graduation design is
about, and the case the plan's own text partially walks through):

- Day N−1's tier1 section covered material that was (24h,48h] old *as of day N−1*. One day later (day N)
  its age is (48h,72h] → **bucket=48h**. This is the case the plan's prose explicitly walks through
  ("under steady daily use `[secs bucket==24h]` is empty — yesterday's 24h cohort graduated to the 48h
  band").
- Day N−1's **tier2** section covered material that was (48h,72h] old as of day N−1. One day later its age
  is (72h,96h] → **bucket=72h**. By the identical logic the plan applies to tier1, this section *also*
  graduates — into the 72h band.
- Day N−1's **tier3** section is already >72h old and, being terminal, only ever gets older →
  **bucket=72h again** (persists, per the owner ruling and plan §1.3's own explicit statement that "the
  prior tier-3 section... is included in `new_72` together with anything newly crossing 72h").

So on day N, `[secs bucket==72h]` = **{yesterday's tier2 section, yesterday's tier3 section}** — two
distinct prior section texts, not zero or one — *every single day*, from the first day the terminal state is
reached onward. This is not a boundary edge case or a rare gap-catch-up scenario; it is the steady-state
norm for exactly the tier the owner just pinned as "re-compacted forever." The plan's one worked example
(the tier1-band-empties-out case) creates a false impression of symmetry — it demonstrates that *incoming*
bands settle to a single fresh contributor under steady use, but the terminal band structurally does the
opposite: it *always* has two prior contributors once terminal, because nothing ever leaves it to make room.

Nothing in plan §1.3, §1.1 (representation), or the `_fold_into_section(inputs, fraction, cap=None)`
sub-op description states how two *prior section texts* (as opposed to one prior text + a raw-turn
transcript, which is what the existing `_FOLD_PROMPT`/`fold_existing_summary` machinery in
`compaction.py:107-140,325-334` is built for — a single `{prior_summary}` slot) get combined before or
during the fold call. `inputs` is generically named and plausibly *could* mean "render every item (prior
section texts and raw-turn batches alike) into one combined transcript, then run one fold call" — but the
plan never says so explicitly, and this is precisely the kind of representation-to-mechanism gap that round
3's F3 finding (position-sensitivity / age-laundering) turned out to hinge on. If an implementer instead
reuses `compact_conversation`'s existing single-`prior_summary` contract as-is and picks (or the code
happens to pick) only one of the two texts — most naturally the more `_FOLD_PROMPT`-idiomatic "existing
summary" (which section: whichever was read first, an unspecified tie-break) — the OTHER prior tier-3
content is silently dropped from that cycle's fold. Because the write is atomic (C17) this wouldn't corrupt
the row, but it would silently discard content the terminal-tier ruling explicitly promises never to drop
("stays in tier 3... re-compacted forever... not gone/archived" — decisions.md OWNER DESIGN RULINGS item 2,
verbatim: "the 'falls off the bottom → archived out' leg is REMOVED"). A silent content-loss bug in exactly
the mechanism the owner spent two tie-break rounds pinning would be a serious regression, and it would be
invisible to every test in §6 because of the coverage gap below.

**Coverage consequence (feeds CH8/CH9 below):** C14's oracle description — "sow an identifiable turn at a
known ts; run many consecutive daily passes; assert the marker's tier by pass number" — as worded, sows
*one* marked cohort and does not state that fresh raw material is also injected on each subsequent pass. If
implemented literally as worded (single sow, then N empty passes), `G24/G48/G72` are empty on every pass
after the first, no *second* section is ever reclassified into the 72h band alongside the persisting one,
and the multi-section-fold path described above is never exercised — the fixture would pass cleanly even if
the underlying multi-input fold silently drops content. C3's oracle ("feed an over-cap batch and run many
cycles") has the identical ambiguity. Neither criterion, as worded, is guaranteed to inject *continuous*
new raw material every cycle, which is the condition that produces the two-prior-sections case.

**No other logical defect found.** The G24/G48/G72 partition boundaries and the `bucket_of` classification
boundaries use consistent `≤`/half-open-interval conventions (both put the 48h and 72h boundaries on the
younger side), so no off-by-one contradiction between the two classification methods. The inactivity/catch-up
case (all raw >72h after a long gap → only tier3 populated) is handled by the same bucketing with no special
case, as claimed, and does not exhibit the multi-section issue (no tier1/tier2 sections exist to reclassify).

### Lens 3 — Missed opportunity

**M-1 (Minor) — no criterion pins down the multi-prior-section fold's semantics even as a documented
decision.** Independent of whether F1 is a live bug, the plan never *states* how N>1 prior texts are combined
(concatenate-then-single-fold vs. sequential pairwise fold vs. render-all-as-transcript). Even a one-line
design decision here (e.g., "`_fold_into_section` renders every input — prior section text(s) and raw-turn
batches alike — into one combined transcript via `_render_transcript`-equivalent, then makes exactly one
provider call") would have made F1 either a non-issue (if that's the actual intended mechanism, stated) or
would have made the gap in C14/C3's coverage obvious on inspection.

**M-2 (Minor) — `apply_budget`'s interaction with the sectioned representation is asserted but not tested.**
Plan §1.3 states the `apply_budget` backstop (`budget.py:42-91`) is rewired to call a "24h-only emergency
fold" sub-op (shared with the cascade, per §1.3's "Sub-ops reused by both..." list and the §5 accessor table
row "`cascade_conversation` / `_fold_into_section` (daily tick + backstop + migration)"), replacing its
current call to the legacy `compact_conversation` (confirmed at `budget.py:78-86`, unchanged in today's
code). Build order step 3 ("wire the daily tick + the 24h-only backstop") does schedule this as build work,
so it is not an unscheduled gap — but no criterion in §6 exercises the apply_budget-triggers-mid-turn path
against a sectioned summary row to confirm the rewiring actually happened and that a mid-conversation budget
trip doesn't regress the row back to the pre-change flat shape (which the *legacy*, still-present
`compact_conversation` would do if the rewiring were missed or partially done — it writes
`compaction: {covers_until_ts, folded, gen}` with no `sections` key, `compaction.py:364-374`, clobbering
`text` with a plain re-fold of the *whole* rendered 3-section string as one undifferentiated blob). Given
`apply_budget` fires on ANY turn whose assembled prompt exceeds 80k tokens — plausibly more often than once a
day for an active session — an unverified rewiring here could silently undermine the entire temporal-
structure feature for exactly the sessions long-lived enough to need it. Recommend an explicit criterion (or
an addition to C6/C1) that runs a cascade pass, then fires `apply_budget` over-cap, then re-parses the row and
asserts `sections` (all three) are still present and distinct.

### Lens 4 — Unstated assumptions & risks

**F3 (Minor) — same-tick lock continuity between cascade-fold and weekly-rollover-check is asserted, not
specified.** Plan §2.2 (M2 fix): "Order is cascade-fold-then-rollover under **the one compaction lock**...
Both hold the same per-session lock across the tick, so no interleaving." But `compact_conversation` (today's
code, `compaction.py:254-259,425-426`) acquires and releases the compaction lock **internally**, inside the
function, via a `try/finally`. If `cascade_conversation` follows the same self-contained
acquire-inside/release-inside pattern (a reasonable default, and the plan never states otherwise), then "the
one compaction lock... across the tick" is actually two separate acquire/release cycles (cascade's own, then
the rollover check's own), not one continuous hold. This does not appear to create an actual correctness bug
under the current architecture — the only other writer of the buffer, `ingest_turn`, is explicitly lock-free
by design (`compaction.py:404`, "ingest_turn takes no compaction lock by design — appends must stay fast"),
so nothing besides another fold/rollover call for the *same* session could race into the gap, and the daily
tick's per-session loop (`supervisor.py:1636`) is not shown to be concurrent with itself for one session. But
the plan asserts "no interleaving" as a settled fact when the mechanism that would make it strictly true (one
continuous lock hold vs. two back-to-back acquisitions) is unstated. Recommend the plan say explicitly
whether `cascade_conversation` exposes a lock-already-held variant for the rollover check to reuse, or
whether it is simply "two acquisitions of the same lock in immediate succession, which is safe here because
X" — spelling out X.

**No other unstated-assumption risk found beyond F1/F3 above and the accepted, already-documented ones**
(tier2 unbounded-under-gap — explicitly accepted by the owner in Q8; extraction-cursor race — explicitly
scoped out as pre-existing and not worsened, F4 from round 3, re-verified against `pipeline.py:317-318`
which does show an unguarded cursor read-then-later-write, confirming the plan's characterization).

### Lens 5 — Fidelity (owner mechanism vs. proxy)

Pinning each loaded term against the owner's 2026-08-13 ruling (decisions.md):

- **"Terminal tier 3"** — pins to: material graduates 1→2→3 then *persists*, re-compacted every cycle, no
  4th tier, no evict leg. Spec §2a/§2b, plan §1.3 step 4 state this correctly and explicitly reference "no
  evict-out-of-summary leg." **Mechanism matches the term** — verified against real code paths (the
  `bucket_of`/oldest-edge classification in §1.3 steps 2-3, cross-checked against the actual daily-tick call
  site at `supervisor.py:1619-1646` that will host it). The one place fidelity is *incompletely* verified is
  exactly F1 above: the term "re-compacted every cycle" is asserted for the terminal band, but the mechanism
  by which a terminal section absorbs a second, newly-arrived prior section (not just fresh raw) on every
  cycle is not pinned down, so I can confirm the *label* pins to the *design intent* but cannot fully confirm
  it pins to a *specified, testable mechanism* for the (structurally routine) two-prior-section case.
- **Human labels** ("yesterday" / "day before yesterday" / "a few days ago") — pins correctly: spec §2d, plan
  §1.2, criteria C5 all use the exact owner strings, explicitly called out as static (not computed dates) for
  byte-stability, matching the owner's ruling verbatim.
- **Tier-3 hard cap** — pins correctly to `_SECTION_72H_CHAR_CAP = 0.20 × _SECTION_24H_CHAR_CAP`: plan §0 Q8
  and criteria C3 both state the formula and "enforced like tier1" (sentence-boundary truncation + validator
  re-request), matching the owner ruling's "enforced like tier1" phrase exactly. (N1 above is a stale
  intermediate arithmetic figure elsewhere in the same subsection, not a mechanism mismatch.)
- **Oldest-edge graduation, no-remerge** — pins correctly: plan §1.3 step 3 explicitly classifies by
  `covers_from_ts` (oldest edge), not `covers_until_ts` (newest edge) — the round-3 F3 fix the owner
  endorsed keeping. Verified this is a genuine change from what the CURRENT code does (today's
  `compact_conversation` has no age-bucket classification at all — it's a flat single-fold — so there is no
  regression risk of accidentally reverting to newest-edge; the mechanism is being built fresh against the
  correct spec).

**No fidelity mismatch found beyond the F1 gap already covered under Logical.**

---

## Coverage challenge (CH8)

Behaviors the change could alter that no C1-C20/A1-A2 criterion observes:

1. **(= F1 above, Major)** The terminal tier-3 band absorbing *two* prior section texts (persisting tier3 +
   newly-graduated tier2) in the same cascade pass — the steady-state norm once terminal, not an edge case.
   No criterion's fixture is specified to inject continuous fresh raw material across passes, so this path
   may never execute during verification even though it will execute in production on pass 2 after terminal
   is first reached.
2. **(= F2 above, Major)** `/state/{old_sid}` post-swap: response `session_id` field switching to the
   successor (inconsistent with `/chat`'s echo-back contract) and `in_flight` permanently misreporting
   `false` once `in_flight_locks` is re-keyed by resolved sid everywhere except this fifth call site. No
   criterion drives `/state/{old_sid}` (only `/state/{successor}`, as an oracle for other fixes).
3. **(= M-2 above, Minor)** `apply_budget`'s mid-turn backstop path against a sectioned row — no criterion
   confirms the rewiring from legacy `compact_conversation` to the sectioned-aware fold sub-op actually
   happened and preserves `sections`.
4. **(Nitpick, informational)** The migration path's self-healing claim (plan §4, "next daily cascade tick
   re-establishes the full sectioned form" after a backlog-drain flattens a row) is asserted but not covered
   by C12's oracle, which only tests migration idempotency in isolation, not the migration→backlog-drain→
   cascade sequence. Low priority — the claim is plausible by construction (tolerant reader + cascade both
   independently re-derive `sections` from `text`/raw) and the plan explicitly flags this as a documented,
   accepted interaction rather than a gap it missed.

## Label audit (CH9/CH10)

Re-auditing whether each criterion exercises the real path it governs, focused per the charter on C14, C3,
C5, C16, C19, C20:

- **C14** — drives the real cascade pass with a marker-preserving fake provider (not a proxy on
  `covers_until_ts`); genuinely non-tautological. **Gap:** as worded, does not specify continuous raw
  injection across passes, so it may not exercise the multi-prior-section fold path (F1). Recommend the
  criterion text be tightened to require fresh raw turns sown on *every* pass, not just pass 0, so the
  terminal band's steady-state two-input case is forced to occur and be observed.
- **C3** — real over-cap fixture, "many cycles." Same gap as C14: doesn't specify continuous injection, so
  may not force the two-input tier-3 fold either. The caps themselves (length ≤ bound) would still be
  correctly checked even if a fold silently dropped one of two inputs (a dropped-content bug could
  paradoxically make the cap *easier* to satisfy, since less material means a shorter fold) — so C3 could
  pass even in the presence of F1, which is exactly why F1 needs its own explicit multi-input assertion, not
  reliance on C3/C14 as currently worded.
- **C5** — real fold, known ts, asserts coarse span + exact label strings. Exercises the real render path;
  no gap found.
- **C16** — genuinely drives the real four handlers via TestClient with the real `get_or_hydrate_session`
  chokepoint; correctly disqualifies the round-1 bare-`ingest_turn` proxy. **Gap:** doesn't enumerate the
  fifth call site (`/state`, F2) even though `/state/{successor}` is used as ITS OWN oracle helper elsewhere
  (C20) — the label "ALL FOUR handlers" is asserted as exhaustive but the codebase has five
  `get_or_hydrate_session` call sites.
- **C19** — real interleaving on both sids, asserts one lock key. Correctly scoped to the four handlers it
  names; same F2 gap by omission (the fifth, `/state`, reader of `in_flight_locks` is never keyed by anything
  under this fix, so it silently diverges from the four that are).
- **C20** — real TestClient POST to `/sessions/close` with the old sid, asserts successor's registry+lock
  freed and (via `/state/{successor}`) not-live. Genuinely non-proxy for what it tests. Uses `/state` only in
  its *successor*-id form, never noticing `/state` itself is unpatched for the *old*-id form (F2).

**A1 (regression metrics, advisory)** — reason given (no replay/held workload) is accurate and unchanged
since round 1; still correctly advisory, not gating.
**A2 (CI green)** — reasonable as stated; not challenged further this round (build-hygiene gate, not a
behavior criterion, and this round is a plan review — A2 is inherently unverifiable pre-build).

---

## Bottom line

Two Major findings, both new to this round and both squarely inside what round 5 was asked to hunt for
("scrutinize the mechanism once more for a NEW edge case under the terminal ruling"):

- **F1** — the terminal tier-3 band structurally requires folding *two* prior section texts (not one prior +
  raw) on every cycle once terminal is reached under continuous use; the plan never specifies the
  multi-input fold mechanism, and no criterion is guaranteed to exercise it, creating a real risk of silently
  dropping content the owner explicitly ruled must never be dropped.
- **F2** — `get_or_hydrate_session` has five call sites, not four; the fifth (`/state/{session_id}`) is
  unenumerated in spec/plan/criteria despite being a real, tested production endpoint, and will permanently
  misreport `in_flight` (and inconsistently echo `session_id`) for any client still on a redirected old sid
  once the four-handler fix and C19's lock re-keying land — with zero test coverage of this path even though
  the same endpoint is used (in its successor-id form) as another criterion's own oracle.

Both are grounded in direct citation to the plan text and to the actual current source (server.py line
numbers and compaction.py's existing fold-machinery contract), both would ship invisibly under the current
criteria set (C14/C3's fixtures don't force F1's path; no criterion drives F2's path at all), and both bear
directly on the terminal-tier promise the owner spent two tie-break rounds pinning down. Three Minor findings
(M-1, M-2, F3) and two Nitpicks (N1, CH8-4) round out the review; all carried-forward findings from round 4
are otherwise cleanly resolved (5 of 7 outright, 2 surfacing the new gaps above under closer scrutiny rather
than remaining open as originally scoped).

**Routing: Major → return to stage 2 (narrow).** Scope for the next plan revision: (a) specify the
multi-prior-section fold mechanism for the terminal band and strengthen C14/C3 to force continuous-injection
fixtures that exercise it; (b) add `/state/{session_id}` as a fifth site needing the resolved-sid treatment
(both the response body and the `in_flight` lookup) — or, if `/state` is deliberately meant to reflect the
*raw* id's literal state rather than redirect, state that as an explicit design decision and adjust C16/C19's
"ALL FOUR handlers" language so it stops reading as an exhaustiveness claim it doesn't back. This is a narrow,
two-item revision, not a re-open of the terminal-tier design itself — the owner's ruling is otherwise
faithfully and correctly implemented throughout.
