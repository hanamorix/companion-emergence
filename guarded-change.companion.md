# guarded-change config — companion-emergence (Layer 2)

Per-project config for the `guarded-change` skill. Parameterizes the agnostic loop for
companion-emergence on **this** machine (macOS, dev repo, persona `nell`).
See `~/.claude/skills/guarded-change/METHODOLOGY.md` for the contract.

Adapted from the upstream `guarded-change.companion.md` (which targeted a Linux box +
persona "Phoebe"). Path changes here: XDG `~/.local/share/...` → macOS
`~/Library/Application Support/...`; persona `Phoebe` → `nell`; bundled Linux runtime →
the dev repo `brain/` (the authoritative source we edit + test here).

```yaml
project: companion-emergence

redteam_context:          # PRIORITY ORDER — read top-down; a cold subagent can't read the
                          # whole brain tree, so each entry says what to check there first.
  - path: "~/Library/Application Support/companion-emergence/personas/nell/chat_usage.jsonl"
    note: "Ground truth for cost/cache/num_turns claims. Check fields exist before trusting a metric (e.g. there is NO background/foreground marker on generate rows)."
  - path: "~/Library/Application Support/companion-emergence/personas/nell/tool_invocations.log.jsonl"
    note: "Ground truth for tool/file behavior. Note: only request_id groups rows; no session_id, no reply-boundary field — confirm before treating request_id as 'one reply'."
  - path: "the project root/brain"
    note: "The REAL code we edit + test (dev repo source of truth — uv run pytest runs against this). Start at bridge/provider.py and chat/{prompt,engine,tool_loop,salience,tool_recruit}.py for prompt/caching/tool claims."
  - path: "the project root/docs/superpowers/specs"
    note: "Design specs (version-control authoritative). Find the relevant version's design before trusting a behavior claim."

measurement:
  baseline:               # capture the CURRENT version's behavior before a change
    how: >
      Read the tail of chat_usage.jsonl + tool_invocations.log.jsonl for the current
      version and compute the per-message metrics below (mean + tail). Record the version
      string (brain.__version__). Read-only analysis of the app's own logs — no patching.
    output: "changes/<slug>/0-baseline.md"
  check:                  # measure the NEW build's behavior the same way, post-change
    how: >
      After running the new version through representative chat turns (incl. any file-tool
      use the change touches), recompute the same metrics from the fresh log rows and the
      new version string.
    output: "changes/<slug>/8-harness.md"

metrics:                  # standing regression metrics (source: the JSONL logs)
  # CURRENT CAPABILITY (read first): A COMPARABLE REPLAY WORKLOAD NOW EXISTS —
  # `scripts/cache_replay_workload.py` (landed 2026-06-22, prompt-caching-adopt P1). It fires a
  # deterministic N-turn single-session sequence against a scratch persona (real claude calls) and
  # emits per-`call_type=="chat"` cache_creation/cache_read with an OLD-vs-NEW `--compare` A/B. The
  # `gating: true` cache/cost metrics below ARE gating when measured via this replay (same seed both
  # arms isolates a change's own contribution — the false-regression guard the methodology requires).
  # The cache_debug.jsonl probe (NELL_CACHE_DEBUG=1) adds C1 system-prompt byte-stability.
  # STILL ADVISORY: deltas computed over "whatever live turns happened to run" (no fixed workload) —
  # those remain advisory; confirm against the replay A/B + conformance. The live streaming+tools
  # path also already logs cache tokens (provider.py chat_stream), so a comparable live turn-set is
  # a valid gating measure too (see C2-live in a change's 1.5-criteria). See the workload note under Notes.
  # --- GATING-WHEN-WORKLOAD-EXISTS: measurable per chat call in chat_usage.jsonl (call_type=="chat") ---
  - name: cost_per_chat_call_usd
    source: "chat_usage.jsonl: total_cost_usd where call_type==chat, mean"
    direction: lower_is_better
    regression_threshold: "+10%"
    gating: true
  - name: num_turns_per_chat_call
    source: "chat_usage.jsonl: num_turns where call_type==chat, mean and max(tail)"
    direction: lower_is_better
    regression_threshold: "+1 turn mean, or any new tail above prior max"
    gating: true
  - name: cache_creation_per_chat_call
    source: "chat_usage.jsonl: cache_creation_input_tokens where call_type==chat, mean"
    direction: lower_is_better
    regression_threshold: "+10%"
    gating: true
  - name: cache_read_ratio
    source: >
      Per-run cache-break check over chat_usage.jsonl rows (call_type==chat; one run = one slice
      of rows in log order, e.g. one replay). The run's first chat row (the cold write) is never
      judged. Only single-call rows (num_turns == 1) are judged. The first judged row establishes S;
      every later judged row's cache_read_input_tokens is compared with S, the minimum of the earlier
      nonzero judged reads. A judged row is a cache break if its read is 0 or below S (T below is 0:
      read * 100 < S * (100 - T)). The run needs at least 2 judged rows (one establishes S, one is
      compared against it), else the verdict is UNMEASURED. Multi-call rows (num_turns != 1, tool
      turns) are not judged and impose no rule: a row's usage is the SUM over the turn's API calls,
      so its read is not comparable, and a tool-using message's first call has the same cached
      prefix (system prompt and tool block) as a plain message's, the model decides on tools only
      after that call, so a prefix break shows on plain rows too (micro-acks are exempt from the
      monologue call only). Each run is measured against its own reads, so no OLD/NEW baseline,
      call count or logging change is needed and a change that shrinks the cached prefix cannot
      trip it. Run `uv run python scripts/cache_replay_workload.py --cache-break-check
      <chat_usage.jsonl> [--from-row N]`, or run a replay (same codes after the metrics JSON is
      written; the stderr line `cache_break_check: <STATUS>` and the `cache_break_check` key in the
      JSON carry the verdict; a replay sends a warm-up turn, one plain micro-ack, the content turns
      and one trailing plain micro-ack). Exit codes of a verdict: 0 PASS, 10 FAIL, 11 UNMEASURED
      (an unreadable file or config or a corrupt log line is UNMEASURED too). Other exits are never
      a verdict: 1 = a crash, `--turns` below 1 or `--dump-replies` without `--out`; 2 = argparse
      usage errors, no persona dir, and bad `--cache-break-check` arguments (`--from-row` below 1,
      not a file, `--regression-threshold` outside [0, 100)). A replay whose provider never logs
      usage (`--provider fake`, `ollama`) has no rows by design: status NOT_APPLICABLE, exit 0 and
      one stderr line saying so; a real provider with no usage rows is UNMEASURED. `--compare`
      returns 0 / 10 / 11 (C8 pass / fail / not comparable). The replay parses chat_usage.jsonl
      once, and only the bytes this run appended, so an older corrupt line never matters and a
      corrupt line inside the run's own rows makes it UNMEASURED. The result names the judged and
      multi-call rows; a FAIL names each break row with its read, the S it was compared against,
      share of S, ts and the gap to the previous row.
    # Not an OLD->NEW delta, unlike the entries above: "stable" means a judged row's read stays at
    # or above the run's own stable read S.
    direction: stable
    # Tolerance below S in percent (0 = none). Not a regression threshold against a baseline.
    regression_threshold: "0%"
    gating: true
    # Reworked in #339/#340 (three versions). The old sum(read)/sum(creation) ratio read a smaller cached prefix as a
    # break (#332: ratio -26%/-18%/-2% while cost fell 23-47%) and was made advisory in #333/#334.
    # The first per-run version (a later row re-writing within 10% of the first row's creation) was
    # shown by the #340 review to FAIL healthy runs: a healthy later row re-writes the volatile tail
    # and the new exchange, which can be as large as the first row's whole write. A real healthy
    # 8-turn replay (2026-10-06) FAILED it on 7 later rows. The second version judged single-call
    # reads against the highest earlier read, and the #340 re-review showed that one high read then
    # FAILed every later healthy row. This version compares READS only, against the minimum of the
    # earlier judged reads.
    # - Meaning of the fields: `direction: stable` and `regression_threshold: "0%"` mean "a judged
    #   row's read stays at or above the run's own stable read S, with 0% tolerance". They are not
    #   an OLD->NEW delta (the old `lower_is_better` / "-10%" no longer applies to this metric).
    # - Basis of S = the minimum of the earlier judged reads: the stable prefix (system prompt and
    #   tool block) is the floor that every healthy plain row reads at least; extra reads above it
    #   (for example cached history) are not the prefix and must not raise the reference. So a high
    #   read in the middle of a run, a history-window slide, compaction or an idle gap after the
    #   first judged row does not fail the later rows. Why the minimum and not a floor that needs two
    #   establishing rows: with two establishing rows a persistent break that starts on the second
    #   judged row would pass (reads 6449, 2000, 2000 give S = 2000 and the third read is not below
    #   it), while the minimum catches it at the second row. Every nonzero judged read enters the
    #   minimum, a zero read is a break and never enters it.
    # - Basis of T = 0 (two measurements, throwaway synthetic sandbox personas, claude CLI 2.1.219):
    #   healthy first-call reads did not vary at all. 2026-10-06, haiku, tools path, 8 turns plus
    #   warm-up: three direct single-call reads were 6449 and the first calls of three two-call rows
    #   (total read minus last-call read) were 6449 as well (these three are the inferred first calls
    #   counted below); two more first calls of three-call rows were only estimated from an assumed
    #   per-call read and two were not derivable. 2026-10-07, sonnet, text path (the replay default), 5 turns plus
    #   warm-up and 2 plain turns: both single-call reads were 8403 (the other five rows were 3-4
    #   call rows). That is 5 direct single-call reads and 3 inferred first calls with zero spread
    #   within each run (6449 on one model, 8403 on the other). (The 2026-10-06 log held two more
    #   chat rows than the nine replay turns, of unidentified origin; the three direct reads are
    #   its last three rows, and two of them are probably those extra rows, so only the first is
    #   certainly a replay turn.) The #340 review reports a constant
    #   15,476 read on real rows as a further datum. So zero wobble is OBSERVED, not bounded: the
    #   sample is two small runs on two models with synthetic sandbox personas, and a handful of
    #   agreeing reads cannot exclude an occasional variation. The consequence is stated: with T = 0 any plain row reading
    #   below the earlier minimum FAILs a gating run, a transient miss included. The FAIL names the
    #   row, read, S, share, ts and gap; if a healthy run ever fails by a small margin, re-measure
    #   and set T here (the check reads it from this entry).
    # - Basis of the minimum of 2 judged rows (derived, not tuned): a single judged row can only
    #   establish S; a comparison needs a second row.
    # - UNMEASURED (exit 11) = an unverified gating criterion: stop for the human, never a pass.
    #   Causes: fewer than 2 chat rows or fewer than 2 judged rows (e.g. most tools-path rows were
    #   multi-call); a judged-candidate row with no int read, a negative read, creation 0 and read 0
    #   (no cache activity at all), or on a different model than the run's (the model of the first
    #   usable single-call later row); a threshold outside [0, 100); an unreadable config; a log
    #   line the shared jsonl reader had to skip within the run's own rows (never judged from the
    #   remaining rows); a log file that got smaller during the run (rotated or truncated).
    # - Replay: after the warm-up the replay sends one plain micro-ack and, after the content
    #   turns, one more, so a break that begins during the content turns and persists is seen by a
    #   plain row (a transient miss that has recovered by the trailing row is not seen). The model
    #   still decides about tools, so these may come back multi-call (UNMEASURED: re-run).
    #   --no-warmup without --no-plain-turns makes the leading plain turn the never-judged first
    #   row, which leaves one judgeable row on the tools path (UNMEASURED): use the two flags
    #   together. Each arm of an A/B replay exits with its own verdict code (an OLD arm may
    #   legitimately exit 10 or 11; the metrics JSON is written first). The means and series of a
    #   replay cover all of its chat rows, warm-up and plain turns included (labelled
    #   `c8_cache.rows_include`; `last_turn_cache_creation` is the last content turn's row, from
    #   exact turn attribution, and `c8_cache.row_turns` lists the turn id of every row). Use the
    #   same --no-warmup / --no-plain-turns settings on both arms of a --compare: it prints
    #   "NOT COMPARABLE" (no deltas, no C8 verdict, exit 11) when the settings, `turns_requested` or
    #   `with_tools` differ, and an arm from an older script (no such keys) counts as having sent
    #   no warm-up and no plain turns.
    # - Limits: (1) A high read on the FIRST judged row followed by normal reads FAILs: reads
    #   15476, 6449, 6449 break at the second, because S is 15476 by then. This is the known limit
    #   of the minimum (the #340 review's exact rows, creation/read 17000/0, 9000/15476, 9000/6449,
    #   9000/6449, all single-call, FAIL at row 3). It has not been observed: both measurements saw
    #   zero spread in healthy single-call reads. The FAIL names the row, the read and S, so it can
    #   be told from a real break by looking at the reads. (2) A partial break on the first judged
    #   row lowers the floor, and a persistent partial break from the first judged row on (e.g. the
    #   system block churning behind an intact tools block) leaves a stable prefix that S then follows,
    #   so it is not caught here; PASS does not certify its absence, cache_creation_per_chat_call and
    #   cost_per_chat_call_usd still gate it. The first judged row is only checked for read > 0.
    #   (3) With exactly 2 judged rows (what a text-path replay yields: the leading and trailing plain
    #   turns) the check is one comparison of two reads with zero tolerance; the minimum differs from
    #   a highest-read reference only with 3 or more judged rows. The "not observed" claim above
    #   covers the measured modes (default replay, no history file); a --history-file replay is
    #   unmeasured. (4) With T > 0 small steps, each within T% of the current minimum, lower S step by
    #   step and pass; T is 0, so this does not apply today. (5) A slice must be ONE run: a prompt or
    #   tools change that shrinks the prefix, or an idle gap inside it, reads as a break. (6) num_turns == 1 meaning one API call
    #   is an assumption (measured n=1 rows had total == last-call numbers; the fixture's num_turns 4
    #   is not its 3 calls, so only == 1 is used). (7) A break confined to tool-using rows, or after
    #   the last plain row, is not seen directly (basis above). (8) The replay reads the rows this run
    #   appended and assumes nothing else writes the log meanwhile; a file smaller after the run than
    #   before is UNMEASURED, a rotation that regrows past the old size within the run is not
    #   detected. Why gate it: a baseline-free within-run detector of a read collapse,
    #   complementary to the creation and cost metrics.

  # --- BLOCKED: not measurable from current logs; needs stage-2 instrumentation ---
  # tool_calls_per_request and file_reread_per_request — BLOCKED. The grouping key `request_id`
  #   is stamped on only ~41% of tool rows: audit.py writes it only when NELL_MCP_AUDIT_REQUEST_ID
  #   is set, which provider.py sets only on the MCP-subprocess path. Computing either metric over
  #   that minority silently reports on a non-random subset — the SAME defect class as the original
  #   removed metric. Additionally, record_monologue bookkeeping rows inflate tool counts. To
  #   RESTORE these as real metrics, a stage-2 instrumentation change must: (1) stamp a correlation
  #   key (request_id or session_id) on ALL tool rows, and (2) tag tool-vs-bookkeeping rows so
  #   record_monologue can be excluded. Until both land, these are not metrics — record the gap.
  #
  # background_generate_per_msg — REMOVED. A `generate` row in chat_usage.jsonl carries NO field
  #   distinguishing background from foreground. To restore, a change must first add a call-origin
  #   field to the log (an instrumentation task per the methodology).
```

## Notes specific to this project

- **Acceptance criteria are per-change** (authored in `1.5-criteria.md`), not here. Example
  for a file-tool fix: "resolves a named folder + approximate filename in ≤2 tool calls; zero
  parent-directory traversal; zero within-`request_id` re-reads of the same file."
- **The two logs cannot be joined.** `chat_usage.jsonl` has `session_id` and **no**
  `request_id`; `tool_invocations.log.jsonl` has `request_id` and **no** `session_id`. There
  is no shared correlation key, so any "tool activity *per chat message*" metric is currently
  uncomputable. Combined with the ~41% `request_id` coverage gap, this is why both tool-log
  metrics are **BLOCKED**. The stage-2 fix that unblocks them — stamp a correlation key on
  **all** tool rows **and** tag tool-vs-bookkeeping rows — also closes this join gap.
- **`request_id` ≈ "one reply" is unverified.** No field in the tool log marks reply/burst
  boundaries. A second reason the tool-log metrics are BLOCKED rather than trusted.
- **Comparable workload required for gating regression.** The metrics above are aggregates;
  to gate (not just advise), baseline and check must run a **comparable set of chat turns**
  (ideally a fixed replay script), else a change that legitimately does more shows a false
  regression. Until a replay harness exists, treat deltas as advisory; confirm with conformance.
- **Running the replay A/B (lessons from #332).** `cache_replay_workload.py` defaults to the
  no-tools TEXT path; pass `--with-tools` to measure the tools-bearing chat prefix production
  uses (the tools path now logs usage rows too). Run OLD from a `git worktree` of the
  merge-base with its own `uv sync --extra dev`, not via `PYTHONPATH`. Use the default home so
  the brain's own `CLAUDE_CONFIG_DIR` applies (a scratch home loads the owner's global config,
  which biases the comparison). Record each arm's exact launch command. If Sonnet's safety
  classifier flags the scripted conversation (it did on both builds, 2026-09-30), run BOTH arms
  on Haiku via `get_provider(..., model_override="haiku")`: the token metrics hold across
  models; cost is then Haiku prices. The replay now also sends a fixed warm-up turn, one plain
  micro-ack and, after the content turns, one more (extra chat rows in the means and series; they
  give the cache_read_ratio check the single-call rows it judges): run both arms with the same
  settings, `--no-warmup --no-plain-turns` on both when the OLD tree's script predates them;
  `--compare` prints NOT COMPARABLE on stdout (no deltas, no C8 verdict, exit 11) when they differ.
- **These metrics exist because the v0.0.38 file-tool token-cost regression was only catchable
  via `tool_invocations.log.jsonl`.** Any change touching an un-instrumented area must add
  logging in stage 2 ("instrument before you build").
- **Reviewer independence matters extra here.** The stage-3/6 cold reviewer must prefer
  arguments from the JSONL data over arguments from reasoning; a clean factual verdict is
  invalid without source citations (see METHODOLOGY). This complements — does not replace —
  the existing `superpowers:requesting-code-review` two-stage pass.
- **changes/<slug>/ lives under `docs/guarded-change/`** in this repo (keeps the repo root
  clean and groups the per-change artefacts with the other design docs).
```
