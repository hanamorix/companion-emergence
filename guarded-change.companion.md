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
      chat_usage.jsonl (call_type==chat): sum(cache_read_input_tokens) /
      sum(cache_creation_input_tokens). RATIO-OF-SUMS, not mean-of-ratios; rows with
      cache_creation==0 contribute to the sums only (no per-row division).
    direction: higher_is_better
    regression_threshold: "-10%"
    gating: true

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
