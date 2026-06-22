# 3 — Red-team of {1-spec, 1.5-criteria, 2-plan} (stage 3, lap 1)

Cold independent reviewer (`general-purpose`, no shared context). Sources consulted + cited:
fork diff `git show 7baa145b` (in object DB), real `brain/chat/{prompt,engine,tool_loop}.py` +
`brain/bridge/{provider,server}.py`, live `chat_usage.jsonl` (87 rows / 45 `call_type=="chat"`),
`guarded-change.companion.md`. Citations spot-verified by author (provider.py:888 streaming
`log_usage`; 43/45 live rows `num_turns>1`; static head ≈4.2K tok vs mean create 28.8K).

**Worst severity: MAJOR (no blockers).** Full findings below; routing + folded revisions in
`decisions.md` (stage-4, lap 1).

## Lens 1 — Factual
- **[minor]** Cost attribution overstated: frozen head ≈4.2K tok (~15% of mean create 28,790),
  not ~25K. Evidence: live log mean create 28,790 (min 21,046/max 37,126), even at `num_turns==1`
  = 27,422; static head = preamble + voice.md (16,280 B ≈ 4,070 tok) + epistemic. Option A alone
  recovers ~4.2K; A+ is the lever. → **folded** (spec §problem corrected).
- **[major]** C5 premise wrong: live UI path is `_StreamingProxy.chat()` (server.py:310) →
  `chat_stream` → `log_usage(call_type="chat")` (provider.py:888). 43/45 live rows `num_turns>1`
  = real tool round-trips, logged with cache tokens. `_chat_with_mcp_tools` (provider.py:1123, no
  log) is NOT the live path. The gap is the *replay* stripping tools (replay:240). → **folded**
  (C5 reframed, C2-live added).
- **[verified-clean]** create+read nonzero on all 45 chat rows (read mean 77,293). Gotcha is
  factually wrong. ✓
- **[verified-clean]** suffix appended in BOTH branches; static/volatile partition complete
  (all ~19 blocks in `build_volatile_context`; static uses only persona name + user_name +
  voice_md + constant epistemic); image path keeps `build_system_message`. ✓

## Lens 2 — Logical
- **[minor]** C1 "exactly 1 distinct system_sha256" too strict for production: `voice_reflection.py`
  can autonomously rewrite voice.md mid-session (voice.py:228) → legitimate 2nd hash, one extra
  create, self-correcting. Fine as a *replay* gate. → **folded** (C1 production caveat added).
- **[minor]** P3 task-9: final-forced-pass passes `options` without `persona_dir` →
  `_maybe_log_cache_debug` early-returns → no `cache_debug` row for it. Assert suffix via
  unit/mock, not the debug log. → **folded**.
- **[clean]** P1→P2→P3 ordering sound; "OLD C1 should FAIL (many hashes)" baseline hygiene good.

## Lens 3 — Missed opportunity
- **[minor]** Dominant cacheable mass = MCP tool schemas; the replay strips them, so the gating
  A/B structurally can't measure the dominant term. Read live streaming+tools rows (already log
  cache tokens) instead of leaning on the tool-stripped replay. → **folded** (C2-live now gates
  the dominant term; A1 demoted).
- **[minor]** C5 option (b) (system_sha256 byte-identity with tools on) measures the wrong thing
  (system file is tool-independent; trivially identical). → **folded** (option b dropped).

## Lens 4 — Unstated assumptions & risks
- **[major]** Load-bearing mechanism unproven: byte-stable `--system-prompt-file` → cacheable
  system+tools prefix *through the CLI's invocation shape* (tools resolve in-subprocess; schema
  position opaque). Scratch A/B weak (−7% create; +21% was read). → **folded** (C4/C5 hard
  pre-ship gates; mechanism flagged in spec §Known gaps).
- **[minor]** Position-shift behavioural risk under-tested (6-turn scratch, no tools/history). C7
  (human, ≥8 turns, ≥1 tools, seeded) is the right + only guard. Adequate, noted.
- **[minor]** C7-auto image byte-identity needs the `now` seam pinned (else `Current time:`
  nondeterministic). → **folded**.
- **[clean]** Build-hold/rebase-onto-post-Phase-7-main safe (Phase 7 touches none of the 4 files;
  `respond()` signature stable → P1 can land early). 80-msg window doesn't clip the suffix
  (threaded outside `apply_budget`); replay gap 3s < 5-min TTL. ✓

## Bottom line (reviewer)
Core premise factually sound; gotcha genuinely wrong. Fix before build: (1) C5 framing (live
streaming path already logs — gap is replay tool-stripping); (2) cost attribution (~4.2K not
25K); (3) prove the mechanism via realistic+tools measurement before any ship claim, since the
gating replay can't see the dominant tool-schema term. All three folded in lap-1 revision.
