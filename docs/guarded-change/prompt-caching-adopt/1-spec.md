# 1 — Spec: adopt prompt-caching (Options A + A+) + cache instrumentation + replay harness

**Slug:** `prompt-caching-adopt`
**Date:** 2026-06-22
**Origin:** fork `ThinkerOfThoughts/companion-emergence`, commit `7baa145b`
("feat(caching): freeze system prompt + push volatile to stdin tail (Options A + A+)").
Validated by a local A/B replay on 2026-06-22 (see `0-baseline.md` notes).

## The problem

Every chat turn rebuilds the persona **system prompt** with ~19 volatile blocks baked in
(emotions, body energy, recall, felt-time, monologue/interior frame, attunement, …). Because
that content changes byte-for-byte every turn, the `--system-prompt-file` passed to the
`claude` CLI is never the same twice. Anthropic server-side **prompt caching keys on an exact
prefix match**, so the system+tools prefix is **re-created (`cache_creation`, billed ~1.25×)
every single turn** instead of **read (`cache_read`, ~0.1×)**.

This is a real, measured per-turn cost. Live `nell` `chat_usage.jsonl` shows **mean ~28.8K
`cache_creation_input_tokens` on every chat row** (min 21K / max 37K, 45 rows).

**Cost-attribution correction (stage-3 red-team finding, evidence: live log + byte count).**
That ~28.8K is NOT mostly the frozen head. The static block this change freezes — preamble +
`voice.md` (~16KB ≈ **~4.2K tokens**) + epistemic — is only ~15% of per-turn `cache_creation`.
The dominant terms are the **volatile blocks**, the **MCP tool-definition schemas** (injected by
the CLI subprocess), and **history**. So **Option A (freeze the head) alone recovers ~4.2K at
most**; the real lever is **Option A+** (moving the ~19 volatile blocks OUT of the cacheable
prefix to the stdin tail) plus whatever the opaque CLI tool-schema placement allows to cache.
The change is sound; the earlier "~25K = frozen part re-paid" framing was wrong and is corrected
here so the plan measures the right thing.

## Why this was missed for ~30 versions (correct the record)

The standing CLAUDE.md gotcha — *"The Claude Code CLI provider does NO cross-call prompt
caching"* — is **wrong**. The v0.0.31 P0 spike refuted only the *prompt-reorder* hypothesis
(you cannot win by reordering the conversation string toward a cache) and over-generalised it
to "there is no cache." There was a server-side prefix cache the whole time; we simply never
made the prefix byte-stable. We never measured `cache_creation`/`cache_read`, so the cost hid
in plain sight. **This change includes correcting that gotcha** (and the deferred-memory copy)
— see §Wiring / §Out-of-scope for sequencing.

## The change (three parts)

**Part 1 — Instrumentation (lowest risk, lift first).**
- `cache_debug.jsonl` behind `NELL_CACHE_DEBUG=1` (off by default): one row per chat call with
  `system_sha256`, char/token estimates, `volatile_present` — the live byte-stability probe.
- `scripts/cache_replay_workload.py`: a deterministic N-turn single-session replay against a
  scratch persona (real `claude` calls), emitting C1/C8/C9 metrics + a C7 reply side-by-side,
  with an A/B `--compare` mode. **This is the comparable replay workload the project's
  guarded-change config declares missing** — adopting it flips the cost/cache regression
  metrics from *advisory-only* to *gating*.

**Part 2 — Option A: freeze the system prompt.**
- Split `brain/chat/prompt.py::build_system_message` into:
  - `build_static_system_message` — preamble + `voice.md` + epistemic instruction; **no
    per-turn state** (byte-stable within a session).
  - `build_volatile_context` — all ~19 per-turn blocks.
- `build_system_message` itself is **left intact** and still used by the **image path** (the
  image fold can't carry a stdin suffix), keeping that path byte-identical.

**Part 3 — Option A+: push volatile to the stdin tail.**
- `engine.respond` threads the volatile chunk to the provider as a stdin **suffix** appended
  **after** history + the new user turn, via `run_tool_loop` chat_options → every
  `provider.chat` call (main loop, recruit-rerun, final forced pass).
- Relocate the block-level `Current time:` anchor out of the JSONL history top into the tail
  (per-message `ts` preserved). The reply frame stays genuinely last.
- Net effect: the **system+tools prefix** AND the **history prefix** both become cacheable
  (the per-turn-changing bytes all move to the very end).

## Constraints / invariants (must not break)

1. **Image path byte-identical.** Do not unify the image path onto the split; keep
   `build_system_message` for it (matches the v0.0.36 image-fallback design — there are no
   token deltas to stream, the path is blocking `chat()` → `_chat_with_images`).
2. **Voice fidelity + ambient use must not regress.** Moving emotional/body/recall state from
   the system prompt to the stdin tail must not flatten the persona or cause her to treat the
   ambient tail as the *task* instead of answering the user. (My 6-turn A/B C7 showed no
   regression — even richer — but that sample is thin: scratch persona, no tools, 6 turns.)
3. **Salience / tool-recruitment / reflection-debounce paths unchanged.** A+ threads a suffix;
   it must not alter which tools are recruited or the reflection gate.
4. **Fail-soft instrumentation.** A `cache_debug.jsonl` write failure must never affect a turn
   (the fork already wraps it in bare `except` — keep).
5. **No new user-facing surface.** Per the project's user-surface principle. `NELL_CACHE_DEBUG`
   is a dev env var; the replay script is a dev tool. Nothing in NellFace.

## Prior art / evidence

- Fork commit `7baa145b` (+1315/−34 across provider.py, engine.py, prompt.py, tool_loop.py,
  the replay script, and tests `test_prompt_caching_split.py` C1–C4 + engine test).
- Local A/B (2026-06-22, isolated worktree, scratch persona, 6 real `claude` turns/arm):
  **C1 PASS** (frozen system byte-stable, 1 distinct `system_sha256`); **C8 PASS direction**
  (`cache_creation` 29038→27086 = −7%; `cache_read` 50511→61229 = +21%); **C7 no voice
  regression.** Memory: `[[project-companion-emergence-cli-prompt-caching-real]]`.

## Known gaps carried into the plan (not hand-waved)

- **Magnitude unconfirmed on our realistic workload.** The fork's headline "~21% / ~4.2K fewer
  `cache_creation`/turn" did **not** replicate on scratch (I got −7% creation; the +21% I saw
  was on *read*). Cause: scratch persona + no seeded history → the frozen-block and A+
  history-prefix benefit barely exercised. The plan must measure with a **real 16KB `voice.md`
  + a real `--history-file`** under the 80-msg window.
- **Replay covers only the text path; the live tools path IS already logged (corrected).**
  Earlier framing ("the real path is unmeasured") was **wrong** — stage-3 red-team verified the
  live UI uses `_StreamingProxy.chat()` → `chat_stream`, which **does** log
  `cache_creation`/`cache_read` (`provider.py:888`); 43/45 live chat rows carry `num_turns>1`
  (real tool round-trips, logged with cache tokens). The genuine gap is narrower: the **replay
  A/B force-strips tools** (`cache_replay_workload.py` `_TextPathProvider`) to get usage rows, so
  the gating A/B exercises the text `chat()` path and **omits the MCP tool-schema term — the
  single largest cacheable mass.** The plan must therefore measure the dominant term via the
  **live streaming+tools `chat_usage.jsonl` rows before/after** (a real, already-instrumented
  signal), not lean on the tool-stripped replay. `_chat_with_mcp_tools` (the non-streaming tools
  path, which logs nothing) is NOT the live path — don't instrument it for this.
- **Load-bearing mechanism is unproven.** That a byte-stable `--system-prompt-file` lands at a
  stable Anthropic cache breakpoint *through the CLI's invocation shape* (tools resolve inside
  the subprocess; tool-schema position relative to the system file is opaque to this repo) is
  assumed, not demonstrated. The scratch A/B was weak (−7% creation; the +21% was on read). The
  realistic + tools measurement (criteria C4/C5) MUST land before any ship claim — the change
  could be directionally right yet recover far less than implied.
- **Rebase target.** The fork is on the public v0.0.38 base and lacks all kindled-link work.
  `7baa145b` will conflict in `prompt.py`/`engine.py`/`provider.py`/`tool_loop.py`. We
  **re-apply the design onto our `main`**, not cherry-pick the commit. Build is **held until
  the kindled-link Phase 7 session frees the main worktree** to avoid churn against a moving
  branch.

## Out of scope (this change)

- Adopting any other fork commit (the fork also carries our own v0.0.38 maker/file-write/notes
  — already ours).
- The CLAUDE.md/deferred-memory gotcha correction is **part of this change's wiring** but the
  repo edit is sequenced after the kindled session releases the tree (the memory copy is
  already corrected).
