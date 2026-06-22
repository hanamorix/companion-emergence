# decisions.md — gate log (append-only)

Change: `prompt-caching-adopt` (fork `7baa145b`, Options A + A+ + cache instrumentation + replay harness)

---

**Stage 4 (plan gate) — lap 1 — 2026-06-22**
- Worst severity: **MAJOR** (no blockers).
- Stage-3 cold red-team (`general-purpose`, no shared context; read fork diff `7baa145b`, real
  `brain/` code, live `chat_usage.jsonl` 45 chat rows, the config). Verdict verified: direction
  sound (caching IS active — 45/45 live rows nonzero create+read; CLAUDE.md "no caching" gotcha
  factually wrong), partition clean, both-branch suffix append + image-path preservation confirmed.
- Three MAJOR/material findings, all cited + folded into spec+plan+criteria (lap-1 revision):
  1. **C5 framing wrong** — live path is `chat_stream` which logs cache tokens (`provider.py:888`;
     43/45 rows `num_turns>1`). Tools-path cache is already observable; the gap is the *replay*
     stripping tools. → reframed C5, added **C2-live** as the gating measure of the dominant
     tool-schema term; dropped the planned `_chat_with_mcp_tools` instrumentation (wrong path).
  2. **Cost attribution overstated** — frozen head ≈4.2K tok, not ~25K (mean create 28.8K is
     mostly volatile + tool schemas + history). → spec §problem corrected; A+ is the real lever.
  3. **Mechanism unproven** — frozen file → stable CLI cache breakpoint is assumed; replay can't
     see the tool-schema term. → C4/C5 (realistic + live tools) made hard pre-ship gates.
  - Minors folded: C1 annotated for the autonomous `voice_reflection` writer (replay gates,
    production self-corrects); P3 task-9 assert via mock not `cache_debug` (final pass has no
    `persona_dir`); C7-auto must pin the `now` seam.
- **Route: MAJOR → return to stage 2 (re-plan).** Done in place (findings were cited corrections,
  not contested calls — no human tie to break). Spec direction unchanged (not a blocker → no
  stage-1 restart).
- **Owed before build:** a lap-2 stage-3 confirmation pass (lightweight — verify the three
  corrections landed, no new majors) per the loop. Cheap; the reviewer just confirms.
- **Build status: HELD** — pending the kindled-link Phase 7 session merging to `main` (P2/P3 edit
  `prompt.py`/`engine.py`/`provider.py`/`tool_loop.py`; rebase onto post-Phase-7 main, apply once).
- Human override: none.
- Iteration-cap state: stage-4 bounce count for finding-class {plan/measurement} = 1.
