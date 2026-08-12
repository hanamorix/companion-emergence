# P0 image-tool-route — Decisions log

## Run-start path validation (CFG3) — 2026-08-11
All reviewer-context paths validated readable at run start (worktree
`/home/zero/Desktop/companion-emergence/.claude/worktrees/agent-a3a3740a6bd46a3a9`):
- brain/, brain/bridge/provider.py, brain/chat/engine.py, brain/chat/tool_loop.py,
  brain/mcp_server/tools.py, brain/tools/impls/read_file.py, brain/bridge/server.py, brain/images.py,
  app/src/components/ChatPanel.tsx — OK.
- Stale-config (read-only) paths: ~/Downloads/Phoebe/chat_usage.jsonl,
  ~/Downloads/Phoebe/tool_invocations.log.jsonl, ~/companion-token-trace — OK.
- Config note override: `guarded-change.companion.md` redteam_context points at
  `~/Desktop/companion-emergence/brain`; per task instruction the reviewer is pointed at THIS worktree's
  `brain/` instead (same code, isolated phase). No dead paths.

## Run-resume path re-validation (CFG3) — 2026-08-12
Prior stage-3 red-team returned MAJOR but its findings were NOT captured on disk (no
`3-redteam-plan.md`, decisions.md held only stubs). Per resume instruction, re-running the
cold stage-3 red-team. Re-validated all reviewer-context paths readable in this worktree:
brain/{bridge/provider.py, chat/engine.py, chat/tool_loop.py, chat/tool_recruit.py,
mcp_server/tools.py, tools/impls/read_file.py, tools/schemas.py, bridge/server.py, images.py,
tunables.py}, app/src/components/ChatPanel.tsx — all OK. Spike harness ~/p0-image-spike/ and
spike plan present. redteam_context override: worktree `brain/` (not the config's
`~/Desktop/companion-emergence/brain`), same code, isolated phase.

## OWNER RULING (Roy) — 2026-08-12, relayed via coordinator/main (durable source: the coordinator message in this run's transcript)
Verbatim, on the open axes escalated at the gate-4 lap-2 stop:
1. **SCOPE = A (widen `/upload` NOW).** "Close #43 in THIS PR: widen `/upload` (server.py ~2150-2201) + the
   allowed-types set + add a non-image storage path so a text/PDF file reaches disk and is readable via the
   generalized read tool. Add the end-to-end #43 criterion (send a non-image file → stored → readable).
   server.py is a Hana-coordination surface — flag the footprint in the spec."
2. **IMAGE-SEEING = OPT-IN.** "Keep the opt-in design; C13 (organic uninstructed seeing works, its failure
   routes to the owner as a blocker) stays a GATING criterion. Deliberate — aligned with issue #124 (more
   Kindled agency over automatic actions)."
3. **Minor axes ratified:** "`image_max_bytes` default ≥ the /upload cap (20MB) — no dead zone; the
   pre-upgrade-buffer 'replays image-less' is an ACCEPTED one-time risk (name it in decisions.md). Your
   other plumbing/wording calls stand."
- **Mapping (RAT1):** axis 5 (#43 scope) → option **A**; axis 4 (seeing) → **opt-in, C13 gating**; axis 2
  (`image_max_bytes`) → **≥ 20MB**; migration → **accepted one-time risk (named here, this line)**. Each
  owner phrase disambiguates its flagged axis; no partial/adjacent resolution.
- **NAMED ACCEPTED RISK (Roy, per ruling item 3):** on upgrade, a session buffer that shared an image before
  the transport deletion replays image-less (image drops from later context). Accepted as a one-time
  pre-release cost.

## Gate log
- **Gate 4 (plan), lap 3 = FINAL confirmation red-team + route-to-build — 2026-08-12.** Owner ruled scope A;
  artifacts revised for scope A (steps 9-11: `brain/files.py` store + `/upload` widening + engine file-ref
  resolution; C8b/C16 gating). Final cold red-team (opus, fresh subagent `a709f806a20d8cc8e`), with the owner
  ruling supplied for the **ratification audit**. Result: **ratification VALID on every axis** (CH11/CH12 —
  #43=A, seeing=opt-in/C13-gating, image_max_bytes≥20MB, migration accepted-risk all map to the owner's
  verbatim words); scope-A steps feasible against source; C16/C8b/C13/C17 discriminating; factual/logical/
  position/concurrency lenses earned-clean (new `files.py` store adds no new RMW window — mirrors idempotent
  `save_image_bytes`). Two MAJOR + two ADVISORY:
  - **Finding 1 (MAJOR, PDF):** owner's verbatim "text/PDF … readable" vs. mechanism (PDF sniffs non-image →
    text path → UTF-8 decode fails → "binary — not shown"; `read_file.py:135-145`), so PDF is stored-not-
    readable. This is the #43-prose-vs-mechanism finding class for the **3rd time** (lap-1 F-5 → lap-2 text-
    claim → lap-3 PDF) → **iteration-cap human tie-break fires.** Disposition: reconciled the spec prose to
    C16's honest scope (text readable; PDF/binary stored + path-surfaced, NOT text-extracted this PR) and
    **escalated PDF-text-readability to the owner at the post-build gate** — NOT resolved by runner fiat.
  - **Finding 2 (MAJOR, path-traversal):** the new store derived on-disk ext from the client filename, no
    criterion gating traversal. **Fixed by design:** on-disk path is the validated 64-hex sha ONLY (no client
    filename/ext as a path component); filename kept as display metadata; added **gating C17** (traversal-
    proof). Dissolves Finding 4 (dedup-key wrinkle → pure-sha dedup).
  - **Finding 3 (ADV):** `upload_max_bytes` mislabeled as owner-ratified → relabeled under "plumbing calls
    stand" (runner's call). **Finding 4 (ADV):** dissolved by the sha-only path.
  **ROUTE = to build.** Runner fixes applied; the sole owner-facing residue (PDF-readability) is escalated to
  the post-build gate (the checkpoint the owner named), honoring the cap tie-break; criteria are internally
  clean + discriminating. The stage-6 code red-team provides the cold review of C17 + built code.
- **CRITERIA FROZEN (FRZ) 2026-08-12** at route-to-build. `1.5-criteria.md`
  sha256=`3f90cd52ddc92465dc5032ba46e8f7af828f6ec6e953deb0a611b141fb734846`. (spec
  sha256=`bc29385b7c1f673b825ec9e10aa477273e062f5f7c6fd9b40e5f24d11884d820`, plan
  sha256=`12f9f6479476e11317bfdd3c6070dc3a72ef121a4947fea2ae7684913a9c55e7`.) Stage 8 verifies the criteria
  file still matches this hash; any post-freeze edit needs a logged entry + targeted re-red-team.
- **Gate 4 (plan), lap 2 — 2026-08-12.** Re-ran stage-3 cold red-team (opus, fresh subagent
  `a2b082d90d127cdd0`) on the lap-1-revised artifacts, carrying lap-1 findings forward. Verdict: the four
  lap-1 findings (F-1, F-3/F-5/G2, F-4/G4, G5) all **genuinely addressed** with discriminating criteria and
  correct fix locations; factual/position/concurrency lenses earned clean; C12/C13/C14/C15 close their
  gaps; CH11/CH12 correctly do not fire. **One surviving MAJOR:** the retained "#43 closed for text files"
  claim is itself un-mechanized — `/upload` (`server.py:2150-2201`) hard-refuses every non-image at 415/422
  and stores images only, so no text/PDF file can reach disk; no build step widens `/upload` and no
  criterion exercises text end-to-end. Plus MINORs (C12 vacuous if `image_max_bytes ≤ 256KB`; upload-cap
  20MB vs read-cap dead zone; pre-existing-`image_shas`-buffer migration replays image as text).
  **Iteration-cap boundary:** #43-closure claim is now a 2nd bounce at gate 4 on the same finding-class
  (lap-1 F-5 → lap-2). Its resolution is a genuine **owner scope decision** (widen `/upload` to close #43 in
  this PR vs. defer #43, ship images-viewable + volatile-fix now) = the human tie-break the cap prescribes.
  **Disposition (runner, this lap):** removed the un-mechanized "#43 closed" claim entirely; rewrote the
  spec/plan to the **images-only default (option B)** and surfaced #43 as OWNER DECISION open-axis 5
  (A widen-now / B defer). Split E3→E3a (image send-path, IN SCOPE) / E3b (send-file relabel + non-image,
  CONTINGENT on A); C8→C8a (gating, image send-path) / C8b (advisory-contingent). Folded the MINORs:
  C12 precondition `image_max_bytes > 256KB` (and ≥20MB to avoid the dead zone); named migration
  accepted-risk in plan step 6. **Route:** STOP for the human — owner scope/sign-off gate (per run rules +
  cap tie-break); no build before sign-off. After the owner settles scope, a final confirmation red-team
  lap runs on the settled criteria before build. Criteria NOT frozen (freeze is on route-to-build).
- **Gate 4 (plan), lap 1 — 2026-08-12.** Stage-3 cold red-team (opus) worst severity = **MAJOR** →
  route to stage 2 (revise artifacts), then re-red-team. Record: `3-redteam-plan.md`. Three MAJOR:
  F-1 (image cap-ordering defeats `image_max_bytes`), F-3/F-5/G2 (organic image-seeing unmeasured),
  F-5 (#43 prose overreach). No blocker (premise spike-proven; factual/concurrency lenses clean).
  First bounce for these finding classes (iteration cap not tripped). No human stop required at this
  gate (no blocker, no missing criteria/config, no ratification). Disposition: F-1→plan step 1 reorder
  + new C12; F-3/F-5/G2→new C13 (gating, failure=blocker-to-owner) + spec caveats + open axes 4/5;
  F-4/G4→plan step 5 injection-point pin + new C14; G5→new C15; C1 pinned-model caveat→measurement
  table; G3→open axis 6 (out of scope). Criteria NOT yet frozen (freeze happens on route-to-build,
  which is gated behind owner sign-off per this run's rules).
