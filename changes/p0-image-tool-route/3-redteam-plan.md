# P0 image-tool-route — Stage-3 red-team (plan) — VERBATIM RECORD

Re-run 2026-08-12 (the prior interrupted run's stage-3 findings were not captured on disk; this
record is authoritative). Cold, independent reviewer, no shared context with the author.

## Provenance
- **Reviewer:** `general-purpose` subagent, model **opus** (chosen: adversarial review of a
  position-sensitive prompt-assembly change with load-bearing caching/MCP-posture correctness claims;
  a missed finding here is expensive downstream).
- **Context (closed set):** the three stage artifacts (1-spec, 1.5-criteria, 2-plan) + the config
  `redteam_context` paths, pointed at THIS worktree's `brain/` (isolated-phase override, same code) +
  the spec's touched-files list + prior-art (spike plan `~/.claude/plans/memory-dream-rework-p0-image-tool-spike.md`,
  `~/p0-image-spike/`, verify-not-authority).
- **Charter given:** the METHODOLOGY red-team charter core verbatim (five lenses + discipline bullets),
  the **position-sensitivity conditional lens** (fires — prompt-assembly edit), a concurrency-lens
  judgment call, and the stage-3 additions (CH8 coverage challenge, CH9/CH10 label + verification-table
  audit). No recorded owner-ruling exists, so CH11/CH12 (ratification audit) do not fire — the reviewer
  was asked to confirm that reading.
- **Reviewer-reported context-file sha256** (spot-checkable):

| sha256 | file |
|---|---|
| `3e1146f2a9a208db7e4784153b3865e0f2e1322f345b390baf1ea5df6bfdd39b` | brain/bridge/provider.py |
| `d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260` | brain/chat/engine.py |
| `ae2408b2723b616b8fd21d16217da48d214dc5ae51cedea3ce57df6ff57a2e2f` | brain/chat/tool_loop.py |
| `a6483bc91b809a7624f7172a1e041bcda7694ae32c18245c70b30f7dbff15d37` | brain/chat/tool_recruit.py |
| `f553930835e238dff355bd1dd765406854783c329a6bbd346e2cd030bacc9647` | brain/mcp_server/tools.py |
| `11485c6be582dbe73ce8bb20653c7c7ca1ae0348095e9ec95e1c4f012c6e357e` | brain/tools/impls/read_file.py |
| `9c783357e4cce338caaae4779b65375b33aa7959ed63783629a99554b8eb6c48` | brain/tools/schemas.py |
| `f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e` | brain/bridge/server.py |
| `d8a5619549af38fc22227051663bc9d45d5cbf7bdd68a74d79cbba77e016030e` | brain/images.py |
| `4e5e1be91504a6d400b44c6b6463b873018d8a3b5c0b5846af730da986fc92fb` | brain/tunables.py |
| `01885a618fdc428c7ce5970c815d0fc5c2588c1017782f4ac71e979b4ce091f8` | app/src/components/ChatPanel.tsx |
| `ddd611d36559004896016cab2964cd3888bb10fe9c97d5ce29ece6daaa95a206` | changes/…/1-spec.md (pre-revision) |
| `9338b9b63d91f423b3b6f67ef946f3be41f3da8fc4cf8194df26810d71616b1b` | changes/…/1.5-criteria.md (pre-revision) |
| `029d9f917bea5e90142e0f9303b29646ee399e197e4fe637bde1f6f50fc09de9` | changes/…/2-plan.md (pre-revision) |

Reviewer also ran `python -c "from mcp.types import ImageContent"` (OK; fields type/data/mimeType/…),
confirming the `ImageContent(type="image", data=…, mimeType=…)` constructor, and confirmed
`python -m brain.mcp_server --persona-dir <path>` entrypoint exists (`brain/mcp_server/__main__.py`).

## Reviewer verdict — worst severity: **MAJOR** (no blocker; premise spike-proven and factually sound)

### Three MAJOR findings (routed to stage 2)
- **F-1 (MAJOR, logical).** Plan step 1 inserts the image branch "before the UTF-8 decode" (`read_file.py:131`),
  but the text size-cap check fires earlier at `read_file.py:102-114` (`if size > files.read_max_bytes:
  return "file too large"`). Any image 256KB–`image_max_bytes` is refused before the image branch runs;
  the separate `files.image_max_bytes` cap (E2) is dead code, effective cap = `min(256KB, image_max_bytes)`.
  No criterion caught it (see G1). **Fix:** branch to the image path *before* the text-cap check at :102.
- **F-3 / F-5 / G2 (MAJOR, fidelity + unstated risk + coverage).** Image visibility silently shifts from
  automatic (pixels pushed by `_chat_with_images`) to opt-in (model must choose to call `read_file`). The
  owner's hard #48 requirement ("must still let her *see* images") is delegated to C1/C2, which test only the
  **instructed** read — the **organic** image-share turn (user shares a photo, asks "what is this?") is
  unmeasured. If the model doesn't self-initiate, #48 regresses silently.
- **F-5 (MAJOR, fidelity).** "closes #43" overreaches: E2/step-1 keep non-image binaries (incl. PDF —
  #43's own example) refused by default, so the mechanism closes #43 only for text files. The prose claim
  outruns the mechanism.

### Lower-severity (fixed in place / carried)
- **F-4 / G4 (ADVISORY→addressed).** `build_volatile_context`/`assess_salience` are fed `user_input`; if the
  surfaced `[the user shared a file: <path>]` line is injected into that input, it perturbs salience/volatile.
  No criterion checked volatile *content* on a file-send turn (C5 checks presence only).
- **G1 (MAJOR coverage), G3 (MINOR multi-file), G5 (ADVISORY audit base64).** Named coverage gaps.
- **C1 caveat.** Harness must resolve the pinned model from persona config, not hardcode an alias.

### Lenses the reviewer earned clean
- **Factual:** every cited symbol/line verified against source (line refs off by ≤9 in a couple of spots,
  immaterial). The volatile-drop mechanism (engine forks on this-turn `image_shas`; provider forks on
  replayed-history `has_images`; `_chat_with_images` carries no `volatile_suffix`) is confirmed exact.
- **Concurrency lens:** judged NOT to fire — `/upload` reuses `images.save_image_bytes`, content-addressable,
  idempotent tmp-then-rename; no new read-modify-write over shared state.
- **CH9/CH10 label audit (earned):** all gating labels honest; C8/C11 advisory reasons legitimate (not
  dodges). The gaps are coverage (CH8) and two prose claims, not mislabeling.
- **CH11/CH12:** correctly do not fire — the E-marked elaborations are the runner's proposal awaiting the
  owner-sign-off gate, not smuggled under owner authority.

*(The reviewer's full verbatim report is preserved in the run's agent transcript;*
*`agentId: ae30f20bf86f65d5f`. Summary above is faithful; author interpretation → decisions.md.)*

## Author disposition (→ stage-2 revision, this lap)
- **F-1:** plan step 1 rewritten — sniff media type from a byte prefix and gate images on `image_max_bytes`
  BEFORE the `:102` text cap; text files fall through unchanged. New **C12** guards it (>256KB image
  returned, with a mandatory shown-able-to-fail against mis-ordered/pre-change code).
- **F-3/F-5/G2:** new **C13 [GATING]** — organic (uninstructed) image-share turn on the real server/pinned
  model reports the secret; its failure is a **blocker routing to owner** (design proactively emits the
  read). Spec caveats added (automatic→opt-in shift; #43 scope). Open axis 4 (owner-awareness) + axis 5
  (#43 scope) added to the plan.
- **F-5 prose:** "closes #43" softened to "closes #43 for text + images; PDF/binary per axis 2" in spec +E3.
- **F-4/G4:** plan step 5 pins the injection point (path line only on the outgoing user message, not into
  salience/volatile input). New **C14 [GATING]** asserts volatile/salience equivalence on a file-send turn.
- **G5:** new **C15 [ADVISORY]** — image audit rows summarize, no base64 blob.
- **C1 caveat:** measurement table now requires the harness to resolve the model from persona config.
- **G3 (multi-file):** recorded as out-of-scope (open axis 6), single-file per turn matches current UI.
