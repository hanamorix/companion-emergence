# P0 image-tool-route — FINAL confirmation red-team (lap 3, scope A) — VERBATIM RECORD

2026-08-12. Cold, independent reviewer, no shared context. Run after the owner ratified scope A; the owner's
verbatim ruling was supplied so the reviewer could audit the ratification (CH11/CH12).

## Provenance
- **Reviewer:** `general-purpose` subagent, model **opus** (adversarial final gate on a scope-A build with
  a new store + a backend Hana surface; ratification audit against owner's verbatim words).
- **Context (closed set):** the three revised stage artifacts + decisions.md (owner ruling) + 3-redteam-plan.md
  + the config `redteam_context` (this worktree's `brain/`) + the spec touched-files.
- **Charter:** METHODOLOGY core (five lenses + discipline) + position lens (fires) + concurrency lens
  (re-check, new store) + stage-3 additions CH8/CH9/CH10 + **CH11/CH12 ratification audit (fires — owner
  ruling present)**.
- **Reviewer-reported sha256** (artifacts + source):

| sha256 | file |
|---|---|
| `6f9485a2c37af0a7b3a081af92081d75e8dcb6647d6a2222fdefa3f8ae08824f` | 1-spec.md (pre-Finding-fix) |
| `3b05cc2f9c483456278080f2c41b9dc1bab1c7da65b37114bb9c37399a7a39ca` | 1.5-criteria.md (pre-Finding-fix) |
| `b52155e6fa67407a2b9805344daeca919ecb0dd95371bc01820ce63bc702744d` | 2-plan.md (pre-Finding-fix) |
| `11485c6be582dbe73ce8bb20653c7c7ca1ae0348095e9ec95e1c4f012c6e357e` | brain/tools/impls/read_file.py |
| `d8a5619549af38fc22227051663bc9d45d5cbf7bdd68a74d79cbba77e016030e` | brain/images.py |
| `f0b0b715746bc9f8e27964ec7c24301a14b6b80dab4d38db8f8b533d1df62d7e` | brain/bridge/server.py |
| `d359be520046a66c502ca5f0b56a0c61e8f4a13fbef93868e6fe127bed0d1260` | brain/chat/engine.py |
| `3e1146f2a9a208db7e4784153b3865e0f2e1322f345b390baf1ea5df6bfdd39b` | brain/bridge/provider.py |
| `a6483bc91b809a7624f7172a1e041bcda7694ae32c18245c70b30f7dbff15d37` | brain/chat/tool_recruit.py |
| `f553930835e238dff355bd1dd765406854783c329a6bbd346e2cd030bacc9647` | brain/mcp_server/tools.py |
| `4e5e1be91504a6d400b44c6b6463b873018d8a3b5c0b5846af730da986fc92fb` | brain/tunables.py |
| `bf86fb817903e52a66c5763e59789823915c0d1178eedabf1a5dfd8ff378ff86` | brain/bridge/chat.py |
| `9c783357e4cce338caaae4779b65375b33aa7959ed63783629a99554b8eb6c48` | brain/tools/schemas.py |

## Verdict — worst severity: MAJOR (two)
- **Ratification audit (CH11/CH12): VALID on every axis.** #43→A, seeing→opt-in/C13-gating,
  image_max_bytes≥20MB, migration accepted-risk — each maps to the owner's verbatim words; no partial/adjacent
  resolution. One ADVISORY inflation (Finding 3): `upload_max_bytes` mislabeled owner-ratified.
- **Scope-A steps 9-11 feasible**; C16 (pre-change /upload 415 on text = real shown-able-to-fail), C8b, C13,
  C17 discriminating; factual/logical/position/concurrency lenses earned-clean; new `files.py` store adds no
  new RMW window (mirrors idempotent `save_image_bytes`).
- **Finding 1 (MAJOR):** spec claimed PDF "readable"; owner's verbatim named PDF; mechanism stores PDF
  unreadable (`read_file.py:135-145` binary note). → 3rd occurrence of the #43-prose finding class →
  iteration-cap tie-break. **Disposition:** spec reconciled to C16 (text readable; PDF stored+surfaced, not
  extracted); **PDF-readability escalated to owner at the post-build gate.**
- **Finding 2 (MAJOR):** new store derived on-disk ext from client filename, no traversal criterion.
  **Disposition:** on-disk path = validated-sha only (traversal-proof); filename = display metadata; added
  gating **C17**. Dissolves Finding 4 (dedup wrinkle).
- **Finding 3/4 (ADVISORY):** provenance label fixed; dedup wrinkle dissolved by sha-only path.

*(Reviewer's full verbatim report preserved in the run transcript; `agentId: a709f806a20d8cc8e`.)*

## Post-review artifact fixes (applied this lap, then criteria FROZEN)
- Plan step 9/10/11: sha-only on-disk path (Finding 2); ref shape `{kind,sha,media_type?,filename?}`.
- Criteria: added **C17** (traversal-proof store); C16 note + spec prose reconciled to honest PDF scope
  (Finding 1); plan axis-2 provenance relabel (Finding 3).
- Frozen criteria sha256 recorded in decisions.md.
