# P0 image-tool-route — Stage-6 red-team (code) — VERBATIM RECORD

2026-08-12. Cold, independent reviewer, no shared context. Reviewed the built code against the FROZEN
criteria + plan.

## Provenance
- **Reviewer:** `general-purpose` subagent, model **opus** (coordinated multi-file correctness review of a
  transport-deletion + wire-contract change on a position-sensitive assembly).
- **Reviewed diff — generated MECHANICALLY (ST6d):** `git --no-pager diff cd29bc61 ffa907c9` (single commit,
  31 files, +1927/−967). Full mechanical diff, not hand-curated.
- **Context:** the reviewed diff + `{1.5-criteria (frozen), 2-plan, 1-spec, decisions}` + the config
  `redteam_context` (this worktree's `brain/` + frontend + touched tests).
- **Charter:** METHODOLOGY core (five lenses + discipline + spot-verify) + position lens (fires) +
  concurrency lens (re-check, new `file_store.py`) + stage-6 mechanical-diff duty.
- **Reviewer confirmed** the frozen criteria file hash matches decisions.md's FRZ record
  (`3f90cd52…734846`).

## Verdict — worst severity: ADVISORY (nothing above MINOR). Routing: **CLEAN → proceed.**

Earned-clean lenses (with citations):
- **Factual CLEAN** — `read_file.py:131` image branch precedes text cap `:178`; `provider.py:248` `"Read"` in
  `_BUILTIN_TOOLS_DISALLOWED`, `:265-266` `_apply_lean_flags` emits `--disallowedTools … Read`;
  `tools.py:70-91` emits `ImageContent` on the `image` key, base64-free audit; `server.py:725` `SharedFileRef`
  sha-pattern.
- **Fidelity CLEAN (terms pinned)** — transport symbols (`_chat_with_images`, `_message_has_image`,
  `_build_stream_json_user_message`, `has_images` branch) grep-absent from non-test `brain/`; ImageContent
  emission real; `Read` untouched in the disallowed tuple; `file_store.file_path` sha-only; #43 text
  end-to-end readable, PDF-readability honestly escalated (not claimed).
- **Position-sensitivity CLEAN** — C9 text-turn assembly byte-identical to pre-change `else` branch
  (`engine.py:192-193,270-272`); C14 salience/volatile fed raw `user_input`, path line only on
  `outgoing_user_text` (`engine.py:193-198` vs `:255-256`); C5 fork structurally removed.
- **Concurrency CLEAN** — `file_store.save_file_bytes` faithfully mirrors `images.save_image_bytes`
  (content-addressed, dedup, unique per-writer tmp, atomic `os.replace`, post-race recheck); no new RMW
  window.
- **Gating-criterion verification table** (governed path per criterion) + **deviation audit** (file_store
  naming = real collision; turn_logger UP035 = inert, == main's `2ef26f23`; deleted tests = transport-only,
  no retained coverage dropped) — all OK.
- **CI:** ruff clean; non-live suite 4235 passed + one pre-existing date-dependent failure
  (`initiate/test_review.py::test_review_tick_gate_blocks_send_records_hold`) reproduced identically on the
  base commit, outside the footprint. Live C1/C2/C13 + C16 live arm = stage-8 usage-gated (correct deferral).

MINOR/ADVISORY residue (fix-in-place-or-note, NOT a bounce):
- **ADVISORY (over-recruit):** `force_files` unioned the whole `_FILE_TOOLS` (incl. write-capable
  `propose_write`), broader than plan step 4's `{read_file}`. → **FIXED IN PLACE (stage 7)**: narrowed to
  `keep.add("read_file")`; test updated to assert only read_file forced.
- **MINOR:** C5's pre-change reproduce not executed in-tree (structurally impossible post-change; the
  base-commit repro is a stage-8 harness item).
- **ADVISORY:** dead `_SHA256_HEX_RE` constant (`server.py:101`) after the ChatReq validator removal
  (harmless; ruff doesn't flag module constants); dormant `ImageBlock` (plan permits). → NOTED, left.

*(Reviewer's full verbatim report preserved in the run transcript; `agentId: a1d673607448c4ed8`.)*
