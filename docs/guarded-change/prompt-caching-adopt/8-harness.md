# 8 — Harness: conformance + regression (stage 8)

Build: `feat/prompt-caching-build` @ `0fd1e4a7` (P2+P3) on top of P1 (`4546c24e`).
Method: `scripts/cache_replay_workload.py`, scratch persona (16KB seeded voice.md +
deterministic memory fixtures), 8 real `claude` turns/arm, single session, gap 3s (< 5-min TTL).
OLD arm = main's code (P1 instrumentation, no A/A+); NEW arm = the build. `--dump-replies` for C7.

## Conformance (vs 1.5-criteria.md)

| Criterion | Result | Evidence |
|---|---|---|
| **C1** frozen system byte-stable (replay) | **PASS** | NEW: **1 distinct `system_sha256`** over 8 turns (`byte_stable: True`). The make-or-break — the static head is frozen. |
| **C2-text** create drops, read not collapsed | **PASS** | mean `cache_creation` 51087 → 48200 (**−6%**, ≥5% bar); `cache_read` 47384 → 75195 (rose, did not collapse). |
| **C3** cache_read_ratio improves | **PASS** | ratio-of-means OLD 0.93 → NEW 1.56 (NEW ≥ OLD; more prefix served from cache). |
| **C7-auto** image path byte-identical | **PASS** | `test_respond_image_turn_system_message_is_unsplit_full` green; image turn keeps `build_system_message`, `volatile_suffix=None`. |
| **C6** existing suite | **PARTIAL** | chat+bridge+streaming subset **860 passed**, ruff clean. Full backend suite owed pre-merge. |
| **C4** realistic magnitude (16KB voice + real history < 80-msg) | **PARTIAL** | Magnitude recorded on scratch-8turn (−6% create / **+59% read**). The formal real-`--history-file` run not done — `active_conversations` was empty (no live session). The +59% read already shows history-prefix caching engaging as the 8-turn buffer accumulates. |
| **C5 / C2-live** dominant tool-schema term on live streaming+tools path | **NOT RUN** | Requires driving the real bridge with tools OLD vs NEW. Stage-6 confirmed the code reaches the live `_StreamingProxy`→`chat_stream` path; the measurement is a human-gated decision (below). |
| **C7** voice fidelity + ambient use (human) | **PENDING HANA** | Side-by-side dumped: `/tmp/pc-old.replies.json`, `/tmp/pc-new.replies.json`. Author spot-check: NEW fully in voice, uses ambient state (names love=7/grief, recalls leaving/coin), no task-hijack. |

## The headline number

**`cache_read` +59% per turn** (47384 → 75195) with **`cache_creation` −6%**. On scratch (no deep
seeded history, no tools) the create drop is modest — the frozen head (~4.2K) is small vs the
~50K of volatile + tool schemas + accumulating history, exactly as the stage-3 red-team predicted.
The win shows up on **read**: A+ pushing the per-turn-changing bytes to the tail makes the
system+history prefix cacheable, so far more of it is served from cache (+59%) instead of
re-created. Mechanism confirmed: C1 byte-stable + read climbing per-turn as history accumulates.

## Regression (advisory — no comparable live baseline)

Not gating: the project has no replay baseline captured pre-P1, and live aggregates aren't a
comparable workload. The A/B above IS the comparable measurement (same seed both arms). No
regression signal — NEW strictly improves cache_read and slightly improves cache_creation.

## Verdict

Core mechanism **proven** (C1 + C2-text + C3 + C7-auto all PASS). Remaining before merge:
**(a)** Hana's C7 voice read (gating human); **(b)** decision on C5/C2-live — measure the live
tools path, or accept (C1 byte-stability proof + text-path +59% read + stage-6 code verification)
as sufficient; **(c)** full backend suite (C6). Not a blocker or major — the change is sound;
these are the remaining conformance confirmations.
