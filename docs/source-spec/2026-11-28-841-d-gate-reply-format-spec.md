# The outreach review's D gate parses its reply (#841)

Decision sheet v3, 2026-11-28 (lines S6–S27). Every statement below cites the sheet line it comes from.

## The problem

The D-reflection gate of the outreach review asks the model for a promote or filter decision on each queued candidate:
- `run_initiate_review_tick` in `brain/initiate/review.py` calls `run` in `brain/initiate/reflection.py`;
- `parse_structured_response` reads a JSON object with `decisions` (each `candidate_index`, `decision`, `reason`, `confidence`) and an optional `tick_note`;
- no prompt describes that object, and no reply has ever parsed. Every tick takes the `malformed_json` path, which promotes every candidate.

One long-running persona's `initiate_d_calls.jsonl` holds 245 calls from 2026-05-15 to 2026-11-27, all `malformed_json`. Two more defects sit in the same path: the candidate index is shown counting from 1 but read counting from 0, and the adaptive calibration block is never passed in.

## Design

### The reply-format paragraph [S7, S13]

- **Where.** The last segment of `[initiate.reflection] user_message_segments` in `brain/prompt_strings.toml`, after "Promote at most 2. The default is filter.", gains this text. Each line ends with a newline [S27].
  ```text
  Reply with only this JSON object and nothing else:
  {"decisions": [{"candidate_index": 1, "decision": "filter", "reason": "one short sentence", "confidence": "high"}], "tick_note": null}
  Give one decision for each candidate. candidate_index is the number in brackets before the candidate. decision is "promote" or "filter". confidence is "high", "medium" or "low". tick_note is one short sentence about this tick, or null.
  ```
- **The parser.** `parse_structured_response` still raises `ValueError` for a reply it cannot use, and the callers in `run` still catch only `ValueError` [S7]. Three cases now raise it:
  - an item with a missing field, or a field of the wrong type [S22]: `candidate_index` must be an integer (not a bool, a string or a float such as `2.0`), and `decision`, `reason` and `confidence` must be strings (`null` is wrong) [S27];
  - a `decision` other than `promote` or `filter` [S23];
  - a `confidence` other than `high`, `medium` or `low` [S23].

  Values are compared without regard to case and stored in lower case, so `"Filter"` counts as `filter` [S23]. A reply that raises takes the existing malformed path: the retry, then promote all, with a `malformed_json` row [S22, S23, S12].
- **Alternatives considered:** the other option posed with S7, loosening the parser to read a plain-text reply, was not chosen [S7].

### The candidate index [S8]

- **Meaning.** `candidate_index` is the number shown in brackets, counting from 1. In `run_initiate_review_tick`, the decision loop checks `1 <= candidate_index <= len(candidates)` before it maps the index to `candidates[candidate_index - 1]`, so an index of 0 can never select the last candidate. The covered set is built from the same shifted indices [S8].
- **Out of range.** A `candidate_index` outside 1 to `len(candidates)` is logged, as the existing warning does, and its decision is skipped. The candidate it failed to cover is then promoted by the existing not-covered default [S8, S12].
- **Duplicates.** When a reply holds more than one decision for the same candidate, the first one for it is applied. Any later decision for it is logged and ignored, so each candidate gets exactly one outcome [S17].
- **Synthetic decisions are unchanged.** That covers the rate-limit path's synthetic `DDecision(candidate_index=0, …)`, which is never mapped to a candidate, and the low-confidence forcing in `run`, which copies the model's index [S26].
- **Counts.** `run` computes the row's promote and filter counts over every parsed decision, as today. The first-decision rule and the out-of-range skip apply where `review.py` maps decisions to candidates [S25].

### The calibration block [S9]

`run` passes `persona_dir=deps.persona_dir` to `build_system_message`.
- **Adaptive persona.** A persona whose `d_mode.json` says `"adaptive"` gets `build_calibration_block` prepended.
- **Every other persona.** A persona whose `d_mode.json` does not say `"adaptive"`, or that has no `d_mode.json`, gets the same system message as today [S9]. The adaptive file holds `{"mode": "adaptive"}` [S27].

### The drift history [S18]

`detect_drift` (`brain/initiate/adaptive.py`) leaves out rows of calls that made no decision. Those are the rows whose `failure_type` is `timeout`, `rate_limit`, `provider_error` or `malformed_json`. It keeps `both_low_confidence` rows, which carry real promote and filter counts. Its two existing bootstrap floors (100 rows in all, 50 historical) then count only kept rows [S27], so it stays quiet until enough real decisions build up, as on a new persona.

### The draft-space reason [S19]

`demote_to_draft_space` writes `d_reason: ` followed by `json.dumps(reason, ensure_ascii=False)`. Quotes and line breaks in a model's reason are escaped, and characters such as an em-dash are written as they are [S19, S24]. Nothing reads the frontmatter back.

### Unchanged [S10, S11, S12]

- **Rollout [S10].** D's decisions take effect from the first tick after the update. The drift check (`detect_drift`, inside the compose slot) keeps running, with the history above [S18].
- **The retry [S11].** It still calls the same provider and is still logged as `"sonnet"`; that is #533.
- **What #829 revisits [S12].** The single batched call, "Promote at most 2", and every fallback:
  - a candidate the reply leaves out is promoted;
  - a reply still malformed after the retry promotes all;
  - a rate limit demotes all to draft space;
  - three failed calls in a row promote all (the code counts timeouts and provider errors).

## Acceptance

| Sheet | Check | Before ship? |
|---|---|---|
| S7, S13 | The rendered user message ends with the paragraph above, byte for byte (a test renders `build_user_message`). | yes |
| S14 | A canary: every key `parse_structured_response` reads (`decisions`, `candidate_index`, `decision`, `reason`, `confidence`) appears in the paragraph, and the paragraph's example line parses to one filter decision for candidate 1. | yes |
| S8, S14 | Through `run_initiate_review_tick` with a fake gate provider answering in the documented format for 3 candidates (`[1]` filter, `[2]` promote, `[3]` filter): candidate 1 and candidate 3 are demoted to `draft_space.md` and removed from the queue; candidate 2 reaches composition. A reply of `[1]` filter, `[2]` promote and index 4 filter, for 3 candidates, logs the warning for index 4. Candidate 3, which nothing covered, is promoted. A reply with index 0 logs the warning and does not touch candidate 3. | yes |
| S22, S23 | Each of these replies is malformed and takes the retry, then promote all:
- an item with no `reason`;
- `candidate_index` null;
- `decision` `"skip"`;
- `confidence` `"unsure"`.

The tick writes a `malformed_json` row, and nothing raises out of `run_initiate_review_tick`. A reply with `"Filter"` and `"High"` parses as filter, high. | yes |
| S17 | A reply with filter then promote for `[1]` demotes candidate 1 and does not compose it. The second decision is logged. | yes |
| S18 | Two checks of `detect_drift` with 245 `malformed_json` rows in the log. With 40 parsed rows besides, it returns `None`, because fewer than 100 rows are kept. With 150 parsed rows besides, one of them `both_low_confidence`, whose per-row promote rates alternate 0.0 and 1.0, its median (0.5) and spread come from those 150 rows only, and a current window at rate 0.5 raises no alert. | yes |
| S19, S24 | A filtered reason containing `"`, a line break and an em-dash is written on the `d_reason:` line as one JSON string. The quote and line break are escaped, and the em-dash is kept as it is. | yes |
| S9, S14 | `run` sends a system message that contains the calibration block for a persona with `d_mode.json` `{"mode": "adaptive"}`, and none for a persona without the file. | yes |
| S10 | The tick's `initiate_d_calls.jsonl` row for a parsed reply has `failure_type` null and non-zero promote or filter counts. | yes |
| S15, S21 | A live check, through `scripts/clean_shell.sh` on a temp `KINDLED_HOME` with made-up candidates and never the live persona: the real Haiku D call parses for a 1-, a 2- and a 3-candidate tick (3–6 model calls). The result is recorded in the PR. It also records whether any reply's reason is the example's "one short sentence" [S27]. If any reply does not parse, the PR is not marked ready and the result goes back to the owner [S21]. | yes, before the PR is marked ready |

Existing assertions on the old end of the user message (`tests/unit/brain/initiate/test_reflection_prompt.py`) are updated to the new text. All but S15 run in the normal suite. The live check is a script run once, and its replies are not committed.

## Invariants [sheet invariants table]

| Invariant | Holds? | Where |
|---|---|---|
| I1, user surface | holds | no knob is added (Design) |
| I5, older brain reads state | holds | no field is added; `initiate_d_calls.jsonl` rows keep their shape |
| I7, autonomous behaviour recipe | holds for #841 | no new call; D's existing bypass of the daily cap is unchanged (Unchanged) |
| I12, Claude CLI on subscription | holds | the same provider and tier |
| I13, byte-stable chat prefix | n.a. | the D gate's system message is its own `build_system_message`, not the chat prompt prefix |
| I14, macOS, Windows and Linux | holds | `d_reason` keeps non-ASCII text as written (S24), and the file is written as UTF-8 as today |
| I16, canary for an invariant | holds | the S14 canary |
| the others | n.a. | no organ, store, cadence, body, gate, prompt-prefix, platform or overlay change |

## Build

- **The loop.** The build goes through guarded-change [S16].
- **Sequencing.** #827 (ready, not merged) changes `demote_to_draft_space` in `reflection.py`, which S19 also changes, and does not touch `run`. Whichever of the two merges second may need a small rebase.
- **Build questions.** A question at build time that changes a mechanism comes back to the owner as a stage-7 fork.
