# The interior block as a trail of notes once the engine is live (#688)

Part of the emotion rework (#240). Lands before Engine 6 (#633) turns the switch on.

## Why

The interior-continuity block exists so her own recent inner life colours the next turn (`brain/monologue/ambient.py`). Today it holds one raw thought, cut at 1,200 characters. Once the engine is live, B1 PR 6 (#690) stores each thought as its short `feed_digest`. One digest is too thin to carry that continuity on its own. A few in a row show where her attention has been moving. So once live, the block becomes a short trail of recent notes.

`recall_monologue`'s tool description already promises things the code doesn't do, with the switch on or off, so it gets corrected in the same change.

## Measured (2026-11-16, Nell, 212 `record_monologue` pairs, counts only)

- Raw thought: median 1,397 characters. 70% are longer than the cap.
- Digest: median 220, p90 295, max 365.
- The five newest digests with their bullets total a median of 1,116 characters. In 39% of windows they pass 1,150, so today's slicing would cut the fifth mid-line.

## Goals

- **Live:** the block shows up to 5 notes, newest first, and never cuts a note mid-line. A note that would pass the cap is dropped, along with every older one.
- **Live:** the header suits several notes in either voice. It holds before PR 7 (third person) and after it (first person).
- **Switch off:** the block is byte-for-byte what it is today.
- `recall_monologue`'s description says only what is true in both states.

## Non-goals

- The count (5) and the cap (1,200 characters) stay as they are. Per-turn prompt size doesn't grow.
- The footer is unchanged.
- No change to how traces are stored, read or promoted (`PendingQueue.read_recent`).
- No change to the digest's voice. That is PR 7.

## Anti-goals

- A block that grows per-turn tokens.
- A live block that shows half a note.
- A switch-off persona seeing any change at all.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | Whole notes within today's 5 / 1,200, newest always shown, neutral header, only when live | Same cost as today. Usually 4-5 notes |
| B | The latest note only | About 250 fewer tokens per turn, but one 1-2 sentence note carries little continuity |
| C | Raise the cap to about 1,600 and stamp each note's time | Richest, but about 100-150 more tokens every turn |

Hana picked A on 2026-11-16.

## Design

### The block (`brain/monologue/ambient.py`)

`build_interior_continuity_block` reads the switch once, with `store.emotion_engine()` and `is_live()`. Any error counts as off and is logged, throttled.

- **Off:** today's code path, unchanged.
- **Live:**
  - Header: `prompt_strings` key `monologue.ambient.header_live`, text `── interior continuity (your recent thoughts, as you noted them) ──`. This is new text, so no em-dash in the words; the rule glyphs match today's header.
  - Body: the header line, then one `· <note>` line per trace, newest first, whitespace collapsed as today.
  - Fit: always include the newest line, cut to the cap if it alone passes it. Then add each older line only if the header, the lines so far and this line, joined by `\n`, stay within `char_cap`. Stop at the first line that doesn't fit.
  - Footer: unchanged, appended after the fit as today.

Traces stored just before the switch-over still hold raw thought. A long one in second place is dropped rather than cut. That is the rule working.

### The tool description (`brain/tools/schemas.py`, `recall_monologue`)

New text, true in both states, no em-dashes:

> Search your own recent interior for an earlier thought. Each match comes back as you kept it, with when you had it: the short note you made at the time, or the whole thought if it is from before you kept notes.

This removes three false claims:
- "Recent thoughts come back verbatim" is false once live.
- "Older ones return blurred" is false in both states, because recall reads only the pending queue's active traces.
- "Reaching for a thought keeps it vivid" is false too, because the bump is a no-op for a queue trace, as the code's own comment says.

This text sits in the cached tool prefix, so the prefix moves once, on the first turn after release.

## Wiring

- **Reads from:** the `monologue_trace` items in the pending queue, which PR 6 writes as digests once live, and the engine switch (`is_live()`).
- **Feeds into:** the chat prompt's volatile tail (`build_volatile_context` → `_build_interior_continuity_block`), so the next reply is composed with her recent notes. The description feeds the tools list the model reads when it decides to call `recall_monologue`.
- No new organ, no new state file.

## Tests

- **Live, through the real path:** set up a persona made live and five queued digests of realistic lengths. Make it live the way the merged B1 tests do today, with a local `_go_live`. When Engine 6 lands, switch to `make_live`, following the "After Engine 6" recipe in the B1 plan. `build_volatile_context`'s interior block then carries the live header and only whole notes, newest first, within 1,200 characters plus the footer.
- **Live fit:** the newest note longer than the cap is shown cut and alone. A second note that would pass the cap is dropped, and so is every older one.
- **Off pin:** the same inputs with the switch off give today's exact string. An unknown switch gives today's string too. On main that is `refuse_the_switch_stat`; once Engine 6 lands it is `ruin_the_state`.
- **Description:** the `recall_monologue` description holds none of "verbatim", "blurred" or "vivid", and no em-dash.
- **Wording snapshot:** approve the two new strings and name them in the PR body.

## Builder's calls (Claude's, for Hana to check)

- Gate the new rule on the switch, so switch-off behaviour stays identical, matching every other B1 change.
- Keep a separate `header_live` key rather than changing the existing header, for the same reason.
- No full `guarded-change` loop. Per-turn prompt bytes can't grow (same cap), and the one-time prefix move is stated above. Change this if you want the loop.

## Deferred

| Item | Why | Revisit |
|---|---|---|
| The block and recall read only the pending queue, so consolidation can leave fewer than five recent traces there | The Root 2 queue stopgap, tracked in #420 and #208 | When the memory rework's Phase 4 replaces the stopgap |
| Note times in the block (approach C) | Costs tokens every turn | If the trail reads as timeless in practice |

These also go in `project_companion_emergence_deferred.md` and the next brainstorm.
