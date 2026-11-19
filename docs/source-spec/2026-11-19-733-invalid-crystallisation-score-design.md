# One bad crystallisation item costs only that item (#733)

Follow-up to #640 (PR #728). Severity low.

## Why

Pass 2's extractor (`extract_from_thinking`, `brain/chat/extractor.py`) validates the whole reply with one `ExtractorOutput.model_validate`. Since #640 an item with no `importance` (missing or null) is removed before that call and recorded. An item whose `importance` is there but invalid (`""`, `0`, `11`, `"high"`, `9.5`) still fails the whole reply. That loses the scored items, memory writes, emotion deltas and reflex audits in the same reply, for one rare bad field.

## Goals

- A crystallisation item that fails its own validation is dropped, and the rest of the reply applies.
- The drop is recorded in the parse-failure log, as a missing score is.
- One parse-failure row per reply, as #640 set: if the reply still fails validation for another reason, the single `schema:` row carries the drop note.

## Non-goals

- `memory_writes`, `reflex_audit`, `emotion_delta` and `interest_candidate` keep today's behaviour: a bad one still fails the whole reply. That is its own change if wanted.
- No clamping. An `11` is not turned into a `10`: that would queue the item as a soul candidate (the threshold is 8) on a score the model never gave.
- No change to what counts as valid. Pydantic's own rules for `CrystallisationCandidate` decide (it accepts `"9"` and rejects `9.5`), so this code doesn't copy them.

## Anti-goals

- Reply text in the parse-failure row's `error` field (#427). Pydantic's error messages include the input value, so they are never written there.
- An invalid item reaching the soul queue.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | Validate each crystallisation item with `CrystallisationCandidate.model_validate` before the whole-reply validation, and drop the ones that fail | Covers every bad field on an item, with pydantic as the one source of what is valid |
| B | Extend #640's pre-filter to match bad `importance` values by hand | Copies pydantic's coercion rules, and misses other bad fields (a whitespace-only theme) |

Hana approved A on 2026-11-19.

## Design

`_drop_unscored_crystallisation(data) -> int` becomes `_drop_bad_crystallisation(data) -> tuple[int, int]`, returning `(unscored, invalid)`:

- An item that is a dict with `importance` missing or null counts as unscored and is removed, as today.
- Every other item is checked with `CrystallisationCandidate.model_validate(item)`. One that raises `ValidationError` counts as invalid and is removed. An item that isn't a dict is invalid.
- A `crystallisation` value that isn't a list is left alone, for the whole-reply validation to judge, as today.
- The kept items stay in the reply as the model sent them (not the validated copies), so the whole-reply validation sees what it sees today.

The note names only counts and the field, never values:

- unscored only: `N crystallisation item(s) without importance` (today's text, so #640's tests and log readers keep working);
- invalid only: `N invalid crystallisation item(s)`;
- both: the two joined with `, `.

As today, the note is written as `"<note> dropped"` on its own row after a successful validation, or as the `schema: <note>; <error>` prefix when the reply fails anyway.

## Wiring

No new organ. The extractor's output feeds `apply_side_effects` as before. The parse-failure log row is the existing one, `site="extractor"`.

## Tests (written first)

- Through `run_pass2_monologue`: one item scored `"high"` and one scored 9. The scored item is in `soul_candidates.jsonl`, the reflex audit lands, and one row is recorded whose error names the count and none of the reply's text. Fails on main.
- `extract_from_thinking` with each of `""`, `0`, `11`, `9.5`: the item is dropped, the other is kept, one row.
- A whitespace-only theme next to a good item: only the bad one is dropped.
- A missing score and an invalid one in the same reply: both counts in one row.
- An invalid item plus a bad memory write: one `schema:` row that carries the drop note.
- #640's existing tests pass unchanged.
