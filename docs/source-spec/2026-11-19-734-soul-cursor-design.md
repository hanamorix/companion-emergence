# get_soul: page by cursor, read a clipped field, and filter by any stored love type (#734)

Follow-ups from #596 (PR #731), which made `get_soul` return one bounded page with `offset`/`limit`/`love_type`. Severity low.

## Why

1. **Offset paging breaks when the soul changes during a walk.** The list is newest first. A crystallization added between two calls shifts every later item down a place, so one item comes back twice. A revoked one shifts them up, so one item is never returned.
2. **A clipped item's full text can't be read.** An item too large for a page has its text fields cut and is marked `clipped: true`, and nothing returns the rest.
3. **Some stored love types can't be used as a filter.** `brain/migrator/emergence_kit.py` stores `love_type` unchecked, so `by_love_type` can name a type that `get_soul(love_type=...)` refuses. The schema's `enum` also stops a model from sending it.

## Goals

- Walking every page while items are added or revoked between calls returns each item that was active the whole time exactly once (the acceptance test).
- A clipped field can be read to its end, within the page's byte bound.
- Every type `by_love_type` shows can be used as `love_type`.
- `offset` paging, `boot` and every current argument behave as before.

## Non-goals

- No change to what the migrator stores, or to the order of the soul.
- No new index: the order already has one (`idx_crystal_created`), and the id is the primary key.

## Anti-goals

- A cursor that depends on how a timestamp was written (legacy rows, migrated rows).
- A slice of a field that pushes the result over `MAX_RESULT_BYTES`.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | The cursor is the last item's `id`; the store compares against that row's stored `(crystallized_at, id)` in SQL | Opaque to the model (it copies an id it already has), immune to timestamp formats, survives the cursor item being revoked |
| B | Cursor = `crystallized_at` + `id` text from the result | The result's timestamp is re-serialised, and may not match the stored text for older rows |
| C | Keep offset and return a snapshot token | Needs server-side state per walk |

Hana approved the three parts on 2026-11-19. The cursor's form (A) and the schema change in part 3 are my calls, made reading the code.

## Design

### 1. The cursor (`after`)

- `SoulStore.list_active_page(*, offset, limit, love_type=None, after=None)`. With `after`, the query adds `AND (crystallized_at, id) < (SELECT crystallized_at, id FROM crystallizations WHERE id = ?)` and uses no OFFSET. `SoulStore.get(id)` already exists for checking the id.
- `get_soul(after=...)`:
  - An unknown id gives `{"loaded": false, "error": ...}`.
  - `after` and a nonzero `offset` together are refused, with an error naming both.
  - With `after`, `offset` in the result is null.
- Every result with more to come has `next_after`, the last item's id, as well as `next_offset`, which stays for compatibility. The "more remain" message suggests `after`. `boot`'s lifted message follows automatically, since it comes from get_soul.
- `truncated` with `after`: the count of items older than the cursor (same filter) is more than the items returned.

### 2. A clipped field (`id`, `field`, `char_offset`)

- `get_soul(id=X, field=F, char_offset=N)`, with `F` one of `moment`, `why_it_matters`, `who_or_what`. It returns `{"loaded": true, "id", "field", "char_offset", "length", "text", "next_char_offset"}`.
  - `text` is the longest slice from `N` that keeps the result within the byte bound.
  - `next_char_offset` is null at the end of the field.
- **Errors:** an unknown id, a missing or unknown `field`, or `id` together with paging arguments (`after`, `love_type`, a nonzero `offset`) each give an error naming the fix.
- `char_offset` is clamped to the field, like `read_file`'s.
- A clipped item gains `clipped_at: {field: chars_shown}` for each field it cut, so the next call is `char_offset=chars_shown`. When a page holds a clipped item, the message names that call.

### 3. Love types

- `love_type` is accepted when it is in `LOVE_TYPES` or among the soul's active types (`active_love_type_counts()`). The error lists both.
- The schema's `love_type` loses its `enum`. Its description lists the standard types and says `by_love_type` shows the ones in her soul.

### Schema (`brain/tools/schemas.py`)

`get_soul` gains `after`, `id`, `field` (an enum of the three fields) and `char_offset`, all optional. The description mentions `next_after`. No name changes.

## Wiring

No new organ. `get_soul` is reached through `dispatch()`, the MCP server and `boot`, as before.

## Tests (written first, one at a time)

- **Acceptance:** walk with `after` while items are added and revoked between pages; each item active the whole time comes back exactly once. The same walk with `offset` shows the old bug (documentation, not a pin).
- **The cursor:**
  - `after` an id that was revoked mid-walk still continues;
  - an unknown `after` and `after` with `offset` are errors;
  - `after` with `love_type` stays in the filter;
  - `next_after` is null on the last page.
- **A clipped field:**
  - an item too large for a page is clipped, and reading each clipped field with `id`/`field`/`char_offset` returns the whole text across calls;
  - every result stays within `MAX_RESULT_BYTES`;
  - the errors above.
- **Love types:** a migrated item whose type isn't in `LOVE_TYPES` is reachable with that `love_type`, and an unknown type is still an error.
- **Existing tests:** #596's tests (`test_get_soul_bounded.py`) and `boot`'s pass, with only the message wording updated.
