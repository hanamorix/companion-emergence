# Visible inner life — v0.0.13-alpha.2

**Status:** approved 2026-05-17 (brainstorm)
**Target release:** v0.0.13-alpha.2
**Roadmap item:** Tier 1 #3 — *"The companion's visible inner life"*

## Goal

Replace the existing "Recent Interior" snapshot panel (last-of-each across four engines) with a chronological **journal feed** that reads like checking in on someone you care about rather than a debug dashboard. The brain already produces the content; this is a presentation layer over existing substrate.

No new architecture, no new LLM calls, no schema changes. One new bridge endpoint, one new frontend component, and a small server-side merge helper.

## Scope (locked from brainstorm)

| Decision | Choice |
|---|---|
| **Surface** | Replace `InteriorPanel.tsx`; the feed is the single inner-life view (Q1 = A) |
| **Content scope** | Dreams + research completions + soul crystallizations + outbound initiations (delivered only) + voice-edit proposals (Q2 = B) |
| **Phrasing** | Light template framing — type → opener phrase ("I dreamed…", "I've been researching…", etc.) prepended to the existing summary; no per-item LLM rewrite (Q3 = B) |
| **Update cadence** | Polling-only, piggybacks the existing 5s `/persona/state` cadence (Q4 = A) |
| **History depth** | Fixed top-50 entries, no pagination, no cursor (Q5 = A) |
| **Item layout** | List with colored type-dots per engine, no card borders, body indented under a hairline rule (item layout B) |

## Architecture

Three layers; nothing crosses boundaries unexpectedly.

```
brain/bridge/feed.py
    Read 5 source streams, normalize to FeedEntry, merge by ts desc,
    apply selection rules, return top 50.

brain/bridge/server.py (extended)
    GET /persona/feed (bearer auth, mirrors /persona/state contract)
    -> { entries: [FeedEntry, ...] }

app/src/components/panels/FeedPanel.tsx (new)
    Replaces InteriorPanel in LeftPanel slot. Polls the feed endpoint
    alongside the existing state poll. Renders Layout B.
```

## Data shape

### `FeedEntry` (server-side dataclass + serialised JSON shape)

```python
@dataclass(frozen=True)
class FeedEntry:
    type: Literal["dream", "research", "soul", "outreach", "voice_edit"]
    ts: str          # ISO8601 with timezone; used for sort + "Xm ago"
    opener: str      # journal-voice phrase, server-computed via fixed map
    body: str        # existing summary content; already in Nell's voice
    audit_id: str | None = None   # cross-reference; None for sources without one
```

### Type → opener phrase map

| `type` | Opener |
|---|---|
| `dream` | `"I dreamed"` |
| `research` | `"I've been researching"` |
| `soul` | `"I noticed"` |
| `outreach` | `"I reached out"` |
| `voice_edit` | `"I wanted to change"` |

Hardcoded constant in `brain/bridge/feed.py`. Future tweaks live here; no config surface (per user-surface principle in CLAUDE.md).

### TypeScript mirror

```typescript
export type FeedEntryType = "dream" | "research" | "soul" | "outreach" | "voice_edit";

export interface FeedEntry {
  type: FeedEntryType;
  ts: string;
  opener: string;
  body: string;
  audit_id?: string;
}
```

## Source-log mapping

The bridge already streams JSONL logs via `iter_jsonl_streaming` (`brain/health/jsonl_reader.py` — no full-file loads, closed the 2026-05-11 memory-spike vector).

| `type` | Source path | Selection rule | Fields to populate |
|---|---|---|---|
| `dream` | `<persona_dir>/dreams.log.jsonl` | every entry with a non-empty summary | `ts` from row, `body` from summary/content field, `audit_id=None` |
| `research` | source path determined during implementation by reading how `InteriorPanel` is fed today (the bridge already exposes the latest research entry via `state.interior.research`); use the same data source | every entry with a non-empty summary | `ts`, `body`, `audit_id=None` |
| `soul` | `<persona_dir>/soul_audit.jsonl` | only rows where the entry represents a **crystallization** event (skip routine audit lines) — verify the actual event-type discriminator during implementation | `ts`, `body` from the crystallization summary/text, `audit_id` if the row has one |
| `outreach` | `<persona_dir>/initiate_audit.jsonl` | rows where `decision in {"send_notify", "send_quiet"}` **AND** `delivery.current_state == "delivered"`. Held / errored / hold-via-gate decisions are noise and stay out. | `ts`, `body=tone_rendered`, `audit_id=audit_id` |
| `voice_edit` | `<persona_dir>/soul_audit.jsonl` (or the dedicated voice-edit log if one exists — verify during implementation) | only "proposed" events; accepted/rejected/withdrawn are out of v1 scope | `ts`, `body` from the proposal text/rationale, `audit_id` if available |

Merge logic:

1. Load up to N×2 candidates from each stream (N=50; over-read so post-filter we still have ≥50).
2. Apply each stream's selection rule.
3. Concatenate, sort by `ts` desc.
4. Slice top 50.
5. Wrap each with the type's opener phrase.

Fault isolation: any one source failing to load/parse logs the exception and continues with the others — empty feed is preferable to a 500.

## Bridge endpoint

`GET /persona/feed`

- Auth: bearer token, same as `/persona/state`
- Query params: none in v1
- Response: `200 { "entries": FeedEntry[] }` — `entries` is at most 50, always sorted ts-desc
- Error semantics: same as `/persona/state` — 401 on missing/bad token, 503 if persona dir not resolvable

Polled by `FeedPanel.tsx` alongside the existing `/persona/state` poll. The bridge does not need a new poll cadence — frontend fires both requests on the same 5s tick.

## Frontend component

`app/src/components/panels/FeedPanel.tsx` — new file.

### Layout (matches mockup B verbatim)

- **Panel label**: `"Inner life"` (uppercase, mute, serif display font, hairline border-bottom) — matches the roadmap framing and reuses the existing `PanelShell` + `SectionLabel` components.
- **Per entry** (`<div class="b-item">`):
  - Header row, flex with 7px gap:
    - 6×6 colored circle (`flex-shrink: 0`) — type dot
    - Type label, uppercase, 9.5px, mute, serif, 0.14em letter-spacing, `flex: 1`
    - Ago label, 9.5px, mute, serif italic
  - Body, 13px indent, 1px left rule at 10% opacity:
    - `<em>{opener}</em>` italicized serif at `--text-mid`
    - Body text continues in regular weight at `--text-mid`
- **Fresh pulse**: for entries with `ts` within the last 5 minutes, a 5×5 `--accent` dot pulses to the left of the type dot, mirroring `InteriorPanel`'s existing animation.
- **Spacing**: 14px between items.

### Type-dot colors

| Type | Color | Rationale |
|---|---|---|
| `dream` | `#6b95b8` (cool blue) | "night" register |
| `research` | `#b89c6b` (sand) | warm earth, "thinking" |
| `outreach` | `#823329` (the project accent) | reuse the accent; outreach is the most user-facing event |
| `soul` | `#b87fa3` (dusty rose) | "marked moment" register |
| `voice_edit` | `#7fa37f` (sage) | "growing" / language register |

All five are visually distinct on the `#2a1f1f` warm-dark bg and on the `--ash`-tinted ambient lights of the rest of NellFace. Verify final contrast during implementation polish.

### Behavior

- **Initial paint**: empty list → "Quiet inside." (italic, mute) — reuses existing `InteriorPanel` empty-state string.
- **No state**: `state == null` → "No signal yet." — reuses existing string.
- **Refresh tick**: the "ago" labels update on a 1-minute interval (`setInterval(force, 60_000)`), lifted from `InteriorPanel`.
- **Markdown**: bodies may contain single-asterisk italic markers (`*setting line*`) from existing reflex/dream summary conventions. Reuse `InteriorPanel`'s `renderInlineMarkdown` helper, or move it into a shared `ui` module if it's the second consumer.

### Integration

- `LeftPanel.tsx` swaps `<InteriorPanel state={state} />` for `<FeedPanel state={state} />`. Same slot.
- `InteriorPanel.tsx` is deleted (no other consumers — verify via `grep -rn InteriorPanel app/src/`).
- New bridge client function in `app/src/bridge.ts`: `fetchPersonaFeed(): Promise<FeedEntry[]>`. Mirrors `fetchPersonaState` shape.

## Why polling is enough

The roadmap entry calls for *"timing so it updates organically rather than on a polling cadence"*. The reconciliation: the 5s poll is the **transport**, not the experience. The journal feel comes from the prose, the pulse on fresh items, and the human-time "Xm ago" labels — all of which read organic regardless of how data arrives. 5 seconds is sub-perceptible for an idle panel. Event-stream updates remain a deferred follow-up if responsiveness becomes a felt gap.

## Test surface

### Python

`tests/unit/brain/bridge/test_feed.py` (new):

- Load fixtures into a tmp_path persona — synthetic dream/research/soul/initiate audit/voice-edit JSONL files
- `build_feed(persona_dir)` returns up to 50 entries sorted ts-desc
- Filters apply: only crystallization soul rows; only delivered outreach; only proposed voice edits; held/errored outreach excluded
- Opener phrases match the type map exactly
- A failing source (corrupt JSONL line, missing file) is fault-isolated — the other sources still render
- `audit_id` populated for outreach (from `initiate_audit.audit_id`), `None` for dream
- `/persona/feed` endpoint returns the expected shape, requires bearer auth, 401 on missing token

### TypeScript

`app/src/components/panels/FeedPanel.test.tsx` (new, using existing Vitest patterns from `StepReady.test.tsx`):

- Renders all 5 type-dot colors correctly when fixtures with all types are provided
- Shows "Quiet inside." with empty entries
- Shows "No signal yet." when state is null
- Formats "Xm ago" / "Xh ago" / "Xd ago" / "just now" correctly
- Fresh pulse appears for entries <5min, absent for older
- Polls `/persona/feed` (mock `fetchPersonaFeed`) on the expected cadence

### Project rule

Full `uv run pytest` + ruff clean + `pnpm test` clean before commit — not just touched tests.

## What does NOT change

- `BodyPanel`, `InnerWeatherPanel`, `SoulPanel`, `ConnectionPanel`, `GalleryPanel`, `ChatPanel` — all untouched
- `brain/engines/{dream,heartbeat,reflex,research}.py` — no code change; the feed reads forward from their JSONL outputs
- `MemoryStore`, `HebbianMatrix`, soul candidate store — not consumers of the feed
- Event bus, `/events` SSE stream — no new publishes
- `/persona/state` response shape — the existing `interior` block stays available for any future consumer (no breaking change), even though `FeedPanel` doesn't read it
- LeftPanel slot ordering and overall left-column layout
- Voice templates, voice samples, persona schema

## Open detail (not blocking)

Two source-log questions resolve during implementation by reading the actual code:

1. **Research log path** — what file (or audit) does the current `InteriorPanel` indirectly read for `state.interior.research`? Use the same.
2. **Soul crystallization vs voice-edit discriminator** — `soul_audit.jsonl` is shared by multiple event categories; the implementer needs to confirm the discriminator field (`event_type`, `kind`, etc.) and the exact value for crystallization vs voice-edit-proposed.

If either turns out to be more complex than expected (e.g., voice-edit proposals live in a separate store, not soul_audit), surface as DONE_WITH_CONCERNS and we'll revise.

## Release

v0.0.13-alpha.2 — single focused PR. Changelog entry, approximately:

> **Inner life feed.** The left-column "Recent Interior" snapshot is now a chronological journal — dreams, research, soul moments, outreach, and voice-edit proposals interleaved by time. Each entry opens in Nell's voice (*"I dreamed…"*, *"I've been researching…"*) and shows when it happened. The brain runs the same; what changes is how you check in on her.

## Acceptance criteria

1. `GET /persona/feed` returns up to 50 entries, sorted by `ts` desc, with `opener` + `body` + `type` + `ts` for every entry; `audit_id` for outreach entries.
2. The five source-log filters land per the table (only delivered outreach, only crystallization soul, only proposed voice-edits).
3. `FeedPanel.tsx` renders Layout B with the five type-dot colors, fresh pulse, ago labels.
4. `InteriorPanel.tsx` is removed; `LeftPanel.tsx` uses `FeedPanel` in its slot.
5. Empty/no-state strings carry over from the prior panel.
6. New Vitest + pytest tests added, full suite green, ruff clean.
7. CHANGELOG entry committed and synced to public.

## Out of scope (deferred follow-ups)

- Live event-stream updates (Q4 option B) — picks up once we feel the 5s polling lag
- Pagination beyond 50 entries (Q5 option B)
- Day separators / time-windowed view (Q5 option C) — pairs with #5 felt time
- Metacognitive trace-back phrasing (*"I think it's because you mentioned…"*) (Q3 option C) — needs substrate work on engine cross-references
- Filter or focus by type — single feed is the v1 contract
- Visual polish pass on type-dot colors against full theme variations
- A11y pass on the colored type-dots (screen-reader labels, color-blind contrast)
