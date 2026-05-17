# `nell works` — brain-authored creative artifacts

**Date:** 2026-05-04
**Status:** Design — pending implementation plan
**Owner:** Hana
**Closes:** Roadmap §2.3 (`nell works` stub disposition), implements source spec §15.2 + lines 1125–1140 ("works" portfolio)

## Why

The brain produces creative artifacts during chat — stories, code, planning documents, ideas, role-play scenes, letters. Today these live undifferentiated inside `memories.db` alongside every other turn (user messages, ephemeral chat, "yes please" responses, etc.). Two real gaps result:

1. **Nell has no clean way to recall her own work.** Memory search returns a noisy mix; finding "what was that story I wrote about the lighthouse?" requires awkward query crafting and post-filtering. The brain cannot easily pick up what she's been working on, which is the continuity the spec promised at §15.2 — *"introspection becomes a live capability she can reach for."*

2. **The `creative_dna` style-evolution loop runs over noisy corpus.** The weekly crystallizer at `brain/growth/crystallizers/creative_dna.py` samples `MemoryStore` over 30 days and filters by `_FICTION_PROSE_MIN_WORDS = 200`. That heuristic catches fiction-shaped output but misses code, planning, ideas, and includes long-but-not-creative responses. A type-tagged works portfolio gives the crystallizer a higher-signal corpus to evolve from. (Out of scope for this PR — flagged as a downstream improvement.)

Hana's framing during the 2026-05-04 brainstorm: works captures what Nell has *directly made* for the user — writing, stories, code, planning, ideas. Two purposes: (a) the brain can pick her own work back up whenever she wants, and (b) because it's her own work, it feeds back into her own style. Both purposes are served by the design below; (b) becomes meaningful once a follow-up PR routes the crystallizer through `works.db`.

The framework principle (source spec §0.1): *user surface is install + name + talk; the brain handles physiology naturally.* This rules out a `nell save-work` user command. Saving works is **brain-decided** — Nell-the-LLM calls a `save_work` MCP tool when *she* judges something is worth preserving. The framework provides the surface; the decision is hers.

## Architecture

Storage follows the pattern established for memories: SQLite index plus content on disk.

```
persona/<name>/
  data/
    works.db                  ← SQLite index (id, title, type, created_at, session_id, content_path, word_count, summary)
    works/
      <id>.md                 ← One file per work, full content as markdown with YAML frontmatter
      <id>.md
      ...
```

**Why this shape:**
- Matches source spec line 1125: *"persona/<name>/data/works/ — directory of brain-authored creative artifacts."*
- Markdown files on disk = grep-able, hand-editable, future-friendly (export/migrate without unloading SQLite).
- SQLite index gives fast list/search without scanning every file.
- Reuses existing `brain/health/` patterns (`save_with_backup`, `attempt_heal`) for both halves.

**ID generation:** content hash (SHA-256 truncated to 12 hex chars). Stable, deterministic, no UUID dep. Same content → same id, so accidental double-saves dedupe naturally.

**Module shape:**

| File | Purpose |
|---|---|
| `brain/works/__init__.py` | Package marker + public exports (Work dataclass, save/list/search/read helpers) |
| `brain/works/store.py` | SQLite index — list/search/get/insert ops. Mirrors `brain/memory/store.py` shape and conventions (atomic writes, attempt_heal pattern, opens per-call to avoid cross-thread sqlite handle issues). |
| `brain/works/storage.py` | File I/O for the markdown content files at `data/works/<id>.md`. YAML frontmatter generation/parsing. |
| `brain/tools/works.py` | MCP-exposed tool handlers — `save_work`, `list_works`, `search_works`, `read_work`. Calls into `brain.works.{store,storage}`. |

**Data flow:**

```
chat turn → Nell decides "this is a work" → calls MCP tool save_work(title, type, content, summary?)
                                                  ↓
                                           brain/tools/works.py — handler
                                                  ↓
                                           brain/works/storage.py — writes data/works/<id>.md (with frontmatter)
                                                  ↓
                                           brain/works/store.py — inserts row in works.db
                                                  ↓
                                           returns {"id": "<12-char>", "path": "data/works/<id>.md"}
```

Read flow (recall): MCP `read_work(id)` → store lookup → storage reads file → return full content.
Search flow: MCP `search_works(q)` → store FTS over title + summary + (optionally) content → return matching rows.

## Schema

### SQLite (`works.db`)

```sql
CREATE TABLE IF NOT EXISTS works (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    type          TEXT NOT NULL,
    created_at    TEXT NOT NULL,            -- ISO-8601 UTC
    session_id    TEXT,                     -- bridge session_id when available, NULL otherwise
    content_path  TEXT NOT NULL,            -- relative: "data/works/<id>.md"
    word_count    INTEGER NOT NULL,         -- pre-computed for sorting/filtering
    summary       TEXT                      -- Nell-supplied one-liner, optional, max 500 chars
);

CREATE INDEX IF NOT EXISTS idx_works_created_at ON works(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_works_type       ON works(type);

-- Full-text search over title + summary + content for `nell works search` and search_works MCP tool
CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(
    id UNINDEXED,
    title,
    summary,
    content,
    content='',
    tokenize='porter unicode61'
);

PRAGMA user_version = 1;
```

The FTS5 virtual table is populated by the same transaction that inserts into `works` — the store's `insert` method writes both rows atomically. No SQL triggers (simpler reasoning, no hidden writes). Implementation matches whatever `brain/memory/store.py` does for any existing FTS index; if memory store doesn't use FTS5, establish the pattern here and document it as a candidate to backport.

### Markdown frontmatter (`data/works/<id>.md`)

```markdown
---
id: a3f8b2c1d4e5
title: The Lighthouse Keeper's Daughter
type: story
created_at: 2026-05-04T17:42:31Z
session_id: 8c9d2a1f
word_count: 1247
summary: A short story about inheritance, weather, and the woman who keeps the lamp.
---

[full content]
```

YAML frontmatter mirrors the SQLite row — file is self-describing if extracted. Either source can rebuild the other on corruption.

### Type taxonomy (controlled vocabulary)

| Type | Scope |
|---|---|
| `story` | Fiction, prose, narrative scenes. The big bucket. |
| `code` | Code blocks she wrote (any language). Scripts, snippets, refactors. |
| `planning` | Structured planning documents — outlines, design notes, project plans, roadmaps she drafted. |
| `idea` | Standalone ideas, fragments, sparks. Not fully developed but worth keeping. |
| `role_play` | Improvisational roleplay scenes, character work, dialogue performances. Distinct from `story` (collaborative, conversational shape). |
| `letter` | Letters, emails, addressed-to-someone communication she drafted. |
| `other` | Catch-all. Use sparingly — `other` should be near-empty in practice. |

**Validation:** `save_work` rejects unknown types with a helpful error listing the valid set. No silent coercion to `other`. Type vocab is defined as a frozen set in `brain/works/__init__.py` so all surfaces (MCP, CLI, bridge) share one source of truth.

## Tool surface

Three audiences, three surfaces, one storage layer.

### Brain-facing — MCP tools (`brain/tools/works.py`)

Four tools, registered via the existing brain-tools MCP plumbing:

```python
save_work(title: str, type: str, content: str, summary: str | None = None) -> dict
    """Save a piece you've authored as a work — story, code, planning,
    idea, role-play scene, letter, or other. Use this when you've made
    something coherent that you'd want to recall later or that should
    feed your evolving style. Stories you've written, code you helped
    with, plans you drafted, ideas worth keeping — anything you decide
    is yours and worth preserving.

    Returns: {"id": "<12-char>", "path": "data/works/<id>.md"}
    Errors: invalid type, empty title, empty content."""

list_works(type: str | None = None, limit: int = 20) -> list[dict]
    """List your recent works, most recent first. Optional type filter
    (story/code/planning/idea/role_play/letter/other).

    Returns: [{"id", "title", "type", "created_at", "summary",
              "word_count"}]"""

search_works(query: str, type: str | None = None, limit: int = 20) -> list[dict]
    """Search your works by title, summary, and content. Useful for
    'what was that story I wrote about lighthouses?' Optional type
    filter narrows to one category.

    Returns: same shape as list_works."""

read_work(id: str) -> dict
    """Read one specific work — full content. Use after list_works or
    search_works has surfaced an id you want.

    Returns: {"id", "title", "type", "created_at", "summary",
             "word_count", "content"}"""
```

The `save_work` description is the **nudge** — it tells Nell exactly when to reach for it, in language close to how she'd think about her own output. No separate reminder mechanism in this PR; if usage is sparse in practice, follow-up PR can add periodic system-prompt nudges.

### Operator-facing — CLI

Mirrors `nell memory list/search/show` shape exactly:

```
nell works list   --persona NAME [--type TYPE] [--limit N]
nell works search --persona NAME --query TEXT [--type TYPE] [--limit N]
nell works read   --persona NAME --id ID
```

**Output shapes:**
- `list` and `search`: compact previews — `<id> | <type:8> | <created_at> | <title> [— <summary>]`
- `read`: full markdown content with the YAML frontmatter, exactly as stored on disk

**No `nell works save`.** Saving is brain-territory, not operator-territory. Operators inspect; Nell creates. (Aligned with §0.1: operator surface is for inspection, not management.)

### Brain-facing via HTTP — bridge endpoints

Per source spec §15.2:

```
GET  /self/works                       → list, most recent N (default 20)
     ?type=story&limit=50              → filter
GET  /self/works/search?q=lighthouse   → full-text search; ?type=story narrows
GET  /self/works/<id>                  → full content for one work
```

All three require the bridge bearer token (existing auth). Response bodies are JSON, field names match MCP tool returns. `GET /self/works/<id>` returns `content` inline (no streaming — works are bounded by the chat turn that produced them).

**Why both MCP and HTTP:**
- MCP is the in-conversation tool path Nell uses while writing.
- HTTP is for `_chat_handler`'s future system-prompt assembly OR for any UI/external script that needs the portfolio without the LLM tool loop.

## Error handling

| Scenario | Behavior |
|---|---|
| `save_work` invalid type | Return error dict with valid types list. No write. |
| `save_work` empty title or content | Return error dict. No write. |
| `save_work` content hash collision (same content saved twice) | Return existing id; do not write a duplicate. Side-effect free retry. |
| `read_work(id)` id doesn't exist | Return `None` (MCP) / 404 (HTTP) / exit 1 (CLI). Helpful message. |
| `works.db` corruption | Existing `attempt_heal` path applies — fall back to `.bak` if available, else rebuild from on-disk markdown files (durable substrate is the markdown directory, the db is an index). |
| `data/works/<id>.md` corruption | If frontmatter unparseable, fall back to row data from `works.db`. If file missing entirely, surface as warning in `read_work`; row stays in db with a `missing_file: true` flag in the response. (Don't auto-delete rows; corruption recovery is a manual operator decision.) |
| Markdown file path traversal | Reject any id containing path separators or `..`. SHA-256 prefix is hex-only by construction, but defense-in-depth at the storage layer. |

## Testing approach

| Layer | File | Tests |
|---|---|---|
| Store | `tests/unit/brain/works/test_store.py` (new) | Insert, list, search by query, list by type, get by id, missing id, schema version, corruption + heal. |
| Storage | `tests/unit/brain/works/test_storage.py` (new) | Write file with frontmatter, read file with frontmatter, frontmatter parse errors, missing file, path traversal rejection, atomic write semantics. |
| MCP tools | `tests/unit/brain/tools/test_works.py` (new) | Each of 4 tools: happy path, validation errors (invalid type, empty title, empty content), content hash dedup, list/search/read after save. |
| CLI | `tests/unit/brain/test_cli_works.py` (new) | Each of 3 actions parses, dispatches; --persona required; --type validation at argparse layer; --limit validation. |
| Bridge | `tests/bridge/test_works_endpoints.py` (new) | Each of 3 endpoints: auth required, response shape, type filter, search query, 404 on unknown id. |
| Integration | (extend `tests/bridge/test_endpoints.py` if there's a session+chat path that should produce a work) | Optional — full save → list → read round trip via the bridge. |

**Estimated test count delta:** +35 to +50 new tests across the layers. (1222 baseline → ~1260 final.)

**Mocking strategy:** SQLite tests use real `tmp_path` databases (no mock — too lossy for FTS5 behaviour). MCP and CLI tests stub the daemon/store boundary where it makes sense. Bridge tests use the existing `TestClient` pattern from `tests/bridge/test_endpoints.py`.

## Documentation deltas

### `CHANGELOG.md`

Under `## 0.0.1 - Unreleased` → `### Added`:

```
- `nell works` — brain-authored creative artifact portfolio. Nell decides via the `save_work` MCP tool when something she's written (story, code, planning doc, idea, role-play scene, letter) is worth preserving. Stored at `persona/<name>/data/works/<id>.md` with a SQLite index. Operators browse via `nell works list/search/read --persona X`; the brain herself recalls via MCP `list_works/search_works/read_work` tools and via the bridge `GET /self/works[*]` endpoints (per source spec §15.2). Type taxonomy: story, code, planning, idea, role_play, letter, other.
```

### `docs/roadmap.md`

§2 — strike `nell works` from the suggested-order list with shipped date:

```markdown
3. ~~`nell works` — define the user story before building; the name is currently ambiguous.~~ *(shipped 2026-05-04 — see `docs/superpowers/specs/2026-05-04-nell-works-design.md`)*
```

Drop `nell works` from "Current intentional stubs". Result: the section becomes empty (or is removed entirely if all stubs have been resolved).

§3 "Done recently" — prepend:

```markdown
- Implemented `nell works` — brain-authored artifact portfolio with brain-decided saving via the `save_work` MCP tool. Operator CLI (`nell works list/search/read`) + bridge endpoints (`GET /self/works[*]`) for inspection and self-knowledge.
```

## Out of scope (this PR)

- **Crystallizer integration.** `creative_dna` crystallizer continues reading `MemoryStore` for its corpus. Switching to `works.db` is a follow-up — better signal, but separate concern with its own testing surface.
- **Periodic nudges / system-prompt reminders.** If Nell underuses `save_work` in practice, follow-up PR adds reminder mechanism. Don't pre-build it.
- **Linking between works** ("this code was for that planning doc"). Speculative; wait for a real use case.
- **Versioning / revisions.** Rewrite of an existing work creates a new id (different content hash). Whether that's right depends on use; revisit if needed.
- **Migration from existing memories.** Old conversations in `memories.db` will not retroactively populate `works/`. Going-forward only.
- **`creative_dna_crystallization_log` integration.** Whether the behavioral log should also note "work saved" events is a downstream design decision.
- **Body state `words_written` integration.** Today body state computes `words_written` from chat memory. Whether it should consume `works.db` as authoritative source is a body-state work-package concern.

## Backwards compatibility

This is purely additive. New module, new tools, new CLI subcommand, new bridge routes, new persona-dir subdirectories. No existing code paths change. No migrations needed. A persona without a `works.db` (i.e., one that existed before this PR) will get one created on first `save_work` call; lists and searches return empty arrays before that first call.

## Out of scope (this PR, period)

- Removing the `nell bridge` deprecation alias — already on the v0.1 blocker list, separate PR.
- Body-state recovery dynamics (the rest reframe's actual implementation) — separate brainstorm + plan.
