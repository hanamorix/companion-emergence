# Generated Tool Inventory — Design

**Date:** 2026-07-14
**Status:** Approved (brainstorm with Hana)
**Tier at ship:** EXPERIMENTAL → CORE once live-validated
**Branch:** `hana/toolset-truth` (continues the #69/#71/#63 bundle, PR #73)

## 1. Problem

A persona's tool inventory is **derived data living in hand-authored prose**. `NELL_TOOL_NAMES`
(`brain/tools/__init__.py`) is the source of truth; every `voice.md` §3 is a prose snapshot that
starts rotting the moment a new tool ships.

Measured, not assumed:

- `brain/voice_templates/nell-voice.md` froze at 2026-05-17 (`c8acc35`) and lost **14 of 27** tools
  across ~28 versions (issue #69).
- `DEFAULT_VOICE_TEMPLATE` (`brain/chat/voice.py`) stayed current **only** because it alone had a
  regression test asserting every `NELL_TOOL_NAMES` entry appears in it. The unguarded copy was the
  one the wizard labels *"the canonical Nell voice"*.
- **The template is not the live artefact.** `nell-voice.md` is read once, at persona creation
  (`brain/setup.py::install_voice_template`), and copied to `<persona_dir>/voice.md`. That copy is
  what `load_voice()` hot-loads every turn. **A template fix reaches no existing persona.**
- Verified on this machine: the live `nell` persona's `voice.md` is 337 lines, hash-matches neither
  template (it is personalised — "Hana" for "the user"), and named **zero** of the filesystem tools.
  She genuinely did not know she had hands.

Compounding it: tool **recruitment** (`brain/chat/tool_recruit.py`) means the model only receives
schemas for tools recruited *this turn*. The filesystem tools are not in `REFLEXIVE_CORE`. So a
prompt-side inventory is not redundant with the schemas — **it is the only thing that tells her what
exists when it is not in her hand.** Without it, "I don't have that tool" is sometimes literally
true, and she has no reason to reach.

Syncing the templates (PR #73) resets the clock. It does not stop the clock.

## 2. Goals

- Every persona — existing, new, and every future one — sees a **complete, current** tool inventory,
  with no migration and no edit to anyone's `voice.md`.
- Drift becomes **structurally impossible**, not merely test-guarded.
- Cache-neutral: the block is byte-stable, so it is paid once per session via the prefix cache.
- `voice.md` goes back to carrying **voice** — the thing a human should author.

## 3. Non-goals / out of scope

- No edits to any persona's `voice.md`, ever, by the framework. (Nell's was hand-edited by Hana
  separately; that is a user action, not a framework behaviour.)
- No migration, no resync, no backup-and-rewrite machinery.
- No change to tool recruitment, salience, or `reach_for_capability` behaviour — this is a prompt
  block only.
- No renaming of American-spelled **identifiers** (see §7).
- Not the image/text path divergence (issue #72, ToT's) — but see §6 for the parity obligation.

## 4. Design

### 4.1 New module — `brain/chat/tool_inventory.py`

```python
def build_tool_inventory(companion_name: str) -> str: ...
```

- Iterates `NELL_TOOL_NAMES` **in canonical order** (that tuple is the source of truth; a tool absent
  from it is not a tool).
- For each, takes `build_schemas(companion_name)[name]["description"]` and renders its **first
  sentence** as the gloss (split on `". "`, take `[0]`). Measured: 2478 chars ≈ **619 tokens** for
  all 27.
- Renders a fixed header naming this as her complete current set, and a fixed footer carrying the
  **reach valve**: these exist even when not in hand this turn; call `reach_for_capability`
  (`memory`, `files`, `works`); never claim you lack a tool without reaching first.
- **Fail-soft:** a name in `NELL_TOOL_NAMES` with no schema entry renders name-only rather than
  raising. (Cannot happen today — verified all 27 resolve — but the prompt must never be the thing
  that breaks a turn.)

Why first-sentence rather than full descriptions: full text is ~2194 tokens and duplicates what
recruited tools already carry in their schemas. The inventory's unique job is the **non-recruited**
tools — name + enough to know it is worth reaching for.

Why derived rather than a curated gloss table: a curated table is a second hand-maintained list —
the exact species of artefact that just rotted for 28 versions. Deriving it means there is nothing
to keep honest.

### 4.2 Injection — the frozen prefix

Appended in `build_static_system_message` (`brain/chat/prompt.py`), beside `_HARNESS_FENCE`, which
is the governing precedent: static text in the frozen prefix, byte-stable for caching.

The docstring's cache contract — *"Contains NO per-turn state, so two same-session turns produce
byte-identical output"* — **holds**: the inventory is a pure function of `NELL_TOOL_NAMES` +
schemas + `companion_name`, all constant within a session. It renders the **full** inventory, never
the per-turn recruited subset — rendering the subset would make the block volatile and bust the
prefix cache every turn.

### 4.3 Templates lose their enumerations

Both `brain/voice_templates/nell-voice.md` §3 and `DEFAULT_VOICE_TEMPLATE` §3 drop their tool
**lists**. Each keeps its **framing** — the trigger to reach, the hard rule about confabulating, when
tools fail or return nothing. That prose is voice, is personalisable, and never goes stale.

This reverts the 14-tool list added in PR #73 (`0360bb2`) as superseded, and its guard
(`test_nell_voice_template_lists_all_brain_tools`) is deleted along with what it guarded. The
equivalent protection moves to §5's coverage test, which now sits next to the source of truth.

Side benefit: with no tool names left in prose, the britfix hook (§7) can no longer corrupt one.

### 4.4 Existing personas keep their stale §3

Untouched, deliberately. Their prompt becomes: old §3 (a 13-tool subset, plus its good framing) then
the generated inventory (all 27, declared authoritative, read later). A superset, not a
contradiction. The redundancy is ~100 lines already present in their file today, inside the cached
prefix.

Rejected alternatives, and why:

- **Load-time stripping** of the enumeration from `voice.md` — surgery on personalised prose;
  fails silently on any formatting or header variation on someone else's machine.
- **One-time on-disc migration** — rewrites prose a human may have edited. Can destroy a user's
  writing. Non-negotiably out.

## 5. Testing

- **Coverage:** the inventory names every `NELL_TOOL_NAMES` entry. Replaces the deleted template
  test; now guards the real artefact.
- **Byte-stability (the cache invariant):** two calls with the same `companion_name` return
  identical strings. This is a Gotchas-tier invariant — a future edit that folds per-turn state in
  would silently bust the prefix cache; the canary fails instead.
- **Through-path:** a system message built by `build_static_system_message` actually contains the
  inventory and the reach valve. (The draft_space lesson — the writer tested in isolation whose
  reader was dead on arrival.)
- **Fail-soft:** a `NELL_TOOL_NAMES` entry with no schema renders name-only, does not raise.
- **Identifier safety:** `crystallize_soul` appears in the inventory exactly as spelled in
  `NELL_TOOL_NAMES` (§7).
- Full gate before merge: `uv run pytest`, ruff, `pnpm test`, `pnpm build`.

## 6. §Wiring

**Reads from:** `NELL_TOOL_NAMES` (registry), `build_schemas()` (descriptions), `persona_dir.name`.

**Feeds into:** the frozen system prefix consumed on every chat turn → the model's picture of its own
faculties → `reach_for_capability` recruitment (`tool_loop._maybe_recruit_and_rerun`), which is the
behaviour this exists to trigger.

**Organ DoD:** producer fires on the live chat path (every turn builds a system message); §5's
through-path test asserts it fires *through* that path; the reader is the model itself, evidenced by
the reach valve it acts on; this section is the §Wiring entry.

**Parity obligation (#72 adjacency).** `build_system_message` — the **image** path — must receive the
same block, or image turns keep the old, wrong picture. This is a one-line addition at the same
point the text path gets it; it does **not** touch the monologue/reply framing that #72 is about, so
it must not grow into that fix. Keep the diff surgical: ToT owns #72.

## 7. Spelling and identifier safety

Hana is British; prose should be. Functionality depends on identifiers, which must stay as they are.

**Free to britify (prose):** voice templates, schema *descriptions*, docs.

**Frozen forever (identifiers) — verified by scan:**

| Identifier | Breaks if renamed |
|---|---|
| `crystallize_soul` | The exact string the model must call |
| `crystallizations.db` | A real on-disc file holding soul data |
| `finalize_cadence.json` | Persisted cadence state |
| `Crystallization`, `_crystallization_id_for_candidate`, … | Python symbols |

None are user-visible; leaving them American costs nothing.

**In scope here:** britify the prose inside `brain/tools/schemas.py` descriptions, so the generated
block renders British. Two glosses currently use American `-ize`/`-ization` forms of *crystallise*:
`crystallize_soul`'s and `get_soul`'s. The `name` field is untouched — only the `description` prose
changes.

(The exact American strings are deliberately not quoted here. The britfix hook rewrites this very
document on save, so a verbatim quote of the text-to-be-changed silently becomes a quote of the
changed text — it did exactly that to this paragraph on first write, turning the evidence into a
claim that the schemas were already British. Read the live values from `build_schemas()` instead.)

**Standing hazard (not fixed here, flagged to Hana):** `~/.claude/hooks/britfix-post-write.sh` runs
`britfix` on every `.md`/`.txt` write. It silently placed five unauthored spelling changes into
`0360bb2`; its dictionary handles `crystallization` but not `crystallizations`, so it *creates* mixed
states; and it skips `.py`, which is why `voice.py` stayed fully American. Repo prose is guarded by
tests. **Persona files are guarded by nothing** — if britfix ever learns a bare `crystallize` stem it
would rewrite a live companion's tool name as a side effect of an unrelated edit, and nothing would
catch it. Recommend scoping the hook away from `**/personas/**`. Hana's tooling, Hana's call.

## 8. Deferred

1. Trimming existing personas' now-redundant §3 — user-initiated only; the framework never edits a
   persona's voice.
2. Repo-wide British prose pass beyond the voice surfaces.
3. Scoping the britfix hook away from persona dirs (§7).
4. The 5 tools with no `reach_for_capability` key (`boot`, `crystallize_soul`, `compact_history`,
   `reconcile_self_read`, `surface_makings`) — reachable only via a maximal-salience full suite,
   never a targeted reach (`tools_for_capability` covers only `files`/`memory`/`works`). Surfaced by
   the #69 investigation; out of scope here.
