# Felt-time anchors from the real writers (#676)

Build after #385 merges.
Citations `[Dn]` name the sheet line each statement comes from; `[DERIVED from Dn]` marks a mechanical consequence.

## Intent

Felt time is meant to notice the marked moments of the persona's life (dreams, growth, soul crystallisations) and to show them in her sense of time: when each last happened, and how the stretch since the newest one has felt. Today none fire: `brain/felt_time/anchors.py::_SOURCES` reads files and fields that no writer produces (#676). This spec points felt time at the real writers.

**In scope:** felt time notices dreams, growth and soul moments, and the current stretch restarts on them [D1].
**Out of scope:** narrative arcs and grief from these moments. They wait for a second spec, which first gives each moment the lived age felt time saw it at [D1, D8]. This spec keeps them off explicitly [D9].

## Design

### Sources [D2, D3, D5]

| type | file | rows that count | time | label |
|---|---|---|---|---|
| dream | `dreams.log.jsonl` (written by `DreamEngine._log`) | every row | `timestamp` | text of the memory named by `dream_id` (below) |
| growth | `emotion_growth.log.jsonl` (written by `brain/growth/log.py::append_growth_event`) | `type` ends in `_added` (today: `emotion_added`, `arc_added`); `arc_pruned_by_brain` and every other type never count | `timestamp` | `name` |
| soul | `soul_audit.jsonl` (written by `brain/soul/audit.py::append_audit_entry`) | `crystallization_id` is a non-empty string and `dry_run` is exactly `false` [D3, D19]; defer, reject and parse-error rows never count | `ts` | `candidate_text` [D4] |
| arc | `arcs.log.jsonl` | unchanged | `ts_iso` | `title` |

`_SOURCES` is the one source map; nothing else keeps a copy. Narrative memory's copy is deleted [D9]. The loader's rebuild trigger (`state.py` `LOG_SOURCES` / `_newest_log_ts`, which today reads `dreams.log.jsonl`, `growth.log.jsonl`, `soul.log.jsonl` and a `ts` key) is replaced: the loader rebuilds from logs when `extract_all` finds any counting row of dream, growth, soul or arc, each under its own counting rules and with parsed times, and otherwise cold-starts [D15]. A growth log holding only prunings therefore cold-starts, and a persona whose only logs are arcs rebuilds and gets its arc anchors back. A growth row with no `name` or a soul row with empty `candidate_text` is skipped, as today; only dreams keep an empty label [D17]. Each source reads its active file through the canonical JSONL reader, as today. Rotated archives (`dreams`/`emotion_growth` roll at 5 MB, `soul_audit` yearly) are not read [D2, D3, D5 name the active files]. Just after a rotation the active file can be empty: the stored anchor is kept (D7 only moves anchors forward), and a rebuild (D15) sees no row there.

### Dream labels [D5, D6]

The label of a dream comes first from `daemon_state.json` `last_dream.summary`, when that record's parsed timestamp is at or after the newest dream row's. The heartbeat writes that record in the same heartbeat, just after the log row, so in the normal case the label is there on the first tick and `memories.db` is not opened [D26]. Otherwise the label is the `content` that `MemoryStore.get(dream_id, bump=False)` returns on `<persona>/memories.db`; a faded memory's summary is used as returned [D5, D20]. The lookup runs first. Only when it returns `None` (a hard-deleted or unknown id), the row has no string `dream_id`, or `memories.db` cannot be opened on that tick, does the dream count with an empty label; a missing `memories.db` counts as unopenable and is never created, and the store case logs a warning [D6, D20].

A new dream's memory is still in the pending queue when felt time first sees the dream. `route_write` enqueues it, and the next heartbeat's consolidation commits it. So while a stored dream anchor has an empty label and is still the newest dream row, each tick looks it up again, and the label appears about one heartbeat later. A dream that is no longer the newest is never relabelled. A dream that deduplication dropped stays unlabelled [D23]. A leading `DREAM: ` is dropped from the label [D24].

`extract_all` and `scan_since` stay pure: they never open `memories.db`, and a dream anchor leaves them with its label unresolved (its `dream_id` travels with it, not persisted). The lookup runs only on the tick path, when a tick stores a dream anchor, so every stored dream anchor is resolved exactly once [D16]. A rebuild from logs (`_replay_from_logs`, inside the read-only loader) stores only arc anchors; dream, growth and soul are filled by the next tick under D7, so the loader never opens `memories.db` [D16]. An empty dream label is valid: #385's `_check_anchor` (`brain/felt_time/state.py`) requires only that `label` is a string, the pre-#385 loader does not check it, and `felt_time_now` returns `""` [D6, D16].

### Time order [D10, D12]

Every timestamp is parsed to an aware UTC `datetime` before it is compared or sorted. Dream and growth rows end in `Z`; soul rows and the tick end in `+00:00`; parsing makes them comparable. A row whose time does not parse, or parses with no timezone, is skipped like a corrupt row. A stored `Anchor.ts` keeps the row's original text [D10].

A row restarts the stretch only when its parsed time is strictly after the last tick's parsed time [D12]; which row is each type's anchor is decided separately, by D7. "Newest of each type" is decided by parsed time [D10].

### Which row is each type's anchor [D7, D12]

On every tick, for each of dream, growth and soul, felt time takes the newest counting row (by parsed time; on equal times the later row in its file wins) [D22]. It becomes that type's anchor when it is newer than the stored anchor of that type, or when none is stored; a stored anchor whose `ts` no longer parses counts as none [D7, D22]. So the persona's existing history fills each type on the first tick after the fix, and a row that arrived mid-tick (stamped before the tick's `now_iso`) is still taken on the next tick instead of being lost. Running the rule again changes nothing, so a restore from an older backup or an older brain's rewrite is harmless. No state field is added [D7].

### Pressure [D11, D12, D18]

A row taken under D7 restarts the current stretch (`pressure.apply_tick` with a non-empty `new_anchors`) only when its parsed time is strictly after the last tick's parsed time; several in one tick restart it once [D11, D12]. One exception: on the tick where her anchors go from holding no dream, growth or soul anchor to holding one, the stretch restarts once, whatever D12 says for that tick. Her saved counters have never been reset, so they would otherwise show months of activity under "since the dream" [D25]. A row taken from older history does not restart it. With no last tick (a fresh persona, or the first tick after a rebuild) no row of any type, arcs included, restarts it, and the stretch counts from that tick's own heartbeats and turns [D12]. Arc rows keep today's fold and labels, compared by parsed time and restarting the stretch under the same rule; while D9 holds, the narrative pass opens no new arc [DERIVED D18]. Horizon buckets and lived age are unchanged.

### Prompt and tool [D6, D13]

`brain/felt_time/prompt.py` keeps its wording byte for byte: `last dream`, `last crystallization` (growth), `last soul moment`, labels cut to 40 characters, except the dream line with no label [D13]. On the dream, growth and soul lines only, line breaks inside a label become spaces before the cut [D21]; arc lines are unchanged; the stored label and `felt_time_now` keep the full text. A dream with an empty label renders as `  last dream: 5 hours ago`, with no dash and no quotes [D6]. The block stays where it is: in `build_volatile_context` on the text path, and in the image-path `build_system_message` (both `brain/chat/prompt.py`) (I13).

### Arcs and grief kept off [D9]

In `_run_narrative_memory_pass` (`brain/bridge/supervisor.py`), narrative memory's copy of the source map, the `anchor_sources` dict, is deleted, so the pass takes no dream, growth or soul moments at all [D9]. What happens to the helpers that read through it (`_AnchorAdapter`, `_extract_seed_memory_ids`, `_adapt_anchors`) is left to the builder, provided the pass's anchor sweep yields nothing. It opens no arc and writes no grief from dream, growth or soul moments; the rest of the pass (membership and staleness of existing arcs) is unchanged [D9]. The second spec has narrative read felt time's one map [D9].

### New parts and the alternatives

- **Dream-memory lookup.** Reuses `MemoryStore.get(..., bump=False)`. Nothing new.
- No new store, file, field or process.

## Wiring

- **Reads from:** `dreams.log.jsonl` and dream memories in `memories.db` (DreamEngine), `emotion_growth.log.jsonl` (growth scheduler and crystallisers via `append_growth_event`), `soul_audit.jsonl` (soul review via `append_audit_entry`).
- **Feeds:**
  - the felt-time ambient block (chat volatile tail), via the `last …` lines and the `current stretch (since the …)` line;
  - the `felt_time_now` MCP tool (`anchors`);
  - the pressure counters, which reset on new moments.
- **Explicitly not fed:** narrative memory (arcs, grief) [D9].
- **Tier:** `felt_time` stays CORE in `docs/maturity-manifest.md`. This is a repair of an existing organ, and its producer (the supervisor felt-time tick) already fires live.

## Invariants

| # | Invariant | holds / n.a. | where |
|---|---|---|---|
| I1 | install + name + talk | holds | no setting added |
| I2 | reads and feeds the loops; §Wiring | holds | Sources, Wiring |
| I3 | organ DoD; tier | holds | Wiring (CORE repair); Acceptance D14 |
| I4 | store API only | holds | Dream labels (`MemoryStore.get`) |
| I5 | additive state, defaults | holds | no state field added (Which row is each type's anchor); an empty dream label is a value, not a field |
| I6–I12, I15 | cadence, autonomy, body, gates, salience, bridge, provider, overlay | n.a. | not touched |
| I13 | stable prefix; volatile tail | holds | Prompt and tool (block placement unchanged on both paths) |
| I14 | cross-platform | holds | no new text read: logs go through the canonical JSONL reader, labels through `MemoryStore` |
| I16 | canary per physiology invariant | holds | Acceptance D9, D10/D12, D14 |
| I17 | no real names | holds | tests use temp `KINDLED_HOME` fixtures |

## Acceptance criteria

Every test runs in a temp `KINDLED_HOME` and drives the real writers. The live persona is never a fixture.

- **D2:** an `emotion_added` row and an `arc_added` row each become the growth anchor, labelled by `name`. An `arc_pruned_by_brain` row never does.
- **D3, D4:** a crystallised, non-dry-run audit row becomes the soul anchor, labelled by `candidate_text`. Accept-without-id, dry-run, defer, reject and parse-error rows never do.
- **D5:** a dream run through `DreamEngine` with `FakeProvider` becomes the dream anchor, labelled with the dream memory's content. The memory's recall count is unchanged.
- **D23:** a dream written by `DreamEngine` is stored with label `""` on the same heartbeat's tick. After the next heartbeat's consolidation, the following tick fills the label, with no `DREAM: ` prefix [D24]. An older dream with an empty label is never relabelled.
- **D25:** a persona whose saved counters are high and whose anchors are empty gets its anchors filled from history, and the stretch restarts once on that tick.
- **D6:** with the dream memory deleted, the dream still becomes the anchor with label `""`. The prompt line is exactly `  last dream: <ago>`, where `<ago>` is today's hours-or-days form.
- **D19:** a crystallised row with `crystallization_id` `""`, or with `dry_run` missing, never becomes the soul anchor.
- **D20, D21:** a faded dream memory labels with its summary; a row without `dream_id` counts unlabelled; with `memories.db` absent, no file is created; a label with a line break renders on one line.
- **D22:** two rows with equal parsed times: the later in the file wins; a stored anchor with an unparseable `ts` is replaced on the next tick.
- **D7, D12:** a persona with saved state and only older rows gets the newest of each type as its anchors on the next tick, with pressure counters not reset; a second tick changes nothing. A row stamped before the last tick but absent from it is taken on the next tick without a reset. A fresh persona's first tick takes the newest rows and restarts the stretch once, because its anchors first fill then (D25).
- **D9:** with real dream, growth (with `evidence_memory_ids` naming real memories) and soul rows present, `_run_narrative_memory_pass` opens no arc, appends nothing to `arcs.log.jsonl`, and writes no grief memory. The growth row is the guard: an adapter restored with the corrected file names would seed an arc from it, so the test fails.
- **D10, D12:** a `Z` row and a `+00:00` row in the same second order correctly when one has no fraction. A row with no timezone is skipped.
- **D11:** a new soul row after the last tick restarts the stretch, and three in one tick restart it once. **Measured pre-ship (guarded-change):** a replay of the reference persona's real log timestamps, read-only, reports the distribution of stretch lengths, so the owner can see how long stretches would run.
- **D13:** the prompt block's text for fixed state is byte-identical to today's apart from the dream-fallback line (D6) and D21's line-break rule.
- **D15, D16:** with the state file and its backups missing and real dream rows present, `load_or_recover` rebuilds from the logs instead of cold-starting, stores no dream, growth or soul anchor, and does not open `memories.db`; the next tick fills all three. With only `arc_pruned_by_brain` growth rows it cold-starts; with only arc rows it rebuilds with those arc anchors.
- **D16, D6:** with `memories.db` unopenable, a new dream becomes the anchor with label `""` and a warning is logged; after the store recovers, the next tick fills the label while that dream is still the newest [D23].
- **D17:** a soul row with empty `candidate_text` never becomes the soul anchor.
- **D14:** for each source, the real writer, then the supervisor's felt-time tick, then `_build_felt_time_block` shows its `last …` line.

## Left to the builder

- Where parsing lives: a helper in `anchors.py`, or sorting by a parsed key, provided every comparison in `tick`, `_replay_from_logs`, `scan_since`, the D15 rebuild trigger, `prompt.py`'s newest-anchor pick for the `since the …` line and `tool.py`'s `hours_ago` uses it [D10]. Parsed comparison also settles stamps of different fractional precision.
- How a dream anchor carries its `dream_id` from `extract_all` to the storing step without persisting it.
- A `last_tick_ts` that does not parse (only felt time writes it) is treated as no last tick under D12. A stored anchor whose `ts` does not parse when `prompt.py` or `tool.py` read it: skip that anchor's line, as D22 does on the tick path.
- How the store is opened for the dream lookup (once per tick that needs it, closed after).
- Merge order with #655, which edits `load_or_recover` and `FeltTimeState` in `state.py`, as D15 does here.

## Deferred

- **Known states, accepted (guarded-change stage 3):**
  - dream and soul labels are stored in full in `felt_time_state.json` (and its backups) and returned in full by `felt_time_now`; only the prompt cuts at 40;
  - a newest dream that deduplication dropped, or whose consolidation is deferred until a newer dream arrives, stays unlabelled. While it is the newest, it is looked up on each tick.

- **Narrative arcs and grief from dream, growth and soul moments** need the second spec. It builds the per-moment lived-age stamp together with its reader and reconnects narrative memory to `_SOURCES` [D1, D8, D9]; what it does with moments from before it ships, and with soul's repeated labels, is the second spec's decision.
