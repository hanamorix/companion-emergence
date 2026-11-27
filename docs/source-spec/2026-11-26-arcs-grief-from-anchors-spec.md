# Narrative arcs and grief from felt-time moments

Build after the #676 build (spec `docs/source-spec/2026-11-26-676-felt-time-anchors-spec.md`, draft #824), which builds on #385.
Citations `[Dn]` name the decision-sheet line each statement comes from; `[DERIVED from Dn]` marks a mechanical consequence.

## Intent

The #676 spec makes felt time notice dreams, growth and soul moments, and keeps narrative arcs and grief off. This spec turns them on [D1]:
- an arc opens from a real dream, growth or soul moment and gathers the memories related to it;
- while it is open, it makes her time feel heavier and slows the fading of her memories, in proportion to how much its members matter;
- it closes, with grief scaled the same way, only when its thread has truly gone quiet in her lived time.

Today none of this happens on a real persona. Arcs never open, because anchors never fired (#676) and the adapter reads seed keys no dream or soul row has. And an open arc weighs 0, because its emotion weight is set only at close.

**Out of scope:** moments from before this ships [D2]; any change to forgetting's mechanism [D5]; new grief kinds or emotion representations [D15]; any setting (I1).

## Design

### Where moments come from [D13, D2, D3, D4]

`_run_narrative_memory_pass` (`brain/bridge/supervisor.py`) gets its anchor sweep back. It reads moments through felt time's one source map (`brain/felt_time/anchors.py::_SOURCES`, `extract_all`), with #676's counting rules and parsed times [DERIVED D13]. It takes only dream, growth and soul moments, never `arcs.log.jsonl` rows [DERIVED D18].

Because #676 keeps `extract_all` pure, a dream moment arrives with its label unresolved. The pass resolves it itself, as #676 does: the label is the content of `MemoryStore.get(dream_id, bump=False)`, with a leading `DREAM: ` dropped (#676 D24) [D18].

A new dream's memory stays in the pending queue until the next heartbeat's consolidation promotes it, and this pass can run inside that window (a heartbeat can run between forgetting's items, and this pass follows forgetting). When `memories.db` has no memory for the dream, the pass uses felt time's stored dream label, if felt time's stored dream anchor is that same log row (same `source_ref` and `ts`) and its label is not empty. #676 fills that label from the heartbeat's last-dream record [D21]. When neither gives a label, the moment opens no arc, and the cursor still moves [D18, D21, D11]. That happens to a dream felt time never labelled: a CLI dream read before the next felt tick, a second dream pending behind a newer one, or a dream whose felt tick failed. A newest dream that deduplication dropped still has felt time's label, so it opens an arc whose `dream_id` member never resolves; that member adds no feelings [DERIVED from D21]. For seed ids, the pass reads the row named by each moment's `source_ref`. It uses the same reader and valid-row indexing as `extract_all` (`iter_jsonl_skipping_corrupt`), so a corrupt line never shifts a reference [DERIVED D20].

One new field in `arcs_state.json` (`brain/narrative_memory/state.py`), `anchor_cursor_iso`, is the anchor sweep's own cursor. It defaults to `None`, and `save_state` and `load_or_recover` read and write it by name [D19].

- **Sweep time.** This is the time read immediately before the pass reads the moment logs [D4].
- **Cutover.** It runs first. A pass that finds the cursor `None` (the first pass of this build) sets it to its sweep time, whether or not it knows her lived age, and takes no moments. So moments from before this ships never become arcs [D2].
- **After that,** a moment is taken only when its parsed time is after the cursor. At the end of a pass that knew her lived age, the cursor moves to that pass's sweep time [D2, D4].
- **Unknown lived age.** When it is unknown (`felt_time_state.lived_age_hours` is `None` or not finite) and a cursor exists, the pass takes no moments and leaves the cursor where it was. Those moments are taken on the next pass that knows it [D4].
- **Reading twice.** A row written between the sweep time and the read is read twice. The second read seeds the same memories, so it adds nothing to that arc, except re-adding a seed that the 50-member cap evicted [D4].
- **Missed row.** A row stamped before the sweep time but appended after the read (a growth event builds its timestamp just before appending) is missed. This window of milliseconds is accepted [D4].
- **Recovery.** `last_pass_ts_iso` keeps its role in the state-recovery check against `arcs.log.jsonl` and is not the anchor cursor [D4]. A rebuild from `arcs.log.jsonl` (`_replay_from_log`) sets `anchor_cursor_iso` to the newest parsed event time (`ts_iso`) in that log [D19]. That log is written only on open, member and close events, so its newest time can be days behind the last pass. A rebuild can therefore re-read days of moments. That adds nothing to the arcs they reached, except re-adding seeds the 50-member cap evicted. When the log is empty, the cursor is `None` and the cutover reruns, so moments since the real cutover never become arcs [D19].
- **A pass that failed halfway.** The pass appends its log events as it goes and saves the state only at the end. Loading the state also rebuilds it from `arcs.log.jsonl` when the log is ahead of it: the log has an open arc (an `arc_opened` event with no `arc_closed` for the same arc) that the state lacks, or shows closed an arc the state still holds open. A state arc the log does not know is kept, as today when the log is missing. So a pass that failed after appending events is rebuilt from its own events on the next load, and the same moment never opens a second arc. The rebuilt open arcs weigh 0 until the next pass recomputes them; the `arc_closed` event records the closed arc's weight and dominant feeling, so closed arcs keep theirs [D22, D9].
- **A rebuilt cursor after a failed pass.** If the failed pass had appended events stamped at pass time, the newest event time is later than that pass's sweep time, so a moment written in the seconds between the log read and those events is never taken. Accepted [D19, D22].

Each moment taken is stamped with her lived age at that pass, as candidate members are stamped today [D3]. Its seed members join at that age, and the arc's `lived_age_at_open` is that age. No file or writer is added [D3].

### Seeds [D6]

| moment | seed memory ids |
|---|---|
| dream (`dreams.log.jsonl`) | `seed_id`, every id in `neighbour_ids`, `dream_id` |
| growth (`emotion_growth.log.jsonl`) | every id in `evidence_memory_ids` |
| soul (`soul_audit.jsonl`) | every id in `related_memories` |

Ids are kept in that order, without repeats, and non-string values are dropped (a builder-level detail of D6; today's `_extract_seed_memory_ids` coerces with `str()` instead). A moment that names no id opens no arc and extends none [D6]. `_extract_seed_memory_ids`' old key list (`seed_memory_ids`, `linked_memory_ids`, `memory_ids`, `memory_id`) is replaced by this table (builder-level detail of D6).

### Opening and extending [D7]

Any of the three types may open an arc, and there is no cap on open arcs. A moment extends an open arc with its other seeds, instead of opening a new one, in two cases [D7]:
- one of its seed memories already belongs to that arc (today's rule, `policy.should_open`);
- it names at least one seed memory, and its title, compared after the D10 cut, equals that arc's title. This includes an arc opened earlier in the same pass.

A moment with no seed memory does nothing, even when its title matches an open arc [D6, D7].

So two open arcs never share a title [D7]. Felt time's title-keyed arc lines (`brain/felt_time/prompt.py`) need no change. In one case they can still show a new open arc as closed: its moment fell between the previous sweep time and the close of an arc with the same title. That window lasts seconds and is accepted.

### Titles [D10, D11]

An arc's title is its moment's label (#676: dream memory text, growth `name`, soul `candidate_text`) with each line break replaced by a space; `\r\n` counts as one [D10]. Its total length is at most `ARC_TITLE_MAX_CHARS` (60, in `brain/dev_constants.py`). When the label is longer, it is cut at the last word boundary within the first 59 characters, trailing spaces are dropped, and it ends with `…` [D10]. When the 60th character is a space, the first 59 already end on a word and are kept whole; a space with nothing before it is not a word boundary [DERIVED from D10]. A label longer than 60 with no word boundary in its first 59 characters is cut at 59 characters plus `…` [D11]. The stored `Arc.title` is that string, and every surface shows it [D10]:
- the `arc_opened` event in `arcs.log.jsonl`, which felt time's arc anchors and `_replay_from_log` read;
- the arcs ambient block;
- `get_narrative_threads` and `get_arc_detail`;
- the grief block;
- the arc-close grief memory, via `arc_close_phrase`. A moment with an empty label opens no arc, but it may extend one by its seeds; a label of only spaces counts as empty [D11].

### Weight while open [D9]

On every pass, each open arc's strongest member emotion is recomputed with `_compute_arc_close_emotion_inputs`, the same function the close uses. The result is saved on the arc as `max_member_emotion_normalised` and `dominant_non_grief_emotion` [D9]. When the recompute raises, the arc keeps its saved values and the pass goes on [D9]. When the pass cannot open `memories.db` at all, open arcs keep their saved values and no arc closes that pass, because a close reads the members' feelings and writes its grief there. A stale arc then closes, with its grief, on the next pass that can read the store; the rest of the pass goes on [D23]. Time's heaviness (`brain/felt_time/lived_age.py::narrative_weight`, fed by the supervisor's drivers) and forgetting's fade gate read the saved value. The close recomputes it fresh, as today [D9].

### Fading [D5]

Unchanged. `brain/forgetting/policy.py` lowers the fade gate for all her memories by `narrative_weight` [D5]. It becomes non-zero because of D9.

### Closing and grief [D12, D14, D15]

The numbers stay as they are [D12]:
- an open arc not touched this pass closes as `stale_72h` when lived age now minus its newest member's join age is at least `ARC_STALE_LIVED_HOURS` (72.0);
- `MAX_ARC_MEMBERS` (50) evicts the lowest-salience members first, as today;
- a close writes an `arc_close` grief breadcrumb when `ARC_CLOSE_SCALE` (7.0) × the strongest member emotion is at least `THRESHOLD` (3.0);
- a forgotten member of an open arc fires today's `recall_touch` grief, as today.

Note (not a decision): today's `all_members_lost` close (`not arc.members`) is left as it is. No path empties an arc's members, because lost members stay listed, so it does not fire.

Membership keeps the hebbian-or-embedding test and reads hebbian only through `HebbianView.weight` [D14]. Grief keeps `brain.grief.breadcrumb.write_breadcrumb` and adds no emotion representation [D15].

### New parts and the alternatives

- **`anchor_cursor_iso`.** The alternatives were to reuse `last_pass_ts_iso`, to read felt time's stored anchors, or to keep two fields.
  - `last_pass_ts_iso` lets moments from the last pass window before shipping through (against D2). It also cannot be held back without triggering the recovery replay (D4).
  - Felt time's stored anchors hold only the newest moment of each type, so moments between passes would be lost.
  - A separate set-once cutover field would duplicate the cursor's first value.
- **Nothing else new.** Seeds, titles and weights reuse the existing rows, labels and close function.

## Wiring

- **Reads from:**
  - felt time's `_SOURCES` rows: `DreamEngine._log`, `append_growth_event`, `append_audit_entry`;
  - felt-time lived age;
  - `MemoryStore` (candidate pool, member feelings through the emotion engine);
  - `HebbianView` and embeddings (membership);
  - the forgetting graveyard (lost members).
- **Feeds:**
  - `narrative_weight`, which feeds the lived-age rate and forgetting's fade gate (D9);
  - the arcs ambient block (`brain/narrative_memory/prompt.py`, volatile tail);
  - the felt-time arc lines (`arc_anchors` from `arcs.log.jsonl`); a newer arc row also restarts felt time's current stretch (#676 D12, D18);
  - the MCP tools `get_narrative_threads` and `get_arc_detail`;
  - grief breadcrumbs (memories) and the grief block;
  - the Loss panel through `/persona/state`.
- **Tier [D16]:** `docs/maturity-manifest.md` moves `narrative_memory` from CORE-STABLE to EXPERIMENTAL until the D17 test passes and it is validated live. The grief row stays CORE-STABLE, with a note that its arc-close leg has never fired.

## Invariants

| # | Invariant | holds / n.a. | where |
|---|---|---|---|
| I1 | install + name + talk | holds | no setting |
| I2 | reads and feeds the loops; §Wiring | holds | Wiring |
| I3 | organ DoD; tier | holds | Wiring (tier); Acceptance D17 |
| I4 | store API only | holds | Seeds re-read rows; membership via `HebbianView`; feelings via the engine |
| I5 | additive state, defaults | holds | `anchor_cursor_iso` defaults to `None`, is saved and read by name, and is rebuilt from the log [D19]. If an older brain rewrites the file, it drops the field, so the cutover reruns; D9 reuses existing fields |
| I6 | long cadences persisted | holds | runs on the existing persisted maintenance cadence (6 h) |
| I7, I8–I12, I15 | autonomy, body, gates, salience, bridge, provider, overlay | n.a. | no LLM call or new behaviour |
| I13 | stable prefix; volatile tail | holds | arcs and grief blocks stay where they are |
| I14 | cross-platform | holds | canonical JSONL reader and stores |
| I16 | canary | holds | Acceptance D17, D2, D4 |
| I17 | no real names | holds | temp `KINDLED_HOME`; the D8 replay reports counts only |

## Acceptance criteria

Every test runs in a temp `KINDLED_HOME` with the real writers. The live persona is never a fixture.

- **D2:** with older moments present, the first pass sets `anchor_cursor_iso` and opens nothing, even with lived age unknown; a later moment opens an arc; an older one never does.
- **D18:** an `arc_opened` row in `arcs.log.jsonl` never becomes a moment; a dream opens an arc titled from its memory text; a dream whose memory is gone opens no arc.
- **D21:** a dream whose memory is still in the pending queue at the pass takes its title from felt time's stored label for that same row, without `DREAM: `; with neither label, it opens no arc.
- **D22:** a pass that raises after appending an `arc_opened` event leaves no saved arc; the next load rebuilds it from the log, and the next pass opens no second arc for the same moment. A state holding an open arc the log does not know is not rebuilt. An arc closed before a rebuild keeps its weight.
- **D23:** with `memories.db` unopenable, a stale arc stays open that pass with its saved weight and closes, with grief, on the next pass that can read the store.
- **D19:** a state file without the field loads with `None`; after a rebuild from `arcs.log.jsonl`, the cursor is the log's newest event time, and a moment after it still opens an arc.
- **D20:** with a corrupt line before a dream row, the dream still seeds its own memories.
- **D3:** an arc opened at a pass has `lived_age_at_open` equal to felt time's lived age at that pass.
- **D4:** with lived age `None`, a pass opens nothing and leaves `anchor_cursor_iso` unchanged; the next pass with a lived age opens the arc. Holding the cursor triggers no recovery replay.
- **D6:** a dream opens an arc with `seed_id`, the neighbours and `dream_id` as members; a soul row with empty `related_memories` opens none; a growth row seeds its evidence.
- **D7:** a second moment sharing a seed extends the open arc instead of opening another; a moment with no shared seed but the same title as an open arc extends it too; a seedless moment with that title changes nothing.
- **D9:** after a pass, an open arc whose member feels 8/10 has `max_member_emotion_normalised` 0.8, and the supervisor's `narrative_weight` is above 0 once the arc has been open long enough.
- **D10, D11:** a 300-character label becomes a title of at most 60 characters ending in `…` at a word boundary; a 70-character single word becomes 59 characters plus `…`; two moments with the same text make one arc, the second extending the first; a label with a line break is one line; an empty-label dream opens no arc.
- **D12:** a quiet arc closes as `stale_72h` at 72 lived hours, not before; a close whose strongest member feels 5/10 writes an `arc_close` breadcrumb, and one at 4/10 does not.
- **D17 (through path):** `DreamEngine` with `FakeProvider` writes a dream, then the felt-time tick runs, then `_run_narrative_memory_pass`. The arc opens with its seeds, the open arc gives a non-zero `narrative_weight`, and after 72 lived hours with no joins it closes and writes grief for a strong member.
- **D8, measured pre-ship (guarded-change):** a read-only replay of the reference persona's log ids and timestamps reports the figures below. It also reads each seed memory's emotion values from `memories.db`, through `MemoryStore` (I4), and converts wall hours to lived hours with her current lived-age-to-wall ratio. It uses seeds only, labelled as a lower bound, and reports counts and durations only:
  - how many arcs would be open at once;
  - how long arcs stay open;
  - how many closes per week would write grief;
  - how many seeds the 50-member cap evicts.

  The owner sees it before ship (option context of D8 and D12, "so you see the volume before it goes live").

## Left to the builder

- How the seed read reuses `_adapt_anchors`' structure, provided D20's reader and indexing hold.
- When a moment's seeds match one open arc and its title matches another: extend the seed-matching arc (today's first-match order) unless there is reason otherwise, provided open arcs never share a title. As today, an arc is marked touched when it gains a member.
- Where in the pass the D9 recompute runs, provided the saved values are in place before the pass saves state.
- Moments in a pass are applied in parsed-time order, so the earliest of two same-title moments opens the arc.
- Where the title helper lives (`narrative_memory`), provided every arc is created through it.
- Merge order with #655, #656 and #803, which touch `supervisor.py` near this pass.

## Deferred

- **#676's D9 acceptance** ("opens no arc … writes no grief") is superseded by this spec's D17 once this ships.

- **Volume limits.** You decide whether to cap open arcs or change the numbers after seeing the D8 replay, before it ships [D7, D8, D12].
- **Known states.** A dream felt time never labelled opens no arc (D21). A felt-time cold start reads as lived age 0.0, which counts as known, so a pass in that state stamps 0.0; after #385 that needs every felt-time backup to be unreadable. An unreadable moment log makes every pass raise, so no arc closes until the file is readable; the log line repeats each pass.
- **Members-only fade protection.** Holding an open arc's own members against fading, rather than all memories, is not built [D5].
