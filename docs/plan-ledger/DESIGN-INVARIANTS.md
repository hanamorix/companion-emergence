# companion-emergence — Design invariants

status: FOR OWNER REVIEW     <!-- FOR OWNER REVIEW | REVIEWED <date> -->

Standing architectural rules the owners have set. Every plan-ledger sheet, spec, red-team and drift
check is checked against these, row by row. A line is added only with an owner's words and a date; a
line is removed only by an owner. Keep each line to one checkable sentence; put the why in the second
column.

Seeded 2026-09-22 from rulings that can be quoted (issue titles and bodies on the tracker, dated by
issue creation). Every row is provisional until an owner marks the file REVIEWED.

| # | Invariant (one checkable sentence) | Why (owner's words, date, source) |
|---|---|---|
| I1 | Hebbian associations live inside memories.db, not in a separate database file. | "Fold hebbian_edges into memories.db to remove the separate-file orphan/corruption risk" — ThinkerOfThoughts, 2026-08-12, #128 |
| I2 | A per-memory attribute (an embedding, a score) is stored on the memory row or a memories-side table keyed by memory id, never in a content-hash side cache. | "Store per-memory embeddings on the memory row, not in a content-hash side cache (embeddings.db)" — ThinkerOfThoughts, 2026-09-13, #259 |
| I3 | A behaviour threshold is derived or self-calibrating; a hardcoded empirical constant is tracked as a defect, not shipped as the answer. | "Replace the hardcoded RERANK_FLOOR relevance threshold with a derived mechanism" — ThinkerOfThoughts, 2026-09-12, #250 |
| I4 | Emotion signals and journal entries are not stored as standalone memory rows; each gets its own bounded store or field. | "Emotion signals are stored as standalone memory rows in memories.db instead of a field / dedicated store" (#222) and "Journal entries are stored as first-class memory rows in memories.db — consider a dedicated store" (#224) — ThinkerOfThoughts, 2026-09-07 |
| I5 | The companion never receives its owner's own CLAUDE.md: the CLI is spawned from a working directory whose ancestor chain carries no CLAUDE.md, on every supported platform. | "the companion still receives the owner's global ~/.claude/CLAUDE.md because no CLAUDE.md-free working directory exists" — hanamorix, 2026-09-12, #252 (gap statement; the #122 fix is the rule) |
| I6 | Mechanics constants and model-facing prompt strings live in dedicated editable files, separate from mechanism code, and there is one such surface, not several. | "Externalize mechanics constants and prompt strings into dedicated editable files, separate from mechanism code" — ThinkerOfThoughts, 2026-08-12, #129 |
| I7 | Model-facing prompt and tool-description strings carry no persona-name holdovers and no LLM verbal tics. | "Sweep 'Nell' (and other) persona-name holdovers from prompt strings" (#135, 2026-08-18) and "Sweep claudisms / LLM verbal tics from model-facing prompt & tool-description strings" (#184, 2026-09-02) — ThinkerOfThoughts |
| I8 | Updating a bundled install is a standalone script, never a Python subcommand that runs from the runtime it replaces. | "Design decision 2 chose a standalone bash script over a `nell update` subcommand: a Python subcommand runs from the runtime it replaces" — hanamorix, 2026-09-12, #256 (docs/source-spec/2026-09-13-update-script-design.md) |
| I9 | The three service backends (launchd, systemd, Task Scheduler) promise the same per-user, login-started, restart-on-failure shape and are tested in the same string-equality shape. | "the per-user install this backend promises (the launchd gui/<uid>/ and systemd --user analog)" — toutounnis, 2026-09-13, #260; brain/service/windows_service.py docstring |
| I10 | Live tests never drive the live persona; every live run goes through the harness sandbox, which refuses to run beside a live companion service. | tests/harness/__init__.py: "nothing it does may touch anything outside its temp sandbox"; #264 — hanamorix, 2026-09-14 |
