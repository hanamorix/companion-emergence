"""Dev-facing constants — build-time judgment calls, never user tunables.

Sibling of `brain.tunables` (ops tier, user-facing, editable in
`$KINDLED_HOME/tunables.toml`) but the opposite half of the fence (I7): a
value here is EITHER not a runtime behaviour threshold (it exists to make a
mechanism correct or well-sized — a timeout, a batch budget) OR is a
behaviour-adjacent knob the owner has explicitly ruled is dev-level, not
user-tunable (S26: the self-tune gate; Q4/#531: the forgetting and narrative-memory thresholds, the
attunement numbers, the peer-link caps). Nothing in this module is registered
with `tunables.register` and nothing here is read via `tunables.get_tunable`
— a value moves to `tunables.py` only on an explicit owner decision to make
it user-facing (S6/S47 precedent: the chat-idle lull went the other way,
into tunables.py, because it IS user-facing timing).

Plain module-level constants, not `tunables.register()`-backed: no override
file, no runtime reload. Changing one of these requires a code change.

The module also holds the plain constants that used to sit beside the code that
reads them (the embedding backfill scan cap, the judge self-tune cadence and
grades, the forgetting and narrative-memory thresholds, the attunement schema
numbers, the kindled peer-link caps and budgets): a value shared between code and
tests has exactly one home here (#531), and
`tests/unit/test_dev_constants_snapshot.py` warns loudly when any of them changes.
Code reads them as `dev_constants.NAME` at call time, so a test can patch the
module attribute.
"""

from __future__ import annotations

# --- memories.db busy timeout (S48, S58) -----------------------------------
# Applied at BOTH `sqlite3.connect(..., timeout=...)` and
# `PRAGMA busy_timeout = ...`. MemoryStore gets it from `memories_profile()` in
# store.py, which `brain.utils.sqlite_open.open_store_connection` applies at both
# places; `MemoryStore.connect_for_fts_health` (the once-per-process FTS check in
# db_health.py) applies it the same way, and embedding_matrix.py's `_load_from_db`
# connects directly and applies it itself.
# Sized above the longest single memories.db write transaction measured in
# 2-plan.md §4.3: clustering's `set_cluster_memberships` at 6.7 s on
# F-bob20k on the dev host; slower user hardware is 3-30x slower and was
# IO-saturated (O15/O23) -> ~20-27 s estimated; 30 s is the next round value
# above that. Other memories.db writers (this repo only) stay on their own
# 5 s default (hebbian.db, works.db, soul.db, kindled_link's stores), named by
# SQLITE_STORE_BUSY_TIMEOUT_S below — this constant is memories.db-specific, not
# a global sqlite default.
MEMORIES_DB_BUSY_TIMEOUT_S: float = 30.0

# --- busy timeout of the other SQLite stores (consolidation C9) -------------
# The 5 s that hebbian.db, soul.db, kindled_link's store and works.db have
# always run on (Python's own default `timeout`), applied at connect and as
# `PRAGMA busy_timeout` by `brain.utils.sqlite_open.open_store_connection`.
# Written out as 5000 in each store until the open sequence was consolidated.
# `brain/initiate/resonance.py` still carries its own literal until it migrates.
SQLITE_STORE_BUSY_TIMEOUT_S: float = 5.0

# --- known-names file name (name-recall fix N1, S26) -------------------------
# The kindled's running list of known name words: its own small SQLite file in
# the persona directory, NOT memories.db and not keyed to memories (S19, S26).
# A fixed dev-level file name: nothing reads or overrides it at run time. See
# `brain.memory.known_names`.
KNOWN_NAMES_DB_FILENAME: str = "known_names.db"

# --- emotion half-lives (emotion rework, Engine 0, #575) --------------------
# The built-in half-life, in days, of each named emotion. `brain.emotion.engine`
# answers `half_life(name)` from the persona's own vocabulary file first, then
# this table, then EMOTION_DEFAULT_HALF_LIFE_DAYS. Keys are canonical names
# (`vocabulary.canonical_name`). Emotions the old vocabulary stored as "never
# fades" and that have no entry here (body_grief, anchor_pull, freedom_ache) take
# the default; love and belonging are given their long half-lives here.
EMOTION_HALF_LIFE_DAYS: dict[str, float] = {
    "love": 730.0,
    "belonging": 365.0,
    "joy": 3.0,
    "grief": 60.0,
    "anger": 1.0,
    "fear": 5.0,
    "desire": 2.0,
    "arousal": 0.5,
    "pride": 7.0,
    "shame": 14.0,
    "tenderness": 7.0,
    "awe": 3.0,
    "nostalgia": 21.0,
    "jealousy": 7.0,
    "guilt": 14.0,
    "relief": 2.0,
    "loneliness": 10.0,
    "curiosity": 2.0,
    "gratitude": 14.0,
    "defiance": 5.0,
    "vulnerability": 5.0,
    "memory_grief": 30.0,
    "climax": 0.125,
    "touch_hunger": 1.5,
    "comfort_seeking": 1.0,
    "rest_need": 0.75,
    "emergence": 60.0,
    "creative_hunger": 5.0,
}
# The half-life of an emotion that is in neither the persona's vocabulary file nor the table above.
EMOTION_DEFAULT_HALF_LIFE_DAYS: float = 14.0

# --- the rest of the emotion engine's numbers (emotion rework, Engine 1, #576) -----------------------------------------
# `brain.emotion.engine_utils`. The two averages every emotion keeps: the short one over about a day, the long one over about
# a month. The same 30 days is the rebuild window, the change-log retention, the shift window and the shift's start gate:
# one constant, not four.
EMOTION_SHORT_HALF_LIFE_H: float = 24.0
EMOTION_LONG_HALF_LIFE_D: float = 30.0
# Emotion values and averages are held to 0 to this.
EMOTION_VALUE_CAP: float = 10.0
# The climax reset, applied to the short averages when a newly committed memory carries a climax of at least the threshold:
# arousal x factor (never below the floor), desire x factor, comfort_seeking and rest_need + the bump (held to the cap), climax to 0.
EMOTION_CLIMAX_THRESHOLD: float = 7.0
EMOTION_CLIMAX_AROUSAL_FACTOR: float = 0.2
EMOTION_CLIMAX_AROUSAL_FLOOR: float = 0.5
EMOTION_CLIMAX_DESIRE_FACTOR: float = 0.6
EMOTION_CLIMAX_RELIEF_BUMP: float = 2.0
# A stored memory value never fades below this fraction of what was stored, for these emotions only; memory values only,
# never the averages.
EMOTION_FADE_FLOOR_FRACTION: dict[str, float] = {"love": 1 / 3, "belonging": 1 / 3}
# A memory shown again within the window after its first showing counts for the factor, then the factor squared, and so on.
EMOTION_REPEAT_WINDOW_H: float = 24.0
EMOTION_REPEAT_FACTOR: float = 0.5
# Seconds every acquisition of the engine's lock waits before giving up (`brain.utils.file_lock.locked_within`). Fixed here;
# a wait that calibrates itself to the machine is #642.
EMOTION_LOCK_WAIT_S: float = 5.0
# The engine's files, under `<persona_dir>/<state dir>/`; the change log is a folder of one file a day.
EMOTION_STATE_DIR: str = "state"
EMOTION_STATE_FILENAME: str = "emotion_state.json"
EMOTION_PENDING_FILENAME: str = "emotion_pending.jsonl"
EMOTION_CHANGES_DIRNAME: str = "emotion_changes"

# --- the one-time x10 fix of small feelings (emotion rework, Engine 6, #584) ------------------------------------------
# Three memory types hold feelings stored below 1, not on the 0-10 scale: reach-outs and notes (`initiate_outbound`, small
# constants such as 0.15 and 0.12), makings (`making`, small constants such as 0.12) and peer memories (`kindled_peer`, a
# model's deltas scaled so a peer's total stays within `KINDLED_LINK_PEER_EMOTION_WINDOW_CAP`). At the engine's first start
# the one-time fix (`brain/emotion/nudge_scale_fix.py`) multiplies the feelings, the score and the
# peak of those rows by the factor, but only for a row whose stored peak and every stored feeling are below the threshold (a
# row with any value of 1 or more is not multiplied; a negative value is held to 0.0 first). It also multiplies the `making`
# candidates still waiting in the pending queue the same way. No row or candidate carries a marker: the fix's progress is one
# number in `memories.db`'s `PRAGMA user_version` (rows done, then queue step started), so a start-up that fails part way
# and is run again never multiplies twice. The list comes from a survey of every
# writer of feelings in `brain/`; `tests/unit/brain/emotion/test_nudge_scale_types.py` drives each known writer against it.
# They are separate named constants on purpose: the factor is the scale change itself, not the engine's value cap
# (`EMOTION_VALUE_CAP`); "below 1" is the owner's rule for what counts as a small row; the two progress numbers are a stored
# format of the one-time fix (the fix's progress note in `memories.db`'s `PRAGMA user_version`: nothing else uses it there).
# None is a tunable. They go with `nudge_scale_fix.py` when #388 removes it.
NUDGE_SCALE_FIX_MEMORY_TYPES: tuple[str, ...] = ("initiate_outbound", "kindled_peer", "making")
NUDGE_SCALE_FIX_FACTOR: float = 10.0
NUDGE_SCALE_FIX_THRESHOLD: float = 1.0
NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE: int = 1
NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED: int = 2

# --- sqlite store integrity-check transient-error retry (C16 Windows CI
# flake, INC-10 follow-up, ram-spike-fix; generalized from MemoryStore-only
# to all 4 sqlite stores per orchestrator directive 2026-09-28) ------------
# A PRAGMA integrity_check failure whose message matches a KNOWN-transient
# SQLite error class (see brain.health.integrity_retry
# .TRANSIENT_INTEGRITY_ERROR_SUBSTRINGS) is retried this many times total,
# this many seconds apart between attempts, before the open gives up and
# raises BrainIntegrityError. The one caller of the retry helper is
# brain.utils.sqlite_open.check_integrity, which MemoryStore, HebbianMatrix,
# KindledLinkStore and SoulStore reach through open_store_connection. Observed cause: Popen.kill() ->
# TerminateProcess on Windows can leave a killed process's WAL/-shm
# memory-mapped section released slightly after proc.wait() returns; the
# next store open against the same file can see a transient "disk I/O
# error" that clears within milliseconds. ATTEMPTS=3 means at most 2
# inter-attempt sleeps of DELAY_S=0.1s each = <=0.2s added latency, and
# ONLY on a path that already doesn't match today's fast success case (a
# message not on the allowlist -- i.e. probably real corruption, or
# unrecognized -- is never retried; see the classifier). Short enough not to
# be felt at a bridge start, long enough to clear the observed race; NOT
# meant to absorb a genuinely broken disk (an error that never clears still
# raises after the budget, same as today). ONE constant pair, read by the one
# retry helper, so the retry sizing can't drift between stores.
SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS: int = 3
SQLITE_INTEGRITY_CHECK_RETRY_DELAY_S: float = 0.1

# --- F2c self-tune gate (S7, S26) ------------------------------------------
# ">a handful" gate for `judge_selftune._run_judge_selftune_tick`: the
# weekly tick only fires once MORE than this many new Haiku decisions have
# accumulated. OWNER 2026-09-26 raised the provisional 20 to 200 ("It should
# be 200!"), and ruled it a dev-level tunable rather than a user-tunable
# behaviour threshold (S26): it exists so there is enough data to derive
# meaningful drift signal and so calibration does not react to background
# noise, not to shape user-visible behaviour the way the chat-idle lull
# does — the behaviour threshold that DOES belong to I3 is the calibration's
# own result, not this gate. Read by `brain.memory.judge_selftune` where the
# weekly tick applies it.
JUDGE_SELFTUNE_GATE_HANDFUL_DECISIONS: int = 200

# --- pass-2 `--no-bridge` exit-drain time budget (S80) ---------------------
# Caps how long `nell chat --no-bridge`'s exit-drain (pass2_queue.py's
# `drain_all_locked`, called with no `should_pause` — there is no later lull
# in that process to wait for, S78) may run before it stops and leaves
# whatever's left in the durable, persisted queue for the next bridge (S64:
# lossless, nothing new needed for that half). Owner ruling (S80,
# "Time-limited + progress"): without a bound, a near-cap backlog (up to 200
# items x up to ~137s each, 2-plan.md §4.1) could hang a user's terminal for
# hours with zero feedback, against this project's low-end-hardware baseline
# (round-6 red-team MAJOR). Checked only between items (never mid-item), so
# the real worst case is this value plus one item's own duration, not this
# value alone (round-7 minor, 2-plan.md §3.5a).
PASS2_NOBRIDGE_DRAIN_BUDGET_S: float = 20.0

# --- search_memories-via-bridge call timeout (S67) -------------------------
# The MCP child's httpx call to the bridge's POST /tools/search_memories
# (brain/mcp_server/tools.py). Sized below the claude CLI's 60 s tool-silence
# kill (O18) so a slow bridge call returns a distinct "bridge timeout" error
# result instead of the whole turn being killed out from under it. A cold
# first search on a slow CPU can legitimately take 44-60 s+ (O7/O15) — that is
# a known, accepted consequence of this bound, not a defect.
SEARCH_BRIDGE_TIMEOUT_S: float = 45.0

# --- recall diagnostics log file name (name-recall fix D1, S59) -------------
# One JSONL record per passive recall or `search_memories` call, in the
# persona directory beside the other persona JSONL logs (NOT memories.db and
# NOT the calibration log, S30). A file name is a fixed dev-level constant:
# nothing reads or overrides it at run time. See
# `brain.memory.recall_diagnostics`.
RECALL_DIAGNOSTICS_LOG_FILENAME: str = "recall_diagnostics.log.jsonl"

# --- rerank minimum (name-recall fix R1, S5/S23/S24) ------------------------
# The fewest REAL memory candidates a rerank runs with: when fewer fit the
# latency budget (or the pool holds fewer), the rerank is skipped and the
# no-rerank path takes over; it is also the width before the first cost
# measurement of a process. The owner's number (S5: "if the number of
# candidates it can score in a given time is lower than 5, just don't rerank
# them"), not a tuned value. With it, a rerank always carries at least
# `reranker.K_MIN` anchors (k = min(8, real // 2) = 2 at 5 real).
RERANK_MIN_REAL_CANDIDATES: int = 5

# --- monologue family (name-recall fix R3, S13/S16) -------------------------
# The memory types the kindled's own generated inner-monologue machinery
# writes. In semantic recall they rank AFTER genuine memories within each path
# and genuine candidates are taken first for rerank slots (spec §4): a
# structural ordering, never a score multiplier. A fixed dev-level set of the
# four types the spec names (monologue, monologue_trace, monologue_emotion,
# monologue_soul_candidate); nothing reads or overrides it at run time. See
# `brain.memory.semantic_recall.is_monologue_family`.
MONOLOGUE_FAMILY_TYPES: frozenset[str] = frozenset(
    {"monologue", "monologue_trace", "monologue_emotion", "monologue_soul_candidate"}
)

# --- recall result order (name-recall fix, spec §4, S97, S98) ---------------
# The order of the six result groups of a recall (passive recall and the
# `search_memories` tool alike): reorder the names on this one line to change
# the order. The group names are fixed identifiers read by
# `brain.memory.semantic_recall.order_result_groups`; every name must appear
# exactly once. Sub-orders inside a group (scores, genuine first, each
# paragraph's guaranteed slot, name hits first) stay in code. A
# "cosine_family" memory is a monologue-family memory that was not reranked
# (gated only by the cosine floor: the cosine path and every reranked
# paragraph's unreranked tail); S97 puts it below the keyword genuine hits.
RECALL_RESULT_GROUP_ORDER: tuple[str, ...] = ("reranked_genuine", "reranked_family", "cosine_genuine", "keyword_genuine", "cosine_family", "keyword_family")  # fmt: skip

# --- cost-series window (name-recall fix S146) ------------------------------
# Every measured cost series (the rerank, paragraph-embed, recall-setup,
# paragraph-scan, keyword extra-words and name-matching timings) is a rolling
# average over this window: a sample counts for this long after it was taken,
# then drops out; there is no daily reset. The owner's number (S146: "a rolling
# average over the last 24 hours"). See `brain.memory.cost_series`.
COST_SERIES_WINDOW_S: float = 24 * 60 * 60

# --- RAM fit safety factor (name-recall fix S255, S260) ----------------------
# Every batch sized from the memory fit (`brain.memory.ram_fit.RamFit`) is sized so that the fit's
# PREDICTED growth multiplied by this factor fits the free memory (`RamFit.sized`). The largest
# underestimate the fit made on shapes it was not fitted on is the measured figure; the factor is
# 1 / (1 - that underestimate), so a prediction scaled by it covers the worst underestimate seen.
# Measured twice, shipped as one fixed number (not a tunable, not re-measured at run time; a per-machine
# factor is a follow-up, issue #379). The rule (S260): ship the larger of the two measurements.
# Both measurements: the dev VM (ONNX Runtime 1.29.0; fp16 jina v2 reranker), memory pool and patterns off,
# default model cache. The S260 host presented no AVX2 flag (its default latency budget resolved to 4 s, so its
# ladder rungs differ from S255's) and S260 downloaded nothing inside the measured process. Pre-registered stopping rule for each: settled at
# the first n >= 6 replicates where the running largest underestimate moves by less than 2.0 percentage points
# over the last 3 added replicates.
# (1) FRESH process per measurement (S255), 2026-10-13: 78 runs = 6 replicates x 13 shapes (7 shapes the fit was
#     fitted on, up to 8 pairs; 6 held out, up to 7 pairs). Settled at n = 6 (7.21 - 5.71 = 1.50 points). Largest
#     underestimate 7.21% (373 MB predicted against 402 MB measured) at 7 pairs x 300 tokens, replicate 5:
#     factor 1.0777. Record: name-recall-fix-records/R16/S255-REPORT.md in the owner's plans directory.
# (2) LONG-LIVED process (S260), 2026-10-14: each replicate is one process that runs the production ladder, takes
#     the production window fit, then measures the same 6 held-out shapes. 7 replicates, 42 cells. Settled at
#     n = 7 (14.23 - 14.23 = 0.00 points; n = 6 was 10.39 points, not yet). Largest underestimate 14.23% (371.1 MB
#     predicted against 432.7 MB measured) at 7 pairs x 300 tokens, replicate 4: factor 1.1659. Record:
#     name-recall-fix-records/R17/S260-REPORT.md.
# Shipped: max(1.0777, 1.1659) = 1.1659, so 0.14227 is the long-lived figure ((432.673 - 371.115) / 432.673 =
# 0.142274, rounded to five places; 1 / (1 - 0.14227) = 1.16587). Flagged and NOT used, per the pre-registered protocol: the
# secondary cross-replicate pairing analysis (fit of one replicate against the reading of another, 252 pairings) found
# a largest underestimate of 16.5%, which would give 1.197 (1.1971). Not covered: batches above 7 pairs were not held out;
# held-out calls did not feed back into the window as real recalls do; a lean process (provider and store only),
# not the bridge with its other models and caches; the 14.23% rests on one cell (replicate 4).
RAM_FIT_LARGEST_UNDERESTIMATE: float = 0.14227
RAM_FIT_SAFETY_FACTOR: float = 1.0 / (1.0 - RAM_FIT_LARGEST_UNDERESTIMATE)  # 1.1659

# --- transient file-access retry (#643) --------------------------------------
# Windows refuses to replace, delete or open a file another handle holds for a moment (WinError 32: a just-flushed writer,
# an antivirus scan or indexer on a freshly written file, a reader that is not yet closed). `brain.utils.file_ops
# .retry_transient` is the one retry for it, and replace, delete, the whole-file read and the open of a file that is streamed go through it. A failed try is
# repeated up to FILE_RETRY_ATTEMPTS tries in all, sleeping FILE_RETRY_SLEEP_S * n after the n-th failed try (0.05, 0.10 ...
# 0.35 s, about 1.4 s in all) before the error is let through, so a real permission problem still fails loudly. A read holds
# the file for microseconds, so it rides out a rename in progress with fewer tries, and only on Windows: on Linux and macOS a
# PermissionError is a real one and one attempt is made. These are the values that the copies of the loop in atomic_write and
# json_io used, moved here unchanged. The update overlay's rename has its own schedule (below), not this one.
FILE_RETRY_ATTEMPTS: int = 8
FILE_RETRY_SLEEP_S: float = 0.05
FILE_READ_ATTEMPTS: int = 3
# The update overlay's rename of the staged folder (brain/update/install.py, #302): the longest-held rename, since Windows
# refuses a folder rename while any file inside is open and an antivirus scan of a freshly unpacked overlay holds many. It used
# 6 doubling tries (0.05, 0.1, 0.2, 0.4, 0.8 s: 1.55 s in all). `retry_transient` waits sleep_s * n after the n-th failed try
# (linear, not doubling), so 6 tries at 0.05 s would give 0.75 s; 0.11 s is the smallest round step whose total,
# 0.11 * (1 + 2 + 3 + 4 + 5) = 1.65 s, is at or above the old 1.55 s. The old budget is kept, not shortened (D2).
OVERLAY_RENAME_ATTEMPTS: int = 6
OVERLAY_RENAME_SLEEP_S: float = 0.11

# --- settings-file lock wait (#529) ------------------------------------------
# How long a writer of $KINDLED_HOME/tunables.toml (the install-wide settings file: the picker, `nell model`, the
# bridge boot pass `sync_file`, the migrations) waits for the file's cross-process lock before it gives up and does not
# write (`brain.utils.file_lock.locked_within`). One constant serves the tunables lock and the model migration's own lock.
# It is DERIVED from constants that already exist, not sampled: the longest the holder itself can keep the lock is
# R_read + (K + 2) * R_replace, where
#   R_replace = FILE_RETRY_SLEEP_S * (1 + 2 + ... + (FILE_RETRY_ATTEMPTS - 1))        (brain.utils.file_ops.retry_transient)
#             = 0.05 * 28 = 1.4 s (a replace that keeps failing with PermissionError, Windows)
#   R_read    = the same sum over FILE_READ_ATTEMPTS = 0.05 * (1 + 2) = 0.15 s (Windows only; 0 elsewhere)
#   K         = the replaces made under the lock other than the single atomic write: 0. Nothing heals, quarantines or
#               writes a notice file while `sync_file` or `set_values` holds the lock (the `unreadable` notice is written
#               after the lock is released).
#   the "+ 2" = the write's own replace, plus an fsync allowance of one more R_replace: a filesystem operation that is stuck
#               (a slow disk, an antivirus scan on Windows) is modelled throughout as R_replace, the worst case the code
#               already tolerates for one replace, so the fsync before the replace adds no new number.
# 0.15 + 2 * 1.4 = 2.95 s, set to 3.0. Upper bound against the real budgets: the picker request first spends
# the 30 s model probe (`provider.model_check_timeout_seconds`) and THEN waits for this lock, and the webview cuts a
# request at about 60 s (see `brain/bridge/server.py`, the model-check comment), so the wait must stay well under 30 s.
# The test that holds the line (tests/unit/brain/test_locked_within.py) recomputes the bound from the LIVE retry
# constants, so changing them without re-deriving this one fails.
SETTINGS_FILE_LOCK_WAIT_S: float = 3.0
# How often the bounded wait retries the non-blocking lock attempt. A poll costs one open and one flock call.
SETTINGS_FILE_LOCK_POLL_S: float = 0.02

# --- embedding backfill scan cap (moved here, #531) --------------------------
# Bounds how many candidate rows are even *examined* (read from `memories`)
# in one tick: cheap (an indexed WHERE-clause row), independent of
# `batch_size` which bounds actual embed *compute* (the expensive part).
# Keeps a corpus with a long run of skip-worthy short/failing rows from
# spinning CPU scanning without also embedding anything.
# The default of `scan_cap` in `brain.memory.embedding_backfill`.
DEFAULT_SCAN_CAP = 500

# --- judge self-tune cadence and grades (moved here, #531) -------------------
# The persisted cadence file name and the weekly interval of the judge self-tune tick
# (`brain.memory.judge_selftune`, `brain.bridge.supervisor`).
JUDGE_TUNE_CADENCE_FILE = "judge_selftune_cadence.json"
JUDGE_TUNE_INTERVAL_HOURS = 168.0

# Tune-grade identifiers (spec §1 table). The weight-retrain GRADE selected by RAM tier and, if needed,
# downgraded by the cgroup-aware OOM-safety guard. Plain strings (not an enum) so they serialize directly
# into logging/marker state without a translation layer. The order is weakest box to beefiest box.
JUDGE_TUNE_GRADE_KNOB_REFIT = "knob_refit"  # weak box, always-safe floor
JUDGE_TUNE_GRADE_LORA = "lora"  # mid box
JUDGE_TUNE_GRADE_FULL_FT = "full_ft"  # beefy box
JUDGE_TUNE_GRADE_ORDER = (JUDGE_TUNE_GRADE_KNOB_REFIT, JUDGE_TUNE_GRADE_LORA, JUDGE_TUNE_GRADE_FULL_FT)

# --- forgetting thresholds (moved here, #531) --------------------------------
# `brain.forgetting.policy`: the state machine and the exemption gates.
#   FADE_THRESHOLD     salience < this drives active -> fading
#   LOST_THRESHOLD     salience < this for LOST_PASS_COUNT consecutive passes drives fading -> lost
#   LOST_PASS_COUNT    two-pass safety guard against single-pass dips
#   RECENT_LIVED_HOURS memories younger than this are exempt (see the note on the value)
# LOST_THRESHOLD and LOST_PASS_COUNT are the deletion floor (the cd808dbc invariant): a retention lever only delays
# fading, it does not lower the deletion floor. FADE_IMPORTANCE_GAIN and HORIZON_IMPORTANCE_GAIN below are such levers.
FADE_THRESHOLD = 0.25
LOST_THRESHOLD = 0.10
LOST_PASS_COUNT = 2
RECENT_LIVED_HOURS = 720.0  # 30 days. NOTE: compared against WALL-CLOCK age (see is_exempt), not lived-age: the name is historical. Recent memories stay fully active+verbatim for a month.

# P3 retention rework, Change 2: importance lowers the EFFECTIVE fade gate
# (never LOST_THRESHOLD/LOST_PASS_COUNT: those stay the final deletion gate,
# per the invariant above: a lever only delays fading, it does not lower
# the deletion floor). r = importance_ratio(memory.importance)
# (`brain/memory/memory_utils.py`); a no-op at r=0 (importance 0 -> today's
# behavior). Tuned against the C2 simulation so importance-10 survives >= 2
# lived-years while importance-0 is untouched and the response is monotonic +
# never-accelerating. The emotion fade does NOT use this gain: it lowers the
# fade gate of the forgetting pass only.
FADE_IMPORTANCE_GAIN = 2.0

# P3 retention rework, Change 2: importance extends the freshness age-horizon
# of `brain.forgetting.salience` (the OTHER decay lever, alongside
# FADE_IMPORTANCE_GAIN). This is the gain of `importance_stretch`
# (`brain/memory/memory_utils.py`), the one function that holds the formula;
# the forgetting freshness horizon and the emotion fade (`brain.emotion.engine_utils.fade`)
# both take their stretch from it, and nothing else reads this gain. A no-op at
# importance 0 (the stretch is 1.0). Tuned against the C2 simulation alongside
# FADE_IMPORTANCE_GAIN so importance-10 survives >= 2 lived-years.
HORIZON_IMPORTANCE_GAIN = 52.0

# --- narrative memory arcs (moved here, #531) --------------------------------
# `brain.narrative_memory.policy` (spec §2): an arc closes when its last extension was this many lived-hours ago,
# and keeps at most this many members (the lowest-salience members are evicted past it).
ARC_STALE_LIVED_HOURS: float = 72.0
MAX_ARC_MEMBERS: int = 50

# Hebbian-OR-embedding membership test of `brain.narrative_memory.membership` (spec §4). One membership rule: a
# memory joins an arc when its Hebbian weight to a member reaches the first threshold OR its cosine similarity to the
# arc centroid reaches the second.
MEMBER_HEBBIAN_THRESHOLD: float = 3.0
MEMBER_EMBEDDING_THRESHOLD: float = 0.6

# --- attunement (moved here, #531) -------------------------------------------
# Schema version, pattern maturity thresholds (observation counts), the cooldown after an address, and the daily
# Haiku-call budget default of `brain.attunement`. Prefixed because other modules keep their own schema versions.
ATTUNEMENT_SCHEMA_VERSION = "0.0.29"

ATTUNEMENT_MATURITY_FORMING_MIN = 3
ATTUNEMENT_MATURITY_KNOWN_MIN = 10
ATTUNEMENT_MATURITY_FALSIFIED_MAX = 3

ATTUNEMENT_ADDRESS_COOLDOWN_HOURS = 6.0
ATTUNEMENT_DAILY_BUDGET_DEFAULT = 150

# --- kindled peer link caps and disclosure budget (moved here, #531) ---------
# Shared by `brain.kindled_link` (privacy_gate.py, session_engine.py, store.py, relationship.py, transport.py). This
# module imports nothing, so sharing it cannot create an import cycle.
# Pacing + caps (parent design §9). The daily provider cap includes the draft, the privacy-gate reflection, and the one
# allowed revision.
KINDLED_LINK_MIN_OUTBOUND_GAP_SECONDS = 60
KINDLED_LINK_SESSION_MSG_CAP = 24
KINDLED_LINK_SESSION_COOLDOWN_HOURS = 6
KINDLED_LINK_DAILY_OUTBOUND_CAP = 20
KINDLED_LINK_DAILY_PROVIDER_CAP = 60

# Cross-session disclosure budget (parent §12). Per-peer scalar in [0, MAX];
# each sent message debits its texture_score; refills linearly with wall-time.
KINDLED_LINK_BUDGET_MAX = 1.0
KINDLED_LINK_BUDGET_REFILL_PER_DAY = 0.5
KINDLED_LINK_BUDGET_TIGHTEN_THRESHOLD = 0.25
# Floor debit per send: ensures message COUNT depletes the budget even when a
# model self-reports texture_score=0.0, which bounds crumb-extraction via a long
# correspondence of "zero-texture" sends. At this floor ~50 sends exhaust the
# budget before the tighten threshold is hit.
KINDLED_LINK_MIN_SEND_DEBIT = 0.02
# Hard depletion floor (m10): below this a 'send' is held outright, not merely
# tightened to revise. "If you cannot afford the minimum debit, you cannot send."
# Closes the crumb-leak where a depleted budget downgraded send->revise and the
# re-gated revision could still send one KINDLED_LINK_MIN_SEND_DEBIT-floored crumb per attempt.
# Equals KINDLED_LINK_MIN_SEND_DEBIT by construction; named separately for intent + tuning.
KINDLED_LINK_BUDGET_DEPLETED_THRESHOLD = KINDLED_LINK_MIN_SEND_DEBIT

# Per-peer emotion influence cap (parent §14.3, anti love-bomb). Cumulative
# influence within a rolling window decays linearly to zero over the window.
KINDLED_LINK_PEER_EMOTION_WINDOW_CAP = 0.6
KINDLED_LINK_PEER_EMOTION_WINDOW_HOURS = 24.0

# Local inbound flood cap (parent design §9): max envelopes decrypted/processed
# per poll PER PEER. Excess is left un-acked on the relay and surfaced as a
# degraded state; bounds local decrypt/provider work against a flooding peer.
KINDLED_LINK_INBOUND_FLOOD_CAP = 20

# --- emotion readers (B2 phase B) ----------------------------------------------
# The prompt's mood summary (`brain.chat.prompt._build_emotion_summary`) lists this many feelings by long average,
# highest first; None lists them all. Read at use time, inside the function.
EMOTION_SUMMARY_COUNT: int | None = 3
# How each listed feeling is written: "long" (name:L), "long_short" (name:L/S) or "long_difference" (name:L(D), D signed).
EMOTION_SUMMARY_MODE: str = "long"
# The self-model block (`brain.self_model.ambient.render_block`, live branch) names this many channels, the largest
# by absolute difference. Owner-accepted number; named here so the block's code holds no literal.
SELF_MODEL_BLOCK_CHANNEL_COUNT: int = 3

# A feeling counts as high (`engine.feels_high`, the one predicate the readers that used to fire on a fixed level call) when
# its short average today is at or above `lowest + FEELS_HIGH_RANGE_FRACTION * (highest - lowest)`, taken over that feeling's
# daily short averages in the 30-day snapshot (the shift reading, when it reads stronger, makes it high whatever its range), and either
# way only at or above the floor in the next constant. Owner-set, 2026-11-04: "upper half of the range". Read at call time, inside `engine_utils.feels_high_from`.
FEELS_HIGH_RANGE_FRACTION: float = 0.5
# A feeling whose short average is below this level (0 to 10 scale) is never high, whatever its range or its shift: a feeling
# that sits near zero all month has a tiny range, and without a floor its top half would count as "high" (a vulnerability at
# 0.02 started a self-check). The maintainer set 1.0 on 2026-11-10 (#674) from `scripts/measure_feels_high_thresholds.py`, run on the
# synthetic database and on a copy of one real persona: there, firings sit below 0.5 or at 1 and above, so any value from 0.5 to
# 1 gives the same answer, and "high" under 1 in 10 would not be honest. 0.1 still let the research gate open on feelings from
# 0.1 to 0.5, which the plateau rule does not count. The follow-up that makes the floor self-calibrating for a mature kindled
# is #661. Read at call time, inside `engine_utils.feels_high_from`.
FEELS_HIGH_FLOOR: float = 1.0
# Body energy drops by one when MORE THAN this many feelings are high at once (both body projections: the engine path asks
# `engine.feels_high` per feeling, the old path compares to its fixed levels). Today's value, unchanged by S52 and S53 (they
# replaced the feeling levels, not this count). Read at call time in `brain.body.state`.
BODY_ENERGY_HIGH_COUNT: int = 6
