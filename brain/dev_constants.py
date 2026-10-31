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

# --- heartbeat decay batch time budget (S24, S45, S47) ---------------------
# Caps how long a single heartbeat decay batch (one `memories.db` write
# transaction, `HeartbeatEngine._apply_emotion_decay`) may run before it
# commits and the heartbeat saves its resume cursor, so a decay pass over a
# large corpus never holds the write lock for longer than about this many
# seconds at a stretch (S48/S58's busy-timeout sizing assumes this bound).
# Checked after each row, not each batch (S45). ~2,500 rows/batch at the
# measured 0.4 ms/row in-txn rate (2-plan.md §5.2, F-bob20k). A batch may
# also end earlier if the rows run out.
HEARTBEAT_DECAY_BATCH_BUDGET_S: float = 1.0

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

# --- settings-file lock wait (#529) ------------------------------------------
# How long a writer of $KINDLED_HOME/tunables.toml (the install-wide settings file: the picker, `nell model`, the
# bridge boot pass `sync_file`, the migrations) waits for the file's cross-process lock before it gives up and does not
# write (`brain.utils.file_lock.locked_within`). One constant serves the tunables lock and the model migration's own lock.
# It is DERIVED from constants that already exist, not sampled: the longest the holder itself can keep the lock is
# R_read + (K + 2) * R_replace, where
#   R_replace = _REPLACE_RETRY_SLEEP_S * (1 + 2 + ... + (_REPLACE_RETRIES - 1))        brain/health/attempt_heal.py
#             = 0.05 * 28 = 1.4 s (a replace that keeps failing with PermissionError, Windows)
#   R_read    = the same sum over json_io._READ_ATTEMPTS = 0.05 * (1 + 2) = 0.15 s (Windows only; 0 elsewhere)
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
# the deletion floor). r = clamp(memory.importance / 10, 0, 1); a no-op at
# r=0 (importance 0 -> today's behavior, byte-identical). Tuned against the
# C2 simulation so importance-10 survives >= 2 lived-years while importance-0
# is untouched and the response is monotonic + never-accelerating.
FADE_IMPORTANCE_GAIN = 2.0

# P3 retention rework, Change 2: importance extends the freshness age-horizon
# of `brain.forgetting.salience` (the OTHER decay lever, alongside
# FADE_IMPORTANCE_GAIN). r = clamp(memory.importance / 10, 0, 1); a no-op at
# r=0 (importance 0 -> the horizon there, unchanged). Tuned against the
# C2 simulation alongside FADE_IMPORTANCE_GAIN so importance-10 survives >= 2 lived-years.
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
