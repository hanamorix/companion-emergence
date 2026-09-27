"""Dev-facing constants — build-time judgment calls, never user tunables.

Sibling of `brain.tunables` (ops tier, user-facing, overridable via
`$KINDLED_HOME/tunables.json`) but the opposite half of the fence (I7): a
value here is EITHER not a runtime behaviour threshold (it exists to make a
mechanism correct or well-sized — a timeout, a batch budget) OR is a
behaviour-adjacent knob the owner has explicitly ruled is dev-level, not
user-tunable (S26: the self-tune gate). Nothing in this module is registered
with `tunables.register` and nothing here is read via `tunables.get_tunable`
— a value moves to `tunables.py` only on an explicit owner decision to make
it user-facing (S6/S47 precedent: the chat-idle lull went the other way,
into tunables.py, because it IS user-facing timing).

Plain module-level constants, not `tunables.register()`-backed: no override
file, no runtime reload. Changing one of these requires a code change.
"""

from __future__ import annotations

# --- memories.db busy timeout (S48, S58) -----------------------------------
# Applied at BOTH `sqlite3.connect(..., timeout=...)` and
# `PRAGMA busy_timeout = ...` at the memories.db connect sites
# (store.py MemoryStore.__init__, embedding_matrix.py's `_load_from_db`).
# Sized above the longest single memories.db write transaction measured in
# 2-plan.md §4.3: clustering's `set_cluster_memberships` at 6.7 s on
# F-bob20k on the dev host; Phoebe-class hardware is 3-30x slower and was
# IO-saturated (O15/O23) -> ~20-27 s estimated; 30 s is the next round value
# above that. Other memories.db writers (this repo only) stay on their own
# 5 s default (hebbian.db, works.db, soul.db, kindled_link's stores) — this
# constant is memories.db-specific, not a global sqlite default.
MEMORIES_DB_BUSY_TIMEOUT_S: float = 30.0

# --- F2c self-tune gate (S7, S26) ------------------------------------------
# ">a handful" gate for `judge_selftune._run_judge_selftune_tick`: the
# weekly tick only fires once MORE than this many new Haiku decisions have
# accumulated. OWNER 2026-09-26 raised the provisional 20 to 200 ("It should
# be 200!"), and ruled it a dev-level tunable rather than a user-tunable
# behaviour threshold (S26): it exists so there is enough data to derive
# meaningful drift signal and so calibration does not react to background
# noise, not to shape user-visible behaviour the way the chat-idle lull
# does — the behaviour threshold that DOES belong to I3 is the calibration's
# own result, not this gate. See `brain.memory.judge_selftune.
# JUDGE_TUNE_GATE_HANDFUL_DECISIONS`, which re-exports this value.
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
