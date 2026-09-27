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
