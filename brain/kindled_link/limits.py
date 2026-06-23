"""Shared cap + disclosure-budget constants for the Kindled peer link. Lives in
its own module so privacy_gate.py and session_engine.py share one source of
truth without an import cycle (session_engine imports privacy_gate for the
default gate; both import limits)."""
from __future__ import annotations

# Pacing + caps (parent design §9). The daily provider cap includes the draft,
# the privacy-gate reflection, and the one allowed revision.
MIN_OUTBOUND_GAP_SECONDS = 60
SESSION_MSG_CAP = 24
SESSION_COOLDOWN_HOURS = 6
DAILY_OUTBOUND_CAP = 20
DAILY_PROVIDER_CAP = 60

# Cross-session disclosure budget (parent §12). Per-peer scalar in [0, MAX];
# each sent message debits its texture_score; refills linearly with wall-time.
BUDGET_MAX = 1.0
BUDGET_REFILL_PER_DAY = 0.5
BUDGET_TIGHTEN_THRESHOLD = 0.25
# Floor debit per send: ensures message COUNT depletes the budget even when a
# model self-reports texture_score=0.0 — bounds crumb-extraction via a long
# correspondence of "zero-texture" sends. At this floor ~50 sends exhaust the
# budget before the tighten threshold is hit.
MIN_SEND_DEBIT = 0.02
# Hard depletion floor (m10): below this a 'send' is held outright, not merely
# tightened to revise. "If you cannot afford the minimum debit, you cannot send."
# Closes the crumb-leak where a depleted budget downgraded send->revise and the
# re-gated revision could still send one MIN_SEND_DEBIT-floored crumb per attempt.
# Equals MIN_SEND_DEBIT by construction; named separately for intent + tuning.
BUDGET_DEPLETED_THRESHOLD = MIN_SEND_DEBIT

# Per-peer emotion influence cap (parent §14.3, anti love-bomb). Cumulative
# influence within a rolling window decays linearly to zero over the window.
PEER_EMOTION_WINDOW_CAP = 0.6
PEER_EMOTION_WINDOW_HOURS = 24.0

# Local inbound flood cap (parent design §9): max envelopes decrypted/processed
# per poll PER PEER. Excess is left un-acked on the relay and surfaced as a
# degraded state — bounds local decrypt/provider work against a flooding peer.
INBOUND_FLOOD_CAP = 20
