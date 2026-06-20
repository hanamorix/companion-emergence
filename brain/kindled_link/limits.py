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
