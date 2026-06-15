"""Maker tuning constants. Calibration target: a making roughly every few
emotionally-active days, never multiple per day. Adjust against live
maker_charge.json traces, not in isolation."""
W_EMOTION = 0.15
W_SOUL = 2.0
W_DREAM = 0.6
DECAY_PER_HOUR = 0.97          # ~half-life ~23h of idle charge
DISCHARGE_THRESHOLD = 12.0
COOLDOWN_HOURS = 18.0
DAILY_CAP = 2
FAILED_MAKE_CHARGE = 6.0       # partial value after a malformed making (below threshold)
SHARE_DELAY_HOURS = 12.0       # eventual_share → feed readiness delay
