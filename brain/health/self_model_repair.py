"""One-time reset of the persisted self-model gap (self_model_state.json).

The v0.0.36 derived read used a total-mass-normalised mean that produced a large
uniform-negative gap artifact (magnitude in the hundreds — "almost everything I
claim to feel is arriving at a fraction of the strength") for any populated
persona. The windowed-peak fix corrects the computation, but a persona that ran
the old code has a bogus OPEN gap persisted, carrying a confabulated note. This
clears ``current_gap`` ONCE so the next reflection tick recomputes honestly and
no false "natural resolution" fires from the collapsing artifact. ``gap_history``
is preserved. Marker-gated → runs at most once per persona.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.self_model import state as sm_state

logger = logging.getLogger(__name__)

_MARKER_FILENAME = "self_model_repair_state.json"


def should_run_self_model_repair(persona_dir: Path) -> bool:
    """True until the one-time repair marker is written."""
    return not (persona_dir / _MARKER_FILENAME).exists()


def run_self_model_repair(persona_dir: Path) -> None:
    """Clear any persisted current_gap once, then write the marker. Fail-soft."""
    try:
        state, _recovered = sm_state.load_or_recover(persona_dir)
        if state.current_gap is not None:
            sm_state.save(
                persona_dir,
                sm_state.SelfModelState(current_gap=None, gap_history=state.gap_history),
            )
            logger.info(
                "self_model_repair: cleared a persisted gap (windowed-peak migration)"
            )
    except Exception:
        logger.exception("self_model_repair: failed (non-fatal)")
    finally:
        try:
            persona_dir.mkdir(parents=True, exist_ok=True)
            (persona_dir / _MARKER_FILENAME).write_text(
                json.dumps({"done": True}), encoding="utf-8"
            )
        except OSError:
            logger.exception("self_model_repair: could not write marker")
