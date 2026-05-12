"""New v0.0.10 candidate emitters: reflex firings and research completions.

Each emitter has a gate function that decides whether to emit, plus an
emit function that calls into brain.initiate.emit.emit_initiate_candidate.

Rejected gate checks write to gate_rejections.jsonl (separate from the
main audit log — rejection volume would otherwise drown signal).

Thresholds are loaded from gate_thresholds.json in the persona dir,
with defaults baked in. Operator can tune without code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateThresholds:
    reflex_confidence_min: float = 0.70
    reflex_flinch_intensity_min: float = 0.60
    reflex_anti_flood_hours: float = 4.0
    research_maturity_min: float = 0.75
    research_topic_overlap_min: float = 0.30
    research_freshness_minutes: float = 30
    meta_anti_flood_minutes: float = 30
    meta_max_queue_depth: int = 6


def load_gate_thresholds(persona_dir: Path) -> GateThresholds:
    """Load thresholds from <persona_dir>/gate_thresholds.json with defaults.

    Defaults defined on the dataclass. Persona file overrides any subset
    of fields. Missing file => all defaults.
    """
    path = persona_dir / "gate_thresholds.json"
    if not path.exists():
        return GateThresholds()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("gate_thresholds.json read failed (%s); using defaults", exc)
        return GateThresholds()
    if not isinstance(raw, dict):
        logger.warning("gate_thresholds.json is not a JSON object; using defaults")
        return GateThresholds()
    valid_names = {f.name for f in fields(GateThresholds)}
    overrides: dict[str, Any] = {k: v for k, v in raw.items() if k in valid_names}
    return GateThresholds(**overrides)
