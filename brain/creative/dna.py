"""brain.creative.dna — load/save creative_dna with default fallback.

Per spec §3.1: creative_dna is the brain's evolved writing voice (active /
emerging / fading + influences/avoid). Distinct from voice.md, which is the
authored static persona.

This module owns the file I/O. The crystallizer
(brain/growth/crystallizers/creative_dna.py) is the only auto-evolution
caller. Migration imports from OG via brain/migrator/og_journal_dna.py.

Atomic writes via save_with_backup; reads via attempt_heal so corruption
falls back to .bak rotation; if all backups corrupt, framework default
applies (per spec §3.2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brain.health.adaptive import compute_treatment
from brain.health.attempt_heal import attempt_heal, save_with_backup

logger = logging.getLogger(__name__)

CREATIVE_DNA_FILENAME = "creative_dna.json"
_DEFAULT_PATH = Path(__file__).parent / "default_creative_dna.json"


def _default_factory() -> dict[str, Any]:
    """Return the framework-shipped default. Read from the bundled JSON file."""
    return json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))


def _validate_schema(data: object) -> None:
    """Minimal schema validation. Raises ValueError if malformed."""
    if not isinstance(data, dict):
        raise ValueError("creative_dna must be a dict")
    required_keys = {"version", "core_voice", "strengths", "tendencies", "influences", "avoid"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"creative_dna missing keys: {missing}")
    tendencies = data.get("tendencies", {})
    if not isinstance(tendencies, dict):
        raise ValueError("creative_dna.tendencies must be a dict")
    for bucket in ("active", "emerging", "fading"):
        if bucket not in tendencies:
            raise ValueError(f"creative_dna.tendencies missing bucket: {bucket}")
        if not isinstance(tendencies[bucket], list):
            raise ValueError(f"creative_dna.tendencies.{bucket} must be a list")


def load_creative_dna(persona_dir: Path) -> dict[str, Any]:
    """Load creative_dna.json. Falls back to framework default if missing/corrupt.

    On first-call (file missing), the default is COPIED to the persona dir so
    subsequent reads are stable. Per spec §5.7: brand-new personas grow into
    their style from this default; first crystallizer tick populates active/
    emerging from observed patterns.
    """
    path = persona_dir / CREATIVE_DNA_FILENAME

    if not path.exists():
        # Seed the persona dir with the default. Never bypass attempt_heal
        # for live usage — but on first-creation it's a clean copy.
        default = _default_factory()
        save_creative_dna(persona_dir, default)
        return default

    data, anomaly = attempt_heal(path, _default_factory, schema_validator=_validate_schema)
    if anomaly is not None:
        logger.warning(
            "creative_dna at %s anomaly %s (action=%s); using recovered/default",
            path,
            anomaly.kind,
            anomaly.action,
        )
    return data


def save_creative_dna(persona_dir: Path, data: dict[str, Any]) -> None:
    """Atomic write with .bak rotation. Validates schema before writing."""
    _validate_schema(data)
    path = persona_dir / CREATIVE_DNA_FILENAME
    persona_dir.mkdir(parents=True, exist_ok=True)
    try:
        treatment = compute_treatment(persona_dir, CREATIVE_DNA_FILENAME)
        backup_count = treatment.backup_count
    except Exception:  # noqa: BLE001
        backup_count = 3
    save_with_backup(path, data, backup_count=backup_count)
