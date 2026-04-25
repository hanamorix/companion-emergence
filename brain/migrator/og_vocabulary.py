"""Extract persona-vocabulary entries from OG memories.

Pure function: walks memory dicts, collects unique emotion names not in
the framework baseline, returns JSON-shaped entries ready for writing
to `{persona_dir}/emotion_vocabulary.json`. Handles all three OG-user
classes uniformly:

- Nell — every nell_specific emotion she used gets a canonical entry
- Other OG users with same defaults — same canonical entries
- Power users with runtime-registered customs — placeholder entries
  the user can refine later

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md §6
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from brain.emotion._canonical_personal_emotions import _CANONICAL


def extract_persona_vocabulary(
    memories: Iterable[dict],
    *,
    framework_baseline_names: set[str],
) -> list[dict[str, Any]]:
    """Return persona-vocabulary entries for emotions referenced in memories
    that are NOT already shipped by the framework baseline.

    Entries for known nell_specific emotions use canonical descriptions +
    decay values from `_CANONICAL`. Entries for unknown emotion names
    use a placeholder description + sensible default decay (14 days).

    Output is sorted by name for deterministic diffs.
    """
    seen: set[str] = set()
    for mem in memories:
        emotions = mem.get("emotions") if isinstance(mem, dict) else None
        if not isinstance(emotions, dict):
            continue
        seen.update(emotions.keys())

    out: list[dict[str, Any]] = []
    for name in seen - framework_baseline_names:
        if name in _CANONICAL:
            canonical = _CANONICAL[name]
            out.append(
                {
                    "name": canonical.name,
                    "description": canonical.description,
                    "category": "persona_extension",
                    "decay_half_life_days": canonical.decay_half_life_days,
                    "intensity_clamp": canonical.intensity_clamp,
                }
            )
        else:
            out.append(
                {
                    "name": name,
                    "description": "(migrated from OG; edit to refine)",
                    "category": "persona_extension",
                    "decay_half_life_days": 14.0,
                    "intensity_clamp": 10.0,
                }
            )

    out.sort(key=lambda d: d["name"])
    return out
