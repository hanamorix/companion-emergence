"""Extract OG nell_interests.json into new-schema interest dicts.

Pure JSON read (no AST needed — OG interests live in a JSON file, not
a Python module). Scope classification: interests whose topic mentions
any name from the persona's soul.json are tagged "internal" (never
web-search Hana herself, Jordan, etc). Everything else defaults to
"either" — safe default, web-priority but memory-fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brain.utils.time import iso_utc, parse_iso_utc


def extract_interests_from_og(
    og_interests_path: Path, *, soul_names: set[str]
) -> list[dict[str, Any]]:
    """Return new-schema interest dicts extracted from OG nell_interests.json.

    Raises FileNotFoundError if the path doesn't exist, ValueError if it
    can't be parsed. soul_names is lowercased for case-insensitive match.
    """
    if not og_interests_path.exists():
        raise FileNotFoundError(f"OG interests file not found: {og_interests_path}")

    try:
        data = json.loads(og_interests_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {og_interests_path}: {exc}") from exc

    og_items = data.get("interests", [])
    if not isinstance(og_items, list):
        raise ValueError(f"{og_interests_path}: 'interests' is not a list")

    soul_lower = {s.lower() for s in soul_names}

    result: list[dict[str, Any]] = []
    for item in og_items:
        if not isinstance(item, dict):
            continue
        transformed = _transform_interest(item, soul_lower)
        if transformed is not None:
            result.append(transformed)
    return result


def extract_soul_names_best_effort(og_dir: Path) -> set[str]:
    """Read soul names from NellBrain/data/nell_soul.json. Best-effort; returns
    empty set if file missing/corrupt.
    """
    candidates = [
        og_dir / "nell_soul.json",
        og_dir / "data" / "nell_soul.json",
        og_dir.parent / "nell_soul.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return set()
            names: set[str] = set()
            for c in data.get("crystallizations", []):
                who = c.get("who_or_what", "")
                if isinstance(who, str) and who:
                    names.update(w.lower() for w in who.split() if len(w) > 2)
            return names
    return set()


def _transform_interest(og: dict[str, Any], soul_lower: set[str]) -> dict[str, Any] | None:
    required = (
        "id",
        "topic",
        "pull_score",
        "first_seen",
        "last_fed",
        "feed_count",
        "source_types",
        "related_keywords",
        "notes",
    )
    for key in required:
        if key not in og:
            return None

    topic = str(og["topic"])
    scope = _classify_scope(topic, soul_lower)

    # Normalise timestamps through parse → iso to collapse to Z format.
    first_seen = iso_utc(parse_iso_utc(og["first_seen"]))
    last_fed = iso_utc(parse_iso_utc(og["last_fed"]))

    return {
        "id": str(og["id"]),
        "topic": topic,
        "pull_score": float(og["pull_score"]),
        "scope": scope,
        "related_keywords": list(og["related_keywords"]),
        "notes": str(og["notes"]),
        "first_seen": first_seen,
        "last_fed": last_fed,
        "last_researched_at": None,
        "feed_count": int(og["feed_count"]),
        "source_types": list(og["source_types"]),
    }


def _classify_scope(topic: str, soul_lower: set[str]) -> str:
    topic_lower = topic.lower()
    for name in soul_lower:
        if name in topic_lower:
            return "internal"
    return "either"
