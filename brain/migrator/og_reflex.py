"""Extract OG reflex arc dicts from reflex_engine.py via AST, and migrate them.

We parse the file's AST rather than importing it — the OG module's
imports depend on nell_brain.py and other top-level modules that are
not available in the new framework's environment.
"""

from __future__ import annotations

import ast
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ACTION_RENAMES = {
    "generate_story_pitch": "generate_pitch",
    "write_journal": "generate_journal",
    "write_gift": "generate_gift",
    "write_memory": "generate_reflection",
}

_OUTPUT_RENAMES = {
    "journal": "reflex_journal",
    "gifts": "reflex_gift",
    "memories": "reflex_memory",
}


def extract_arcs_from_og(og_reflex_engine_path: Path) -> list[dict[str, Any]]:
    """Return a list of new-schema arc dicts extracted from OG source."""
    source = og_reflex_engine_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    arcs_node: ast.Dict | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REFLEX_ARCS":
                    if isinstance(node.value, ast.Dict):
                        arcs_node = node.value
                        break
            if arcs_node is not None:
                break

    if arcs_node is None:
        raise ValueError(f"REFLEX_ARCS assignment not found in {og_reflex_engine_path}")

    result: list[dict[str, Any]] = []
    for key_node, value_node in zip(arcs_node.keys, arcs_node.values, strict=True):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        if not isinstance(value_node, ast.Dict):
            continue
        name = key_node.value
        arc_dict = ast.literal_eval(value_node)
        transformed = _transform_og_arc(name, arc_dict)
        if transformed is not None:
            result.append(transformed)
    return result


def _transform_og_arc(name: str, og: dict[str, Any]) -> dict[str, Any] | None:
    """Map one OG arc dict to the new schema. Returns None if invalid."""
    required = (
        "trigger",
        "days_since_min",
        "action",
        "output",
        "cooldown_hours",
        "description",
        "prompt_template",
    )
    for key in required:
        if key not in og:
            return None

    return {
        "name": name,
        "description": str(og["description"]),
        "trigger": dict(og["trigger"]),
        "days_since_human_min": float(og["days_since_min"]),
        "cooldown_hours": float(og["cooldown_hours"]),
        "action": _ACTION_RENAMES.get(og["action"], og["action"]),
        "output_memory_type": _OUTPUT_RENAMES.get(og["output"], og["output"]),
        "prompt_template": str(og["prompt_template"]),
    }


def _default_arcs_path() -> Path:
    """Return the bundled default_reflex_arcs.json shipped with the engines module."""
    return Path(__file__).parent.parent / "engines" / "default_reflex_arcs.json"


def migrate_reflex_arcs(
    persona_dir: Path,
    *,
    og_reflex_engine_path: Path | None = None,
    force: bool = True,
) -> list[dict[str, Any]]:
    """Write (or update) reflex_arcs.json in `persona_dir` with provenance stamps.

    Behaviour
    ---------
    - If `og_reflex_engine_path` is provided, extract arcs from OG source via AST.
    - Otherwise, load the bundled ``default_reflex_arcs.json``.
    - Stamp every arc with ``created_by="og_migration"`` and a ``created_at``
      timestamp (ISO 8601 UTC).
    - **Idempotent**: if ``reflex_arcs.json`` already exists, re-read it and
      preserve the original ``created_at`` / ``created_by`` for any arc whose
      name is already present.  Only newly introduced arc names get a fresh
      ``created_at`` of *now*.
    - Writes atomically via a ``.new`` temp file + ``os.replace``.

    Returns the list of arc dicts written.
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    arcs_path = persona_dir / "reflex_arcs.json"

    # 1. Build the source arc list (OG extraction or default bundled arcs).
    if og_reflex_engine_path is not None:
        source_arcs: list[dict[str, Any]] = extract_arcs_from_og(og_reflex_engine_path)
    else:
        default_data = json.loads(_default_arcs_path().read_text(encoding="utf-8"))
        source_arcs = list(default_data.get("arcs", []))

    # 2. Load existing arcs (for idempotency — preserve created_at/created_by).
    existing_by_name: dict[str, dict[str, Any]] = {}
    if arcs_path.exists():
        try:
            existing_data = json.loads(arcs_path.read_text(encoding="utf-8"))
            for a in existing_data.get("arcs", []):
                if isinstance(a, dict) and "name" in a:
                    existing_by_name[a["name"]] = a
        except (json.JSONDecodeError, OSError):
            pass  # corrupt existing file — treat as empty, re-stamp everything

    # 3. Stamp each arc with provenance.
    now_iso = datetime.now(UTC).isoformat()
    arcs_to_write: list[dict[str, Any]] = []
    for arc in source_arcs:
        arc_dict = dict(arc)  # copy — don't mutate source
        existing = existing_by_name.get(arc_dict.get("name", ""))
        if existing is not None and "created_at" in existing:
            # Already stamped on a previous migration run — preserve the original.
            arc_dict["created_by"] = existing.get("created_by", "og_migration")
            arc_dict["created_at"] = existing["created_at"]
        else:
            # First time seeing this arc — stamp now.
            arc_dict.setdefault("created_by", "og_migration")
            arc_dict.setdefault("created_at", now_iso)
        arcs_to_write.append(arc_dict)

    # 4. Atomic write.
    tmp_path = arcs_path.with_suffix(arcs_path.suffix + ".new")
    tmp_path.write_text(
        json.dumps({"version": 1, "arcs": arcs_to_write}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, arcs_path)

    return arcs_to_write
