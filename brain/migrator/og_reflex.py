"""Extract OG reflex arc dicts from reflex_engine.py via AST.

We parse the file's AST rather than importing it — the OG module's
imports depend on nell_brain.py and other top-level modules that are
not available in the new framework's environment.
"""

from __future__ import annotations

import ast
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
