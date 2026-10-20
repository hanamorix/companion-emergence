"""Robust JSON extraction from LLM responses.

LLMs sometimes wrap JSON output in prose ("Here is my answer: {...} Thanks!")
or markdown fences (```json ... ```). This helper extracts the embedded JSON
object tolerantly: fenced match first (handles multi-line and nested braces),
loose fallback (first { to last }).

Used by:
- brain.initiate.reflection.parse_structured_response (D-reflection output)
- brain.engines.research._compute_topic_overlap_via_haiku (topic-overlap score)
"""

from __future__ import annotations

import json
import re

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_LOOSE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json_object(raw: str) -> str:
    """Return the JSON object substring inside `raw`.

    Raises ValueError if no JSON object can be located.
    """
    fenced = _JSON_FENCE.search(raw)
    if fenced is not None:
        return fenced.group(1)
    loose = _JSON_LOOSE.search(raw)
    if loose is None:
        raise ValueError("no JSON object found in response")
    return loose.group(1)


def loads_json_object_ex(raw: str) -> tuple[dict, bool]:
    """Like :func:`loads_json_object`, plus whether the object was found INSIDE PROSE.

    The flag is False when the object is the whole reply or the whole reply is just its fenced block, and True when other text
    surrounds it. A caller that expects particular keys uses it: an object that is the whole reply is an answer (possibly an empty
    one), but an unrelated object found in a sentence is not.
    """
    try:
        data = json.loads(raw.strip())
        in_prose = False
    except ValueError:
        fenced = _JSON_FENCE.search(raw)
        if fenced is not None:
            matched, text = fenced.group(0), fenced.group(1)
        else:
            loose = _JSON_LOOSE.search(raw)
            if loose is None:
                raise ValueError("no JSON object found in response") from None
            matched, text = loose.group(1), loose.group(1)
        data = json.loads(text)
        in_prose = bool(raw.replace(matched, "", 1).strip())
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data, in_prose


def loads_json_object(raw: str) -> dict:
    """Parse a reply that should be one JSON object: the whole reply first,
    then the tolerant ``extract_json_object`` scrape. Raises ValueError when no
    object can be read (#364)."""
    return loads_json_object_ex(raw)[0]
