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
