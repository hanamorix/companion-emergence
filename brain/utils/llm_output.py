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


_OBJECT_START = re.compile(r'\{\s*"')  # the opening of a JSON object with at least one key


def _balanced_block(text: str, start: int) -> str | None:
    """The `{...}` block that opens at ``start``, counting braces OUTSIDE string literals only (#478; moved here from soul/review so every model-reply reader shares it), or None if it never closes."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def find_json_object_block(text: str, prefer_key: str | None = None) -> str | None:
    """Find the JSON object that is the model's answer in ``text``. Tolerant of preamble and of braces inside strings (#478).

    Tries each ``{`` in turn with ``json.JSONDecoder().raw_decode``: the first object that carries ``prefer_key`` wins, else the
    first object that decodes at all. An object that opens inside a ``{...}`` block that did NOT decode is part of that malformed
    reply and is not taken as the answer (a decision quoted inside invalid JSON must not be accepted). That includes a TRUNCATED outer object (one that opens
    like JSON, `{"`, and never closes): everything after it is part of it. When no ``{`` starts a decodable object (a malformed reply) it falls back to the first
    string-aware balanced block, so a caller can still report "the reply is not valid JSON" rather than "no JSON object"; None
    when no `{` starts a block that closes.
    """
    decoder = json.JSONDecoder()
    first_object: str | None = None
    first_balanced: str | None = None
    inside_failed_until = 0  # an object that opened inside a block that did not decode belongs to that malformed block
    start = text.find("{")
    while start >= 0:
        if start < inside_failed_until:
            start = text.find("{", start + 1)
            continue
        try:
            obj, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):  # RecursionError: nested past the limit (#406's tuple)
            obj = None
            balanced = _balanced_block(text, start)
            if balanced is not None:
                inside_failed_until = start + len(balanced)
            elif _OBJECT_START.match(text, start):
                # An object that opens like JSON (`{"`) and never closes is a truncated reply: everything after it is part of it.
                # (A `{` that opens prose, such as "I considered { maybe", is not, and does not hide a later answer.)
                inside_failed_until = len(text)
        else:
            balanced = None
            if isinstance(obj, dict):
                block = text[start:end]
                if prefer_key is not None and prefer_key in obj:
                    return block
                if first_object is None:
                    first_object = block
        if first_balanced is None:
            first_balanced = balanced if balanced is not None else _balanced_block(text, start)
        start = text.find("{", start + 1)
    return first_object if first_object is not None else first_balanced
