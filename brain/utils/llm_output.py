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
# What JSON allows right after an object's `{`: whitespace, then a key's quote or the closing brace. Any other `{` cannot start an
# object, so the decoder need not be asked: its failure builds a JSONDecodeError that counts the lines before the position
# (linear in the text), which made a reply of many prose `{` quadratic even with the block scan made linear.
_CAN_OPEN_OBJECT = re.compile(r'\{[ \t\n\r]*["}]')


def _balanced_block(text: str, start: int) -> str | None:
    """The `{...}` block that opens at ``start``, counting braces OUTSIDE string literals only (#478; moved here from soul/review so every model-reply reader shares it), or None if it never closes.

    One scan from ``start``: linear in the text left, so a caller that asks for MANY starts must use ``_balanced_ends``.
    """
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


_OUT, _IN, _ESC = 0, 1, 2  # the string-literal scanner's states: outside a string, inside one, just after a backslash inside one


def _string_step(state: int, ch: str) -> int:
    if state == _IN:
        return _ESC if ch == "\\" else (_OUT if ch == '"' else _IN)
    if state == _ESC:
        return _IN
    return _IN if ch == '"' else _OUT


def _balanced_ends(text: str) -> dict[int, int]:
    """For every `{` that opens a balanced block in ``text``: ``{start: end}`` (``end`` exclusive), the same answer as
    ``_balanced_block(text, start)`` for each (end = start + len), computed in ONE linear pass instead of a scan per start.

    A scan's string state does not depend on its depth, so three scanners started at position 0 in each of the three states
    cover every start: a start sits outside a string by definition, so it follows whichever of the three is outside at that
    position (they agree from there on, the scanner being deterministic). Each scanner has a running depth (``{`` minus ``}``
    outside strings); the block that opens at ``s`` ends at the first later position where that depth is back to its value
    before ``s``, found for all starts at once by walking right to left with the nearest position of each depth value.
    Starts no scanner has outside a string (none should exist) are left to the direct scan.
    """
    n = len(text)
    braces = [i for i, ch in enumerate(text) if ch == "{"]
    if not braces:
        return {}
    states: list[list[int]] = []  # states[q][j]: scanner q's state BEFORE the character at j
    depths: list[list[int]] = []  # depths[q][k]: scanner q's depth after the first k characters
    for q in (_OUT, _IN, _ESC):
        state, depth = q, 0
        st = [0] * n
        dp = [0] * (n + 1)
        for j, ch in enumerate(text):
            st[j] = state
            if state == _OUT:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            state = _string_step(state, ch)
            dp[j + 1] = depth
        states.append(st)
        depths.append(dp)
    owner = {}  # start -> the first scanner that is outside a string there
    for s in braces:
        for q in (_OUT, _IN, _ESC):
            if states[q][s] == _OUT:
                owner[s] = q
                break
    ends: dict[int, int] = {}
    for q in (_OUT, _IN, _ESC):
        mine = [s for s in braces if owner.get(s) == q]
        if not mine:
            continue
        wanted = set(mine)
        nearest: dict[int, int] = {}  # depth value -> the nearest position k > current with depths[q][k] == value
        dp = depths[q]
        for k in range(n, -1, -1):
            if k in wanted:
                found = nearest.get(dp[k])
                if found is not None:
                    ends[k] = found
            nearest[dp[k]] = k
    for s in braces:
        if s not in owner:  # not expected; keep the answer exact anyway
            block = _balanced_block(text, s)
            if block is not None:
                ends[s] = s + len(block)
    return ends


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
    ends: dict[int, int] | None = None  # `_balanced_ends`, built once, on the first `{` that fails to decode
    inside_failed_until = 0  # an object that opened inside a block that did not decode belongs to that malformed block
    start = text.find("{")
    while start >= 0:
        if start < inside_failed_until:
            start = text.find("{", start + 1)
            continue
        decoded = None
        if _CAN_OPEN_OBJECT.match(text, start):
            try:
                decoded = decoder.raw_decode(text, start)
            except (ValueError, RecursionError):  # RecursionError: nested past the limit (#406's tuple)
                decoded = None
        if decoded is None:
            if ends is None:
                ends = _balanced_ends(text)  # one linear pass: a scan per failed start is quadratic on many unclosed `{`
            balanced = text[start : ends[start]] if start in ends else None
            if balanced is not None:
                inside_failed_until = start + len(balanced)
            elif _OBJECT_START.match(text, start):
                # An object that opens like JSON (`{"`) and never closes is a truncated reply: everything after it is part of it.
                # (A `{` that opens prose, such as "I considered { maybe", is not, and does not hide a later answer.)
                inside_failed_until = len(text)
            if first_balanced is None:
                first_balanced = balanced  # None stays None: the table already says this start never closes
        else:
            obj, end = decoded
            if isinstance(obj, dict):
                block = text[start:end]
                if prefer_key is not None and prefer_key in obj:
                    return block
                if first_object is None:
                    first_object = block
            if first_balanced is None:
                first_balanced = _balanced_block(text, start)  # a decoded object closes: one scan, bounded by it
        start = text.find("{", start + 1)
    return first_object if first_object is not None else first_balanced
