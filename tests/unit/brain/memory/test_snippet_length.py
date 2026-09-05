"""snippet_length — proportional snippet-length formula (recall-reinforcement,
CHANGE 2 — G8/G9/G10/G11).

``snippet_length(body_len) = min(max(SNIPPET_MIN_CHARS, body_len // 5), SNIPPET_MAX_CHARS)``.
Table-driven over the load-bearing lengths named in the criteria: a short
memory (12 chars, at/below the floor), a sub-140 memory (100 chars, the
Defect-A case), a memory just over the cap boundary (200 chars), and a long
memory (800 chars).
"""

from __future__ import annotations

import pytest

from brain.memory.relevance import SNIPPET_MAX_CHARS, SNIPPET_MIN_CHARS, snippet_length


@pytest.mark.parametrize(
    ("body_len", "expected"),
    [
        (12, SNIPPET_MIN_CHARS),  # G11: tiny body — floored, not ~3 useless chars
        (100, 20),  # G9 (Defect-A fix): sub-140 body now truncates, 100 // 5 == 20
        (200, 40),  # 200 // 5 == 40, still below the cap
        (800, SNIPPET_MAX_CHARS),  # G10: long body still caps at 140
    ],
)
def test_g8_snippet_length_formula(body_len: int, expected: int) -> None:
    assert snippet_length(body_len) == expected


def test_g9_sub_140_memory_now_truncates() -> None:
    """Defect-A fix: a memory shorter than 140 but longer than the floor no
    longer surfaces as its own full body — the snippet is strictly shorter,
    so a follow-up read_full_memory always adds real text."""
    body = "x" * 100
    max_chars = snippet_length(len(body))
    assert max_chars < len(body)
    snippet = body[: max_chars - 1].rstrip() + "…"
    assert snippet.endswith("…")
    assert len(snippet) < len(body)


def test_g10_long_memory_caps_at_snippet_max_chars() -> None:
    body = "y" * 800
    max_chars = snippet_length(len(body))
    assert max_chars == SNIPPET_MAX_CHARS
    snippet = body[: max_chars - 1].rstrip() + "…"
    assert snippet.endswith("…")
    assert len(snippet) == SNIPPET_MAX_CHARS


def test_g11_tiny_memory_shows_whole_body() -> None:
    """A body at/below the floor isn't reduced to a useless fragment: its
    snippet_length is >= its own length, so the (len > max_chars) truncation
    gate never fires and the whole tiny body renders."""
    body = "z" * 12
    max_chars = snippet_length(len(body))
    assert max_chars >= len(body)
