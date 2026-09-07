"""_format_claude_context_block adds a 'Current time' preamble line, rendered
in the companion's local time (issue #217), not UTC."""

import json
import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import _format_claude_context_block

# A fixed non-UTC zone (not the host's OS timezone): brain.utils.time.to_local
# treats a non-zero offset as already local and returns it unconverted,
# whereas a UTC-aware `now` gets re-converted via astimezone() to whatever
# zone the test host runs in. Using a fixed LA offset makes the assertion
# deterministic regardless of the CI runner's TZ — same pattern as
# tests/unit/brain/initiate/test_gates.py.
_LOCAL = ZoneInfo("America/Los_Angeles")


def test_preamble_includes_current_time_in_local_zone():
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
    ]
    now = datetime(2026, 5, 20, 7, 30, 0, tzinfo=_LOCAL)
    block = _format_claude_context_block(msgs, includes_latest_user=True, now=now)
    # Local-offset ISO-8601, no UTC 'Z' suffix — LA is UTC-7 in May (PDT).
    assert "Current time: 2026-05-20T07:30:00-07:00." in block
    assert re.search(r"Current time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", block) is None


def test_preamble_explains_ts_field():
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b", ts="2026-05-20T10:05:00Z"),
    ]
    block = _format_claude_context_block(msgs, includes_latest_user=True)
    assert "ts" in block  # the explanatory line about the field
    assert "wall-clock" in block.lower()


def test_per_message_ts_rendered_in_local_zone_not_utc():
    """Discriminating regression test for #217: a per-message ts stored as
    UTC must render as local wall-clock time in the JSONL the companion
    reads, not as the raw UTC string relabeled.

    The exact offset depends on the CI runner's OS timezone (unlike the
    `now` anchor above, a stored `ts` string can't be swapped for an
    already-non-UTC-offset test double without changing what's under test —
    the conversion itself is the thing being verified). So this asserts two
    things that hold true regardless of the runner's zone: (1) the raw 'Z'
    (UTC) suffixed string must NOT survive verbatim into the rendered JSONL
    - fails pre-fix, where it's passed through unchanged - and (2) the
    rendered value must resolve back to the exact same instant, so the
    conversion is a true zone change and not just a stray reformat."""
    raw_ts = "2026-05-20T17:00:00Z"
    msgs = [ChatMessage(role="user", content="a", ts=raw_ts)]
    block = _format_claude_context_block(msgs, includes_latest_user=True)
    records = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    rendered_ts = records[0]["ts"]

    assert rendered_ts != raw_ts
    assert not rendered_ts.endswith("Z")
    original_instant = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
    assert datetime.fromisoformat(rendered_ts).astimezone(UTC) == original_instant
