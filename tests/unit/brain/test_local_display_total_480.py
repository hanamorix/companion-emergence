"""#480: `local_display` is a display seam and must be TOTAL: it returns the raw text for any stamp it cannot render, and never raises.

It caught only (ValueError, TypeError), so a non-string input (`AttributeError` in `parse_iso_utc`) and a stamp with no local
equivalent (`OverflowError` from `astimezone()`: year 1 in a zone behind UTC, year 9999 in a zone ahead) escaped, and
`initiate/ambient.py` calls it with no guard. Now it catches everything at the seam, and the review tick's call to the outbound-recall
block is guarded too.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.utils.time import local_display

pytestmark = pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")


@pytest.fixture
def zone(monkeypatch):
    def set_zone(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield set_zone
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


@pytest.mark.parametrize("bad", [None, 5, 1.5, b"2026-01-01T00:00:00Z", ["x"], {"a": 1}])
def test_a_non_string_stamp_does_not_raise(bad):
    assert local_display(bad) == ""  # nothing to echo for a value that is not text


def test_year_one_in_a_zone_behind_utc_returns_the_raw_text(zone):
    zone("America/Los_Angeles")

    assert local_display("0001-01-01T00:00:00Z") == "0001-01-01T00:00:00Z"


def test_year_9999_in_a_zone_ahead_of_utc_returns_the_raw_text(zone):
    zone("Pacific/Kiritimati")

    assert local_display("9999-12-31T23:59:59Z") == "9999-12-31T23:59:59Z"


@pytest.mark.parametrize("junk", ["not a time", "", "2026-13-45T99:99:99Z"])
def test_a_malformed_string_still_returns_the_raw_text(junk):
    # PIN: this already worked.
    assert local_display(junk) == junk


def test_a_normal_stamp_still_converts(zone):
    # PIN
    zone("America/Los_Angeles")

    assert local_display("2026-06-01T12:00:00Z") == "2026-06-01T05:00:00"


def test_the_outbound_recall_block_survives_a_stamp_with_no_local_equivalent(zone, monkeypatch, tmp_path):
    from brain.initiate import ambient

    zone("Pacific/Kiritimati")
    row = SimpleNamespace(
        ts="9999-12-31T23:59:59Z",
        decision="send_quiet",
        subject="a far-future stamp",
        delivery={"current_state": "acknowledged_unclear"},
    )
    monkeypatch.setattr(ambient, "read_recent_audit", lambda *a, **k: [row])

    block = ambient.build_outbound_recall_block(tmp_path)

    assert block is not None and "a far-future stamp" in block and "9999-12-31T23:59:59Z" in block


def test_the_narrative_time_helper_still_returns_empty_for_an_unrenderable_stamp(zone):
    # PIN: its raw-text rule differs (never echo an unparseable value), and its own widened except is no longer needed.
    from brain.bridge.provider import _narrative_time

    zone("America/Los_Angeles")

    assert _narrative_time("0001-01-01T00:00:00Z") == ""
    assert _narrative_time(None) == ""


def test_the_review_tick_guards_its_outbound_block_call():
    # The call at review.py (the D-reflection gate setup) must sit inside a try, so a block that cannot be built never aborts the tick.
    path = Path(__file__).resolve().parents[3] / "brain" / "initiate" / "review.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_outbound_recall_block"]
    assert calls, "the call moved: update this test"

    def in_try_body(node) -> bool:
        while node in parents:
            child, node = node, parents[node]
            if isinstance(node, ast.Try) and child in node.body:
                return True
        return False

    for call in calls:
        assert in_try_body(call), f"build_outbound_recall_block call at review.py:{call.lineno} is not inside a try"
