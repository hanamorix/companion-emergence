"""#830: the attunement detector's live call is shown the learned patterns her prompt block could show, with their ids.

One shared selection (ambient.select_surfaceable_patterns) feeds both her prompt block and the detector's per-turn
message; the list is recomputed when the detector runs; the backfill call carries no list.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from brain import dev_constants
from brain.attunement.ambient import (
    _ATTUNEMENT_RENDER_CAP,
    build_attunement_block,
    select_surfaceable_patterns,
)
from brain.attunement.detector import _build_user_message, run_detector
from brain.attunement.schemas import LearnedPattern
from brain.attunement.store import BufferTurn, _append_pattern, mark_addressed
from brain.chat.tool_loop import _run_attunement_pass2, _spawn_pass2_attunement

_MSG = "I had a long day today, love."


def _drain_pass2_queue(persona_dir: Path) -> None:
    from brain.bridge import cli_throttle
    from brain.chat import pass2_queue

    cli_throttle.reset()
    pass2_queue.drain_pending(persona_dir)


def _pattern(pid: str, maturity: str = "known", evidence: int = 5, addressed: str | None = None,
             description: str | None = None) -> LearnedPattern:
    return LearnedPattern(
        id=pid,
        category="tone",
        canonical_key=f"key-{pid}",
        description=description or f"description of {pid}",
        evidence_count=evidence,
        maturity=maturity,
        first_seen_at="2026-04-01T00:00:00Z",
        last_confirmed_at="2026-05-31T12:00:00Z",
        last_addressed_at=addressed,
        crystallised_at=None,
        falsified_at=None,
        examples=[],
        schema_version=dev_constants.ATTUNEMENT_SCHEMA_VERSION,
    )


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _buffer() -> list[BufferTurn]:
    return [BufferTurn(id="t1", content=_MSG)]


# --- the shared selection ------------------------------------------------------------------------------------------

def test_selection_orders_known_first_then_evidence_and_drops_ineligible():
    patterns = [
        _pattern("forming-low", "forming", 3),
        _pattern("known-low", "known", 4),
        _pattern("forming-high", "forming", 9),
        _pattern("known-high", "known", 8),
        _pattern("immature", "immature", 99),
        _pattern("recent", "known", 50, addressed=_now_iso()),
        _pattern("old-address", "known", 1, addressed="2020-01-01T00:00:00Z"),
    ]
    ids = [p.id for p in select_surfaceable_patterns(patterns)]
    assert ids == ["known-high", "known-low", "old-address", "forming-high", "forming-low"]


def test_selection_caps_at_the_render_cap():
    patterns = [_pattern(f"p{i:02d}", "known", i) for i in range(_ATTUNEMENT_RENDER_CAP + 4)]
    assert len(select_surfaceable_patterns(patterns)) == _ATTUNEMENT_RENDER_CAP


def test_prompt_block_shows_exactly_the_selected_patterns(tmp_path: Path):
    for i in range(_ATTUNEMENT_RENDER_CAP + 3):
        _append_pattern(tmp_path, _pattern(f"p{i:02d}", "known", i, description=f"Pattern number {i:02d}"))
    block = build_attunement_block(tmp_path)
    selected = select_surfaceable_patterns(
        [_pattern(f"p{i:02d}", "known", i, description=f"Pattern number {i:02d}") for i in range(_ATTUNEMENT_RENDER_CAP + 3)]
    )
    assert {line[2:] for line in block.splitlines() if line.startswith("- ")} == {p.description for p in selected}


# --- the per-turn message ------------------------------------------------------------------------------------------

def test_message_without_a_list_is_unchanged():
    expected = (
        "USER'S RECENT TURNS:\n[turn id=t1] " + _MSG + "\n\n"
        "LATEST REPLY (for context only, do not extract patterns from it):\nok\n\n"
        "Return JSON only. No prose."
    )
    assert _build_user_message(_buffer(), "ok") == expected
    assert _build_user_message(_buffer(), "ok", shown_patterns=None) == expected
    assert _build_user_message(_buffer(), "ok", shown_patterns=[]) == expected


def test_message_with_a_list_carries_heading_ids_and_descriptions_in_order():
    shown = [_pattern("aaaa111122223333", description="first thing"), _pattern("bbbb111122223333", description="second thing")]
    msg = _build_user_message(_buffer(), "ok", shown_patterns=shown)
    assert msg == (
        "USER'S RECENT TURNS:\n[turn id=t1] " + _MSG + "\n\n"
        "LATEST REPLY (for context only, do not extract patterns from it):\nok\n\n"
        "LEARNED PATTERNS:\n- aaaa111122223333: first thing\n- bbbb111122223333: second thing\n\n"
        "Return JSON only. No prose."
    )


def test_a_multi_line_description_stays_on_one_line():
    shown = [_pattern("aaaa111122223333", description="line one\n\nline   two\n- fake: bullet")]
    msg = _build_user_message(_buffer(), "ok", shown_patterns=shown)
    assert "- aaaa111122223333: line one line two - fake: bullet\n" in msg


def test_run_detector_threads_the_list_into_the_message():
    captured: list[str] = []

    def fake_haiku(system_prompt: str, user_message: str, **_kw: object) -> str:
        captured.append(user_message)
        return ""

    with patch("brain.attunement.detector._call_haiku", side_effect=fake_haiku):
        run_detector(buffer_slice=_buffer(), reply_text="ok", shown_patterns=[_pattern("aaaa111122223333")])
        run_detector(buffer_slice=_buffer(), reply_text="ok")
    assert "aaaa111122223333" in captured[0]
    assert "LEARNED PATTERNS" not in captured[1]


# --- the live call -------------------------------------------------------------------------------------------------

def test_live_call_is_given_the_selection_recomputed_when_it_runs(tmp_path: Path):
    _append_pattern(tmp_path, _pattern("stays", "known", 5))
    _append_pattern(tmp_path, _pattern("stamped-later", "known", 9))
    with patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _spawn_pass2_attunement(tmp_path, turn_id="t1", user_message=_MSG, reply_text="ok", buffer_slice=_buffer())
        # an earlier run stamps a pattern after this record was queued and before it runs
        mark_addressed(tmp_path, ["stamped-later"], now_iso=_now_iso())
        _drain_pass2_queue(tmp_path)
    shown = mock_detector.call_args.kwargs["shown_patterns"]
    assert [p.id for p in shown] == ["stays"]


def test_the_queued_record_carries_no_pattern_list(tmp_path: Path):
    from brain.chat import pass2_queue

    _append_pattern(tmp_path, _pattern("p1"))
    _spawn_pass2_attunement(tmp_path, turn_id="t1", user_message=_MSG, reply_text="ok", buffer_slice=_buffer())
    (record,) = pass2_queue._load_queue_unlocked(tmp_path)
    assert set(record) == {"id", "kind", "turn_id", "user_message", "reply_text", "buffer_slice"}


def test_an_unreadable_pattern_file_logs_an_error_and_skips_the_detector(tmp_path: Path):
    with patch("brain.chat.tool_loop.read_learned_patterns", side_effect=OSError("disk")), \
         patch("brain.chat.tool_loop.run_detector") as mock_detector, \
         patch("brain.chat.tool_loop._attunement_consume_call") as mock_consume:
        _run_attunement_pass2(tmp_path, "t1", _MSG, "ok", _buffer())
    mock_detector.assert_not_called()
    mock_consume.assert_not_called()
    assert "disk" in (tmp_path / "attunement_errors.jsonl").read_text()


def test_an_exhausted_budget_returns_silently_before_the_pattern_read(tmp_path: Path):
    with patch("brain.chat.tool_loop._attunement_budget_remaining", return_value=0), \
         patch("brain.chat.tool_loop.read_learned_patterns", side_effect=OSError("disk")), \
         patch("brain.chat.tool_loop.run_detector") as mock_detector:
        _run_attunement_pass2(tmp_path, "t1", _MSG, "ok", _buffer())
    mock_detector.assert_not_called()
    assert not (tmp_path / "attunement_errors.jsonl").exists()
