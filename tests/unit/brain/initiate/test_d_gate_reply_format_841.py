"""#841: the outreach review's D gate tells the model its reply format, and reads the reply it gets.

Spec: `docs/source-spec/2026-11-28-841-d-gate-reply-format-spec.md`. No prompt described the JSON object the parser reads,
so no reply ever parsed and every candidate was promoted.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

REPLY_FORMAT = (
    "Reply with only this JSON object and nothing else:\n"
    '{"decisions": [{"candidate_index": 1, "decision": "filter", "reason": "one short sentence", "confidence": "high"}], '
    '"tick_note": null}\n'
    "Give one decision for each candidate. candidate_index is the number in brackets before the candidate. "
    'decision is "promote" or "filter". confidence is "high", "medium" or "low". '
    "tick_note is one short sentence about this tick, or null.\n"
)


def _user_message() -> str:
    from brain.initiate.reflection import build_user_message

    return build_user_message(
        user_name="the user",
        now=datetime(2026, 11, 28, 9, 30, tzinfo=UTC),
        outbound_recall_block="(no recent outbound)",
        candidate_summaries=["source: dream", "source: research"],
    )


# --- the reply-format paragraph (S7, S13) ---------------------------------------------------------------------------


def test_the_user_message_ends_with_the_reply_format_after_the_task() -> None:
    rendered = _user_message()
    assert rendered.endswith("Promote at most 2. The default is filter.\n" + REPLY_FORMAT)


def test_the_reply_format_names_every_key_the_parser_reads_and_its_example_parses() -> None:
    """A canary: the prompt and the parser cannot drift apart again (I16)."""
    from brain import prompt_strings
    from brain.initiate.reflection import parse_structured_response

    segments = prompt_strings.get_segments("initiate.reflection.user_message_segments")
    assert len(segments) == 7  # build_user_message joins 6 values between these 7 segments (#129)
    task = segments[-1]
    for key in ("decisions", "candidate_index", "decision", "reason", "confidence", "tick_note"):
        assert f'"{key}"' in task, key
    example = next(line for line in task.splitlines() if line.startswith('{"decisions"'))
    result = parse_structured_response(example)
    assert [(d.candidate_index, d.decision, d.confidence) for d in result.decisions] == [(1, "filter", "high")]


# --- the parser (S22, S23) ------------------------------------------------------------------------------------------

_GOOD = {"candidate_index": 1, "decision": "filter", "reason": "quiet weather", "confidence": "high"}


def _reply(**item: object) -> str:
    import json

    return json.dumps({"decisions": [{**_GOOD, **item}], "tick_note": None})


@pytest.mark.parametrize(
    "item",
    [
        {"reason": None},  # null
        {"candidate_index": None},
        {"candidate_index": "2"},
        {"candidate_index": True},
        {"candidate_index": 2.0},
        {"decision": "skip"},
        {"decision": None},
        {"confidence": "unsure"},
    ],
    ids=["reason-null", "index-null", "index-string", "index-bool", "index-float", "decision-skip", "decision-null",
         "confidence-unsure"],
)
def test_a_badly_shaped_or_valued_item_makes_the_reply_malformed(item: dict) -> None:
    from brain.initiate.reflection import parse_structured_response

    with pytest.raises(ValueError):
        parse_structured_response(_reply(**item))


def test_an_item_missing_a_field_makes_the_reply_malformed_not_a_key_error() -> None:
    import json

    from brain.initiate.reflection import parse_structured_response

    item = {k: v for k, v in _GOOD.items() if k != "reason"}
    with pytest.raises(ValueError):
        parse_structured_response(json.dumps({"decisions": [item]}))


def test_values_are_read_without_regard_to_case_and_stored_in_lower_case() -> None:
    from brain.initiate.reflection import parse_structured_response

    (d,) = parse_structured_response(_reply(decision="Filter", confidence="High")).decisions
    assert (d.decision, d.confidence) == ("filter", "high")


# --- through the real review tick (S8, S10, S17, S22) ---------------------------------------------------------------

_NOW = datetime(2026, 11, 28, 10, 0, tzinfo=UTC)


class _Gate:
    """A stand-in for the Haiku gate provider: answers every D call with the next canned reply."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.systems: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.systems.append(system or "")
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _tick(tmp_path, monkeypatch, gate: _Gate, n: int = 3):
    """Queue n candidates, run one real review tick with `gate` as the D provider; return (persona, candidate ids, composed ids)."""
    from brain.initiate.emit import emit_initiate_candidate, read_candidates
    from brain.initiate.review import run_initiate_review_tick
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "persona"
    for i in range(n):
        emit_initiate_candidate(
            persona, kind="message", source="dream", source_id=f"d{i + 1}", semantic_context=SemanticContext(), now=_NOW
        )
    ids = [c.candidate_id for c in read_candidates(persona)]
    composed: list[str] = []
    monkeypatch.setattr("brain.bridge.model_tier.build_tier_provider", lambda *_a, **_k: gate)
    monkeypatch.setattr(
        "brain.initiate.review._process_one_candidate", lambda _pd, candidate, **_k: composed.append(candidate.candidate_id)
    )
    run_initiate_review_tick(persona, provider=object(), voice_template="", cap_per_tick=3, now=_NOW)
    return persona, ids, composed


def _decisions(*items: tuple[int, str]) -> str:
    import json

    return json.dumps(
        {"decisions": [{"candidate_index": i, "decision": d, "reason": f"r{i}", "confidence": "high"} for i, d in items]}
    )


def _d_rows(persona) -> list[dict]:
    import json

    return [json.loads(line) for line in (persona / "initiate_d_calls.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_parsed_reply_filters_and_promotes_the_candidates_its_bracket_numbers_name(tmp_path, monkeypatch) -> None:
    from brain.initiate.emit import read_candidates

    persona, ids, composed = _tick(tmp_path, monkeypatch, _Gate(_decisions((1, "filter"), (2, "promote"), (3, "filter"))))
    draft = (persona / "draft_space.md").read_text(encoding="utf-8")
    assert ids[0] in draft and ids[2] in draft and ids[1] not in draft
    assert composed == [ids[1]]
    assert [c.candidate_id for c in read_candidates(persona)] == [ids[1]]  # the two filtered ones left the queue
    (row,) = _d_rows(persona)
    assert (row["failure_type"], row["promoted_out"], row["filtered_out"]) == (None, 1, 2)


def test_an_index_past_the_last_candidate_is_skipped_and_the_uncovered_candidate_is_promoted(tmp_path, monkeypatch) -> None:
    persona, ids, composed = _tick(tmp_path, monkeypatch, _Gate(_decisions((1, "filter"), (2, "promote"), (4, "filter"))))
    draft = (persona / "draft_space.md").read_text(encoding="utf-8")
    assert ids[0] in draft and ids[1] not in draft and ids[2] not in draft
    assert sorted(composed) == sorted([ids[1], ids[2]])  # candidate 3 was covered by nothing


def test_an_index_of_0_never_selects_the_last_candidate(tmp_path, monkeypatch) -> None:
    persona, ids, composed = _tick(tmp_path, monkeypatch, _Gate(_decisions((0, "filter"), (1, "promote"), (2, "promote"))))
    assert not (persona / "draft_space.md").exists()  # nothing was demoted
    assert sorted(composed) == sorted(ids)


def test_a_second_decision_for_the_same_candidate_is_ignored(tmp_path, monkeypatch) -> None:
    persona, ids, composed = _tick(tmp_path, monkeypatch, _Gate(_decisions((1, "filter"), (1, "promote"))), n=1)
    assert ids[0] in (persona / "draft_space.md").read_text(encoding="utf-8")
    assert composed == []  # demoted once, never also composed


def test_a_badly_shaped_reply_takes_the_malformed_path_and_keeps_the_raw_reply(tmp_path, monkeypatch) -> None:
    import json

    bad = json.dumps({"decisions": [{"candidate_index": None, "decision": "filter", "reason": "x", "confidence": "high"}]})
    persona, ids, composed = _tick(tmp_path, monkeypatch, _Gate(bad), n=2)  # both attempts get the bad reply
    (row,) = _d_rows(persona)
    assert row["failure_type"] == "malformed_json"
    assert sorted(composed) == sorted(ids)  # promote all, as before
    sites = [json.loads(line)["site"] for line in (persona / "parse_failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sites == ["d_reflection_haiku", "d_reflection_sonnet"]  # #427: the raw reply is kept for each attempt


# --- the calibration block (S9) -------------------------------------------------------------------------------------


def _adaptive(persona) -> None:
    from brain.initiate.adaptive import CalibrationRow, append_calibration_row

    persona.mkdir(parents=True, exist_ok=True)
    append_calibration_row(
        persona,
        CalibrationRow(
            ts_decision="2026-11-20T10:00:00+00:00",
            ts_closed="2026-11-20T11:00:00+00:00",
            candidate_id="ic_x",
            source="dream",
            decision="promote",
            confidence="high",
            model_tier="haiku",
            promoted_to_state="replied_explicit",
            filtered_recurred=None,
            reason_short="x",
        ),
    )
    (persona / "d_mode.json").write_text('{"mode": "adaptive"}', encoding="utf-8")


def test_an_adaptive_persona_gets_its_calibration_block_in_the_d_call(tmp_path, monkeypatch) -> None:
    _adaptive(tmp_path / "persona")
    gate = _Gate(_decisions((1, "promote")))
    _tick(tmp_path, monkeypatch, gate, n=1)
    assert "=== Your recent editorial track record ===" in gate.systems[0]


@pytest.mark.no_base_failure(reason="#841 pins that a stateless persona still gets no calibration block; main omitted it for every persona")
def test_a_persona_without_d_mode_gets_no_calibration_block(tmp_path, monkeypatch) -> None:
    gate = _Gate(_decisions((1, "promote")))
    persona, _ids, _composed = _tick(tmp_path, monkeypatch, gate, n=1)
    assert not (persona / "d_mode.json").exists()
    assert gate.systems and "editorial track record" not in gate.systems[0]


# --- the drift history (S18) ----------------------------------------------------------------------------------------


def _log_d_calls(persona, *, failed: int, parsed: int, both_low_at: int | None = None) -> None:
    """`failed` malformed_json rows (no decision), then `parsed` rows whose per-row promote rates alternate 0.4 / 0.6
    (5 candidates, 2 or 3 promoted), all failed rows dated before every parsed row."""
    from datetime import timedelta

    from brain.initiate.audit import append_d_call_row
    from brain.initiate.d_call_schema import DCallRow

    persona.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 5, 15, tzinfo=UTC)
    for i in range(failed + parsed):
        ok = i >= failed
        promoted = (2 if (i - failed) % 2 == 0 else 3) if ok else 0
        append_d_call_row(
            persona,
            DCallRow(
                d_call_id=f"dc_{i}", ts=(start + timedelta(hours=i)).isoformat(), tick_id=f"t{i}",
                model_tier_used="haiku", candidates_in=5 if ok else 1, promoted_out=promoted,
                filtered_out=(5 - promoted) if ok else 0, latency_ms=0, tokens_input=0, tokens_output=0,
                failure_type=None if ok else "malformed_json",
            ),
        )
    if both_low_at is not None:  # a call that decided (forced filters) under a failure type is kept
        rows = (persona / "initiate_d_calls.jsonl").read_text(encoding="utf-8").splitlines()
        import json

        row = json.loads(rows[failed + both_low_at])
        row["failure_type"] = "both_low_confidence"
        rows[failed + both_low_at] = json.dumps(row)
        (persona / "initiate_d_calls.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_failed_calls_do_not_count_toward_the_drift_floor(tmp_path) -> None:
    from brain.initiate.adaptive import detect_drift

    _log_d_calls(tmp_path, failed=245, parsed=40)
    assert detect_drift(tmp_path) is None  # 40 kept rows: still below the 100-row floor


def test_drift_is_judged_on_the_calls_that_decided(tmp_path) -> None:
    from brain.initiate.adaptive import detect_drift

    _log_d_calls(tmp_path, failed=245, parsed=150, both_low_at=10)
    # Kept: 150 rows (the both_low_confidence one included). Historical = the 120 oldest, median 0.5; the newest 30 run
    # at 0.5, so nothing drifted. Counting the 245 failed rows as rate 0 would put the median at 0 and alert.
    assert detect_drift(tmp_path) is None


def test_a_both_low_confidence_row_counts_and_the_four_no_decision_types_do_not() -> None:
    from types import SimpleNamespace

    from brain.initiate.adaptive import _rows_that_decided

    types = [None, "both_low_confidence", "timeout", "rate_limit", "provider_error", "malformed_json"]
    kept = _rows_that_decided([SimpleNamespace(failure_type=t) for t in types])
    assert [r.failure_type for r in kept] == [None, "both_low_confidence"]


# --- the draft-space reason (S19, S24) ------------------------------------------------------------------------------


def test_a_filtered_reason_is_written_as_one_json_string_that_keeps_non_ascii(tmp_path) -> None:
    from brain.initiate.reflection import DDecision, demote_to_draft_space
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    reason = 'she said "not yet"\nand meant it — later'
    candidate = InitiateCandidate(
        candidate_id="ic_a", ts=_NOW.isoformat(), kind="message", source="dream", source_id="d1",
        semantic_context=SemanticContext(),
    )
    demote_to_draft_space(tmp_path, candidate=candidate, decision=DDecision(1, "filter", reason, "high"), now=_NOW)
    lines = (tmp_path / "draft_space.md").read_text(encoding="utf-8").splitlines()
    (line,) = [x for x in lines if x.startswith("d_reason: ")]
    assert line == 'd_reason: "she said \\"not yet\\"\\nand meant it — later"'
