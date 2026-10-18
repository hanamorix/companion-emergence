"""#440 - recognising a Claude CLI model rejection, and remembering which models were rejected.

The strings come from the committed fixture ``cli_2_1_219_model_not_found.ndjson``: the scrubbed
``assistant`` and ``result`` frames the real CLI 2.1.219 emitted for an unknown id (claude-sonet-5)
and a retired id (claude-3-opus-20240229). A replay proves the parser and the wire; whether the
CURRENT CLI still words it this way is checked by tests/bridge/test_model_rejection_live.py.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest

from brain.bridge import model_error, provider_auth
from brain.bridge import provider as provider_mod

FIXTURE = Path(__file__).resolve().parents[3] / "bridge" / "fixtures" / "cli_2_1_219_model_not_found.ndjson"
SENTENCE = "There's an issue with the selected model ({model}). It may not exist or you may not have access to it."
AUTH = "Failed to authenticate: OAuth session expired and could not be refreshed"
QUOTA = "api_error_status=429; is_error=True; You've hit your session limit"


def _frames() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result_frames() -> list[dict]:
    return [f for f in _frames() if f.get("type") == "result"]


@pytest.fixture(autouse=True)
def _reset():
    model_error.reset()
    yield
    model_error.reset()


def _clock(monkeypatch, start: float = 1000.0):
    t = {"now": start}
    monkeypatch.setattr(model_error, "_monotonic", lambda: t["now"])
    return t


# --- C5: recognising the rejection -----------------------------------------------------------


def test_fixture_has_the_real_rejection_for_both_probes():
    results = _result_frames()
    assert [r["api_error_status"] for r in results] == [404, 404]
    assert all(r["is_error"] is True for r in results)
    assert SENTENCE.format(model="claude-sonet-5") == results[0]["result"]
    assert SENTENCE.format(model="claude-3-opus-20240229") == results[1]["result"]


@pytest.mark.parametrize("idx", [0, 1])
def test_streaming_path_text_is_a_rejection(idx):
    # The real _cli_error_detail turns the result frame into the detail the stream path raises with.
    detail = provider_mod._cli_error_detail(_result_frames()[idx])
    assert detail is not None
    assert model_error.is_model_rejection(f"[claude_cli_error] {detail}") is True


@pytest.mark.parametrize("idx", [0, 1])
def test_blocking_path_text_is_a_rejection(idx):
    proc = subprocess.CompletedProcess(args=[], returncode=1, stdout=json.dumps(_result_frames()[idx]), stderr="")
    detail = provider_mod._claude_failure_detail(proc)
    assert "api_error_status=404" in detail
    assert model_error.is_model_rejection(f"[claude_cli_exit] exit 1: {detail}") is True


@pytest.mark.parametrize("text", [AUTH, QUOTA, "", "[claude_cli_exit] exit 1: ", "ClaudeCliProvider failed (exit 1): boom"])
def test_other_failures_are_not_a_rejection(text):
    assert model_error.is_model_rejection(text) is False


# --- C6: the model name ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "configured", "expected"),
    [
        (SENTENCE.format(model="claude-sonet-5"), "sonnet", "claude-sonet-5"),
        (SENTENCE.format(model="opus"), "sonnet", "opus"),
        (SENTENCE.format(model="claude-3-opus-20240229"), None, "claude-3-opus-20240229"),
        # not valid model grammar: fall back to the configured model
        (SENTENCE.format(model="<script>"), "claude-sonnet-5", "claude-sonnet-5"),
        (SENTENCE.format(model="Claude-X"), "sonnet", "sonnet"),
        (SENTENCE.format(model="has space"), "sonnet", "sonnet"),
        # truncated by the provider's 200-character cap: no closing parenthesis
        ("There's an issue with the selected model (claude-sonet-", "claude-sonnet-5", "claude-sonnet-5"),
        # no parenthesised name at all
        ("issue with the selected model", "opus", "opus"),
        # nothing usable
        ("issue with the selected model", None, None),
        ("issue with the selected model", "gpt4", None),
        (SENTENCE.format(model="<script>"), None, None),
    ],
)
def test_rejected_model_name(text, configured, expected):
    assert model_error.rejected_model_name(text, configured) == expected


def test_an_overlong_named_model_is_not_returned():
    # more than 64 characters is not valid model grammar, so the configured model is used instead
    long_id = "claude-" + "a" * 58
    assert model_error.rejected_model_name(SENTENCE.format(model=long_id), "sonnet") == "sonnet"
    assert model_error.rejected_model_name(SENTENCE.format(model=long_id), None) is None


def test_rejected_model_name_only_ever_returns_valid_grammar():
    from brain.persona_config import is_valid_model

    for text in (SENTENCE.format(model=m) for m in ["a" * 500, "claude-" + "x" * 300, "‮", "../etc"]):
        out = model_error.rejected_model_name(text, None)
        assert out is None or is_valid_model(out)


# --- C16: visibility state ------------------------------------------------------------------------

REJ = SENTENCE.format(model="claude-sonet-5")


def test_failure_is_recorded_with_count_and_times():
    model_error.note_failure(REJ)
    model_error.note_failure(REJ)
    (rec,) = model_error.state()
    assert rec["model"] == "claude-sonet-5"
    assert rec["count"] == 2
    assert rec["since"] and rec["last_at"]
    assert set(rec) == {"model", "since", "last_at", "count"}


def test_non_rejection_text_records_nothing():
    model_error.note_failure(AUTH)
    model_error.note_failure(QUOTA)
    model_error.note_failure("")
    assert model_error.state() == []


def test_warning_names_the_model_and_is_throttled(monkeypatch, caplog):
    t = _clock(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=model_error.log.name):
        model_error.note_failure(REJ)
        first = [r for r in caplog.records if "claude-sonet-5" in r.getMessage()]
        assert len(first) == 1
        model_error.note_failure(REJ)
        t["now"] += model_error.WARN_INTERVAL_S - 1
        model_error.note_failure(REJ)
        assert len([r for r in caplog.records if "claude-sonet-5" in r.getMessage()]) == 1
        t["now"] += 2
        model_error.note_failure(REJ)
        assert len([r for r in caplog.records if "claude-sonet-5" in r.getMessage()]) == 2


def test_warn_interval_default_matches_provider_auth_reprobe():
    assert model_error.WARN_INTERVAL_S == 900.0


def test_clean_frame_for_the_same_model_clears_a_different_model_does_not(tmp_path):
    from brain.bridge.usage_log import log_usage

    model_error.note_failure(REJ)
    log_usage(tmp_path, call_type="chat", model="haiku", frame={"is_error": False, "usage": {}})
    assert [r["model"] for r in model_error.state()] == ["claude-sonet-5"]
    # an error frame for the same model must NOT clear it
    log_usage(tmp_path, call_type="chat", model="claude-sonet-5", frame={"is_error": True, "usage": {}})
    assert [r["model"] for r in model_error.state()] == ["claude-sonet-5"]
    log_usage(tmp_path, call_type="chat", model="claude-sonet-5", frame={"is_error": False, "usage": {}})
    assert model_error.state() == []


def test_funnel_feeds_the_state():
    # provider_auth.note_cli_failure is the one funnel every CLI failure text passes through.
    provider_auth.note_cli_failure(REJ)
    assert [r["model"] for r in model_error.state()] == ["claude-sonet-5"]


def test_read_time_filter_hides_unrelated_models():
    model_error.note_failure(REJ)
    model_error.note_failure(SENTENCE.format(model="claude-other-1"))
    assert {r["model"] for r in model_error.state()} == {"claude-sonet-5", "claude-other-1"}
    assert [r["model"] for r in model_error.state(only={"claude-sonet-5", "haiku"})] == ["claude-sonet-5"]
    assert model_error.state(only={"opus"}) == []


def test_switch_drops_old_record_and_ignores_a_late_failure(caplog):
    model_error.note_failure(REJ)
    model_error.switch("claude-sonet-5", "sonnet", keep=frozenset())
    assert model_error.state() == []
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=model_error.log.name):
        model_error.note_failure(REJ)  # an in-flight ambient call that read the old id
    assert model_error.state() == []
    assert not [r for r in caplog.records if "claude-sonet-5" in r.getMessage()]
    # switching back un-ignores it
    model_error.switch("sonnet", "claude-sonet-5", keep=frozenset())
    model_error.note_failure(REJ)
    assert [r["model"] for r in model_error.state()] == ["claude-sonet-5"]


def test_switch_never_ignores_a_tier_model():
    tier_model = "claude-tier-9"
    model_error.switch(tier_model, "sonnet", keep=frozenset({tier_model}))
    model_error.note_failure(SENTENCE.format(model=tier_model))
    assert [r["model"] for r in model_error.state()] == [tier_model]


def test_suppressed_check_records_nothing_and_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=model_error.log.name):
        with model_error.suppressed():
            model_error.note_failure(REJ)
            provider_auth.note_cli_failure(REJ)
    assert model_error.state() == []
    assert not [r for r in caplog.records if "claude-sonet-5" in r.getMessage()]
    model_error.note_failure(REJ)  # suppression ended with the context
    assert len(model_error.state()) == 1


def test_suppression_is_per_thread():
    import threading

    seen = []

    def other():
        model_error.note_failure(REJ)
        seen.append(len(model_error.state()))

    with model_error.suppressed():
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert seen == [1]


def test_public_functions_fail_open(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(model_error, "_lock", None)  # any use raises
        model_error.note_failure(REJ)
        model_error.note_success("claude-sonet-5")
        model_error.switch("a", "b", keep=frozenset())
        assert model_error.state() == []


def test_log_usage_never_raises_on_a_non_dict_frame(tmp_path):
    # the clear-on-success hook sits inside log_usage's fail-safe ("instrumentation must never break a turn")
    from brain.bridge.usage_log import log_usage

    for bad in (None, "x", 5, [1]):
        log_usage(tmp_path, call_type="chat", model="sonnet", frame=bad)  # type: ignore[arg-type]
