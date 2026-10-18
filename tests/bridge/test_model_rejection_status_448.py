"""#448 - a rejected model is recognised by the CLI's own error signals, not only by its sentence.

Signals (real frames from claude 2.1.219 and 2.1.302, scrubbed, in tests/bridge/fixtures): the result frame's
``api_error_status: 404``, the assistant frame's ``error: "model_not_found"`` and, as the backup, the sentence.
Rule (main's ruling M3): a rejection is (404 AND model_not_found) OR (404 AND the sentence) OR model_not_found
alone OR the sentence alone; a 404 with none of those is the generic error, logged with its status (D8a in
decisions.md: an accepted trade-off, pinned by test_pure_404_is_generic_and_nothing_is_recorded).

Real ClaudeCliProvider, real builders, real routes; only the subprocess is faked.
"""

from __future__ import annotations

import getpass
import io
import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain.bridge import model_error, provider_auth
from brain.bridge import provider as provider_mod
from brain.bridge import server as server_mod
from brain.bridge.chat import ChatMessage, StreamError
from brain.bridge.provider import ClaudeCliProvider, ProviderError
from brain.bridge.server import build_app
from brain.utils import log_throttle

FIXDIR = Path(__file__).parent / "fixtures"
F219 = [json.loads(x) for x in (FIXDIR / "cli_2_1_219_model_not_found.ndjson").read_text(encoding="utf-8").splitlines() if x.strip()]
F302_TEXT = (FIXDIR / "cli_2_1_302_model_not_found.ndjson").read_text(encoding="utf-8")
F302 = [json.loads(x) for x in F302_TEXT.splitlines() if x.strip()]
F302_JSON = json.loads((FIXDIR / "cli_2_1_302_model_not_found_json.json").read_text(encoding="utf-8"))
BAD = "claude-sonet-5"
SENT_219 = F219[1]["result"]
SENT_302 = F302[1]["result"]
REWORDED = "Model unknown."
AUTH = "Failed to authenticate: OAuth session expired and could not be refreshed"


@pytest.fixture(autouse=True)
def _reset():
    model_error.reset()
    log_throttle._last.clear()
    yield
    model_error.reset()


def _reword(frames: list[dict]) -> list[dict]:
    out = json.loads(json.dumps(frames))
    for f in out:
        if f.get("type") == "result":
            f["result"] = REWORDED
        if f.get("type") == "assistant":
            f["message"]["content"] = [{"type": "text", "text": REWORDED}]
    return out


def _ndjson(frames: list[dict]) -> str:
    return "".join(json.dumps(f) + "\n" for f in frames)


def _proc(rc: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


class _FakePopen:
    def __init__(self, lines, rc: int = 1):
        self.stdin = io.StringIO()
        self.stdout = iter(lines)
        self.stderr = io.StringIO("")
        self.pid = 4242
        self._rc = rc

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass

    def kill(self):
        pass


def _lines(frames: list[dict]) -> list[str]:
    return [json.dumps(f) + "\n" for f in frames]


# --- MB1: the detector rule -------------------------------------------------------------------------

TRUE_CASES = [
    "api_error_status=404; error=model_not_found; " + REWORDED,
    "[claude_cli_error] api_error_status=404; error=model_not_found; " + REWORDED,
    "api_error_status=404; is_error=True; " + SENT_219,  # 404 AND the sentence (blocking json argv)
    "api_error_status=404; is_error=True; " + SENT_302,  # ... with the 2.1.302 suffix
    "error=model_not_found; " + REWORDED,  # the code alone
    SENT_219,  # the sentence alone (backup)
    SENT_302,
    "[claude_cli_error] " + SENT_302,
]
FALSE_CASES = [
    "api_error_status=404; is_error=True; Not Found",  # 404 with neither signal (D8a)
    "[claude_cli_error] api_error_status=404; " + REWORDED,
    "api_error_status=404; error=authentication_failed; " + REWORDED,  # conflicting code
    "api_error_status=429; is_error=True; You've hit your session limit",
    AUTH,
    "api_error_status=4041; error=model_not_found_extra; " + REWORDED,
    "api_error_status=40; " + REWORDED,
    "x_api_error_status=404; " + REWORDED,
    "api_error_status=500; " + REWORDED,
    "error=model_not_found_extra; " + REWORDED,
    "",
    "[claude_cli_exit] exit 1: ",
]


@pytest.mark.parametrize("text", TRUE_CASES)
def test_rejection_shapes_are_recognised(text):
    assert model_error.is_model_rejection(text) is True


@pytest.mark.parametrize("text", FALSE_CASES)
def test_everything_else_is_not_a_rejection(text):
    assert model_error.is_model_rejection(text) is False


def test_auth_wins_over_a_rejection_text():
    text = f"{AUTH}; api_error_status=404; error=model_not_found; {SENT_302}"
    assert server_mod._chat_error_code(RuntimeError(text)) == "auth_expired"


def _results(frames):
    return [f for f in frames if f.get("type") == "result"]


@pytest.mark.parametrize("frames,code", [(F219, "model_not_found"), (F302, "model_not_found")])
def test_real_builders_on_real_frames_give_a_rejection(frames, code):
    for res in _results(frames):
        detail = provider_mod._cli_error_detail(res, error_code=code, model=BAD)
        assert detail is not None and model_error.is_model_rejection(f"[claude_cli_error] {detail}")
        assert "api_error_status=404" in detail
        blocking = provider_mod._claude_failure_detail(_proc(1, json.dumps(res)), model=BAD)
        assert model_error.is_model_rejection(f"[claude_cli_exit] exit 1: {blocking}")
        ndjson = provider_mod._claude_failure_detail(_proc(1, _ndjson(frames)), model=BAD)
        assert "error=model_not_found" in ndjson and model_error.is_model_rejection(ndjson)


def test_json_argv_object_from_302_is_a_rejection_through_the_blocking_builder():
    assert "error" not in F302_JSON  # the json argv carries the status and the sentence, no error code (probe 2)
    detail = provider_mod._claude_failure_detail(_proc(1, json.dumps(F302_JSON)), model=BAD)
    assert "api_error_status=404" in detail and model_error.is_model_rejection(detail)


def test_pure_404_is_generic_and_nothing_is_recorded(caplog):
    """D8a (accepted trade-off): a 404 with neither the code nor the sentence is NOT a model rejection."""
    frame = {**F302_JSON, "result": "Not Found"}
    detail = provider_mod._claude_failure_detail(_proc(1, json.dumps(frame)), model=BAD)
    assert "api_error_status=404" in detail
    assert server_mod._chat_error_code(RuntimeError(f"[claude_cli_exit] exit 1: {detail}")) == "provider_failed"
    with caplog.at_level(logging.WARNING):
        provider_auth.note_cli_failure(detail)
    assert model_error.state() == []
    assert any("404" in r.getMessage() for r in caplog.records)  # logged with its status
    with patch("subprocess.run", return_value=_proc(1, json.dumps(frame))):  # Apply: inconclusive, saved unchecked
        assert server_mod._check_model_blocking(BAD, 5) == ("inconclusive", "failed")


# --- MB2: the streaming path carries the status ---------------------------------------------------------


def _stream_error(frames: list[dict], model: str = BAD) -> StreamError:
    provider = ClaudeCliProvider(model=model)
    with patch("subprocess.Popen", return_value=_FakePopen(_lines(frames))):
        events = list(provider.chat_stream([ChatMessage(role="user", content="hi")]))
    errors = [e for e in events if isinstance(e, StreamError)]
    assert errors, events
    return errors[0]


def test_stream_result_frame_detail_carries_status_and_code():
    err = _stream_error(F302)
    assert err.stage == "claude_cli_error"
    assert "api_error_status=404" in err.detail and "error=model_not_found" in err.detail
    assert model_error.is_model_rejection(f"[{err.stage}] {err.detail}")


def test_stream_with_the_sentence_reworded_away_is_still_a_rejection():
    err = _stream_error(_reword(F302))
    assert REWORDED in err.detail and SENT_302 not in err.detail
    assert model_error.is_model_rejection(f"[{err.stage}] {err.detail}")


def test_stream_last_string_error_code_wins_and_later_frames_do_not_erase_it():
    frames = _reword(F302)
    extra = {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}  # no error key
    err = _stream_error([frames[0], extra, frames[1]])
    assert "error=model_not_found" in err.detail


def _client(persona_dir: Path, monkeypatch, model: str = BAD) -> TestClient:
    monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: ClaudeCliProvider(model=model))
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


@pytest.mark.parametrize("frames", [F302, _reword(F302)], ids=["sentence", "reworded"])
def test_routes_answer_model_unavailable_with_the_model(persona_dir: Path, monkeypatch, frames):
    with _client(persona_dir, monkeypatch) as c:
        sid = c.post("/session/new", json={}).json()["session_id"]
        with patch("subprocess.Popen", return_value=_FakePopen(_lines(frames))), patch(
            "subprocess.run", return_value=_proc(1, _ndjson(frames))
        ):
            r = c.post("/chat", json={"session_id": sid, "message": "hello"})
            assert r.status_code == 502 and r.json()["detail"] == f"model_unavailable:{BAD}"
        with patch("subprocess.Popen", return_value=_FakePopen(_lines(frames))), patch(
            "subprocess.run", return_value=_proc(1, _ndjson(frames))
        ):
            with c.websocket_connect(f"/stream/{sid}") as ws:
                ws.send_json({"message": "hello"})
                while True:
                    f = ws.receive_json()
                    if f.get("type") in ("error", "done"):
                        break
    assert f == {"type": "error", "code": "model_unavailable", "model": BAD, "done": True}


# --- MB3: blocking paths -----------------------------------------------------------------------------------


def _persona(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "scratch"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps({"mcp_audit_log_level": "redacted"}), encoding="utf-8")
    return d


def test_generate_json_argv_failure_is_a_rejection():
    provider = ClaudeCliProvider(model=BAD)
    with patch("brain.bridge.provider.subprocess.run", return_value=_proc(1, json.dumps(F302_JSON))):
        with pytest.raises(RuntimeError) as err:
            provider.generate("hi")
    assert model_error.is_model_rejection(str(err.value)) and f"model={BAD}" in str(err.value)


def test_generate_exit0_in_frame_failure_is_a_rejection():
    provider = ClaudeCliProvider(model=BAD)
    with patch("brain.bridge.provider.subprocess.run", return_value=_proc(0, json.dumps(F302_JSON))):
        with pytest.raises(RuntimeError) as err:
            provider.generate("hi")
    assert model_error.is_model_rejection(str(err.value))


@pytest.mark.parametrize("frames", [F302, _reword(F302)], ids=["sentence", "reworded"])
def test_blocking_text_chat_exit1_is_a_rejection(tmp_path: Path, frames):
    provider = ClaudeCliProvider(model=BAD)
    with patch("brain.bridge.provider.subprocess.run", return_value=_proc(1, _ndjson(frames))):
        with pytest.raises(ProviderError) as err:
            provider.chat([ChatMessage(role="user", content="hi")], tools=None, options={"persona_dir": str(_persona(tmp_path))})
    assert model_error.is_model_rejection(str(err.value))


def test_blocking_text_chat_exit0_in_frame_is_a_rejection(tmp_path: Path):
    provider = ClaudeCliProvider(model=BAD)
    with patch("brain.bridge.provider.subprocess.run", return_value=_proc(0, _ndjson(_reword(F302)))):
        with pytest.raises(ProviderError) as err:
            provider.chat([ChatMessage(role="user", content="hi")], tools=None, options={"persona_dir": str(_persona(tmp_path))})
    assert model_error.is_model_rejection(str(err.value))


def test_blocking_tools_chat_exit1_is_a_rejection(tmp_path: Path):
    provider = ClaudeCliProvider(model=BAD)
    with patch("brain.bridge.provider.subprocess.run", return_value=_proc(1, _ndjson(_reword(F302)))):
        with pytest.raises(ProviderError) as err:
            provider.chat([ChatMessage(role="user", content="hi")], tools=[{"name": "search_memories"}], options={"persona_dir": str(_persona(tmp_path))})
    assert model_error.is_model_rejection(str(err.value)) and f"model={BAD}" in str(err.value)


def test_blocking_builder_output_is_unchanged_without_a_model_not_found_code():
    frame = {"type": "result", "is_error": True, "api_error_status": 401, "result": AUTH}
    assert provider_mod._claude_failure_detail(_proc(1, json.dumps(frame)), model="sonnet") == f"api_error_status=401; is_error=True; {AUTH}"
    quota = {"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"}
    assert provider_mod._claude_failure_detail(_proc(1, json.dumps(quota))) == "api_error_status=429; is_error=True; You've hit your session limit"


# --- M4: hostile stdout never breaks the failure path ---------------------------------------------------------


def _hostile_lines() -> list[str]:
    # Every line here makes json.loads raise ValueError / RecursionError or is not an assistant dict.
    return [
        "9" * 5000,
        "[" * 100000,
        json.dumps([1, 2, 3]),
        json.dumps("a string"),
        '{"type": "assistant", "error": 5}',
        "x" * 200000,
        '{"type": "assistant", "error": "' + "A" * 80 + '"}',
        "\ud800 not json",
    ]


def _hostile_stdout() -> str:
    hostile = "\n".join(_hostile_lines())
    return hostile + "\n" + _ndjson(_reword(F302)) + hostile


def test_assistant_error_scan_survives_hostile_lines():
    assert provider_mod._assistant_error_code(_hostile_stdout()) == "model_not_found"
    assert provider_mod._assistant_error_code("") is None
    assert provider_mod._assistant_error_code("9" * 5000) is None
    assert provider_mod._assistant_error_code("[" * 100000) is None


def test_failure_detail_with_benign_hostile_lines_is_still_a_detected_rejection():
    """The builder around the scan: lines the existing result-line parser already tolerates (non-dict JSON,
    huge or invalid text) placed before and after the real frames. (ValueError / RecursionError lines in the
    EXISTING result-line parser are PR #406's fix, not this change's.)"""
    benign = ["x" * 200000, json.dumps([1, 2, 3]), json.dumps("a string"), '{"type": "assistant", "error": 5}', "\ud800 not json"]
    stdout = "\n".join(benign) + "\n" + _ndjson(_reword(F302)) + "\n".join(benign)
    detail = provider_mod._claude_failure_detail(_proc(1, stdout), model=BAD)
    assert model_error.is_model_rejection(detail) and "error=model_not_found" in detail, detail[:300]


# --- MB4: naming and validation (the executed argv, never provider._model) ------------------------------------------


def test_name_from_sentence_when_valid_else_argv_token_else_none():
    name = model_error.rejected_model_name
    assert name(f"model={BAD}; error=model_not_found; {REWORDED}", None) == BAD
    assert name("There's an issue with the selected model (claude-other-1). It may not exist.", None) == "claude-other-1"
    # a sentence that names text outside the model grammar falls back to the argv token
    assert name(f"model={BAD}; There's an issue with the selected model (a<b). x", None) == BAD
    assert name("There's an issue with the selected model (a b). x", None) is None
    assert name(f"model=not valid!; {REWORDED}", None) is None
    assert name("", None) is None


def test_chat_error_model_never_uses_the_provider_model():
    text = f"model={BAD}; error=model_not_found; {REWORDED}"
    assert server_mod._chat_error_model(RuntimeError(text), ClaudeCliProvider(model="opus")) == BAD
    # nothing usable in the text: no name, even though the provider has a model
    assert server_mod._chat_error_model(RuntimeError("error=model_not_found; Model unknown."), ClaudeCliProvider(model="opus")) is None
    assert server_mod._chat_error_detail(RuntimeError("error=model_not_found; Model unknown."), ClaudeCliProvider(model="opus")) == "model_unavailable"


def test_hot_swap_during_a_streaming_turn_still_names_the_executed_model(persona_dir: Path, monkeypatch):
    with _client(persona_dir, monkeypatch) as c:
        provider = c.app.state.bridge.provider
        sid = c.post("/session/new", json={}).json()["session_id"]

        def _popen(cmd, *a, **k):
            assert cmd[cmd.index("--model") + 1] == BAD
            provider._model = "opus"  # the picker switched while the call was running
            return _FakePopen(_lines(_reword(F302)))

        with patch("subprocess.Popen", side_effect=_popen), patch("subprocess.run", return_value=_proc(1, _ndjson(_reword(F302)))):
            r = c.post("/chat", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 502 and r.json()["detail"] == f"model_unavailable:{BAD}"


def test_hot_swap_on_the_blocking_path_names_and_records_the_executed_model(tmp_path: Path):
    provider = ClaudeCliProvider(model=BAD)

    def _run(cmd, **kw):
        provider._model = "opus"
        return _proc(1, _ndjson(_reword(F302)))

    with patch("brain.bridge.provider.subprocess.run", side_effect=_run):
        with pytest.raises(ProviderError) as err:
            provider.chat([ChatMessage(role="user", content="hi")], tools=None, options={"persona_dir": str(_persona(tmp_path))})
    assert server_mod._chat_error_model(err.value, provider) == BAD
    assert [r["model"] for r in model_error.state()] == [BAD]


# --- MB5: background visibility without the sentence ----------------------------------------------------------


def test_reworded_rejection_on_the_ndjson_path_is_recorded_warned_and_reported(tmp_path: Path, caplog):
    provider = ClaudeCliProvider(model=BAD)
    with caplog.at_level(logging.WARNING), patch("brain.bridge.provider.subprocess.run", return_value=_proc(1, _ndjson(_reword(F302)))):
        with pytest.raises(ProviderError):
            provider.chat([ChatMessage(role="user", content="hi")], tools=None, options={"persona_dir": str(_persona(tmp_path))})
    state = model_error.state()
    assert [r["model"] for r in state] == [BAD] and state[0]["count"] == 1
    assert len([r for r in caplog.records if BAD in r.getMessage() and r.name.endswith("model_error")]) == 1
    from brain.bridge.persona_state import build_persona_state

    d = tmp_path / "p2"
    d.mkdir()
    (d / "persona_config.json").write_text(json.dumps({"provider": "claude-cli", "model": BAD}))
    assert [r["model"] for r in build_persona_state(d)["model_rejections"]] == [BAD]


# --- MB7: non-rejection frames unchanged; downstream classifiers unchanged ---------------------------------------


@pytest.mark.parametrize(
    "frame,expected",
    [
        ({"is_error": True, "result": AUTH}, AUTH),
        ({"is_error": True, "api_error_status": 401, "result": AUTH}, AUTH),
        ({"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"}, "You've hit your session limit"),
        ({"is_error": True, "api_error_status": 500, "result": "boom"}, "boom"),
        ({"is_error": True, "result": ""}, None),  # replaced below: the "no result text" form
    ],
)
def test_detail_is_todays_value_for_every_frame_that_is_not_a_404(frame, expected):
    detail = provider_mod._cli_error_detail(frame, error_code="rate_limit", model="sonnet")
    if expected is None:
        assert detail.startswith("is_error with no result text: ")
    else:
        assert detail == expected


def test_over_budget_and_clean_frames_are_unchanged():
    assert provider_mod._cli_error_detail({"is_error": True, "subtype": "error_max_budget_usd", "result": "x"}) is None
    provider_auth.note_cli_failure(AUTH + " oauth session expired")
    assert provider_mod._cli_error_detail({"is_error": False, "result": "ok"}) is None


def test_status_404_prefix_is_outside_the_200_char_cut():
    long = "There's an issue with the selected model (" + BAD + "). " + "x" * 400
    detail = provider_mod._cli_error_detail({"is_error": True, "api_error_status": 404, "result": long}, model=BAD)
    assert detail.startswith(f"model={BAD}; api_error_status=404; ")
    body = detail.split("; ", 2)[2]
    assert len(body) == 200


def test_review_string_classifier_reads_the_same_text_as_before():
    """brain/initiate/review.py maps an exception to a rate limit when its lowercased text contains "429" or
    "rate". The new tokens must not change that answer for any detail. (The wrapper itself raises TypeError for
    every non-timeout exception today, because of `except (TimeoutError, TimeoutError.__class__)`: an existing
    bug outside this change, reported to main.)"""

    def hit(detail: str) -> bool:
        low = detail.lower()
        return "429" in low or "rate" in low

    quota = "You've hit your session limit"
    in_frame_429 = provider_mod._cli_error_detail({"is_error": True, "api_error_status": 429, "result": quota})
    assert in_frame_429 == quota and hit(in_frame_429) is hit(quota)
    rej = provider_mod._cli_error_detail(F302[1], error_code="model_not_found", model=BAD)
    assert hit(rej) is hit(SENT_302)


# --- MB8: fixture hygiene -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cli_2_1_302_model_not_found.ndjson", "cli_2_1_302_model_not_found_json.json"])
def test_new_fixtures_are_scrubbed(name):
    text = (FIXDIR / name).read_text(encoding="utf-8")
    for needle in ("/home/", getpass.getuser(), '"cwd"', "memory"):
        assert needle not in text, needle
    for line in [x for x in text.splitlines() if x.strip()]:
        obj = json.loads(line)
        for key in ("session_id", "uuid", "request_id"):
            assert obj.get(key) in (None, "REDACTED"), (key, obj.get(key))


def test_ndjson_fixture_has_only_assistant_and_result_frames():
    assert [f["type"] for f in F302] == ["assistant", "result"]
    assert F302[0]["error"] == "model_not_found" and F302[0]["api_error_status"] == 404
    assert F302[1]["api_error_status"] == 404 and F302[1]["is_error"] is True


def test_apply_check_marks_a_status_and_code_rejection_as_rejected(monkeypatch):
    with patch("subprocess.run", return_value=_proc(1, json.dumps(F302_JSON))):
        assert server_mod._check_model_blocking(BAD, 5) == ("rejected", "")
