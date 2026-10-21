"""POST /persona/config/model: set the install-wide chat model (#529: C11, C35, C42 e).

The chat model is the `model.interactive_chat` line of the tunables file; the endpoint applies the one write rule to
the raw body value, probes only a value it is about to write that is not an alias word, and answers with the id the
chat job runs after the write. Nothing is hot-swapped: the next chat call reads the file.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import dev_constants, tunables
from brain.bridge import model_error, server
from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT
from brain.bridge.server import build_app

_CHAT = "model.interactive_chat"
_OPUS = "claude-opus-5"


def _make_client(tmp_path: Path, *, provider: str = "claude-cli", auth_token: str | None = None):
    persona_dir = tmp_path / "personas" / "test"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(json.dumps({"provider": provider, "searcher": "noop"}))
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    return TestClient(app), persona_dir


def _overrides() -> dict:
    p = tunables._file_path()
    return json.loads(p.read_text(encoding="utf-8")).get("overrides", {}) if p.exists() else {}


def _post(client, model):
    return client.post("/persona/config/model", json={"model": model})


def test_sonnet_removes_an_existing_line_and_asks_no_claude(tmp_path, capture_claude_argv) -> None:
    tunables.set_overrides({_CHAT: _OPUS})
    client, _ = _make_client(tmp_path)
    with client:
        r = _post(client, "sonnet")
    assert r.status_code == 200 and r.json() == {"ok": True, "model": "claude-sonnet-5"}
    assert _CHAT not in _overrides()
    assert capture_claude_argv.argvs == []


def test_sonnet_with_no_line_writes_nothing(tmp_path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        r = _post(client, "sonnet")
    assert r.status_code == 200 and r.json()["model"] == "claude-sonnet-5"
    assert not tunables._file_path().exists() or _CHAT not in _overrides()


def test_opus_writes_the_opus_id_after_one_probe(tmp_path, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        r = _post(client, "opus")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "model": _OPUS, "checked": True}
    assert _overrides()[_CHAT] == _OPUS
    assert capture_claude_argv.models == [_OPUS]


def test_haiku_writes_the_alias_with_no_probe(tmp_path, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        r = _post(client, "haiku")
    assert r.json() == {"ok": True, "model": "haiku"}
    assert _overrides()[_CHAT] == "haiku" and capture_claude_argv.argvs == []


def test_a_full_id_is_written_verbatim_even_when_it_equals_the_default(tmp_path, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        r = _post(client, "claude-sonnet-5")
    assert r.status_code == 200 and r.json()["model"] == "claude-sonnet-5" and r.json()["checked"] is True
    assert _overrides()[_CHAT] == "claude-sonnet-5"
    assert capture_claude_argv.models == ["claude-sonnet-5"]


def test_the_body_never_carries_the_retired_fallback_fields(tmp_path, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        body = _post(client, "claude-opus-4-9").json()
    assert "fallbacks" not in body and "tiers_unchecked" not in body


def test_an_unknown_model_is_refused_with_the_three_words(tmp_path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        r = _post(client, "gpt-4")
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "unknown_model" and body["valid"] == ["haiku", "opus", "sonnet"]
    assert _CHAT not in _overrides()


def test_a_rejected_probe_writes_nothing(tmp_path, capture_claude_argv) -> None:
    capture_claude_argv.reject("claude-sonet-5")
    client, _ = _make_client(tmp_path)
    with client:
        before = tunables._file_path().read_bytes()  # after the start-up rewrite of the defaults section
        r = _post(client, "claude-sonet-5")
        assert tunables._file_path().read_bytes() == before
    assert r.status_code == 400
    assert r.json() == {"ok": False, "error": "model_unavailable", "model": "claude-sonet-5"}


def test_an_inconclusive_probe_saves_unchecked(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server, "_check_model_blocking", lambda model, timeout_s=None: ("inconclusive", "timeout"))
    client, persona_dir = _make_client(tmp_path)
    with client:
        r = _post(client, "claude-opus-4-9")
        state = client.get("/persona/state").json()
    assert r.status_code == 200 and r.json() == {"ok": True, "model": "claude-opus-4-9", "checked": False}
    assert state["model_unconfirmed"] == ["claude-opus-4-9"]
    assert _overrides()[_CHAT] == "claude-opus-4-9"


@pytest.mark.parametrize("provider", ["ollama", "fake"])
def test_a_non_claude_persona_is_refused_and_nothing_is_written(tmp_path, provider, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path, provider=provider)
    with client:
        r = _post(client, "opus")
        state = client.get("/persona/state").json()
    assert r.status_code == 400 and r.json() == {"ok": False, "error": "model_not_applicable"}
    assert _CHAT not in _overrides() and capture_claude_argv.argvs == []
    assert state["connection"]["model"] is None and state["connection"]["model_choice"] is None


def test_a_damaged_settings_file_answers_500_and_is_not_touched(tmp_path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        tunables._file_path().write_text("{not json")  # damaged while the bridge runs (a writer never heals it)
        r = _post(client, "haiku")
    assert r.status_code == 500
    body = r.json()
    assert body["ok"] is False and body["error"] == "tunables_unwritable" and "not usable" in body["detail"]
    assert tunables._file_path().read_text() == "{not json"


def test_a_wedged_lock_answers_500_and_writes_nothing(tmp_path, wedged_lock, monkeypatch) -> None:
    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    client, _ = _make_client(tmp_path)
    with client:
        before = tunables._file_path().read_bytes()
        with wedged_lock(tunables._file_path()):
            r = _post(client, "haiku")
        assert tunables._file_path().read_bytes() == before
    assert r.status_code == 500 and r.json()["error"] == "tunables_unwritable"


def test_state_reports_the_effective_id_and_the_picker_choice_and_the_provider_needs_no_swap(
    tmp_path, capture_claude_argv
) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        provider = client.app.state.bridge.provider
        assert client.get("/persona/state").json()["connection"]["model_choice"] == "sonnet"
        assert _post(client, "opus").status_code == 200
        conn = client.get("/persona/state").json()["connection"]
        assert conn["model"] == _OPUS and conn["model_choice"] == "opus"
        provider.generate("x")  # the provider built at start, never told about the change
        assert capture_claude_argv.last_model == _OPUS
        assert _post(client, "claude-sonnet-4-6").status_code == 200
        conn = client.get("/persona/state").json()["connection"]
        assert conn["model"] == "claude-sonnet-4-6" and conn["model_choice"] is None
        assert _post(client, "sonnet").status_code == 200  # the removed line still highlights Sonnet
        conn = client.get("/persona/state").json()["connection"]
        assert conn["model"] == "claude-sonnet-5" and conn["model_choice"] == "sonnet"


def test_the_picker_switch_ignores_the_old_id_until_it_is_configured_again(tmp_path, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        _post(client, "claude-opus-4-9")
        _post(client, "sonnet")
    assert "claude-opus-4-9" not in [r["model"] for r in model_error.state()]


def test_concurrent_saves_run_their_checks_one_at_a_time(tmp_path, monkeypatch) -> None:
    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow_check(model, timeout_s=None):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.15)
        with lock:
            live["now"] -= 1
        return "ok", ""

    monkeypatch.setattr(server, "_check_model_blocking", slow_check)
    client, _ = _make_client(tmp_path)
    results: list[int] = []
    with client:
        threads = [
            threading.Thread(target=lambda m=m: results.append(_post(client, m).status_code))
            for m in ("claude-opus-4-1", "claude-opus-4-2", "claude-opus-4-3")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert results == [200, 200, 200] and live["max"] == 1


def test_endpoint_requires_bearer_auth(tmp_path) -> None:
    client, _ = _make_client(tmp_path, auth_token="secret")
    with client:
        assert client.post("/persona/config/model", json={"model": "opus"}).status_code == 401


def test_endpoint_rejects_a_malformed_body(tmp_path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        assert client.post("/persona/config/model", json={"unrelated": "x"}).status_code == 422


def test_the_chat_job_is_the_only_line_the_endpoint_writes(tmp_path, capture_claude_argv) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        _post(client, "claude-opus-4-9")
    assert set(_overrides()) == {_CHAT}
    assert TIER_INTERACTIVE_CHAT == "interactive-chat"
