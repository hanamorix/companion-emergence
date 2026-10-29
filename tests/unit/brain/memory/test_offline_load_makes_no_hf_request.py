"""The app never asks Hugging Face for telemetry, so an offline model load makes no request (#554).

huggingface_hub 1.30.0 builds an "is an AI agent calling me" user agent every time it builds request headers. Unless
telemetry is off it reads `$HF_HOME/.agent_harnesses.json` and, when that file is missing or a day old, does
`GET {endpoint}/api/agent-harnesses`. `hf_hub_download` builds its headers before its offline short-circuit, so a model
that resolves entirely from the local cache still made that one request on the first load of each day. The app wants no
telemetry or agent detection at all, online or offline, so `brain/__init__.py` sets the library's own opt-out
(`HF_HUB_DISABLE_TELEMETRY=1`) before any module can import the library, unless the user set a value.

The library reads that setting from the environment when it is first imported, so every test here is a fresh interpreter
(`python -P -c`). Its environment has none of the opt-outs the library knows (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
`HF_HUB_DISABLE_TELEMETRY`, `DISABLE_TELEMETRY`, `DO_NOT_TRACK`) and no `HF_ENDPOINT`, so a developer's or CI's exported
variable cannot hide a request, and `HF_HOME` is an empty tmp dir: the registry file is absent, which is the first header
build of a day. Each script checks that precondition itself. `PYTHONPATH` puts this checkout first, because `-P` plus an
editable install would otherwise resolve `brain` to another checkout (a worktree's tests would test main's code).

Requests are counted at `httpx.Client.send`, under every module's session, and the recorded URLs are asserted on. A request
is refused (`ConnectError`) unless the test installs a stand-in server: the registry URL answers a fake harness registry
keyed to `TEST_AGENT_ENV` (so an `agent/` field shows up in the user agent whenever telemetry is on), and the model file URL
answers a HEAD and a GET. Nothing here loads a downloaded model: the real-library tests build a from-scratch tiny model into a
fabricated cache.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tests.source_tree import REPO

_REGISTRY_URL_SUFFIX = "/api/agent-harnesses"
_LIBRARY_ENV = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
    "HF_ENDPOINT",
)
_TIMEOUT_S = 240

# Shared by every script. Defines the request recorder and the precondition checks; imports neither brain nor the library,
# so each script decides the import order that it tests.
_PRELUDE = textwrap.dedent(
    """
    import json
    import os
    import sys

    import httpx

    REPO_ROOT = os.path.realpath(os.environ["TEST_REPO_ROOT"])
    SERVE_REGISTRY = os.environ.get("TEST_SERVE_REGISTRY") == "1"
    SERVE_FILE = os.environ.get("TEST_SERVE_FILE") == "1"
    FILE_BODY = b'{"model_type": "gpt2"}'
    requests = []


    def _send(self, request, **kwargs):
        url = str(request.url)
        requests.append({"method": request.method, "url": url, "user_agent": request.headers.get("user-agent", "")})
        if SERVE_REGISTRY and url.endswith("/api/agent-harnesses"):
            registry = {"standardEnvVars": [], "harnesses": {"testagent": {"envVars": {"TEST_AGENT_ENV": "*"}}}}
            return httpx.Response(200, json=registry, request=request)
        if SERVE_FILE and url.endswith("/org/tiny/resolve/main/config.json"):
            headers = {"ETag": '"abc"', "X-Repo-Commit": "c" * 40, "Content-Length": str(len(FILE_BODY))}
            if request.method == "HEAD":
                return httpx.Response(200, headers=headers, request=request)
            return httpx.Response(200, headers=headers, content=FILE_BODY, request=request)
        raise httpx.ConnectError("network blocked in this test", request=request)


    httpx.Client.send = _send


    def check_brain_is_this_checkout():
        import brain

        assert os.path.realpath(brain.__file__).startswith(REPO_ROOT), brain.__file__


    def check_library_state():
        # The precondition of every request-counting claim: the library is neither offline nor holding a registry file.
        import huggingface_hub.constants as constants

        assert constants.HF_HUB_OFFLINE is False, "the library is in offline mode, so it would never ask for the registry"
        assert not os.path.exists(constants.AGENT_HARNESSES_PATH), constants.AGENT_HARNESSES_PATH


    def headers_made_by_the_real_library():
        from huggingface_hub.utils import build_hf_headers

        return build_hf_headers(token=False)["user-agent"]
    """
)


def _run(script: str, tmp_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run `script` in a fresh interpreter whose environment is the library-neutral one described in the module docstring,
    plus `env`; return the JSON dict on its last stdout line."""
    child_env = {k: v for k, v in os.environ.items() if k not in _LIBRARY_ENV}
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir(exist_ok=True)
    child_env["HF_HOME"] = str(hf_home)
    child_env["PYTHONPATH"] = os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])])
    child_env["TEST_REPO_ROOT"] = str(REPO)
    child_env["OMP_NUM_THREADS"] = "1"
    child_env.update(env or {})
    proc = subprocess.run(
        [sys.executable, "-P", "-c", _PRELUDE + textwrap.dedent(script)],
        cwd=str(REPO),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"no stdout: stderr={proc.stderr!r}"
    return json.loads(lines[-1])


def _registry_requests(requests: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in requests if r["url"].endswith(_REGISTRY_URL_SUFFIX)]


# ---------------------------------------------------------------------------
# S1: the real offline loads, on the first header build of a day, make no request.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.unit.brain.memory.test_judge_full_ft import _build_tiny_model_dir

    return _build_tiny_model_dir(tmp_path_factory.mktemp("offline_load_tiny") / "tiny_model")


@pytest.fixture
def hf_cache(tiny_model_dir: Path, tmp_path: Path) -> Path:
    """A Hugging Face cache dir holding the tiny model as repo `org/tiny` at revision `rev1` (the hub layout)."""
    cache = tmp_path / "hf-cache"
    repo = cache / "models--org--tiny"
    shutil.copytree(tiny_model_dir, repo / "snapshots" / "rev1")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    return cache


_JUDGE_SITE = """
    import brain.memory.relevance_judge as rj
    check_brain_is_this_checkout()
    check_library_state()
    cache = os.environ["TEST_HF_CACHE"]
    assert rj.offline_load_kwargs("org/tiny", cache), "the fabricated cache must count as cached, or this is not an offline load"
    score = rj.TorchCrossEncoderJudge("org/tiny", cache).score("query", "relevant")
    print(json.dumps({"requests": requests, "score_is_float": isinstance(score, float)}))
"""

_FULL_SCORER_SITE = """
    import brain.memory.judge_full_ft as full_ft
    import brain.memory.relevance_judge as rj
    check_brain_is_this_checkout()
    check_library_state()
    cache = os.environ["TEST_HF_CACHE"]
    assert rj.offline_load_kwargs("org/tiny", cache), "the fabricated cache must count as cached, or this is not an offline load"
    score = full_ft.load_full_scorer("org/tiny", cache_dir=cache)(("query", "relevant"))
    print(json.dumps({"requests": requests, "score_is_float": isinstance(score, float)}))
"""


@pytest.mark.xdist_group("torch-training")
@pytest.mark.parametrize("site", [_JUDGE_SITE, _FULL_SCORER_SITE], ids=["judge-init", "load-full-scorer"])
def test_real_offline_load_makes_no_request_on_first_load_of_a_day(site: str, hf_cache: Path, tmp_path: Path) -> None:
    result = _run(site, tmp_path, {"TEST_HF_CACHE": str(hf_cache)})

    assert result["requests"] == [], f"an offline load made requests: {[r['url'] for r in result['requests']]}"
    assert result["score_is_float"]


# ---------------------------------------------------------------------------
# S2, S3: the opt-out is on after importing any entry module, and when the library was imported first.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["brain", "brain.cli", "brain.bridge.runner", "brain.mcp_server"])
def test_importing_an_entry_module_turns_the_library_telemetry_off(module: str, tmp_path: Path) -> None:
    result = _run(
        f"""
        import importlib
        importlib.import_module({module!r})
        check_brain_is_this_checkout()
        assert "huggingface_hub" not in sys.modules, "importing the entry module imported the library"
        import huggingface_hub.constants as constants
        print(json.dumps({{"env": os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "flag": constants.HF_HUB_DISABLE_TELEMETRY}}))
        """,
        tmp_path,
    )

    assert result == {"env": "1", "flag": True}


def test_a_library_imported_before_brain_is_covered_too(tmp_path: Path) -> None:
    result = _run(
        """
        import huggingface_hub.constants as constants
        check_library_state()
        before = constants.HF_HUB_DISABLE_TELEMETRY
        import brain
        check_brain_is_this_checkout()
        user_agent = headers_made_by_the_real_library()
        print(json.dumps({"before": before, "after": constants.HF_HUB_DISABLE_TELEMETRY, "requests": requests,
                          "user_agent": user_agent}))
        """,
        tmp_path,
        {"TEST_SERVE_REGISTRY": "1", "TEST_AGENT_ENV": "1"},
    )

    assert result["before"] is False, "the apparatus did not start from the library's default"
    assert result["after"] is True
    assert result["requests"] == [], f"header building made requests: {[r['url'] for r in result['requests']]}"
    assert "torch/" not in result["user_agent"] and "agent/" not in result["user_agent"], result["user_agent"]


# ---------------------------------------------------------------------------
# S4: a value the user set is theirs.
# ---------------------------------------------------------------------------

_HEADERS_AFTER_BRAIN = """
    import brain
    check_brain_is_this_checkout()
    import huggingface_hub.constants as constants
    check_library_state()
    user_agent = headers_made_by_the_real_library()
    print(json.dumps({"env": os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "flag": constants.HF_HUB_DISABLE_TELEMETRY,
                      "requests": requests, "user_agent": user_agent}))
"""
_SERVED = {"TEST_SERVE_REGISTRY": "1", "TEST_AGENT_ENV": "1"}


@pytest.mark.no_base_failure(
    reason="the user's own value is respected both before and after this change, so the old code passes by design; this also "
    "proves the apparatus can see the registry request and the agent field when telemetry is on"
)
def test_a_user_set_zero_is_respected_and_gets_the_library_default(tmp_path: Path) -> None:
    result = _run(_HEADERS_AFTER_BRAIN, tmp_path, {**_SERVED, "HF_HUB_DISABLE_TELEMETRY": "0"})

    assert result["env"] == "0"
    assert result["flag"] is False
    assert [r["url"].rsplit("/api/", 1)[-1] for r in _registry_requests(result["requests"])] == ["agent-harnesses"]
    assert "agent/testagent" in result["user_agent"], result["user_agent"]


@pytest.mark.no_base_failure(reason="the user's own value is respected both before and after this change, so the old code passes by design")
def test_a_user_set_one_is_respected(tmp_path: Path) -> None:
    result = _run(_HEADERS_AFTER_BRAIN, tmp_path, {**_SERVED, "HF_HUB_DISABLE_TELEMETRY": "1"})

    assert result["env"] == "1"
    assert result["flag"] is True
    assert result["requests"] == []


@pytest.mark.parametrize("empty", ["", "  "])
def test_an_empty_value_is_not_a_choice(empty: str, tmp_path: Path) -> None:
    result = _run(_HEADERS_AFTER_BRAIN, tmp_path, {**_SERVED, "HF_HUB_DISABLE_TELEMETRY": empty})

    assert result["env"] == "1"
    assert result["flag"] is True
    assert result["requests"] == [], f"header building made requests: {[r['url'] for r in result['requests']]}"


# ---------------------------------------------------------------------------
# S5: the online download path still works; only the headers are trimmed.
# ---------------------------------------------------------------------------

_ONLINE_DOWNLOAD = """
    import brain
    check_brain_is_this_checkout()
    check_library_state()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("org/tiny", "config.json", cache_dir=os.environ["TEST_ONLINE_CACHE"])
    with open(path, "rb") as f:
        content = f.read().decode()
    print(json.dumps({"content": content, "requests": requests}))
"""


def _download(tmp_path: Path, env: dict[str, str]) -> dict[str, Any]:
    return _run(
        _ONLINE_DOWNLOAD,
        tmp_path,
        {"TEST_SERVE_REGISTRY": "1", "TEST_SERVE_FILE": "1", "TEST_AGENT_ENV": "1", "TEST_ONLINE_CACHE": str(tmp_path / "online"), **env},
    )


def test_online_download_still_works_with_only_the_telemetry_fields_trimmed(tmp_path: Path) -> None:
    result = _download(tmp_path, {})

    assert result["content"] == '{"model_type": "gpt2"}'
    assert [(r["method"], r["url"].rsplit("/org/tiny/", 1)[-1]) for r in result["requests"]] == [
        ("HEAD", "resolve/main/config.json"),
        ("GET", "resolve/main/config.json"),
    ], f"unexpected requests: {[r['url'] for r in result['requests']]}"
    for request in result["requests"]:
        agent = request["user_agent"]
        assert agent.startswith("unknown/None; hf_hub/"), agent
        assert "torch/" not in agent and "agent/" not in agent, agent


@pytest.mark.no_base_failure(
    reason="control: with the user's own HF_HUB_DISABLE_TELEMETRY=0 the old and the new code both do the library's default, so this "
    "passes on the old code by design; it shows the same download works with the registry request and the agent field present"
)
def test_online_download_control_with_the_user_default_asks_for_the_registry(tmp_path: Path) -> None:
    result = _download(tmp_path, {"HF_HUB_DISABLE_TELEMETRY": "0"})

    assert result["content"] == '{"model_type": "gpt2"}'
    assert len(_registry_requests(result["requests"])) == 1
    model_requests = [r for r in result["requests"] if not r["url"].endswith(_REGISTRY_URL_SUFFIX)]
    assert [r["method"] for r in model_requests] == ["HEAD", "GET"]
    assert all("agent/testagent" in r["user_agent"] for r in model_requests), [r["user_agent"] for r in model_requests]
