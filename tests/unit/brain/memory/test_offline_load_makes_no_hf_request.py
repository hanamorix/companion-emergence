"""An offline model load makes no Hugging Face request, even on the first load of a day (#554).

huggingface_hub 1.30.0 builds an "is an AI agent calling me" user agent every time it builds request headers. Unless
telemetry is off it reads `$HF_HOME/.agent_harnesses.json` and, when that file is missing or a day old, does
`GET {endpoint}/api/agent-harnesses`. `hf_hub_download` builds its headers before its offline short-circuit, so a model
that resolves entirely from the local cache still made that one request on the first load of each day. The scope
`relevance_judge.offline_load_scope` holds the library's telemetry flag off around each offline construction.

Every test here starts from "the first load of a day": the registry file path is absent and the in-process registry is
empty. The environment that would make the old code pass (the library's telemetry or offline flags) is switched off
first, so a developer's or CI's exported `HF_HUB_OFFLINE` / `DO_NOT_TRACK` cannot hide the request. Requests are counted
at `httpx.Client.send`, under every module's session, and the recorded URLs are asserted on. Nothing here loads a
downloaded model: the real-library test builds a from-scratch tiny model into a fabricated cache.
"""

from __future__ import annotations

import shutil
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from brain.memory import judge_full_ft, judge_lora
from brain.memory import relevance_judge as rj

_REGISTRY_URL_SUFFIX = "/api/agent-harnesses"


class _StopError(Exception):
    """Raised by a stub to end a retrain function right after its model construction."""


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


@pytest.fixture(autouse=True)
def fresh_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, str]]:
    """The first header build of a day, with every request recorded (and refused, as offline)."""
    import huggingface_hub.constants as hf_constants
    import huggingface_hub.utils._detect_agent as detect_agent

    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_TELEMETRY", False, raising=True)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False, raising=True)
    monkeypatch.setattr(hf_constants, "AGENT_HARNESSES_PATH", str(tmp_path / "absent" / "agent_harnesses.json"), raising=True)
    monkeypatch.setattr(detect_agent, "_registry", None, raising=True)

    requests: list[tuple[str, str]] = []

    def refuse(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        raise httpx.ConnectError("network blocked in this test", request=request)

    monkeypatch.setattr(httpx.Client, "send", refuse)
    return requests


def _flag() -> bool:
    import huggingface_hub.constants as hf_constants

    return hf_constants.HF_HUB_DISABLE_TELEMETRY


def _registry_requests(requests: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [r for r in requests if r[1].endswith(_REGISTRY_URL_SUFFIX)]


# ---------------------------------------------------------------------------
# R1a: the real judge, the real CrossEncoder, a tiny model in a fabricated cache.
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("torch-training")
@pytest.mark.usefixtures("one_torch_thread")
def test_real_judge_load_from_cache_makes_no_request_on_first_load_of_a_day(
    hf_cache: Path, fresh_day: list[tuple[str, str]]
) -> None:
    assert rj.offline_load_kwargs("org/tiny", hf_cache), "the fabricated cache must count as cached, or this is not an offline load"

    judge = rj.TorchCrossEncoderJudge("org/tiny", hf_cache)
    score = judge.score("query", "relevant")

    assert fresh_day == [], f"an offline judge load made requests: {fresh_day}"
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# R1b: all four torch load sites, with a stub CrossEncoder that makes the same library calls an offline load makes.
# ---------------------------------------------------------------------------


def _install_stub(monkeypatch: pytest.MonkeyPatch, *, stop: bool) -> dict[str, Any]:
    """Replace `sentence_transformers.CrossEncoder` with a stub that resolves `config.json` and the PEFT adapter probe
    through transformers exactly as `CrossEncoder`'s own load does, with the kwargs it was handed."""
    import sentence_transformers

    record: dict[str, Any] = {"flag": [], "config": None}

    class _Stub:
        def __init__(self, model_name_or_path: str, *args: Any, cache_folder: str | None = None, **kwargs: Any) -> None:
            from transformers.utils import find_adapter_config_file
            from transformers.utils.hub import cached_file

            record["flag"].append(_flag())
            offline = bool(kwargs.get("local_files_only", False))
            adapter_kwargs = (kwargs.get("model_kwargs") or {}).get("adapter_kwargs", {})
            record["config"] = cached_file(model_name_or_path, "config.json", cache_dir=cache_folder, local_files_only=offline)
            find_adapter_config_file(model_name_or_path, cache_dir=cache_folder, **adapter_kwargs)
            if stop:
                raise _StopError

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _Stub)
    return record


def _fake_training_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retrain functions import peft and the sentence-transformers trainer stack before they build the model.
    peft is not a local dependency and the trainer imports are slow, so those modules are stand-ins here."""

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    for name, mod in {
        "peft": module("peft", LoraConfig=object, get_peft_model=object),
        "sentence_transformers.cross_encoder.losses": module("losses", BinaryCrossEntropyLoss=object),
        "sentence_transformers.cross_encoder.trainer": module("trainer", CrossEncoderTrainer=object),
        "sentence_transformers.cross_encoder.training_args": module("training_args", CrossEncoderTrainingArguments=object),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


_TRIPLES = [("query", "relevant", "relevant"), ("query", "irrelevant", "irrelevant")]


def _site_judge(cache: Path) -> None:
    rj.TorchCrossEncoderJudge("org/tiny", cache)


def _site_load_full_scorer(cache: Path) -> None:
    judge_full_ft.load_full_scorer("org/tiny", cache_dir=cache)


def _site_full_ft_retrain(cache: Path) -> None:
    with pytest.raises(_StopError):
        judge_full_ft.build_full_ft_retrain_fn("org/tiny", cache_dir=cache, epochs=1)(_TRIPLES)


def _site_lora_retrain(cache: Path) -> None:
    with pytest.raises(_StopError):
        judge_lora.build_lora_retrain_fn(
            "org/tiny", target_modules=["q"], modules_to_save=["classifier"], cache_dir=cache, epochs=1
        )(_TRIPLES)


@pytest.mark.parametrize(
    ("site", "stops"),
    [
        pytest.param(_site_judge, False, id="judge-init"),
        pytest.param(_site_load_full_scorer, False, id="load-full-scorer"),
        pytest.param(_site_full_ft_retrain, True, id="full-ft-retrain"),
        pytest.param(_site_lora_retrain, True, id="lora-retrain"),
    ],
)
def test_every_offline_torch_load_site_makes_no_request_on_first_load_of_a_day(
    site: Callable[[Path], None],
    stops: bool,
    hf_cache: Path,
    fresh_day: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert rj.offline_load_kwargs("org/tiny", hf_cache), "the fabricated cache must count as cached, or this is not an offline load"
    record = _install_stub(monkeypatch, stop=stops)
    _fake_training_modules(monkeypatch)

    site(hf_cache)

    assert fresh_day == [], f"an offline load made requests: {fresh_day}"
    assert isinstance(record["config"], str) and record["config"].endswith("config.json"), "the stub did not resolve the cached file"
    assert record["flag"] == [True], f"the library's telemetry flag was not held off during the load: {record['flag']}"


# ---------------------------------------------------------------------------
# R2, R2b: the online path is unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.no_base_failure(
    reason="guards that the fix does not widen to the online path: on the old code nothing is held off, so this passes by design"
)
def test_online_judge_load_still_reaches_the_registry_with_the_flag_untouched(
    hf_cache: Path, fresh_day: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rj, "offline_load_kwargs", lambda *a, **kw: {})
    record = _install_stub(monkeypatch, stop=False)

    rj.TorchCrossEncoderJudge("org/tiny", hf_cache)

    assert record["flag"] == [False]
    assert _registry_requests(fresh_day), f"an online load did not do the library's usual registry fetch: {fresh_day}"


def test_online_load_after_an_offline_load_still_fetches_the_registry(
    hf_cache: Path, fresh_day: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_stub(monkeypatch, stop=False)

    rj.TorchCrossEncoderJudge("org/tiny", hf_cache)
    assert fresh_day == [], f"the offline load made requests: {fresh_day}"

    monkeypatch.setattr(rj, "offline_load_kwargs", lambda *a, **kw: {})
    rj.TorchCrossEncoderJudge("org/tiny", hf_cache)

    assert record["flag"] == [True, False]
    assert _registry_requests(fresh_day), "the offline scope left the library unable to do its normal registry fetch afterwards"


# ---------------------------------------------------------------------------
# R3, R4: the scope puts the flag back, also across threads.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [False, True])
def test_scope_holds_the_flag_on_and_puts_it_back(start: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_TELEMETRY", start, raising=True)

    with rj.offline_load_scope({"local_files_only": True}):
        assert _flag() is True
    assert _flag() is start

    with pytest.raises(RuntimeError, match="boom"):
        with rj.offline_load_scope({"local_files_only": True}):
            raise RuntimeError("boom")
    assert _flag() is start


@pytest.mark.parametrize("start", [False, True])
def test_scope_with_no_offline_kwargs_leaves_the_flag_alone(start: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_TELEMETRY", start, raising=True)
    with rj.offline_load_scope({}):
        assert _flag() is start
    assert _flag() is start


def test_overlapping_scopes_in_two_threads_lose_no_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """A enters, B enters, A leaves while B is still loading: the flag stays on for B; B leaves: it is back to its start."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_TELEMETRY", False, raising=True)
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen_by_b_after_a_left: list[bool] = []
    errors: list[BaseException] = []

    def a() -> None:
        try:
            with rj.offline_load_scope({"local_files_only": True}):
                a_in.set()
                assert b_in.wait(10)
        except BaseException as exc:  # noqa: BLE001 - reported to the test thread
            errors.append(exc)
        finally:
            a_out.set()

    def b() -> None:
        try:
            assert a_in.wait(10)
            with rj.offline_load_scope({"local_files_only": True}):
                b_in.set()
                assert a_out.wait(10)
                seen_by_b_after_a_left.append(_flag())
        except BaseException as exc:  # noqa: BLE001 - reported to the test thread
            errors.append(exc)

    threads = [threading.Thread(target=a), threading.Thread(target=b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)

    assert errors == []
    assert seen_by_b_after_a_left == [True], "the first thread to leave switched the flag off under the second thread's load"
    assert _flag() is False, "the flag was left on after both loads finished"


# ---------------------------------------------------------------------------
# R5: the judge's online retry runs outside the scope.
# ---------------------------------------------------------------------------


def test_judge_online_retry_after_a_failed_offline_attempt_runs_with_the_flag_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    flags: list[bool] = []

    class _OfflineFailsOnlineWorks:
        def __init__(self, model_id: str, cache_folder: str | None = None, device: str | None = None, **kwargs: Any) -> None:
            flags.append(_flag())
            if kwargs.get("local_files_only"):
                raise OSError("simulated corrupted offline load")

    monkeypatch.setattr(rj, "offline_load_kwargs", lambda *a, **kw: {"local_files_only": True})
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _OfflineFailsOnlineWorks)

    rj.TorchCrossEncoderJudge("fake-judge-model", "/tmp/fake-cache-dir")

    assert flags == [True, False], f"expected the flag on for the offline attempt and back for the online retry, got {flags}"


# ---------------------------------------------------------------------------
# R6: the real library's behaviour under the scope.
# ---------------------------------------------------------------------------


def test_scope_stops_the_real_header_build_from_asking_for_the_registry(
    fresh_day: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub.utils._detect_agent as detect_agent
    from huggingface_hub.utils import build_hf_headers

    with rj.offline_load_scope({"local_files_only": True}):
        headers = build_hf_headers(token=False)
    assert fresh_day == [], f"header building under the scope made requests: {fresh_day}"
    assert "agent/" not in headers["user-agent"]

    monkeypatch.setattr(detect_agent, "_registry", None, raising=True)
    build_hf_headers(token=False)
    assert _registry_requests(fresh_day), "outside the scope the library no longer asks for the registry; this test would prove nothing"
