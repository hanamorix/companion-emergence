"""S222: the reranker's ONNX session runs with the CPU memory pool and the memory-pattern
planner off. No model is loaded: a stand-in for fastembed's `TextCrossEncoder`, whose inner
model is a REAL `fastembed.common.onnx_model.OnnxModel` (so its own option handling is the
one under test) and a real `onnxruntime.SessionOptions`."""

from __future__ import annotations

from typing import Any

import onnxruntime as ort
import pytest
from fastembed.common.onnx_model import OnnxModel

from brain.memory import reranker as reranker_mod


class _FakeTextCrossEncoder:
    instances: list[_FakeTextCrossEncoder] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model = OnnxModel()
        _FakeTextCrossEncoder.instances.append(self)


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch):
    _FakeTextCrossEncoder.instances = []
    monkeypatch.setattr(
        "fastembed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder", _FakeTextCrossEncoder
    )
    return _FakeTextCrossEncoder


def test_the_provider_asks_fastembed_for_no_memory_pool(fake_fastembed, tmp_path) -> None:
    reranker_mod.CrossEncoderProvider("some/model", tmp_path)
    (inst,) = fake_fastembed.instances
    assert inst.kwargs["enable_cpu_mem_arena"] is False
    assert inst.kwargs["lazy_load"] is True
    assert inst.kwargs["model_name"] == "some/model"


def test_the_options_hook_turns_both_flags_off_on_a_real_session_options(
    fake_fastembed, tmp_path
) -> None:
    reranker_mod.CrossEncoderProvider("some/model", tmp_path)
    (inst,) = fake_fastembed.instances
    options = ort.SessionOptions()
    default_level = options.graph_optimization_level
    assert options.enable_cpu_mem_arena is True and options.enable_mem_pattern is True, (
        "ORT defaults"
    )
    # exactly how fastembed's `_load_onnx_model` calls it
    inst.model.add_extra_session_options(options, {"enable_cpu_mem_arena": False})
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False
    assert options.graph_optimization_level == default_level, "nothing else is touched"


def test_the_hook_is_per_instance_and_leaves_fastembed_alone(fake_fastembed, tmp_path) -> None:
    reranker_mod.CrossEncoderProvider("some/model", tmp_path)
    untouched = ort.SessionOptions()
    OnnxModel.add_extra_session_options(untouched, {"enable_cpu_mem_arena": False})
    assert untouched.enable_cpu_mem_arena is False
    assert untouched.enable_mem_pattern is True, "the class method is not patched for other users"


def test_a_failure_to_install_the_hook_warns_and_construction_still_works(
    monkeypatch, tmp_path, caplog
) -> None:
    class _NoInner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(
        "fastembed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder", _NoInner
    )
    with caplog.at_level("WARNING"):
        provider = reranker_mod.CrossEncoderProvider("some/model", tmp_path)
    assert provider.model_id() == "some/model"
    assert any("memory patterns" in r.getMessage() for r in caplog.records)


def _provider_with_session(options_or_error, fake_fastembed, tmp_path):
    provider = reranker_mod.CrossEncoderProvider("some/model", tmp_path)
    session = type("S", (), {})()
    if isinstance(options_or_error, Exception):

        def _raise():
            raise options_or_error

        session.get_session_options = _raise
    else:
        session.get_session_options = lambda: options_or_error
    provider._model.model.model = session  # noqa: SLF001
    return provider


def _opts(arena: bool, pattern: bool):
    return type("O", (), {"enable_cpu_mem_arena": arena, "enable_mem_pattern": pattern})()


@pytest.mark.parametrize("arena,pattern", [(True, False), (False, True), (True, True)])
def test_a_live_session_with_the_pool_or_patterns_on_is_reported_once(
    fake_fastembed, tmp_path, caplog, arena, pattern
) -> None:
    provider = _provider_with_session(_opts(arena, pattern), fake_fastembed, tmp_path)
    with caplog.at_level("WARNING"):
        provider._check_session_options_once()  # noqa: SLF001
        provider._check_session_options_once()  # noqa: SLF001
    warnings = [r for r in caplog.records if "memory pool" in r.getMessage()]
    assert len(warnings) == 1


def test_a_live_session_with_both_off_is_silent(fake_fastembed, tmp_path, caplog) -> None:
    provider = _provider_with_session(_opts(False, False), fake_fastembed, tmp_path)
    with caplog.at_level("WARNING"):
        provider._check_session_options_once()  # noqa: SLF001
    assert not [r for r in caplog.records if "memory pool" in r.getMessage()]


def test_an_unreadable_session_never_raises(fake_fastembed, tmp_path) -> None:
    provider = _provider_with_session(RuntimeError("no session yet"), fake_fastembed, tmp_path)
    provider._check_session_options_once()  # noqa: SLF001 - must not raise


# --- S249: the provider remembers that THIS process downloaded the model file ---


@pytest.mark.parametrize(("cached", "downloaded"), [(False, True), (True, False), (None, False)])
def test_the_provider_notes_a_download_only_when_the_file_was_not_cached(
    fake_fastembed, tmp_path, monkeypatch, cached, downloaded
) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        reranker_mod, "_model_file_cached", lambda model_id, cache_dir: seen.append((model_id, str(cache_dir))) or cached
    )
    reranker_mod.CrossEncoderProvider("some/model", tmp_path)
    assert seen == [("some/model", str(tmp_path))], "decided BEFORE fastembed's construction could download"
    assert ("some/model" in reranker_mod._downloaded_model_ids) is downloaded  # noqa: SLF001
