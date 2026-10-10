"""S223: a reply that starts while the RAM ladder's call runs STOPS that call (ONNX Runtime
`RunOptions.terminate`), instead of letting the reply wait for it.

fastembed's own call passes no run options, so `CrossEncoderProvider` wraps the loaded session's
`run` (only after something armed a stop) and hands the arming thread's `RunOptions` to ONNX
Runtime. Real `onnxruntime.RunOptions`, a real `OnnxModel`, the real provider, the real
`cli_throttle` hooks and the real ladder / `floor_startup` thread; the session is scripted (it
polls `RunOptions.terminate` the way ONNX Runtime does). The real-model oracle is the records'
`real_terminate_check.py`. No model is loaded here. Synthetic data only."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import onnxruntime as ort
import pytest
from fastembed.common.onnx_model import OnnxModel

from brain.bridge import cli_throttle
from brain.memory import reranker as reranker_mod
from brain.memory.reranker import RamMeasurementAbandoned, RerankerProvider, RerankTerminated
from brain.memory.store import MemoryStore
from tests.unit.brain.memory._ram_support import (
    FULL,
    ScriptedLadderProvider,
    shapes_in_window,
    true_growth,
)
from tests.unit.brain.memory.test_ram_figure_abandon import _Loop

_MODEL = "scripted-ladder"  # the id test_ram_figure_abandon._Loop flags as needed
_TIMEOUT = 10.0
_TERMINATED = "Exiting due to terminate flag being set to true."


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 0.0)  # phase 1 only
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)


class _Session:
    """A scripted ONNX session. `run` honours a real `RunOptions.terminate` the way ONNX Runtime
    does (it polls the flag while it works), records every call, and can block."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict, Any]] = []
        self.entered = threading.Event()  # set on the first call that blocks
        self.release = threading.Event()
        self.block_lengths: set[int] = set()  # calls with this padded length block until released/stopped
        self.block_all = False
        self.stopped_by_flag = 0
        self.fail_with: Exception | None = None
        self.last_growth = 0.0

    def run(self, output_names, input_feed, run_options=None):
        self.calls.append((output_names, input_feed, run_options))
        self.last_growth = true_growth(input_feed["pairs"], input_feed["length"])
        blocking = self.block_all or input_feed["length"] in self.block_lengths
        if blocking:
            self.entered.set()
        deadline = time.monotonic() + _TIMEOUT
        while True:
            if run_options is not None and run_options.terminate:
                self.stopped_by_flag += 1
                raise RuntimeError(_TERMINATED)
            if not blocking or self.release.is_set():
                break
            if time.monotonic() > deadline:
                raise AssertionError("the call was never stopped")
            time.sleep(0.001)
        if self.fail_with is not None:
            raise self.fail_with
        return [[0.0]]


class _Tokenizer:
    def encode_batch(self, pairs):
        return [SimpleNamespace(attention_mask=[1] * _tokens(q, d)) for q, d in pairs]


def _tokens(query: str, document: str) -> int:
    return min(FULL, (len(query) + len(document)) // 4)


class _FakeEncoder:
    """Stands in for fastembed's `TextCrossEncoder`: `rerank` batches the documents and calls
    `self.model.model.run(...)` once per batch, like fastembed's `OnnxCrossEncoderModel`."""

    batch_size = 64
    reaches_session = True

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model = OnnxModel()  # the real fastembed inner model
        self.model.tokenizer = _Tokenizer()

    def rerank(self, query, documents):
        docs = list(documents)
        scores: list[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start : start + self.batch_size]
            if self.reaches_session:
                length = max(_tokens(query, d) for d in batch)
                self.model.model.run(None, {"pairs": len(batch), "length": length})
            scores += [0.0] * len(batch)
        return scores


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("fastembed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder", _FakeEncoder)

    def _make(session: _Session | None = None, model_id: str = _MODEL):
        provider = reranker_mod.CrossEncoderProvider(model_id, tmp_path)
        if session is not None:
            provider._model.model.model = session  # noqa: SLF001
        # the peak reader is the real Linux one elsewhere; here the "process" grows as the scripted law says
        monkeypatch.setattr(reranker_mod.peak_rss, "begin", lambda: 0.0)
        monkeypatch.setattr(
            reranker_mod.peak_rss, "growth_since", lambda before: session.last_growth if session else None
        )
        return provider

    return _make


def _on_thread(fn, *args) -> tuple[threading.Thread, dict]:
    result: dict[str, Any] = {}

    def _body() -> None:
        try:
            result["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_body, daemon=True)
    thread.start()
    return thread, result


# ---------------------------------------------------------------------------
# T3: wiring
# ---------------------------------------------------------------------------


def test_a_provider_nobody_armed_never_touches_the_sessions_run(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    original = session.run
    provider.rerank("q", ["d"])
    provider.rerank_timed("q", ["d"])
    assert session.run == original, "no wrapper was installed: the bridge and plain recall are unchanged"
    assert [c[2] for c in session.calls] == [None, None], "no RunOptions are created for unarmed callers"


def test_the_wrapper_is_installed_once_after_the_first_arm_and_only_before_a_scored_call(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    original = session.run
    provider.arm_terminate()
    assert session.run == original, "arming alone changes nothing; the install happens inside the next call's lock"
    provider.rerank("q", ["d"])
    wrapper = session.run
    assert wrapper != original
    provider.rerank_timed("q", ["d"])
    provider.rerank("q", ["d"])
    assert session.run is wrapper, "installed once, never wrapped again"
    provider.disarm_terminate()


def test_an_armed_call_hands_a_real_run_options_to_the_session_and_returns_the_scores(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    provider.arm_terminate()
    scores = provider.rerank("q", ["a", "b"])
    assert scores == [0.0, 0.0]
    (_names, feed, options), = session.calls
    assert isinstance(options, ort.RunOptions) and options.terminate is False
    assert feed["pairs"] == 2


def test_after_disarm_and_for_other_threads_the_wrapper_forwards_untouched(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    provider.arm_terminate()
    provider.rerank("q", ["d"])  # installs the wrapper
    provider.disarm_terminate()
    provider.rerank("q", ["d"])
    other, result = _on_thread(provider.rerank, "q", ["d"])
    other.join(_TIMEOUT)
    assert "error" not in result
    assert [c[2] is not None for c in session.calls] == [True, False, False]


def test_options_supplied_by_the_caller_are_passed_through(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    provider.arm_terminate()
    provider.rerank("q", ["d"])
    mine = ort.RunOptions()
    session.run(None, {"pairs": 1, "length": 1}, mine)  # the installed wrapper
    assert session.calls[-1][2] is mine


def test_a_session_without_a_session_gets_loaded_by_the_install(make_provider) -> None:
    provider = make_provider(None)
    inner = provider._model.model  # noqa: SLF001
    inner.model = None
    session = _Session()
    loads: list[int] = []

    def _load() -> None:
        loads.append(1)
        inner.model = session

    inner.load_onnx_model = _load
    provider.arm_terminate()
    provider.rerank("q", ["d"])
    assert loads == [1] and len(session.calls) == 1
    assert isinstance(session.calls[0][2], ort.RunOptions), "the loaded session is the wrapped one"


def test_a_session_whose_run_cannot_be_replaced_warns_once_and_everything_still_works(
    make_provider, caplog: pytest.LogCaptureFixture
) -> None:
    class _Frozen:
        __slots__ = ("calls",)

        def __init__(self) -> None:
            self.calls = 0

        def run(self, output_names, input_feed, run_options=None):
            self.calls += 1
            return [[0.0]]

    frozen = _Frozen()
    provider = make_provider(None)
    provider._model.model.model = frozen  # noqa: SLF001
    stop = provider.arm_terminate()
    with caplog.at_level("WARNING", logger=reranker_mod.log.name):
        assert provider.rerank("q", ["d"]) == [0.0]
        assert provider.rerank("q", ["d"]) == [0.0]
    stop()  # harmless
    assert frozen.calls == 2
    assert len([r for r in caplog.records if "stoppable" in r.getMessage()]) == 1


def test_a_scoring_call_that_never_reaches_the_session_warns_once_but_an_empty_list_does_not(
    make_provider, caplog: pytest.LogCaptureFixture
) -> None:
    session = _Session()
    provider = make_provider(session)
    provider._model.reaches_session = False  # noqa: SLF001
    provider.arm_terminate()
    with caplog.at_level("WARNING", logger=reranker_mod.log.name):
        provider.rerank("q", [])
        assert not [r for r in caplog.records if "did not go through" in r.getMessage()]
        provider.rerank("q", ["d"])
        provider.rerank("q", ["d"])
    assert len([r for r in caplog.records if "did not go through" in r.getMessage()]) == 1


def test_the_base_provider_cannot_stop_its_call() -> None:
    provider = ScriptedLadderProvider()
    stop = provider.arm_terminate()
    assert stop() is None
    assert provider.disarm_terminate() is None
    assert isinstance(provider, RerankerProvider)


def test_without_onnxruntime_run_options_arming_is_a_warned_noop(make_provider, monkeypatch, caplog) -> None:
    provider = make_provider(_Session())

    def _boom() -> None:
        raise RuntimeError("no run options")

    monkeypatch.setattr(ort, "RunOptions", _boom)
    with caplog.at_level("WARNING", logger=reranker_mod.log.name):
        stop = provider.arm_terminate()
    assert stop() is None and provider._want_terminable is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# T2: the stop
# ---------------------------------------------------------------------------


def test_a_stop_during_the_call_ends_it_with_rerank_terminated_chained_to_the_sessions_error(make_provider) -> None:
    """The scope is armed on the thread that makes the call; the stop is called from another one."""
    session = _Session()
    session.block_all = True
    provider = make_provider(session)
    armed = threading.Event()
    holder: dict[str, Any] = {}

    def _call() -> Any:
        holder["stop"] = provider.arm_terminate()
        armed.set()
        return provider.rerank("q", ["d"])

    thread, result = _on_thread(_call)
    assert armed.wait(_TIMEOUT) and session.entered.wait(_TIMEOUT)
    holder["stop"]()
    thread.join(_TIMEOUT)
    error = result["error"]
    assert isinstance(error, RerankTerminated) and isinstance(error.__cause__, RuntimeError)
    assert session.stopped_by_flag == 1


def test_a_stop_made_on_another_thread_than_the_one_that_armed_does_not_touch_that_threads_nothing(
    make_provider,
) -> None:
    """The options belong to the arming thread: a call made by a thread that did not arm uses none."""
    session = _Session()
    provider = make_provider(session)
    provider.arm_terminate()  # armed on THIS thread
    provider.rerank("q", ["d"])  # installs the wrapper
    other, result = _on_thread(provider.rerank, "q", ["d"])
    other.join(_TIMEOUT)
    assert result == {"value": [0.0]} and session.calls[-1][2] is None


def test_a_stop_before_the_call_makes_the_call_start_terminated(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    stop = provider.arm_terminate()
    stop()
    started = time.monotonic()
    with pytest.raises(RerankTerminated):
        provider.rerank_timed("q", ["d"])
    assert time.monotonic() - started < _TIMEOUT / 2
    assert session.stopped_by_flag == 1


def test_an_error_that_is_not_ours_propagates_unchanged(make_provider) -> None:
    session = _Session()
    session.fail_with = ValueError("a real model error")
    provider = make_provider(session)
    provider.arm_terminate()
    with pytest.raises(ValueError) as excinfo:
        provider.rerank("q", ["d"])
    assert excinfo.value is session.fail_with


def test_a_stale_stop_cannot_poison_a_later_arm_or_an_unarmed_call(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    old_stop = provider.arm_terminate()
    provider.rerank("q", ["d"])
    provider.disarm_terminate()
    new_stop = provider.arm_terminate()
    old_stop()  # a late callback of the earlier scope
    assert provider.rerank("q", ["d"]) == [0.0], "the new scope has its own options"
    provider.disarm_terminate()
    old_stop()
    new_stop()
    assert provider.rerank("q", ["d"]) == [0.0], "an unarmed call is not affected by any stop"


def test_a_stop_after_the_last_batch_changes_nothing(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    stop = provider.arm_terminate()
    assert provider.rerank("q", ["d"]) == [0.0]
    stop()
    provider.disarm_terminate()
    assert provider.rerank("q", ["d"]) == [0.0]


def test_a_many_batch_rerank_stops_at_the_first_terminated_batch(make_provider) -> None:
    session = _Session()
    provider = make_provider(session)
    provider._model.batch_size = 2  # noqa: SLF001
    stop = provider.arm_terminate()
    stop()
    with pytest.raises(RerankTerminated):
        provider.rerank("q", ["d"] * 7)
    assert len(session.calls) == 1, "no further batch is run"


def test_a_stop_cannot_reach_another_threads_running_call_and_the_armed_call_starts_terminated(make_provider) -> None:
    """Review M1: thread Y (the floor bootstrap's stand-in) is mid-call and holds the provider; thread X
    arms, waits behind the lock, and its stop fires meanwhile. Y's call is untouched; X's call is
    terminated when it finally starts."""
    session = _Session()
    session.block_all = True
    provider = make_provider(session)
    y_thread, y_result = _on_thread(provider.rerank, "q", ["d"])
    assert session.entered.wait(_TIMEOUT)

    def _x() -> Any:
        stop = provider.arm_terminate()
        x_armed.set()
        stop()  # the reply's start, while X waits for Y's call to finish
        return provider.rerank("q", ["d"])

    x_armed = threading.Event()
    x_thread, x_result = _on_thread(_x)
    assert x_armed.wait(_TIMEOUT)
    time.sleep(0.05)
    session.block_all = False
    session.release.set()  # Y's call finishes normally
    y_thread.join(_TIMEOUT)
    x_thread.join(_TIMEOUT)
    assert y_result == {"value": [0.0]}, "the other thread's call was not stopped"
    assert isinstance(x_result["error"], RerankTerminated)
    assert session.calls[0][2] is None, "Y ran with no run options"


# ---------------------------------------------------------------------------
# T4/T6: the ladder
# ---------------------------------------------------------------------------


class _CountingProvider(ScriptedLadderProvider):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.armed = 0
        self.disarmed = 0
        self.stops: list[int] = []
        self.raise_in_timed: Exception | None = None

    def arm_terminate(self):
        self.armed += 1
        return lambda: self.stops.append(1)

    def disarm_terminate(self) -> None:
        self.disarmed += 1

    def rerank_timed(self, query, documents):
        if self.raise_in_timed is not None:
            raise self.raise_in_timed
        return super().rerank_timed(query, documents)


def _rung(provider, store, watch):
    base = "The quick brown fox jumps over the lazy dog. " * 200
    return reranker_mod._run_rung(  # noqa: SLF001
        provider, store, _MODEL, base, reranker_mod._Rung(1, FULL, True), FULL, watch  # noqa: SLF001
    )


@pytest.mark.parametrize(
    "error", [None, ValueError("boom"), RerankTerminated("x"), RamMeasurementAbandoned("y")], ids=lambda e: type(e).__name__
)
def test_a_rung_arms_before_the_call_and_unarms_on_every_exit(tmp_path, error) -> None:
    store, provider = MemoryStore(tmp_path / "memories.db"), _CountingProvider(model=_MODEL)
    provider.raise_in_timed = error
    watch = cli_throttle.reply_watch()
    try:
        if error is None:
            assert _rung(provider, store, watch) is True
        else:
            with pytest.raises(type(error) if not isinstance(error, RerankTerminated) else RamMeasurementAbandoned):
                _rung(provider, store, watch)
    finally:
        store.close()
    assert (provider.armed, provider.disarmed) == (1, 1)
    assert cli_throttle._start_hooks == [], "the hook is unregistered"  # noqa: SLF001


def test_a_terminated_call_is_an_abandon_and_records_nothing(tmp_path) -> None:
    store, provider = MemoryStore(tmp_path / "memories.db"), _CountingProvider(model=_MODEL)
    provider.raise_in_timed = RerankTerminated("stopped")
    with pytest.raises(RamMeasurementAbandoned):
        _rung(provider, store, cli_throttle.reply_watch())
    assert shapes_in_window(store, _MODEL) == {}
    assert reranker_mod._rerank_cost.state(store, _MODEL).sums.n == 0  # noqa: SLF001
    store.close()


def test_no_watch_and_a_plain_callable_watch_arm_nothing(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    for watch in (None, lambda: True):
        provider = _CountingProvider(model=_MODEL)
        assert _rung(provider, store, watch) is True
        assert (provider.armed, provider.disarmed) == (0, 0)
    store.close()


def test_the_reply_start_hook_calls_the_providers_stop(tmp_path) -> None:
    store, provider = MemoryStore(tmp_path / "memories.db"), _CountingProvider(model=_MODEL)
    watch = cli_throttle.reply_watch()
    provider.on_timed = lambda pairs, length: cli_throttle.note_user_message()  # the reply starts in the call
    with pytest.raises(RamMeasurementAbandoned):
        _rung(provider, store, watch)  # the post-call check discards (this provider's stop is scripted)
    assert provider.stops == [1]
    store.close()


def test_a_reply_stops_the_running_call_discards_its_sample_and_the_rerun_records_it(
    make_provider, tmp_path
) -> None:
    session = _Session()
    session.block_lengths = {FULL}
    provider = make_provider(session)
    store = MemoryStore(tmp_path / "memories.db")
    watch = cli_throttle.reply_watch()

    def _measure() -> bool:
        own = MemoryStore(tmp_path / "memories.db")  # a store belongs to the thread that opened it
        try:
            return reranker_mod.measure_ram_figure(provider, own, watch=watch)
        finally:
            own.close()

    worker, result = _on_thread(_measure)
    assert session.entered.wait(_TIMEOUT), "the first rung's call is running"
    cli_throttle.note_user_message()  # a reply starts
    # the reply's own rerank takes the provider's lock as soon as the call has been stopped
    reply, reply_result = _on_thread(provider.rerank_timed, "q", ["d"])
    reply.join(_TIMEOUT / 2)
    assert not reply.is_alive() and "error" not in reply_result, "the reply did not wait for the full call"
    worker.join(_TIMEOUT)
    assert isinstance(result.get("error"), RamMeasurementAbandoned)
    assert session.stopped_by_flag == 1, "stopped by the flag, not by the test's timeout"
    assert shapes_in_window(store, _MODEL) == {}
    assert reranker_mod._rerank_cost.state(store, _MODEL).sums.n == 0  # noqa: SLF001
    assert cli_throttle._start_hooks == []  # noqa: SLF001
    # the reply ends; the pass runs again and records the rung exactly once
    cli_throttle.note_reply_end()
    session.block_lengths = set()
    assert reranker_mod.measure_ram_figure(provider, store, watch=cli_throttle.reply_watch()) is True
    assert sorted(shapes_in_window(store, _MODEL)) == [(1, 128), (1, 256), (1, 512), (1, 1024)]
    assert all(n == 1 for n, _total in shapes_in_window(store, _MODEL).values())
    store.close()


def test_without_a_reply_the_same_run_records_every_rung(make_provider, tmp_path) -> None:
    session = _Session()
    provider = make_provider(session)
    store = MemoryStore(tmp_path / "memories.db")
    assert reranker_mod.measure_ram_figure(provider, store, watch=cli_throttle.reply_watch()) is True
    assert sorted(shapes_in_window(store, _MODEL)) == [(1, 128), (1, 256), (1, 512), (1, 1024)]
    assert session.stopped_by_flag == 0
    store.close()


def test_a_reply_during_another_threads_call_does_not_stop_it_and_the_ladders_call_starts_terminated(
    make_provider, tmp_path
) -> None:
    """Review M1 at the ladder level: the floor-bootstrap thread's call holds the provider while the
    ladder's rung is armed and waiting; a reply starts. The bootstrap's call is not stopped."""
    session = _Session()
    session.block_all = True
    provider = make_provider(session)
    store = MemoryStore(tmp_path / "memories.db")
    bootstrap, bootstrap_result = _on_thread(provider.rerank, "q", ["d"])
    assert session.entered.wait(_TIMEOUT)
    watch = cli_throttle.reply_watch()
    ladder, ladder_result = _on_thread(_rung, provider, store, watch)
    deadline = time.monotonic() + _TIMEOUT
    while not cli_throttle._start_hooks and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.001)  # the rung has armed and is waiting for the lock
    cli_throttle.note_user_message()
    time.sleep(0.05)
    assert bootstrap.is_alive(), "the bootstrap's call is still running: the reply did not stop it"
    session.block_all = False
    session.release.set()
    bootstrap.join(_TIMEOUT)
    ladder.join(_TIMEOUT)
    assert bootstrap_result == {"value": [0.0]}
    assert isinstance(ladder_result["error"], RamMeasurementAbandoned)
    assert session.calls[0][2] is None
    store.close()


def test_through_the_real_floor_startup_thread_a_reply_stops_the_call_and_the_rerun_after_the_reply_completes(
    monkeypatch: pytest.MonkeyPatch, make_provider, tmp_path: Path
) -> None:
    from brain.memory import floor_startup

    monkeypatch.setattr(floor_startup, "_background_inhibited", False)
    session = _Session()
    session.block_lengths = {FULL}
    provider = make_provider(session)
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert session.entered.wait(_TIMEOUT)
        cli_throttle.note_user_message()
        reply, reply_result = _on_thread(provider.rerank_timed, "q", ["d"])
        reply.join(_TIMEOUT / 2)
        assert not reply.is_alive() and "error" not in reply_result
        assert loop.entered[1].wait(_TIMEOUT), "after the abandon the thread waits for the reply to end"
        assert session.stopped_by_flag == 1
        assert shapes_in_window(loop.store, _MODEL) == {}
        session.block_lengths = set()
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert reranker_mod.ram_figure_ready(_MODEL, loop.store) is True
        assert sorted(shapes_in_window(loop.store, _MODEL)) == [(1, 128), (1, 256), (1, 512), (1, 1024)]
    finally:
        session.release.set()
        loop.finish(thread)
