"""S218 (with S220/S222): under `nell chat --no-bridge` the RAM ladder never records a sample that
overlapped a reply.

A reply that starts while a rung runs abandons that rung: nothing from it is recorded (the
process-wide peak would include the reply's allocations) and the pass runs again once that reply has
ended, resuming from the window (earlier rungs' samples stay). With the ONNX memory pool off an
abandoned call leaves no residue, so the re-run is valid and simply measures the same rung again.

Real `cli_throttle` hooks, a real `floor_startup` thread and the real `run_ram_figure` /
`measure_ram_figure`, with a scripted provider; no model is loaded. Synthetic data only."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.memory import floor_startup
from brain.memory import reranker as reranker_mod
from brain.memory.reranker import RamMeasurementAbandoned
from brain.memory.store import MemoryStore
from tests.unit.brain.memory._ram_support import FULL, ScriptedLadderProvider, shapes_in_window

_MODEL = "scripted-ladder"
_TIMEOUT = 10.0
_PHASE_ONE = [(1, 1024), (1, 512), (1, 256), (1, 128)]
# S245: a pass is complete (and the window gives a model) only with a measured batch of more than one
# pair, so the scripted pass is phase 1 plus ONE batch rung (the batch sizes are cut to [2]).
_PASS = [*_PHASE_ONE, (2, 949)]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 2.0)
    monkeypatch.setattr(reranker_mod, "_batch_sizes", lambda: [2])  # phase 1 plus one batch rung (S245)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)  # S261: ample (unreadable now means no rerank)
    # S253/S257: the first rung and the model load need the model file's size (none exists for these doubles); 1 byte.
    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", lambda model_id: 1.0)
    monkeypatch.setattr(floor_startup, "_background_inhibited", False)


class _Watch:
    """A scripted watch: clear until `trip()`."""

    def __init__(self) -> None:
        self.clear = True

    def __call__(self) -> bool:
        return self.clear

    def trip(self) -> None:
        self.clear = False


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db")


def _nothing_recorded(store: MemoryStore) -> None:
    assert shapes_in_window(store, _MODEL) == {}
    assert reranker_mod._rerank_cost.state(store, _MODEL).sums.n == 0  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_MODEL, store) is False


# ---------------------------------------------------------------------------
# The abandon points of measure_ram_figure
# ---------------------------------------------------------------------------


def test_a_watch_that_is_already_false_abandons_before_the_provider_is_asked_for_anything(tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    watch.trip()
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert (provider.probes, provider.plain, provider.timed) == (0, [], [])
    _nothing_recorded(store)


def test_a_reply_starting_during_the_pair_length_probe_abandons_before_the_warm_ups(tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    provider.on_probe = lambda n: watch.trip() if n >= 2 else None  # the probe's last call
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert provider.plain == [] and provider.timed == []
    _nothing_recorded(store)


def test_a_reply_starting_during_the_warm_ups_abandons_before_the_first_call(tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    provider.on_warm = lambda n: watch.trip() if n == 2 else None
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert len(provider.plain) == 2 and provider.timed == []
    _nothing_recorded(store)


def test_a_reply_starting_during_a_call_discards_that_rungs_sample_only(tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    provider.on_timed = lambda pairs, length: watch.trip() if length == 256 else None  # rung 3
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert provider.timed == [(1, 1024), (1, 512), (1, 256)], "this provider cannot stop its call (S218 fallback): it finishes"
    assert sorted(shapes_in_window(store, _MODEL)) == [(1, 512), (1, 1024)], "rungs 1 and 2 stay; 3 is not recorded"
    assert reranker_mod._rerank_cost.state(store, _MODEL).sums.n == 2  # noqa: SLF001


def test_the_next_pass_resumes_from_the_window_and_measures_the_abandoned_rung_again(tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    provider.on_timed = lambda pairs, length: watch.trip() if length == 256 else None
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    provider.on_timed = None
    provider.timed.clear()
    assert reranker_mod.measure_ram_figure(provider, store, watch=_Watch()) is True
    assert provider.timed == [(1, 256), (1, 128), (2, 949)], "an abandoned call leaves no residue: it is simply measured again"
    assert sorted(shapes_in_window(store, _MODEL)) == [(1, 128), (1, 256), (1, 512), (1, 1024), (2, 949)]


def test_a_reply_during_the_first_call_leaves_nothing_and_the_rerun_measures_it_again(tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    provider.on_timed = lambda pairs, length: watch.trip()
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    _nothing_recorded(store)
    provider.on_timed = None
    assert reranker_mod.measure_ram_figure(provider, store, watch=_Watch()) is True
    assert reranker_mod.ram_figure_ready(_MODEL, store) is True


@pytest.mark.parametrize("watch", [None, _Watch()], ids=["no-watch", "always-clear-watch"])
def test_an_always_clear_watch_and_no_watch_measure_exactly_as_before(tmp_path, watch) -> None:
    store, provider = _store(tmp_path), ScriptedLadderProvider(model=_MODEL)
    assert reranker_mod.measure_ram_figure(provider, store, watch=watch) is True
    assert provider.timed == _PASS


def test_every_abandon_is_logged(caplog: pytest.LogCaptureFixture, tmp_path) -> None:
    store, watch = _store(tmp_path), _Watch()
    watch.trip()
    with caplog.at_level("INFO", logger=reranker_mod.log.name):
        with pytest.raises(RamMeasurementAbandoned):
            reranker_mod.measure_ram_figure(ScriptedLadderProvider(model=_MODEL), store, watch=watch)
    assert any("abandoned" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# run_ram_figure
# ---------------------------------------------------------------------------


def test_run_ram_figure_checks_the_watch_before_it_builds_the_provider(monkeypatch, tmp_path) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_MODEL)
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: pytest.fail("a reply is in flight"))
    watch = _Watch()
    watch.trip()
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.run_ram_figure("m1", store, watch=watch)
    assert _MODEL not in reranker_mod._ram_figure_attempt, "the stamp is removed"  # noqa: SLF001
    assert reranker_mod._ram_figure_needed == {_MODEL}, "the need flag stays"  # noqa: SLF001


def test_a_reply_starting_during_the_provider_build_abandons_before_any_call(monkeypatch, tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    reranker_mod.note_ram_figure_needed(_MODEL)

    def _build(**kw):
        watch.trip()
        return provider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", _build)
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.run_ram_figure("m1", store, watch=watch)
    assert (provider.probes, provider.plain, provider.timed) == (0, [], [])
    assert _MODEL not in reranker_mod._ram_figure_attempt  # noqa: SLF001
    assert reranker_mod._ram_figure_needed == {_MODEL}  # noqa: SLF001


def test_run_ram_figure_passes_the_watch_on_and_the_stamp_survives_a_clean_run(monkeypatch, tmp_path) -> None:
    store, provider, watch = _store(tmp_path), ScriptedLadderProvider(model=_MODEL), _Watch()
    reranker_mod.note_ram_figure_needed(_MODEL)
    provider.on_timed = lambda pairs, length: watch.trip()  # tripped during a call: only a passed-on watch sees it
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: provider)
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.run_ram_figure("m1", store, watch=watch)
    clean = ScriptedLadderProvider(model=_MODEL)
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: clean)
    assert reranker_mod.run_ram_figure("m2", store, watch=_Watch()) is True
    assert reranker_mod._ram_figure_attempt[_MODEL] == "m2"  # noqa: SLF001, kept after success
    assert reranker_mod._ram_figure_needed == set()  # noqa: SLF001


# ---------------------------------------------------------------------------
# The real thread, the real cli_throttle hooks (executed interleavings)
# ---------------------------------------------------------------------------


class _Loop:
    """Drives the direct chat's background measurement with real threads. The body's wait is
    wrapped by a spy that counts entries, so the test knows the thread is blocked on it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: ScriptedLadderProvider) -> None:
        self.provider = provider
        self.waits = 0
        self.entered: list[threading.Event] = [threading.Event() for _ in range(8)]
        self.builds = 0
        self.runs = 0
        self.store = MemoryStore(tmp_path / "memories.db")
        real_wait = cli_throttle.wait_for_no_reply_in_flight

        def _spy_wait() -> None:
            self.entered[min(self.waits, 7)].set()
            self.waits += 1
            real_wait()

        monkeypatch.setattr(cli_throttle, "wait_for_no_reply_in_flight", _spy_wait)
        real_run = reranker_mod.run_ram_figure

        def _spy_run(marker, store, **kw):
            self.runs += 1
            assert kw.get("watch") is not None, "the body passes a watch to every run"
            return real_run(marker, store, **kw)

        monkeypatch.setattr(reranker_mod, "run_ram_figure", _spy_run)

        def _build(**kw):
            self.builds += 1
            return provider

        monkeypatch.setattr(reranker_mod, "build_reranker_provider", _build)
        monkeypatch.setattr(floor_startup, "_marker_provider", lambda: "m1")  # a CONSTANT marker
        floor_startup.enable_background_ram_figure(tmp_path)
        reranker_mod.note_ram_figure_needed(_MODEL)
        assert floor_startup.request_ram_figure() is True

    def start(self) -> threading.Thread:
        thread = floor_startup.start_requested_ram_figure()
        assert thread is not None
        return thread

    def finish(self, thread: threading.Thread) -> None:
        cli_throttle.reset()  # releases a waiter if a test failed half way
        thread.join(_TIMEOUT)
        assert not thread.is_alive(), "the measurement thread must end"
        assert not floor_startup.inflight("ram-figure")
        self.store.close()


def _blocking_call(provider: ScriptedLadderProvider, at_length: int):
    """The call of the rung measured at `at_length` blocks until released; returns the events."""
    started, release = threading.Event(), threading.Event()

    def _hook(pairs: int, length: int) -> None:
        if length == at_length:
            started.set()
            assert release.wait(_TIMEOUT)

    provider.on_timed = _hook
    return started, release


def test_a_reply_starting_during_a_call_abandons_that_rung_and_the_rerun_after_the_reply_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    started, release = _blocking_call(provider, 256)  # the third rung
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert started.wait(_TIMEOUT)
        cli_throttle.note_user_message()  # a reply starts while the third rung's call runs
        release.set()
        assert loop.entered[1].wait(_TIMEOUT), "after the abandon the thread waits for the reply to end"
        assert thread.is_alive() and provider.timed == _PHASE_ONE[:3], "no run while the reply is in flight"
        assert sorted(shapes_in_window(loop.store, _MODEL)) == [(1, 512), (1, 1024)], "the third sample was discarded"
        # one thread only: a second start during the wait starts nothing
        assert floor_startup.request_ram_figure() is True
        assert floor_startup.start_requested_ram_figure() is None
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert provider.timed == [*_PHASE_ONE[:3], (1, 256), (1, 128), (2, 949)], "resumed from the window, no residue"
        assert reranker_mod.ram_figure_ready(_MODEL, loop.store) is True
        assert reranker_mod._ram_figure_needed == set()  # noqa: SLF001
    finally:
        release.set()
        loop.finish(thread)


def test_a_reply_that_started_and_ended_inside_a_call_is_seen_too(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    started, release = _blocking_call(provider, 1024)  # the first rung
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert started.wait(_TIMEOUT)
        cli_throttle.note_user_message()
        cli_throttle.note_reply_end()  # the whole reply fits inside the call
        release.set()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert provider.timed == [(1, 1024), *_PASS], "the sample was discarded; the pass ran again at once"
        assert loop.runs == 2 and reranker_mod.ram_figure_ready(_MODEL, loop.store) is True
    finally:
        release.set()
        loop.finish(thread)


def test_a_reply_starting_during_the_probe_abandons_before_any_call_and_the_rerun_measures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constant marker provider: the re-run is due only because the abandon removed the
    once-per-marker stamp."""
    provider = ScriptedLadderProvider(model=_MODEL)
    fired: list[int] = []

    def _probe(n: int) -> None:
        if not fired:
            fired.append(n)
            cli_throttle.note_user_message()  # a reply starts during the probe, before any call

    provider.on_probe = _probe
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert loop.entered[1].wait(_TIMEOUT)
        assert provider.timed == [] and thread.is_alive()
        _nothing_recorded(loop.store)
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert provider.timed == _PASS
        assert reranker_mod.ram_figure_ready(_MODEL, loop.store) is True
        assert reranker_mod._ram_figure_needed == set()  # noqa: SLF001
    finally:
        loop.finish(thread)


def test_a_reply_starting_during_the_warm_ups_abandons_and_the_rerun_measures(monkeypatch, tmp_path) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    fired: list[int] = []

    def _warm(n: int) -> None:
        if n == 2 and not fired:
            fired.append(n)
            cli_throttle.note_user_message()

    provider.on_warm = _warm
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert loop.entered[1].wait(_TIMEOUT)
        assert provider.timed == [] and thread.is_alive()
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert provider.timed == _PASS
    finally:
        loop.finish(thread)


def test_the_loop_closes_the_store_it_opened_for_every_run_abandoned_or_not(monkeypatch, tmp_path) -> None:
    closed: list[str] = []
    real_close = MemoryStore.close

    def _close(self):
        closed.append(threading.current_thread().name)
        return real_close(self)

    monkeypatch.setattr(MemoryStore, "close", _close)
    provider = ScriptedLadderProvider(model=_MODEL)
    fired: list[int] = []

    def _probe(n: int) -> None:
        if not fired:
            fired.append(n)
            cli_throttle.note_user_message()

    provider.on_probe = _probe
    loop = _Loop(monkeypatch, tmp_path, provider)
    closed.clear()
    thread = loop.start()
    try:
        assert loop.entered[1].wait(_TIMEOUT)
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        own = [name for name in closed if name == "ram-figure-measure"]
        # due-check store + run store for the abandoned pass, the same for the measuring pass
        assert len(own) == 4, own
    finally:
        loop.finish(thread)


def test_a_reply_in_flight_when_the_thread_starts_delays_everything_until_it_ends(monkeypatch, tmp_path) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    loop = _Loop(monkeypatch, tmp_path, provider)
    cli_throttle.note_user_message()  # the user's next message began its reply already
    thread = loop.start()
    try:
        assert loop.entered[0].wait(_TIMEOUT)
        assert thread.is_alive()
        assert (loop.builds, provider.probes, provider.plain, provider.timed) == (0, 0, [], [])
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert loop.builds == 1 and provider.timed == _PASS
        assert reranker_mod.ram_figure_ready(_MODEL, loop.store) is True
        assert loop.waits == 1, "no abandon: one wait, one run"
    finally:
        loop.finish(thread)


def test_a_reply_starting_between_the_wait_and_the_snapshot_abandons_at_once(monkeypatch, tmp_path) -> None:
    """The wait returned (idle), then a reply starts before the snapshot: the snapshot sees it in
    flight, the run abandons before the provider is built, the loop waits again."""
    provider = ScriptedLadderProvider(model=_MODEL)
    loop = _Loop(monkeypatch, tmp_path, provider)
    real_watch = cli_throttle.reply_watch
    calls: list[int] = []

    def _late_reply_watch():
        if not calls:
            calls.append(1)
            cli_throttle.note_user_message()
        return real_watch()

    monkeypatch.setattr(cli_throttle, "reply_watch", _late_reply_watch)
    thread = loop.start()
    try:
        assert loop.entered[1].wait(_TIMEOUT)
        assert loop.builds == 0 and provider.probes == 0, "nothing built while a reply is in flight"
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert loop.builds == 1 and provider.timed == _PASS
    finally:
        loop.finish(thread)


def test_a_run_that_is_not_abandoned_ends_the_loop_after_one_pass(monkeypatch, tmp_path) -> None:
    provider = ScriptedLadderProvider(model=_MODEL, growth_fn=lambda *_a: None)  # unreadable: the first rung ends it
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert loop.waits == 1 and provider.timed == [(1, FULL)]
        _nothing_recorded(loop.store)
    finally:
        loop.finish(thread)
