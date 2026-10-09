"""S218: under `nell chat --no-bridge` the one-pair RAM measurement never overlaps a reply.

A reply that starts while the measurement runs abandons it: nothing is recorded (the
process-wide RSS reading would be inflated by the reply) and it is run again once that
reply has ended. If the pair itself already ran, the ONNX arena holds its memory, so no
clean sample can come from this process: the re-run is refused (a second pair would read 0
or page noise, and any positive value would be recorded as the figure).

Real `cli_throttle` hooks, a real `floor_startup` thread and the real
`run_ram_figure` / `measure_ram_figure`, with scripted providers; no model is loaded.
Synthetic data only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.memory import floor_startup
from brain.memory import reranker as reranker_mod
from brain.memory.reranker import RamMeasurementAbandoned, RerankerProvider
from brain.memory.store import MemoryStore
from tests.unit.brain.memory.test_monologue_last import _RERANKER_ID

_FULL = 1024
_TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod, "_current_rss_bytes", lambda: 1.0e9)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _RERANKER_ID)
    monkeypatch.setattr(floor_startup, "_background_inhibited", False)


class _Watch:
    """A scripted watch: clear until `trip()`."""

    def __init__(self) -> None:
        self.clear = True
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.clear

    def trip(self) -> None:
        self.clear = False


class _Scripted(RerankerProvider):
    """Token count = characters / 4 per pair, capped at the input limit. Every stage can
    run a hook; `rss` is a list consumed one value per timed call (the last repeats)."""

    def __init__(self, *, rss: tuple[float, ...] = (245e6,), seconds: float = 2.0) -> None:
        self.rss_values = list(rss)
        self.seconds = seconds
        self.probes = 0
        self.plain: list[tuple[str, list[str]]] = []
        self.timed: list[tuple[str, list[str]]] = []
        self.on_probe: Callable[[int], None] | None = None
        self.on_warm: Callable[[int], None] | None = None
        self.on_timed: Callable[[], None] | None = None

    def rerank(self, query, documents):
        self.plain.append((query, list(documents)))
        if self.on_warm is not None:
            self.on_warm(len(self.plain))
        return [0.0] * len(documents)

    def rerank_timed(self, query, documents):
        self.timed.append((query, list(documents)))
        if self.on_timed is not None:
            self.on_timed()
        index = min(len(self.timed), len(self.rss_values)) - 1
        return [0.0] * len(documents), self.seconds, self.rss_values[index]

    def pair_token_lengths(self, query, documents):
        self.probes += 1
        if self.on_probe is not None:
            self.on_probe(self.probes)
        return [min(_FULL, (len(query) + len(d)) // 4) for d in documents]

    def model_id(self) -> str:
        return _RERANKER_ID


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db")


def _nothing_recorded(store: MemoryStore) -> None:
    assert reranker_mod._ram_cost.state(store, _RERANKER_ID).sums.n == 0  # noqa: SLF001
    assert reranker_mod._rerank_cost.state(store, _RERANKER_ID).sums.n == 0  # noqa: SLF001
    assert _RERANKER_ID not in reranker_mod._ram_state  # noqa: SLF001
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


# ---------------------------------------------------------------------------
# G3: the four abandon points of measure_ram_figure
# ---------------------------------------------------------------------------


def test_a_watch_that_is_already_false_abandons_before_the_provider_is_asked_for_anything(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider, watch = _Scripted(), _Watch()
    watch.trip()
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert (provider.probes, provider.plain, provider.timed) == (0, [], [])
    _nothing_recorded(store)
    assert _RERANKER_ID not in reranker_mod._pair_ran_unrecorded  # noqa: SLF001


def test_a_reply_starting_during_the_pair_length_probe_abandons_before_the_warm_ups(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider, watch = _Scripted(), _Watch()
    provider.on_probe = lambda n: watch.trip() if n >= 2 else None  # the probe's last call
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert provider.probes >= 2 and provider.plain == [] and provider.timed == []
    _nothing_recorded(store)
    assert _RERANKER_ID not in reranker_mod._pair_ran_unrecorded  # noqa: SLF001


def test_a_reply_starting_during_the_warm_ups_abandons_before_the_pair(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider, watch = _Scripted(), _Watch()
    provider.on_warm = lambda n: watch.trip() if n == 2 else None  # the second warm-up
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert len(provider.plain) == 2 and provider.timed == []
    _nothing_recorded(store)
    assert _RERANKER_ID not in reranker_mod._pair_ran_unrecorded  # noqa: SLF001


def test_a_reply_starting_during_the_pair_discards_its_sample_and_marks_the_pair_as_run(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider, watch = _Scripted(), _Watch()
    provider.on_timed = watch.trip  # the reply starts while the pair runs
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(provider, store, watch=watch)
    assert len(provider.timed) == 1, "the pair ran (it cannot be interrupted)"
    _nothing_recorded(store)
    assert _RERANKER_ID in reranker_mod._pair_ran_unrecorded  # noqa: SLF001


@pytest.mark.parametrize("watch", [None, _Watch()], ids=["no-watch", "always-clear-watch"])
def test_an_always_clear_watch_and_no_watch_measure_exactly_as_before(tmp_path: Path, watch) -> None:
    store = _store(tmp_path)
    provider = _Scripted()
    assert reranker_mod.measure_ram_figure(provider, store, watch=watch) is True
    assert provider.plain == [("warm up", ["warm up"])] * 2 and len(provider.timed) == 1
    ram = reranker_mod._ram_cost.state(store, _RERANKER_ID).sums  # noqa: SLF001
    assert (ram.n, ram.sum_x, ram.sum_y) == (1, _FULL, 245e6)
    assert _RERANKER_ID not in reranker_mod._pair_ran_unrecorded  # noqa: SLF001


# ---------------------------------------------------------------------------
# G3b: a pair that ran without a sample is never run a second time
# ---------------------------------------------------------------------------


def test_after_an_abandon_past_the_pair_a_later_measurement_is_refused_and_asks_for_nothing(
    tmp_path: Path,
) -> None:
    """A second pair on the grown arena would read 0 or page noise; the scripted second
    pair below would read a few KB of "growth", which would be recorded as the figure."""
    store = _store(tmp_path)
    first, watch = _Scripted(), _Watch()
    first.on_timed = watch.trip
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.measure_ram_figure(first, store, watch=watch)

    noisy = _Scripted(rss=(5_000.0,))
    assert reranker_mod._ram_figure_would_refuse(_RERANKER_ID) is True  # noqa: SLF001
    assert reranker_mod.measure_ram_figure(noisy, store) is False
    assert reranker_mod.measure_ram_figure(noisy, store, watch=_Watch()) is False
    assert (noisy.probes, noisy.plain, noisy.timed) == (0, [], [])
    _nothing_recorded(store)
    assert reranker_mod.ram_figure_ready(_RERANKER_ID, store) is False


def test_the_run_function_refuses_without_building_the_provider_after_a_pair_ran_unrecorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod._pair_ran_unrecorded.add(_RERANKER_ID)  # noqa: SLF001
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    monkeypatch.setattr(
        reranker_mod, "build_reranker_provider", lambda **kw: pytest.fail("refused: nothing is built")
    )
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert reranker_mod._ram_figure_attempt[_RERANKER_ID] == "m1"  # noqa: SLF001
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}  # noqa: SLF001, the flag stays


def test_the_test_reset_forgets_the_pair_ran_mark() -> None:
    reranker_mod._pair_ran_unrecorded.add(_RERANKER_ID)  # noqa: SLF001
    reranker_mod._reset_rerank_cost_model()  # noqa: SLF001
    assert reranker_mod._ram_figure_would_refuse(_RERANKER_ID) is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# G4: run_ram_figure
# ---------------------------------------------------------------------------


def test_run_ram_figure_checks_the_watch_before_it_builds_the_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    monkeypatch.setattr(
        reranker_mod, "build_reranker_provider", lambda **kw: pytest.fail("a reply is in flight")
    )
    watch = _Watch()
    watch.trip()
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.run_ram_figure("m1", store, watch=watch)
    assert _RERANKER_ID not in reranker_mod._ram_figure_attempt, "the stamp is removed"  # noqa: SLF001
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}, "the need flag stays"  # noqa: SLF001


def test_a_reply_starting_during_the_provider_build_abandons_before_any_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    provider, watch = _Scripted(), _Watch()

    def _build(**kw):
        watch.trip()
        return provider

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", _build)
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.run_ram_figure("m1", store, watch=watch)
    assert (provider.probes, provider.plain, provider.timed) == (0, [], [])
    assert _RERANKER_ID not in reranker_mod._ram_figure_attempt  # noqa: SLF001
    assert reranker_mod._ram_figure_needed == {_RERANKER_ID}  # noqa: SLF001


def test_run_ram_figure_passes_the_watch_on_and_the_stamp_survives_a_clean_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    provider, watch = _Scripted(), _Watch()
    provider.on_timed = watch.trip  # tripped during the pair: only a passed-on watch sees it
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: provider)
    with pytest.raises(RamMeasurementAbandoned):
        reranker_mod.run_ram_figure("m1", store, watch=watch)
    reranker_mod._reset_rerank_cost_model()  # noqa: SLF001
    reranker_mod.note_ram_figure_needed(_RERANKER_ID)
    clean = _Scripted()
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: clean)
    assert reranker_mod.run_ram_figure("m2", store, watch=_Watch()) is True
    assert reranker_mod._ram_figure_attempt[_RERANKER_ID] == "m2"  # noqa: SLF001, kept after success


# ---------------------------------------------------------------------------
# G5 / G6: the real thread, the real cli_throttle hooks (executed interleavings)
# ---------------------------------------------------------------------------


class _Loop:
    """Drives the direct chat's background measurement with real threads. The body's wait
    is wrapped by a spy that counts entries, so the test knows the thread is blocked on it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: _Scripted) -> None:
        self.provider = provider
        self.waits = 0
        self.entered: list[threading.Event] = [threading.Event() for _ in range(8)]
        self.builds = 0
        self.watches: list[Callable[[], bool] | None] = []
        self.store = MemoryStore(tmp_path / "memories.db")
        real_wait = cli_throttle.wait_for_no_reply_in_flight

        def _spy_wait() -> None:
            self.entered[min(self.waits, 7)].set()
            self.waits += 1
            real_wait()

        monkeypatch.setattr(cli_throttle, "wait_for_no_reply_in_flight", _spy_wait)
        real_run = reranker_mod.run_ram_figure

        def _spy_run(marker, store, **kw):
            self.watches.append(kw.get("watch"))
            return real_run(marker, store, **kw)

        monkeypatch.setattr(reranker_mod, "run_ram_figure", _spy_run)

        def _build(**kw):
            self.builds += 1
            return provider

        monkeypatch.setattr(reranker_mod, "build_reranker_provider", _build)
        monkeypatch.setattr(floor_startup, "_marker_provider", lambda: "m1")  # a CONSTANT marker
        floor_startup.enable_background_ram_figure(tmp_path)
        reranker_mod.note_ram_figure_needed(_RERANKER_ID)
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


def test_a_reply_starting_during_the_pair_abandons_it_and_the_rerun_after_the_reply_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = threading.Event()
    provider = _Scripted(rss=(245e6, 5_000.0))
    pair_started = threading.Event()

    def _pair() -> None:
        pair_started.set()
        assert release.wait(_TIMEOUT)

    provider.on_timed = _pair
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert pair_started.wait(_TIMEOUT)
        cli_throttle.note_user_message()  # a reply starts while the pair runs
        release.set()
        assert loop.entered[1].wait(_TIMEOUT), "after the abandon the thread waits for the reply to end"
        assert thread.is_alive() and len(provider.timed) == 1, "no second run while the reply is in flight"
        _nothing_recorded(loop.store)
        # one thread only: a second start during the wait starts nothing
        assert floor_startup.request_ram_figure() is True
        assert floor_startup.start_requested_ram_figure() is None
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert len(provider.timed) == 1, "the re-run was refused: no second pair on the grown arena"
        _nothing_recorded(loop.store)
        assert reranker_mod._ram_figure_needed == {_RERANKER_ID}  # noqa: SLF001
        assert all(w is not None for w in loop.watches), "the body passes a watch to every run"
    finally:
        release.set()
        loop.finish(thread)


def test_a_reply_that_started_and_ended_before_the_pair_returned_is_seen_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = threading.Event()
    provider = _Scripted(rss=(245e6, 5_000.0))
    pair_started = threading.Event()

    def _pair() -> None:
        pair_started.set()
        assert release.wait(_TIMEOUT)

    provider.on_timed = _pair
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert pair_started.wait(_TIMEOUT)
        cli_throttle.note_user_message()
        cli_throttle.note_reply_end()  # the whole reply fits inside the pair
        release.set()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert len(provider.timed) == 1, "the sample was discarded and the re-run refused"
        _nothing_recorded(loop.store)
        assert len(loop.watches) == 2, "run, abandoned, run again (refused)"
    finally:
        release.set()
        loop.finish(thread)


def test_a_reply_starting_before_the_pair_abandons_it_and_the_rerun_after_the_reply_measures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constant marker provider: the re-run is due only because the abandon removed the
    once-per-marker stamp."""
    provider = _Scripted()
    fired: list[int] = []

    def _probe(n: int) -> None:
        if not fired:
            fired.append(n)
            cli_throttle.note_user_message()  # a reply starts during the probe, before the pair

    provider.on_probe = _probe
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        assert loop.entered[1].wait(_TIMEOUT)
        assert provider.timed == [] and thread.is_alive()
        _nothing_recorded(loop.store)
        assert _RERANKER_ID not in reranker_mod._pair_ran_unrecorded  # noqa: SLF001
        cli_throttle.note_reply_end()
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert len(provider.timed) == 1, "the re-run ran the pair"
        assert reranker_mod.ram_figure_ready(_RERANKER_ID, loop.store) is True
        ram = reranker_mod._ram_cost.state(loop.store, _RERANKER_ID).sums  # noqa: SLF001
        assert (ram.n, ram.sum_y) == (1, 245e6)
        assert reranker_mod._ram_figure_needed == set()  # noqa: SLF001
    finally:
        loop.finish(thread)


def test_a_reply_in_flight_when_the_thread_starts_delays_everything_until_it_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _Scripted()
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
        assert loop.builds == 1 and len(provider.timed) == 1
        assert reranker_mod.ram_figure_ready(_RERANKER_ID, loop.store) is True
        assert loop.waits == 1, "no abandon: one wait, one run"
    finally:
        loop.finish(thread)


def test_a_reply_starting_between_the_wait_and_the_snapshot_abandons_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wait returned (idle), then a reply starts before the snapshot: the snapshot sees it in
    flight, the run abandons before the provider is built, the loop waits again."""
    provider = _Scripted()
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
        assert loop.builds == 1 and len(provider.timed) == 1
        assert reranker_mod.ram_figure_ready(_RERANKER_ID, loop.store) is True
    finally:
        loop.finish(thread)


def test_a_run_that_is_not_abandoned_ends_the_loop_after_one_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _Scripted(rss=(0.0,))  # a pair with no growth: not a figure, not abandoned
    loop = _Loop(monkeypatch, tmp_path, provider)
    thread = loop.start()
    try:
        thread.join(_TIMEOUT)
        assert not thread.is_alive()
        assert loop.waits == 1 and len(provider.timed) == 1
        _nothing_recorded(loop.store)
        assert _RERANKER_ID not in reranker_mod._pair_ran_unrecorded  # noqa: SLF001
    finally:
        loop.finish(thread)


def test_the_body_waits_for_and_uses_the_cli_throttle_signal() -> None:
    """Source-level wiring pin (the executed interleavings above carry the behaviour)."""
    import inspect

    source = inspect.getsource(floor_startup._ram_figure_body)  # noqa: SLF001
    assert "cli_throttle.wait_for_no_reply_in_flight()" in source
    assert "cli_throttle.reply_watch()" in source
    assert "watch=watch" in source
    assert "RamMeasurementAbandoned" in source
