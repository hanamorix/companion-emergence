"""Tests for issue #132 — supervisor per-tick DB overhead + two small tick fixes.

Part A: sweep/maker/notes share ONE MemoryStore/HebbianMatrix connection per tick
(instead of 3 memories.db opens), integrity_check=False at the per-tick sites.
Part B: session-silence threshold bumped 5 -> 10 minutes (behavioral gate test lives
in tests/unit/brain/ingest/test_pipeline.py; the default/wiring checks live here).
Part C: the 6h forgetting/narrative maintenance pass is throttled behind
cli_throttle.background_slot(), matching interest-sweep's existing idiom.

All Part-A tests drive run_folded() SYNCHRONOUSLY, single-threaded, no polling or
event-timing — see _drive_ticks below. Termination is bounded deterministically by
hooking the sweep's MemoryStore construction (reached every iteration, success or
failure), not by any later call in the tick body or by a wall-clock timeout.
"""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import brain.bridge.server as server
import brain.bridge.supervisor as supervisor
import brain.health.self_model_repair
import brain.ingest.pipeline as pipeline
import brain.memory.hebbian
import brain.memory.store
from brain.bridge.provider import FakeProvider
from tests.unit.brain.bridge.test_supervisor import _CapturingBus, _persona_dir

# ---------------------------------------------------------------------------
# Part A — the synchronous, bounded-termination test primitive
# ---------------------------------------------------------------------------


class _TickCounter:
    """Shared, externally-readable construction ordinal."""

    def __init__(self) -> None:
        self.n = 0


class _DriveResult:
    def __init__(self, bus, constructions, closes, maker_calls, notes_calls) -> None:
        self.bus = bus
        self.constructions = constructions
        self.closes = closes
        self.maker_calls = maker_calls
        self.notes_calls = notes_calls


def _isolating_kwargs(**overrides):
    """The Part-A isolating configuration: every cadence other than sweep/maker/notes
    disabled, so only the three targeted sites can construct a MemoryStore."""
    kwargs = {
        "tick_interval_s": 0.05,
        "maker_enabled": True,
        "notes_enabled": True,
        "heartbeat_interval_s": None,
        "soul_review_interval_s": None,
        "finalize_interval_s": None,
        "log_rotation_interval_s": None,
        "initiate_review_interval_s": None,
        "voice_reflection_interval_s": None,
        "self_model_interval_s": None,
        "compaction_interval_s": None,
        "interest_sweep_interval_s": None,
        "kindled_link_enabled": False,
    }
    kwargs.update(overrides)
    return kwargs


def _drive_ticks(
    persona_dir: Path,
    *,
    num_ticks: int = 1,
    bus=None,
    memstore_raises_on_tick=None,
    on_notes_call=None,
    **extra,
):
    """Run run_folded synchronously, single-threaded, terminating deterministically
    after exactly num_ticks constructions of the sweep's MemoryStore — whether that
    construction succeeds or is made to raise. Hooking the SWEEP's construction
    (reached every iteration, success or failure) is what makes termination
    unconditional; tick_interval_s=0.05 keeps the one intentional stop_event.wait()
    between ticks fast (not a correctness dependency — termination is by the
    counter, not the timer).
    """
    stop_event = threading.Event()
    bus = bus or _CapturingBus()
    counter = _TickCounter()
    constructions: list[dict] = []
    closes: list[dict] = []
    maker_calls: list[dict] = []
    notes_calls: list[dict] = []

    real_memory_store = brain.memory.store.MemoryStore
    real_hebbian_matrix = brain.memory.hebbian.HebbianMatrix
    real_maybe_maker = supervisor._maybe_run_maker_tick
    real_maybe_notes = supervisor._maybe_run_notes_tick

    class _CountingStore(real_memory_store):
        def close(self):
            closes.append({"tick": counter.n, "obj": self})
            super().close()

    def _make_memory_store(*args, **kwargs):
        counter.n += 1
        this_tick = counter.n
        if this_tick >= num_ticks:
            stop_event.set()
        if memstore_raises_on_tick and this_tick in memstore_raises_on_tick:
            raise RuntimeError(f"simulated store-open failure on tick {this_tick}")
        obj = _CountingStore(*args, **kwargs)
        constructions.append({"kind": "MemoryStore", "tick": this_tick, "kwargs": kwargs, "obj": obj})
        return obj

    def _make_hebbian(*args, **kwargs):
        obj = real_hebbian_matrix(*args, **kwargs)
        constructions.append({"kind": "HebbianMatrix", "tick": counter.n, "kwargs": kwargs, "obj": obj})
        return obj

    def _spy_maker(*a, **kw):
        maker_calls.append({"tick": counter.n, "store": kw.get("store")})
        return real_maybe_maker(*a, **kw)

    def _spy_notes(*a, **kw):
        notes_calls.append({"tick": counter.n, "store": kw.get("store")})
        if on_notes_call is not None:
            on_notes_call(counter.n, kw.get("store"))
        return real_maybe_notes(*a, **kw)

    with (
        patch.object(supervisor, "MemoryStore", side_effect=_make_memory_store),
        patch("brain.memory.store.MemoryStore", side_effect=_make_memory_store),
        patch.object(supervisor, "HebbianMatrix", side_effect=_make_hebbian),
        patch.object(supervisor, "_maybe_run_maker_tick", side_effect=_spy_maker),
        patch.object(supervisor, "_maybe_run_notes_tick", side_effect=_spy_notes),
        patch.object(supervisor, "_attunement_should_run_backfill", return_value=False),
        patch.object(supervisor, "_attunement_should_run_supplementary_backfill", return_value=False),
        patch.object(supervisor, "_emotion_backfill_should_run", return_value=False),
        patch.object(supervisor, "_vocab_repair_should_run", return_value=False),
        patch.object(supervisor, "_soul_candidate_repair_should_run", return_value=False),
        patch(
            "brain.health.self_model_repair.should_run_self_model_repair",
            return_value=False,
        ),
    ):
        kwargs = _isolating_kwargs(**extra)
        supervisor.run_folded(
            stop_event,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=bus,
            **kwargs,
        )
    return _DriveResult(bus, constructions, closes, maker_calls, notes_calls)


def _mem_constructions(result: _DriveResult) -> list[dict]:
    return [c for c in result.constructions if c["kind"] == "MemoryStore"]


def _hebbian_constructions(result: _DriveResult) -> list[dict]:
    return [c for c in result.constructions if c["kind"] == "HebbianMatrix"]


# ---------------------------------------------------------------------------
# C1 / C2 / C3 — one connection per tick, no per-tick integrity_check
# ---------------------------------------------------------------------------


def test_run_folded_opens_memories_db_once_per_tick_sweep_maker_notes(tmp_path):
    """C1: sweep/maker/notes share one memories.db connection per tick."""
    persona_dir = _persona_dir(tmp_path)
    result = _drive_ticks(persona_dir, num_ticks=1)
    assert len(_mem_constructions(result)) == 1


def test_run_folded_opens_hebbian_db_once_per_tick(tmp_path):
    """C2 (regression guard): hebbian.db opened at most once per tick."""
    persona_dir = _persona_dir(tmp_path)
    result = _drive_ticks(persona_dir, num_ticks=1)
    assert len(_hebbian_constructions(result)) == 1


def test_run_folded_per_tick_opens_skip_integrity_check(tmp_path):
    """C3: the per-tick memories.db/hebbian.db opens pass integrity_check=False."""
    persona_dir = _persona_dir(tmp_path)
    result = _drive_ticks(persona_dir, num_ticks=1)
    mem = _mem_constructions(result)
    heb = _hebbian_constructions(result)
    assert len(mem) == 1 and mem[0]["kwargs"].get("integrity_check") is False
    assert len(heb) == 1 and heb[0]["kwargs"].get("integrity_check") is False


def test_maker_and_notes_reuse_the_sweeps_live_store_object(tmp_path):
    """C1/C3 strengthened (round-5 MAJOR-2/MAJOR-4): maker and notes are called with
    the EXACT store object the sweep constructed, not merely *a* connection — this
    directly rules out a mis-build where the shared store is closed too early
    (maker/notes would then hold a dead connection) while still passing a naive
    open-count assertion.
    """
    persona_dir = _persona_dir(tmp_path)
    result = _drive_ticks(persona_dir, num_ticks=1)
    mem = _mem_constructions(result)
    assert len(mem) == 1
    shared_obj = mem[0]["obj"]
    assert len(result.maker_calls) == 1
    assert len(result.notes_calls) == 1
    assert result.maker_calls[0]["store"] is shared_obj
    assert result.notes_calls[0]["store"] is shared_obj
    # And it must still be usable (not closed) at the point maker/notes received it —
    # by the time _drive_ticks returns the tick has finished and store IS closed, but
    # confirm close() was only called once, after both consumers ran.
    assert len(result.closes) == 1
    assert result.closes[0]["obj"] is shared_obj


def test_startup_repair_sites_unchanged(tmp_path):
    """C4 (regression guard): the two one-shot startup MemoryStore opens (vocab-repair,
    soul-candidate-repair) still pass integrity_check=False, unaffected by this change.
    Anchored on the `while not stop_event.is_set():` structural marker, not a line
    number, so this survives surrounding edits — and runs entirely against the
    working tree (no external git ref needed, unlike an origin/main diff, which
    cannot run in this repo's shallow-checkout CI).
    """
    source = Path(supervisor.__file__).read_text(encoding="utf-8")
    anchor = "while not stop_event.is_set():"
    assert anchor in source
    prefix = source.split(anchor, 1)[0]
    assert prefix.count("integrity_check=False") == 2
    assert "_vocab_repair_should_run(persona_dir)" in prefix
    assert "_soul_candidate_repair_should_run(persona_dir)" in prefix
    assert 'MemoryStore(str(db_path), integrity_check=False)' in prefix


# ---------------------------------------------------------------------------
# C8 — fail-soft: a shared-store open failure skips maker/notes for that tick only
# ---------------------------------------------------------------------------


def test_maker_notes_skip_cleanly_when_shared_store_open_fails(tmp_path):
    persona_dir = _persona_dir(tmp_path)
    result = _drive_ticks(persona_dir, num_ticks=2, memstore_raises_on_tick={1})
    assert not any(c["tick"] == 1 for c in result.maker_calls)
    assert not any(c["tick"] == 1 for c in result.notes_calls)
    assert [c["tick"] for c in result.maker_calls] == [2]
    assert [c["tick"] for c in result.notes_calls] == [2]
    mem = _mem_constructions(result)
    tick2_store = next(c["obj"] for c in mem if c["tick"] == 2)
    assert result.maker_calls[0]["store"] is tick2_store
    assert result.notes_calls[0]["store"] is tick2_store


# ---------------------------------------------------------------------------
# C11 — shared-connection visibility is unchanged
# ---------------------------------------------------------------------------


def test_shared_store_sees_commits_from_independent_connection(tmp_path):
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona_dir(tmp_path)
    shared = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    try:
        other = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        try:
            memory = Memory.create_new(content="hello", memory_type="meta", domain="test")
            other.create(memory)
        finally:
            other.close()
        fetched = shared.get(memory.id, bump=False)
        assert fetched is not None
        assert fetched.id == memory.id
    finally:
        shared.close()


# ---------------------------------------------------------------------------
# C12 — no connection leak across ticks, including on an uncaught mid-tick exception
# ---------------------------------------------------------------------------


def test_store_construct_close_count_matches_across_ticks(tmp_path):
    """C12 test 1 (regression guard, explicitly labeled): true both pre- and post-fix."""
    persona_dir = _persona_dir(tmp_path)
    result = _drive_ticks(persona_dir, num_ticks=3)
    mem = _mem_constructions(result)
    assert len(mem) == len(result.closes) == 3
    constructed_objs = {c["obj"] for c in mem}
    closed_objs = {c["obj"] for c in result.closes}
    assert constructed_objs == closed_objs


def test_store_closed_before_uncaught_exception_propagates(tmp_path):
    """C12 test 2 (claim-bearing): the shared store is closed before an exception
    raised elsewhere in the tick body (soul_cadence.save_cadence_state, genuinely
    unguarded pre-fix and post-fix) propagates out of run_folded.
    """
    persona_dir = _persona_dir(tmp_path)
    stop_event = threading.Event()
    closes: list[int] = []
    real_memory_store = brain.memory.store.MemoryStore

    class _CloseSpyingStore(real_memory_store):
        def close(self):
            closes.append(1)
            super().close()

    with (
        patch.object(
            supervisor, "MemoryStore", side_effect=lambda *a, **kw: _CloseSpyingStore(*a, **kw)
        ),
        patch.object(
            supervisor.soul_cadence, "save_cadence_state", side_effect=RuntimeError("boom")
        ),
        patch.object(supervisor, "_attunement_should_run_backfill", return_value=False),
        patch.object(supervisor, "_attunement_should_run_supplementary_backfill", return_value=False),
        patch.object(supervisor, "_emotion_backfill_should_run", return_value=False),
        patch.object(supervisor, "_vocab_repair_should_run", return_value=False),
        patch.object(supervisor, "_soul_candidate_repair_should_run", return_value=False),
        patch(
            "brain.health.self_model_repair.should_run_self_model_repair",
            return_value=False,
        ),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            supervisor.run_folded(
                stop_event,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=_CapturingBus(),
                tick_interval_s=0.05,
                maker_enabled=True,
                notes_enabled=True,
                heartbeat_interval_s=None,
                soul_review_interval_s=0.0,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
                voice_reflection_interval_s=None,
                self_model_interval_s=None,
                compaction_interval_s=None,
                interest_sweep_interval_s=None,
                kindled_link_enabled=False,
            )
    assert closes == [1]


# ---------------------------------------------------------------------------
# C16 — the widened connection lifetime coexists with a real second connection
# ---------------------------------------------------------------------------


def test_shared_store_coexists_with_finalize_on_same_tick(tmp_path):
    """C16: finalize fires on the same tick (its own cadence state also treats a
    missing next_at as due), opening its OWN independent MemoryStore to the same
    file WHILE the shared store from the sweep is still held open — nothing
    deadlocks or raises, and the shared connection survives the encounter usable.

    Corrected (stage-6 round 1, MAJOR-2): the shared store's lifetime does overlap
    finalize's — both are open simultaneously during _run_finalize_tick's body —
    but `_run_finalize_tick` closes its OWN connection (via its own ExitStack)
    before returning, well before the notes block where the liveness probe below
    fires. So the probe proves the shared connection is still healthy AFTER a
    second connection to the same file was opened, used, and closed during this
    tick — not that the two were simultaneously queried at the same instant. The
    substantive "did the two connections coexist without a lock conflict" claim
    is what `len(mem) == 2` plus `run_folded` returning without raising already
    proves; the probe adds "and the shared connection wasn't left in a broken
    state by that encounter."
    """
    persona_dir = _persona_dir(tmp_path)
    live_probe_results = []

    def _probe(tick, store_obj):
        # Called from the notes block, after finalize's own tick (and its own,
        # already-closed connection to the same file) has run earlier in this
        # same iteration — confirms the shared connection is still usable.
        if store_obj is not None:
            row = store_obj._conn.execute("SELECT 1").fetchone()  # noqa: SLF001
            live_probe_results.append(tuple(row))

    result = _drive_ticks(
        persona_dir, num_ticks=1, finalize_interval_s=0.0, on_notes_call=_probe
    )
    mem = _mem_constructions(result)
    # Two independent MemoryStore connections open on this tick: the sweep's shared
    # one (reused by maker/notes) and finalize's own, separate one — exactly the
    # "widened lifetime overlapping a real second connection" scenario this
    # criterion targets. The sweep's is always constructed first each iteration.
    # run_folded returning normally (no exception) is itself part of the proof
    # that the two same-thread connections coexisted without a lock conflict.
    assert len(mem) == 2
    shared_obj = mem[0]["obj"]
    assert len(result.maker_calls) == 1
    assert result.maker_calls[0]["store"] is shared_obj
    assert len(result.notes_calls) == 1
    assert result.notes_calls[0]["store"] is shared_obj
    # The shared connection is still live and queryable after finalize's
    # independent connection to the same file was opened and closed earlier in
    # this same tick.
    assert live_probe_results == [(1,)]


# ---------------------------------------------------------------------------
# Part B — session-silence threshold defaults + wiring
# ---------------------------------------------------------------------------


def test_silence_minutes_defaults_are_ten_minutes():
    """C14 test 1 (claim-bearing)."""
    assert inspect.signature(supervisor.run_folded).parameters["silence_minutes"].default == 10.0
    assert inspect.signature(server.build_app).parameters["silence_minutes"].default == 10.0
    assert (
        inspect.signature(pipeline.snapshot_stale_sessions).parameters["silence_minutes"].default
        == 10.0
    )


def test_build_app_default_reaches_run_folded():
    """C14 test 3: build_app's own silence_minutes PARAMETER is what actually gets
    threaded into run_folded's kwargs dict (not a separately-hardcoded value) —
    proves the composed production path, not just three independently-agreeing
    signatures.

    Corrected (stage-6 round 1, BLOCKER-1): the original version walked the WHOLE
    module for any ast.keyword named silence_minutes, which spuriously matched
    _drain_sessions_blocking's unrelated keyword argument at server.py — a
    different function this change explicitly does NOT touch — and could never
    have matched the real target, the sp7-supervisor kwargs={...} DICT LITERAL
    inside build_app, which is an ast.Dict, not an ast.keyword. That made the
    test vacuous: it would pass even if the dict literal were hardcoded to 5.0.
    This version scopes the walk to build_app's own FunctionDef body and looks
    for the actual dict entry.
    """
    import ast

    source = Path(server.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    build_app_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_app"
    )

    found = False
    for node in ast.walk(build_app_node):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "silence_minutes"
                and isinstance(value, ast.Name)
                and value.id == "silence_minutes"
            ):
                found = True
                break
        if found:
            break

    assert found, (
        "build_app's own sp7-supervisor kwargs dict must forward its own "
        "silence_minutes parameter, not a separately hardcoded value"
    )


# ---------------------------------------------------------------------------
# Part C — maintenance pass throttled behind cli_throttle.background_slot()
# ---------------------------------------------------------------------------


def _drive_one_maintenance_tick(persona_dir: Path, *, throttle_grants: bool):
    stop_event = threading.Event()
    real_memory_store = brain.memory.store.MemoryStore

    def _stop_after_sweep(*a, **kw):
        stop_event.set()
        return real_memory_store(*a, **kw)

    # A shared call-order list, not just two independent MagicMocks — two
    # independent mocks cannot observe RELATIVE order between them, which C15's
    # "forgetting before narrative, unchanged from pre-fix" claim requires
    # (stage-6 round 1, MAJOR-3: the original version only asserted each was
    # called once, which cannot catch an order regression).
    call_order: list[str] = []
    forgetting_spy = MagicMock(side_effect=lambda *a, **kw: call_order.append("forgetting"))
    narrative_spy = MagicMock(side_effect=lambda *a, **kw: call_order.append("narrative"))
    pending_spy = MagicMock()
    sidecar_spy = MagicMock()

    @contextmanager
    def _fake_slot(*, now=None):
        yield throttle_grants

    with (
        patch.object(supervisor, "MemoryStore", side_effect=_stop_after_sweep),
        patch("brain.memory.store.MemoryStore", side_effect=_stop_after_sweep),
        patch.object(supervisor, "HebbianMatrix"),
        patch.object(supervisor.cli_throttle, "background_slot", side_effect=_fake_slot),
        patch.object(supervisor, "forgetting_run_pass", forgetting_spy),
        patch.object(supervisor, "_run_narrative_memory_pass", narrative_spy),
        patch("brain.files.pending.sweep_expired", pending_spy),
        patch("brain.health.sidecar_sweep.sweep_stale_sidecars", sidecar_spy),
        patch.object(supervisor, "_attunement_should_run_backfill", return_value=False),
        patch.object(supervisor, "_attunement_should_run_supplementary_backfill", return_value=False),
        patch.object(supervisor, "_emotion_backfill_should_run", return_value=False),
        patch.object(supervisor, "_vocab_repair_should_run", return_value=False),
        patch.object(supervisor, "_soul_candidate_repair_should_run", return_value=False),
        patch(
            "brain.health.self_model_repair.should_run_self_model_repair",
            return_value=False,
        ),
    ):
        supervisor.run_folded(
            stop_event,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=_CapturingBus(),
            tick_interval_s=0.05,
            maker_enabled=False,
            notes_enabled=False,
            heartbeat_interval_s=None,
            soul_review_interval_s=0.0,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
            self_model_interval_s=None,
            compaction_interval_s=None,
            interest_sweep_interval_s=None,
            kindled_link_enabled=False,
        )
    return forgetting_spy, narrative_spy, pending_spy, sidecar_spy, call_order


def test_maintenance_pass_defers_expensive_work_when_throttle_denies(tmp_path):
    persona_dir = _persona_dir(tmp_path)
    forgetting_spy, narrative_spy, pending_spy, sidecar_spy, _order = _drive_one_maintenance_tick(
        persona_dir, throttle_grants=False
    )
    forgetting_spy.assert_not_called()
    narrative_spy.assert_not_called()
    pending_spy.assert_called_once()
    sidecar_spy.assert_called_once()
    cadence_state = supervisor.persisted_cadence.load_cadence(
        persona_dir, "maintenance_cadence.json"
    )
    assert cadence_state.next_at is not None


def test_maintenance_pass_runs_expensive_work_when_throttle_grants(tmp_path):
    persona_dir = _persona_dir(tmp_path)
    forgetting_spy, narrative_spy, pending_spy, sidecar_spy, order = _drive_one_maintenance_tick(
        persona_dir, throttle_grants=True
    )
    forgetting_spy.assert_called_once()
    narrative_spy.assert_called_once()
    pending_spy.assert_called_once()
    sidecar_spy.assert_called_once()
    # C15's stated claim: forgetting before narrative, unchanged from pre-fix.
    assert order == ["forgetting", "narrative"]


def test_forgetting_failure_does_not_skip_narrative_pass(tmp_path):
    """The two try/except blocks under the throttle stay independent — a forgetting
    exception must not also skip narrative for that cycle (round-5 finding: an
    earlier draft merged them into one handler, which this test would have caught).
    """
    persona_dir = _persona_dir(tmp_path)
    stop_event = threading.Event()
    real_memory_store = brain.memory.store.MemoryStore

    def _stop_after_sweep(*a, **kw):
        stop_event.set()
        return real_memory_store(*a, **kw)

    narrative_spy = MagicMock()

    @contextmanager
    def _granting_slot(*, now=None):
        yield True

    with (
        patch.object(supervisor, "MemoryStore", side_effect=_stop_after_sweep),
        patch("brain.memory.store.MemoryStore", side_effect=_stop_after_sweep),
        patch.object(supervisor, "HebbianMatrix"),
        patch.object(supervisor.cli_throttle, "background_slot", side_effect=_granting_slot),
        patch.object(supervisor, "forgetting_run_pass", side_effect=RuntimeError("boom")),
        patch.object(supervisor, "_run_narrative_memory_pass", narrative_spy),
        patch("brain.files.pending.sweep_expired"),
        patch("brain.health.sidecar_sweep.sweep_stale_sidecars"),
        patch.object(supervisor, "_attunement_should_run_backfill", return_value=False),
        patch.object(supervisor, "_attunement_should_run_supplementary_backfill", return_value=False),
        patch.object(supervisor, "_emotion_backfill_should_run", return_value=False),
        patch.object(supervisor, "_vocab_repair_should_run", return_value=False),
        patch.object(supervisor, "_soul_candidate_repair_should_run", return_value=False),
        patch(
            "brain.health.self_model_repair.should_run_self_model_repair",
            return_value=False,
        ),
    ):
        # Must not raise — forgetting's exception is caught locally, exactly as pre-fix.
        supervisor.run_folded(
            stop_event,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=_CapturingBus(),
            tick_interval_s=0.05,
            maker_enabled=False,
            notes_enabled=False,
            heartbeat_interval_s=None,
            soul_review_interval_s=0.0,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
            self_model_interval_s=None,
            compaction_interval_s=None,
            interest_sweep_interval_s=None,
            kindled_link_enabled=False,
        )
    narrative_spy.assert_called_once()
