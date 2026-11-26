"""ram-spike-fix INC-6, criterion C4 (AC4, S5/S6/S28/S29/S40): the lull tunable key has exactly one
reader, and every former idle-check caller routes through is_chat_idle. (The retired idle-tuning keys
and identifiers are listed in tests/unit/test_removed_names_stay_removed.py.)"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BRAIN_DIR = REPO_ROOT / "brain"

assert (BRAIN_DIR / "bridge" / "cli_throttle.py").is_file(), (
    f"sanity: expected brain/ at {BRAIN_DIR}"
)


def _iter_call_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


# ---------------------------------------------------------------------------
# C4(b): the lull tunable key is read in EXACTLY one place — cli_throttle's
# private _lull_seconds(), called only by is_chat_idle.
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason='1.5 s of CPU in its call phase on a 4-core dev VM, 2.0 s wall (serial run for #540, J12)')
def test_lull_tunable_key_read_in_exactly_one_place() -> None:
    """AST-based functional-read count — a docstring/comment mentioning the key (tunables.py's own fence
    note, cli_throttle.py's module docstring, server.py's lifespan comment,
    tunables_migration.py's prose) is correctly excluded; only an actual
    `tunables.get_tunable("chat.idle_lull_seconds", ...)` call counts as a
    read. The one `tunables.register(...)` call (tunables.py) is excluded
    from THIS check by name (it's the registration, not a read)."""
    key = "chat.idle_lull_seconds"
    read_hits: list[str] = []
    for path in BRAIN_DIR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for call in _iter_call_nodes(tree):
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if name != "get_tunable":
                continue
            for arg in call.args:
                if isinstance(arg, ast.Constant) and arg.value == key:
                    read_hits.append(f"{path.relative_to(REPO_ROOT)}:{call.lineno}")

    assert len(read_hits) == 1, (
        f"expected exactly one get_tunable read of {key!r} (cli_throttle._lull_seconds), "
        f"found {len(read_hits)}:\n" + "\n".join(read_hits)
    )
    assert "cli_throttle.py" in read_hits[0]


# ---------------------------------------------------------------------------
# C4(c): every former idle-check caller routes through is_chat_idle — flip a
# fake "not idle" and assert deferral, for both the shared cli_throttle
# wrapper functions AND the two direct-import callers this increment touched.
# ---------------------------------------------------------------------------


def test_shared_wrapper_functions_all_gate_on_is_chat_idle(monkeypatch) -> None:
    from brain.bridge import cli_throttle

    monkeypatch.setattr(cli_throttle, "is_chat_idle", lambda **_: False)
    assert cli_throttle.should_yield() is True
    assert cli_throttle.slot_available() is False
    assert cli_throttle.acquire_background() is False
    with cli_throttle.background_slot() as ok:
        assert ok is False


def test_self_model_articulate_defers_when_not_idle(monkeypatch, tmp_path: Path) -> None:
    from brain.bridge import cli_throttle
    from brain.self_model.articulate import articulate
    from brain.self_model.gap import Gap

    monkeypatch.setattr(cli_throttle, "is_chat_idle", lambda **_: False)

    class _CountingProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, *_a, **_kw):
            self.calls += 1
            return "note"

    provider = _CountingProvider()
    gap = Gap(per_channel={"grief": 5.0}, magnitude=5.0, unnamed_pressure=0.0)
    with pytest.raises(cli_throttle.ThrottleDeferred):
        articulate(gap, provider=provider, persona_dir=tmp_path)
    assert provider.calls == 0


def test_pass2_queue_defers_when_not_idle(monkeypatch, tmp_path: Path) -> None:
    """C4 evidence: pass 2 defers (does not run) while chat is not idle.

    Exercises the REAL production gate `cli_throttle.acquire_background()` —
    the slot the pass-2 job (supervisor `_build_gated_jobs`, ram-spike-fix
    INC-9; it replaced the old worker thread) takes before ever invoking
    `drain_all_locked` — rather than re-deriving the gating decision inside
    the test itself. The central cadence function additionally asks
    is_chat_idle before the job at all (test_central_cadence.py)."""
    from brain.bridge import cli_throttle
    from brain.chat import pass2_queue

    monkeypatch.setattr(cli_throttle, "is_chat_idle", lambda **_: False)
    pass2_queue.reset()
    ran = {"n": 0}
    record_id = pass2_queue.new_record_id()
    pass2_queue.register_test_side_effect(record_id, lambda: ran.__setitem__("n", ran["n"] + 1))
    pass2_queue.enqueue({"id": record_id, "kind": "test_probe"}, persona_dir=tmp_path)

    # The real pre-flight gate denies a background slot while not idle, so
    # the pass-2 job never even calls drain_all_locked.
    assert cli_throttle.acquire_background() is False
    assert ran["n"] == 0
    assert pass2_queue._queue_size(tmp_path) == 1  # left queued, not dropped
    pass2_queue.reset()


def test_emotion_backfill_yields_when_not_idle(monkeypatch, tmp_path: Path) -> None:
    from brain.bridge import cli_throttle
    from brain.ingest.emotion_backfill import run_emotion_backfill
    from brain.memory.store import Memory, MemoryStore

    monkeypatch.setattr(cli_throttle, "is_chat_idle", lambda **_: False)

    store = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    m = Memory.create_new(
        content="a memory long enough to clear the embed min-chars floor",
        memory_type="conversation",
        domain="us",
    )
    store.create(m)
    store.close()

    calls = {"n": 0}

    def tagger(_memory):
        calls["n"] += 1
        return {"loneliness": 7.0}

    state = run_emotion_backfill(tmp_path, tagger_fn=tagger, cap=50, delay_s=0)
    assert calls["n"] == 0
    assert state.status != "complete"


# The full 2-plan §3.2 caller enumeration (file:line as the plan states them),
# minus the 3 callers already given dedicated runtime tests above
# (self_model/articulate.py, chat/pass2_queue.py, ingest/emotion_backfill.py)
# and the supervisor.py session-snapshot/prune block (its own dedicated
# runtime test lives in test_supervisor.py). Stage-6 code red-team MINOR
# (label-audit gap): C4(c)'s own wording ("every caller... a test flips a
# fake 'not idle' and asserts EACH caller defers") is only executed at
# runtime for a handful of these; the plan's "zero lines changed" argument
# for the rest (they already call cli_throttle.background_slot()/
# slot_available() with no arguments, so the internals change transparently
# underneath them) rested on inspection, not an executable check. This
# completeness sweep converts that argument into one: EVERY enumerated
# caller file must (a) still reference a cli_throttle idle-gating function,
# and (b) carry no `min_idle=` keyword anywhere near it (the retired
# per-caller override mechanism).
_ENUMERATED_CALLER_FILES = (
    "engines/dream.py",
    "engines/research.py",
    "engines/reflex.py",
    "notes/runner.py",
    "notes/__init__.py",
    "maker/making_runner.py",
    "maker/__init__.py",
    "soul/review.py",
    "attunement/backfill.py",
    "health/vocab_repair.py",
    "initiate/voice_reflection.py",
    "initiate/review.py",
    "kindled_link/session_engine.py",
    "kindled_link/privacy_gate.py",
    "kindled_link/relationship.py",
)


@pytest.mark.parametrize("relpath", _ENUMERATED_CALLER_FILES)
def test_enumerated_caller_still_routes_through_cli_throttle_no_min_idle(relpath: str) -> None:
    path = BRAIN_DIR / relpath
    assert path.is_file(), f"expected caller file at {path}"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls_cli_throttle_gate = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name in ("background_slot", "slot_available", "acquire_background", "should_yield"):
            calls_cli_throttle_gate = True
            for kw in node.keywords:
                assert kw.arg != "min_idle", (
                    f"{relpath}:{node.lineno} still passes the retired min_idle= kwarg"
                )
    assert calls_cli_throttle_gate, (
        f"{relpath} no longer calls any cli_throttle idle-gating function — "
        "either it was migrated to call is_chat_idle() directly (update this "
        "list) or the caller was silently dropped"
    )


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_the_central_pass_asks_is_chat_idle_by_default() -> None:
    """The supervisor calls ``central_cadence.run_central_pass`` without an ``is_idle`` argument, so the idle gate
    every gated job (the session snapshot/prune included) waits on is the function's default, which must be
    ``cli_throttle.is_chat_idle``. The tests of the central function inject their own ``is_idle``, so only this one
    pins the default the production call relies on."""
    import inspect

    from brain.bridge import central_cadence, cli_throttle, supervisor

    default_is_idle = inspect.signature(central_cadence.run_central_pass).parameters["is_idle"].default
    assert default_is_idle is cli_throttle.is_chat_idle
    assert "session_snapshot_prune" in central_cadence.GATED_JOB_ORDER
    assert "central_cadence.run_central_pass(" in inspect.getsource(supervisor.run_folded)
