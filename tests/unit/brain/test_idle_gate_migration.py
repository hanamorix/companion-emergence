"""ram-spike-fix INC-6, criterion C4 (AC4, S5/S6/S28/S29/S40): the retired
idle-tuning keys are gone from brain/, the lull tunable key has exactly one
reader, and every former idle-check caller routes through is_chat_idle."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BRAIN_DIR = REPO_ROOT / "brain"

assert (BRAIN_DIR / "bridge" / "cli_throttle.py").is_file(), (
    f"sanity: expected brain/ at {BRAIN_DIR}"
)

_RETIRED_TUNABLE_KEYS = (
    "throttle.background_min_idle_seconds",
    "chat.pass2_min_idle_seconds",
    "self_model.articulate_min_idle_seconds",
)
_RETIRED_IDENTIFIER = "_ACTIVE_CHAT_IDLE_MINUTES"


def _iter_call_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _is_tunables_register_or_get(call: ast.Call) -> bool:
    func = call.func
    name = None
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    return name in ("register", "get_tunable")


def _find_functional_tunable_reads(pattern: str) -> list[str]:
    """AST-based: a "functional read" is `tunables.register(<pattern>, ...)`
    or `tunables.get_tunable(<pattern>, ...)` — NOT a bare string mention in a
    docstring/comment (which ast never turns into a Call argument here)."""
    hits: list[str] = []
    for path in BRAIN_DIR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for call in _iter_call_nodes(tree):
            if not _is_tunables_register_or_get(call):
                continue
            for arg in call.args:
                if isinstance(arg, ast.Constant) and arg.value == pattern:
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{call.lineno}")
    return hits


def _find_identifier_uses(identifier: str) -> list[str]:
    """AST-based: an actual Python NAME node (Load/Store/assignment/reference)
    — never a docstring/comment mention (those are ast.Constant strings, not
    ast.Name nodes, so they never match here)."""
    hits: list[str] = []
    for path in BRAIN_DIR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == identifier:
                hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return hits


# ---------------------------------------------------------------------------
# C4(a): no code under brain/ reads the four retired keys/constants, or calls
# run_folded(silence_minutes=...). Positive control: the scanner must find a
# planted reference. Scoped to FUNCTIONAL reads (an executable tunables call,
# or an actual Python identifier) — a docstring/comment explaining the
# migration history (e.g. tunables_migration.py's own module docstring, or
# its plain string list of keys to strip) is prose, not a read, and is
# correctly excluded (only tunables_migration.py's own removal logic touches
# these strings, as DATA to delete, never via tunables.register/get_tunable).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", _RETIRED_TUNABLE_KEYS)
def test_retired_tunable_key_not_functionally_read(pattern: str) -> None:
    hits = _find_functional_tunable_reads(pattern)
    assert hits == [], f"retired tunable key {pattern!r} still read via tunables.*:\n" + "\n".join(hits)


def test_retired_identifier_not_used_as_a_python_name() -> None:
    hits = _find_identifier_uses(_RETIRED_IDENTIFIER)
    assert hits == [], f"retired identifier {_RETIRED_IDENTIFIER!r} still used:\n" + "\n".join(hits)


def test_positive_control_ast_scanner_finds_a_planted_functional_read(tmp_path: Path) -> None:
    """Demonstrates the AST scanners above actually discriminate — a planted
    FUNCTIONAL call (not a docstring mention) must be found."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        'from brain import tunables\n'
        'X = tunables.register("throttle.background_min_idle_seconds", 300.0)\n',
        encoding="utf-8",
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    found = any(
        _is_tunables_register_or_get(call)
        and any(
            isinstance(a, ast.Constant) and a.value == "throttle.background_min_idle_seconds"
            for a in call.args
        )
        for call in _iter_call_nodes(tree)
    )
    assert found, "scanner failed to find its own planted positive control"

    # And a DOCSTRING-only mention (prose) must NOT be found by the same logic.
    prose_only = tmp_path / "prose_only.py"
    prose_only.write_text(
        '"""Mentions throttle.background_min_idle_seconds in prose only."""\n'
        'X = 1\n',
        encoding="utf-8",
    )
    tree2 = ast.parse(prose_only.read_text(encoding="utf-8"))
    found2 = any(
        _is_tunables_register_or_get(call)
        for call in _iter_call_nodes(tree2)
    )
    assert not found2, "scanner must not flag a bare docstring mention as a functional read"


def test_run_folded_silence_minutes_kwarg_absent(tmp_path: Path) -> None:
    """No code constructs a run_folded(...) call/kwargs dict carrying a
    "silence_minutes" key (the AST check underlying
    test_build_app_kwargs_to_run_folded_carry_no_silence_minutes_key in
    test_supervisor_db_overhead.py; duplicated here, scoped to this
    criterion's own positive control, per C4(a))."""
    import ast

    server_src = (BRAIN_DIR / "bridge" / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(server_src)
    build_app_node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_app"
    )
    for node in ast.walk(build_app_node):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                assert not (
                    isinstance(key, ast.Constant) and key.value == "silence_minutes"
                ), "build_app must not forward silence_minutes into run_folded's kwargs"


def test_positive_control_identifier_scanner_finds_a_planted_assignment(tmp_path: Path) -> None:
    """The identifier scanner (_find_identifier_uses) must find a planted
    actual assignment, but NOT a bare comment mention of the same name."""
    planted = tmp_path / "planted.py"
    planted.write_text("_ACTIVE_CHAT_IDLE_MINUTES = 5.0\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    found = any(
        isinstance(n, ast.Name) and n.id == "_ACTIVE_CHAT_IDLE_MINUTES" for n in ast.walk(tree)
    )
    assert found, "scanner failed to find its own planted positive control"

    comment_only = tmp_path / "comment_only.py"
    comment_only.write_text(
        "X = 1  # references _ACTIVE_CHAT_IDLE_MINUTES for the positive control\n",
        encoding="utf-8",
    )
    tree2 = ast.parse(comment_only.read_text(encoding="utf-8"))
    found2 = any(
        isinstance(n, ast.Name) and n.id == "_ACTIVE_CHAT_IDLE_MINUTES" for n in ast.walk(tree2)
    )
    assert not found2, "scanner must not flag a bare comment mention"


# ---------------------------------------------------------------------------
# C4(b): the lull tunable key is read in EXACTLY one place — cli_throttle's
# private _lull_seconds(), called only by is_chat_idle. time_since_last_message
# (the prune's own accessor) reads a DIFFERENT anchor and must NOT count.
# ---------------------------------------------------------------------------


def test_lull_tunable_key_read_in_exactly_one_place() -> None:
    """AST-based functional-read count (see _find_functional_tunable_reads
    above) — a docstring/comment mentioning the key (tunables.py's own fence
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


def test_time_since_last_message_reads_a_different_anchor_not_the_tunable_key() -> None:
    """S72/2-plan §3.3a: time_since_last_message() must not itself reference
    the chat.idle_lull_seconds key (it reads the monotonic anchor only) — so
    it correctly does NOT count against the "one reader" claim above."""
    src = (BRAIN_DIR / "bridge" / "cli_throttle.py").read_text(encoding="utf-8")
    match = re.search(
        r"def time_since_last_message\(.*?\n(?:.*\n)*?(?=\ndef |\Z)", src
    )
    assert match, "time_since_last_message not found in cli_throttle.py"
    body = match.group(0)
    assert "chat.idle_lull_seconds" not in body
    assert "_lull_seconds" not in body


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


def test_supervisor_snapshot_prune_block_gated_by_is_chat_idle(monkeypatch) -> None:
    """Static confirmation that supervisor.py's session-snapshot/prune work is
    idle-gated (the executable per-caller test lives in test_supervisor.py /
    test_central_cadence.py; this is the C4(c) grep-style companion for THIS
    specific caller). Since ram-spike-fix INC-9 the snapshot/prune is a gated
    job of the central cadence function: its only call site is inside
    ``_build_gated_jobs`` (never in run_folded's own body), and the central
    function asks ``cli_throttle.is_chat_idle`` before every job."""
    import inspect

    from brain.bridge import central_cadence, cli_throttle, supervisor

    src = (BRAIN_DIR / "bridge" / "supervisor.py").read_text(encoding="utf-8")
    assert src.count("reports = snapshot_stale_sessions(") == 1
    idx = src.index("reports = snapshot_stale_sessions(")
    assert src.rfind("def _build_gated_jobs(", 0, idx) > src.rfind("def run_folded(", 0, idx)
    default_is_idle = inspect.signature(central_cadence.run_central_pass).parameters["is_idle"].default
    assert default_is_idle is cli_throttle.is_chat_idle
    assert "session_snapshot_prune" in central_cadence.GATED_JOB_ORDER
    assert "central_cadence.run_central_pass(" in inspect.getsource(supervisor.run_folded)
