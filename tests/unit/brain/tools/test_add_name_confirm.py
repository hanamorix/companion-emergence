"""Names increment (name-recall fix): the remove_name tool, the rejects list, and the
add_name confirm step (S105, S135, S137, S138; criteria N14, N30-N34).

"Her next tool call" (S138) = the next call reaching brain tool dispatch, including
the MCP search_memories forward; the Claude CLI's built-in web tools and calls the
MCP library rejects before our handler are not counted. Synthetic persona dirs,
invented names, no model, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from brain import prompt_strings
from brain.memory import known_names as kn
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.dispatch import ToolDispatchError, dispatch
from brain.tools.impls.remove_known_name import remove_known_name
from brain.tools.naming import TOOL_PREFIX_SLOT, fill_tool_prefix
from brain.tools.schemas import ADD_NAME_TOOL_NAME, REMOVE_NAME_TOOL_NAME

_REPO = Path(__file__).resolve().parents[4]
ASK = prompt_strings.get("tools.add_name.rejected_reply").format(name="Tuesday", tool_prefix=TOOL_PREFIX_SLOT)  # wording: test_persona_wording_snapshot.py


@pytest.fixture(autouse=True)
def _no_nested_lock(monkeypatch):
    """Every test here runs with a depth-checking wrapper around the known-names
    file lock: the lock is not re-entrant, so a nested acquisition would deadlock;
    this makes it fail the test at once instead (criterion N34, mutant D25)."""
    real = kn.file_lock
    depth = [0]
    nested: list[str] = []

    @contextlib.contextmanager
    def checked(path, *args, **kwargs):
        if depth[0]:
            nested.append(str(path))
            raise AssertionError("nested known-names file_lock (not re-entrant)")
        depth[0] += 1
        try:
            with real(path, *args, **kwargs) as got:
                yield got
        finally:
            depth[0] -= 1

    monkeypatch.setattr(kn, "file_lock", checked)
    yield
    assert nested == [], f"nested file_lock acquisitions: {nested}"


def _ctx(persona: Path) -> dict:
    return {"store": MemoryStore(":memory:"), "hebbian": HebbianMatrix(":memory:"), "persona_dir": persona}


def _q(persona: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(str(kn.known_names_path(persona)))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _listed(persona: Path) -> dict[str, str]:
    return dict(_q(persona, "SELECT name_lower, source FROM known_names"))


def _rejected(persona: Path) -> dict[str, str]:
    return dict(_q(persona, "SELECT name_lower, rejected_by FROM rejected_names"))


def _pending(persona: Path) -> list[str]:
    return [r[0] for r in _q(persona, "SELECT name_lower FROM pending_readd")]


def _add(persona: Path, name, ctx=None) -> dict:
    return dispatch(ADD_NAME_TOOL_NAME, {"name": name}, **(ctx or _ctx(persona)))


def _removed_tuesday(persona: Path) -> None:
    kn.admit_names(persona, ["Tuesday"], "gate")
    assert dispatch(REMOVE_NAME_TOOL_NAME, {"name": "Tuesday"}, **_ctx(persona)) == {
        "removed": True,
        "name": "Tuesday",
    }


# --------------------------------------------------------------------------
# N14 remove_name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["gate", "reappraiser"])
def test_n14_remove_name_removes_a_mechanically_added_name(tmp_path, source):
    kn.admit_names(tmp_path, ["O'Brien"], source)
    assert remove_known_name("O'Brien", persona_dir=tmp_path) == {"removed": True, "name": "O'Brien"}
    assert "o brien" not in kn.load_known_names(tmp_path).names
    assert _rejected(tmp_path) == {"o brien": "tool"}


def test_n14_remove_name_removes_her_own_names_and_a_stopword_name(tmp_path):
    for name in ("Wren", "Will"):
        assert kn.confirm_readd(tmp_path, name) == "added"
    assert _q(tmp_path, "SELECT name_lower FROM stopword_names") == [("will",)]
    assert remove_known_name("Wren", persona_dir=tmp_path)["removed"] is True
    assert remove_known_name("Will", persona_dir=tmp_path)["removed"] is True
    assert _listed(tmp_path) == {}
    assert _q(tmp_path, "SELECT name_lower FROM stopword_names") == []
    assert _rejected(tmp_path) == {"wren": "tool", "will": "tool"}


def test_n14_a_name_not_listed_is_not_removed_and_not_rejected(tmp_path):
    kn.admit_names(tmp_path, ["Wren"], "gate")
    assert remove_known_name("Pretzel", persona_dir=tmp_path) == {"removed": False, "name": "Pretzel"}
    assert _rejected(tmp_path) == {}


@pytest.mark.parametrize("bad", [None, 7, "", "   ", ["Wren"]])
def test_n14_a_bad_argument_is_false_and_never_raises(tmp_path, bad):
    assert remove_known_name(bad, persona_dir=tmp_path)["removed"] is False


def test_n14_no_file_is_created_by_a_removal(tmp_path):
    assert remove_known_name("Wren", persona_dir=tmp_path)["removed"] is False
    assert not kn.known_names_path(tmp_path).exists()


# --------------------------------------------------------------------------
# N30 the rejects list and the mechanical adders
# --------------------------------------------------------------------------


def test_n30_every_removal_is_on_the_rejects_list_and_again_after_a_readd(tmp_path):
    _removed_tuesday(tmp_path)
    assert _rejected(tmp_path) == {"tuesday": "tool"}
    assert _add(tmp_path, "Tuesday")["added"] is False
    assert _add(tmp_path, "Tuesday")["added"] is True
    assert _rejected(tmp_path) == {}
    assert kn.remove_names(tmp_path, ["Tuesday"], "tool") == ["tuesday"]
    assert _rejected(tmp_path) == {"tuesday": "tool"}


@pytest.mark.parametrize("source", ["gate", "reappraiser"])
def test_n30_a_mechanical_source_is_refused_a_rejected_name_and_asks_nothing(tmp_path, source):
    _removed_tuesday(tmp_path)
    assert kn.admit_names(tmp_path, ["Tuesday", "Pretzel"], source) == ["pretzel"]
    assert "tuesday" not in _listed(tmp_path)
    assert _pending(tmp_path) == []


def test_n30_the_tool_path_of_admit_names_refuses_a_rejected_name_too(tmp_path):
    _removed_tuesday(tmp_path)
    assert kn.admit_names(tmp_path, ["Tuesday"], "tool") == []
    assert _pending(tmp_path) == [] and "tuesday" not in _listed(tmp_path)


def test_n30_a_future_mechanical_source_is_refused_by_the_rule_not_a_list(tmp_path, monkeypatch):
    # The scratch file carries a widened source CHECK, so an "importer" row CAN be
    # written; both halves are asserted, which separates the rule from a source list.
    path = kn.known_names_path(tmp_path)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute(
        "CREATE TABLE known_names (name_lower TEXT PRIMARY KEY, display TEXT NOT NULL, source TEXT"
        " NOT NULL CHECK (source IN ('gate', 'reappraiser', 'tool', 'importer')), first_seen TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(kn, "SOURCES", (*kn.SOURCES, "importer"))
    _removed_tuesday(tmp_path)
    assert kn.admit_names(tmp_path, ["Pretzel"], "importer") == ["pretzel"]
    assert _listed(tmp_path)["pretzel"] == "importer"
    assert kn.admit_names(tmp_path, ["Tuesday"], "importer") == []
    assert "tuesday" not in _listed(tmp_path) and _pending(tmp_path) == []


# --------------------------------------------------------------------------
# N31 the confirm path
# --------------------------------------------------------------------------


def test_n31_the_first_add_asks_and_the_next_same_name_add_confirms(tmp_path):
    _removed_tuesday(tmp_path)
    ctx = _ctx(tmp_path)
    assert _add(tmp_path, " Tuesday ", ctx) == {"added": False, "name": "Tuesday", "message": ASK}
    assert "tuesday" not in _listed(tmp_path)
    assert _rejected(tmp_path) == {"tuesday": "tool"}
    assert _pending(tmp_path) == ["tuesday"]
    assert _add(tmp_path, "tuesday", ctx) == {"added": True, "name": "tuesday"}
    assert _listed(tmp_path)["tuesday"] == "tool"
    assert _rejected(tmp_path) == {} and _pending(tmp_path) == []


def test_n31_confirm_readd_outcomes(tmp_path):
    _removed_tuesday(tmp_path)
    assert kn.confirm_readd(tmp_path, "Tuesday") == "asked"
    assert kn.confirm_readd(tmp_path, "Tuesday") == "added"
    assert kn.confirm_readd(tmp_path, "   ") == "refused"  # nothing to store
    assert kn.confirm_readd(tmp_path, "!!!") == "added"  # S166: no ASCII letter or digit is admitted, keyed raw
    assert kn.confirm_readd(tmp_path, "I am") == "added"  # S144: an all-stopword entry is admitted as given
    assert _pending(tmp_path) == []


def test_n31_the_question_survives_a_process_boundary(tmp_path):
    _removed_tuesday(tmp_path)
    code = (
        "import json, sys; from pathlib import Path\n"
        "from brain.memory.hebbian import HebbianMatrix\n"
        "from brain.memory.store import MemoryStore\n"
        "from brain.tools.dispatch import dispatch\n"
        "out = dispatch('add_name', {'name': 'Tuesday'}, store=MemoryStore(':memory:'),"
        " hebbian=HebbianMatrix(':memory:'), persona_dir=Path(sys.argv[1]))\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=_REPO, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"added": False, "name": "Tuesday", "message": ASK}
    kn._reset_cache()
    assert _add(tmp_path, "Tuesday")["added"] is True


def _mcp_call(persona: Path, name: str, arguments: dict) -> str:
    from mcp.server import Server
    from mcp.types import CallToolRequest, CallToolRequestParams

    from brain.mcp_server.tools import register_tools

    server = Server("brain-tools")
    register_tools(server, persona_dir=persona, store=MemoryStore(":memory:"), hebbian=HebbianMatrix(":memory:"))
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(method="tools/call", params=CallToolRequestParams(name=name, arguments=arguments))
    result = asyncio.run(handler(request))
    return result.root.content[0].text


def test_n31_the_confirm_path_through_the_mcp_entry_point(tmp_path):
    _removed_tuesday(tmp_path)
    first = _mcp_call(tmp_path, "add_name", {"name": "Tuesday"})
    assert fill_tool_prefix(ASK, "") in first and json.loads(first)["added"] is False
    second = json.loads(_mcp_call(tmp_path, "add_name", {"name": "Tuesday"}))
    assert second == {"added": True, "name": "Tuesday"}


# --------------------------------------------------------------------------
# N32 lapse paths
# --------------------------------------------------------------------------


def _asked(persona: Path) -> None:
    _removed_tuesday(persona)
    assert _add(persona, "Tuesday")["added"] is False
    assert _pending(persona) == ["tuesday"]


def _still_rejected_and_asks_again(persona: Path) -> None:
    assert _pending(persona) == []
    assert _rejected(persona).get("tuesday") == "tool"
    assert _add(persona, "Tuesday")["added"] is False, "asks again instead of adding"


def test_n32_a_search_lapses_the_question(tmp_path):
    _asked(tmp_path)
    dispatch("search_memories", {"query": "harbour"}, **_ctx(tmp_path))
    _still_rejected_and_asks_again(tmp_path)


def test_n32_a_remove_of_another_name_lapses_it(tmp_path):
    _asked(tmp_path)
    dispatch(REMOVE_NAME_TOOL_NAME, {"name": "Pretzel"}, **_ctx(tmp_path))
    _still_rejected_and_asks_again(tmp_path)


def test_n32_an_add_of_a_different_name_lapses_it_and_is_handled_on_its_own(tmp_path):
    _asked(tmp_path)
    assert _add(tmp_path, "Wren") == {"added": True, "name": "Wren"}
    _still_rejected_and_asks_again(tmp_path)
    kn.admit_names(tmp_path, ["Harbour"], "gate")
    kn.remove_names(tmp_path, ["Harbour"], "tool")
    assert _add(tmp_path, "Harbour")["added"] is False
    assert _pending(tmp_path) == ["harbour"], "a second rejected name asks its own question"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("no_such_tool", {}), (ADD_NAME_TOOL_NAME, {}), (ADD_NAME_TOOL_NAME, {"name": 5})],
)
def test_n32_a_malformed_call_reaching_dispatch_lapses_it(tmp_path, name, arguments):
    _asked(tmp_path)
    with contextlib.suppress(ToolDispatchError):
        dispatch(name, arguments, **_ctx(tmp_path))
    _still_rejected_and_asks_again(tmp_path)


def test_n32_the_mcp_forward_lapses_it_even_when_the_bridge_is_unreachable(tmp_path):
    _asked(tmp_path)
    out = json.loads(_mcp_call(tmp_path, "search_memories", {"query": "x"}))
    assert out == {"error": "bridge unreachable"}
    _still_rejected_and_asks_again(tmp_path)


@pytest.mark.parametrize("arguments", [{}, {"name": 5}])
def test_n32_a_call_the_mcp_library_rejects_is_not_counted(tmp_path, arguments):
    _asked(tmp_path)
    text = _mcp_call(tmp_path, "add_name", arguments)
    assert "Input validation error" in text
    assert _pending(tmp_path) == ["tuesday"], "S138: not counted"


def test_n32_mechanical_writes_and_a_restart_do_not_lapse_it(tmp_path):
    _asked(tmp_path)
    kn.admit_names(tmp_path, ["Pretzel"], "gate")
    kn.remove_names(tmp_path, ["Pretzel"], "reappraiser")
    kn._reset_cache()
    assert _pending(tmp_path) == ["tuesday"]
    assert _add(tmp_path, "Tuesday")["added"] is True


# --------------------------------------------------------------------------
# N33 the hook: no lock, no write, unless a question is pending
# --------------------------------------------------------------------------


class _LockSpy:
    def __init__(self, real):
        self.real = real
        self.calls = 0
        self.depth = 0

    @contextlib.contextmanager
    def __call__(self, path, *args, **kwargs):
        self.calls += 1
        self.depth += 1
        if self.depth > 1:
            self.depth -= 1
            raise AssertionError("nested file_lock (it is not re-entrant)")
        try:
            with self.real(path, *args, **kwargs) as got:
                yield got
        finally:
            self.depth -= 1


def test_n33_no_file_means_nothing_created_and_no_lock(tmp_path, monkeypatch):
    spy = _LockSpy(kn.file_lock)
    monkeypatch.setattr(kn, "file_lock", spy)
    kn.note_tool_call(tmp_path, "search_memories", {"query": "x"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert spy.calls == 0 and list(tmp_path.iterdir()) == []


def test_n33_no_pending_question_means_a_read_only(tmp_path, monkeypatch):
    kn.admit_names(tmp_path, ["Wren"], "gate")
    before = kn._signature(kn.known_names_path(tmp_path))
    spy = _LockSpy(kn.file_lock)
    monkeypatch.setattr(kn, "file_lock", spy)
    kn.note_tool_call(tmp_path, "search_memories", {"query": "x"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert spy.calls == 0
    assert kn._signature(kn.known_names_path(tmp_path)) == before


def test_n33_a_lapse_takes_the_lock_once_and_twice_is_the_same_as_once(tmp_path, monkeypatch):
    _asked(tmp_path)
    spy = _LockSpy(kn.file_lock)
    monkeypatch.setattr(kn, "file_lock", spy)
    kn.note_tool_call(tmp_path, "search_memories", {"query": "x"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert spy.calls == 1 and _pending(tmp_path) == []
    kn.note_tool_call(tmp_path, "search_memories", {"query": "x"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert spy.calls == 1 and _pending(tmp_path) == []


def test_n33_the_hook_never_raises(tmp_path, monkeypatch, caplog):
    _asked(tmp_path)

    @contextlib.contextmanager
    def broken(*_a, **_k):
        raise OSError("lock broken")
        yield  # pragma: no cover

    monkeypatch.setattr(kn, "file_lock", broken)
    out = dispatch("search_memories", {"query": "harbour"}, **_ctx(tmp_path))
    assert "memories" in out or "mode" in out
    assert any("hook failed" in r.message for r in caplog.records)
    monkeypatch.undo()
    kn.known_names_path(tmp_path).write_bytes(b"garbage, not a database" * 100)
    kn.note_tool_call(tmp_path, "search_memories", {}, add_tool_name=ADD_NAME_TOOL_NAME)


# --------------------------------------------------------------------------
# N34 confirm atomicity
# --------------------------------------------------------------------------


def test_n34_confirm_readd_takes_the_lock_exactly_once(tmp_path, monkeypatch):
    _removed_tuesday(tmp_path)
    spy = _LockSpy(kn.file_lock)
    monkeypatch.setattr(kn, "file_lock", spy)
    assert kn.confirm_readd(tmp_path, "Tuesday") == "asked"
    assert kn.confirm_readd(tmp_path, "Tuesday") == "added"
    assert spy.calls == 2
    for source in kn.SOURCES:
        out = kn.admit_names(tmp_path, ["Pretzel"], source)
        assert isinstance(out, list) and all(isinstance(x, str) for x in out)


def test_n34_other_writes_between_the_ask_and_the_confirm(tmp_path):
    _removed_tuesday(tmp_path)
    kn.admit_names(tmp_path, ["Harbour"], "gate")
    assert kn.confirm_readd(tmp_path, "Tuesday") == "asked"
    kn.remove_names(tmp_path, ["Harbour"], "reappraiser")
    assert kn.admit_names(tmp_path, ["Tuesday"], "gate") == []
    assert kn.confirm_readd(tmp_path, "Tuesday") == "added"
    assert _q(tmp_path, "SELECT count(*) FROM known_names WHERE name_lower = 'tuesday'") == [(1,)]
    assert _rejected(tmp_path) == {"harbour": "reappraiser"}


def test_n34_a_failed_add_loses_neither_the_reject_nor_the_question(tmp_path, monkeypatch):
    _removed_tuesday(tmp_path)
    assert kn.confirm_readd(tmp_path, "Tuesday") == "asked"

    def boom(conn, row):
        raise sqlite3.IntegrityError("injected")

    monkeypatch.setattr(kn, "_write_tool_row", boom)
    assert kn.confirm_readd(tmp_path, "Tuesday") == "refused"
    monkeypatch.undo()
    assert _rejected(tmp_path) == {"tuesday": "tool"}
    assert _pending(tmp_path) == ["tuesday"]
    assert "tuesday" not in _listed(tmp_path)


# --------------------------------------------------------------------------
# Code review F3-F5: confirm-step gaps
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("listed", "asked", "confirmed"), [("O'Brien", "O'Brien", "o'brien"),
                                                             ("New York", "New York", "new   york")])
def test_f3_a_punctuated_or_multi_word_rejected_name_confirms_through_dispatch(tmp_path, listed, asked, confirmed):
    kn.admit_names(tmp_path, [listed], "gate")
    assert kn.remove_names(tmp_path, [listed], "tool")
    assert _add(tmp_path, asked)["added"] is False
    assert _add(tmp_path, confirmed)["added"] is True
    assert kn.normalize_name(listed) in _listed(tmp_path)
    assert _rejected(tmp_path) == {} and _pending(tmp_path) == []


def test_f4_a_question_for_one_name_never_confirms_another(tmp_path):
    kn.admit_names(tmp_path, ["Tuesday", "Harbour"], "gate")
    kn.remove_names(tmp_path, ["Tuesday", "Harbour"], "tool")
    assert kn.confirm_readd(tmp_path, "Tuesday") == "asked"
    assert kn.confirm_readd(tmp_path, "Harbour") == "asked", "a pending question for Tuesday is not a yes for Harbour"
    assert _pending(tmp_path) == ["harbour"]
    assert set(_rejected(tmp_path)) == {"tuesday", "harbour"}


def test_f5_an_ordinary_add_clears_a_stale_question(tmp_path):
    _removed_tuesday(tmp_path)
    assert kn.confirm_readd(tmp_path, "Tuesday") == "asked"
    assert kn.confirm_readd(tmp_path, "Wren") == "added"
    assert _pending(tmp_path) == []
