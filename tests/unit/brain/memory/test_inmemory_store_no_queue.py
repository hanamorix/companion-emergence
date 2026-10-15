"""#375: an in-memory MemoryStore has no persona directory, so nothing may be queued for it.

``Path(":memory:").parent`` is ``Path(".")``, so ``MemoryStore.persona_dir`` used to hand back the
process's working directory and ``route_write`` appended gated candidates to
``./pending_candidates.jsonl`` (17 tests did it on every run). The docstring promised in-memory
stores "never enqueue"; these tests make it true and guard the cwd against the regression.
"""

from __future__ import annotations

import pytest

from brain.memory.pending import PendingQueue, route_write
from brain.memory.store import Memory, MemoryStore


def _mem(memory_type: str) -> Memory:
    return Memory.create_new(
        content="a thought", memory_type=memory_type, domain="us", emotions={"tenderness": 5.0}
    )


def test_in_memory_store_has_no_persona_dir() -> None:
    store = MemoryStore(":memory:")
    with pytest.raises(ValueError, match="in-memory"):
        store.persona_dir  # noqa: B018 - the property access is the behaviour under test
    store.close()


def test_route_write_refuses_a_gated_type_on_an_in_memory_store_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(":memory:")
    with pytest.raises(ValueError, match="in-memory"):
        route_write(store, _mem("dream"), source="test")
    assert list(tmp_path.iterdir()) == []  # no queue file, no lock sidecar
    store.close()


def test_route_write_still_commits_a_bypass_type_on_an_in_memory_store(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(":memory:")
    mem = _mem("journal_entry")
    assert route_write(store, mem, source="test") == mem.id
    assert store.get(mem.id) is not None
    assert list(tmp_path.iterdir()) == []
    store.close()


def test_route_write_still_enqueues_a_gated_type_for_a_file_backed_store(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem("dream")
    assert route_write(store, mem, source="test") == mem.id
    assert (tmp_path / "pending_candidates.jsonl").exists()
    assert [m.id for m in PendingQueue(store.persona_dir).read_recent("dream", limit=5)] == [mem.id]
    store.close()


def test_cwd_leak_guard_errors_a_test_that_grows_the_cwd_queue(pytester) -> None:
    pytester.makeconftest("from tests.conftest import _no_cwd_queue_leak  # noqa: F401\n")
    pytester.makepyfile(
        "def test_leaks():\n"
        "    with open('pending_candidates.jsonl', 'a', encoding='utf-8') as fh:\n"
        "        fh.write('{}\\n')\n"
    )
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, errors=1)  # body passes, the guard's teardown errors
    result.stdout.fnmatch_lines(["*pending_candidates.jsonl*"])


def test_cwd_leak_guard_ignores_a_queue_that_pre_exists_and_does_not_grow(pytester) -> None:
    pytester.makeconftest("from tests.conftest import _no_cwd_queue_leak  # noqa: F401\n")
    (pytester.path / "pending_candidates.jsonl").write_text("{}\n", encoding="utf-8")
    pytester.makepyfile("def test_quiet():\n    assert True\n")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
