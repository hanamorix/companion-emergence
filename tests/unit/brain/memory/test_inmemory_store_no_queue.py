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
    result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly")
    result.assert_outcomes(passed=1, errors=1)  # body passes, the guard's teardown errors
    result.stdout.fnmatch_lines(["*pending_candidates.jsonl*"])


def test_cwd_leak_guard_ignores_a_queue_that_pre_exists_and_does_not_grow(pytester) -> None:
    pytester.makeconftest("from tests.conftest import _no_cwd_queue_leak  # noqa: F401\n")
    (pytester.path / "pending_candidates.jsonl").write_text("{}\n", encoding="utf-8")
    pytester.makepyfile("def test_quiet():\n    assert True\n")
    result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly")
    result.assert_outcomes(passed=1)


# --- #540: the guard attributes a write to the test whose process made it (it must hold under pytest-xdist) ---

_GUARD_CONFTEST = "from tests.conftest import _no_cwd_queue_leak  # noqa: F401\n"

# The previous guard, kept here as a control: it compares the file's size before and after the test.
_SIZE_CHECK_CONFTEST = '''
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _size_check():
    queue = Path.cwd() / "pending_candidates.jsonl"
    before = queue.stat().st_size if queue.exists() else None
    yield
    assert (queue.stat().st_size if queue.exists() else None) == before
'''

_FOREIGN_PROCESS_WRITES = '''
import subprocess
import sys


def test_innocent():
    code = "open('pending_candidates.jsonl', 'a', encoding='utf-8').write('{}\\\\n')"
    subprocess.run([sys.executable, "-c", code], check=True)
'''


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_another_process_growing_the_cwd_queue_does_not_fail_an_innocent_test(pytester) -> None:
    """Under xdist the repo root is shared by every worker: a leak made by another process is not this test's."""
    pytester.makeconftest(_GUARD_CONFTEST)
    pytester.makepyfile(_FOREIGN_PROCESS_WRITES)
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly").assert_outcomes(passed=1)


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_previous_size_check_fails_an_innocent_test_on_the_same_foreign_write(pytester) -> None:
    """Control arm: the scenario above really is one the size comparison got wrong."""
    pytester.makeconftest(_SIZE_CHECK_CONFTEST)
    pytester.makepyfile(_FOREIGN_PROCESS_WRITES)
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly").assert_outcomes(passed=1, errors=1)


_LEAKS = {
    "pending_queue": (
        "from brain.memory.pending import PendingQueue\n"
        "from brain.memory.store import Memory\n"
        "PendingQueue('.').enqueue(\n"
        "    Memory.create_new(content='x', memory_type='dream', domain='us', emotions={'tenderness': 5.0}), source='t')\n"
    ),
    "replace": "import os\nopen('t.tmp', 'w').write('{}')\nos.replace('t.tmp', 'pending_candidates.jsonl')\n",
    "remove": "import os\nos.remove('pending_candidates.jsonl')\n",
    "truncate": "import os\nos.truncate('pending_candidates.jsonl', 0)\n",
    "rename_away": "import os\nos.rename('pending_candidates.jsonl', 'moved.jsonl')\n",
}


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.parametrize("kind", sorted(_LEAKS))
def test_every_way_of_changing_the_cwd_queue_errors_the_test_that_did_it_and_names_the_call_site(pytester, kind) -> None:
    pytester.makeconftest(_GUARD_CONFTEST)
    (pytester.path / "pending_candidates.jsonl").write_text("{}\n", encoding="utf-8")
    body = "".join("    " + line + "\n" for line in _LEAKS[kind].splitlines())
    pytester.makepyfile(test_leaks="def test_leaks():\n" + body)
    result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*pending_candidates.jsonl*", "*test_leaks.py:[0-9]* in test_leaks*"])


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_reading_the_cwd_queue_is_not_a_write(pytester) -> None:
    pytester.makeconftest(_GUARD_CONFTEST)
    # bytes, not write_text: on Windows a text write turns "\n" into "\r\n" and the inner size check would see 4
    (pytester.path / "pending_candidates.jsonl").write_bytes(b"{}\n")
    pytester.makepyfile(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def test_reads():\n"
        "    assert Path('pending_candidates.jsonl').read_text(encoding='utf-8') == '{}\\n'\n"
        "    open('pending_candidates.jsonl', 'rb').close()\n"
        "    assert os.stat('pending_candidates.jsonl').st_size == 3\n"
    )
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly").assert_outcomes(passed=1)


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_a_same_named_file_in_another_directory_is_not_the_cwd_queue(pytester) -> None:
    """`shutil.rmtree` unlinks by bare name relative to a directory descriptor; a TemporaryDirectory holding its own
    `pending_candidates.jsonl` (a test that chdirs into it) is not the queue of the cwd the test started in."""
    pytester.makeconftest(_GUARD_CONFTEST)
    pytester.makepyfile(
        "import tempfile\n"
        "from pathlib import Path\n"
        "\n"
        "def test_elsewhere():\n"
        "    with tempfile.TemporaryDirectory() as other:\n"
        "        Path(other, 'pending_candidates.jsonl').write_text('{}\\n', encoding='utf-8')\n"
    )
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly").assert_outcomes(passed=1)
