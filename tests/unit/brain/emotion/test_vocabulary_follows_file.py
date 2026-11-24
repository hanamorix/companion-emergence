"""#803 step 15 (VC2): every process's copy of her vocabulary follows the file.

A process that keeps a copy of her vocabulary (the bridge, the owner; the memory server and a `nell` command, read-only copies)
keeps the stat key of the file its copy came from, and before every read of the copy (`vocabulary.get`, `list_all`,
`by_category`) and every write (`vocabulary_file._transact`) stats the file: a different key reloads the copy first, exactly. A hand edit
is therefore seen on the next read. All personas here are tmp directories with synthetic entries.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest

from brain.emotion import persona_loader, vocabulary, vocabulary_file
from brain.emotion.vocabulary import Emotion
from brain.health import json_io
from brain.memory.store import Memory, MemoryStore
from brain.utils import file_ops

VOCAB = "emotion_vocabulary.json"


def _entry(name: str, description: str = "her own words", days: float | None = 6.0, category: str = "persona_extension") -> dict:
    return {"name": name, "description": description, "category": category, "decay_half_life_days": days, "intensity_clamp": 10.0}


def _hand_edit(path: Path, entries: list[dict]) -> None:
    """Replace the file the way an editor does: a new file moved over the old one, so its stat key is new."""
    tmp = path.with_name(path.name + ".hand")
    tmp.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")
    os.replace(tmp, path)


def _file_names(path: Path) -> list[str]:
    return [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]


def _registry_extensions() -> set[str]:
    return {e.name for e in vocabulary.list_all() if not vocabulary.is_baseline(e.name)}


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / VOCAB


@pytest.fixture
def owner() -> None:
    vocabulary.set_bridge_owner(True)  # the autouse registry fixture puts the flag back


# ----------------------------------------------------------------------------------------------------------------
# Criterion 1: a hand edit that adds, changes and removes an entry is seen on the next read
# ----------------------------------------------------------------------------------------------------------------


def _edit_then_read(path: Path) -> None:
    _hand_edit(path, [_entry("keep one", "first"), _entry("drop me"), _entry("change me", "before")])


def _assert_edit_seen(path: Path) -> None:
    _hand_edit(path, [_entry("keep one", "first"), _entry("change me", "after"), _entry("brand new")])
    assert vocabulary.get("brand new") is not None  # added, seen by get
    assert vocabulary.get("drop me") is None  # removed
    assert vocabulary.get("change me").description == "after"  # changed
    _hand_edit(path, [_entry("keep one", "first"), _entry("change me", "third"), _entry("brand new"), _entry("later one")])
    assert "later one" in {e.name for e in vocabulary.list_all()}  # list_all
    assert [e.name for e in vocabulary.by_category("persona_extension")].count("later one") == 1  # by_category
    _hand_edit(path, [_entry("keep one", "first")])
    assert [e.name for e in vocabulary.by_category("persona_extension")] == ["keep one"]


def test_the_owner_sees_a_hand_edit_on_the_next_read(path, owner):
    _edit_then_read(path)
    persona_loader.load_persona_vocabulary(path)
    assert vocabulary.get("drop me") is not None
    _assert_edit_seen(path)


def test_a_non_owner_sees_a_hand_edit_on_the_next_read_the_memory_server_path(path):
    _edit_then_read(path)
    persona_loader.load_persona_vocabulary(path, read_only=True)  # what the memory server's start calls
    assert vocabulary.get("drop me") is not None
    _assert_edit_seen(path)


# ----------------------------------------------------------------------------------------------------------------
# Criterion 2: a core entry or a twin added by hand
# ----------------------------------------------------------------------------------------------------------------


def _hand_add_core_and_twin(path: Path) -> None:
    _hand_edit(
        path,
        [_entry("her word"), _entry("joy", "hand-written core"), _entry("tender longing", "one"), _entry("tender_longing", "two")],
    )


def test_the_owners_reload_cleans_a_hand_added_core_entry_and_twin_in_the_file(path, owner):
    _hand_edit(path, [_entry("her word")])
    persona_loader.load_persona_vocabulary(path)
    _hand_add_core_and_twin(path)
    assert vocabulary.get("tender longing") is not None  # the first read after the edit reloads and cleans
    assert _file_names(path) == ["her word", "tender longing"]  # core removed, twins merged under the first spelling
    assert vocabulary.get("joy").description == "happiness, delight, warmth"  # the framework's entry


def test_a_non_owners_reload_only_reads_a_core_entry_and_a_twin(path):
    _hand_edit(path, [_entry("her word")])
    persona_loader.load_persona_vocabulary(path, read_only=True)
    _hand_add_core_and_twin(path)
    before = path.read_bytes()
    assert vocabulary.get("tender longing").description == "one"  # the first twin wins
    assert path.read_bytes() == before  # nothing written, nothing cleaned
    assert vocabulary.get("joy").description == "happiness, delight, warmth"
    assert not list(path.parent.glob("*.bak*"))


def test_the_load_not_a_get_is_the_first_reader_of_a_hand_added_core_entry_and_twin_and_cleans_the_file(path, owner):
    """R3-1: a load that sees the hand edit first must not record its key with a core name or twin left in the file."""
    _hand_edit(path, [_entry("her word")])
    persona_loader.load_persona_vocabulary(path)
    _hand_add_core_and_twin(path)
    persona_loader.load_persona_vocabulary(path)
    assert _file_names(path) == ["her word", "tender longing"]


def test_a_failed_owner_clean_up_is_retried_by_the_next_explicit_load(path, owner, monkeypatch):
    _hand_edit(path, [_entry("her word")])
    persona_loader.load_persona_vocabulary(path)
    _hand_add_core_and_twin(path)
    real = vocabulary_file.remove_core_entries
    monkeypatch.setattr(vocabulary_file, "remove_core_entries", lambda p: vocabulary_file.CoreRemoval([], cannot_write=True))
    persona_loader.load_persona_vocabulary(path)
    assert "joy" in _file_names(path)
    monkeypatch.setattr(vocabulary_file, "remove_core_entries", real)
    persona_loader.load_persona_vocabulary(path)  # the next tick's load
    assert "joy" not in _file_names(path)


# ----------------------------------------------------------------------------------------------------------------
# Criterion 3: a write after a hand edit keeps the hand edit
# ----------------------------------------------------------------------------------------------------------------


def test_a_write_after_a_hand_edit_keeps_the_hand_edit_in_the_file_and_the_copy(path, owner):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    _hand_edit(path, [_entry("one"), _entry("hand added")])
    results = vocabulary_file.add(path, [("written after", None)])
    assert results[0].changed
    assert _file_names(path) == ["one", "hand added", "written_after"]
    assert _registry_extensions() == {"one", "hand added", "written_after"}


def test_the_copy_matches_the_file_after_a_write_so_the_next_read_does_not_reload(path, owner, monkeypatch):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    vocabulary_file.add(path, [("two", None)])
    calls = []
    real = vocabulary_file.reload
    monkeypatch.setattr(vocabulary_file, "reload", lambda p: calls.append(p) or real(p))
    assert vocabulary.get("two") is not None
    assert calls == []


# ----------------------------------------------------------------------------------------------------------------
# Criterion 4: an unreadable file
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_owner", [True, False])
def test_an_unreadable_file_leaves_the_copy_and_is_logged_once_until_it_changes(path, caplog, is_owner):
    vocabulary.set_bridge_owner(is_owner)
    _hand_edit(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path, read_only=not is_owner)
    path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            assert vocabulary.get("kept one") is not None  # the copy is as it was
        assert [e.name for e in vocabulary.list_all() if e.name == "kept one"] == ["kept one"]
    assert path.read_text(encoding="utf-8") == "{ not json"  # left as found: no repair here
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "emotion_vocabulary" in r.getMessage()]
    assert len(warnings) == 1
    _hand_edit(path, [_entry("after the fix")])
    assert vocabulary.get("after the fix") is not None  # the change of file reloads
    assert vocabulary.get("kept one") is None


# ----------------------------------------------------------------------------------------------------------------
# Criterion 5: an absent file
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_owner", [True, False])
def test_an_absent_file_drops_every_non_framework_entry(path, is_owner):
    vocabulary.set_bridge_owner(is_owner)
    _hand_edit(path, [_entry("one"), _entry("two")])
    persona_loader.load_persona_vocabulary(path, read_only=not is_owner)
    assert _registry_extensions() == {"one", "two"}
    path.unlink()
    assert vocabulary.get("one") is None
    assert _registry_extensions() == set()
    assert vocabulary.get("joy") is not None  # the framework's own stay


def test_the_owner_with_memories_naming_a_feeling_rebuilds_a_vanished_file_with_a_short_lived_store(path, owner, monkeypatch):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    store = MemoryStore(path.parent / "memories.db")
    store.create(
        Memory.create_new(content="x", memory_type="conversation", domain="us", emotions={"emergence": 5.0}),
        check_names=False,
        fold=False,
    )
    store.close()
    opened: list[object] = []
    closed: list[object] = []

    class Recording(MemoryStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def close(self):
            closed.append(self)
            super().close()

    monkeypatch.setattr(persona_loader, "MemoryStore", Recording)
    path.unlink()
    assert vocabulary.get("emergence") is not None  # the rebuild wrote the name her memories hold
    assert "emergence" in _file_names(path)
    assert vocabulary.get("one") is None
    assert len(opened) == 1 and closed == opened  # opened for the rebuild, closed when done


def test_an_absent_file_with_no_memories_database_stays_absent_and_opens_no_store(path, owner, monkeypatch):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    monkeypatch.setattr(persona_loader, "MemoryStore", lambda *a, **k: pytest.fail("a store was opened"))
    path.unlink()
    assert vocabulary.get("one") is None
    assert not path.exists()
    assert not (path.parent / "memories.db").exists()


def test_a_rebuild_store_that_raises_does_not_stop_the_read(path, owner, monkeypatch, caplog):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    (path.parent / "memories.db").write_bytes(b"")
    monkeypatch.setattr(persona_loader, "MemoryStore", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("store broke")))
    path.unlink()
    with caplog.at_level(logging.WARNING):
        assert vocabulary.get("joy") is not None  # the read goes on
        assert vocabulary.get("one") is None
        vocabulary.get("joy")
    assert sum("store broke" in r.getMessage() or (r.exc_info and "store broke" in str(r.exc_info[1])) for r in caplog.records) == 1


# ----------------------------------------------------------------------------------------------------------------
# Criterion 6: a second load of an unchanged file reports 0
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.no_base_failure(reason="control: the old add-only load was idempotent too, so a second explicit load reported 0; this pins that the exact reload keeps it")
def test_a_second_explicit_load_of_an_unchanged_file_reports_zero(path):
    _hand_edit(path, [_entry("one"), _entry("two")])
    assert persona_loader.load_persona_vocabulary(path, read_only=True) == 2
    assert persona_loader.load_persona_vocabulary(path, read_only=True) == 0


def test_a_reload_of_an_unchanged_file_reports_zero_and_a_changed_one_reports_the_changes(path):
    _hand_edit(path, [_entry("one", "a"), _entry("two")])
    persona_loader.load_persona_vocabulary(path, read_only=True)
    assert vocabulary_file.reload(path)[0] == 0
    _hand_edit(path, [_entry("one", "b"), _entry("three")])  # one changed, two removed, three added
    assert vocabulary_file.reload(path)[0] == 3
    assert vocabulary_file.reload(path)[0] == 0


def test_an_exact_load_removes_a_name_the_file_no_longer_holds(path):
    _hand_edit(path, [_entry("one"), _entry("two")])
    persona_loader.load_persona_vocabulary(path, read_only=True)
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path, read_only=True)
    assert _registry_extensions() == {"one"}


# ----------------------------------------------------------------------------------------------------------------
# Criterion 7: structure, no recursion, no deadlock
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.no_base_failure(reason="control: vocabulary.py already imported neither module at 51048e29; this pins that it keeps importing neither")
def test_the_registry_module_imports_neither_the_file_module_nor_the_access_module():
    source = Path(vocabulary.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from ", "    import ", "    from "))]
    assert imports  # the guard sees the module's imports
    assert not [line for line in imports if "vocabulary_file" in line or "vocabulary_access" in line]


def _within(seconds: float, target) -> bool:
    """Run `target` in a thread; True when it returned in time (a deadlock fails the test instead of hanging it)."""
    done = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            target()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    finished = done.wait(seconds)
    if errors:
        raise errors[0]
    return finished


@pytest.mark.parametrize("is_owner", [True, False])
def test_an_explicit_load_with_the_file_changed_by_hand_returns_and_matches_the_file(path, is_owner):
    vocabulary.set_bridge_owner(is_owner)
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path, read_only=not is_owner)
    _hand_edit(path, [_entry("two")])  # one removed, two added: the load must match the file exactly
    assert _within(10, lambda: persona_loader.load_persona_vocabulary(path, read_only=not is_owner)), "the explicit load deadlocked"
    assert _registry_extensions() == {"two"}


@pytest.mark.parametrize("is_owner", [True, False])
def test_a_read_with_the_file_changed_by_hand_returns_and_reloads(path, is_owner):
    vocabulary.set_bridge_owner(is_owner)
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path, read_only=not is_owner)
    _hand_edit(path, [_entry("one"), _entry("two")])
    assert _within(10, lambda: vocabulary.get("two")), "the check's reload deadlocked"
    assert vocabulary.get("two") is not None


def test_a_registry_step_that_calls_get_inside_the_registry_lock_with_the_file_just_changed_returns(path, owner):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    _hand_edit(path, [_entry("one"), _entry("hand")])
    assert _within(10, lambda: vocabulary_file.add(path, [("added", None)])), "the write deadlocked"
    assert _registry_extensions() == {"one", "hand", "added"}


def test_the_check_is_off_per_thread_a_second_threads_read_is_still_checked_while_the_first_reloads(path):
    entered = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def slow_reload() -> None:
        seen.append("slow")
        entered.set()
        release.wait(10)
        vocabulary.note_copy_key(path, file_ops.stat_path(path).key)

    _hand_edit(path, [_entry("one")])
    vocabulary.set_file_copy(path, slow_reload, key=file_ops.stat_path(path).key, stale=False)
    _hand_edit(path, [_entry("one"), _entry("two")])
    first = threading.Thread(target=lambda: vocabulary.get("one"), daemon=True)
    first.start()
    assert entered.wait(10)
    # the first thread is inside the reload with ITS check off; this thread's read must still stat and reload
    second = threading.Thread(target=lambda: vocabulary.get("one"), daemon=True)
    second.start()
    second.join(0.5)
    release.set()
    first.join(10)
    second.join(10)
    assert seen.count("slow") == 2  # both threads ran the reload: the second was checked on its own


def test_a_reload_that_raises_is_logged_once_and_the_read_goes_on(path, caplog):
    def broken() -> None:
        raise RuntimeError("reload broke")

    _hand_edit(path, [_entry("one")])
    vocabulary.set_file_copy(path, broken, key=file_ops.stat_path(path).key, stale=False)
    _hand_edit(path, [_entry("one"), _entry("two")])
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert vocabulary.get("joy") is not None
    assert sum("reload broke" in str(r.exc_info[1]) for r in caplog.records if r.exc_info) == 1


def test_set_file_copy_stale_reloads_at_the_next_read_and_a_noted_key_clears_it(path):
    calls: list[int] = []
    _hand_edit(path, [_entry("one")])

    def reload() -> None:
        calls.append(1)
        vocabulary.note_copy_key(path, file_ops.stat_path(path).key)

    vocabulary.set_file_copy(path, reload)  # stale by default: not known to match
    vocabulary.get("joy")
    vocabulary.get("joy")
    assert calls == [1]
    assert vocabulary.copy_matches(path, file_ops.stat_path(path).key)
    assert not vocabulary.copy_matches(path, (0, 0, 0))
    assert vocabulary.copy_matches(path.parent / "other.json", (0, 0, 0))  # a copy of another path is not this file's
    vocabulary.clear_file_copy()
    assert vocabulary.followed_path() is None


# ----------------------------------------------------------------------------------------------------------------
# The write paths of `_transact`: the in-lock check (M1), the read-back, the key
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.no_base_failure(
    reason="the base never reloads a copy, so no reload can interleave with a write there and its copy equals its file by construction; this pins the new rule that the key advances only when the copy still matched at the registry step (D6 M1)"
)
def test_a_reload_between_the_writers_read_and_its_registry_step_leaves_no_stale_copy_m1(path, owner, monkeypatch):
    """D6 M1: a hand edit lands after the writer read the file, a reader thread reloads it into the copy, then the writer's
    replace lands. The writer must not advance the key (the copy then holds the hand edit the file lost): the next read reloads."""
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    real_write = vocabulary_file.write_json

    def interleaved(p, data, **kwargs):
        _hand_edit(p, [_entry("one"), _entry("hand edit")])  # H lands after the writer's read, before its write
        reader = threading.Thread(target=lambda: vocabulary.get("one"), daemon=True)  # another thread's read reloads H
        reader.start()
        reader.join(10)
        assert not reader.is_alive()
        return real_write(p, data, **kwargs)  # the writer's data, derived from the earlier read, replaces the file

    monkeypatch.setattr(vocabulary_file, "write_json", interleaved)
    results = vocabulary_file.add(path, [("written", None)])
    monkeypatch.setattr(vocabulary_file, "write_json", real_write)
    assert results[0].changed
    assert vocabulary.get("one") is not None  # this read reloads when the copy is behind the file
    assert _registry_extensions() == set(_file_names(path)), "the copy holds an entry the file lacks"


@pytest.mark.no_base_failure(
    reason="the base never reloads a copy, so no reload can interleave with a write there and its registry holds the written name by construction; this pins that the registry step is skipped, not only the key, when a reload got in first"
)
def test_a_reload_after_the_write_and_before_its_registry_step_does_not_leave_the_written_name_in_the_copy(path, owner, monkeypatch):
    """Review 1 of the code: the hand edit lands after the writer's file write and removes the written name; a reader thread reloads
    that file (and notes its key, which is the file's real key) before the writer's registry step. The step must not put the name
    back into the copy: nothing would reload it while the file's key stays the same."""
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    real_write = vocabulary_file.write_json

    def interleaved(p, data, **kwargs):
        ok = real_write(p, data, **kwargs)  # the writer's file now holds "written"
        _hand_edit(p, [_entry("one")])  # then a hand edit takes it out again
        reader = threading.Thread(target=lambda: vocabulary.get("one"), daemon=True)  # a reader's reload sees only the hand edit
        reader.start()
        reader.join(10)
        assert not reader.is_alive()
        return ok

    monkeypatch.setattr(vocabulary_file, "write_json", interleaved)
    vocabulary_file.add(path, [("written", None)])
    monkeypatch.setattr(vocabulary_file, "write_json", real_write)
    assert _file_names(path) == ["one"]
    assert vocabulary.get("written") is None
    assert _registry_extensions() == {"one"}, "the copy holds a name the file lacks under the file's current key"


def test_a_hand_edit_between_the_pre_lock_check_and_the_in_lock_read_does_not_advance_the_key_and_loses_nothing(path, owner, monkeypatch):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    real_backup_count = vocabulary_file.backup_count

    def late(p):
        _hand_edit(path, [_entry("one"), _entry("late hand edit")])  # after the check before the lock, before the read
        return real_backup_count(p)

    monkeypatch.setattr(vocabulary_file, "backup_count", late)
    results = vocabulary_file.add(path, [("written", None)])
    monkeypatch.setattr(vocabulary_file, "backup_count", real_backup_count)
    assert results[0].changed
    assert _file_names(path) == ["one", "late hand edit", "written"]  # the write was made on what the lock read
    assert vocabulary.get("one") is not None
    assert _registry_extensions() == {"one", "late hand edit", "written"}


def test_a_read_back_that_raises_leaves_the_write_standing_and_the_next_read_reloads(path, owner, monkeypatch):
    _hand_edit(path, [_entry("one")])
    persona_loader.load_persona_vocabulary(path)
    real_read = vocabulary_file.read
    state = {"fail": True}

    def flaky(p, **kwargs):
        if state["fail"] and kwargs.get("recover") is json_io.Recovery.RAISE:
            raise OSError("read-back failed")
        return real_read(p, **kwargs)

    monkeypatch.setattr(vocabulary_file, "read", flaky)
    result = vocabulary_file.add(path, [("two", None)])[0]
    state["fail"] = False
    assert result.outcome == "used" and result.changed
    assert _file_names(path) == ["one", "two"]
    assert vocabulary.get("two") is not None
    _hand_edit(path, [_entry("one"), _entry("two"), _entry("hand")])  # the copy still follows the file after the failed read-back
    assert vocabulary.get("hand") is not None


# ----------------------------------------------------------------------------------------------------------------
# json_io: the key of the bytes read
# ----------------------------------------------------------------------------------------------------------------


def test_a_read_returns_the_key_of_the_bytes_it_read_not_of_a_later_stat(path):
    _hand_edit(path, [_entry("one")])
    key_before = file_ops.stat_path(path).key
    res = vocabulary_file.read(path, recover=json_io.Recovery.RAISE)
    _hand_edit(path, [_entry("one"), _entry("changed after the read")])
    assert res.sig == key_before
    assert res.sig != file_ops.stat_path(path).key


def test_an_absent_file_reads_with_no_key(path):
    res = vocabulary_file.read(path, recover=json_io.Recovery.RAISE)
    assert res.outcome == "missing" and res.sig is None


def test_the_registry_test_only_load_registers_what_the_file_holds_exactly(path):
    _hand_edit(path, [_entry("one")])
    vocabulary.register(Emotion("stray", "d", "persona_extension", 1.0))
    count, anomaly = vocabulary_file.load(path)
    assert anomaly is None and count == 2  # one added, the stray removed
    assert _registry_extensions() == {"one"}
