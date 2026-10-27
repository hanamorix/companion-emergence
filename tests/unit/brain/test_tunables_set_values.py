"""``tunables.set_values`` and the hardened reader (#529: C7, C9, C36, C55).

``set_values`` edits ``tunables.toml`` IN PLACE: only the value text of the named entries changes. These tests pin that on
hand-edited layouts (the golden-file half of C7), the reset rule, ``only_if_default``, the refusal on an unreadable file, the
fsync-before-replace order, and the reader's live, lock-free, cross-process behaviour.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.tunables_file import FILE_NAME, read_values, write_text, write_values


@pytest.fixture()
def t(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as mod

    mod._reset_for_tests()
    yield mod
    mod._reset_for_tests()


def _file(tmp_path: Path) -> Path:
    return tmp_path / FILE_NAME


def _bytes(tmp_path: Path) -> bytes:
    return _file(tmp_path).read_bytes()


# ---------------------------------------------------------------------------
# C7: what the writer changes and what it leaves alone
# ---------------------------------------------------------------------------

HAND_EDITED = (
    "# My own notes. Keep these.\n"
    "\n"
    "# The live chat turn.\n"
    '# default: "claude-sonnet-5"\n'
    '"model.interactive_chat" = "claude-sonnet-5"  # my note stays\n'
    "\n"
    "# I moved this one up\n"
    "# default: 600.0\n"
    '"chat.idle_lull_seconds" = 600.0\n'
    "\n"
    '"some.unknown.key" = "keep me"\n'
    "unquoted.dotted = 5   # also kept\n"
    "\n"
    "multi = [\n"
    "  1,\n"
    "  2,\n"
    "]\n"
)


def test_only_the_named_value_changes_every_other_byte_is_identical(t, tmp_path) -> None:
    t.register("model.interactive_chat", "claude-sonnet-5")
    t.register("chat.idle_lull_seconds", 600.0)
    write_text(tmp_path, HAND_EDITED)
    assert t.set_values({"model.interactive_chat": "claude-opus-5"}) == {"model.interactive_chat": "claude-opus-5"}
    expect = HAND_EDITED.replace('= "claude-sonnet-5"  # my note stays', '= "claude-opus-5"  # my note stays')
    assert _bytes(tmp_path) == expect.encode()


def test_a_crlf_file_with_a_bom_keeps_both_and_added_lines_use_crlf(t, tmp_path) -> None:
    t.register("a", 1)
    raw = b'\xef\xbb\xbf# head\r\n\r\n# default: 1\r\n"a" = 1 # x\r\n"b" = 2\r\n'
    write_text(tmp_path, raw)
    t.set_values({"a": 7, "c": 3})
    out = _bytes(tmp_path)
    assert out.startswith(b'\xef\xbb\xbf# head\r\n\r\n# default: 1\r\n"a" = 7 # x\r\n"b" = 2\r\n')
    assert out.replace(b"\r\n", b"").count(b"\n") == 0  # no bare LF anywhere
    assert read_values(tmp_path) == {"a": 7, "b": 2, "c": 3}


def test_a_value_with_hash_quote_backslash_and_equals_is_edited_without_touching_the_trailing_comment(t, tmp_path) -> None:
    write_text(tmp_path, '"k" = "a # b \\" \\\\ = c"  # keep = # this\n')
    t.set_values({"k": "new"})
    assert _bytes(tmp_path) == b'"k" = "new"  # keep = # this\n'


@pytest.mark.parametrize("line", ["'a.b' = 1", "a.b = 1", '"a"."b" = 1', '"a.b"=1', "  a . b  =  1  "])
def test_every_spelling_is_edited_in_place_and_never_duplicated(t, tmp_path, line) -> None:
    t.register("a.b", 1)
    write_text(tmp_path, f"{line}\n")
    t.set_values({"a.b": 2})
    assert _bytes(tmp_path) == f"{line}\n".replace("1", "2").encode()
    t.sync_file()
    from brain.utils import toml_io

    doc = toml_io.TomlDocument.from_bytes(_bytes(tmp_path))
    assert doc.values() == {"a.b": 2}
    assert doc.state("a.b") == "single"  # present in any spelling counts as present: no second copy was appended


def test_an_entry_inside_a_users_table_is_edited_in_place(t, tmp_path) -> None:
    t.register("a.b", 1)
    write_text(tmp_path, "[a]\nb = 1  # c\n")
    t.set_values({"a.b": 9})
    assert _bytes(tmp_path) == b"[a]\nb = 9  # c\n"


def test_a_key_in_two_spellings_makes_the_writer_raise_naming_the_key_and_leaves_both_lines(t, tmp_path) -> None:
    t.register("a.b", 1)
    text = '"a.b" = 1\na.b = 2\n'
    write_text(tmp_path, text)
    with pytest.raises(t.TunablesWriteError) as err:
        t.set_values({"a.b": 3})
    assert "a.b" in str(err.value)
    assert _bytes(tmp_path) == text.encode()


def test_a_multi_line_value_makes_the_writer_raise_naming_the_key(t, tmp_path) -> None:
    t.register("multi", [1, 2])
    write_text(tmp_path, HAND_EDITED)
    with pytest.raises(t.TunablesWriteError) as err:
        t.set_values({"multi": [3]})
    assert "multi" in str(err.value)
    assert _bytes(tmp_path) == HAND_EDITED.encode()
    assert t.get_tunable("multi", [0]) == [1, 2]  # still read normally


def test_none_resets_the_entry_to_the_registered_default_and_keeps_the_entry(t, tmp_path) -> None:
    t.register("a", 5)
    write_text(tmp_path, "# default: 5\n\"a\" = 9\n\"b\" = 2\n")
    assert t.set_values({"a": None}) == {"a": 5}
    assert _bytes(tmp_path) == b'# default: 5\n"a" = 5\n"b" = 2\n'


def test_none_for_an_unregistered_key_raises_value_error(t, tmp_path) -> None:
    write_text(tmp_path, '"x" = 1\n')
    with pytest.raises(ValueError, match="x"):
        t.set_values({"x": None})
    assert _bytes(tmp_path) == b'"x" = 1\n'


def test_an_equal_value_writes_nothing_and_does_not_touch_the_file(t, tmp_path) -> None:
    t.register("a", 5)
    path = write_text(tmp_path, '"a" = 5\n')
    os.utime(path, ns=(1_700_000_000_000_000_000,) * 2)
    before = path.stat().st_mtime_ns
    assert t.set_values({"a": 5}) == {}
    assert path.stat().st_mtime_ns == before


def test_reset_of_a_value_that_already_equals_the_default_writes_nothing_and_does_not_create_the_file(t, tmp_path) -> None:
    t.register("a", 5)
    assert t.set_values({"a": None}) == {}
    assert not _file(tmp_path).exists()
    assert t.set_values({"a": 5}) == {}
    assert not _file(tmp_path).exists()


def test_only_if_default_writes_only_keys_whose_value_still_equals_the_default(t, tmp_path) -> None:
    t.register("a", 1)
    t.register("b", 1)
    t.register("c", 1)
    write_text(tmp_path, '"a" = 5\n"b" = 1\n')  # a was changed by the user; b is untouched; c is absent
    written = t.set_values({"a": 2, "b": 3, "c": 4}, only_if_default=True)
    assert written == {"b": 3, "c": 4}
    assert read_values(tmp_path) == {"a": 5, "b": 3, "c": 4}


def test_only_if_default_treats_a_typed_default_equal_int_as_default(t, tmp_path) -> None:
    t.register("a", 600.0)
    write_text(tmp_path, '"a" = 600\n')
    assert t.set_values({"a": 30.0}, only_if_default=True) == {"a": 30.0}


def test_an_invalid_text_in_a_model_key_survives_a_write_of_other_keys(t, tmp_path) -> None:
    write_text(tmp_path, '"model.interactive_chat" = "banana"\n')
    t.set_values({"other.key": 3})
    assert read_values(tmp_path) == {"model.interactive_chat": "banana", "other.key": 3}
    assert _bytes(tmp_path).startswith(b'"model.interactive_chat" = "banana"\n')


def test_a_zero_byte_file_is_no_values_for_the_reader_and_the_writer(t, tmp_path) -> None:
    _file(tmp_path).write_bytes(b"")
    assert t.get_tunable("a", 7) == 7
    assert t.set_values({"a": 8}) == {"a": 8}
    assert read_values(tmp_path) == {"a": 8}


@pytest.mark.parametrize(
    ("content", "line"),
    [
        (b"a = 1\nb = \n", 2),
        (b'k = 1\n\nk = 2\n', 3),
        (b"a = 1\n[t\nb = 2\n", 2),
        (b'a = 1\nb = "caf\xe9"\n', 2),
    ],
)
def test_a_damaged_file_raises_naming_file_line_and_column_and_is_not_modified(t, tmp_path, content, line) -> None:
    _file(tmp_path).write_bytes(content)
    mtime = _file(tmp_path).stat().st_mtime_ns
    with pytest.raises(t.TunablesWriteError) as err:
        t.set_values({"a": 1})
    text = str(err.value)
    assert FILE_NAME in text and f"line {line}" in text and "column" in text
    assert _bytes(tmp_path) == content
    assert _file(tmp_path).stat().st_mtime_ns == mtime
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name or "bak" in p.name or p.name.endswith(".tmp")]


def test_an_unreadable_file_raises_and_is_not_modified(t, tmp_path, monkeypatch) -> None:
    write_text(tmp_path, '"a" = 1\n')
    before = _bytes(tmp_path)

    def boom(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("brain.utils.toml_io._read_bytes", boom)
    with pytest.raises(t.TunablesWriteError):
        t.set_values({"a": 2})
    monkeypatch.undo()
    assert _bytes(tmp_path) == before


def test_an_absent_file_is_created_by_the_sync_pass_then_edited(t, tmp_path) -> None:
    t.register("reg.one", 1, description="One thing.")
    t.register("reg.two", "x", description="Another thing.")
    assert t.set_values({"reg.one": 2}) == {"reg.one": 2}
    assert read_values(tmp_path) == {"reg.one": 2, "reg.two": "x"}  # the other registered key got its default
    text = _bytes(tmp_path).decode()
    assert "# default: 1\n\"reg.one\" = 2" in text and "# One thing." in text


def test_the_write_is_fsynced_before_the_replace(t, tmp_path, monkeypatch) -> None:
    from brain.utils import atomic_write, toml_io

    t.register("a", 1)
    write_text(tmp_path, '"a" = 1\n')
    order: list[str] = []
    real_fsync, real_replace = os.fsync, atomic_write._replace_with_retry
    monkeypatch.setattr(toml_io.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(atomic_write, "_replace_with_retry", lambda s, d: (order.append("replace"), real_replace(s, d))[1])
    t.set_values({"a": 2})
    assert order[:2] == ["fsync", "replace"]  # the data is on disk before the replace; the folder fsync comes after


def test_a_failed_verify_before_the_replace_leaves_the_file_unchanged(t, tmp_path, monkeypatch) -> None:
    from brain.utils import toml_io

    t.register("a", 1)
    write_text(tmp_path, '"a" = 1\n')
    monkeypatch.setattr(toml_io.TomlDocument, "verify", lambda self, expected: False)
    with pytest.raises(t.TunablesWriteError):
        t.set_values({"a": 2})
    assert _bytes(tmp_path) == b'"a" = 1\n'


def test_a_reader_polling_during_many_writes_never_sees_a_torn_or_default_value(t, tmp_path) -> None:
    """Replacement is atomic: an unchanged key never flickers to its default while another key is written many times."""
    write_text(tmp_path, '"keep" = 7\n')
    stop = threading.Event()
    seen: list[object] = []

    def reader() -> None:
        while not stop.is_set():
            seen.append(t.get_tunable("keep", 0))

    th = threading.Thread(target=reader)
    th.start()
    try:
        for i in range(60):
            t.set_values({"moving": i})
    finally:
        stop.set()
        th.join()
    assert seen and set(seen) == {7}


def test_unregistered_keys_and_user_comments_are_untouched_by_a_write_of_another_key(t, tmp_path) -> None:
    t.register("a", 1)
    text = '# mine\n"stranger" = "keep"\n# default: 1\n"a" = 1\n'
    write_text(tmp_path, text)
    t.set_values({"a": 2})
    assert _bytes(tmp_path) == text.replace('"a" = 1', '"a" = 2').encode()


# ---------------------------------------------------------------------------
# C9: live, lock-free, and not stale across processes
# ---------------------------------------------------------------------------


def test_a_write_is_visible_in_process_at_once(t) -> None:
    assert t.get_tunable("a", 1) == 1
    t.set_values({"a": 2})
    assert t.get_tunable("a", 1) == 2


def test_two_contents_with_identical_mtime_ns_are_both_seen(t, tmp_path) -> None:
    """The cache key is (mtime_ns, size, inode): a coarse timestamp tick cannot hide a second write."""
    path = write_values(tmp_path, {"a": 1})
    fixed = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed, fixed))
    assert t.get_tunable("a", 0) == 1
    path.write_text('"a" = 22\n', encoding="utf-8")  # a different size
    os.utime(path, ns=(fixed, fixed))
    assert path.stat().st_mtime_ns == fixed
    assert t.get_tunable("a", 0) == 22


def test_same_size_replacement_with_identical_mtime_ns_is_seen_through_the_inode(t, tmp_path) -> None:
    path = write_values(tmp_path, {"a": 1})
    fixed = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed, fixed))
    assert t.get_tunable("a", 0) == 1
    other = tmp_path / "other.toml"
    other.write_text('"a" = 2\n', encoding="utf-8")  # same length, new inode
    os.utime(other, ns=(fixed, fixed))
    os.replace(other, path)
    assert t.get_tunable("a", 0) == 2


def test_a_second_process_sees_a_write_made_after_it_started(t, tmp_path, repo_root) -> None:
    # The child polls for up to 20 s and stops at the first sight of the value, so the deadline only matters when the write
    # is never seen; it is long so that a loaded host (a slow interpreter start) cannot miss it.
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from brain import tunables\n"
        "print('READY', flush=True)\n"
        "end = time.monotonic() + 20.0\n"
        "while time.monotonic() < end:\n"
        "    if tunables.get_tunable('a', 1) == 5:\n"
        "        print('SEEN'); break\n"
        "    time.sleep(0.01)\n"
    )
    env = {**os.environ, "KINDLED_HOME": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    t.set_values({"a": 5})
    out, _ = proc.communicate(timeout=60)  # longer than the child's own 20 s, so a missing write fails the assertion below
    assert "SEEN" in out


# ---------------------------------------------------------------------------
# C36: I/O failures are not cached, content failures are
# ---------------------------------------------------------------------------


def test_an_io_failure_is_not_cached_and_the_previous_values_stay_in_force(t, tmp_path, monkeypatch) -> None:
    path = write_values(tmp_path, {"a": 1})
    assert t.get_tunable("a", 0) == 1
    path.write_text('"a" = 22\n', encoding="utf-8")  # a change the reader must pick up
    from brain.utils import toml_io

    real = toml_io._read_bytes
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("busy")
        return real(p)

    monkeypatch.setattr(toml_io, "_read_bytes", flaky)
    assert t.get_tunable("a", 0) == 1  # the failed read did not wipe the values
    assert t.get_tunable("a", 0) == 22  # and was not cached: the next call re-read the file


def test_a_content_failure_is_cached_once_and_leaves_the_notice(t, tmp_path, monkeypatch) -> None:
    write_text(tmp_path, "a = = 1\n")
    from brain.utils import toml_io

    real = toml_io.read_toml_result
    reads = {"n": 0}

    def counting(*args, **kwargs):
        reads["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(toml_io, "read_toml_result", counting)
    assert t.get_tunable("a", 5) == 5
    assert t.get_tunable("a", 5) == 5
    assert reads["n"] == 1
    notice = t.current_notice()
    assert notice is not None and notice["kind"] == "unreadable" and notice["line"] == 1


# ---------------------------------------------------------------------------
# Non-re-entrancy is a documented fact the callers rely on (C33, second test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["lone \ud800 surrogate", object(), {"a": 1}])
def test_a_value_the_file_cannot_hold_raises_the_documented_tunables_write_error(t, tmp_path, bad) -> None:
    """The docstring promises ``TunablesWriteError`` for every refusal; the pre-check used to leak ``TomlFileError`` (rev-4 S6)."""
    from brain.utils import toml_io

    write_text(tmp_path, 'a = 1\n')
    with pytest.raises(t.TunablesWriteError) as err:
        t.set_values({"a": bad})
    assert not isinstance(err.value, toml_io.TomlFileError)
    assert _bytes(tmp_path) == b"a = 1\n"


def test_a_registered_key_reset_to_a_default_the_file_cannot_hold_also_raises_tunables_write_error(t, tmp_path) -> None:
    t.register("odd", object())
    with pytest.raises(t.TunablesWriteError):
        t.set_values({"odd": None})
    assert not _file(tmp_path).exists()


def test_set_values_writes_through_a_symlinked_settings_file(t, tmp_path) -> None:
    """A dotfile manager's link stays a link; the target holds the change (rev-4 review bycatch)."""
    t.register("a", 1, description="A.")
    target_dir = tmp_path / "dotfiles"
    target_dir.mkdir()
    target = target_dir / "tunables.toml"
    target.write_bytes(b'# default: 1\n"a" = 1\n')
    try:
        _file(tmp_path).symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    assert t.set_values({"a": 5}) == {"a": 5}
    assert _file(tmp_path).is_symlink()
    assert target.read_bytes() == b'# default: 1\n"a" = 5\n'
    t.sync_file()
    assert _file(tmp_path).is_symlink() and read_values(tmp_path) == {"a": 5}


def test_set_values_does_not_reacquire_a_lock_its_caller_holds(t, tmp_path, monkeypatch) -> None:
    """A nested acquire of the same sidecar is NOT granted (the lock is not re-entrant): a caller holding the tunables
    lock gets a TunablesWriteError from the bounded wait instead of hanging. Nothing in the product nests."""
    from brain import dev_constants
    from brain.utils.file_lock import file_lock

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    with file_lock(_file(tmp_path)):
        with pytest.raises(t.TunablesWriteError):
            t.set_values({"a": 1})
    assert not _file(tmp_path).exists()
