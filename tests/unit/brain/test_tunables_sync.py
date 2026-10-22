"""``tunables.sync_file`` (the boot pass) and the layout of ``tunables.toml`` (#529: C46, C47, C49, C54).

The file is the single source: one flat quoted entry per registered key, value pre-filled, the shipped default in a
``# default:`` comment above it. At every boot the pass adds missing entries and moves a value that still equals the
RECORDED default onto the new default (owner ruling E2: "The bridge updates any value that still matches the old default,
since the user never touched it. Values the user did change are left alone."). It never repairs a file that does not parse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from brain.utils import toml_io
from tests.tunables_file import FILE_NAME, read_values, write_text


@pytest.fixture()
def t(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as mod

    mod._reset_for_tests()
    yield mod
    mod._reset_for_tests()


def _file(home: Path) -> Path:
    return home / FILE_NAME


def _text(home: Path) -> str:
    return _file(home).read_text(encoding="utf-8")


def _age(path: Path) -> int:
    """Set the file's mtime well in the past and return it, so a later write shows as a changed mtime."""
    old = 1_700_000_000_000_000_000
    os.utime(path, ns=(old, old))
    return path.stat().st_mtime_ns


# ---------------------------------------------------------------------------
# C46: the generated layout
# ---------------------------------------------------------------------------


def _register_samples(t) -> dict[str, Any]:
    samples = {
        "area.float": (600.0, "How long to wait, in seconds."),
        "area.int": (20 * 1024 * 1024, "A size in bytes."),
        "area.str": ("claude-sonnet-5", "A model name."),
        "area.flag": (True, "A switch."),
        "area.list": (["png", "jpg"], "File types."),
        "area.tuple": (("a", "b"), "A pair."),
        "area.negative_zero": (-0.0, "A float zero."),
    }
    for key, (default, desc) in samples.items():
        t.register(key, default, description=desc)
    return {k: v[0] for k, v in samples.items()}


def test_sync_on_an_empty_home_writes_every_registered_key_once_with_description_and_default_above(t, tmp_path) -> None:
    defaults = _register_samples(t)
    t.sync_file()
    text = _text(tmp_path)
    lines = text.splitlines()
    for key, default in defaults.items():
        entry = f'{toml_io.dumps_value(key)} = {toml_io.dumps_value(default)}'
        assert lines.count(entry) == 1, key
        i = lines.index(entry)
        assert lines[i - 1] == f"# default: {toml_io.dumps_value(default)}", key
        assert lines[i - 2].startswith("# ") and lines[i - 2].endswith("."), key  # the description
    assert not any(line.lstrip().startswith("[") for line in lines)  # no table header at all
    for gone in ("_readme", "defaults", "overrides"):
        assert gone not in text.replace("# default:", "")
    parsed = toml_io.flatten_toml(toml_io.parse_toml(text))
    assert {k: tuple(v) if isinstance(v, list) and isinstance(defaults[k], tuple) else v for k, v in parsed.items()} == defaults
    assert "—" not in text and "–" not in text  # no em-dash or en-dash anywhere in the generated text


def test_the_generated_file_has_a_header_that_says_not_to_edit_the_default_lines(t, tmp_path) -> None:
    t.register("a", 1, description="A thing.")
    t.sync_file()
    head = _text(tmp_path).split("\n\n", 1)[0]
    assert head.startswith("# ") and "default" in head and "Safe to edit by hand" in head
    assert 'Do not edit the "# default:" lines' in head


def test_entries_are_sorted_by_key_on_creation(t, tmp_path) -> None:
    for key in ("zeta", "alpha", "mid"):
        t.register(key, 1, description="Thing.")
    t.sync_file()
    keys = list(toml_io.parse_toml(_text(tmp_path)))
    assert keys == ["alpha", "mid", "zeta"]


def test_a_second_pass_writes_nothing_bytes_and_mtime_identical(t, tmp_path) -> None:
    _register_samples(t)
    t.sync_file()
    path = _file(tmp_path)
    mtime = _age(path)
    before = path.read_bytes()
    t.sync_file()
    t.sync_file()
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime


def test_no_notice_file_is_written_for_a_healthy_pass(t, tmp_path) -> None:
    t.register("a", 1, description="A thing.")
    t.sync_file()
    assert t.current_notice() is None
    assert not (tmp_path / "tunables.notice.json").exists()


# ---------------------------------------------------------------------------
# C47: the bump rule
# ---------------------------------------------------------------------------


def _entry(value_text: str, default_text: str | None, key: str = "k") -> str:
    comment = f"# default: {default_text}\n" if default_text is not None else ""
    return f'# What it does.\n{comment}"{key}" = {value_text}\n'


def _run(t, tmp_path, text: str, default: Any, desc: str = "What it does.") -> str:
    write_text(tmp_path, text)
    t.register("k", default, description=desc)
    t.sync_file()
    return _text(tmp_path)


def test_a_value_equal_to_the_old_default_follows_the_new_default_value_and_comment(t, tmp_path) -> None:
    out = _run(t, tmp_path, _entry("60.0", "60.0"), 90.0)
    assert out == _entry("90.0", "90.0")


def test_a_value_the_user_changed_is_kept_and_the_comment_is_refreshed(t, tmp_path) -> None:
    out = _run(t, tmp_path, _entry("75.0", "60.0"), 90.0)
    assert out == _entry("75.0", "90.0")


def test_a_deleted_default_comment_keeps_the_value_and_reinserts_the_comment(t, tmp_path) -> None:
    """The comment is the only record of the old default: with it gone, one bump is lost (stated, R16)."""
    out = _run(t, tmp_path, _entry("60.0", None), 90.0)
    assert out == _entry("60.0", "90.0")


def test_an_unparsable_default_comment_is_replaced_and_the_value_kept(t, tmp_path) -> None:
    out = _run(t, tmp_path, '# What it does.\n# default: ???\n"k" = 60.0\n', 90.0)
    assert out == _entry("60.0", "90.0")


def test_a_blank_line_between_comment_and_entry_inserts_an_extra_comment_and_loses_nothing(t, tmp_path) -> None:
    text = '# default: 60.0\n\n"k" = 60.0\n'
    out = _run(t, tmp_path, text, 90.0)
    assert out == '# default: 60.0\n\n# default: 90.0\n"k" = 60.0\n'


def test_numerically_equal_int_and_float_count_as_equal(t, tmp_path) -> None:
    """A user who typed 600 for a float default 600.0 still follows bumps."""
    out = _run(t, tmp_path, _entry("600", "600.0"), 900.0)
    assert out == _entry("900.0", "900.0")


def test_a_bool_never_equals_a_number(t, tmp_path) -> None:
    out = _run(t, tmp_path, _entry("1", "true"), False)
    assert out == _entry("1", "false")  # 1 is not true: the user's value is kept
    out = _run(t, tmp_path, _entry("true", "true"), False)
    assert out == _entry("false", "false")


def test_a_wrong_typed_value_is_kept(t, tmp_path) -> None:
    out = _run(t, tmp_path, _entry('"fast"', "60.0"), 90.0)
    assert out == _entry('"fast"', "90.0")


def test_a_list_default_follows_when_untouched_and_a_tuple_default_is_compared_as_a_list(t, tmp_path) -> None:
    out = _run(t, tmp_path, _entry('["png"]', '["png"]'), ["png", "jpg"])
    assert out == _entry('["png", "jpg"]', '["png", "jpg"]')
    t._reset_for_tests()
    out = _run(t, tmp_path, _entry('["a", "b"]', '["a", "b"]'), ("a", "b", "c"))
    assert out == _entry('["a", "b", "c"]', '["a", "b", "c"]')


def test_a_default_that_changes_type_follows_through_the_same_rule(t, tmp_path) -> None:
    out = _run(t, tmp_path, _entry("60.0", "60.0"), "sixty")
    assert out == _entry('"sixty"', '"sixty"')
    t._reset_for_tests()
    out = _run(t, tmp_path, _entry("75.0", "60.0"), "sixty")  # the old-typed user value differs from the old default: kept
    assert out == _entry("75.0", '"sixty"')


def test_a_default_that_did_not_change_leaves_the_file_alone(t, tmp_path) -> None:
    text = _entry("75.0", "60.0")
    out = _run(t, tmp_path, text, 60.0)
    assert out == text


def test_a_new_key_is_added_with_description_and_default_and_others_stay(t, tmp_path) -> None:
    write_text(tmp_path, '# mine\n"old" = 1\n')
    t.register("new.key", 3, description="A new thing.")
    t.sync_file()
    out = _text(tmp_path)
    assert out.startswith('# mine\n"old" = 1\n')
    assert out.endswith('# A new thing.\n# default: 3\n"new.key" = 3\n')


def test_a_key_missing_from_a_file_that_ends_inside_a_users_table_goes_before_the_first_table(t, tmp_path) -> None:
    """M1: appended at the end it would parse as model."reranker.precision" and never be read."""
    write_text(tmp_path, 'a = 1\n\n[model]\ncompaction = "x"\n')
    t.register("reranker.precision", "high", description="How exact.")
    t.sync_file()
    parsed = toml_io.parse_toml(_text(tmp_path))
    assert parsed["reranker.precision"] == "high"
    assert parsed["model"] == {"compaction": "x"}
    assert t.get_tunable("reranker.precision", "d") == "high"
    mtime = _age(_file(tmp_path))
    before = _file(tmp_path).read_bytes()
    t.sync_file()  # the second pass finds it (read back as top level) and writes nothing
    assert _file(tmp_path).read_bytes() == before and _file(tmp_path).stat().st_mtime_ns == mtime


def test_a_file_already_holding_exactly_the_result_is_not_rewritten(t, tmp_path) -> None:
    text = _entry("90.0", "90.0")
    write_text(tmp_path, text)
    mtime = _age(_file(tmp_path))
    t.register("k", 90.0, description="What it does.")
    t.sync_file()
    assert _file(tmp_path).stat().st_mtime_ns == mtime


def test_a_value_in_a_users_table_or_a_multi_line_value_is_skipped_with_a_log_not_duplicated(t, tmp_path, caplog) -> None:
    text = "k = [\n  1,\n  2,\n]\n"
    write_text(tmp_path, text)
    t.register("k", [1, 2], description="A list.")
    with caplog.at_level("WARNING", logger="brain.tunables"):
        t.sync_file()
    assert _text(tmp_path) == text
    assert any("k" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# C49: keys this process does not register are kept untouched
# ---------------------------------------------------------------------------


def test_an_unregistered_key_and_every_user_comment_are_byte_identical_after_a_pass(t, tmp_path) -> None:
    text = (
        "# my top note\n"
        '"stranger.key" = "keep me"  # stays\n'
        "\n"
        "# default: 60.0\n"
        '"k" = 60.0\n'
        "# trailing note\n"
    )
    out = _run(t, tmp_path, text, 90.0)
    assert out == text.replace("60.0", "90.0").replace('"k" = 90.0', '"k" = 90.0')
    assert '"stranger.key" = "keep me"  # stays\n' in out and "# my top note\n" in out and "# trailing note\n" in out


def test_a_pass_never_deletes_a_key_the_registry_lacks(t, tmp_path) -> None:
    write_text(tmp_path, '"gone.from.code" = 5\n')
    t.register("other", 1, description="Other.")
    t.sync_file()
    assert read_values(tmp_path)["gone.from.code"] == 5


# ---------------------------------------------------------------------------
# C54: an unreadable file is left alone, the notice names the line
# ---------------------------------------------------------------------------

BAD_FILES = {
    "syntax error on line 7": (b"# a\n# b\n\n\"x\" = 1\n\n# note\nthis is = = broken\n\"y\" = 2\n", 7),
    "duplicate key": (b'k = 1\n\nk = 2\n', 3),
    "bad value": (b'a = 1\nb = nope\n', 2),
    "cp1252 byte without a BOM": (b'a = 1\nb = 2\nc = "caf\xe9"\n', 3),
}


@pytest.mark.parametrize("name", sorted(BAD_FILES))
def test_sync_leaves_an_unreadable_file_alone_and_records_the_line(t, tmp_path, name) -> None:
    content, line = BAD_FILES[name]
    path = tmp_path / FILE_NAME
    path.write_bytes(content)
    mtime = path.stat().st_mtime_ns
    t.register("k2", 1, description="Another.")
    t.sync_file()
    assert path.read_bytes() == content and path.stat().st_mtime_ns == mtime
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name or "quarantine" in p.name or ".bak" in p.name]
    notice = t.current_notice()
    assert notice is not None
    assert notice["kind"] == "unreadable" and notice["line"] == line and isinstance(notice["column"], int) and notice["reason"]
    assert t.get_tunable("k2", 5) == 5  # defaults apply while the file is unreadable


def test_the_notice_text_fields_are_clean_and_short(t, tmp_path) -> None:
    (tmp_path / FILE_NAME).write_bytes(b'a = "x\x01y"\n')
    t.sync_file()
    n = t.current_notice()
    assert n is not None and len(n["reason"]) <= 120 and all(ch.isprintable() for ch in n["reason"])


def test_a_bom_file_with_valid_toml_is_read_not_damaged(t, tmp_path) -> None:
    (tmp_path / FILE_NAME).write_bytes(b'\xef\xbb\xbf"k" = 4\r\n')
    t.register("k", 1, description="A thing.")
    assert t.get_tunable("k", 1) == 4
    t.sync_file()
    assert t.current_notice() is None
    data = (tmp_path / FILE_NAME).read_bytes()
    assert data.startswith(b"\xef\xbb\xbf") and b'"k" = 4\r\n' in data  # the BOM and the user's value are kept
    assert data.replace(b"\r\n", b"").count(b"\n") == 0  # the lines the pass added use the file's own CRLF


def test_the_notice_ends_when_the_file_is_fixed(t, tmp_path) -> None:
    path = tmp_path / FILE_NAME
    path.write_bytes(b"a = = 1\n")
    t.sync_file()
    assert t.current_notice() is not None
    path.write_bytes(b"a = 1\n")
    os.utime(path, ns=(1_800_000_000_000_000_000,) * 2)
    assert t.current_notice() is None


def test_the_notice_stays_while_the_file_is_edited_but_still_broken(t, tmp_path) -> None:
    path = tmp_path / FILE_NAME
    path.write_bytes(b"a = = 1\n")
    t.sync_file()
    path.write_bytes(b"a = = 2\n")
    os.utime(path, ns=(1_800_000_000_000_000_000,) * 2)
    assert t.current_notice() is not None


@pytest.mark.parametrize(
    "marker",
    [
        '{"kind": "reset", "kept_as": "tunables.json.corrupt-1", "at": "2026-01-01T00:00:00Z", "overrides_at_heal": {}}',
        '{"kind": "restored", "kept_as": "x", "at": "2026-01-01T00:00:00Z"}',
        '{"kind": "unreadable", "kept_as": null, "at": "2026-01-01T00:00:00Z", "mtime_ns": 1}',
    ],
)
def test_a_pre_upgrade_marker_is_inactive_and_removed_at_boot(t, tmp_path, marker) -> None:
    (tmp_path / "tunables.notice.json").write_text(marker, encoding="utf-8")
    t.register("a", 1, description="A.")
    t.sync_file()
    assert t.current_notice() is None
    assert not (tmp_path / "tunables.notice.json").exists()


def test_a_zero_byte_file_is_filled_by_the_next_pass(t, tmp_path) -> None:
    (tmp_path / FILE_NAME).write_bytes(b"")
    t.register("a", 1, description="A.")
    t.sync_file()
    assert read_values(tmp_path) == {"a": 1}
    assert t.current_notice() is None


def test_sync_writes_no_notice_file_while_holding_the_lock(t, tmp_path, monkeypatch) -> None:
    """The unreadable notice is written AFTER the lock is released (it would otherwise lengthen the lock hold)."""
    (tmp_path / FILE_NAME).write_bytes(b"a = = 1\n")
    seen: list[bool] = []
    real = t._write_marker

    def spy(record):
        from brain.utils.file_lock import file_lock

        with file_lock(tmp_path / FILE_NAME, blocking=False) as got:
            seen.append(bool(got))  # True only if sync_file has already released the tunables lock
        real(record)

    monkeypatch.setattr(t, "_write_marker", spy)
    t.sync_file()
    assert seen == [True]


def test_a_lock_timeout_skips_the_pass_and_writes_nothing(t, tmp_path, monkeypatch, caplog) -> None:
    from brain import dev_constants
    from brain.utils.file_lock import file_lock

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.2)
    t.register("a", 1, description="A.")
    with file_lock(_file(tmp_path)), caplog.at_level("WARNING", logger="brain.tunables"):
        t.sync_file()
    assert not _file(tmp_path).exists()
    assert any("not" in r.getMessage() for r in caplog.records)


def test_sync_never_raises_into_startup(t, tmp_path, monkeypatch) -> None:
    from brain.utils import toml_io as tio

    t.register("a", 1, description="A.")
    monkeypatch.setattr(tio, "write_toml_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    t.sync_file()  # logged and swallowed
