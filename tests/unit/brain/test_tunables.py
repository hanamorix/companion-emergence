"""Canaries for the ops-tunables invariants (spec 2026-07-04-ops-tunables-design §5, reshaped for tunables.toml in #529).

Fail-open, hot-reload, and type-guard are load-bearing: a regression here
silently strips or corrupts operational settings on live installs.
"""
from __future__ import annotations

import os
import time

import pytest

from tests.tunables_file import write_text, write_values


@pytest.fixture()
def tunables(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield t
    t._reset_for_tests()


def test_missing_file_returns_default(tunables):
    assert tunables.get_tunable("a.b", 60.0) == 60.0


def test_a_value_in_the_file_wins(tunables, tmp_path):
    write_values(tmp_path, {"a.b": 180.0})
    assert tunables.get_tunable("a.b", 60.0) == 180.0


def test_register_returns_default_and_records(tunables):
    assert tunables.register("a.c", 15.0) == 15.0
    assert tunables._registry["a.c"] == 15.0


def test_register_stores_the_description_beside_the_registry_without_changing_its_shape(tunables):
    assert tunables.register("a.d", 3, description="How many things.") == 3
    assert tunables._registry["a.d"] == 3  # the key -> default dict keeps its shape: other tests read it
    assert tunables._descriptions["a.d"] == "How many things."
    tunables.register("a.e", 4)  # throwaway registrations without a description still work
    assert "a.e" not in tunables._descriptions or tunables._descriptions["a.e"] == ""


def test_the_description_can_be_the_third_positional_argument(tunables):
    tunables.register("a.f", 1.0, "Seconds to wait.")
    assert tunables._descriptions["a.f"] == "Seconds to wait."


def test_fails_open_on_a_damaged_file(tunables, tmp_path):
    (tmp_path / "tunables.toml").write_text("a.b = = 1\n", encoding="utf-8")
    assert tunables.get_tunable("a.b", 60.0) == 60.0  # no raise


def test_type_guard_ignores_wrong_type(tunables, tmp_path):
    write_values(tmp_path, {"a.b": "fast"})
    assert tunables.get_tunable("a.b", 60.0) == 60.0


def test_type_guard_accepts_int_for_float(tunables, tmp_path):
    write_values(tmp_path, {"a.b": 90})
    val = tunables.get_tunable("a.b", 60.0)
    assert val == 90.0 and isinstance(val, float)


def test_type_guard_bool_is_not_int(tunables, tmp_path):
    write_values(tmp_path, {"a.n": True})
    assert tunables.get_tunable("a.n", 5) == 5  # bool must not satisfy int


def test_a_wrong_typed_value_warns_once_and_says_value_not_override(tunables, tmp_path, caplog):
    write_values(tmp_path, {"a.b": "fast"})
    with caplog.at_level("WARNING", logger="brain.tunables"):
        tunables.get_tunable("a.b", 60.0)
        tunables.get_tunable("a.b", 60.0)
    texts = [r.getMessage() for r in caplog.records if "a.b" in r.getMessage()]
    assert len(texts) == 1 and "value" in texts[0] and "override" not in texts[0]


def test_hot_reload_on_mtime_change(tunables, tmp_path):
    write_values(tmp_path, {"a.b": 90.0})
    assert tunables.get_tunable("a.b", 60.0) == 90.0
    write_values(tmp_path, {"a.b": 240.0})
    # ensure mtime actually advances on coarse-granularity filesystems
    os.utime(tmp_path / "tunables.toml", (time.time() + 2, time.time() + 2))
    assert tunables.get_tunable("a.b", 60.0) == 240.0


# --- the reader reads every spelling (C55) -------------------------------------------------------------------------


def test_nested_tables_and_bare_dotted_keys_are_read_as_dotted_keys(tunables, tmp_path):
    write_text(tmp_path, '[model]\ncompaction = "x"\n\n[a.b]\nc = 5\n')
    assert tunables.get_tunable("model.compaction", "d") == "x"
    assert tunables.get_tunable("a.b.c", 1) == 5
    write_text(tmp_path, 'model.compaction = "y"\n')
    os.utime(tmp_path / "tunables.toml", (time.time() + 3, time.time() + 3))
    assert tunables.get_tunable("model.compaction", "d") == "y"


def test_a_key_in_two_spellings_reads_as_the_flat_quoted_one(tunables, tmp_path):
    write_text(tmp_path, '"a.b" = 1\n[a]\nb = 2\n')
    assert tunables.get_tunable("a.b", 0) == 1


def test_a_list_read_for_a_tuple_default_is_a_tuple(tunables, tmp_path):
    write_text(tmp_path, '"files.image_types" = ["png", "jpg"]\n')
    got = tunables.get_tunable("files.image_types", ("gif",))
    assert got == ("png", "jpg") and isinstance(got, tuple)


def test_a_list_default_round_trips_as_a_list(tunables, tmp_path):
    write_text(tmp_path, '"files.image_types" = ["png"]\n')
    got = tunables.get_tunable("files.image_types", ["gif"])
    assert got == ["png"] and isinstance(got, list)


def test_get_raw_returns_the_value_as_the_file_holds_it_with_no_type_guard(tunables, tmp_path):
    write_values(tmp_path, {"a.b": "fast", "model.x": "banana"})
    assert tunables.get_raw("a.b") == "fast"
    assert tunables.get_raw("model.x") == "banana"
    assert tunables.get_raw("absent") is None


def test_file_mtime_is_the_files_modification_time_or_none(tunables, tmp_path):
    assert tunables.file_mtime() is None
    path = write_values(tmp_path, {"a": 1})
    assert tunables.file_mtime() == path.stat().st_mtime


def test_no_names_of_the_old_json_file_survive(tunables):
    for gone in ("set_overrides", "get_override", "overrides_mtime", "write_defaults_section", "_README", "_overrides_is_object",
                 "_load_overrides_locked", "_record_heal", "_same_content"):
        assert not hasattr(tunables, gone), gone
