"""#462 reshaped (#529): `/persona/state` reports the persisted tunables notice for EVERY persona sharing the home (no bridge-local state).

The notice is written next to tunables.toml by whichever process found the file unreadable; every other bridge, for every other
persona, reads the same marker on each poll. It carries the failing line, column and reason (owner ruling E9). Synthetic names
only; `KINDLED_HOME` is a tmp dir.
"""

from __future__ import annotations

import os

import pytest

from brain.bridge.persona_state import build_persona_state


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield tmp_path
    t._reset_for_tests()


def _personas(home):
    out = []
    for name in ("canary", "wren"):
        d = home / "personas" / name
        d.mkdir(parents=True)
        out.append(d)
    return out


def _notice(persona_dir):
    return build_persona_state(persona_dir)["tunables_notice"]


def test_a_healthy_home_reports_no_notice_for_both_personas(home):
    import brain.tunables as t

    p1, p2 = _personas(home)
    t.sync_file()

    assert _notice(p1) is None and _notice(p2) is None


def test_one_unreadable_file_is_reported_for_both_personas_with_the_line(home):
    # only `sync_file` ran, once; both personas' state shows it.
    import brain.tunables as t

    p1, p2 = _personas(home)
    (home / "tunables.toml").write_text("a = 1\n\nb = = 2\n", encoding="utf-8")
    t.sync_file()

    for p in (p1, p2):
        n = _notice(p)
        assert n["kind"] == "unreadable" and n["line"] == 3 and isinstance(n["column"], int) and n["reason"]
        assert set(n) == {"kind", "line", "column", "reason"}  # no kept_as: nothing is ever set aside now


def test_a_second_boot_does_not_clear_it_and_polling_never_writes(home):
    import brain.tunables as t

    p1, p2 = _personas(home)
    (home / "tunables.toml").write_text("b = = 2\n", encoding="utf-8")
    t.sync_file()
    t.sync_file()
    marker = home / "tunables.notice.json"
    before = (marker.read_bytes(), marker.stat().st_mtime_ns)

    assert _notice(p1) is not None and _notice(p2) is not None
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before


def test_the_user_fixing_the_file_clears_it_for_both_personas(home):
    import brain.tunables as t

    p1, p2 = _personas(home)
    path = home / "tunables.toml"
    path.write_text("b = = 2\n", encoding="utf-8")
    t.sync_file()

    path.write_text('"x" = 2\n', encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    assert _notice(p1) is None and _notice(p2) is None


def test_the_key_is_the_last_key_of_the_state(home):
    (p1,) = _personas(home)[:1]

    assert list(build_persona_state(p1))[-1] == "tunables_notice"
