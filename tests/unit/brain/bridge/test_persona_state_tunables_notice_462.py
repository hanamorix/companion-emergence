"""#462: `/persona/state` reports the persisted tunables notice for EVERY persona sharing the home (no bridge-local state).

The notice is written next to tunables.json by whichever bridge healed the file; every other bridge, for every other persona,
reads the same marker on each poll. Synthetic names only; `KINDLED_HOME` is a tmp dir.
"""

from __future__ import annotations

import json

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
    # C462-e
    import brain.tunables as t

    p1, p2 = _personas(home)
    t.write_defaults_section()

    assert _notice(p1) is None and _notice(p2) is None


def test_one_heal_is_reported_for_both_personas(home):
    # C462-e: only `write_defaults_section` ran, once; both personas' state shows it.
    import brain.tunables as t

    p1, p2 = _personas(home)
    (home / "tunables.json").write_bytes(b"{corrupt")
    t.write_defaults_section()
    kept = sorted(home.glob("tunables.json.corrupt-*"))[0].name

    assert _notice(p1) == {"kind": "reset", "kept_as": kept}
    assert _notice(p2) == {"kind": "reset", "kept_as": kept}


def test_a_second_boot_does_not_clear_it_and_polling_never_writes(home):
    # C462-f
    import brain.tunables as t

    p1, p2 = _personas(home)
    (home / "tunables.json").write_bytes(b"{corrupt")
    t.write_defaults_section()
    t.write_defaults_section()
    marker = home / "tunables.notice.json"
    before = (marker.read_bytes(), marker.stat().st_mtime_ns)

    assert _notice(p1) is not None and _notice(p2) is not None
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before


def test_the_user_editing_the_overrides_clears_it_for_both_personas(home):
    # C462-g
    import brain.tunables as t

    p1, p2 = _personas(home)
    (home / "tunables.json").write_bytes(b"{corrupt")
    t.write_defaults_section()

    (home / "tunables.json").write_text(json.dumps({"overrides": {"x": 2}}), encoding="utf-8")

    assert _notice(p1) is None and _notice(p2) is None


def test_the_key_is_the_last_key_of_the_state(home):
    # C462-i
    (p1,) = _personas(home)[:1]

    assert list(build_persona_state(p1))[-1] == "tunables_notice"
