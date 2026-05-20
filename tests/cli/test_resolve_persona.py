"""_resolve_persona_or_exit() — exit hygiene for the new no-default policy."""

from __future__ import annotations

import pytest

from brain.cli import _resolve_persona_or_exit


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    return tmp_path


def test_explicit_arg_returns_verbatim(tmp_home, capsys):
    # No personas installed; explicit arg means resolver doesn't consult
    # the filesystem at all.
    name = _resolve_persona_or_exit("alex")
    assert name == "alex"
    assert capsys.readouterr().err == ""


def test_zero_personas_exits_nonzero_with_creation_hint(tmp_home, capsys):
    (tmp_home / "personas").mkdir()
    with pytest.raises(SystemExit) as exc:
        _resolve_persona_or_exit(None)
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "nell init" in err.lower() or "create" in err.lower()


def test_single_persona_returns_silently(tmp_home, capsys):
    (tmp_home / "personas" / "mira").mkdir(parents=True)
    name = _resolve_persona_or_exit(None)
    assert name == "mira"
    # No noisy output on the silent branch.
    assert capsys.readouterr().err == ""


def test_multiple_personas_exits_with_listing(tmp_home, capsys):
    for n in ["alex", "mira", "nell"]:
        (tmp_home / "personas" / n).mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        _resolve_persona_or_exit(None)
    assert exc.value.code != 0
    err = capsys.readouterr().err
    for n in ["alex", "mira", "nell"]:
        assert n in err
    assert "--persona" in err  # hint to use the flag


def test_invalid_explicit_arg_validates(tmp_home):
    # Grammar violation: spaces forbidden.
    with pytest.raises(SystemExit) as exc:
        _resolve_persona_or_exit("has space")
    assert exc.value.code != 0


def test_explicit_arg_with_no_installed_personas_still_returns(tmp_home, capsys):
    # Explicit arg means "use this name even if not yet installed" —
    # downstream code (e.g., `nell init`) will create the persona dir.
    name = _resolve_persona_or_exit("brand-new")
    assert name == "brand-new"
