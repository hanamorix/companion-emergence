"""The overlay activation hook (#286 §3.4). The hook must never import `brain`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

from brain.update import overlay_hook


def test_home_matches_brain_paths_for_kindled_home(tmp_path, monkeypatch):
    from brain.paths import get_home

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "k"))
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    assert overlay_hook.home() == get_home()


def test_home_matches_brain_paths_for_deprecated_nellbrain_home(tmp_path, monkeypatch):
    from brain.paths import get_home

    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "n"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert overlay_hook.home() == get_home()


def test_home_matches_brain_paths_for_the_platform_default(monkeypatch):
    from brain.paths import get_home

    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    assert overlay_hook.home() == get_home()


def _world(tmp_path, *, active_bundle_id="bundle-1", site_bundle_id="bundle-1", make_dir=True, state=None):
    site = tmp_path / "site"
    site.mkdir()
    overlay_hook.install_hook(site, site_bundle_id)
    (site / "demo_pkg").mkdir()
    (site / "demo_pkg" / "__init__.py").write_text("WHERE = 'bundle'\n", encoding="utf-8")
    home = tmp_path / "home"
    root = home / "brain-overlay"
    folder = root / "aaaa"
    if make_dir:
        (folder / "demo_pkg").mkdir(parents=True)
        (folder / "demo_pkg" / "__init__.py").write_text("WHERE = 'overlay'\n", encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    if state is None:
        state = json.dumps({"active": {"dir": "aaaa", "commit": "c", "brain_version": "0.0.42",
                                       "bundle_id": active_bundle_id}, "previous": None})
    (root / "current.json").write_text(state, encoding="utf-8")
    return site, home


def _where(site, home):
    code = ("import site, sys; site.addsitedir(%r); import demo_pkg; "  # noqa: UP031
            "print(demo_pkg.WHERE, 'brain' in sys.modules)") % str(site)
    env = {**os.environ, "KINDLED_HOME": str(home)}
    env.pop("NELLBRAIN_HOME", None)
    out = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True,
                         encoding="utf-8", env=env, cwd=str(site.parent))
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def test_unload_platformdirs_drops_it_from_sys_modules():
    """After activation, an overlay's own (possibly newer) platformdirs must be able
    to load — so this helper drops what the hook imported. Direct unit test on the
    helper (subprocess env control for the full through-path is awkward: it needs
    HOME/XDG pointed at a temp dir with KINDLED_HOME/NELLBRAIN_HOME both unset, which
    the other tests in this file deliberately avoid)."""
    sys.modules["platformdirs"] = object()
    sys.modules["platformdirs.macos"] = object()
    sys.modules["platformdirs_unrelated"] = object()  # must survive: prefix trap
    try:
        overlay_hook._unload_platformdirs()
        assert "platformdirs" not in sys.modules
        assert "platformdirs.macos" not in sys.modules
        assert "platformdirs_unrelated" in sys.modules
    finally:
        for k in ("platformdirs", "platformdirs.macos", "platformdirs_unrelated"):
            sys.modules.pop(k, None)


def test_active_overlay_wins_and_brain_is_not_imported(tmp_path):
    site, home = _world(tmp_path)
    assert _where(site, home) == ["overlay", "False"]


def test_kindled_no_overlay_env_var_disables_an_active_overlay(tmp_path):
    site, home = _world(tmp_path)
    code = ("import site, sys; site.addsitedir(%r); import demo_pkg; "  # noqa: UP031
            "print(demo_pkg.WHERE, 'brain' in sys.modules)") % str(site)
    env = {**os.environ, "KINDLED_HOME": str(home), "KINDLED_NO_OVERLAY": "1"}
    env.pop("NELLBRAIN_HOME", None)
    out = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True,
                         encoding="utf-8", env=env, cwd=str(site.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["bundle", "False"]


def test_overlay_for_another_bundle_is_ignored(tmp_path):
    site, home = _world(tmp_path, active_bundle_id="some-other-bundle")
    assert _where(site, home) == ["bundle", "False"]


def test_missing_overlay_folder_is_ignored(tmp_path):
    site, home = _world(tmp_path, make_dir=False)
    assert _where(site, home) == ["bundle", "False"]


def test_corrupt_state_is_ignored(tmp_path):
    site, home = _world(tmp_path, state="{not json")
    assert _where(site, home) == ["bundle", "False"]


def test_activation_does_not_load_hashlib_or_shutil(tmp_path):
    """hashlib/shutil load OpenSSL etc. at every interpreter start; activate() itself
    never needs them (only compute_bundle_id/install_hook, run during an update, do).
    """
    site, home = _world(tmp_path)
    code = ("import site, sys; site.addsitedir(%r); "  # noqa: UP031
            "print('hashlib' in sys.modules, 'shutil' in sys.modules)") % str(site)
    env = {**os.environ, "KINDLED_HOME": str(home)}
    env.pop("NELLBRAIN_HOME", None)
    out = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True,
                         encoding="utf-8", env=env, cwd=str(site.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"], out.stdout


def test_activation_imports_only_stdlib_and_platformdirs(tmp_path):
    """Start-up cost (spec §9): the hook runs in every interpreter start."""
    site, home = _world(tmp_path)
    code = ("import sys; before = set(sys.modules); import site; site.addsitedir(%r); "  # noqa: UP031
            "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
            "extra = sorted(m for m in new if m not in sys.stdlib_module_names "
            "and m not in ('platformdirs', '_ce_overlay', '_virtualenv')); print(extra)") % str(site)
    env = {**os.environ, "KINDLED_HOME": str(home)}
    env.pop("NELLBRAIN_HOME", None)
    out = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True,
                         encoding="utf-8", env=env, cwd=str(site.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_module_docstring_documents_the_escape_hatch():
    assert "KINDLED_NO_OVERLAY" in overlay_hook.__doc__


def test_install_hook_writes_the_three_files(tmp_path):
    overlay_hook.install_hook(tmp_path, "abc")
    assert (tmp_path / "_ce_overlay.pth").read_text(encoding="utf-8") == overlay_hook.PTH_LINE
    assert (tmp_path / "_ce_bundle_id").read_text(encoding="utf-8").strip() == "abc"
    assert (tmp_path / "_ce_overlay.py").read_text(encoding="utf-8") == \
        __import__("pathlib").Path(overlay_hook.__file__).read_text(encoding="utf-8")


def test_bundle_id_changes_with_requirements_or_wheel(tmp_path):
    req = tmp_path / "r.txt"
    req.write_text("a==1\n", encoding="utf-8")
    one = overlay_hook.compute_bundle_id(req, "companion_emergence-0.0.42-py3-none-any.whl")
    assert one != overlay_hook.compute_bundle_id(req, "companion_emergence-0.0.43-py3-none-any.whl")
    req.write_text("a==2\n", encoding="utf-8")
    assert one != overlay_hook.compute_bundle_id(req, "companion_emergence-0.0.42-py3-none-any.whl")


def test_dir_that_is_not_a_plain_name_is_ignored(tmp_path):
    outside = tmp_path / "outside"
    (outside / "demo_pkg").mkdir(parents=True)
    (outside / "demo_pkg" / "__init__.py").write_text("WHERE = 'outside'\n", encoding="utf-8")
    for bad in (str(outside), "../outside", "..", ".", "a/b", "a\\b", ""):
        world = tmp_path / f"w{abs(hash(bad))}"
        world.mkdir()
        state = json.dumps({"active": {"dir": bad, "commit": "c", "brain_version": "0.0.42",
                                       "bundle_id": "bundle-1"}, "previous": None})
        site, home = _world(world, state=state)
        assert _where(site, home)[0] == "bundle", bad
