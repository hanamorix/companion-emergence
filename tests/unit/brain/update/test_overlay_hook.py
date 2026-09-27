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


def test_active_overlay_wins_and_brain_is_not_imported(tmp_path):
    site, home = _world(tmp_path)
    assert _where(site, home) == ["overlay", "False"]


def test_overlay_for_another_bundle_is_ignored(tmp_path):
    site, home = _world(tmp_path, active_bundle_id="some-other-bundle")
    assert _where(site, home) == ["bundle", "False"]


def test_missing_overlay_folder_is_ignored(tmp_path):
    site, home = _world(tmp_path, make_dir=False)
    assert _where(site, home) == ["bundle", "False"]


def test_corrupt_state_is_ignored(tmp_path):
    site, home = _world(tmp_path, state="{not json")
    assert _where(site, home) == ["bundle", "False"]


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
