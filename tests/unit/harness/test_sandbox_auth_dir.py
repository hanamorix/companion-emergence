"""#236 — the sandbox's ``CLAUDE_CONFIG_DIR`` must be a dir the CLI can actually authenticate in.

An explicit ``CLAUDE_CONFIG_DIR`` does NOT fall back to the default Keychain entry: the CLI keys a
per-dir credential (a hash of the dir path), so a fresh tempdir is always "Not logged in". The fix is a
stable, harness-owned config dir the developer logs into once (``scripts/setup_harness_claude_login.sh``
writes ``.harness-authed`` on success) that the sandbox reuses when present.
"""

from __future__ import annotations

import importlib
import os
import warnings
from pathlib import Path

import pytest

sandbox_mod = importlib.import_module("tests.harness.sandbox")
sandbox = sandbox_mod.sandbox


def _fake_home_without_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    fake_home = tmp_path / "fake-home"
    (fake_home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    return fake_home


def test_sandbox_uses_harness_authed_dir_when_marker_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_home_without_credentials(monkeypatch, tmp_path)
    authed = tmp_path / "harness-claude-config"
    authed.mkdir()
    (authed / ".harness-authed").write_text("ok")
    monkeypatch.setattr(sandbox_mod, "_harness_config_dir", lambda: authed)

    with sandbox() as sb:
        assert sb.claude_config_dir == authed
        assert os.environ["CLAUDE_CONFIG_DIR"] == str(authed)
        assert sb.auth_source == "harness-dir"
        # Carrier B (synthetic oauthAccount) still applies to the authed dir.
        assert (authed / ".claude.json").is_file()
    # The authed dir is persistent — teardown must not remove it.
    assert authed.is_dir()


def test_sandbox_without_marker_or_credentials_file_is_unauthenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No credentials file and no authed harness dir ⇒ honestly 'unauthenticated' on EVERY platform.

    The old darwin branch returned 'keychain-or-inherited' on the false premise that a fresh explicit
    ``CLAUDE_CONFIG_DIR`` still reads the default Keychain entry."""
    _fake_home_without_credentials(monkeypatch, tmp_path)
    monkeypatch.setattr(sandbox_mod, "_harness_config_dir", lambda: tmp_path / "absent")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sandbox() as sb:
            assert sb.auth_source == "unauthenticated"
    assert any("setup_harness_claude_login" in str(w.message) for w in caught)


def test_default_harness_config_dir_is_outside_every_guarded_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The authed dir accumulates CLI session files across runs, so it must never sit under a
    leak-guarded root or every live run would trip the guard."""
    _fake_home_without_credentials(monkeypatch, tmp_path)
    d = sandbox_mod._harness_config_dir().resolve()
    for root in sandbox_mod._guarded_roots():
        r = root.resolve()
        assert d != r and r not in d.parents, f"{d} is under guarded root {r}"


def test_live_example_skips_when_unauthenticated() -> None:
    from types import SimpleNamespace

    from tests.harness.examples import test_generic_run as example

    with pytest.raises(pytest.skip.Exception) as info:
        example._skip_unless_authed(SimpleNamespace(auth_source="unauthenticated"))
    assert "setup_harness_claude_login" in str(info.value)
    example._skip_unless_authed(SimpleNamespace(auth_source="harness-dir"))  # no raise
    example._skip_unless_authed(SimpleNamespace(auth_source="credentials-file"))


def test_darwin_never_trusts_a_copied_credentials_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On macOS the CLI's credential is a per-dir Keychain entry; a copied ``.credentials.json``
    (often a stale leftover) does not authenticate a fresh dir, so it must not be reported as auth."""
    fake_home = _fake_home_without_credentials(monkeypatch, tmp_path)
    (fake_home / ".claude" / ".credentials.json").write_text('{"stale": true}')
    monkeypatch.setattr(sandbox_mod, "_harness_config_dir", lambda: tmp_path / "absent")
    monkeypatch.setattr(sandbox_mod.sys, "platform", "darwin")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox() as sb:
            assert sb.auth_source == "unauthenticated"
            assert not (sb.claude_config_dir / ".credentials.json").exists()


def test_non_darwin_still_seeds_the_credentials_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = _fake_home_without_credentials(monkeypatch, tmp_path)
    (fake_home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    monkeypatch.setattr(sandbox_mod, "_harness_config_dir", lambda: tmp_path / "absent")
    monkeypatch.setattr(sandbox_mod.sys, "platform", "linux")

    with sandbox() as sb:
        assert sb.auth_source == "credentials-file"
        assert (sb.claude_config_dir / ".credentials.json").is_file()
