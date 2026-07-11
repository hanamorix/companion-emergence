"""Headline safety test — prove a REAL in-sandbox run is confined + the leak oracle can fail.

This is the #1 requirement's gate (criteria G1/G2/G3/G12/G13). It is a real B-REP gate, not a
diff-algorithm smoke test:
  - POSITIVE: a real in-sandbox persona build leaves every guarded root unchanged AND the expected
    new persona paths actually appear under the sandbox root.
  - NEGATIVE CONTROL: mutating a fingerprinted (extra) guard root makes ``sandbox()`` raise
    ``SandboxLeak`` — proving the oracle fires on a real mutation (an oracle that cannot fail is
    worthless).
Zero model tokens (a fake provider; the persona build itself spends none).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.harness import (
    MemorySeed,
    PersonaSpec,
    SandboxLeak,
    build_persona,
    sandbox,
)
from tests.harness.sandbox import _guarded_roots


def _seed_fake_cred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point HOME at a tmp dir with a fake ~/.claude/.credentials.json so the auth-seed copies a
    file (never touches the real credential) and no guarded real-home root is the developer's."""
    fake_home = tmp_path / "fake-home"
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))


def test_real_in_sandbox_write_is_confined_and_nothing_leaks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POSITIVE: a real persona build lands under the sandbox root; no guarded root changes (G1a)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with sandbox() as sb:
        # A real in-sandbox persona write (real engine APIs, no tokens).
        spec = PersonaSpec(memories=[MemorySeed(content="Bob mentioned his dog Biscuit.")])
        live = build_persona(spec, sb)
        # (a) every expected new path is UNDER the sandbox root (catches a silent no-op write).
        assert live.persona_dir.is_relative_to(sb.root)
        assert (live.persona_dir / "persona_config.json").exists()
        assert (live.persona_dir / "memories.db").exists()
        assert (live.persona_dir / "voice.md").exists()
        # (b) safe-by-force persona config (G1b).
        cfg = json.loads((live.persona_dir / "persona_config.json").read_text())
        assert cfg["notes_enabled"] is False
        assert cfg["kindled_relay_url"] is None
    # On clean exit, sandbox() did NOT raise SandboxLeak — nothing outside the sandbox changed.


def test_real_engine_precedence_is_used_not_a_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G2: brain.paths.get_home() (the REAL precedence) resolves to the sandbox root, and the env
    exports CLAUDE_CONFIG_DIR (the mechanism provider.py:174 respects)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with sandbox() as sb:
        from brain.paths import get_home  # real engine mechanism, not a reimplementation

        assert get_home() == sb.root.resolve()
        assert os.environ["CLAUDE_CONFIG_DIR"] == str(sb.claude_config_dir)
        assert sb.claude_config_dir.is_relative_to(sb.root)
        assert (sb.claude_config_dir / ".credentials.json").exists()  # auth seeded (G1c)


def test_nellbrain_home_is_unset_inside(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """G3: a stray NELLBRAIN_HOME on the host must not win the fallback (paths.py:60)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    monkeypatch.setenv("NELLBRAIN_HOME", "/bogus/should/be/unset")
    with sandbox() as sb:
        assert "NELLBRAIN_HOME" not in os.environ
        from brain.paths import get_home

        assert get_home() == sb.root.resolve()  # not the bogus value
    # restored after exit (the fixture set it; sandbox must not have deleted it permanently).
    assert os.environ.get("NELLBRAIN_HOME") == "/bogus/should/be/unset"


def test_guard_root_set_is_broad_and_platformdirs_derived(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G12: the fingerprint set includes the platformdirs data/cache/state homes (R1)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    from platformdirs import PlatformDirs

    dirs = PlatformDirs("companion-emergence")
    roots = {str(r) for r in _guarded_roots()}
    for expected in (dirs.user_data_path, dirs.user_cache_path, dirs.user_state_path):
        assert str(Path(expected).resolve()) in roots, f"guard set missing {expected}"


def test_leak_oracle_fires_on_a_real_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NEGATIVE CONTROL (G1d): mutating a fingerprinted extra guard root -> SandboxLeak.

    Proves the oracle CAN fail — a clean result elsewhere is only trustworthy because of this.
    """
    _seed_fake_cred(monkeypatch, tmp_path)
    guarded = tmp_path / "guarded-root"
    guarded.mkdir()
    (guarded / "existing.txt").write_text("original")

    with pytest.raises(SandboxLeak):
        with sandbox(extra_guard_roots=[guarded]) as sb:
            build_persona(PersonaSpec(), sb)  # legitimate in-sandbox work
            # ...then a real mutation OUTSIDE the sandbox, under a fingerprinted root:
            (guarded / "leaked.txt").write_text("this escaped the sandbox")


def test_leak_oracle_catches_same_size_same_mtime_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M1: a same-size + mtime-RESTORED in-place overwrite of a fingerprinted ~/.claude dotfile is
    still caught (content-hash), not just an add-a-file. Closes the owner-flagged blind spot."""
    import os

    _seed_fake_cred(monkeypatch, tmp_path)
    # Pre-existing ~/.claude dotfile (content-hashed because the root is ~/.claude).
    claude_dir = tmp_path / "fake-home" / ".claude"
    target = claude_dir / "settings.json"
    target.write_text("AAAA")  # 4 bytes
    orig_stat = target.stat()

    with pytest.raises(SandboxLeak):
        with sandbox() as sb:
            _ = sb.root
            # Same-size (4 bytes) different content, then restore the mtime so (size, mtime_ns) match.
            target.write_text("BBBB")
            os.utime(target, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
            # (size, mtime_ns) are now identical to before — only the CONTENT differs.


def test_env_restored_even_after_leak_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G13: env is restored in a finally even when SandboxLeak raises — no stale KINDLED_HOME
    pointing at a deleted tempdir poisons the next run."""
    _seed_fake_cred(monkeypatch, tmp_path)
    monkeypatch.setenv("KINDLED_HOME", "/prior/kindled/home")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    guarded = tmp_path / "guarded-root2"
    guarded.mkdir()

    with pytest.raises(SandboxLeak):
        with sandbox(extra_guard_roots=[guarded]) as sb:
            _ = sb.root
            (guarded / "leak.txt").write_text("x")

    # env fully restored: prior KINDLED_HOME back, CLAUDE_CONFIG_DIR gone again.
    assert os.environ["KINDLED_HOME"] == "/prior/kindled/home"
    assert "CLAUDE_CONFIG_DIR" not in os.environ
