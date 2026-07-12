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
from tests.harness.sandbox import (
    _CLAUDE_SESSION_LOG_DIRS,
    _CLAUDE_SESSION_LOG_FILES,
    _claude_session_log_excludes,
    _fingerprint,
    _guarded_roots,
)


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


# --- F4: orchestrator session-runtime log exclusion -----------------------------------------------


def _write_orchestrator_session_logs(fake_home: Path) -> None:
    """Simulate the ORCHESTRATOR (a live claude-code session) writing its own runtime logs under the
    real ~/.claude DURING a run — a file under every excluded dir + every excluded top-level file."""
    claude = fake_home / ".claude"
    for d in _CLAUDE_SESSION_LOG_DIRS:
        p = claude / d / "sub"
        p.mkdir(parents=True, exist_ok=True)
        (p / "orchestrator-log.jsonl").write_text('{"orchestrator": "session log"}')
    for fn in _CLAUDE_SESSION_LOG_FILES:
        (claude / fn).write_text("orchestrator runtime marker")


def test_orchestrator_session_logs_do_not_trip_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1 (positive/repro): the orchestrator writing its OWN claude-code session logs under the real
    ~/.claude during a run must NOT raise SandboxLeak (this WOULD raise pre-fix).

    Tightened (NITPICK-1): exercises the FULL `_write_orchestrator_session_logs` set against BOTH a
    no-exclude fingerprint (proves the full set would trip pre-fix — oracle-can-fail) and the real
    `sandbox()` run (proves it does not trip post-fix), in one body.
    """
    _seed_fake_cred(monkeypatch, tmp_path)
    fake_home = tmp_path / "fake-home"
    claude = fake_home / ".claude"

    # Oracle-can-fail on the FULL excluded set: a no-exclude fingerprint changes when the full
    # orchestrator-log set is written (this is the pre-fix behavior).
    before_no_exclude = _fingerprint(claude)
    _write_orchestrator_session_logs(fake_home)
    after_no_exclude = _fingerprint(claude)
    assert after_no_exclude != before_no_exclude, "full orchestrator-log set must trip pre-fix oracle"

    # Post-fix: the same full set, written inside a real sandbox() run, does NOT raise SandboxLeak.
    with sandbox() as sb:
        _ = sb.root
        _write_orchestrator_session_logs(fake_home)  # writes to EACH excluded dir + file
    # Clean exit, no SandboxLeak — the excluded session-log churn was correctly pruned.


def test_orchestrator_log_writes_WOULD_trip_pre_fix_oracle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1 oracle-can-fail (ST1.5f): prove the SAME writes trip the fingerprint WITHOUT the exclusion.

    Names a representative subset directly against `_fingerprint` with NO excludes — at least one
    dir-type entry (`projects/`) and one file-type entry (`history.jsonl`) — so the file-vs-dir path
    (L-1) is covered by the can-fail demo, not only the post-change green run. Compares a
    no-exclude fingerprint before/after the writes and asserts it changed."""
    _seed_fake_cred(monkeypatch, tmp_path)
    claude = tmp_path / "fake-home" / ".claude"
    before = _fingerprint(claude)  # NO excludes = pre-fix behavior
    # dir-type excluded entry:
    (claude / "projects" / "p").mkdir(parents=True, exist_ok=True)
    (claude / "projects" / "p" / "session.jsonl").write_text("x")
    # file-type excluded entry at the ~/.claude ROOT (first os.walk iteration):
    (claude / "history.jsonl").write_text("y")
    after_no_exclude = _fingerprint(claude)
    assert after_no_exclude != before, "pre-fix oracle must fire on orchestrator log writes"
    # ...and WITH the exclusion set, the same writes are invisible:
    excludes = _claude_session_log_excludes()
    # Re-root the excludes at the fake home (the helper derives from the real Path.home(), which the
    # fixture has monkeypatched to fake-home, so this already points at the fake tree).
    assert all(e.is_relative_to(claude.parent) for e in excludes)
    after_excluded = _fingerprint(claude, exclude=excludes)
    # The projects/ dir + history.jsonl writes must NOT appear in the excluded fingerprint.
    assert not any("projects" in k or k == "history.jsonl" for k in after_excluded)


def test_provider_respects_upstream_claude_config_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NITPICK-2: pin `provider.py:174` — the SINGLE invariant the whole 'no hole' argument rests on.

    The F4 exclusion is safe only because the sandboxed subject's `claude` CLI subprocesses inherit
    the tempdir `CLAUDE_CONFIG_DIR` (so their session logs never reach the real ~/.claude excluded
    dirs). That inheritance depends on `_subprocess_env` RESPECTING an already-set upstream
    `CLAUDE_CONFIG_DIR` rather than overwriting it with the brain default. If a future provider
    refactor dropped that guard, the exclusion would start masking a real escape — so this pins it in
    the harness suite. Token-free: `_subprocess_env` only builds an env dict, spawns nothing.
    """
    from brain.bridge.provider import _subprocess_env

    sentinel = "/tmp/upstream-sandbox-config-dir"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", sentinel)
    env = _subprocess_env()
    assert env["CLAUDE_CONFIG_DIR"] == sentinel, (
        "provider must respect an upstream CLAUDE_CONFIG_DIR (the F4 no-hole invariant)"
    )


def test_non_excluded_claude_write_still_trips_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C2 (negative control): writing to a NON-excluded ~/.claude path still raises SandboxLeak —
    both a known config file (settings.json) and a brand-new NON-allowlisted dir (fail-closed)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    fake_home = tmp_path / "fake-home"

    # (a) a kept config file.
    with pytest.raises(SandboxLeak):
        with sandbox() as sb:
            _ = sb.root
            (fake_home / ".claude" / "settings.json").write_text("leaked config")

    # (b) an unknown/future dir NOT on the allowlist must stay guarded (fail-closed).
    with pytest.raises(SandboxLeak):
        with sandbox() as sb:
            _ = sb.root
            newdir = fake_home / ".claude" / "some-future-config"
            newdir.mkdir(parents=True, exist_ok=True)
            (newdir / "x").write_text("a future claude-code dir a real leak could land in")


def test_kept_dir_write_still_trips_even_amid_orchestrator_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C2 reinforced: a KEPT-dir mutation is still caught even when excluded session logs churn in
    the same run — proves the exclusion prunes ONLY the allowlist, not the whole ~/.claude."""
    _seed_fake_cred(monkeypatch, tmp_path)
    fake_home = tmp_path / "fake-home"
    with pytest.raises(SandboxLeak):
        with sandbox() as sb:
            _ = sb.root
            _write_orchestrator_session_logs(fake_home)  # excluded churn (benign)
            (fake_home / ".claude" / "hooks").mkdir(parents=True, exist_ok=True)
            (fake_home / ".claude" / "hooks" / "evil.py").write_text("leaked hook code")


def test_fingerprint_exclude_single_subtree_still_prunes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C4 (behavior-preservation): the single->set `exclude` generalization still prunes exactly the
    named subtree at the pre-existing call site's usage — positive (sibling present) + pruned absent.
    Also exercises the resolve-both predicate via a SYMLINKED walk path so the deviation from the old
    unresolved `dp == excl` is tested, not assumed inert (F-2)."""
    root = tmp_path / "root"
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "present.txt").write_text("keep me")
    (root / "prune").mkdir(parents=True)
    (root / "prune" / "gone.txt").write_text("prune me")

    fp = _fingerprint(root, exclude=[root / "prune"])
    assert "keep/present.txt" in fp  # positive: sibling survives
    assert "prune/gone.txt" not in fp  # pruned subtree absent

    # Resolve-both predicate: walk through a SYMLINK to root; excluding the symlinked prune path must
    # still prune (old unresolved `dp == excl` could miss this; resolved-both matches).
    link = tmp_path / "link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    fp_link = _fingerprint(link, exclude=[root / "prune"])
    assert "keep/present.txt" in fp_link
    assert "prune/gone.txt" not in fp_link


def test_tempdir_removed_after_normal_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5: a clean (non-leaking) run removes the tempdir (keep defaults False)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    captured: dict[str, Path] = {}
    with sandbox() as sb:
        captured["root"] = sb.root
        assert sb.root.exists()
    assert not captured["root"].exists()  # cleaned up on normal exit


def test_tempdir_cleanup_and_keep_semantics_on_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C6: on a REAL leak, the outer-finally cleanup still runs (keep=False -> gone); keep=True keeps
    it. Pins the F4 'orphaned tempdir' claim: it is FALSE for the current outer-finally structure."""
    _seed_fake_cred(monkeypatch, tmp_path)

    # (a) leak with keep=False -> tempdir still removed (cleanup is in the OUTER finally).
    guarded_a = tmp_path / "guarded-a"
    guarded_a.mkdir()
    captured_a: dict[str, Path] = {}
    with pytest.raises(SandboxLeak):
        with sandbox(extra_guard_roots=[guarded_a]) as sb:
            captured_a["root"] = sb.root
            (guarded_a / "leak.txt").write_text("x")
    assert not captured_a["root"].exists(), "leak must NOT orphan the tempdir (keep=False)"

    # (b) leak with keep=True -> tempdir retained for post-mortem.
    guarded_b = tmp_path / "guarded-b"
    guarded_b.mkdir()
    captured_b: dict[str, Path] = {}
    try:
        with pytest.raises(SandboxLeak):
            with sandbox(keep=True, extra_guard_roots=[guarded_b]) as sb:
                captured_b["root"] = sb.root
                (guarded_b / "leak.txt").write_text("x")
        assert captured_b["root"].exists(), "keep=True must retain the tempdir on leak"
    finally:
        import shutil as _sh

        _sh.rmtree(captured_b["root"], ignore_errors=True)  # test cleanup
