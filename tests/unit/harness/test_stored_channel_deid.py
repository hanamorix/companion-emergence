"""Stored-channel de-identification seals (carriers A/B/C) — token-free.

Covers 1.5-criteria.md for the harness-stored-channel-deid change:
- C2 (teardown symmetry: HOME/cwd restored), C3 (F1/F2 $USER/$LOGNAME no-regression),
- C1b on-disk half (carrier B: synthetic oauthAccount pre-seeded into .claude.json),
- carrier A mechanism (neutral cwd is a de-identified scratch dir, not the repo),
- C5 (the leak guard still targets the REAL home under the synthetic-HOME seal).

Zero model tokens: real `sandbox()`, no `claude` subprocess. The CLI-overwrite half of carrier B
(does the CLI refetch and rewrite the real identity) is only testable in the live assembled arm — see
`changes/harness-stored-channel-deid/8-harness.md`.

NOTE ON THE FIXTURE (P-1, stage-3): the sibling `_seed_fake_cred` helpers patch `Path.home` to a
*constant* that ignores `$HOME`. That would MASK the carrier-C seal (which sets `os.environ["HOME"]`),
so the C5 guard-targeting test below uses a **dynamic** `Path.home` that reads `os.environ["HOME"]`,
letting the seal's HOME swap actually move `Path.home()` — which is exactly what the guard must survive.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from tests.harness import SandboxLeak, sandbox
from tests.harness.config import (
    SYNTHETIC_DISPLAY_NAME,
    SYNTHETIC_EMAIL,
    SYNTHETIC_ORG,
    SYNTHETIC_USER,
)

sandbox_mod = importlib.import_module("tests.harness.sandbox")


def _seed_dynamic_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> Path:
    """Point HOME at `home` with a fake ~/.claude/.credentials.json, and make `Path.home()` follow
    `$HOME` DYNAMICALLY (not a constant) so the carrier-C HOME swap is observable. Also pin the
    Documents resolver so the notes shallow-scan stays inside the fake home."""
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    (home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    (home / "Documents").mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(os.environ["HOME"])))
    monkeypatch.setattr(sandbox_mod, "_documents_dir", lambda: Path(os.environ["HOME"]) / "Documents")
    return home


# ─────────────────────────── C3 — F1/F2 $USER/$LOGNAME no-regression ───────────────────────────
def test_user_logname_synthetic_in_restored_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C3: inside sandbox() $USER == $LOGNAME == SYNTHETIC_USER; after exit both restored, incl.
    'was unset' → unset (the seal must not regress the input-side de-id)."""
    _seed_dynamic_home(monkeypatch, tmp_path / "real-home")
    monkeypatch.setenv("USER", "realuser")
    monkeypatch.delenv("LOGNAME", raising=False)  # 'was unset' branch

    with sandbox(live_check="off") as sb:
        assert os.environ["USER"] == SYNTHETIC_USER
        assert os.environ["LOGNAME"] == SYNTHETIC_USER
        assert sb is not None

    assert os.environ["USER"] == "realuser"          # restored
    assert "LOGNAME" not in os.environ               # 'was unset' → unset


# ─────────────────────────── C2 — HOME + cwd teardown symmetry ───────────────────────────
def test_home_and_cwd_synthetic_in_restored_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C2: carrier A (neutral cwd) + carrier C (synthetic HOME) are installed inside the window and
    restored on exit. cwd is a de-identified scratch dir under the sandbox root (NOT the repo)."""
    real_home = _seed_dynamic_home(monkeypatch, tmp_path / "real-home")
    outer_cwd = os.getcwd()

    with sandbox(live_check="off") as sb:
        # carrier C: HOME points into the sandbox root, with an empty .claude (no CLAUDE.md).
        seal_home = Path(os.environ["HOME"])
        assert seal_home == sb.root / "home"
        assert seal_home != real_home
        assert (seal_home / ".claude").is_dir()
        assert not (seal_home / ".claude" / "CLAUDE.md").exists()
        # carrier A: cwd is the neutral scratch dir under the sandbox root — not the repo, not real home.
        cwd = Path(os.getcwd()).resolve()
        assert cwd == (sb.root / "work").resolve()
        assert sb.root.resolve() in cwd.parents or cwd == (sb.root / "work").resolve()

    assert os.environ["HOME"] == str(real_home)      # restored
    assert Path(os.getcwd()) == Path(outer_cwd)      # cwd restored (before rmtree)
    assert not sb.root.exists()                       # rmtree happened, cwd was outside it


# ─────────────────────────── carrier B — synthetic oauthAccount pre-seeded ───────────────────────────
def test_carrier_b_oauthaccount_scrubbed_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1b (on-disk half): _seed_scrubbed_config pre-seeds a SYNTHETIC oauthAccount into the CLI's
    <CLAUDE_CONFIG_DIR>/.claude.json — no real displayName/email/org. (The CLI-refetch half is a live
    arm concern.)"""
    _seed_dynamic_home(monkeypatch, tmp_path / "real-home")

    with sandbox(live_check="off") as sb:
        cfg = sb.claude_config_dir / ".claude.json"
        assert cfg.is_file(), "carrier B must pre-seed .claude.json under CLAUDE_CONFIG_DIR"
        oa = json.loads(cfg.read_text())["oauthAccount"]
        assert oa["displayName"] == SYNTHETIC_DISPLAY_NAME == SYNTHETIC_USER
        assert oa["emailAddress"] == SYNTHETIC_EMAIL
        assert oa["organizationName"] == SYNTHETIC_ORG
        # fresh profileFetchedAt (present + plausibly-now) so a short arm stays inside the cache TTL.
        assert isinstance(oa["profileFetchedAt"], int) and oa["profileFetchedAt"] > 0


# ─────────────────────────── C5 — guard still targets the REAL home under synthetic HOME ───────────────────────────
def test_guard_roots_target_real_home_not_synthetic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5(a): even though the carrier-C seal swaps $HOME to a synthetic dir for the run window, the
    leak-guard roots point at the REAL ~/.claude (computed before the swap), never the synthetic one.
    Uses a DYNAMIC Path.home (reads $HOME) so the swap is real — under the constant-patch fixture this
    assertion would be vacuous (P-1)."""
    real_home = _seed_dynamic_home(monkeypatch, tmp_path / "real-home")
    real_claude = (real_home / ".claude").resolve()

    with sandbox(live_check="off") as sb:
        # The seal DID move Path.home() (proving the swap is genuine under this fixture)...
        assert Path.home() == sb.root / "home"
        # ...yet the guard roots still target the REAL ~/.claude, not <synthetic HOME>/.claude.
        assert real_claude in sb.guard_roots
        assert (sb.root / "home" / ".claude").resolve() not in sb.guard_roots


def test_guard_roots_follow_home_dynamically_oracle_can_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5 oracle-can-fail demonstrator: `_guarded_roots()` derives ~/.claude from `Path.home()` →
    `$HOME`, so IF the seal set the synthetic HOME *before* computing guard roots, the guard would
    target the synthetic home. This proves the C5 assertion above can actually fail on a mis-ordered
    build (it is not vacuous). Here we set HOME to two different dirs and show guard roots follow."""
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    for h in (home_a, home_b):
        (h / ".claude").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(os.environ["HOME"])))

    monkeypatch.setenv("HOME", str(home_a))
    roots_a = sandbox_mod._guarded_roots()
    monkeypatch.setenv("HOME", str(home_b))
    roots_b = sandbox_mod._guarded_roots()

    assert (home_a / ".claude").resolve() in roots_a
    assert (home_a / ".claude").resolve() not in roots_b
    assert (home_b / ".claude").resolve() in roots_b


def test_leak_guard_still_fires_under_seals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5(b): the SandboxLeak negative control still fires while the carrier-A/C seals are active —
    mutating a fingerprinted extra guard root raises, so the seals did not gut the #1 guarantee."""
    _seed_dynamic_home(monkeypatch, tmp_path / "real-home")
    guarded = tmp_path / "guarded-root"
    guarded.mkdir()
    (guarded / "before.txt").write_text("x")

    with pytest.raises(SandboxLeak):
        with sandbox(live_check="off", extra_guard_roots=[guarded]) as sb:
            assert Path.home() == sb.root / "home"          # seal active
            (guarded / "leaked.txt").write_text("mutation during the run")


def test_clean_run_under_seals_does_not_trip_spurious_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5 symmetry: a clean run (no guarded mutation) does NOT raise a spurious SandboxLeak — proving
    the real-HOME restore before the after-snapshot keeps the ~/.claude content-hash symmetric.
    (Discrimination note: a broken build that left HOME synthetic during the after-snapshot would flip
    the real ~/.claude fingerprint but that lone diff DOWNGRADES to a warning, so this test alone does
    not prove ordering — that is carried by ``test_guard_roots_target_real_home_not_synthetic``.)"""
    _seed_dynamic_home(monkeypatch, tmp_path / "real-home")
    with sandbox(live_check="off") as sb:
        assert sb.root.exists()
    # no raise == symmetric before/after fingerprint under the HOME swap


def test_write_to_real_home_dotfile_trips_leak_under_synthetic_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stage-6 MAJOR fix: carrier C sets a synthetic HOME, which would weaken brain's write_guard
    (it denies writes under Path.home()/<.ssh,.aws,...> and Path.home() is synthetic during the run).
    The leak fingerprint now covers those REAL-home sensitive paths (computed before the swap), so a
    write to the REAL ~/.ssh during a run still trips SandboxLeak — restoring the detection layer.

    Oracle-can-fail: without the sensitive-home fingerprint, this write (to a path not under any other
    guard root, and — under synthetic HOME — not under write_guard's deny set) would go UNDETECTED."""
    real_home = _seed_dynamic_home(monkeypatch, tmp_path / "real-home")
    real_ssh = real_home / ".ssh"
    real_ssh.mkdir()  # exists before the run, so it is fingerprinted as a dir

    with pytest.raises(SandboxLeak):
        with sandbox(live_check="off") as sb:
            assert Path.home() == sb.root / "home"  # synthetic HOME active
            # A companion write to the REAL ~/.ssh (absolute path — the risk the seal opened).
            (real_ssh / "authorized_keys").write_text("ssh-ed25519 AAAA... attacker")
