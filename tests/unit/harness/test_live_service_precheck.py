"""Live-companion-service pre-check (Phase 2) — token-free unit tests.

Proves the pre-check that `sandbox()` runs at entry detects a RUNNING companion bridge up front and
fails/warns with a distinct, actionable `LiveServiceDetected` instead of letting the run proceed and
die later with a misleading spurious `SandboxLeak`. Zero model tokens; no `claude -p`; no real files
touched (a fake HOME + a monkeypatched `brain.paths.get_home`).

Fidelity: the SCAN root is the engine's real resolver `brain.paths.get_home()` (F1), and liveness is
decided by the REAL `brain.bridge.state_file.pid_is_alive` (P2) — the tests control the scan root by
monkeypatching `get_home` to a fixture dir, but never reimplement pid-liveness. Each oracle is shown
able to fail (H6): the live/dead and pidfile-arm negative controls are exercised.
"""

from __future__ import annotations

import importlib
import json
import os
import warnings
from pathlib import Path

import pytest

from tests.harness import (
    LiveServiceDetected,
    SandboxLeak,
    sandbox,
)
from tests.harness.sandbox import _guarded_roots, _live_bridges

# `tests.harness.__init__` re-exports the `sandbox` FUNCTION, which shadows the submodule attribute
# on the package — so `import tests.harness.sandbox as m` binds the function. Fetch the real module
# object explicitly for monkeypatching module-level helpers.
sandbox_mod = importlib.import_module("tests.harness.sandbox")


def _clear_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    """U1/F1 seed-integrity: clear the XDG_*_HOME family so PlatformDirs resolves under the fake
    HOME on Linux/CI (a stray XDG_DATA_HOME would otherwise redirect the guarded roots away)."""
    for var in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)


def _seed_fake_cred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HOME at a tmp dir with a fake ~/.claude/.credentials.json (never the real one).

    Returns the fake home path.
    """
    fake_home = tmp_path / "fake-home"
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    _clear_xdg(monkeypatch)
    return fake_home


def _seed_bridge(
    monkeypatch: pytest.MonkeyPatch,
    engine_home: Path,
    *,
    pid: object,
    persona: str = "Canary",
    filename: str = "bridge.json",
    raw: bytes | None = None,
) -> Path:
    """Seed a `bridge.json` under `engine_home/personas/<persona>/` AND point the REAL resolver
    `brain.paths.get_home()` at `engine_home` so `_live_bridges` scans there (F1/P12).

    `raw` overrides the JSON body (for the corrupt-primary test). Returns the seeded file path.
    """
    persona_dir = engine_home / "personas" / persona
    persona_dir.mkdir(parents=True, exist_ok=True)
    target = persona_dir / filename
    if raw is not None:
        target.write_bytes(raw)
    else:
        target.write_text(json.dumps({"persona": persona, "pid": pid, "port": 8931}))
    # Point the engine resolver at this fixture home so the scan roots there (real fn, patched root).
    import brain.paths as brain_paths

    monkeypatch.setattr(brain_paths, "get_home", lambda: engine_home)
    return target


# ─────────────────────────── P1 / P4 — live bridge raises up front ───────────────────────────


def test_live_bridge_raises_live_service_detected_up_front(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P1: a live `bridge.json` (own pid) under the real-resolver home raises `LiveServiceDetected`
    at ENTRY (before yielding), NOT a post-run `SandboxLeak`; the message names pid + persona."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid(), persona="Canary")

    entered = False
    with pytest.raises(LiveServiceDetected) as ei:
        with sandbox() as sb:  # default live_check="raise"
            entered = True  # pragma: no cover - must NOT be reached
            _ = sb.root
    assert entered is False, "pre-check must raise BEFORE yielding the handle"
    # Distinct exception type — NOT a SandboxLeak (so `except SandboxLeak` won't swallow it).
    assert not isinstance(ei.value, SandboxLeak)
    msg = str(ei.value)
    assert str(os.getpid()) in msg and "Canary" in msg


def test_live_check_off_disables_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P1/P4 oracle-can-fail: the SAME live pidfile with live_check="off" does NOT raise — proving
    the trip is caused by the live pidfile, not something unconditional; and "off" skips the scan."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())

    with sandbox(live_check="off") as sb:  # must NOT raise
        assert sb.root.exists()


def test_live_check_warn_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P4: live_check="warn" emits a RuntimeWarning whose MESSAGE names pid+persona (L1: not just
    the category, so `_seed_auth`'s unrelated warning can't satisfy it) and CONTINUES the run."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())

    body_ran = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sandbox(live_check="warn") as sb:
            body_ran = True
            assert sb.root.exists()
    assert body_ran is True
    live_msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and str(os.getpid()) in str(w.message)
        and "Canary" in str(w.message)
    ]
    assert live_msgs, "warn mode must emit a RuntimeWarning naming the live pid + persona"


def test_invalid_live_check_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P4: an invalid live_check value fails fast with ValueError (never silent-off)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="live_check"):
        with sandbox(live_check="bogus"):
            pass  # pragma: no cover


# ─────────────────────────── P2 — real mechanism, read-only ───────────────────────────


def test_scan_uses_real_pid_is_alive_and_not_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P2: the scan calls the REAL `state_file.pid_is_alive` and does NOT call the heal-on-read
    `state_file.read` (which can WRITE). Spies count the calls."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())

    import brain.bridge.state_file as sf

    alive_calls: list[int] = []
    read_calls: list[object] = []
    real_alive = sf.pid_is_alive
    monkeypatch.setattr(
        sf, "pid_is_alive", lambda pid: alive_calls.append(pid) or real_alive(pid)
    )
    monkeypatch.setattr(sf, "read", lambda *a, **k: read_calls.append(a) or None)

    live = _live_bridges()
    assert (os.getpid(), "Canary") in live
    assert os.getpid() in alive_calls, "must call the real pid_is_alive"
    assert read_calls == [], "must NOT call heal-on-read state_file.read()"


def test_scan_is_read_only_leaves_root_byte_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P2 (read-only): scanning does not mutate the seeded bridge.json (byte-for-byte unchanged).
    Oracle-can-fail: this assertion fails if the scan wrote anything."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    path = _seed_bridge(monkeypatch, engine_home, pid=os.getpid())
    before = path.read_bytes()
    _ = _live_bridges()
    assert path.read_bytes() == before


def test_pid_is_alive_verdict_is_load_bearing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P2 oracle-can-fail: patch pid_is_alive->False for the live pid; the pre-check must NOT trip
    — proving the real function's verdict (not the pidfile's mere presence) is load-bearing."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())
    import brain.bridge.state_file as sf

    monkeypatch.setattr(sf, "pid_is_alive", lambda pid: False)
    assert _live_bridges() == []


# ─────────────────────────── P3 — dead / stale pid does not false-positive ───────────────────────────


@pytest.mark.parametrize("dead_pid", [None, -1, 0])
def test_dead_pid_does_not_false_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dead_pid: object
) -> None:
    """P3 (U3): pid=None / negative / zero are cross-OS-deterministic dead cases → no trip; the run
    proceeds and yields a handle."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=dead_pid)
    with sandbox() as sb:  # must NOT raise
        assert sb.root.exists()


def test_never_allocated_pid_is_dead_guarded_in_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P3 secondary case (U3): a large never-allocated pid, guarded by an IN-TEST liveness
    pre-assert so a flaky 'this box owns that pid' can't silently mask a regression."""
    import brain.bridge.state_file as sf

    big = 2**31 - 1
    if sf.pid_is_alive(big):  # pragma: no cover - vanishingly rare; skip if the box owns it
        pytest.skip("this machine happens to own the 'dead' pid")
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=big)
    with sandbox() as sb:
        assert sb.root.exists()


# ─────────────────────────── P5 — warn-mode SandboxLeak annotation ───────────────────────────


def test_warn_mode_annotates_subsequent_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P5: under warn mode, a live pidfile + a real in-body mutation of an extra guarded root →
    the SandboxLeak message carries BOTH the changed-root text AND the pre-check annotation."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())
    guarded = tmp_path / "guarded-extra"
    guarded.mkdir()

    with pytest.raises(SandboxLeak) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sandbox(live_check="warn", extra_guard_roots=[guarded]) as sb:
                _ = sb.root
                (guarded / "leaked.txt").write_text("escaped")
    msg = str(ei.value)
    assert "mutated during a sandboxed run" in msg
    assert "live companion service was detected at pre-check" in msg


def test_leak_without_live_service_is_not_annotated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P5 oracle-can-fail: the SAME leak WITHOUT a live pidfile → a SandboxLeak with NO pre-check
    annotation, proving the annotation is driven by the finding, not always appended."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    # No bridge.json seeded, but point the resolver at an empty engine home.
    engine_home.mkdir()
    import brain.paths as brain_paths

    monkeypatch.setattr(brain_paths, "get_home", lambda: engine_home)
    guarded = tmp_path / "guarded-extra2"
    guarded.mkdir()

    with pytest.raises(SandboxLeak) as ei:
        with sandbox(live_check="warn", extra_guard_roots=[guarded]) as sb:
            _ = sb.root
            (guarded / "leaked.txt").write_text("escaped")
    assert "detected at pre-check" not in str(ei.value)


# ─────────────────────────── P6 — opt-in probe + generic message ───────────────────────────


def test_probe_off_by_default_does_not_trip_on_external_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P6 (default-off): with probe=False (default) and NO live pidfile, an external writer that
    would trip the probe does NOT raise — the probe arm is off by default."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    engine_home.mkdir()
    import brain.paths as brain_paths

    monkeypatch.setattr(brain_paths, "get_home", lambda: engine_home)
    guarded = tmp_path / "probe-root"
    guarded.mkdir()

    # Install an INJECTING probe that WOULD trip (it mutates a guarded root in its window). With the
    # default probe=False the helper must never be called, so this run must NOT raise — proving
    # default-off standalone (not only via the opt-in pairing). Also assert the injector never ran.
    injected: list[bool] = []

    def _injecting_probe(snapshot_fn, wait_s):  # type: ignore[no-untyped-def]
        injected.append(True)
        before = snapshot_fn()
        (guarded / "external.txt").write_text("would trip if the probe ran")
        after = snapshot_fn()
        return before != after

    monkeypatch.setattr(sandbox_mod, "_probe_external_writer", _injecting_probe)

    with sandbox(extra_guard_roots=[guarded]) as sb:  # probe defaults False
        assert sb.root.exists()  # no LiveServiceDetected
    assert injected == [], "the probe helper must not run when probe=False (default-off)"


def test_probe_opt_in_fires_with_generic_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P6 (opt-in): probe=True + an injected external write during the probe window → raises with a
    GENERIC 'external process' message DISTINCT from the companion-service message.
    Oracle-can-fail: no injected write → no trip (see the next test)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    engine_home.mkdir()
    import brain.paths as brain_paths

    monkeypatch.setattr(brain_paths, "get_home", lambda: engine_home)
    guarded = tmp_path / "probe-root"
    guarded.mkdir()
    (guarded / "seed.txt").write_text("orig")

    # Inject the mutation deterministically INTO the probe window: patch the probe to mutate the
    # guarded root between its two snapshots (Phase-1 negative-control style — no sleep race).
    def _injecting_probe(snapshot_fn, wait_s):  # type: ignore[no-untyped-def]
        before = snapshot_fn()
        (guarded / "external.txt").write_text("a foreign process wrote this")
        after = snapshot_fn()
        return before != after

    monkeypatch.setattr(sandbox_mod, "_probe_external_writer", _injecting_probe)

    with pytest.raises(LiveServiceDetected) as ei:
        with sandbox(probe=True, extra_guard_roots=[guarded]):
            pass  # pragma: no cover
    msg = str(ei.value)
    assert "external process mutated a guarded root" in msg
    assert "live companion service" not in msg  # generic, not companion-specific


def test_probe_silent_when_nothing_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P6 oracle-can-fail: probe=True with NOTHING mutating during the window → no trip (the probe
    discriminates a live writer from a quiet system). probe_wait=0 keeps it instant."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    engine_home.mkdir()
    import brain.paths as brain_paths

    monkeypatch.setattr(brain_paths, "get_home", lambda: engine_home)
    with sandbox(probe=True, probe_wait=0.0) as sb:  # nothing writes → silent
        assert sb.root.exists()


# ─────────────────────────── P7 — pidfile scan is the load-bearing detector ───────────────────────────


def test_pidfile_scan_carries_detection_default_probe_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P7: a live-but-idle bridge (live pidfile, probe off, nothing writing) is STILL detected —
    carried by the pidfile scan alone."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())
    with pytest.raises(LiveServiceDetected):
        with sandbox():  # probe defaults off
            pass  # pragma: no cover


def test_pidfile_arm_is_load_bearing_executed_negative_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P7 EXECUTED negative control (not a counterfactual): patch `_live_bridges`->[] with the SAME
    live pidfile present + probe off → the pre-check must NOT raise, proving the pidfile arm (not
    something else) carried the detection."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())
    monkeypatch.setattr(sandbox_mod, "_live_bridges", lambda: [])
    with sandbox() as sb:  # must NOT raise now
        assert sb.root.exists()


# ─────────────────────────── P10 — bounded + tolerant scan ───────────────────────────


def test_corrupt_bridge_json_is_tolerated_not_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P10: a corrupt (non-JSON) bridge.json is skipped without raising AND does not count as live;
    a valid live pidfile alongside it still trips ('tolerant' isn't 'blind')."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    # Corrupt one persona's file; seed a valid live one for another persona.
    _seed_bridge(monkeypatch, engine_home, pid=None, persona="Corrupt", raw=b"{not json")
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid(), persona="Canary")
    live = _live_bridges()  # must not raise
    assert (os.getpid(), "Canary") in live
    assert all(name != "Corrupt" for _, name in live)


def test_missing_engine_home_is_tolerated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P10: a missing engine home (no personas dir) → the scan returns no live bridge, no error."""
    _seed_fake_cred(monkeypatch, tmp_path)
    import brain.paths as brain_paths

    monkeypatch.setattr(brain_paths, "get_home", lambda: tmp_path / "does-not-exist")
    assert _live_bridges() == []


# ─────────────────────────── P11 — .bak-recovery gap (disclosed, pinned) ───────────────────────────


def test_bak_recoverable_live_bridge_is_knowingly_not_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P11 (FID1 disclosure): a live bridge whose PRIMARY bridge.json is corrupt but whose .bak
    carries the live pid is a KNOWN, accepted false-negative (the read-only scan does not consult
    .bak, because that would need the heal-on-read read() which WRITES). Backstopped by post-run
    SandboxLeak. Oracle-can-fail: the same live pid in a VALID primary DOES get detected."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    # Corrupt primary + a .bak1 sibling carrying the live pid.
    path = _seed_bridge(
        monkeypatch, engine_home, pid=None, persona="Canary", raw=b"{corrupt primary"
    )
    (path.parent / "bridge.json.bak1").write_text(
        json.dumps({"persona": "Canary", "pid": os.getpid()})
    )
    assert _live_bridges() == [], "corrupt-primary + .bak live pid is knowingly NOT detected"

    # Oracle-can-fail: a VALID primary with the same live pid IS detected.
    path.write_text(json.dumps({"persona": "Canary", "pid": os.getpid()}))
    assert (os.getpid(), "Canary") in _live_bridges()


# ─────────────────────────── P12 — scan root == real resolver (F1 fix) ───────────────────────────


def test_scan_root_is_the_real_resolver_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P12 (BLOCKER F1 fix): `_live_bridges` scans `brain.paths.get_home()/personas`, NOT a bare
    PlatformDirs path. Assert positively that a pidfile seeded under the resolver home IS found and
    that one seeded under a divergent (no-appauthor) PlatformDirs path is NOT (documents the bug the
    fix defends)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid(), persona="Canary")

    # Positive: the resolver-home pidfile is found.
    assert (os.getpid(), "Canary") in _live_bridges()

    # A pidfile placed under a DIFFERENT root (what a bare PlatformDirs(_APP) without appauthor
    # could resolve to on Windows/Mac) is NOT found — the scan follows the real resolver, not that.
    from platformdirs import PlatformDirs

    naive = Path(PlatformDirs("companion-emergence").user_data_path)
    if naive.resolve() != engine_home.resolve():  # only meaningful where they diverge
        naive_persona = naive / "personas" / "Ghost"
        naive_persona.mkdir(parents=True, exist_ok=True)
        (naive_persona / "bridge.json").write_text(
            json.dumps({"persona": "Ghost", "pid": os.getpid()})
        )
        assert all(name != "Ghost" for _, name in _live_bridges())


def test_guarded_roots_includes_appauthor_correct_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P12: the Phase-1 leak fingerprint set now INCLUDES the engine's appauthor-correct DEFAULT
    home (`brain.paths._dirs.user_data_path`), fixing the latent Phase-1 appauthor bug so it guards
    the dir the real bridge writes on Windows/Mac. It uses the un-overridden default (NOT
    `get_home()`), so the sandbox's own KINDLED_HOME root is never fingerprinted (that would
    false-trip on legitimate in-sandbox writes — the Phase-1 regression P9 caught here)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    import brain.paths as brain_paths

    expected = str(Path(brain_paths._dirs.user_data_path).resolve())
    roots = {str(Path(r).resolve()) for r in _guarded_roots()}
    assert expected in roots


# ─────────────────────────── P13 — every run routes through sandbox() ───────────────────────────


def test_every_run_path_routes_through_sandbox() -> None:
    """P13 (A2 promoted): a source assert that the run entrypoints reach `sandbox()`. The worked
    example calls `sandbox(`; no harness helper stands up a BridgeServer outside a `sandbox()`."""
    harness_dir = Path(__file__).resolve().parents[2] / "harness"
    example = harness_dir / "examples" / "test_generic_run.py"
    assert "sandbox(" in example.read_text(), "the worked example must run inside sandbox()"

    # No module under tests/harness/ should construct BridgeServer without importing sandbox.
    for py in harness_dir.rglob("*.py"):
        text = py.read_text()
        if "BridgeServer(" in text and py.name != "engine.py":
            assert "sandbox" in text, f"{py.name} builds a bridge but does not reference sandbox()"


# ─────────────────────────── P14 — env restored after LiveServiceDetected raise ───────────────────────────


def test_env_restored_after_live_service_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P14 (CH8-d, sibling of Phase-1 G13 for the new raise site): a `LiveServiceDetected` raise
    still restores env in the finally + removes the tempdir — no stale KINDLED_HOME leaks."""
    _seed_fake_cred(monkeypatch, tmp_path)
    engine_home = tmp_path / "engine-home"
    _seed_bridge(monkeypatch, engine_home, pid=os.getpid())
    monkeypatch.setenv("KINDLED_HOME", "/prior/kindled/home")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    roots_before = set(Path(tempfile_gettempdir()).glob("ce-harness-*"))
    with pytest.raises(LiveServiceDetected):
        with sandbox():
            pass  # pragma: no cover

    # Env fully restored: prior KINDLED_HOME back, CLAUDE_CONFIG_DIR still absent.
    assert os.environ["KINDLED_HOME"] == "/prior/kindled/home"
    assert "CLAUDE_CONFIG_DIR" not in os.environ
    # Tempdir removed (no new ce-harness-* left behind by this run).
    roots_after = set(Path(tempfile_gettempdir()).glob("ce-harness-*"))
    assert roots_after <= roots_before


def tempfile_gettempdir() -> str:
    import tempfile

    return tempfile.gettempdir()
