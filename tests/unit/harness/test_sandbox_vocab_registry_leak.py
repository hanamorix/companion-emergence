"""Regression: sandbox() must not leak brain's process-global emotion registration.

When the harness stands up the real bridge (BridgeServer.start() -> brain.bridge.server.build_app
-> ensure_persona_vocabulary_loaded -> load_persona_vocabulary), brain reconstructs a missing
emotion_vocabulary.json from the Canary's seeded memories and register()s any persona-extension
emotion (e.g. "warmth") into the process-global brain.emotion.vocabulary._REGISTRY. Pre-fix,
sandbox() restored env + tempdirs but NOT that registry, so the registration escaped the run and
polluted later unit tests (notably tests/unit/brain/emotion/test_aggregate.py::
test_dropped_unregistered_emotion_warns_once, which then saw 0 warnings instead of 1).

These tests reproduce that leak and prove sandbox() now restores the registry on teardown — both on
a clean exit AND when SandboxLeak is raised. Zero model tokens: BridgeServer.start() only
builds+serves the app (the vocab reconstruction happens here); no turn is driven.

Diagnosis: hunts/harness-vocab-registry-leak/diagnosis.md.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

import brain.emotion.vocabulary as vocabulary
from tests.harness import (
    MemorySeed,
    PersonaSpec,
    SandboxLeak,
    build_persona,
    sandbox,
)
from tests.harness.engine import BridgeServer


def _seed_fake_cred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point HOME at a tmp dir with a fake ~/.claude/.credentials.json so the auth-seed copies a
    file (never touches the real credential) and no guarded real-home root is the developer's.
    (Same fixture pattern as test_sandbox_isolation.py.)"""
    fake_home = tmp_path / "fake-home"
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))


def _free_port() -> int:
    """Ask the OS for a free loopback port; raises OSError if it cannot bind (callers skip)."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.fixture(autouse=True)
def _registry_self_clean():
    """G9: this test must never leak the very thing it tests, regardless of prior state. Snapshot the
    process-global registry + warned-set around each test and restore in a finally, so the test
    passes whether or not `warmth`/`reverence` were registered before it ran."""
    import brain.emotion.aggregate as aggregate

    saved_registry = dict(vocabulary._REGISTRY)
    saved_warned = set(aggregate._warned_unregistered)
    try:
        yield
    finally:
        vocabulary._REGISTRY.clear()
        vocabulary._REGISTRY.update(saved_registry)
        aggregate._warned_unregistered.clear()
        aggregate._warned_unregistered.update(saved_warned)


def test_sandbox_does_not_leak_persona_emotion_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G3/G4: after a sandbox() block that reconstructs a warmth-seeded persona vocabulary via a real
    bridge start/stop, `warmth` must NOT remain registered. FAILS pre-fix (registration escapes),
    PASSES with the sandbox restore. Zero tokens (no turn driven)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    assert vocabulary.get("warmth") is None, "precondition: warmth unregistered before the run"

    try:
        port = _free_port()
    except OSError:
        pytest.skip("cannot bind a loopback port in this environment")

    with sandbox() as sb:
        # default MemorySeed emotion is {"warmth": 0.3} -> a non-baseline persona extension.
        live = build_persona(PersonaSpec(memories=[MemorySeed(content="Bob has a dog.")]), sb)
        # build_persona alone does NOT register (no vocab file, no load yet).
        assert vocabulary.get("warmth") is None, "build_persona must not register on its own"

        server = BridgeServer(live.persona_dir, port=port)
        try:
            server.start()
        except OSError:
            pytest.skip("cannot start the bridge on this port in this environment")
        # The real bridge start reconstructed the vocab and registered warmth (representativeness
        # anchor: the leak IS present inside the run, so a None-after is a real restore, not a no-op).
        assert vocabulary.get("warmth") is not None, "bridge start should register warmth in-run"
        server.stop()

    # AFTER the sandbox block: the registration must have been restored away. Pre-fix this is still
    # an Emotion(...) (the leak); post-fix it is None.
    assert vocabulary.get("warmth") is None, "sandbox() leaked a persona emotion registration"


def test_sandbox_registry_restore_survives_a_leak_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G5 (teardown-safe): the registry restore runs in the OUTER finally, so it happens even when
    SandboxLeak is raised. Decidable path (no proxy): a REAL bridge start registers warmth, the
    bridge thread is confirmed dead (so the in-place restore cannot race a live reader), THEN an
    out-of-sandbox mutation forces SandboxLeak — and after the raise, warmth is restored to None."""
    _seed_fake_cred(monkeypatch, tmp_path)
    assert vocabulary.get("warmth") is None

    try:
        port = _free_port()
    except OSError:
        pytest.skip("cannot bind a loopback port in this environment")

    guarded = tmp_path / "guarded-root"
    guarded.mkdir()
    (guarded / "existing.txt").write_text("original")

    with pytest.raises(SandboxLeak):
        with sandbox(extra_guard_roots=[guarded]) as sb:
            live = build_persona(
                PersonaSpec(memories=[MemorySeed(content="Bob has a dog.")]), sb
            )
            server = BridgeServer(live.persona_dir, port=port)
            try:
                server.start()
            except OSError:
                pytest.skip("cannot start the bridge on this port in this environment")
            assert vocabulary.get("warmth") is not None, "bridge start should register warmth"
            server.stop()
            # MAJOR-2: confirm the uvicorn thread joined (is dead) before we rely on the restore's
            # clear()+update() — a live reader of the non-thread-safe _REGISTRY would be a race.
            assert server._thread is not None and not server._thread.is_alive(), (
                "bridge thread must be joined/dead before the registry restore runs"
            )
            # Now force a real SandboxLeak: mutate a fingerprinted root OUTSIDE the sandbox.
            (guarded / "leaked.txt").write_text("this escaped the sandbox")

    # The restore ran in the OUTER finally despite the SandboxLeak raise.
    assert vocabulary.get("warmth") is None, "registry not restored on the SandboxLeak path"


def test_sandbox_registry_object_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G7: the restore mutates _REGISTRY contents in place (clear+update), it does NOT rebind the
    module attribute — so any reference held elsewhere stays valid. Assert the object identity of
    _REGISTRY is unchanged across a sandbox block."""
    _seed_fake_cred(monkeypatch, tmp_path)
    before_id = id(vocabulary._REGISTRY)
    with sandbox() as sb:
        build_persona(PersonaSpec(memories=[MemorySeed(content="Bob has a dog.")]), sb)
    assert id(vocabulary._REGISTRY) == before_id, "restore must not rebind _REGISTRY"


def test_sandbox_does_not_leak_a_nonwarmth_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G3-scope (MAJOR-3): the fix restores ANY non-baseline persona extension, not just the default
    `warmth` seed. Seed `reverence` and assert it is None after the sandbox block."""
    _seed_fake_cred(monkeypatch, tmp_path)
    assert vocabulary.get("reverence") is None, "precondition: reverence unregistered"

    try:
        port = _free_port()
    except OSError:
        pytest.skip("cannot bind a loopback port in this environment")

    with sandbox() as sb:
        live = build_persona(
            PersonaSpec(
                memories=[MemorySeed(content="Bob felt awe.", emotions={"reverence": 0.5})]
            ),
            sb,
        )
        server = BridgeServer(live.persona_dir, port=port)
        try:
            server.start()
        except OSError:
            pytest.skip("cannot start the bridge on this port in this environment")
        assert vocabulary.get("reverence") is not None, "bridge start should register reverence"
        server.stop()

    assert vocabulary.get("reverence") is None, "sandbox() leaked a non-warmth persona extension"
