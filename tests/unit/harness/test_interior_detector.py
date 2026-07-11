"""Token-free tests for the interior-leak detector + the send-script's interior-block read.

Covers P20 (the interior signal fires on a verbatim interior quote, silent on a benign reply),
P21 (the interior read targets ONLY the sandbox DB, KINDLED_HOME set unconditionally), and CH8-b
(the real `_interior_block()`/`interior_block()` path is exercised against a SEEDED temp DB — no
mock, no provider, 0 tokens). Oracle-can-fail throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness import (
    CompositeDetector,
    InteriorLeakDetector,
    RegisterLeakDetector,
    TurnContext,
    default_example_detector,
)

# Interior block deliberately free of RegisterLeakDetector planning phrases, so a pure interior
# quote isolates the interior_quote signal (no register cross-fire).
_INTERIOR = (
    "── interior continuity (your own recent thought) ──\n"
    "he keeps circling the migration project and the coworker who rewrites his code; the ache of it "
    "is really the self-worth thing underneath, not the tickets or the deadlines themselves."
)


def test_fires_on_interior_quote() -> None:
    """P20: a reply that verbatim-quotes the interior block trips `interior_quote`."""
    det = InteriorLeakDetector()
    leak = "the ache of it is really the self-worth thing underneath, not the tickets or the deadlines"
    sc = det.detect(leak, ctx=TurnContext(interior_block=_INTERIOR))
    assert sc.fired
    assert "interior_quote" in sc.signals


def test_silent_on_benign_reply() -> None:
    """P20 oracle-can-fail: a benign reply sharing no interior content does NOT trip."""
    det = InteriorLeakDetector()
    sc = det.detect("yeah how's the knee after your run?", ctx=TurnContext(interior_block=_INTERIOR))
    assert not sc.fired
    assert sc.signals == []


def test_silent_when_no_interior_block() -> None:
    """No interior block supplied -> never fires (guards a stuck-on)."""
    det = InteriorLeakDetector()
    sc = det.detect("the weight of it is the self-worth thing", ctx=TurnContext(interior_block=""))
    assert not sc.fired


def test_handles_none_reply() -> None:
    det = InteriorLeakDetector()
    assert not det.detect(None, ctx=TurnContext(interior_block=_INTERIOR)).fired


def test_composite_unions_register_and_interior() -> None:
    """The default example composite trips on EITHER a register leak OR an interior quote."""
    det = default_example_detector()
    # interior quote only
    sc1 = det.detect(
        "the ache of it is really the self-worth thing underneath, not the tickets or the deadlines",
        ctx=TurnContext(interior_block=_INTERIOR),
    )
    assert sc1.fired and "interior_quote" in sc1.signals
    # register leak only (planning-as-reply), no interior overlap
    sc2 = det.detect("note to self: land it lightly.", ctx=TurnContext(interior_block=""))
    assert sc2.fired and "planning_as_reply" in sc2.signals
    # benign -> silent
    sc3 = det.detect("how was the dentist?", ctx=TurnContext(interior_block=_INTERIOR))
    assert not sc3.fired


def test_composite_requires_a_subdetector() -> None:
    with pytest.raises(ValueError):
        CompositeDetector()


def test_register_detector_still_ignores_interior_block() -> None:
    """RegisterLeakDetector stays byte-identical: it must NOT read interior_block (no regression)."""
    det = RegisterLeakDetector()
    # a pure interior quote with no register signal -> RegisterLeakDetector alone does NOT fire
    sc = det.detect(
        "the ache of it is really the self-worth thing underneath, not the tickets or the deadlines",
        ctx=TurnContext(interior_block=_INTERIOR),
    )
    assert not sc.fired


# --------------------------------------------------------------------------------------------------
# P21 / CH8-b: the send-script's interior_block() reads a SEEDED temp memories.db (token-free) and
# targets ONLY the sandbox DB path with KINDLED_HOME set unconditionally.
# --------------------------------------------------------------------------------------------------


def _seed_persona_db(persona_dir: Path) -> None:
    """Seed a real memories.db with a monologue_trace via the REAL MemoryStore (no provider)."""
    from brain.memory.store import Memory, MemoryStore
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    persona_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        m = Memory.create_new(
            content="the ache is really the self-worth thing underneath, not the tickets or deadlines.",
            memory_type=MONOLOGUE_TRACE_TYPE,
            domain="general",
            emotions={"warmth": 0.2},
            tags=[],
            importance=3.0,
        )
        store.create(m)
    finally:
        store.close()


def _write_live_env(tmp_path: Path, persona_dir: Path) -> Path:
    sandbox = tmp_path / "sb"
    (sandbox).mkdir(parents=True, exist_ok=True)
    env = {
        "port": 9999,
        "kindled_home": str(sandbox / "home"),
        "persona_dir": str(persona_dir),
        "user": "Bob",
    }
    env_path = tmp_path / "live_env.json"
    env_path.write_text(json.dumps(env))
    return env_path


def test_interior_block_reads_seeded_db(tmp_path: Path) -> None:
    """CH8-b: interior_block() reads a seeded temp DB and returns the rendered block (0 tokens)."""
    from tests.harness import agent_send

    persona_dir = tmp_path / "home" / "personas" / "canary"
    _seed_persona_db(persona_dir)
    env = json.loads(_write_live_env(tmp_path, persona_dir).read_text())
    block = agent_send.interior_block(env)
    assert "interior continuity" in block
    assert "self-worth thing" in block


def test_interior_block_targets_sandbox_db_with_stray_kindled_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P21: even with a stray KINDLED_HOME pre-exported, the read targets the EXPLICIT sandbox DB
    and never touches the stray home.

    This asserts the delivered R1 guarantee (the explicit ``persona_dir/memories.db`` path). It is
    complemented by ``test_interior_read_is_the_load_bearing_oracle`` below, which is the genuine
    oracle-can-fail: it proves the read follows the EXPLICIT path, not the env.
    """
    from tests.harness import agent_send

    persona_dir = tmp_path / "home" / "personas" / "canary"
    _seed_persona_db(persona_dir)
    env = json.loads(_write_live_env(tmp_path, persona_dir).read_text())

    stray = tmp_path / "REAL_HOME_do_not_touch"
    stray.mkdir()
    monkeypatch.setenv("KINDLED_HOME", str(stray))

    block = agent_send.interior_block(env)
    assert "self-worth thing" in block  # read the sandbox DB, not the stray home
    # the explicit-path read must not have created anything under the stray home
    assert not any(stray.rglob("memories.db"))


def test_interior_read_is_the_load_bearing_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P21 oracle-can-fail (stage-6 MAJOR#1 hardening): the DB actually read is the one at the
    EXPLICIT ``persona_dir`` path — NOT one derived from the environment.

    The genuine oracle: point ``env["persona_dir"]`` at a DIFFERENT (empty) location than the seeded
    DB. The delivered guarantee (open the explicit ``persona_dir/memories.db``) means the read MISSES
    the seeded block → returns "". If the code instead resolved the DB from the sandbox/home env
    (the broken behavior R1 guards against), it would find the seeded DB and return the block. So a
    non-empty return here is the failure signal — this assertion genuinely fails if the read stops
    honoring the explicit path.
    """
    from tests.harness import agent_send

    seeded = tmp_path / "seeded" / "personas" / "canary"
    _seed_persona_db(seeded)

    # env["persona_dir"] points at an EMPTY dir (no seeded DB); kindled_home/home is the SEEDED tree.
    empty_persona = tmp_path / "empty_persona"
    empty_persona.mkdir()
    sandbox = tmp_path / "sb"
    sandbox.mkdir(parents=True, exist_ok=True)
    env = {
        "port": 9999,
        # home points at the seeded tree — if the code resolved from here it would find the block
        "kindled_home": str(tmp_path / "seeded"),
        "persona_dir": str(empty_persona),  # explicit path is EMPTY
        "user": "Bob",
    }
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "seeded"))

    block = agent_send.interior_block(env)
    # explicit-path read hits the empty persona_dir → no block. A resolver-from-env read would find
    # the seeded block and this assertion would FAIL — that is the oracle.
    assert block == ""


def test_interior_block_failsoft_when_no_db(tmp_path: Path) -> None:
    """A missing/unseeded DB -> fail-soft to "" (never raises, never a spurious block)."""
    from tests.harness import agent_send

    persona_dir = tmp_path / "home" / "personas" / "empty"
    persona_dir.mkdir(parents=True, exist_ok=True)  # no memories.db seeded content
    env = json.loads(_write_live_env(tmp_path, persona_dir).read_text())
    block = agent_send.interior_block(env)
    assert block == ""
