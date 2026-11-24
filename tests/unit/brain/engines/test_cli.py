"""Tests for the `nell dream` CLI subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def nell_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal nell persona dir with one recent memory + empty hebbian."""
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))

    from brain.paths import get_persona_dir

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)

    store = MemoryStore(db_path=persona / "memories.db")
    seed = Memory.create_new(
        content="first meeting test seed",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
    )
    seed.importance = 8.0
    store.create(seed)
    store.close()

    from brain.memory.hebbian import HebbianMatrix

    h = HebbianMatrix(db_path=persona / "hebbian.db")
    h.close()
    return persona


def test_nell_dream_dry_run_with_fake_provider(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """nell dream --dry-run --provider fake runs without writes + prints summary."""
    from brain.cli import main
    from brain.emotion.engine import ensure_engine_ready

    ensure_engine_ready(nell_persona)  # a dry run needs a switched persona (it cannot do the switch-over, it writes nothing)
    rc = main(["dream", "--persona", "nell", "--dry-run", "--provider", "fake"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "first meeting test seed" in out or "Dry run" in out


def test_nell_dream_real_cycle_with_fake_provider(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """nell dream --provider fake writes a new dream memory."""
    from brain.cli import main

    rc = main(["dream", "--persona", "nell", "--provider", "fake"])
    assert rc == 0

    # The dream is written as a gated pending-candidate, not straight to
    # memories.db. Promote the queue (id preserved) so the end-state DB
    # assertion — a real dream memory exists — still holds.
    from brain.engines.consolidation import Decision, run_consolidation

    store = MemoryStore(db_path=nell_persona / "memories.db")
    run_consolidation(
        store,
        persona_dir=store.persona_dir,
        classifier=lambda _c, _ctx: Decision("new"),
    )
    dreams = store.list_by_type("dream")
    store.close()
    assert len(dreams) == 1


def test_nell_dream_ollama_surfaces_provider_error_when_unreachable(
    nell_persona: Path,
) -> None:
    """--provider ollama now attempts a real Ollama call.

    SP-1 ships a working OllamaProvider, so the factory no longer raises
    NotImplementedError.  On CI / dev machines without a local Ollama daemon,
    the httpx request fails and surfaces as ProviderError("ollama_request").
    We mock the network layer so the test is deterministic.
    """
    from unittest.mock import patch

    import httpx

    from brain.bridge.provider import ProviderError
    from brain.cli import main

    with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
        with pytest.raises(ProviderError) as exc_info:
            main(["dream", "--persona", "nell", "--provider", "ollama"])

    assert exc_info.value.stage == "ollama_request"


def test_nell_dream_unknown_provider_fails(nell_persona: Path) -> None:
    """Unknown provider name raises ValueError."""
    from brain.cli import main

    with pytest.raises(ValueError, match="Unknown provider"):
        main(["dream", "--persona", "nell", "--provider", "nonsense"])


def test_nell_dream_unknown_persona_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persona dir missing → FileNotFoundError mentioning 'persona'."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    from brain.cli import main

    with pytest.raises(FileNotFoundError, match="persona"):
        main(["dream", "--persona", "ghost", "--provider", "fake"])


# ---- #801 criterion 5: a `nell dream` run with no bridge adds no feeling name and stores none -------------------------------

_STRAY = "stray_longing_zz"
_FILE_TEXT = '{"version": 1, "emotions": []}\n'


@pytest.fixture
def stray_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A synthetic nell persona whose only conversation memory (the dream's seed) carries a feeling name her vocabulary lacks.

    The seed row is planted as an older version stored it (`check_names=False`); her vocabulary file is a real one with no entries.
    """
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))
    from brain.paths import get_persona_dir

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)
    (persona / "emotion_vocabulary.json").write_text(_FILE_TEXT, encoding="utf-8")
    store = MemoryStore(db_path=persona / "memories.db")
    seed = Memory.create_new(
        content="first meeting test seed", memory_type="conversation", domain="us", emotions={"love": 9.0, _STRAY: 6.0}
    )
    seed.importance = 8.0
    store.create(seed, check_names=False)
    store.close()
    from brain.memory.hebbian import HebbianMatrix

    HebbianMatrix(db_path=persona / "hebbian.db").close()
    return persona


def _run_dream_and_promote(persona: Path) -> list[Memory]:
    """`nell dream` with the fake provider, then the queue promoted as the sibling test does; every memory except the seed."""
    from brain.cli import main
    from brain.emotion.engine import ensure_engine_ready
    from brain.engines.consolidation import Decision, run_consolidation

    ensure_engine_ready(persona)  # the dream reads the engine's faded feelings once the persona is switched over
    assert main(["dream", "--persona", "nell", "--provider", "fake"]) == 0
    store = MemoryStore(db_path=persona / "memories.db")
    try:
        run_consolidation(store, persona_dir=store.persona_dir, classifier=lambda _c, _ctx: Decision("new"))
        return [m for m in store.list_by_type("dream")]
    finally:
        store.close()


def test_a_dream_with_no_bridge_stores_no_memory_feeling_under_a_new_name_and_leaves_her_file_alone(stray_persona: Path) -> None:
    file_before = (stray_persona / "emotion_vocabulary.json").read_bytes()
    dreams = _run_dream_and_promote(stray_persona)

    assert len(dreams) == 1
    assert dreams[0]._stored_emotions.get("love", 0.0) > 0.0  # the framework feeling the seed carried is kept
    assert _STRAY not in dreams[0]._stored_emotions
    assert _STRAY not in (stray_persona / "pending_candidates.jsonl").read_text(encoding="utf-8")  # it never reached the queue
    assert (stray_persona / "emotion_vocabulary.json").read_bytes() == file_before  # her vocabulary file is unchanged
    assert sorted(p.name for p in stray_persona.glob("emotion_vocabulary*")) == ["emotion_vocabulary.json"]  # no backup, no heal file


def test_the_same_dream_in_the_owner_stores_the_name_and_adds_it_to_her_file(stray_persona: Path) -> None:
    """The contrast: only the owner flag differs, so the first test's result is the missing bridge, not the setup."""
    from brain.emotion import vocabulary

    vocabulary.set_bridge_owner(True)  # tests/conftest.py puts the flag back
    dreams = _run_dream_and_promote(stray_persona)

    assert len(dreams) == 1 and dreams[0]._stored_emotions.get(_STRAY, 0.0) > 0.0
    import json

    names = [e["name"] for e in json.loads((stray_persona / "emotion_vocabulary.json").read_text(encoding="utf-8"))["emotions"]]
    assert _STRAY in names
