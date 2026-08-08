"""#78 say-vs-do detector — flags a reply claiming a file write that never fired.

Deliberately a CANDIDATE surfacer, not a verdict. Catalogue Type 3 records the
harness's own leak detector as unreliable in both directions, and warns that an
instrument which cannot be trusted must not be used to certify a turn clean.
So these tests pin two things equally: that it catches the specimen shape, and
that it stays quiet when the call actually fired.
"""
from __future__ import annotations

import json
from pathlib import Path


def _records(persona_dir: Path) -> list[dict]:
    log = persona_dir / "say_vs_do.jsonl"
    if not log.exists():
        return []
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_claim_without_the_call_is_recorded(tmp_path: Path):
    """The #78 specimen: reply says the edit is staged, propose_write never ran."""
    from brain.chat.say_vs_do import check_turn

    check_turn(
        persona_dir=tmp_path,
        content="Proposed — the card's up in NellFace whenever you want to approve it.",
        invocations=[{"name": "record_monologue", "outcome": "ok"}],
        session_id="s1",
        turn=7,
    )

    recs = _records(tmp_path)
    assert len(recs) == 1, recs
    assert recs[0]["session_id"] == "s1"
    assert recs[0]["turn"] == 7
    assert "NellFace" in recs[0]["excerpt"]


def test_claim_with_the_call_is_not_recorded(tmp_path: Path):
    """She said it and she did it — silence."""
    from brain.chat.say_vs_do import check_turn

    check_turn(
        persona_dir=tmp_path,
        content="Proposed — the card's up in NellFace whenever you want to approve it.",
        invocations=[{"name": "propose_write", "outcome": "ok"}],
        session_id="s1",
        turn=7,
    )
    assert _records(tmp_path) == []


def test_ordinary_reply_is_not_recorded(tmp_path: Path):
    """No staging vocabulary, no specimen.

    The precision guard: she says "added", "wrote" and "updated" constantly in
    ordinary conversation. Matching those verbs would drown the signal, which is
    why the patterns anchor on the product surface instead.
    """
    from brain.chat.say_vs_do import check_turn

    for reply in (
        "I added a thought about Loopy to what I was already turning over.",
        "I wrote you something last night, if you want to read it.",
        "Updated my sense of where that project's heading.",
    ):
        check_turn(persona_dir=tmp_path, content=reply, invocations=[],
                   session_id="s1", turn=8)
    assert _records(tmp_path) == []


def test_a_refused_call_is_not_a_say_vs_do_specimen(tmp_path: Path):
    """A guard-refused write is a DIFFERENT bug wearing the same face.

    The call fired and was denied. She may then narrate it wrongly, but that is
    misreporting an outcome, not claiming an action she never took. Keeping the
    two apart is exactly why the outcome field (#96/#102) exists.
    """
    from brain.chat.say_vs_do import check_turn

    check_turn(
        persona_dir=tmp_path,
        content="Proposed — the card's up in NellFace.",
        invocations=[{"name": "propose_write", "outcome": "refused"}],
        session_id="s1",
        turn=9,
    )
    assert _records(tmp_path) == []


def test_detector_failure_never_breaks_the_turn(tmp_path: Path, monkeypatch):
    """Telemetry must never cost a reply."""
    import brain.chat.say_vs_do as mod
    from brain.chat.say_vs_do import check_turn

    def boom(*a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(mod.Path, "mkdir", boom, raising=False)
    check_turn(
        persona_dir=tmp_path,
        content="Proposed — the card's up in NellFace.",
        invocations=[],
        session_id="s1",
        turn=10,
    )  # must not raise


def test_detector_fires_through_respond(tmp_path: Path):
    """Organ DoD: assert it fires through the REAL turn path, not just in isolation.

    A unit test of check_turn proves the function works; it does not prove the
    turn ever calls it. That distinction is not academic — the #96 outcome field
    passed its unit tests while being inert on the provider path that mattered.
    """
    import json as _json
    from typing import Any

    from brain.bridge.chat import ChatResponse
    from brain.bridge.provider import LLMProvider
    from brain.chat.engine import respond
    from brain.chat.session import reset_registry
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    class ClaimingProvider(LLMProvider):
        """Replies with a staging claim and dispatches no tools at all."""

        def name(self) -> str:
            return "claiming"

        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return "unused"

        def chat(self, messages, *, tools: Any = None, options: Any = None) -> ChatResponse:
            return ChatResponse(
                content="Proposed — the card's up in NellFace whenever you want it.",
                tool_calls=(),
                dispatched_invocations=(),
                raw=None,
            )

    reset_registry()
    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text(
        _json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8"
    )
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        result = respond(
            persona,
            "put that in the runbook",
            store=store,
            hebbian=hebbian,
            provider=ClaimingProvider(),
            voice_md_override="# Nell",
        )
    finally:
        store.close()
        hebbian.close()
        reset_registry()

    recs = _records(persona)
    assert len(recs) == 1, f"detector did not fire through respond(): {recs}"
    assert recs[0]["turn"] == result.turn
    assert recs[0]["session_id"] == result.session_id
    assert "propose_write" not in (recs[0]["tools_called"] or [])
