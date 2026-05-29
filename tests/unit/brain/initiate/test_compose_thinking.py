"""compose_tone passes thinking options when PersonaConfig has a budget."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


def _make_candidate():
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    return InitiateCandidate(
        candidate_id="c1",
        ts="2026-01-01T00:00:00+00:00",
        kind="message",
        source="dream",
        source_id="d1",
        semantic_context=SemanticContext(topic_tags=["poetry"], linked_memory_ids=[]),
    )


def test_compose_tone_passes_thinking_budget_to_provider(tmp_path: Path):
    from dataclasses import replace
    from brain.persona_config import PersonaConfig
    from brain.initiate.compose import compose_tone

    # Write config with thinking budget
    cfg = PersonaConfig.load(tmp_path / "persona_config.json")
    replace(cfg, thinking_budget_tokens=5000).save(tmp_path / "persona_config.json")

    captured_options: list = []

    class SpyProvider:
        def chat(self, messages, *, tools=None, options=None):
            captured_options.append(options or {})
            resp = MagicMock()
            resp.content = "I wanted to reach out"
            return resp

        def complete(self, prompt):
            return "subject line"

    compose_tone(
        SpyProvider(),
        subject="something I noticed",
        candidate=_make_candidate(),
        voice_template="speak gently",
        persona_dir=tmp_path,
    )

    assert len(captured_options) == 1
    opts = captured_options[0]
    assert opts.get("thinking_budget_tokens") == 5000
    assert opts.get("thinking_call_site") == "compose"


def test_compose_tone_no_thinking_when_budget_none(tmp_path: Path):
    from brain.initiate.compose import compose_tone

    captured_options: list = []

    class SpyProvider:
        def chat(self, messages, *, tools=None, options=None):
            captured_options.append(options or {})
            resp = MagicMock()
            resp.content = "gentle words"
            return resp

        def complete(self, prompt):
            return "fallback text"

    compose_tone(
        SpyProvider(),
        subject="a thought",
        candidate=_make_candidate(),
        voice_template="speak gently",
        persona_dir=tmp_path,
    )

    # When no budget, complete() is called — chat() should NOT be called
    assert len(captured_options) == 0
