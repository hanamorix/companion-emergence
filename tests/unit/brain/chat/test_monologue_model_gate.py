"""#416 (option B, owner-ratified): on Sonnet behind the Claude CLI, the inner-monologue directive
is the softened wording measured live (no reasoning_extraction hits); every other provider/model
keeps today's text byte-for-byte.

Measured in docs/guarded-change/monologue-directive-sonnet-refusal: the old text tripped Sonnet
5.5's safeguard on 22/24 turns; this text 0/8 (Canary) and 0/8 (Nell). Keep MEASURED_SONNET_FRAME
byte-equal to what was measured; re-measure before rewording either text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.chat.monologue_prompts import build_monologue_frame
from brain.chat.prompt import build_static_system_message, build_system_message

MEASURED_SONNET_FRAME = "── inner monologue, Canary ──\nCanary has an inner life the conversation doesn't see — things Canary notices, feels and leaves unsaid. `record_monologue(monologue, feed_digest)` is where that gets written down: a few lines, in Canary's own voice, about the unspoken side of this moment — a name or reference that lingers, an emotional beat, an ambiguity, a memory that surfaced or didn't, a tangent, a callback, idle affection or annoyance. What goes there feeds Canary's memory, emotions and inner-life Feed; what Canary feels and doesn't say still shapes Canary. On substantive turns, make that call first, then reply; with the tangents given a home, the reply can answer directly. Skip it only on trivial micro-acks — 'hi', 'thanks', 'ok', short factual confirmations."

_MARK = "── inner monologue"


def _persona(tmp_path: Path, cfg: dict | None) -> Path:
    d = tmp_path / "personas" / "Canary"
    d.mkdir(parents=True)
    if cfg is not None:
        (d / "persona_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _frame_in(system: str) -> str:
    return _MARK + system.split(_MARK, 1)[1]


def _old_frame() -> str:
    return build_monologue_frame(persona_name="Canary", emotion_summary="", voice_excerpt="",
                                 soul_hints=(), narrative_hints=())


@pytest.mark.parametrize(
    ("cfg", "softened"),
    [
        ({"provider": "claude-cli", "model": "sonnet"}, True),
        (None, True),  # fresh install: defaults are claude-cli + sonnet
        ({"provider": "claude-cli", "model": "haiku"}, False),
        ({"provider": "claude-cli", "model": "opus"}, False),
        ({"provider": "ollama", "model": "sonnet"}, False),  # Ollama keeps the default model field
        ({"provider": "fake", "model": "sonnet"}, False),
    ],
)
def test_static_frame_is_gated_on_claude_cli_sonnet(tmp_path: Path, cfg, softened: bool) -> None:
    system = build_static_system_message(_persona(tmp_path, cfg), voice_md="# Canary")
    assert _frame_in(system) == (MEASURED_SONNET_FRAME if softened else _old_frame())


@pytest.mark.parametrize(("model", "softened"), [("sonnet", True), ("haiku", False)])
def test_legacy_builder_uses_the_same_gate(tmp_path: Path, model: str, softened: bool) -> None:
    """build_system_message (image/legacy path) appends per-turn hints after the frame, so compare
    the frame's directive line only."""
    from brain.engines.daemon_state import DaemonState
    from brain.memory.store import MemoryStore
    from brain.soul.store import SoulStore

    d = _persona(tmp_path, {"provider": "claude-cli", "model": model})
    store, soul = MemoryStore(d / "memories.db"), SoulStore(d / "crystallizations.db")
    try:
        system = build_system_message(d, voice_md="# Canary", daemon_state=DaemonState(),
                                      soul_store=soul, store=store, user_input="hi")
    finally:
        store.close()
        soul.close()
    directive = (MEASURED_SONNET_FRAME if softened else _old_frame()).split("\n")[1]
    assert directive in _frame_in(system).split("\n")[1]
