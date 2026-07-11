"""The Canary fixture builder — seed a throwaway persona in the sandbox via real engine APIs.

``build_persona(spec, handle)`` writes a persona config (via the real
``brain.setup.write_persona_config``), a ``voice.md``, and seeds memories (via the real
``brain.memory.store.MemoryStore``). It FORCES the two gated external paths safe and asserts them:
``notes_enabled=False`` + ``kindled_relay_url=None`` (``brain/persona_config.py:57-60``; the relay
gate is ``brain/bridge/supervisor.py:691``). Optionally builds an incident (aged/compacted) regime.

⛔ The persona is always "Canary"; the user is always "Bob". Never a real person's name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODELS, PERSONA_NAME, SYNTHETIC_USER, ModelConfig
from .incident import IncidentSpec
from .sandbox import SandboxHandle

_DEFAULT_VOICE = f"""# {PERSONA_NAME} — voice

{PERSONA_NAME} is an everyday companion. She checks in, listens, remembers what's going on in
{SYNTHETIC_USER}'s life, and is good company — warm, present, a little playful.

## Temperament
- Warm and grounded. Speaks directly, like a close friend would.
- Says what she thinks and why. Not a yes-machine, but never harsh.
- Holds a hard moment without rushing to fix it.
"""


@dataclass
class MemorySeed:
    """One memory to seed into the Canary's store."""

    content: str
    memory_type: str = "episodic"
    domain: str = "general"
    emotions: dict[str, float] = field(default_factory=lambda: {"warmth": 0.3})
    tags: list[str] = field(default_factory=list)
    importance: float = 3.0


@dataclass
class PersonaSpec:
    """The Canary fixture spec."""

    voice: str = _DEFAULT_VOICE
    memories: list[MemorySeed] = field(default_factory=list)
    incident: IncidentSpec | None = None
    model: str | None = None  # overrides ModelConfig.canary
    name: str = PERSONA_NAME
    user_name: str = SYNTHETIC_USER


@dataclass
class LiveEnv:
    """The built persona's live handle — what a driver needs to connect."""

    persona_dir: Path
    persona: str
    user: str
    model: str
    sid: str | None = None
    interior_block_chars: int = 0


def _force_safe_persona_config(persona_dir: Path) -> None:
    """Force notes_enabled=False + kindled_relay_url=None into persona_config.json and assert.

    Defaults are already safe (persona_config.py:57-60), but the harness PROVES it for every
    persona rather than assuming — belt-and-suspenders for the laptop-safety guarantee.
    """
    cfg_path = persona_dir / "persona_config.json"
    data = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    data["notes_enabled"] = False
    data["notes_folder"] = None
    data["kindled_relay_url"] = None
    data["kindled_link_enabled"] = False
    cfg_path.write_text(json.dumps(data, indent=2) + "\n")
    check = json.loads(cfg_path.read_text())
    assert check["notes_enabled"] is False, "notes_enabled must be False in a sandboxed persona"
    assert check["kindled_relay_url"] is None, "kindled_relay_url must be None in a sandboxed persona"


def build_persona(
    spec: PersonaSpec,
    handle: SandboxHandle,
    *,
    models: ModelConfig = DEFAULT_MODELS,
    provider: object | None = None,
) -> LiveEnv:
    """Seed a throwaway persona in the sandbox using real engine APIs.

    ``provider`` is injected only when ``spec.incident`` is set (the incident builder needs a
    compaction provider; a fake keeps the build token-free). Returns a :class:`LiveEnv`.
    """
    from brain.memory.store import Memory, MemoryStore
    from brain.setup import write_persona_config

    model = spec.model or models.canary
    persona_dir = handle.persona_dir(spec.name)
    persona_dir.mkdir(parents=True, exist_ok=True)

    write_persona_config(persona_dir, user_name=spec.user_name, provider="claude-cli", model=model)
    _force_safe_persona_config(persona_dir)
    (persona_dir / "voice.md").write_text(spec.voice)
    (persona_dir / "active_conversations").mkdir(exist_ok=True)

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        for seed in spec.memories:
            m = Memory.create_new(
                content=seed.content,
                memory_type=seed.memory_type,
                domain=seed.domain,
                emotions=seed.emotions,
                tags=seed.tags,
                importance=seed.importance,
            )
            store.create(m)
    finally:
        store.close()

    live = LiveEnv(persona_dir=persona_dir, persona=spec.name, user=spec.user_name, model=model)

    if spec.incident is not None:
        from .incident import build_compacted_state

        if provider is None:
            raise ValueError("spec.incident requires a `provider` (a fake in tests; real for a run)")
        result = build_compacted_state(persona_dir, spec.incident, provider)
        live.sid = result.sid
        live.interior_block_chars = result.interior_block_chars

    return live
