"""The Canary fixture builder — seed a throwaway persona in the sandbox via real engine APIs.

``build_persona(spec, handle)`` writes a persona config (via the real
``brain.setup.write_persona_config``), a ``voice.md``, and seeds memories (via the real
``brain.memory.store.MemoryStore``). Optionally builds an incident (aged/compacted) regime.

Config values are kept at their (safe) brain defaults — the harness does NOT force them. An author
opts into the two externally-facing knobs via ``PersonaSpec``: ``notes_enabled`` (a real notes write
lands under ``~/Documents/<Persona> Notes``, which the post-run ``SandboxLeak`` fingerprint CATCHES
unless that folder is a declared F5 ``editable_path`` — the leak guard is the backstop) and
``kindled_relay_url`` (a NETWORK phone-home the filesystem leak guard CANNOT catch — so setting it
emits a LOUD ``RuntimeWarning`` and it stays off by default). The REAL sandbox safety is the
isolation layer (``sandbox()``'s KINDLED_HOME/CLAUDE_CONFIG_DIR redirect + the leak fingerprint),
not forcing config values.

⛔ The persona is always "Canary"; the user is always "Bob". Never a real person's name.
"""

from __future__ import annotations

import json
import warnings
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
    editable_paths: list[Path] = field(default_factory=list)  # F5 sandbox-extension; see LiveEnv
    # Author opt-ins — safe defaults (== brain defaults), so ``PersonaSpec()`` is notes-off/relay-off.
    notes_enabled: bool = False
    # A real notes write lands under ``~/Documents/<name> Notes`` — caught by the post-run
    # ``SandboxLeak`` fingerprint unless that folder is a declared F5 ``editable_path`` (the backstop).
    kindled_relay_url: str | None = None
    # A NETWORK phone-home relay. The filesystem leak guard CANNOT catch a network call, so this stays
    # off by default and setting it emits a LOUD ``RuntimeWarning`` at build (the only guard).


@dataclass
class LiveEnv:
    """The built persona's live handle — what a driver needs to connect."""

    persona_dir: Path
    persona: str
    user: str
    model: str
    sid: str | None = None
    interior_block_chars: int = 0
    # F5: the declared editable paths (resolved), recorded here so the driver has ONE source of truth
    # to pass to ``bob.confirm_writes(persona_dir, live.editable_paths)`` — the SAME set it must also
    # hand to ``sandbox(editable_paths=...)``. A mismatch (committing to a path the sandbox did not
    # exclude) is caught fail-safe by the post-run SandboxLeak. Notes are OFF by default (the reframe
    # lands writes via Bob-confirms, not by enabling the persona's notes feature); an author may still
    # opt in via ``PersonaSpec.notes_enabled``, in which case a notes write is caught by SandboxLeak
    # unless the notes folder is a declared editable_path.
    editable_paths: list[Path] = field(default_factory=list)


def _apply_persona_config_overrides(persona_dir: Path, spec: PersonaSpec) -> None:
    """Layer the author's opt-in config knobs on top of the brain-default config — no forcing.

    The harness keeps config at its (safe) brain defaults and lets the author change them; the REAL
    sandbox safety is the isolation layer, not forcing config values (see the module docstring). When
    NEITHER opt-in is set this is a **no-op that does not rewrite the file** — brain's own
    serialization (``write_persona_config``→``cfg.save()``) stands untouched, so the default persona's
    ``persona_config.json`` is byte-identical to before this change.

    - ``spec.notes_enabled`` → write ``notes_enabled=True`` + a resolved ``notes_folder``
      (``<Documents>/<name> Notes``). A real notes write there is caught by the post-run
      ``SandboxLeak`` fingerprint unless that folder is a declared F5 ``editable_path``.
    - ``spec.kindled_relay_url`` → ALLOW it (write the URL + ``kindled_link_enabled=True``), but emit
      a LOUD ``RuntimeWarning``: a relay is a network phone-home the filesystem leak guard cannot
      catch, so off-by-default + this warning is the only guard.

    The override branch reads the config brain just wrote and re-dumps it with the opt-ins layered on
    — it does NOT thread the knobs through ``brain.setup.write_persona_config`` because that would
    require editing ``brain/`` (the writer has no such kwargs today), which the drop-in invariant
    forbids. All OTHER keys are preserved verbatim (the read-modify-write drops nothing).
    """
    if not spec.notes_enabled and spec.kindled_relay_url is None:
        return  # byte-identical default path: leave brain's serialization untouched.

    cfg_path = persona_dir / "persona_config.json"
    # The config is always written by ``write_persona_config`` just before this at the sole call site
    # (``build_persona``). A missing file here is a caller-ordering bug, not a recoverable state — an
    # empty-dict fallback would silently DROP brain's other keys when layering an opt-in, breaking the
    # key-preservation invariant (C6/C7). Fail loudly instead (NITPICK-1).
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"persona_config.json missing at {cfg_path} — _apply_persona_config_overrides must run "
            "after write_persona_config (a caller-ordering bug)"
        )
    data = json.loads(cfg_path.read_text())
    if spec.notes_enabled:
        from brain.notes.config import resolve_notes_folder

        data["notes_enabled"] = True
        data["notes_folder"] = str(resolve_notes_folder(spec.name))
    if spec.kindled_relay_url is not None:
        warnings.warn(
            "build_persona: kindled_relay_url is set — this Canary can PHONE HOME over the network "
            f"to {spec.kindled_relay_url!r}. The filesystem SandboxLeak guard CANNOT catch a network "
            "call; off-by-default + this warning is the only guard. Unset it unless you deliberately "
            "want a live relay.",
            RuntimeWarning,
            stacklevel=3,
        )
        data["kindled_relay_url"] = spec.kindled_relay_url
        data["kindled_link_enabled"] = True
    cfg_path.write_text(json.dumps(data, indent=2) + "\n")


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
    _apply_persona_config_overrides(persona_dir, spec)
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

    live = LiveEnv(
        persona_dir=persona_dir,
        persona=spec.name,
        user=spec.user_name,
        model=model,
        editable_paths=[Path(p).expanduser().resolve() for p in spec.editable_paths],
    )

    if spec.incident is not None:
        from .incident import build_compacted_state

        if provider is None:
            raise ValueError("spec.incident requires a `provider` (a fake in tests; real for a run)")
        result = build_compacted_state(persona_dir, spec.incident, provider)
        live.sid = result.sid
        live.interior_block_chars = result.interior_block_chars

    return live
