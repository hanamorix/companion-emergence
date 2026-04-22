"""Configuration loader for companion-emergence.

Merges settings from three sources in increasing priority:

1. persona/<name>/persona.toml  - persona defaults (baseline)
2. .env in repo root            - local overrides
3. Environment variables        - runtime overrides (highest priority)

Each resolved value is traced so startup can report the effective source.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Config keys supported by the framework. Extended in later weeks.
_SUPPORTED_KEYS: tuple[str, ...] = (
    "BRIDGE_BIND",
    "PROVIDER",
    "MODEL",
    "NELL_IPC_JID",
)


@dataclass
class Config:
    """Resolved configuration for a persona session.

    Attributes:
        persona_name: Name of the active persona (derived from dir basename).
        bridge_bind: Host:port the bridge listens on.
        provider: LLM provider key (ollama, claude, openai, kimi).
        model: Model tag for the active provider.
        source_trace: Maps each resolved key to the source that provided it.
    """

    persona_name: str = ""
    bridge_bind: str = "127.0.0.1:8765"
    provider: str = "ollama"
    model: str = ""
    source_trace: dict[str, str] = field(default_factory=dict)


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. No shell expansion, no exports."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _load_persona_toml(persona_dir: Path) -> dict[str, str]:
    """Flatten persona.toml into the framework's config key namespace."""
    toml_path = persona_dir / "persona.toml"
    if not toml_path.exists():
        return {}
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)

    flat: dict[str, str] = {}
    bridge = data.get("bridge", {})
    if "bind" in bridge:
        flat["BRIDGE_BIND"] = str(bridge["bind"])
    model = data.get("model", {})
    if "provider" in model:
        flat["PROVIDER"] = str(model["provider"])
    if "tag" in model:
        flat["MODEL"] = str(model["tag"])
    return flat


def _read_env_vars() -> dict[str, str]:
    """Pull supported keys from os.environ."""
    return {k: v for k, v in os.environ.items() if k in _SUPPORTED_KEYS}


def load_config(
    persona_dir: Path, env_file: Path | None = None
) -> Config:
    """Load and merge config from persona.toml + .env + env vars.

    Args:
        persona_dir: Directory containing persona.toml and persona data.
        env_file: Optional path to a .env file to load.

    Returns:
        Resolved Config with source_trace recording where each value came from.
    """
    sources: list[tuple[str, dict[str, str]]] = [
        ("persona.toml", _load_persona_toml(persona_dir)),
    ]
    if env_file is not None:
        sources.append((".env", _load_env_file(env_file)))
    sources.append(("env", _read_env_vars()))

    trace: dict[str, str] = {}
    merged: dict[str, str] = {}
    for source_name, source_values in sources:
        for key, value in source_values.items():
            merged[key] = value
            trace[key] = source_name

    return Config(
        persona_name=persona_dir.name,
        bridge_bind=merged.get("BRIDGE_BIND", "127.0.0.1:8765"),
        provider=merged.get("PROVIDER", "ollama"),
        model=merged.get("MODEL", ""),
        source_trace=trace,
    )
