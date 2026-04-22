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

# Framework-level defaults, applied as the lowest-priority source so that
# source_trace explicitly records "default" for unset keys (startup logging
# can then distinguish "value came from default" from "value missing").
_DEFAULTS: dict[str, str] = {
    "BRIDGE_BIND": "127.0.0.1:8765",
    "PROVIDER": "ollama",
    "MODEL": "",
    "NELL_IPC_JID": "",
}


@dataclass
class Config:
    """Resolved configuration for a persona session.

    Attributes:
        persona_name: Name of the active persona (derived from dir basename).
        bridge_bind: Host:port the bridge listens on.
        provider: LLM provider key (ollama, claude, openai, kimi).
        model: Model tag for the active provider.
        ipc_jid: IPC target for outbox delivery (e.g. WhatsApp JID). Empty
            disables outbox→IPC integration.
        source_trace: Maps each resolved key to the source that provided it.
    """

    persona_name: str = ""
    bridge_bind: str = "127.0.0.1:8765"
    provider: str = "ollama"
    model: str = ""
    ipc_jid: str = ""
    source_trace: dict[str, str] = field(default_factory=dict)


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file.

    Supports: comment lines starting with `#`, blank lines, surrounding
    single/double quotes, inline comments after the value (`KEY=val # note`).
    Does not support: shell expansion, `export` prefix, multi-line values.
    Always read as UTF-8 regardless of platform default encoding.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # Strip inline comment (anything after a ` #` segment). The leading
        # space is required so `#` inside a quoted URL or token survives.
        if " #" in value:
            value = value[: value.index(" #")].rstrip()
        result[key.strip()] = value.strip('"').strip("'")
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
        ("default", dict(_DEFAULTS)),
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
        bridge_bind=merged["BRIDGE_BIND"],
        provider=merged["PROVIDER"],
        model=merged["MODEL"],
        ipc_jid=merged["NELL_IPC_JID"],
        source_trace=trace,
    )
