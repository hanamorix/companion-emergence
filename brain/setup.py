"""Persona setup — testable pure functions used by `nell init`.

The `nell init` CLI subcommand wraps these in an interactive wizard.
The functions themselves are pure (no input(), no print() except via
provided callbacks) so they unit-test cleanly without a tty.

Three things every persona needs to start working correctly:

  1. A persona directory at <NELLBRAIN_HOME>/personas/<name>/.
  2. A persona_config.json with user_name set so the ingest extractor
     can disambiguate the user from historical figures (Bug A from the
     2026-05-05 audit-3).
  3. A voice.md (optional but recommended). When absent, the chat
     engine falls back to brain.chat.voice.DEFAULT_VOICE_TEMPLATE.

This module exposes:

  - validate_persona_name(name) — refuses path-traversal etc.
  - write_persona_config(persona_dir, user_name, ...) — writes the
    config preserving any existing fields.
  - install_voice_template(persona_dir, choice) — copies a starter
    voice.md (the canonical Nell example) or leaves the file absent
    so the default applies.
  - VOICE_TEMPLATES — the catalogue of available starters.

`brain.cli._init_handler` orchestrates these from argparse + tty I/O.
"""
from __future__ import annotations

import re
from pathlib import Path

VOICE_TEMPLATE_DEFAULT = "default"
VOICE_TEMPLATE_NELL_EXAMPLE = "nell-example"
VOICE_TEMPLATE_SKIP = "skip"

VOICE_TEMPLATES = {
    VOICE_TEMPLATE_DEFAULT: (
        "Use the framework's generic DEFAULT_VOICE_TEMPLATE on first "
        "chat. You can author voice.md later by writing your own file."
    ),
    VOICE_TEMPLATE_NELL_EXAMPLE: (
        "Copy the canonical Nell voice.md as a starting point. Edit it "
        "to remove Nell-specific content and add your persona's identity."
    ),
    VOICE_TEMPLATE_SKIP: (
        "Same as 'default' — no voice.md will be written. Provided for "
        "scripts that want to be explicit."
    ),
}

_NELL_VOICE_REPO_PATH = Path("docs/voice-drafts/nell-voice.md")
_PERSONA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def validate_persona_name(name: str) -> None:
    """Raise ValueError if `name` would land outside <home>/personas/.

    Persona names become directory names. Reject anything that could
    escape the personas/ root (slashes, dotdot, empty, oversize).
    """
    if not isinstance(name, str) or not _PERSONA_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid persona name {name!r} — must match "
            f"[A-Za-z0-9_-]{{1,40}} (no slashes, dots, or spaces)"
        )


def write_persona_config(
    persona_dir: Path,
    *,
    user_name: str | None,
    provider: str | None = None,
    searcher: str | None = None,
    mcp_audit_log_level: str | None = None,
) -> Path:
    """Write or update <persona_dir>/persona_config.json with user_name.

    Preserves any existing fields when the file is already present —
    this lets `nell init` run safely after `nell migrate --install-as`,
    or be re-run to change just the user_name.

    Returns the config path.
    """
    from brain.persona_config import PersonaConfig

    persona_dir.mkdir(parents=True, exist_ok=True)
    config_path = persona_dir / "persona_config.json"
    cfg = PersonaConfig.load(config_path)
    if user_name is not None:
        cfg.user_name = user_name.strip() or None
    if provider is not None:
        cfg.provider = provider
    if searcher is not None:
        cfg.searcher = searcher
    if mcp_audit_log_level is not None:
        cfg.mcp_audit_log_level = mcp_audit_log_level
    cfg.save(config_path)
    return config_path


def install_voice_template(
    persona_dir: Path,
    template: str,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Drop a starter voice.md into the persona dir based on `template`.

    Returns the written voice.md path, or None when no file is written
    (the framework's DEFAULT_VOICE_TEMPLATE applies on first chat).

    Args:
        persona_dir: Target persona's dir.
        template: One of VOICE_TEMPLATES keys ("default" / "skip" leave
            no file; "nell-example" copies the canonical Nell voice.md
            as a starting point — the user is expected to edit it).
        repo_root: Path to the companion-emergence repo (for locating
            the example voice.md). Defaults to the parent of this file's
            grandparent (which is the package root in dev installs).
            Pass an explicit path in tests.
    """
    if template not in VOICE_TEMPLATES:
        raise ValueError(
            f"unknown voice template {template!r} — must be one of "
            f"{sorted(VOICE_TEMPLATES.keys())}"
        )

    persona_dir.mkdir(parents=True, exist_ok=True)
    voice_path = persona_dir / "voice.md"

    if template in (VOICE_TEMPLATE_DEFAULT, VOICE_TEMPLATE_SKIP):
        return None

    if template == VOICE_TEMPLATE_NELL_EXAMPLE:
        if repo_root is None:
            # Default: this module lives at brain/setup.py inside the repo
            repo_root = Path(__file__).resolve().parent.parent
        src = repo_root / _NELL_VOICE_REPO_PATH
        if not src.exists():
            raise FileNotFoundError(
                f"Nell voice example not found at {src}. The repo may "
                f"have been installed without docs/. Author your voice.md "
                f"by hand or use template='default'."
            )
        voice_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return voice_path

    # Defensive — the membership check above should make this unreachable
    raise ValueError(f"unhandled voice template {template!r}")
