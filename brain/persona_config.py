"""Per-persona configuration — provider + searcher routing.

Lives at `{persona_dir}/persona_config.json`. The brain owns these choices:
the user surfaces in the GUI are name / cadence / face-body / generated
documents, *not* "which LLM". The framework picks `claude-cli` + `ddgs` as
sensible defaults; a persona's owner (developer, or future GUI tooling)
can override them in this file. CLI `--provider` / `--searcher` flags are
developer overrides only — they don't get written back to the file.

See `docs/superpowers/audits/2026-04-25-principle-alignment-audit.md`
(PR-B) for the principle behind this split.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROVIDER = "claude-cli"
DEFAULT_SEARCHER = "ddgs"


@dataclass
class PersonaConfig:
    """Per-persona routing config — currently provider + searcher.

    Future fields (face, body, voice presets) will land here too — the
    file is the persona's "who am I" surface, separate from the heartbeat's
    internal calibration. Hand-edited config with wrong-type values
    degrades to defaults rather than crashing the CLI.
    """

    provider: str = DEFAULT_PROVIDER
    searcher: str = DEFAULT_SEARCHER

    @classmethod
    def load(cls, path: Path) -> PersonaConfig:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()

        provider_raw = data.get("provider", DEFAULT_PROVIDER)
        searcher_raw = data.get("searcher", DEFAULT_SEARCHER)
        provider = provider_raw if isinstance(provider_raw, str) and provider_raw else DEFAULT_PROVIDER
        searcher = searcher_raw if isinstance(searcher_raw, str) and searcher_raw else DEFAULT_SEARCHER
        return cls(provider=provider, searcher=searcher)

    def save(self, path: Path) -> None:
        """Atomic save via .new + os.replace — same pattern as HeartbeatConfig."""
        payload = {"provider": self.provider, "searcher": self.searcher}
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
