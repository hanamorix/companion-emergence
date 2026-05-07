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
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

DEFAULT_PROVIDER = "claude-cli"
DEFAULT_SEARCHER = "ddgs"
DEFAULT_MCP_AUDIT_LOG_LEVEL = "redacted"

# Allowlists for hand-edited or migrated config files. A value outside
# the set degrades to the default with an attempt_heal anomaly logged
# rather than letting an invalid choice surface as a runtime crash at
# bridge startup or heartbeat-close time. Keep these in sync with
# brain/bridge/provider.py:get_provider and brain/search/factory.py.
KNOWN_PROVIDERS = frozenset({"claude-cli", "ollama", "fake"})
KNOWN_SEARCHERS = frozenset({"ddgs", "noop"})

logger = logging.getLogger(__name__)


def _default_persona_config_dict() -> dict:
    return {
        "provider": DEFAULT_PROVIDER,
        "searcher": DEFAULT_SEARCHER,
        "mcp_audit_log_level": DEFAULT_MCP_AUDIT_LOG_LEVEL,
        "user_name": None,
    }


@dataclass
class PersonaConfig:
    """Per-persona routing config — currently provider + searcher.

    Future fields (face, body, voice presets) will land here too — the
    file is the persona's "who am I" surface, separate from the heartbeat's
    internal calibration. Hand-edited config with wrong-type values
    degrades to defaults rather than crashing the CLI.

    user_name: name the persona's user/owner goes by in conversation.
    Used to disambiguate transcript extraction so the LLM doesn't
    conflate the user with historical figures referenced in soul
    crystallizations or memory context (Bug A from the 2026-05-05
    audit-3: extracted memories attributed Hana's actions to Jordan
    because both names appeared in the transcript via assistant
    references). When None, the extractor falls back to the legacy
    "user:" / "assistant:" labels — backward-compatible for forkers
    who haven't set the field yet.
    """

    provider: str = DEFAULT_PROVIDER
    searcher: str = DEFAULT_SEARCHER
    mcp_audit_log_level: str = DEFAULT_MCP_AUDIT_LOG_LEVEL
    user_name: str | None = None

    @classmethod
    def _parse_data(cls, data: object) -> PersonaConfig:
        """Build instance from already-parsed JSON data (dict expected)."""
        if not isinstance(data, dict):
            return cls()
        provider_raw = data.get("provider", DEFAULT_PROVIDER)
        searcher_raw = data.get("searcher", DEFAULT_SEARCHER)
        audit_raw = data.get("mcp_audit_log_level", DEFAULT_MCP_AUDIT_LOG_LEVEL)
        user_name_raw = data.get("user_name")
        provider_str = (
            provider_raw if isinstance(provider_raw, str) and provider_raw else DEFAULT_PROVIDER
        )
        provider = provider_str if provider_str in KNOWN_PROVIDERS else DEFAULT_PROVIDER
        if provider != provider_str:
            logger.warning(
                "PersonaConfig: unknown provider %r — falling back to %r",
                provider_str,
                DEFAULT_PROVIDER,
            )
        searcher_str = (
            searcher_raw if isinstance(searcher_raw, str) and searcher_raw else DEFAULT_SEARCHER
        )
        searcher = searcher_str if searcher_str in KNOWN_SEARCHERS else DEFAULT_SEARCHER
        if searcher != searcher_str:
            logger.warning(
                "PersonaConfig: unknown searcher %r — falling back to %r",
                searcher_str,
                DEFAULT_SEARCHER,
            )
        audit_level = audit_raw.strip().lower() if isinstance(audit_raw, str) else ""
        if audit_level not in {"off", "metadata", "redacted", "full"}:
            audit_level = DEFAULT_MCP_AUDIT_LOG_LEVEL
        user_name = (
            user_name_raw.strip()
            if isinstance(user_name_raw, str) and user_name_raw.strip()
            else None
        )
        return cls(
            provider=provider,
            searcher=searcher,
            mcp_audit_log_level=audit_level,
            user_name=user_name,
        )

    @classmethod
    def load_with_anomaly(cls, path: Path) -> tuple[PersonaConfig, BrainAnomaly | None]:
        """Load with self-healing from .bak rotation if corrupt.

        Returns (instance, anomaly_or_None). Missing file → defaults, no anomaly.
        Corrupt file → quarantine + restore from .bak1/.bak2/.bak3 or reset.
        """
        from brain.health.attempt_heal import attempt_heal

        data, anomaly = attempt_heal(path, _default_persona_config_dict)
        return cls._parse_data(data), anomaly

    @classmethod
    def load(cls, path: Path) -> PersonaConfig:
        instance, anomaly = cls.load_with_anomaly(path)
        if anomaly is not None:
            logger.warning(
                "PersonaConfig anomaly detected: %s action=%s file=%s",
                anomaly.kind,
                anomaly.action,
                anomaly.file,
            )
        return instance

    def save(self, path: Path) -> None:
        """Atomic save via .bak rotation (save_with_backup)."""
        from brain.health.adaptive import compute_treatment
        from brain.health.attempt_heal import save_with_backup

        payload = {
            "provider": self.provider,
            "searcher": self.searcher,
            "mcp_audit_log_level": self.mcp_audit_log_level,
            "user_name": self.user_name,
        }
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(path, payload, backup_count=treatment.backup_count)
        if treatment.verify_after_write:
            self._verify_after_write(path)

    def _verify_after_write(self, path: Path) -> None:
        """Re-read the written file; if corrupt, restore from .bak1."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("non-dict payload after write")
        except (json.JSONDecodeError, ValueError, OSError):
            logger.error(
                "PersonaConfig verify_after_write failed for %s; restoring from .bak1", path
            )
            bak1 = path.with_name(path.name + ".bak1")
            if bak1.exists():
                shutil.copy2(bak1, path)
