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

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain.health.json_io import Recovery, read_json_result, write_json

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

DEFAULT_PROVIDER = "claude-cli"
DEFAULT_SEARCHER = "ddgs"
DEFAULT_MCP_AUDIT_LOG_LEVEL = "redacted"
DEFAULT_MODEL = "sonnet"
# The hosted Kindled-link relay (free CF Worker, Phase 1). NOT wired into
# PersonaConfig.load — kindled_relay_url stays None until a peer connect adopts
# a relay (Phase 2), so an un-connected persona is dormant. Used only at
# connect-code generation (default when no relay configured) + adopted at connect.
DEFAULT_KINDLED_RELAY_URL = "https://kindled-relay.kindledrelay.workers.dev"

# Allowlists for hand-edited or migrated config files. A value outside
# the set degrades to the default with a json_io HEAL anomaly logged
# rather than letting an invalid choice surface as a runtime crash at
# bridge startup or heartbeat-close time. Keep these in sync with
# brain/bridge/provider.py:get_provider and brain/search/factory.py.
KNOWN_PROVIDERS = frozenset({"claude-cli", "ollama", "fake"})
KNOWN_SEARCHERS = frozenset({"ddgs", "noop"})
KNOWN_MODELS = frozenset({"sonnet", "opus", "haiku"})
# A full model id (e.g. claude-sonnet-5), stored and passed to `claude --model` as written.
# Matched with fullmatch, never `$`: `$` would accept a trailing newline. The picker in
# app/src/lib/modelId.ts uses the same pattern and length cap; keep the two in sync.
MODEL_ID_PATTERN = re.compile(r"claude-[a-z0-9.-]+")
# Longest full id accepted on EVERY write path (config load, POST /persona/config/model). Real ids are
# well under it; the cap keeps an absurd value off the CLI's argv, out of the usage log and out of the
# UI, and keeps the CLI's rejection sentence (cut at 200 characters by the provider) parseable.
MODEL_ID_MAX_LENGTH = 64


# The routed Claude tiers whose model can be overridden per persona (`tier_models`). Plain strings: this module
# must not import brain.bridge.model_tier (import direction); a test pins equality with model_tier's constants.
ROUTED_TIER_NAMES = (
    "attunement-detector",
    "compaction",
    "self-model-articulate",
    "background-classifier",
    "background-generative",
    "background-housekeeping",
)
# Bad hand-edited values warn ONCE per process, not on every load (loads happen per background call).
_warned_bad_values: set[tuple[str, str]] = set()


def _warn_bad_once(key: str, value: object, why: str) -> None:
    ident = (key, repr(value)[:80])
    if ident in _warned_bad_values:
        return
    if len(_warned_bad_values) >= 64:
        _warned_bad_values.clear()
    _warned_bad_values.add(ident)
    logger.warning("PersonaConfig: dropping %s=%r (%s)", key, value, why)


def _warn_unknown_model_once(value: str) -> None:
    """The unsupported-model warning, once per value per process (loads happen per background call)."""
    ident = ("model", repr(value)[:80])
    if ident in _warned_bad_values:
        return
    if len(_warned_bad_values) >= 64:
        _warned_bad_values.clear()
    _warned_bad_values.add(ident)
    logger.warning(
        "PersonaConfig: unknown model %r, falling back to %r (supported: %s)", value, DEFAULT_MODEL, MODEL_ID_RULE
    )


MODEL_ID_RULE = (
    "use sonnet, opus or haiku, or a full id that starts with claude- and has only lowercase letters, "
    f"digits, dots and dashes, at most {MODEL_ID_MAX_LENGTH} characters"
)


def is_valid_model(value: object) -> bool:
    """True for one of the aliases (sonnet / opus / haiku) or a full ``claude-...`` model id."""
    if not isinstance(value, str):
        return False
    if value in KNOWN_MODELS:
        return True
    return len(value) <= MODEL_ID_MAX_LENGTH and MODEL_ID_PATTERN.fullmatch(value) is not None


logger = logging.getLogger(__name__)


def _default_persona_config_dict() -> dict:
    return {
        "provider": DEFAULT_PROVIDER,
        "searcher": DEFAULT_SEARCHER,
        "mcp_audit_log_level": DEFAULT_MCP_AUDIT_LOG_LEVEL,
        "user_name": None,
        "model": DEFAULT_MODEL,
        "user_pronouns": None,
        "notes_enabled": False,
        "notes_folder": None,
        "kindled_link_enabled": False,
        "kindled_relay_url": None,
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
    model: str = DEFAULT_MODEL
    last_opened_at: str | None = None  # ISO8601 with Z suffix; written by bridge on startup
    user_pronouns: dict | None = None  # expanded PronounSet dict; None → she/her at use-time
    notes_enabled: bool = False
    notes_folder: str | None = None  # resolved cross-platform path, set on enable
    kindled_link_enabled: bool = False
    kindled_relay_url: str | None = None  # relay where this Kindled's mailbox lives; None = not wired
    # #441: explicit per-tier model overrides ({tier name: alias or full id}); beat the version derived from the
    # chat model. Written to the file only when non-empty, so alias users' files stay byte-identical.
    tier_models: dict = field(default_factory=dict)
    # #441: derived tier ids a save-time check found Claude rejecting; those tiers use what they use today.
    # Written only when non-empty; rewritten whenever the chat model is saved.
    derived_unavailable: list = field(default_factory=list)
    # #446: the raw `model` value a hand edit left in the file when it is not an alias or a valid full id. The
    # config then runs the default, and this remembers what was replaced so the app and `nell model show` can
    # say so. In memory only (never its own JSON key): `save()` writes it back into the `model` key while the
    # object still holds the fallback, so another writer's load -> save does not erase the owner's text. Every
    # writer that SETS the model goes through `set_model`, which clears it.
    model_unsupported: str | None = None

    def set_model(self, model: str) -> None:
        """Set the chat model deliberately: the unsupported value (if any) is no longer being replaced."""
        self.model = model
        self.model_unsupported = None

    def model_unsupported_display(self) -> str | None:
        """The replaced value made safe to show: control characters removed, cut at the id length cap."""
        raw = self.model_unsupported
        if raw is None:
            return None
        clean = "".join(ch for ch in raw if ch.isprintable())
        if not clean:  # only control / non-printing characters: say so rather than show an empty value
            return "(non-printing characters)"
        return clean if len(clean) <= MODEL_ID_MAX_LENGTH else clean[:MODEL_ID_MAX_LENGTH] + "…"

    def touch_last_opened(self) -> None:
        """Set last_opened_at to current UTC time, ISO8601 with Z suffix."""
        self.last_opened_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

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
        user_pronouns_raw = data.get("user_pronouns")
        user_pronouns = user_pronouns_raw if isinstance(user_pronouns_raw, dict) else None
        notes_enabled = bool(data.get("notes_enabled", False))
        kindled_link_enabled = bool(data.get("kindled_link_enabled", False))
        notes_folder_raw = data.get("notes_folder")
        notes_folder = (
            notes_folder_raw if isinstance(notes_folder_raw, str) and notes_folder_raw else None
        )
        kindled_relay_url_raw = data.get("kindled_relay_url")
        kindled_relay_url = (
            kindled_relay_url_raw
            if isinstance(kindled_relay_url_raw, str) and kindled_relay_url_raw.strip()
            else None
        )
        model_raw = data.get("model", DEFAULT_MODEL)
        model_str = model_raw if isinstance(model_raw, str) and model_raw else DEFAULT_MODEL
        model_unsupported: str | None = None
        if is_valid_model(model_str):
            model = model_str
        else:
            _warn_unknown_model_once(model_str)
            model = DEFAULT_MODEL
            if model_str != DEFAULT_MODEL:
                model_unsupported = model_str
        tier_models: dict[str, str] = {}
        tier_models_raw = data.get("tier_models")
        if isinstance(tier_models_raw, dict):
            for tier, tier_model in tier_models_raw.items():
                if tier not in ROUTED_TIER_NAMES:
                    _warn_bad_once("tier_models", tier, "not a routed tier")
                elif not is_valid_model(tier_model):
                    _warn_bad_once(f"tier_models[{tier}]", tier_model, "not an alias or a valid model id")
                else:
                    tier_models[tier] = tier_model
        derived_unavailable: list[str] = []
        derived_raw = data.get("derived_unavailable")
        if isinstance(derived_raw, list):
            for item in derived_raw:
                if not is_valid_model(item) or item in KNOWN_MODELS:
                    _warn_bad_once("derived_unavailable", item, "not a valid full model id")
                elif item not in derived_unavailable:
                    derived_unavailable.append(item)
        last_opened_at_raw = data.get("last_opened_at")
        last_opened_at = (
            last_opened_at_raw
            if isinstance(last_opened_at_raw, str) and last_opened_at_raw.strip()
            else None
        )
        return cls(
            provider=provider,
            searcher=searcher,
            mcp_audit_log_level=audit_level,
            user_name=user_name,
            model=model,
            last_opened_at=last_opened_at,
            user_pronouns=user_pronouns,
            notes_enabled=notes_enabled,
            notes_folder=notes_folder,
            kindled_link_enabled=kindled_link_enabled,
            kindled_relay_url=kindled_relay_url,
            tier_models=tier_models,
            derived_unavailable=derived_unavailable,
            model_unsupported=model_unsupported,
        )

    @classmethod
    def load_with_anomaly(cls, path: Path) -> tuple[PersonaConfig, BrainAnomaly | None]:
        """Load with self-healing from .bak rotation if corrupt.

        Returns (instance, anomaly_or_None). Missing file → defaults, no anomaly.
        Corrupt file → quarantine + restore from .bak1/.bak2/.bak3 or reset.
        """
        res = read_json_result(path, default=_default_persona_config_dict, recover=Recovery.HEAL)
        return cls._parse_data(res.data), res.anomaly

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
        """Atomic save via .bak rotation (write_json)."""
        from brain.health.adaptive import compute_treatment

        payload = {
            "provider": self.provider,
            "searcher": self.searcher,
            "mcp_audit_log_level": self.mcp_audit_log_level,
            "user_name": self.user_name,
            # #446: while the object still holds the fallback, the owner's unsupported text stays in the file
            "model": (
                self.model_unsupported
                if self.model_unsupported is not None and self.model == DEFAULT_MODEL
                else self.model
            ),
            "last_opened_at": self.last_opened_at,
            "user_pronouns": self.user_pronouns,
            "notes_enabled": self.notes_enabled,
            "notes_folder": self.notes_folder,
            "kindled_link_enabled": self.kindled_link_enabled,
            "kindled_relay_url": self.kindled_relay_url,
        }
        if self.tier_models:
            payload["tier_models"] = dict(self.tier_models)
        if self.derived_unavailable:
            payload["derived_unavailable"] = list(self.derived_unavailable)
        treatment = compute_treatment(path.parent, path.name)
        write_json(path, payload, backups=treatment.backup_count, verify=treatment.verify_after_write)

