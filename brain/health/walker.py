"""Proactive walk over every persona file — used by `nell health check`
and triggered automatically when a heartbeat tick produces >=2 anomalies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.health.anomaly import BrainAnomaly, BrainIntegrityError
from brain.health.attempt_heal import attempt_heal

# Atomic-rewrite files this walker checks. Each entry: filename -> default dict.
#
# When the soul module lands as a Phase 2a-extension, add `soul.json` here
# with default `{"version": 1, "crystallizations": []}` (or whatever the
# soul module's schema settles on). See spec §9.1 for the full plan.
_DEFAULTS: dict[str, dict] = {
    "user_preferences.json": {"dream_every_hours": 24.0},
    "persona_config.json": {"provider": "claude-cli", "searcher": "ddgs"},
    "heartbeat_config.json": {},  # all-default HeartbeatConfig
    "heartbeat_state.json": {},  # triggers fresh-init via load fallback
    "interests.json": {"version": 1, "interests": []},
    "reflex_arcs.json": {"version": 1, "arcs": []},
    "emotion_vocabulary.json": {"version": 1, "emotions": []},
}


# Text identity files checked separately (voice.md — plain-text, not JSON).
_TEXT_IDENTITY_FILES: tuple[str, ...] = ("voice.md",)


def walk_persona(persona_dir: Path) -> list[BrainAnomaly]:
    """Check every persona file. Heal what's healable; report anomalies.

    Iterates atomic-rewrite files via attempt_heal (captures corruption +
    repairs in one shot) then opens each SQLite store so the constructor's
    PRAGMA integrity_check runs. Returns an empty list when everything is
    healthy.
    """
    anomalies: list[BrainAnomaly] = []

    # Atomic-rewrite file scan — d=default captures the dict at iteration time
    # to avoid Python's late-binding closure gotcha.
    for name, default in _DEFAULTS.items():
        path = persona_dir / name
        _, anomaly = attempt_heal(path, default_factory=lambda d=default: d)
        if anomaly is not None:
            anomalies.append(anomaly)

    # Plain-text identity file scan (voice.md).
    for name in _TEXT_IDENTITY_FILES:
        path = persona_dir / name
        if not path.exists():
            continue  # Missing voice.md is fine — created on first chat turn.
        from brain.chat.voice import load_voice

        _, anomaly = load_voice(persona_dir)
        if anomaly is not None:
            anomalies.append(anomaly)
        break  # load_voice checks voice.md by name; only one text file for now

    # SQLite integrity — constructor runs PRAGMA integrity_check; catch failures.
    # When SP-5 (soul) added crystallizations.db, walker needs to scan it too —
    # otherwise corrupt soul data goes undetected by `nell health check`.
    for db_name in ("memories.db", "hebbian.db", "crystallizations.db"):
        db_path = persona_dir / db_name
        if not db_path.exists():
            continue
        try:
            if db_name == "memories.db":
                from brain.memory.store import MemoryStore

                MemoryStore(db_path=db_path).close()
            elif db_name == "hebbian.db":
                from brain.memory.hebbian import HebbianMatrix

                HebbianMatrix(db_path=db_path).close()
            else:  # crystallizations.db
                from brain.soul.store import SoulStore

                SoulStore(db_path=db_path).close()
        except BrainIntegrityError as exc:
            anomalies.append(
                BrainAnomaly(
                    timestamp=datetime.now(UTC),
                    file=db_name,
                    kind="sqlite_integrity_fail",
                    action="alarmed_unrecoverable",
                    quarantine_path=None,
                    likely_cause="disk",
                    detail=exc.detail[:500],
                )
            )

    return anomalies
