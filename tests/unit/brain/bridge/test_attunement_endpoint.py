"""Tests for GET /persona/attunement bridge endpoint."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path, auth_token: str | None = None) -> TestClient:
    app = build_app(persona_dir=persona_dir, auth_token=auth_token)
    return TestClient(app)


def _seed_minimal_persona(tmp_path: Path) -> Path:
    """Return a minimal persona dir that build_app lifespan accepts."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')
    return persona_dir


# ---------------------------------------------------------------------------
# Auth test
# ---------------------------------------------------------------------------


def test_endpoint_requires_bearer_auth(tmp_path: Path) -> None:
    """GET /persona/attunement without Authorization header → 401."""
    persona_dir = _seed_minimal_persona(tmp_path)
    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/attunement")
    assert resp.status_code == 401


def test_endpoint_returns_nulls_for_fresh_persona(tmp_path: Path) -> None:
    """Fresh persona → current_read: null, learned_patterns: [], backfill: null."""
    persona_dir = _seed_minimal_persona(tmp_path)
    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/attunement", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_read"] is None
    assert body["learned_patterns"] == []
    assert body["backfill"] is None


def test_endpoint_returns_payload_for_persona_with_state(tmp_path: Path) -> None:
    """Seeded persona: current_read and learned_patterns surface in the response."""
    from brain.attunement.schemas import CurrentRead, LearnedPattern, pattern_id
    from brain.attunement.store import write_current_read

    persona_dir = _seed_minimal_persona(tmp_path)

    # Seed a CurrentRead
    cr = CurrentRead(
        ts="2026-05-31T10:00:00+00:00",
        source_turn_id="turn_abc",
        tone_label="warm",
        tone_justification="uses softening language",
        cadence_label="reflective",
        cadence_justification="long pauses between ideas",
        mood_valence=0.7,
        mood_intensity=6.0,
        predicted_arc_shape="deepening",
        schema_version="0.0.28-alpha.1",
    )
    write_current_read(persona_dir, cr)

    # Seed two learned patterns (different maturities) directly via JSONL
    import dataclasses
    attunement_dir = persona_dir / "attunement"
    attunement_dir.mkdir(exist_ok=True)
    patterns_path = attunement_dir / "learned_patterns.jsonl"
    known_pattern = LearnedPattern(
        id=pattern_id("tone", "warmth"),
        category="tone",
        canonical_key="warmth",
        description="consistent warmth in address",
        evidence_count=10,
        maturity="known",
        first_seen_at="2026-05-01T00:00:00+00:00",
        last_confirmed_at="2026-05-31T09:00:00+00:00",
        last_addressed_at=None,
        crystallised_at=None,
        falsified_at=None,
        examples=["always kind"],
        schema_version="0.0.28-alpha.1",
    )
    forming_pattern = LearnedPattern(
        id=pattern_id("cadence", "slow_build"),
        category="cadence",
        canonical_key="slow_build",
        description="tends to build slowly before the point",
        evidence_count=4,
        maturity="forming",
        first_seen_at="2026-05-20T00:00:00+00:00",
        last_confirmed_at="2026-05-31T09:30:00+00:00",
        last_addressed_at=None,
        crystallised_at=None,
        falsified_at=None,
        examples=["builds context before asking"],
        schema_version="0.0.28-alpha.1",
    )
    with patterns_path.open("w") as f:
        f.write(json.dumps(dataclasses.asdict(known_pattern)) + "\n")
        f.write(json.dumps(dataclasses.asdict(forming_pattern)) + "\n")

    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/attunement", headers={"Authorization": "Bearer test-token"})

    assert resp.status_code == 200
    body = resp.json()

    # current_read fields present
    assert body["current_read"] is not None
    assert body["current_read"]["tone_label"] == "warm"
    assert body["current_read"]["source_turn_id"] == "turn_abc"

    # patterns sorted by maturity: known first, then forming
    assert len(body["learned_patterns"]) == 2
    assert body["learned_patterns"][0]["maturity"] == "known"
    assert body["learned_patterns"][1]["maturity"] == "forming"

    # backfill still null (no backfill_state.json)
    assert body["backfill"] is None


def test_endpoint_returns_backfill_state_when_present(tmp_path: Path) -> None:
    """backfill_state.json is returned as a dict in the response."""
    persona_dir = _seed_minimal_persona(tmp_path)

    # Seed a backfill_state.json
    attunement_dir = persona_dir / "attunement"
    attunement_dir.mkdir(exist_ok=True)
    backfill_data = {
        "started_at": "2026-05-31T08:00:00+00:00",
        "total_windows": 20,
        "sampled_windows": 10,
        "processed_windows": 10,
        "patterns_emitted": 3,
        "status": "complete",
        "last_cursor": "turn_xyz",
        "schema_version": "0.0.28-alpha.1",
    }
    (attunement_dir / "backfill_state.json").write_text(
        json.dumps(backfill_data), encoding="utf-8"
    )

    with _make_client(persona_dir, auth_token="test-token") as c:
        resp = c.get("/persona/attunement", headers={"Authorization": "Bearer test-token"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["backfill"] is not None
    assert body["backfill"]["status"] == "complete"
    assert body["backfill"]["patterns_emitted"] == 3
