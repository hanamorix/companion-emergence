"""ram-spike-fix INC-6, C12: the REAL wiring — server.py's lifespan must call
migrate_idle_keys() against the actual KINDLED_HOME tunables.json, not the
persona directory. Stage-6 code red-team BLOCKER (caught before ship): the
first cut of this call site passed `persona_dir` instead of
`paths.get_home()`, silently no-oping every migration in production while
every direct unit test of `migrate_idle_keys` (test_tunables_migration.py)
passed since they call the function directly with a fabricated path — this
test exercises the REAL call site through `build_app`'s lifespan instead."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def test_migration_runs_against_kindled_home_not_persona_dir(
    persona_dir: Path, tmp_path: Path
) -> None:
    kindled_home = Path(os.environ["KINDLED_HOME"])
    assert kindled_home != persona_dir, "sanity: fixtures must differ for this test to mean anything"
    kindled_home.mkdir(parents=True, exist_ok=True)
    (kindled_home / "tunables.json").write_text(
        json.dumps(
            {"_readme": "x", "defaults": {}, "overrides": {"throttle.background_min_idle_seconds": 450.0}},
            indent=2,
        ),
        encoding="utf-8",
    )

    # A file at persona_dir/tunables.json must NOT exist beforehand — proving
    # the migration can't be accidentally "succeeding" against a stray copy.
    assert not (persona_dir / "tunables.json").exists()

    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass  # lifespan runs the migration synchronously on __enter__

    migrated = json.loads((kindled_home / "tunables.json").read_text(encoding="utf-8"))
    overrides = migrated.get("overrides", {})
    assert "throttle.background_min_idle_seconds" not in overrides
    assert overrides.get("chat.idle_lull_seconds") == 450.0
    assert not (persona_dir / "tunables.json").exists(), (
        "the migration must never create a tunables.json under the persona dir"
    )
