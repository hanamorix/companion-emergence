"""`.github/workflows/runtime-build.yml` PR-trigger paths (final-review finding 11).

`nell update`'s CLI plumbing (brain/cli.py's _update_handler, brain/paths.py's
get_home, brain/bridge/state_file.py's pid_is_alive used by the overlay lock) is
load-bearing for the overlay e2e this workflow runs — a PR touching only those
files should still trigger it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "runtime-build.yml"


def test_pr_trigger_paths_include_the_overlay_cli_plumbing():
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    paths = data[True]["pull_request"]["paths"]  # YAML 1.1 parses bare `on:` as boolean True
    for expected in ("brain/cli.py", "brain/paths.py", "brain/bridge/state_file.py"):
        assert expected in paths, f"{expected} missing from {WORKFLOW}'s PR trigger paths"
