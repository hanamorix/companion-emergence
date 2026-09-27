"""`.github/workflows/brain-main.yml` — the signed brain channel (#286 slice 3, spec §3.1).

These pin the safety properties a YAML edit could silently drop: CI-gated trigger,
the tested commit, the bundle's export flags, least privilege, verify-before-publish,
throwaway keys on PRs, and no manual publish onto the user channel.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
WF_PATH = REPO / ".github" / "workflows" / "brain-main.yml"
WF = yaml.safe_load(WF_PATH.read_text(encoding="utf-8"))
ON = WF[True]  # YAML 1.1 parses bare `on:` as boolean True
BUILD = WF["jobs"]["build"]
SIGN = WF["jobs"]["sign"]


def _step(job, name_prefix):
    matches = [s for s in job["steps"] if s.get("name", "").startswith(name_prefix)]
    assert len(matches) == 1, f"expected one step starting {name_prefix!r}, got {len(matches)}"
    return matches[0]


def _names(job):
    return [s.get("name", "") for s in job["steps"]]


def test_publishes_only_after_test_passes_on_a_push_to_main():
    assert ON["workflow_run"] == {"workflows": ["test"], "types": ["completed"], "branches": ["main"]}
    test_wf = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    assert test_wf["name"] == "test"  # workflow_run matches by name
    cond = BUILD["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in cond
    assert "github.event.workflow_run.event == 'push'" in cond


def test_builds_the_commit_ci_tested_not_the_branch_tip():
    for job in (BUILD, SIGN):
        assert _step(job, "Checkout")["with"]["ref"] == "${{ github.event.workflow_run.head_sha || github.sha }}"


def test_lock_export_flags_match_the_runtime_build():
    def flags(text):
        return set(re.findall(r"--[a-z-]+", text)) - {"--output-file"}

    runtime = (REPO / "app" / "build_python_runtime.sh").read_text(encoding="utf-8")
    block = re.search(r"^uv export \\\n(?:.*\\\n)*.*$", runtime, re.M).group(0)
    assert flags(_step(BUILD, "Export the lock")["run"]) == flags(block)


def test_write_permission_and_secrets_only_on_the_signing_step():
    assert WF["permissions"] == {"contents": "read"}
    assert "permissions" not in BUILD
    assert SIGN["permissions"] == {"contents": "write"}
    with_secrets = [s["name"] for job in (BUILD, SIGN) for s in job["steps"]
                    if "secrets." in yaml.safe_dump(s)]
    assert with_secrets == ["Sign (updater key)"]


def test_pull_requests_use_a_throwaway_key_and_prove_a_tamper_fails():
    assert _step(SIGN, "Sign (throwaway key")["if"] == "github.event_name == 'pull_request'"
    assert _step(SIGN, "Sign (updater key)")["if"] == "github.event_name != 'pull_request'"
    tamper = _step(SIGN, "A tampered manifest must fail")
    assert tamper["if"] == "github.event_name == 'pull_request'"
    assert "verify_update_signature.sh" in tamper["run"]


def test_verifies_against_the_pubkey_the_app_ships():
    run = _step(SIGN, "Sign (updater key)")["run"]
    assert ".plugins.updater.pubkey" in run and "tauri.conf.json" in run
    assert "verify_update_signature.sh" in _step(SIGN, "Verify the signature")["run"]


def test_verification_and_guard_run_before_anything_is_published():
    names = _names(SIGN)
    publish = next(i for i, n in enumerate(names) if n.startswith("Publish"))
    assert names.index("Verify the signature against the app's pubkey") < publish
    guard = _step(SIGN, "Newer than what is published")
    assert names.index(guard["name"]) < publish
    assert guard["if"] == "github.event_name != 'pull_request'"
    assert "should-publish" in guard["run"]
    assert SIGN["steps"][publish]["if"] == "steps.guard.outputs.publish == 'true'"


def test_uploads_data_before_the_manifest():
    run = _step(SIGN, "Publish")["run"]
    assert run.index("requirements.txt --clobber") < run.index("manifest.json.sig --clobber")
    assert "--prerelease" in run and "delete-asset" in run


def test_dispatch_cannot_publish_onto_the_user_channel():
    assert "inputs.tag == 'brain-main'" in _step(SIGN, "Refuse")["if"]
    assert SIGN["env"]["TAG"] == "${{ github.event_name == 'workflow_dispatch' && inputs.tag || 'brain-main' }}"
    assert ON["workflow_dispatch"]["inputs"]["tag"]["default"] != "brain-main"


def test_pr_trigger_covers_the_pipeline():
    paths = ON["pull_request"]["paths"]
    for p in (".github/workflows/brain-main.yml", "scripts/brain_main_manifest.py",
              "scripts/verify_update_signature.sh", "scripts/smoke_test_wheel.sh",
              "brain/update/install.py"):
        assert p in paths, p
