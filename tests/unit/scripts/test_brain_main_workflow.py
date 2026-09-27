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
PUBLISH = WF["jobs"]["publish"]


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
    for job in (BUILD, SIGN, PUBLISH):
        assert _step(job, "Checkout")["with"]["ref"] == "${{ github.event.workflow_run.head_sha || github.sha }}"


def test_lock_export_flags_match_the_runtime_build():
    def flags(text):
        return set(re.findall(r"--[a-z-]+", text)) - {"--output-file"}

    runtime = (REPO / "app" / "build_python_runtime.sh").read_text(encoding="utf-8")
    block = re.search(r"^uv export \\\n(?:.*\\\n)*.*$", runtime, re.M).group(0)
    assert flags(_step(BUILD, "Export the lock")["run"]) == flags(block)


def test_write_permission_is_confined_to_the_publish_job():
    assert WF["permissions"] == {"contents": "read"}
    assert "permissions" not in BUILD
    assert "permissions" not in SIGN
    assert PUBLISH["permissions"] == {"contents": "write"}
    with_secrets = [s["name"] for job in (BUILD, SIGN, PUBLISH) for s in job["steps"]
                    if "secrets." in yaml.safe_dump(s)]
    assert with_secrets == ["Sign (updater key)"]
    assert "github.token" in yaml.safe_dump(PUBLISH)
    assert "github.token" not in yaml.safe_dump(BUILD)
    assert "github.token" not in yaml.safe_dump(SIGN)


def test_tag_move_failure_is_a_warning_not_a_job_failure():
    run = _step(PUBLISH, "Publish")["run"]
    patch_line = next(line for line in run.splitlines() if "git/refs/tags/$TAG" in line)
    assert "::warning::" in patch_line


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


def test_verification_happens_in_sign_and_guard_runs_before_publish():
    assert _step(SIGN, "Verify the signature against the app's pubkey")
    assert "sign" in PUBLISH["needs"]
    assert PUBLISH["if"] == "github.event_name != 'pull_request'"
    names = _names(PUBLISH)
    publish = next(i for i, n in enumerate(names) if n.startswith("Publish"))
    guard = _step(PUBLISH, "Newer than what is published")
    assert names.index(guard["name"]) < publish
    assert "should-publish" in guard["run"]
    assert PUBLISH["steps"][publish]["if"] == "steps.guard.outputs.publish == 'true'"


def test_concurrency_is_scoped_to_the_publish_job():
    assert "concurrency" not in WF
    assert PUBLISH["concurrency"]["cancel-in-progress"] is False
    assert PUBLISH["concurrency"]["group"] == (
        "brain-main-publish-${{ github.event_name == 'workflow_dispatch' && inputs.tag || 'brain-main' }}"
    )


def test_publish_uploads_data_before_the_manifest():
    run = _step(PUBLISH, "Publish")["run"]
    assert run.index("requirements.txt --clobber") < run.index("manifest.json.sig --clobber")
    assert "--prerelease" in run and "delete-asset" in run


def test_workflow_dispatch_cannot_publish_onto_the_user_channel():
    assert PUBLISH["env"]["TAG"] == "${{ github.event_name == 'workflow_dispatch' && inputs.tag || 'brain-main' }}"
    assert ON["workflow_dispatch"]["inputs"]["tag"]["default"] != "brain-main"


def test_dispatch_is_refused_off_the_brain_main_allow_list_or_off_main():
    refuse = _step(SIGN, "Refuse a dispatch outside brain-main-* test tags or off main")
    assert refuse["if"] == "github.event_name == 'workflow_dispatch'"
    assert refuse["env"] == {"TAG_IN": "${{ inputs.tag }}"}
    assert "${{" not in refuse["run"]
    assert "TAG_IN" in refuse["run"]
    assert r"^brain-main-[a-z0-9-]+$" in refuse["run"]
    assert "refs/heads/main" in refuse["run"]

    recheck = PUBLISH["steps"][0]
    assert recheck["name"] == "Re-check the target tag"
    assert "${{" not in recheck["run"]
    assert r"^brain-main-[a-z0-9-]+$" in recheck["run"]
    assert "brain-main" in recheck["run"]

    default_tag = ON["workflow_dispatch"]["inputs"]["tag"]["default"]
    assert re.fullmatch(r"brain-main-[a-z0-9-]+", default_tag)


def test_signing_hygiene():
    tauri_cli = _step(SIGN, "Tauri CLI")
    assert "--ignore-scripts" in tauri_cli["run"]
    for name_prefix in ("Sign (throwaway key", "Sign (updater key)"):
        run = _step(SIGN, name_prefix)["run"]
        sign_lines = [ln for ln in run.splitlines() if "tauri signer sign " in ln]
        assert sign_lines, name_prefix
        for ln in sign_lines:
            assert " -k " not in ln and " -p " not in ln
    throwaway_run = _step(SIGN, "Sign (throwaway key")["run"]
    assert "TAURI_SIGNING_PRIVATE_KEY" in throwaway_run
    assert "TAURI_SIGNING_PRIVATE_KEY_PASSWORD" in throwaway_run
    updater_env = _step(SIGN, "Sign (updater key)")["env"]
    assert updater_env == {
        "TAURI_SIGNING_PRIVATE_KEY": "${{ secrets.TAURI_UPDATER_PRIVATE_KEY }}",
        "TAURI_SIGNING_PRIVATE_KEY_PASSWORD": "${{ secrets.TAURI_UPDATER_KEY_PASSWORD }}",
    }
    dump = yaml.safe_dump(PUBLISH)
    for tool in ("pnpm", "setup-node", "npm"):
        assert tool not in dump


def test_sign_uploads_and_publish_downloads_the_signed_build():
    upload = _step(SIGN, "Upload signed build")
    assert upload["if"] == "github.event_name != 'pull_request'"
    assert "dist/manifest.json.sig" in upload["with"]["path"]
    download = _step(PUBLISH, "Download signed build")
    assert download["with"]["name"] == "brain-main-signed"


def test_pr_trigger_covers_the_pipeline():
    paths = ON["pull_request"]["paths"]
    for p in (".github/workflows/brain-main.yml", "scripts/brain_main_manifest.py",
              "scripts/verify_update_signature.sh", "scripts/smoke_test_wheel.sh",
              "brain/update/install.py"):
        assert p in paths, p


def test_pr_trigger_is_not_branch_limited_and_covers_the_widened_paths():
    pr = ON["pull_request"]
    assert "branches" not in pr
    for p in ("app/package.json", "app/pnpm-lock.yaml", "uv.lock", "pyproject.toml",
              "brain/update/**"):
        assert p in pr["paths"], p
