"""scripts/ci_touches_rust.py + the `tauri (cargo check + test)` job's condition in test.yml — #758.

The macOS cargo job waits on GitHub's short macOS arm64 runner pool, so a PR runs it only when it changes a file the
Rust build or its tests read. Linux and Windows still compile and test the workspace on every PR, and every push to
main runs the macOS job.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("ci_touches_rust_under_test", REPO / "scripts" / "ci_touches_rust.py")
tr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tr
_spec.loader.exec_module(tr)


def test_a_pr_runs_macos_cargo_only_when_it_changes_what_the_rust_build_reads():
    assert tr.touches_rust(["app/src-tauri/src/lib.rs"]) is True
    assert tr.touches_rust(["brain/chat/prompt.py", "app/src-tauri/Cargo.lock"]) is True
    assert tr.touches_rust(["app/src-tauri/nell-shortcut/src/nell_cmd.tmpl"]) is True
    # files outside app/src-tauri that the Rust tests read
    assert tr.touches_rust(["app/src/brainRecovery.ts"]) is True
    assert tr.touches_rust(["tests/fixtures/bridge_watcher_files/stop.json"]) is True
    # the workflow that defines the job
    assert tr.touches_rust([".github/workflows/test.yml"]) is True
    # a brain-only or frontend-only PR waits on no macOS runner for cargo
    assert tr.touches_rust(["brain/chat/prompt.py", "tests/unit/brain/chat/test_prompt.py"]) is False
    assert tr.touches_rust(["app/src/ChatPanel.tsx", "app/src-tauri-notes.md"]) is False
    assert tr.touches_rust([]) is False


def test_every_file_the_rust_code_reads_from_outside_src_tauri_is_listed():
    # include_str!/include_bytes! paths resolve from the file they sit in; a CARGO_MANIFEST_DIR `.join(..)` chain from
    # app/src-tauri. A new read outside app/src-tauri that RUST_INPUTS misses would let a PR changing that file skip
    # the macOS cargo job.
    import os
    import re

    src_tauri = REPO / "app" / "src-tauri"
    include = re.compile(r'include_(?:str|bytes)!\(\s*"([^"]+)"\s*\)')
    manifest = re.compile(r'Path::new\(\s*env!\("CARGO_MANIFEST_DIR"\)\s*\)((?:\s*\.join\(\s*"[^"]*"\s*\))+)')
    reached = []
    for rs in src_tauri.rglob("*.rs"):
        if "target" in rs.relative_to(src_tauri).parts:
            continue
        text = rs.read_text(encoding="utf-8")
        reached += [Path(os.path.normpath(rs.parent / p)) for p in include.findall(text)]
        for chain in manifest.findall(text):
            reached.append(Path(os.path.normpath(src_tauri.joinpath(*re.findall(r'"([^"]*)"', chain)))))
    outside = sorted({p.relative_to(REPO).as_posix() for p in reached if not p.is_relative_to(src_tauri)})
    assert outside, "the scan found no read outside app/src-tauri; the two known ones should be there"
    unlisted = [p for p in outside if not (tr.touches_rust([p]) or tr.touches_rust([p + "/x"]))]
    assert unlisted == []


def test_macos_cargo_waits_for_a_runner_only_on_rust_prs_and_always_on_main():
    import yaml

    jobs = yaml.safe_load((REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))["jobs"]
    changes = jobs["rust-changes"]
    # PRs only (a push to main has no PR diff), on Ubuntu, reading the PR's own diff against its base
    assert changes["if"] == "github.event_name == 'pull_request'"
    assert changes["runs-on"] == "ubuntu-latest"
    (checkout,) = [s for s in changes["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
    assert checkout["with"]["fetch-depth"] == 2  # the PR merge commit and its first parent, the base tip
    (diff,) = [s for s in changes["steps"] if s.get("id") == "diff"]
    assert diff["shell"] == "bash"  # -eo pipefail: a failed git diff fails the step instead of reading as "no Rust"
    assert diff["run"].strip() == 'git diff --name-only HEAD^1 HEAD | python3 scripts/ci_touches_rust.py >> "$GITHUB_OUTPUT"'
    assert changes["outputs"] == {"rust": "${{ steps.diff.outputs.rust }}"}

    tauri = jobs["tauri"]
    assert tauri["runs-on"] == "macos-latest" and tauri["needs"] == "rust-changes"
    cond = " ".join(tauri["if"].split())
    # !cancelled(): runs although rust-changes was skipped (a push to main); and it fails open — any result but a
    # clean "no Rust" (a failed or skipped detector) runs cargo
    assert cond == ("!cancelled() && github.event_name != 'workflow_dispatch' && "
                    "(needs.rust-changes.result != 'success' || needs.rust-changes.outputs.rust == 'true')")
    # Linux and Windows still build and test the workspace on every PR
    assert "needs" not in jobs["tauri-linux-windows"] and "rust" not in jobs["tauri-linux-windows"].get("if", "")
