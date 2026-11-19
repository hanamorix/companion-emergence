"""Leak-guard engine tests (public-sync-reroot P1).

The guard is the one lasting piece of the re-root design
(docs/guarded-change/public-sync-reroot/2-plan.md step 7). These tests pin the
engine's arms in isolation; hooks/test_leak_guard.sh covers the end-to-end
git-push path (C7/C8/C9).

Stdlib-only module under test: hooks/leak_guard.py (runs outside the venv as a
git hook).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "hooks" / "leak_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("leak_guard", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lg():
    return _load()


def test_normalize_nfc_casefold_punct_whitespace(lg):
    # curly quotes / case / punctuation / repeated whitespace all collapse
    assert lg.normalize("Plush,  the “Kraken”  TEAPOT!") == "plush the kraken teapot"


def test_normalize_hardened_backslash_paths_and_zero_width(lg):
    # red-team m1: Windows-style separators fold to /
    assert lg.normalize(r"C:\Users\janedoe\proj") == "c /users/janedoe/proj"
    # red-team m2: zero-width chars (Cf category) are stripped, not tokenized
    assert lg.normalize("can​ary") == "canary"


SALT = "0011deadbeef"


def test_word_ngram_fingerprint_matches_embedded_phrase(lg):
    fp = lg.build_fingerprint("plush kraken teapot", salt_hex=SALT, kind="words")
    line = "+ she mentioned plush kraken teapot at breakfast"
    assert lg.scan_line_fingerprints(line, [fp], salt_hex=SALT)


def test_path_fingerprint_matches_prefix_of_longer_token(lg):
    # N4: the path marker must match as a PREFIX inside a longer token
    fp = lg.build_fingerprint("/Users/janedoe", salt_hex=SALT, kind="path")
    line = "+cd /Users/janedoe/companion-emergence && make"
    assert lg.scan_line_fingerprints(line, [fp], salt_hex=SALT)


def test_scan_range_all_arms_through_scratch_repo(lg, tmp_path):
    """Through-path: a scratch repo with one dirty commit → all arms report."""
    import subprocess

    repo = tmp_path / "scratch"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Hana")
    git("config", "user.email", "user@leakyhost.local")  # leaky identity
    (repo / "base.txt").write_text("clean\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    (repo / "docs" / "superpowers").mkdir(parents=True)
    (repo / "docs" / "superpowers" / "x.md").write_text("private tree\n")
    (repo / "notes.md").write_text("the canary wombat teacup appears\n")
    (repo / "pic.png").write_bytes(b"\x89PNG\r\n")
    git("add", ".")
    git("commit", "-qm", "adds canary wombat teacup in message")

    rules = tmp_path / "rules.local"
    rules.write_text("substr:canary wombat teacup\n")
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("Hana Mori <214302556+hanamorix@users.noreply.github.com>\n")
    denied = tmp_path / "deny.txt"
    denied.write_text("docs/superpowers/\n")
    binaries = tmp_path / "bin.txt"
    binaries.write_text("expressions/\n")

    report = lg.scan_range(
        f"{base}..HEAD",
        repo=str(repo),
        plaintext_rules_file=rules,
        allowlist_file=allowlist,
        denied_paths_file=denied,
        allowed_binary_file=binaries,
    )
    assert report["clean"] is False
    arms = {f["arm"] for f in report["findings"]}
    assert arms == {"content", "message", "metadata", "path", "binary"}
    # never echo matched content back (the finding may BE the secret)
    assert "wombat" not in str(report["findings"])


def test_content_arm_scans_plus_plus_prefixed_added_lines(lg, tmp_path):
    """Red-team B1: an added line whose CONTENT begins with ++ (markdown,
    committed .patch files) must still be scanned — only real `+++ b/` headers
    are structural."""
    rules_file = tmp_path / "rules.local"
    rules_file.write_text("substr:canary wombat teacup\n")
    rules = lg.load_plaintext_rules(rules_file)
    diff = "+++ b/x.txt\n+++ canary wombat teacup smuggled\n"
    findings = lg.scan_diff_text(diff, plain_rules=rules, salt=None, entries=[])
    assert findings, "++-prefixed added line was skipped by the content arm"


def test_content_arm_not_fooled_by_removed_dashes_before_plus_plus(lg, tmp_path):
    """Re-review NEW-1: a REMOVED content line beginning `--` renders as
    `--- …` in the diff; the following added `++ <marker>` line must still be
    scanned — only real old-file headers (`--- a/`, `--- /dev/null`) shield a
    `+++` line."""
    rules_file = tmp_path / "rules.local"
    rules_file.write_text("substr:canary wombat teacup\n")
    rules = lg.load_plaintext_rules(rules_file)
    diff = (
        "diff --git a/x.md b/x.md\n"
        "--- a/x.md\n"
        "+++ b/x.md\n"
        "@@ -1,2 +1,2 @@\n"
        "--- old subheading\n"
        "+++ canary wombat teacup\n"
    )
    findings = lg.scan_diff_text(diff, plain_rules=rules, salt=None, entries=[])
    assert findings, "removed --- content line masked the following ++ added line"


def test_content_arm_catches_committed_then_reverted_marker(lg, tmp_path):
    """Red-team B2: a marker committed then reverted before push has an EMPTY
    endpoint diff but its blob still lands in public history — the content arm
    must scan per-commit, not endpoints."""
    import subprocess

    repo = tmp_path / "r"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "X")
    git("config", "user.email", "x@example.com")
    (repo / "a.txt").write_text("clean\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    (repo / "oops.txt").write_text("canary wombat teacup\n")
    git("add", ".")
    git("commit", "-qm", "adds")
    git("rm", "-q", "oops.txt")
    git("commit", "-qm", "reverts")

    rules = tmp_path / "rules.local"
    rules.write_text("substr:canary wombat teacup\n")
    report = lg.scan_range(f"{base}..HEAD", repo=str(repo), plaintext_rules_file=rules)
    assert report["clean"] is False
    assert any(f["arm"] == "content" for f in report["findings"])


def test_cli_fingerprint_and_check(lg, tmp_path):
    """CLI: `fingerprint` emits salt+entries JSON; `check --report` scans a range."""
    import json
    import subprocess
    import sys

    markers = tmp_path / "markers.txt"
    markers.write_text("# comment\ncanary wombat teacup\n/Users/janedoe\n")
    out = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "fingerprint", str(markers), "--salt-hex", SALT],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    assert data["salt"] == SALT
    kinds = {e["kind"] for e in data["entries"]}
    assert kinds == {"words", "path"}
    # no plaintext markers in the output (C11)
    assert "wombat" not in out and "janedoe" not in out

    # scratch repo: clean commit → check exits 0 with clean report
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "X")
    git("config", "user.email", "x@example.com")
    (repo / "a.txt").write_text("clean\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (repo / "b.txt").write_text("still clean\n")
    git("add", ".")
    git("commit", "-qm", "second")
    fp = tmp_path / "fp.json"
    fp.write_text(out)
    result = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "check", "--range", f"{base}..HEAD",
         "--repo", str(repo), "--fingerprints", str(fp), "--report"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["clean"] is True


def test_path_and_binary_arms(lg):
    deny = ["docs/superpowers/", "docs/audits/", ".public-sync/"]
    assert lg.check_path("docs/superpowers/plans/x.md", deny) is not None
    assert lg.check_path("Docs/Superpowers/plans/x.md", deny) is not None  # case dodge
    assert lg.check_path("docs/roadmap.md", deny) is None
    allowed = ["expressions/", "app/src-tauri/icons/"]
    assert lg.check_binary("wizard-validation/photo.png", allowed) is not None
    assert lg.check_binary("expressions/joy.png", allowed) is None
    assert lg.check_binary("brain/cli.py", allowed) is None  # not an image


ALLOW = [
    "Hana Mori <214302556+hanamorix@users.noreply.github.com>",
    "committer-only:GitHub <noreply@github.com>",
    "ThinkerOfThoughts <thinkerofthoughts42@gmail.com>",
    "glob:*@users.noreply.github.com",
]


def test_metadata_arm_allowlist_and_committer_only(lg):
    # noreply author+committer: allowed
    ok, why = lg.check_identity(
        author="Hana Mori <214302556+hanamorix@users.noreply.github.com>",
        committer="Hana Mori <214302556+hanamorix@users.noreply.github.com>",
        allowlist=ALLOW,
    )
    assert ok, why
    # leaky author: rejected, named in reason
    ok, why = lg.check_identity(
        author="Hana <user@leakyhost.local>",
        committer="Hana Mori <214302556+hanamorix@users.noreply.github.com>",
        allowlist=ALLOW,
    )
    assert not ok and "leakyhost" in why
    # N5: GitHub web-merge committer allowed as committer…
    ok, _ = lg.check_identity(
        author="ThinkerOfThoughts <thinkerofthoughts42@gmail.com>",
        committer="GitHub <noreply@github.com>",
        allowlist=ALLOW,
    )
    assert ok
    # …but never as author
    ok, _ = lg.check_identity(
        author="GitHub <noreply@github.com>",
        committer="GitHub <noreply@github.com>",
        allowlist=ALLOW,
    )
    assert not ok


def test_plaintext_rules_substring_and_regex(lg, tmp_path):
    rules_file = tmp_path / "leak-rules.local"
    rules_file.write_text("# comment\nsubstr:Plush the kraken teapot\nre:Blorpuary 2[01]\n")
    rules = lg.load_plaintext_rules(rules_file)
    assert lg.scan_line_plaintext("+ note: plush, the kraken teapot photo", rules)
    assert lg.scan_line_plaintext("+ dated Blorpuary 21 in the letter", rules)
    assert not lg.scan_line_plaintext("+ dated Blorpuary 25 in the letter", rules)


# ── #761: the pre-push hook reads the main checkout's local rules from a worktree ──
_HOOK_FILES = ("pre-push", "leak_guard.py", "denied-paths.txt", "allowed-binary-paths.txt")
_CANARY = "canary wombat teacup"  # synthetic marker, as in hooks/test_leak_guard.sh
_ZERO = "0" * 40


def _shell():
    import shutil

    sh = shutil.which("sh") or shutil.which("bash")
    if sh is None:
        pytest.skip("no POSIX shell to run the hook with")
    return sh


def _scratch_with_worktree(tmp_path, *, rules: str | None):
    """A scratch main checkout carrying the real hook files, with ``hooks/leak-rules.local``
    gitignored (and holding ``rules`` when given), plus a worktree on branch ``feat`` whose one
    new commit adds the canary. Returns ``(main, worktree, feat_sha)``."""
    import shutil
    import subprocess

    main = tmp_path / "main"
    (main / "hooks").mkdir(parents=True)

    def git(*args, cwd=main):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    hooks_src = _MODULE_PATH.parent
    for name in _HOOK_FILES:
        shutil.copy2(hooks_src / name, main / "hooks" / name)
    # A scratch identity allowlist, so the metadata arm passes without naming a real person.
    (main / "hooks" / "allowed-identities.txt").write_text("glob:*@example.com\n", encoding="utf-8")
    (main / ".gitignore").write_text("hooks/leak-rules.local\n", encoding="utf-8")
    git("init", "-q", "-b", "main")
    git("config", "user.name", "X")
    git("config", "user.email", "x@example.com")
    git("add", ".")
    git("commit", "-qm", "base")
    if rules is not None:
        (main / "hooks" / "leak-rules.local").write_text(rules, encoding="utf-8")
    wt = tmp_path / "wt"
    git("worktree", "add", "-q", str(wt), "-b", "feat")
    (wt / "note.txt").write_text(f"a line with the {_CANARY} in it\n", encoding="utf-8")
    git("add", ".", cwd=wt)
    git("commit", "-qm", "add a note", cwd=wt)
    return main, wt, git("rev-parse", "HEAD", cwd=wt)


def _push_hook(cwd, sha):
    """Run the real pre-push hook from ``cwd`` as git would for a new branch ``feat`` at ``sha``."""
    import os
    import subprocess

    env = {**os.environ, "LEAK_GUARD_FORCE": "1"}
    # Bytes, not text: git writes "\n", and text mode on Windows would send "\r\n", which the hook's `read` keeps as
    # part of the remote sha (so "0000...\r" is not the new-branch zero and rev-list fails).
    done = subprocess.run(
        [_shell(), "hooks/pre-push", "origin", "https://example.invalid/scratch.git"],
        cwd=cwd, input=f"refs/heads/feat {sha} refs/heads/feat {_ZERO}\n".encode("ascii"),
        capture_output=True, env=env,
    )
    return subprocess.CompletedProcess(
        done.args, done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")
    )


def test_a_push_from_a_worktree_uses_the_main_checkouts_local_rules(tmp_path):
    """#761: the rules file is gitignored, so it lives only in the main checkout. A push from a
    worktree must still be checked against it, and refused when an added line matches."""
    _main, wt, sha = _scratch_with_worktree(tmp_path, rules=f"substr:{_CANARY}\n")
    assert not (wt / "hooks" / "leak-rules.local").exists()

    result = _push_hook(wt, sha)

    assert result.returncode == 1, result.stderr
    assert "LEAK-GUARD [content]" in result.stderr and "Traceback" not in result.stderr, result.stderr


@pytest.mark.no_base_failure(reason="pins that a push from the main checkout still uses its own rules (#761); main does too")
def test_a_push_from_the_main_checkout_still_uses_its_rules(tmp_path):
    import subprocess

    main, wt, sha = _scratch_with_worktree(tmp_path, rules=f"substr:{_CANARY}\n")
    subprocess.run(["git", "merge", "-q", "--ff-only", "feat"], cwd=main, check=True, capture_output=True)

    result = _push_hook(main, sha)

    assert result.returncode == 1, result.stderr
    assert "LEAK-GUARD [content]" in result.stderr and "Traceback" not in result.stderr, result.stderr


def test_no_rules_file_anywhere_warns_and_still_passes_a_clean_push(tmp_path):
    """#761: with no local rules at all the hook says so on stderr, and a push with nothing
    else wrong still goes through (CI's marker-scan is the real boundary)."""
    _main, wt, sha = _scratch_with_worktree(tmp_path, rules=None)

    result = _push_hook(wt, sha)

    assert result.returncode == 0, result.stderr
    assert "leak-rules.local" in result.stderr and "not checked" in result.stderr
    assert "Traceback" not in result.stderr
