#!/usr/bin/env python3
"""Report how much a pull request grows the test suite, and flag the growth that costs more than it proves (#746).

For the test files a PR adds, modifies, renames or deletes, the guard collects the tests at the merge-base and at the
head (``pytest --collect-only``, with ``guard_test_growth_plugin`` recording each collected item) and prints a markdown
summary: counts per file, then these flags.

* scanner test: its parametrize values are built by scanning the repository (a glob, ``ls-files`` ...);
* cross-product grid: stacked parametrize axes whose ids outnumber the sum of the axis sizes;
* single-axis grid larger than the change: more new ids than changed product lines;
* likely duplicates: new tests whose normalised bodies nearly match a test in the same folder.

``@pytest.mark.large_grid(reason="...")`` on a test function justifies a flag. Without ``--enforce`` the exit status is
0; with it, an unjustified scanner or cross-product flag, or a marker problem, gives 1. The guard returns 1 in both
modes when it cannot judge (a collection that failed or disagrees with itself, a head file that does not parse), and
exits 2 on a usage or git error. The summary also goes to ``$GITHUB_STEP_SUMMARY`` when that is set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PLUGIN = "guard_test_growth_plugin"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec_module: dataclasses look the module up
    spec.loader.exec_module(module)
    return module


ntf = _load("new_tests_fail_on_base")
rules = _load("guard_test_growth_rules")


class CannotJudgeError(Exception):
    """The guard cannot give a trustworthy report; never a silent pass."""


def _collect_head(tree: Path, files: Sequence[str], plugin_dir: Path, rows_file: Path) -> tuple[list[str], list[dict]]:
    """Node ids and plugin rows of ``files`` collected in the head ``tree`` (no files, no collection)."""
    if not files:
        return [], []
    env_extra = {"NTFOB_PLUGIN_DIR": str(plugin_dir), "TGG_ROWS_FILE": str(rows_file)}
    args = ["--collect-only", "-q", "--continue-on-collection-errors", "-p", _PLUGIN, *files]
    res = ntf._pytest(sys.executable, tree, env_extra, args)
    if res.returncode not in (0, 5):
        raise CannotJudgeError(f"collecting in {tree.name} exited {res.returncode}:\n{(res.stdout + res.stderr)[-1500:]}")
    ids = ntf.parse_collected(res.stdout)
    expected = 0 if res.returncode == 5 else ntf.collected_count(res.stdout)
    if expected != len(ids):
        raise CannotJudgeError(f"collecting in {tree.name}: pytest reported {expected} tests, {len(ids)} ids were read from its output")
    try:
        rows = json.loads(rows_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise CannotJudgeError(f"the plugin rows of {tree.name} are missing or unreadable: {err}") from err
    if {row["nodeid"] for row in rows} != set(ids):
        raise CannotJudgeError(f"the plugin rows of {tree.name} do not match the collected ids")
    return ids, rows


def _read(tree: Path, path: str) -> str:
    return (tree / path).read_text(encoding="utf-8")


def _bodies(tree: Path, path: str) -> dict[str, list[str]]:
    try:
        found = rules.test_bodies(_read(tree, path))
    except SyntaxError as err:
        raise CannotJudgeError(f"{path} does not parse: {err}") from err
    return {f"{path}::{name}": tokens for name, tokens in found.items()}


def _scanners(tree: Path, files: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for path in files:
        try:
            found = rules.scanner_tests(_read(tree, path))
        except SyntaxError as err:
            raise CannotJudgeError(f"{path} does not parse: {err}") from err
        out |= {f"{path}::{name}" for name in found}
    return out


def _duplicates(tree: Path, files: Sequence[str], new_ids: set[str]) -> list[tuple[str, str, float]]:
    new_functions = {rules.function_of(i) for i in new_ids}
    new: dict[str, list[str]] = {}
    for path in files:
        new.update({k: v for k, v in _bodies(tree, path).items() if k in new_functions})
    pool: dict[str, list[str]] = {}
    for folder in sorted({str(Path(p).parent.as_posix()) for p in files}):
        # The test files directly in the folder, named as the new-tests check names them: test_*.py or *_test.py.
        siblings = {*(tree / folder).glob("test_*.py"), *(tree / folder).glob("*_test.py")}
        for sibling in sorted(siblings):
            pool.update(_bodies(tree, sibling.relative_to(tree).as_posix()))
    return rules.likely_duplicates(new, pool)


def _product_diff(repo: Path, base: str, head: str) -> str:
    """The ``-U0`` diff the budget is counted from, as UTF-8 text, free of the user's diff drivers and colour."""
    res = subprocess.run(
        [
            "git", "-c", "core.quotepath=false", "diff", "-U0", "--no-color", "--no-ext-diff", "--no-textconv",
            "--src-prefix=a/", "--dst-prefix=b/", base, head,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if res.returncode != 0:
        print(f"git diff {base} {head} failed ({res.returncode}): {res.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return res.stdout


def _judge(repo: Path, base: str, head: str, files: list[str], renamed: dict[str, str], deleted: list[str], scratch: Path) -> rules.Report:
    plugin_dir = scratch / "plugin"
    plugin_dir.mkdir()
    shutil.copyfile(_HERE / f"{_PLUGIN}.py", plugin_dir / f"{_PLUGIN}.py")
    head_tree, base_tree = scratch / "head", scratch / "base"
    ntf._git(repo, "worktree", "add", "--detach", str(head_tree), head)
    ntf._git(repo, "worktree", "add", "--detach", str(base_tree), base)

    head_ids, head_rows = _collect_head(head_tree, files, plugin_dir, scratch / "head_rows.json")
    # -z: a plain listing quotes a non-ASCII path ("tests/test_caf\303\251.py"), which then matches no changed file.
    at_base = set(filter(None, ntf._git(repo, "ls-tree", "-r", "-z", "--name-only", base).split("\0")))
    base_files = sorted(({renamed.get(f, f) for f in files} | set(deleted)) & at_base)
    # The base rows are never used: the new-tests check's own collection, with its integrity checks, is enough.
    try:
        base_ids = ntf._collect(sys.executable, base_tree, base_files, "", str(plugin_dir), tolerate_import_errors=True) if base_files else []
    except ntf.CheckError as err:
        raise CannotJudgeError(str(err)) from err
    base_set = ntf.remap_base_ids(base_ids, renamed)
    new_ids = set(head_ids) - base_set

    counts = rules.count_files([*sorted(files), *deleted], base_set, set(head_ids))
    return rules.build_report(
        counts=counts,
        head_rows=head_rows,
        new_ids=new_ids,
        scanners=_scanners(head_tree, files),
        budget=rules.product_lines(_product_diff(repo, base, head)),
        duplicates=_duplicates(head_tree, files, new_ids),
    )


def run(repo: Path, base: str, head: str, *, enforce: bool) -> int:
    base = ntf._git(repo, "rev-parse", base).strip()
    head = ntf._git(repo, "rev-parse", head).strip()
    base = ntf._git(repo, "merge-base", base, head).strip()
    entries = ntf.parse_name_status(ntf._git(repo, "diff", "--name-status", "-M30%", "-z", base, head))
    files, renamed = ntf.changed_test_files(entries)
    deleted = sorted({old for status, old, _ in entries if status == "D" and ntf.is_test_file(old)})
    print(f"base {base[:10]}  head {head[:10]}  changed test files: {len(files) + len(deleted)}")
    if not files and not deleted:
        _emit("no test changes\n")
        return 0

    scratch = Path(tempfile.mkdtemp(prefix="tgg-")).resolve()
    try:
        report = _judge(repo, base, head, files, renamed, deleted, scratch)
    except CannotJudgeError as err:
        _emit(f"the guard can't judge this PR: {str(err).rstrip()}\n")
        return 1
    finally:
        for tree in (scratch / "head", scratch / "base"):
            if tree.exists():
                ntf._git(repo, "worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(scratch, ignore_errors=True)

    _emit(rules.render_summary(report))
    return rules.exit_status(report, enforce=enforce)


def _emit(text: str) -> None:
    """Print ``text`` and append it to ``$GITHUB_STEP_SUMMARY`` when that is set."""
    print(text, end="" if text.endswith("\n") else "\n")
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the branch or commit the PR started from; the guard uses its merge-base with --head")
    ap.add_argument("--head", default="HEAD", help="the PR head commit (default HEAD)")
    ap.add_argument("--repo", default=".", help="repository root (default .)")
    ap.add_argument("--enforce", action="store_true", help="exit 1 on an unjustified scanner or cross-product flag or a marker problem")
    ns = ap.parse_args(argv)
    return run(Path(ns.repo).resolve(), ns.base, ns.head, enforce=ns.enforce)


if __name__ == "__main__":
    sys.exit(main())
