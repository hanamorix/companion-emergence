#!/usr/bin/env python3
"""Every NEW test of a pull request must fail against the code the PR started from (#540).

A test that cannot fail proves nothing. This check runs the tests a PR adds against the product
code of the commit the PR branched from, and reports every new test that does not fail there.

Definitions (also in CONTRIBUTING.md):

* A *changed test file* is a file under ``tests/`` named ``test_*.py`` or ``*_test.py`` that the PR
  adds, modifies or renames.
* A *new test* is a test id (``path::Class::name[param]``) collected at the PR head in a changed test
  file that the base commit does not collect. For a renamed file the base ids are compared under the
  new path, so a pure move creates no new test. A test whose body changed but whose id did not, is
  not new. A renamed test function, a new parameter value, and a test moved to another file by hand
  are new.
* *Product code* is every top-level path of the repository except ``tests/``, ``.github/``,
  ``pyproject.toml`` and ``uv.lock``. The new tests run with the PR's own tests, fixtures and
  pytest configuration, against the base commit's product code.
* A new test *bites* when, against the base product code, it fails, errors (in setup too), or cannot
  be collected (its module does not import, or its id does not exist, because the code it tests is
  new). A test that passes or is skipped there does not bite.
* A test marked ``@pytest.mark.no_base_failure(reason="...")`` is exempt (refactors and test
  infrastructure that pass on the old code by design). Exempt tests are counted and listed.

Exit status 0: every new test bites or is exempt. 1: at least one does not (listed), or the check could not
judge (a pytest call failed to collect or crashed, or the canary failed: the old code cannot run the new test
setup). 2: usage or git error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Top-level paths that stay at the PR head when the product code is swapped to the base.
HEAD_ONLY = frozenset({"tests", ".github", "pyproject.toml", "uv.lock"})

# Written to a temp dir and loaded with ``-p``: records every test report so the judge sees the phase
# (setup / call / teardown) and outcome per node id. Only the controller process writes (xdist workers
# forward their reports to it).
_PLUGIN_NAME = "ntfob_outcome_plugin"
_PLUGIN_SOURCE = '''\
import json
import os

_PATH = os.environ["NTFOB_OUTCOME_FILE"]
_OPEN = open  # a test that replaces builtins.open must not break the recorder
_ACTIVE = [True]


def pytest_configure(config):
    _ACTIVE[0] = not hasattr(config, "workerinput")


def pytest_runtest_logreport(report):
    if not _ACTIVE[0]:
        return
    rec = {"id": report.nodeid, "when": report.when, "outcome": report.outcome}
    with _OPEN(_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\\n")
'''


# ----------------------------------------------------------------------------------------------
# Pure helpers (unit-tested; no git, no subprocess)
# ----------------------------------------------------------------------------------------------


def is_test_file(path: str) -> bool:
    """A path under ``tests/`` named ``test_*.py`` or ``*_test.py`` (forward slashes)."""
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    return p.startswith("tests/") and name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def parse_name_status(raw: str) -> list[tuple[str, str, str]]:
    """Parse ``git diff --name-status -M -z`` output into ``(status, old_path, new_path)``.

    Status is the first letter (A, M, D, R, C, T). For A, M, D, T the two paths are equal. A copy (C)
    is treated like an add of the new path.
    """
    fields = raw.split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    out: list[tuple[str, str, str]] = []
    i = 0
    while i < len(fields):
        status = fields[i][:1]
        if status in ("R", "C"):
            old, new = fields[i + 1], fields[i + 2]
            out.append((status, old, new))
            i += 3
        else:
            path = fields[i + 1]
            out.append((status, path, path))
            i += 2
    return out


def changed_test_files(entries: Iterable[tuple[str, str, str]]) -> tuple[list[str], dict[str, str]]:
    """The head paths of the changed test files, and a ``new_path -> old_path`` map for renames.

    Deleted files are dropped; a copy has no base counterpart (its ids are all new).
    """
    head_files: list[str] = []
    renamed: dict[str, str] = {}
    for status, old, new in entries:
        if status == "D" or not is_test_file(new):
            continue
        head_files.append(new)
        if status == "R" and old != new:
            renamed[new] = old
    return sorted(set(head_files)), renamed


def remap_base_ids(base_ids: Iterable[str], renamed: Mapping[str, str]) -> set[str]:
    """Rewrite base ids of renamed files from the old path to the new path."""
    old_to_new = {old: new for new, old in renamed.items()}
    out: set[str] = set()
    for node in base_ids:
        path, sep, rest = node.partition("::")
        out.add(old_to_new.get(path, path) + sep + rest)
    return out


def parse_collected(stdout: str) -> list[str]:
    """Node ids from ``pytest --collect-only -q`` output: lines that start with a ``tests/`` path."""
    ids = []
    for line in stdout.splitlines():
        line = line.rstrip()
        if line.startswith("tests/") and "::" in line:
            ids.append(line)
    return ids


def collected_count(stdout: str) -> int | None:
    """The ``N tests collected`` figure pytest prints last under ``-q --collect-only`` (``N/M`` when some are deselected)."""
    found = re.findall(r"^(\d+)(?:/\d+)? tests? collected", stdout, re.M)
    return int(found[-1]) if found else None


class CheckError(Exception):
    """The check could not judge (never a silent pass): a pytest call crashed or collected inconsistently."""


@dataclass
class Verdict:
    bites: dict[str, str] = field(default_factory=dict)  # id -> how it bit
    violations: dict[str, str] = field(default_factory=dict)  # id -> why it did not bite


def judge(
    to_check: Iterable[str],
    collectable_at_base: set[str],
    outcomes: Mapping[str, Sequence[tuple[str, str]]],
) -> Verdict:
    """Classify each new, non-exempt test id.

    ``collectable_at_base``: ids that collect in the swapped tree (base product code, head tests).
    ``outcomes``: id -> ``[(when, outcome), ...]`` as recorded while running them.
    An id that does not collect there bites (the code it tests is new). An id that collects but has no
    recorded report did not run, which is a violation (never silently green).
    """
    v = Verdict()
    for node in sorted(set(to_check)):
        if node not in collectable_at_base:
            v.bites[node] = "cannot be collected against the base code"
            continue
        reports = list(outcomes.get(node, ()))
        if not reports:
            v.violations[node] = "collected but did not run (no report recorded)"
            continue
        failed = [(when, out) for when, out in reports if out == "failed"]
        if failed:
            v.bites[node] = f"failed in {failed[0][0]}"
            continue
        call = [out for when, out in reports if when == "call"]
        if call and call[0] == "passed":
            v.violations[node] = "passes against the base code"
        else:
            v.violations[node] = "skipped or xfailed against the base code (that proves nothing)"
    return v


def smallest_unchanged_test(ls_tree_l: str, changed: set[str], min_size: int = 300) -> str | None:
    """Canary choice (listed at the head, so it exists in both trees): the smallest unchanged ``tests/unit`` test file of at least ``min_size`` bytes (300 by default).

    ``ls_tree_l`` is ``git ls-tree -r -l <head> tests/unit`` output (``mode type sha size<TAB>path``).
    """
    best: tuple[int, str] | None = None
    for line in ls_tree_l.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not parts[3].isdigit():
            continue
        size = int(parts[3])
        if size < min_size or not is_test_file(path) or path in changed:
            continue
        if best is None or size < best[0]:
            best = (size, path)
    return best[1] if best else None


# ----------------------------------------------------------------------------------------------
# Side-effecting part
# ----------------------------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if check and res.returncode != 0:
        print(f"git {' '.join(args)} failed ({res.returncode}): {res.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return res.stdout


def _pytest(py: str, cwd: Path, env_extra: Mapping[str, str], args: Sequence[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_extra}
    env["PYTEST_ADDOPTS"] = ""  # the output format is parsed below: no inherited -q/-qq/-p/-n
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(cwd), env_extra.get("NTFOB_PLUGIN_DIR", ""), env.get("PYTHONPATH", "")]))
    return subprocess.run(
        [py, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:tdd-guard", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _collect(py: str, cwd: Path, files: Sequence[str], expr: str, plugin_dir: str, *, tolerate_import_errors: bool) -> list[str]:
    """Node ids collected from ``files`` in ``cwd``.

    pytest's exit status 0 (collected) and 5 (nothing selected) are always fine. With ``tolerate_import_errors``
    (the base-code trees, where a module of new code is expected not to import) 1 and 2 are fine too. Anything else
    (3 internal error, 4 usage error such as a refused mark, or an error in the head tree) raises, and the number
    of ids parsed must equal the count pytest printed, so a changed output format cannot read as "no tests".
    """
    args = ["--collect-only", "-q", "--continue-on-collection-errors"]
    if expr:
        args += ["-m", expr]
    res = _pytest(py, cwd, {"NTFOB_PLUGIN_DIR": plugin_dir}, [*args, *files])
    allowed = (0, 5, 1, 2) if tolerate_import_errors else (0, 5)
    if res.returncode not in allowed:
        raise CheckError(f"collecting in {cwd.name} exited {res.returncode}:\n{(res.stdout + res.stderr)[-1500:]}")
    if res.returncode in (1, 2) and not re.search(r"^ERROR |\d+ errors?\b", res.stdout, re.M):
        raise CheckError(f"collecting in {cwd.name} exited {res.returncode} without a collection error:\n{(res.stdout + res.stderr)[-1500:]}")
    ids = parse_collected(res.stdout)
    printed = collected_count(res.stdout)
    if res.returncode == 0 and printed != len(ids):
        raise CheckError(f"collecting in {cwd.name}: pytest reported {printed} tests, {len(ids)} ids were read from its output")
    return ids


def _swap_product_code(head_tree: Path, base: str) -> list[str]:
    """Replace every product path of ``head_tree`` by the base commit's. Returns the swapped names."""
    base_names = set(filter(None, _git(head_tree, "ls-tree", "--name-only", base).splitlines()))
    head_names = {p.name for p in head_tree.iterdir()} - {".git"}
    swapped = []
    for name in sorted((base_names | head_names) - HEAD_ONLY):
        target = head_tree / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
        if name in base_names:
            _git(head_tree, "checkout", base, "--", name)
        swapped.append(name)
    return swapped


def canonical_id(recorded: str, known: set[str]) -> str:
    """The collected id a recorded report id stands for.

    With ``--dist loadgroup`` pytest-xdist renames every test that carries ``@pytest.mark.xdist_group`` to
    ``<id>@<group names joined by "_">`` (``xdist/remote.py``, ``pytest_collection_modifyitems``), and every report of
    the run then carries that suffixed id. The suffix is cut at the last ``@`` that leaves a known collected id. An id
    that matches no known id is returned as recorded.
    """
    node = recorded
    while node not in known:
        stem, sep, _ = node.rpartition("@")
        if not sep:
            return recorded
        node = stem
    return node


def _read_outcomes(path: Path, known_ids: Iterable[str] = ()) -> dict[str, list[tuple[str, str]]]:
    """Reports per test id; ids renamed by an xdist group are read back under their collected id (``known_ids``)."""
    known = set(known_ids)
    out: dict[str, list[tuple[str, str]]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            out.setdefault(canonical_id(rec["id"], known), []).append((rec["when"], rec["outcome"]))
    return out


def run(repo: Path, base: str, head: str, workers: str, expr: str, only: Sequence[str] = ()) -> int:
    py = sys.executable
    base = _git(repo, "rev-parse", base).strip()
    head = _git(repo, "rev-parse", head).strip()
    base = _git(repo, "merge-base", base, head).strip()  # the fork point, so commits main gained since are not "inverted" into the diff
    entries = parse_name_status(_git(repo, "diff", "--name-status", "-M30%", "-z", base, head))
    files, renamed = changed_test_files(entries)
    if only:
        files = [f for f in files if any(part in f for part in only)]
    print(f"base {base[:10]}  head {head[:10]}  changed test files: {len(files)}")
    if not files:
        print("no changed test files: nothing to check")
        return 0

    scratch = Path(tempfile.mkdtemp(prefix="ntfob-"))
    head_tree, base_tree = scratch / "head", scratch / "base"
    plugin_dir = scratch / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / f"{_PLUGIN_NAME}.py").write_text(_PLUGIN_SOURCE, encoding="utf-8")
    try:
        return _judge_pr(py, repo, base, head, workers, expr, files, renamed, entries, scratch, head_tree, base_tree, plugin_dir, mark_args=["-m", expr] if expr else [])
    except CheckError as err:
        print(f"CHECK COULD NOT JUDGE THIS PR: {err}")
        return 1
    finally:
        for tree in (head_tree, base_tree):
            if tree.exists():
                _git(repo, "worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(scratch, ignore_errors=True)


def _judge_pr(py, repo, base, head, workers, expr, files, renamed, entries, scratch, head_tree, base_tree, plugin_dir, *, mark_args):
    _git(repo, "worktree", "add", "--detach", str(head_tree), head)
    _git(repo, "worktree", "add", "--detach", str(base_tree), base)

    head_ids = set(_collect(py, head_tree, files, expr, str(plugin_dir), tolerate_import_errors=False))
    exempt_ids = set(_collect(py, head_tree, files, f"({expr}) and no_base_failure" if expr else "no_base_failure", str(plugin_dir), tolerate_import_errors=False))
    base_files = sorted({renamed.get(f, f) for f in files} & set(filter(None, _git(repo, "ls-tree", "-r", "--name-only", base).splitlines())))
    base_ids = remap_base_ids(_collect(py, base_tree, base_files, expr, str(plugin_dir), tolerate_import_errors=True), renamed) if base_files else set()
    new_ids = head_ids - base_ids
    to_check = new_ids - exempt_ids
    exempt_new = new_ids & exempt_ids
    print(f"head ids {len(head_ids)}  base ids {len(base_ids)}  new {len(new_ids)}  exempt (no_base_failure) {len(exempt_new)}  to check {len(to_check)}")
    for node in sorted(exempt_new):
        print(f"  exempt: {node}")
    if not to_check:
        print("no new non-exempt tests: nothing to check")
        return 0

    swapped = _swap_product_code(head_tree, base)
    print(f"product code swapped to base: {', '.join(swapped)}")
    still_differs = subprocess.run(["git", "diff", "--quiet", base, "--", *swapped], cwd=head_tree, check=False).returncode
    if still_differs:
        raise CheckError("after the swap the product paths still differ from the base commit")
    probe = subprocess.run(
        [py, "-c", "import brain, sys; sys.stdout.write(brain.__file__)"],
        cwd=head_tree, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(head_tree)},
    )
    if not probe.stdout.startswith(str(head_tree)):
        print(f"CANARY FAILED: brain imports from {probe.stdout!r}, not from the swapped tree {head_tree}")
        return 1

    env_extra = {"NTFOB_PLUGIN_DIR": str(plugin_dir), "NTFOB_OUTCOME_FILE": str(scratch / "outcomes.jsonl")}
    par = ["-n", workers, "--dist", "loadgroup"] if workers not in ("", "0", "1") else []

    # Canary: a test that exists at base and is not in this PR must still pass with the head
    # conftest over the base product code; otherwise every "failure" below could be the setup.
    canary = smallest_unchanged_test(_git(repo, "ls-tree", "-r", "-l", head, "tests/unit"), {new for _, _, new in entries})
    if canary is None:
        print("CANARY FAILED: no unchanged unit test file to use as a canary")
        return 1
    canary_out = scratch / "canary.jsonl"
    canary_run = _pytest(py, head_tree, {**env_extra, "NTFOB_OUTCOME_FILE": str(canary_out)}, ["-p", _PLUGIN_NAME, "-q", *mark_args, canary])
    canary_res = _read_outcomes(canary_out)
    bad = [n for n, reps in canary_res.items() if any(o == "failed" for _, o in reps)]
    called = [o for reps in canary_res.values() for when, o in reps if when == "call"]
    if canary_run.returncode != 0 or "passed" not in called or bad:
        print(f"CANARY FAILED: {canary} does not pass against the base product code with the head tests' setup "
              f"(exit {canary_run.returncode}, failed: {bad[:5]}). The base code cannot run the head conftest, so the check "
              f"cannot judge this PR.\n{(canary_run.stdout + canary_run.stderr)[-1200:]}")
        return 1

    collectable = set(_collect(py, head_tree, files, expr, str(plugin_dir), tolerate_import_errors=True))
    runnable = sorted(to_check & collectable)
    outcomes: dict[str, list[tuple[str, str]]] = {}
    if runnable:
        argsfile = scratch / "ids.txt"
        argsfile.write_text("\n".join(runnable) + "\n", encoding="utf-8")
        res = _pytest(py, head_tree, env_extra, ["-p", _PLUGIN_NAME, "-q", "--tb=no", *par, *mark_args, f"@{argsfile}"])
        tail = "\n".join(res.stdout.splitlines()[-3:])
        print(f"base run: {tail}")
        if res.returncode not in (0, 1):
            raise CheckError(f"the base run exited {res.returncode}:\n{(res.stdout + res.stderr)[-1500:]}")
        outcomes = _read_outcomes(Path(env_extra["NTFOB_OUTCOME_FILE"]), runnable)
        if not outcomes:
            raise CheckError("the base run recorded no test report at all")

    verdict = judge(to_check, collectable, outcomes)
    print(f"checked {len(to_check)}: {len(verdict.bites)} bite, {len(verdict.violations)} do not, {len(exempt_new)} exempt")
    _summary(verdict, sorted(exempt_new))
    if verdict.violations:
        print("\nThese new tests do NOT fail against the base code (a test that cannot fail proves nothing):")
        for node, why in sorted(verdict.violations.items()):
            print(f"  {node}: {why}")
        print("\nFix each test so it fails on the old code, or, for a refactor or test-infrastructure test that "
              "passes there by design, mark it @pytest.mark.no_base_failure(reason=\"...\").")
        return 1
    return 0


def _summary(verdict: Verdict, exempt: Sequence[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = [
        "### New tests must fail against the base code",
        f"- bite: {len(verdict.bites)}",
        f"- do not bite: {len(verdict.violations)}",
        f"- exempt (`no_base_failure`): {len(exempt)}",
    ]
    lines += [f"  - `{n}`: {why}" for n, why in sorted(verdict.violations.items())]
    lines += [f"  - exempt: `{n}`" for n in exempt]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the branch or commit the PR started from (e.g. origin/main); the check uses its merge-base with --head")
    ap.add_argument("--head", default="HEAD", help="the PR head commit (default HEAD)")
    ap.add_argument("--repo", default=".", help="repository root (default .)")
    ap.add_argument("--workers", default="auto", help="pytest-xdist workers for the base run; 0 or 1 = serial")
    ap.add_argument("--mark-expr", default="", help="a pytest -m expression applied to every pytest call (local runs)")
    ap.add_argument("--only", action="append", default=[], help="local runs: keep only changed test files whose path contains this text (repeatable)")
    ns = ap.parse_args(argv)
    return run(Path(ns.repo).resolve(), ns.base, ns.head, ns.workers, ns.mark_expr, ns.only)


if __name__ == "__main__":
    sys.exit(main())
