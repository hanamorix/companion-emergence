# Test-growth guard Implementation Plan (#746, part 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A report-only CI check that shows each PR's test growth and flags scanner tests, cross-product grids,
single-axis grids larger than the change, and likely duplicates. The check can later be switched to block on the first two.

**Architecture:** Three files under `scripts/`.
- The pure rules (`guard_test_growth_rules.py`).
- A pytest collection plugin that writes one JSON row per item (`guard_test_growth_plugin.py`).
- The CLI (`guard_test_growth.py`). It makes base and head worktrees, collects the PR's changed test files with the
  plugin, applies the rules, writes a Markdown summary and sets the exit status.

The CLI imports `scripts/new_tests_fail_on_base.py`'s helpers read-only. A new job in `test.yml` runs it with
`continue-on-error: true`.

**Tech Stack:** Python 3.12, stdlib (`ast`, `difflib`, `json`, `subprocess`), pytest, PyYAML (tests only), GitHub Actions.

**Spec:** `docs/source-spec/2026-11-24-746-test-growth-guard-design.md`

## Global Constraints

- Never edit `tests/conftest.py` or `scripts/new_tests_fail_on_base.py`. The CLI imports the latter with
  `importlib.util.spec_from_file_location`, as `tests/unit/scripts/test_new_tests_fail_on_base.py` does.
- Script and module names must not match `test_*.py` or `*_test.py`.
- Marker: `large_grid(reason)`. The reason is mandatory, and the mark goes on the test function only.
- Thresholds, verbatim from the spec:
  - **cross-product:** ≥2 axes and ids > sum of axis sizes;
  - **single-axis:** exactly 1 axis, ≥2 ids, and new ids > changed product lines;
  - **duplicate:** ratio ≥ 0.9, and bodies under 20 tokens are skipped.
- Scanner calls: `glob`, `rglob`, `iterdir`, `listdir`, `scandir`, `walk`, or any string constant containing `ls-files`.
- Product lines: non-blank `+` or `-` lines of the merge-base→head `git diff -U0`, in `.py`, `.ts`, `.tsx` and `.rs`
  files. Excluded: paths under `tests/`, any `*.test.*` file, and any path with a `__tests__/` segment.
- Exit status:
  - without `--enforce`: 0;
  - with `--enforce`: 1 when a scanner or cross-product flag has no exemption, or there is any marker problem;
  - in both modes: 1 when the guard can't judge, and 2 for a usage or git error.
- Every file read as text uses `encoding="utf-8"`. Every subprocess with `text=True` passes `encoding="utf-8"`
  (`scripts/lint_windows_pitfalls.py`).
- Per-test CPU budget is 2 s. Put any subprocess or git-repo setup in a `scope="module"` fixture.
- In a worktree, run red tests with `-o tdd_guard_project_root=<the main checkout's path>`, so tdd-guard reads them.

## Review Focus

1. **A changed test file deleted at head, or new at head with no base version.** Counts treat the missing side as 0
   functions and 0 ids, and don't crash. Pinned in Task 2 and Task 7.
2. **A head test file that fails to import.** The guard reports "can't judge" (exit 1), never "no new tests". Pinned in
   Task 7.
3. **A parametrize value whose `repr` has a memory address** (an object without `__repr__`). Distinct values stay
   distinct, and an unstable `repr` can't merge two values: axes count the distinct `repr`s within one collection. Pinned
   in Task 1.
4. **A scanner helper that isn't in the same file.** It isn't flagged, and nothing crashes. Pinned in Task 4.
5. **A test class method.** Scanner and duplicate detection see `Class::method` qualnames that match the node ids.
   Pinned in Task 4 and Task 5.

---

### Task 1: Collection plugin

**Files:**
- Create: `scripts/guard_test_growth_plugin.py`
- Test: `tests/unit/scripts/test_guard_test_growth_plugin.py`

**Interfaces:**
- Produces:
  - Env var `TGG_ROWS_FILE`. The plugin writes there a JSON list of rows, `{"nodeid": str, "params": {name: repr},
    "large_grid": [reason, ...], "problems": [str, ...]}`, in `pytest_collection_finish`.
  - `large_grid` holds reasons from marks on the function definition (`item.function.pytestmark`).
  - `problems` combines `tests.no_base_failure.marker_problems([item], "large_grid")`, plus
    `"<nodeid>: @pytest.mark.large_grid is on a pytest.param; mark the test function itself"` for a `large_grid` in
    `own_markers` that isn't in the function's `pytestmark`.

- [ ] **Step 1: Write the failing test.** A module fixture writes a temp package containing `tests/__init__.py`, a copy
  of `tests/no_base_failure.py`, and `tests/test_sample.py` with:
  - `test_plain`;
  - `test_cross`, with `@parametrize("a", [1, 2])` and `@parametrize("b", [1, 2, 3])` stacked;
  - `test_justified`, with `@pytest.mark.large_grid(reason="r")` and `@parametrize("x", [1, 2])`;
  - `test_no_reason`, with a bare `@pytest.mark.large_grid` and `@parametrize("x", [1])`;
  - `test_param_mark`, with `@parametrize("x", [pytest.param(1, marks=pytest.mark.large_grid(reason="r"))])`;
  - `test_obj`, with `@parametrize("o", [object(), object()])`.

  The fixture then runs `python -m pytest --collect-only -q -p guard_test_growth_plugin tests/test_sample.py`. It sets
  `PYTHONPATH` to the temp dir and `scripts/`, and `TGG_ROWS_FILE`. Assertions:

```python
def test_rows_record_nodeid_and_params(rows):
    by_id = {r["nodeid"]: r for r in rows}
    assert by_id["tests/test_sample.py::test_plain"]["params"] == {}
    assert by_id["tests/test_sample.py::test_cross[1-3]"]["params"] == {"a": "1", "b": "3"}
    assert len([r for r in rows if "test_cross[" in r["nodeid"]]) == 6
```

- [ ] **Step 2: Run it.** `uv run pytest tests/unit/scripts/test_guard_test_growth_plugin.py -o tdd_guard_project_root=...`.
  Expected: FAIL (the rows file isn't written, because the plugin module doesn't exist).
- [ ] **Step 3: Implement `pytest_collection_finish(session)` in `scripts/guard_test_growth_plugin.py`.** It writes the rows
  with `open(os.environ["TGG_ROWS_FILE"], "w", encoding="utf-8")`, and does nothing when the variable is unset. Params
  come from `getattr(item, "callspec", None)` and are stored as `repr(v)`.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Add one test at a time, red then green, each from the same fixture:**
  - `test_large_grid_reason_on_function_is_recorded`: `test_justified`'s rows have `large_grid == ["r"]` and `problems == []`.
  - `test_large_grid_without_reason_is_a_problem`: a `test_no_reason` row's problems contain `needs reason=`.
  - `test_large_grid_on_param_is_a_problem`: `test_param_mark`'s problems contain `is on a pytest.param`.
  - `test_object_params_stay_distinct`: the two `test_obj` rows have different `params["o"]`.
- [ ] **Step 6: Commit.** `git commit -m "add the test-growth guard's collection plugin (#746)"`

### Task 2: Counts and grid rules

**Files:**
- Create: `scripts/guard_test_growth_rules.py`
- Test: `tests/unit/scripts/test_guard_test_growth_rules.py`

**Interfaces:**
- Produces (in `guard_test_growth_rules`):
  - `function_of(nodeid: str) -> str`: the node id without a trailing `[...]`.
  - `@dataclass(frozen=True) class FileCount: path: str; base_functions: int; head_functions: int; base_ids: int; head_ids: int; added_ids: int; removed_ids: int`
  - `count_files(files: list[str], base_ids: set[str], head_ids: set[str]) -> list[FileCount]`. The ids are already
    remapped to head paths; a file's ids are those starting with `path + "::"`.
  - `axes(rows: list[dict]) -> dict[str, int]`: per param name, the number of distinct `repr`s across the rows.
  - `is_cross_product(ids: int, axis_sizes: dict[str, int]) -> bool`
  - `is_single_axis_over_budget(new_ids: int, axis_sizes: dict[str, int], budget: int) -> bool`

- [ ] **Step 1: Write the failing tests, one per cycle**, each named for its case:
  - `count_files(["tests/test_a.py"], {"tests/test_a.py::t[1]", "tests/test_a.py::old"}, {"tests/test_a.py::t[1]", "tests/test_a.py::t[2]", "tests/test_a.py::u"})`
    gives `FileCount("tests/test_a.py", 2, 2, 2, 3, 2, 1)`.
  - A file with no base ids gives base counts of 0, and a file with no head ids gives head counts of 0.
  - `is_cross_product(4, {"a": 2, "b": 2}) is False`
  - `is_cross_product(6, {"a": 2, "b": 3}) is True`
  - `is_cross_product(3, {"a": 3, "b": 3}) is False`, the zipped-tuple case.
  - `is_single_axis_over_budget(3, {"x": 3}, 2) is True`
  - `is_single_axis_over_budget(2, {"x": 2}, 2) is False`
  - `is_single_axis_over_budget(1, {"x": 1}, 0) is False`, since a single id isn't a grid.
  - `is_single_axis_over_budget(6, {"a": 2, "b": 3}, 0) is False`, since two axes aren't a single-axis grid.
- [ ] **Step 2:** Run each to see it fail, implement, and run to see it pass.
- [ ] **Step 3: Commit.** `git commit -m "add the guard's counts and grid rules (#746)"`

### Task 3: Changed product lines

**Files:**
- Modify: `scripts/guard_test_growth_rules.py`
- Test: `tests/unit/scripts/test_guard_test_growth_rules.py`

**Interfaces:**
- Produces:
  - `is_product_path(path: str) -> bool`
  - `product_lines(diff_u0: str) -> int`, from `git diff -U0` text. It tracks the current file from `+++ b/<path>` lines,
    skips `/dev/null`, and counts `+` and `-` lines (never `+++` or `---`) whose rest is non-blank.

- [ ] **Step 1: Failing tests, one per cycle:**
  - `is_product_path("brain/x.py")` and `"app/src/a.tsx"` are True.
  - `is_product_path("tests/unit/x.py")`, `"app/src/a.test.tsx"`, `"app/src/__tests__/b.ts"` and `"docs/x.md"` are False.
  - `product_lines` on a two-file diff (one `brain/` file with 2 added, 1 removed and 1 blank added line, one `tests/` file
    with 5 added lines) gives `3`.
  - A deleted file (`+++ /dev/null`, preceded by `--- a/brain/gone.py` with 2 removed lines) counts its removed lines. It
    takes the path from the `--- a/` line when `+++` is `/dev/null`, giving `2`.
- [ ] **Step 2:** Implement and see it green.
- [ ] **Step 3: Commit.** `git commit -m "count the PR's changed product lines for the guard (#746)"`

### Task 4: Scanner detection

**Files:**
- Modify: `scripts/guard_test_growth_rules.py`
- Test: `tests/unit/scripts/test_guard_test_growth_rules.py`

**Interfaces:**
- Produces: `scanner_tests(source: str) -> set[str]`, the qualnames (`func` or `Class::func`) of test functions with a
  `parametrize` decorator whose values argument scans:
  - The values argument is the second positional argument, or `argvalues=`.
  - A literal list, tuple or set is never a scan.
  - Otherwise, it's a scan if its subtree has a call to a scanner name (a `Name` or `Attribute.attr` in the Global
    Constraints set) or a string constant containing `ls-files`.
  - Looking up one level: a `Name` resolves to a module-level assignment's value, and a call to a same-file function
    resolves to that function's body.

- [ ] **Step 1: Failing tests, one per cycle**, each on a short source string:
  - `parametrize("p", sorted(Path("x").glob("*.py")))` gives `{"test_a"}`.
  - `os.walk`, `os.listdir`, `Path.iterdir`, `rglob` and `subprocess.run(["git", "ls-files"])` as values are each
    flagged. Use one test over the five sources with a plain `for` loop, not a parametrize.
  - A literal `[1, 2, 3]` gives `set()`.
  - A module-level `FILES = list(Path(".").rglob("*.json"))` with `parametrize("f", FILES)` is flagged.
  - A same-file `def _discover(): return os.listdir(".")` with `parametrize("f", _discover())` is flagged.
  - `from helpers import discover` with `parametrize("f", discover())` gives `set()` (Review Focus 4).
  - A method under `class TestX:` gives `{"TestX::test_m"}` (Review Focus 5).
- [ ] **Step 2:** Implement and see it green.
- [ ] **Step 3: Commit.** `git commit -m "detect scanner tests from their parametrize source (#746)"`

### Task 5: Likely duplicates

**Files:**
- Modify: `scripts/guard_test_growth_rules.py`
- Test: `tests/unit/scripts/test_guard_test_growth_rules.py`

**Interfaces:**
- Produces:
  - `test_bodies(source: str) -> dict[str, list[str]]`: qualname to normalised tokens, for every `test*` function or
    method. Each identifier becomes `v<n>` in first-seen order, each constant becomes `c:<type name>`, and the docstring is
    dropped. Tokens come from `ast.dump` of the body statements with names and constants replaced.
  - `likely_duplicates(new: dict[str, list[str]], pool: dict[str, list[str]], *, threshold: float = 0.9, min_tokens: int = 20) -> list[tuple[str, str, float]]`:
    keys are `path::qualname`. It never pairs a test with itself, and returns each pair once, sorted by ratio, highest
    first.

- [ ] **Step 1: Failing tests, one per cycle:**
  - Two tests that differ only in variable names and literal values come back as one pair with a ratio of at least 0.9.
  - Two tests with different call structure come back as no pair.
  - Bodies under 20 tokens are skipped, even when identical.
  - A test never pairs with itself.
  - Method qualnames are `TestX::test_m` (Review Focus 5).
- [ ] **Step 2:** Implement and see it green.
- [ ] **Step 3: Commit.** `git commit -m "report likely duplicate tests by normalised body (#746)"`

### Task 6: The report, the summary and the exit status

**Files:**
- Modify: `scripts/guard_test_growth_rules.py`
- Test: `tests/unit/scripts/test_guard_test_growth_rules.py`

**Interfaces:**
- Consumes: Tasks 2-5.
- Produces:
  - `@dataclass class Flag: function: str; rule: str; detail: str`. `rule` is one of `"scanner"`, `"cross-product"` and
    `"single-axis"`.
  - `@dataclass class Report: counts: list[FileCount]; flags: list[Flag]; exempt: list[tuple[Flag, str]]; problems: list[str]; duplicates: list[tuple[str, str, float]]; budget: int`
  - `build_report(*, counts, head_rows: list[dict], new_ids: set[str], scanners: set[str], budget: int, duplicates) -> Report`.
    It considers only functions with at least one id in `new_ids`, and collects problems from the rows. A flagged function
    whose rows carry a `large_grid` reason moves to `exempt` with the first reason. `scanners` holds
    `path::qualname` function ids.
  - `render_summary(report: Report) -> str`: Markdown with a counts table (file, functions base→head, ids base→head,
    +added, −removed, and a totals row). Then sections "Flagged (blocks once enforced)", "Reported (never blocks)",
    "Justified with large_grid", "Marker problems" and "Likely duplicates". Each flag line names the function, the rule,
    `detail`, and "trim it, or add @pytest.mark.large_grid(reason=...)".
  - `exit_status(report: Report, *, enforce: bool) -> int`

- [ ] **Step 1: Failing tests, one per cycle:**
  - A 2×3 new function gives one `cross-product` flag.
  - The same function with a `large_grid` reason moves to `exempt`.
  - A function with no new ids is never flagged.
  - `exit_status(..., enforce=False)` is 0 with a scanner flag.
  - `exit_status(..., enforce=True)` is 1 with an unexempted scanner flag, 1 with a marker problem, and 0 with only
    single-axis reports and duplicates.
  - `render_summary` contains the totals row and each section heading that has content.
- [ ] **Step 2:** Implement and see it green.
- [ ] **Step 3: Commit.** `git commit -m "build the guard's report, summary and exit status (#746)"`

### Task 7: The CLI

**Files:**
- Create: `scripts/guard_test_growth.py`
- Test: `tests/unit/scripts/test_guard_test_growth.py`

**Interfaces:**
- Consumes: from `new_tests_fail_on_base`, `parse_name_status`, `changed_test_files`, `remap_base_ids`,
  `parse_collected`, `collected_count`, `_git` and `_pytest`; plus Tasks 1-6.
- Produces:
  - `main(argv: Sequence[str] | None = None) -> int` with `--base`, `--head`, `--enforce` and `--repo` (default `.`).
  - `run(repo: Path, base: str, head: str, *, enforce: bool) -> int`. It writes the summary to stdout, and appends it to
    `$GITHUB_STEP_SUMMARY` when that is set.
- Behaviour:
  - Merge-base, then the changed test files. `changed_test_files` drops deleted files, so it also takes every
    `status == "D"` entry whose old path passes `is_test_file`. Those are collected at base only, and counted with 0 head
    ids. With no changed or deleted test file, it prints "no test changes" and returns 0.
  - It adds detached worktrees for head and base in a temp dir and always removes them (`finally`).
  - It copies `guard_test_growth_plugin.py` into a temp plugin dir.
  - It collects head files in the head tree, and existing base paths in the base tree, with `--collect-only -q
    --continue-on-collection-errors -p guard_test_growth_plugin`. The plugin dir goes on `PYTHONPATH`, through
    `_pytest`'s `NTFOB_PLUGIN_DIR` key.
  - Head collection must exit 0 or 5, and its parsed ids must equal pytest's printed count; otherwise it prints "can't
    judge" and returns 1. Base collection tolerates 1 and 2.
  - The budget is `product_lines(git diff -U0 base head)`.
  - Scanners and duplicates come from head file sources. The duplicate pool is every `test_*.py` file directly in each
    changed file's folder.

- [ ] **Step 1: Failing end-to-end tests**, from a module fixture. The fixture builds a temp git repo with
  `pyproject.toml` (`[tool.pytest.ini_options] testpaths=["tests"]`), `tests/__init__.py`, `tests/no_base_failure.py`
  (copied), `brain/__init__.py` and a one-line `brain/x.py`. That is the base commit. The head commit adds
  `tests/test_new.py` with a scanner test, a 2×3 grid, and two near-identical 25-token tests, and changes one line of
  `brain/x.py`. The fixture runs `main(["--repo", str(repo), "--base", base_sha, "--head", head_sha])` once and keeps the
  return value and the captured stdout. Assertions, one per cycle:
  - The return value is 0.
  - Stdout has a totals row with the right added-id count.
  - Stdout names the scanner flag, the cross-product flag and the duplicate pair.
  - A second run with `--enforce`, in the same fixture, returns 1.
- [ ] **Step 2: A failing test for Review Focus 1 and 2.** A head commit that deletes a changed test file is counted with
  0 head ids. A head test file with a syntax error makes `run` return 1, with "can't judge" in stdout.
- [ ] **Step 3:** Implement and see each green.
- [ ] **Step 4: Commit.** `git commit -m "add the test-growth guard CLI (#746)"`

### Task 8: CI job, marker and contributor docs

**Files:**
- Modify: `.github/workflows/test.yml` (a new job after `new-tests-fail-on-base`)
- Modify: `pyproject.toml` (the `markers` list)
- Modify: `CONTRIBUTING.md` (a new section after "New tests must be able to fail")
- Test: `tests/unit/scripts/test_ci_test_growth_guard_746.py`

**Interfaces:**
- Produces:
  - Job id `test-growth-guard`, named `test growth guard`, with `if: github.event_name == 'pull_request'`,
    `runs-on: ubuntu-latest`, `continue-on-error: true` and `timeout-minutes: 15`.
  - Its steps are the same as `new-tests-fail-on-base` (checkout with `fetch-depth: 0`, uv, Python 3.12, `uv sync --all-extras`), then
    `uv run python scripts/guard_test_growth.py --base "$base" --head HEAD`, with `base` taken from the PR's merge-base
    as in that job.
  - The marker line:
    `"large_grid(reason): a new parametrized or scanning test kept on purpose; exempts it from the test-growth guard's flags (#746, CONTRIBUTING.md). The reason is mandatory and the mark goes on the test function."`

- [ ] **Step 1: Failing tests** (YAML read the way `test_ci_windows_shards_746.py` reads it):
  - the job exists with `continue-on-error: True`;
  - its run step calls `scripts/guard_test_growth.py` without `--enforce`;
  - `pyproject.toml`'s `markers` has a `large_grid(reason)` entry.
- [ ] **Step 2:** Implement and see it green. Write the CONTRIBUTING section the spec lists: what the guard shows, the two
  blocking flags and their rules, the single-axis report, how to write one test that checks every scanned item, when
  `large_grid` is right, and that duplicates are hints.
- [ ] **Step 3: Full gate.** Run each and check:
  - `uv run pytest`: 0 `^FAILED`;
  - `uv run ruff check .`;
  - `uv run python scripts/lint_windows_pitfalls.py`;
  - `uv run python scripts/new_tests_fail_on_base.py --base "$(git merge-base HEAD origin/main)"`.
- [ ] **Step 4: Run the guard on this branch itself**, `uv run python scripts/guard_test_growth.py --base "$(git
  merge-base HEAD origin/main)" --head HEAD`, and paste its summary into the PR description.
- [ ] **Step 5: Commit.** `git commit -m "run the test-growth guard on every PR, report-only (#746)"`
