# A guard against excessive and redundant tests (#746, part 2)

Part 1 of #746 (#774) split the Windows python job into shards, which treats the symptom. The suite keeps growing: on
main it gained about 2,400 collected tests in under three weeks (2026-10-28 to 2026-11-16), about half of them parametrize
expansions. The existing "new tests must fail on the old code" check (#540) catches a new test that can't fail, but not
one that repeats an existing test, a parametrize grid far larger than the behaviour needs, or a test that adds one id per
file it scans. This guard makes a PR's test growth visible at review time and flags those patterns.

## Goals

- Every PR's checks show how many test functions and collected test ids it adds and removes, per changed test file.
- Scanner tests and cross-product parametrize grids are flagged. After a trial they block the merge, unless the test
  carries a stated reason.
- Single-axis grids larger than the change, and likely duplicate tests, are reported to the reviewer and never block.
- "Too many" is derived from the test or the change, never a fixed count.

## Non-goals

- Trimming the existing suite. The guard looks only at what a PR adds.
- Coverage-based redundancy: which lines and assertions a new test covers that an existing one already does (see
  Approaches).
- Changing the "new tests must fail on the old code" check. Its script is imported, never edited.

## Anti-goals

- A guard that stalls good PRs on a guess. Only the two structural flags can ever block, and only after a trial.
- A noisy check people learn to ignore. A flag names the test and the rule it broke, and says how to justify or trim it.
- Adding meaningful CI time. The guard collects only the PR's changed test files, and runs no tests.

## Constraints

- **No edit to `tests/conftest.py`.** ThinkerOfThoughts' #803 and #655 both change it. The marker's "reason is mandatory"
  rule is enforced by the guard itself, not by the conftest collection hook that enforces `no_base_failure`'s.
- **No edit to `scripts/new_tests_fail_on_base.py`.** Its collection code is imported read-only.
- Open PRs pick up the new job on their next CI run, because a PR's checks run the workflow merged with main. During the
  trial the job only adds a summary.
- Python 3.12 or later, standard library plus pytest, which the project already has. Ubuntu runner, like the new-tests job.

## Approaches (picked: A)

| | What | Trade-off |
|---|---|---|
| **A, picked** | A separate script and CI job that imports the new-tests check's collection helpers, plus a small collection plugin loaded with `-p` | One more job. The existing gate is untouched, and `conftest.py` is not edited |
| B | Extend `new_tests_fail_on_base.py` | One job, but it mixes a report-only guard into a working gate, which raises the risk of breaking it |
| C | A pytest plugin loaded from `conftest.py` | It sees parametrize data directly, but it collides with #803 and #655 |

Duplicates: compare normalised test bodies (picked) rather than per-test coverage, which would mean a coverage run of all
~14,000 tests on every PR, the CI time #746 is about. Hana agreed both on 2026-11-24. The plan is posted on #746 for
ThinkerOfThoughts (comment 7138503936).

## Design

### The script: `scripts/guard_test_growth.py`

`uv run python scripts/guard_test_growth.py --base <sha> --head <sha> [--enforce]`

(It isn't named `test_*.py`, which pytest's `python_files` pattern would treat as a test module.)

1. **Base and head.** It resolves the merge-base and takes the PR's changed test files and renames with the new-tests
   script's `parse_name_status` and `changed_test_files`. With no changed test file, it writes "no test changes" and
   exits 0.
2. **Collection.** It collects the changed files in a head worktree, and their base paths in a base worktree (each tree
   with its own tests and code), with pytest's `--collect-only`. A small plugin, loaded with `-p` from a temp directory
   the way the new-tests script loads its recorder, writes one JSON row per collected item:
   - the node id;
   - the function id (the node id without its `[...]` part);
   - the item's parametrize values: its `callspec.params`, each value as its `repr` (see "Calls made after the builder
     check");
   - the `large_grid` reasons on the test function, and any problem with a `large_grid` mark (no reason, or on a class,
     module or `pytest.param`).

   Base ids of renamed files are compared under the new path (`remap_base_ids`), so a moved test isn't new.
3. **Counts.** For each changed test file: functions and ids at base and head, added, and removed. Plus the totals.
4. **Flags**, applied to new test functions only (a function with at least one new id):
   - **Scanner.** The function's parametrize values are built at collection time from the repository. That means the
     values argument in the source (read with `ast`) is not a literal list or tuple, and it calls, or names a
     module-level value built by calling, any of `glob`, `rglob`, `iterdir`, `os.listdir`, `os.scandir`, `os.walk`, or
     a `git ls-files` subprocess. Each scanned item adds an id, so the count grows with the repository, not with what the
     test checks. The fix is usually one test that checks every item and reports all failures together.
   - **Cross-product grid.** A function with two or more parametrize axes whose id count is larger than the sum of its
     axis sizes. That is, more than 2×2 crossed: 2×2 gives 4 against 4 and isn't flagged, while 2×3 gives 6 against 5.
     The rule uses the id count, not the product of the axis sizes. A single `parametrize("a,b", [(1, 2), (3, 4), (5, 6)])`
     has two axes of 3 values but only 3 ids, so it isn't a cross and isn't flagged.
   - **Grid larger than the change (report only, never blocks).** A single-axis grid whose new ids exceed the PR's changed
     product lines: non-blank lines added or removed in `.py`, `.ts`, `.tsx` and `.rs` files outside `tests/`. A
     test-only PR has a budget of 0, so any new grid of 2 or more ids is listed there. It is a hint for the reviewer:
     a small grid on a fix is usually good testing. Hana decided on 2026-11-24 that it never blocks, and that only the
     scanner and cross-product flags can.
5. **Justifying a flag.** `@pytest.mark.large_grid(reason="...")` on the test function turns a scanner or cross-product
   flag into a listed exemption, shown with its reason. It also turns a single-axis report into an exemption. A `large_grid` mark with no reason, or one on a
   class or module, is a problem of its own, found with `tests/no_base_failure.py`'s `marker_problems(items, "large_grid")`.
6. **Likely duplicates (report only).**
   - Each new test function's body is parsed with `ast`. Identifiers are replaced by their first-seen order, constants by
     their type, and docstrings are dropped.
   - It is compared, with `difflib.SequenceMatcher`, against every other test function in the same file and in the same
     folder at head.
   - A pair whose similarity ratio is at least 0.9 is reported, and bodies under 20 tokens are skipped, since tiny tests
     always look alike.
   - The threshold is a similarity, not a count, so it doesn't break the "derived, not fixed" goal.
7. **Output.**
   - A Markdown summary written to `$GITHUB_STEP_SUMMARY` when set, and always to stdout: a counts table, then the flags,
     the exemptions and the likely duplicates.
   - Each flag names the test, the rule, the numbers behind it, and the two ways out: trim the test, or add `large_grid`
     with a reason.
8. **Exit status.**
   - Without `--enforce` (the trial): 0, whatever is flagged.
   - With `--enforce`: 1 when a scanner or cross-product flag has no `large_grid` reason, or a `large_grid` mark has a
     problem. A single-axis grid and a likely duplicate never fail it.
   - In either mode: 1 when the guard can't judge (collection crashed, or the ids read don't match pytest's count, as in
     the new-tests script), so a broken guard is seen and fixed; and 2 for a usage or git error. During the trial the job
     itself doesn't go red on these (see "Calls made after the builder check").

### The job

`.github/workflows/test.yml` gets a job, `test growth guard`, on `pull_request` only. It runs on Ubuntu with Python 3.12,
fetches with `fetch-depth: 0`, and runs the script against the merge-base, without `--enforce` during the trial. It
needs no xdist and runs no tests.

### The marker

`pyproject.toml`'s `markers` list gains `large_grid(reason)` with a one-line description pointing to `CONTRIBUTING.md`.

### `CONTRIBUTING.md`

A section, "Test growth: keep grids and scans small", after "New tests must be able to fail". It covers:
- what the guard shows;
- the two flags that can block (scanner and cross-product) and their rules, and the single-axis report;
- how to write one test that checks every scanned item;
- when `large_grid` is the right answer;
- that duplicates are hints for the reviewer.

## Rollout

1. This PR: the guard in report-only mode.
2. The trial: a couple of weeks of PRs, reading what the flags catch.
3. Before blocking: run the guard with `--enforce` on whatever of #803, #655 and #656 is still open, show
   ThinkerOfThoughts what it flags, and agree the switch with him.
4. The switch: a one-line change adding `--enforce` to the job, plus any rule fixes the trial showed. A separate PR.

## Testing

- Unit tests for every pure step, with synthetic inputs:
  - the counts from base and head rows;
  - each flag rule, at its boundary (2×2 not flagged, 2×3 flagged; ids equal to the budget not flagged, one more
    flagged);
  - scanner detection for each listed call and for a literal list;
  - the `large_grid` exemption and its problems;
  - the body normaliser;
  - the duplicate threshold and the short-body skip;
  - the exit statuses in both modes.
- One end-to-end test on a temp git repository: a base commit and a head commit that adds a scanner test, a 2×3 grid and
  a near-duplicate. The test runs the script and reads its summary. The heavy part goes in a module fixture, to stay within
  the 2 s CPU budget.

## Calls made after the builder check

These are my calls, settled after a cold builder read; Hana can change any of them.
- **Scanner detection depth:** it follows a values argument into a module-level name, or into a function defined in the
  same file, one level deep. It doesn't follow helpers imported from other modules. A miss there only means a scanner
  goes unflagged, and the trial will show whether that matters.
- **Where `large_grid` goes:** on the test function only, because the flags are per function. A `large_grid` mark on a
  `pytest.param` counts as a marker problem ("put it on the function").
- **Grid sizes come from `item.callspec`, not only from `parametrize` marks.** So ids built by parametrized fixtures
  (`params=`), `indirect=` and `pytest_generate_tests` are counted too. For each function, an axis is a `callspec.params`
  key, and its size is the number of distinct values seen across the function's new items. Scanner detection still reads
  only `parametrize` marks' source.
- **Product lines:** the merge-base to head diff (`git diff --numstat`), non-blank lines only, in `.py`, `.ts`, `.tsx` and
  `.rs` files. Excluded: `tests/`, any `*.test.*` file, and any file under a `__tests__` folder.
- **"Same folder" for duplicates:** the new test's own file and the other test files directly in its folder (not
  subfolders), at head, whether or not the PR changed them.
- **The trial job uses `continue-on-error: true`.** A guard crash then shows as a warning on the job, not a red check on
  every open PR. It is removed in the PR that adds `--enforce`.

## Left to the builder

- Module layout inside the script, and the summary's exact wording and table format.
- How the plugin and the script share the row format.
- Whether the base collection reuses the new-tests script's `_collect` directly or a thin wrapper around it.
