# Contributing

This is a two-person project. The goal of this doc is to keep us from stepping on each other's work — no accidental double-fixes, no surprise merge conflicts, no "wait, are you touching that file too?"

If something in here stops making sense or gets in the way, edit it. This is a living doc, not a contract.

## The core idea

**The repo is the source of truth for who's working on what.** Not Discord, not a doc, not memory. If it's not reflected in an issue or a draft PR, it doesn't exist as far as coordination goes.

Two habits make this work:

1. **Assign yourself an issue before you start working on it.** That's your claim.
2. **Open a draft PR as soon as you start writing code.** That's your "hands off this area for now" signal.

Everything below is just the mechanics of those two habits.

## Issues: what needs doing

Every bug, feature, or chunk of work gets an issue in the Issues tab.

- **Before starting work**, check the Issues tab for anything related. If it exists, assign yourself (Assignees field, right sidebar). If it doesn't, open one and assign yourself.
- **If an issue is assigned to the other person, don't start work on it** without talking first. They might be mid-thought on it even if there's no PR yet.
- **Labels are optional** but `bug`, `enhancement`, and `question` are worth using so we can filter.
- **Closing:** put a `Closes #12` line in the PR **description** — one per issue the PR resolves. GitHub auto-closes each when the PR merges. Three things that have bitten us:
  - **One keyword per issue.** `Closes #12` closes #12; a prose list like "#12/#13" closes nothing. Two issues → two lines.
  - **Squash-merge only reads the PR description,** not commit messages — so the `Closes` lines have to be in the PR body, not in a commit.
  - **Check after merging** that each issue actually closed; manually close any that slipped through. (`Fixes` and `Resolves` work the same as `Closes`.)
- **Out-of-scope follow-ups get their own issue, not a PR note.** If you notice separate follow-up work while doing a PR (a related bug, a deferred cleanup, a "should also do X later"), open an issue for it. A note buried in a PR description disappears once the PR merges; an issue stays on the board.

## Branches: never commit to main

Always work on a branch. The naming convention is `yourname/short-description`:

```
ThinkerOfThoughts/fix-provider-path-resolution
hanamorix/systemd-unit-cleanup
```

Create and push a branch:

```bash
git checkout main
git pull                                    # start from current main
git checkout -b yourname/what-you-are-doing
# ... make an initial commit, even a small one ...
git push -u origin yourname/what-you-are-doing
```

The `-u` sets the upstream so future `git push` and `git pull` on this branch just work.

## Draft PRs: your "I'm working here" signal

**Open a draft PR as soon as you have a branch pushed**, even if you've barely started. This is the single most important habit in this doc.

Steps:

1. Push your branch (above).
2. Go to the repo on GitHub. It'll show a yellow banner: "Compare & pull request." Click it.
3. Write a title. `WIP: <what you're doing>` is fine. Description can be one line.
4. **Click the dropdown on the green button and pick "Create draft pull request."** Not the default green button.
5. Done. The other person can now see the branch exists, what files it touches, and roughly what you're up to.

While the PR is a draft:
- It can't be accidentally merged.
- GitHub will show if your changes conflict with anything on main.
- The other person can leave comments if they see a problem early.

When you're actually done:
- Push your final commits.
- Click "Ready for review" on the PR page.
- The other person reviews, comments if needed, and clicks Merge.

## Checking before you start work

Before starting on anything, take 30 seconds to look at:

1. **Issues tab** — is there already an issue for this? Is it assigned?
2. **Pull requests tab** — filter by "Open." Is the other person already touching the files you're about to touch?

If yes to either: talk in Discord first. Otherwise proceed.

## Keeping your branch current

If the other person merges a PR while you're mid-work, your branch is now behind main. Catch up:

```bash
git checkout main
git pull
git checkout yourname/your-branch
git merge main
```

If there are conflicts, git will tell you which files. Open them, look for the `<<<<<<<` / `=======` / `>>>>>>>` markers, decide what the file should actually look like, remove the markers, `git add` the file, `git commit`.

**Do this every day or two, not once at the end.** Small merges are easy. A week-old branch merging into a changed main is a bad afternoon.

## Stacked pull requests

Sometimes one piece of work needs another that isn't merged yet, so you base the second PR on the first PR's branch instead of on main. That's a **stack**: PR B's base is PR A's branch, PR C's base is PR B's branch, and so on.

**The rule: a PR only ever merges into main.** Merge the bottom of the stack first. When PR A merges, GitHub deletes its branch (the repo has "automatically delete head branches" turned on) and **retargets PR B to main by itself**. Then B is merged, then C, one at a time, bottom-up. Never merge a PR into another PR's branch.

### After the PR below is squash-merged: rebase the next one

This repo squash-merges, so main gets PR A as one new commit, while PR B's branch still carries A's original commits. Wherever B changed lines that A also changed, B now conflicts with main, even though nothing is actually wrong. Replay only B's own commits onto main:

```bash
git fetch origin
git rebase --onto origin/main <A's last head commit> <B's branch>
git push --force-with-lease origin <B's branch>
```

`<A's last head commit>` is the commit A's branch pointed at when it merged: `gh pr view <A> --json headRefOid -q .headRefOid` prints it, and the merged PR's page shows it too. The rebase drops everything up to that commit (A's work, already on main as the squash commit) and replays B's commits on top of main. The result has the same content as the old stack. Do this before marking B ready for review. (Found on the #402 → #406 stack, #407.)

### What helps you follow it

- **The `stack-guard` check** is red on any PR whose base isn't main. Its message says what to do: "merge #N first" while the PR below is open, or "retarget this PR to main" if the PR below has already merged. Once the PR targets main it goes green. A base change re-runs it, and the check reads the PR's current base, so re-running it by hand also works.
- **A stranded-changes alarm.** If a PR does get merged into anything other than main, a comment appears on it within seconds. It says whether the changes will still reach main (the branch's own PR is still open: merge that next) or are **stranded** (that PR had already merged before this one, or was closed), and how to recover: open a PR from that branch to main. It stays quiet if the branch's PR merged *after* this one and so carried the changes along.
- Both only show up on branches that include the workflow, so merge main into an older branch to get them.

### It's a signal, not a lock

The red check can't stop the merge button. GitHub's "required checks" belong to the branch a PR merges *into*. A stacked PR merges into a feature branch, which has no protection, so even a check marked required on main never gets a say. (Issue #282 hoped for a hard block. As far as we can tell, the only way to get one, a ruleset requiring the check on every branch, would also get in the way of ordinary pushes to feature branches. That's untested, and we haven't tried it.)

What really keeps stacks safe is the **auto-delete setting**. Once PR A merges, its branch disappears within a couple of seconds and PR B is moved to main, so there's no stale branch left to merge into. The check warns you before a mistake, and the alarm tells you straight after one.

It guards against accidents, not against someone deliberately editing it: a PR runs its own copy of the check. Between the two of us, that's the point.

### Why: the F2 stack (24 September)

A stack of three PRs (#271 at the bottom, then #274, then #277 on top) was merged within 36 seconds, and not bottom-up:

- **#271** merged into main. Auto-delete was **off** then, so its branch stayed.
- **#277** merged into #274's branch while #274 was still open.
- **#274** merged into #271's branch, which had already reached main 36 seconds earlier.

GitHub showed all three as "merged", but #274 and #277 never reached main. That branch sat 23 commits ahead and had to be recovered by hand with #281. Auto-delete has been on since 25 September. With it on, #271's branch would have been deleted and #274 retargeted to main within about two seconds.

The rule is "base must be main" rather than "the PR below has merged", because the second rule would have been **green** for #274: the PR below it (#271) *had* merged.

## New tests must be able to fail

A test that cannot fail proves nothing. The `new tests must fail on the old code` check (the `new-tests-fail-on-base` job in `test.yml`, pull requests only) runs the tests a PR adds against the code the PR started from and goes red for every one that passes there.

**What counts as a new test.** A test id (`path::Class::name[param]`) that the PR's head collects in a test file the PR adds, modifies or renames (a file under `tests/` named `test_*.py` or `*_test.py`) and that the commit the PR branched from does not collect. So these are new: a test added to an old file, a new parameter value of an old test, a test function you rename, and a test you move to another file by hand. These are not new: a test whose body you changed but whose id is the same, a test file you move with `git mv` (git's rename detection, 30% similarity, maps the old ids to the new path), and anything in `conftest.py` or a helper module.

**What the check runs against.** The PR's own tests, fixtures and `pyproject.toml`, with every other top-level path (`brain/`, `scripts/`, `app/`, `docs/`, ...) replaced by its state at the commit the PR branched from. A new test *bites* if it fails there, errors there (setup included), or cannot be collected there (it imports a module or an id the old code does not have). A test that passes or is skipped there does not bite. Before judging, the check runs one unchanged unit test file against the swapped tree; if that fails, the old code cannot run the new conftest and the check says so instead of reporting every test as biting.

**When a test cannot fail on the old code by design** (a refactor guard, a snapshot of what already exists, test-suite machinery, a rewrite of an existing test under a new id), mark it:

```python
@pytest.mark.no_base_failure(reason="guards what the refactor must not change")
def test_the_refactor_keeps_the_behaviour(): ...
```

The reason is mandatory (a mark without one stops the suite at collection). Mark the function, not the module, so tests added to that file later are still checked. The check counts the marked new tests and lists them, so the escape hatch is visible in the job log. A test that skips on Linux can never bite there; a Windows-only or macOS-only new test needs the mark with a reason that says so.

**Run it locally** (needs `pytest-xdist` for `--workers`, otherwise `--workers 0`; it works in its own temporary git worktrees and leaves your checkout alone, but it checks committed work, so commit first):

```bash
python scripts/new_tests_fail_on_base.py --base origin/main --workers 0
```

`--only <text>` limits a local run to changed test files whose path contains the text; `--mark-expr "<expr>"` adds a `-m` expression to every pytest call.

## The suite runs in parallel, in random order, with a time budget

CI runs `uv run pytest -n auto --dist loadgroup -p no:tdd-guard` on all three operating systems (one worker per CPU; `--dist loadgroup` keeps the tests of one `xdist_group` on one worker). A test that shares a real resource with others (the Hugging Face cache, the systemd user manager, a network install, a fixed port, a fixed file name) must carry `@pytest.mark.xdist_group("<name>")`; `tests/unit/test_xdist_groups.py` checks the known ones. To run one test on one operating system in CI, serially: `gh workflow run test.yml --ref <branch> -f tests="tests/x.py::test_y" -f os=windows-latest` (`workers` defaults to `1`; `2`, `4` and `auto` run it in parallel; `seed` reruns a given random order).

The order is random (`pytest-randomly`): the seed is printed in the header and again at the end of the run (`pytest-randomly seed: N`). A test that only passes in file order has an order dependence; fix the test, do not pin the order. To reproduce a red order, run serially with `--randomly-seed=N` (a parallel run with the same seed can still distribute tests differently); `-p no:randomly` switches the shuffling off while you debug.

**The slow-test budget.** In the CI python job (`KINDLED_SLOW_BUDGET=1`; set it locally to try) a test whose call phase takes more than 2 s fails unless it carries `@pytest.mark.slow(reason="...")` on the test function, with a reason a reviewer can weigh. Setup and teardown are not budgeted (do the heavy work in a fixture), and tests marked `requires_network`, `requires_models`, `requires_claude_cli` or `live` are exempt. The number of slow marks is pinned in `tests/unit/test_slow_test_budget.py` and can only go down: when you make a slow test fast and remove its mark, lower the pin to the count that test's failure message shows; a new slow test needs the pin raised there, which shows in the diff. A weekly run (`timing.yml`) uploads a timing report per operating system.

## Tests must not start a real `claude` in CI

The product starts the claude CLI by its bare name. CI has no `claude`, so a test that reaches that spawn unpatched can only fail the spawn, and a test that tolerates the failure passes without testing what it claims (on a machine that has `claude`, it would spend tokens). When `GITHUB_ACTIONS` is set (or `KINDLED_GUARD_CLAUDE=1` locally, to try it), `tests/claude_spawn_guard.py` fails any test without the `requires_claude_cli` marker that starts a `claude` by bare name through `subprocess` (so `run`, `Popen`, `check_output` and asyncio subprocesses), even if the test catches the error. The message says where the spawn came from. Ways out: patch the spawn (a patched `subprocess.Popen` or provider helper is never seen by the guard); put a fake `claude` script under `tmp_path` and spawn it by explicit path, or put its directory first on the `PATH` you pass; or mark the test `@pytest.mark.requires_claude_cli`. Spawns by explicit path are never blocked. Not seen: spawns made by a child process of the test, by module- or session-scoped fixtures, or by a thread that was already running when the test began, and `shell=True`, `os.system` or `os.posix_spawn` spawns.

## What Discord is for

Discord is still useful, just not as the source of truth:

- "Hey, about to force-push to my branch, don't pull for 5 min"
- "Can you look at PR #14 when you get a sec"
- "I'm gonna pick up issue #22 today, cool?"
- Real-time debugging back-and-forth
- Anything conversational that doesn't need to be archived

If it's a decision that affects the code or the plan, it belongs in an issue or PR comment, not just Discord — otherwise it's lost in scrollback in a week.

## Things not to do

- **Don't commit to main directly.** If you catch yourself typing `git checkout main` and then editing a file, stop and make a branch.
- **Don't force-push to a branch the other person has pulled** without warning them first (Discord is fine for this). Force-pushing rewrites history and will confuse their local copy.
- **Don't keep a branch open for weeks.** The longer it lives, the worse the merge. Better to split work into smaller PRs.
- **Don't merge a PR into another PR's branch.** Merge bottom-up into main and let GitHub retarget the next PR (see "Stacked pull requests"). The `stack-guard` check is red until a PR targets main.
- **Don't add a `WIP.md` or `CLAIMS.md` file to the repo to track who's on what.** It just moves the merge-conflict problem into the repo itself. Draft PRs already do this job.

## Quick reference

| I want to... | Do this |
|---|---|
| Start new work | Assign yourself an issue → branch → initial commit → push → open draft PR |
| Check what the other person is doing | Pull requests tab (filter: Open), Issues tab (filter: Assignee) |
| Mark work as ready | Click "Ready for review" on the draft PR |
| Catch up to main | `git checkout main && git pull && git checkout your-branch && git merge main` |
| Claim an issue | Assign yourself in the right sidebar of the issue |
| Close an issue with a PR | One `Closes #N` line per issue, in the PR description (squash-merge ignores commit-message keywords) |
| Merge a stack of PRs | Bottom-up, one at a time, each into main; GitHub retargets the next one (the `stack-guard` check flags a PR that isn't targeting main) |
| Re-run one test on one OS (e.g. a Windows-only failure) | `gh workflow run test.yml --ref <branch> -f tests="tests/x.py::test_y" -f os=windows-latest` (`os`: `all` by default; `tests`: space-separated paths or node ids, no options; runs only the Python test job) |
