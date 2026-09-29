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

### What helps you follow it

- **The `stack-guard` check** is red on any PR whose base isn't main. Its message says what to do: "merge #N first" while the PR below is open, or "retarget this PR to main" if the PR below has already merged. Once the PR targets main it goes green. A base change re-runs it, and the check reads the PR's current base, so re-running it by hand also works.
- **A stranded-changes alarm.** If a PR does get merged into anything other than main, a comment appears on it within seconds. It says whether the changes will still reach main (the branch's own PR is still open: merge that next) or are **stranded** (that PR already merged or was closed), and how to recover: open a PR from that branch to main.

### It's a signal, not a lock

The red check can't stop the merge button. GitHub's "required checks" belong to the branch a PR merges *into*. A stacked PR merges into a feature branch, which has no protection, so even a check marked required on main never gets a say. (Issue #282 hoped for a hard block; there isn't a way to get one without also blocking ordinary pushes to feature branches.)

What really keeps stacks safe is the **auto-delete setting**. Once PR A merges, its branch disappears within a couple of seconds and PR B is moved to main, so there's no stale branch left to merge into. The check warns you before a mistake, and the alarm tells you straight after one.

It guards against accidents, not against someone deliberately editing it: a PR runs its own copy of the check. Between the two of us, that's the point.

### Why: the F2 stack (24 September)

A stack of three PRs (#271 at the bottom, then #274, then #277 on top) was merged within 36 seconds, and not bottom-up:

- **#271** merged into main. Auto-delete was **off** then, so its branch stayed.
- **#277** merged into #274's branch while #274 was still open.
- **#274** merged into #271's branch, which had already reached main 36 seconds earlier.

GitHub showed all three as "merged", but #274 and #277 never reached main. That branch sat 23 commits ahead and had to be recovered by hand with #281. Auto-delete has been on since 25 September. With it on, #271's branch would have been deleted and #274 retargeted to main within about two seconds.

The rule is "base must be main" rather than "the PR below has merged", because the second rule would have been **green** for #274: the PR below it (#271) *had* merged.

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
