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
- **Closing:** if your PR fixes an issue, put `Fixes #12` in the PR description. GitHub auto-closes the issue when the PR merges.

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
- **Don't add a `WIP.md` or `CLAIMS.md` file to the repo to track who's on what.** It just moves the merge-conflict problem into the repo itself. Draft PRs already do this job.

## Quick reference

| I want to... | Do this |
|---|---|
| Start new work | Assign yourself an issue → branch → initial commit → push → open draft PR |
| Check what the other person is doing | Pull requests tab (filter: Open), Issues tab (filter: Assignee) |
| Mark work as ready | Click "Ready for review" on the draft PR |
| Catch up to main | `git checkout main && git pull && git checkout your-branch && git merge main` |
| Claim an issue | Assign yourself in the right sidebar of the issue |
| Link a PR to an issue | Put `Fixes #N` in the PR description |
