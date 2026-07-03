# CLAUDE.md

Rules for Claude Code when working in this repo. Two-person project; the goal is to prevent us from stepping on each other's work.

## Never commit to main

Before any edit, check the current branch:

```bash
git branch --show-current
```

If it's `main`, stop and create a branch: `git checkout -b <username>/<short-description>`. Ask the user for their name if you don't know it. Never commit or push to `main` directly, even for "small" changes.

## Draft PR early, not late

After the first commit on a new branch, remind the user to push and open a **draft** PR:

```bash
git push -u origin <branch-name>
```

Then prompt: "Push done — open a draft PR now so the other dev can see you're working on this area." Don't wait until the work is finished. The draft PR is a coordination signal, not a review artifact.

## Check for collisions before starting new work

When the user says they want to start work on something new, before writing any code:

1. Ask if there's an existing issue for it, or check `gh issue list` if the GitHub CLI is available.
2. Check open PRs (`gh pr list`) to see if the other dev is already touching related files.
3. If there's overlap, flag it and suggest they coordinate in Discord before proceeding.

If an issue exists, remind the user to assign themselves before starting.

## Keep branches current

If the user has been working on a branch for more than a day or two, suggest catching up to main:

```bash
git checkout main && git pull && git checkout - && git merge main
```

Small merges are cheap; stale branches are painful. Nudge, don't nag.

## Don't create coordination files in the repo

Do not create `WIP.md`, `CLAIMS.md`, `TODO.md`, or similar files intended to track who's working on what. Draft PRs and assigned issues do that job. Adding a tracking file moves the coordination problem into the repo itself and creates its own merge conflicts.

## Commits and PRs

- Commit messages: imperative mood, present tense ("fix provider path resolution," not "fixed" or "fixes").
- If a commit or PR resolves an issue, include `Fixes #N` in the PR description so it auto-closes.
- Don't squash unrelated changes into one commit. If the user is fixing two different things, suggest two branches.

## When in doubt

If a situation isn't covered here and it's about coordination between the two devs, err on the side of surfacing it to the user rather than proceeding silently. "The other dev has an open PR touching this file — want me to keep going or check with them first?" is almost always the right move.

See `CONTRIBUTING.md` for the human-facing version of this workflow.
