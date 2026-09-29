#!/usr/bin/env bash
# stack_guard.sh — #282: stacked pull requests merge bottom-up, each into main.
#
# MODE=check (default), on every open PR: fail (a red check) when the PR's base
#   isn't main, naming the PR below and what to do. The base is read LIVE from the
#   API, because a re-run replays the old event, whose payload still names the old
#   base; the payload is only the fallback.
# MODE=merged, after a PR merged into anything but main: comment on it. Its changes
#   are not on main; if the base branch's own PR already merged or closed (or there
#   is none), they are stranded — this is what happened in the F2 stack (#281).
#
# ADVISORY. A required check can't block a stacked merge: branch protection applies
# to a PR's base branch, and a stacked PR's base is an unprotected feature branch.
# The real guard is delete-branch-on-merge (on since 2026-09-25, off during F2):
# GitHub deletes a merged PR's branch and retargets the next PR to main. This script
# is the signal before a mistake (check) and the alarm right after one (merged).
#
# Why "base must be main" and not "the PR below has merged" (#282's first sketch):
# in F2, #274 was merged into #271's branch 36 s AFTER #271 had reached main — the
# sketch's rule would have been green for it.
#
# The pass/fail decision never depends on the API: the lookups only improve the
# message, so a failed lookup still fails. Guards against accidents between trusted
# collaborators, not a malicious PR (a `pull_request` check runs the PR's own copy).
#
# Env: MODE, BASE_REF, DEFAULT_BRANCH (required); REPO (owner/name), PR_NUMBER.
# Needs `gh` with GH_TOKEN (pull-requests: read; write for the merged-mode comment).
set -uo pipefail
: "${BASE_REF:?BASE_REF is required}" "${DEFAULT_BRANCH:?DEFAULT_BRANCH is required}"
MODE="${MODE:-check}"
case "$MODE" in
  check|merged) ;;
  *) echo "::error title=stack guard::unknown MODE '$MODE' (expected check or merged)"; exit 2 ;;
esac
repo_args=()
[ -n "${REPO:-}" ] && repo_args=(--repo "$REPO")

# Workflow-command escaping: % first, then CR and LF.
annotate() {  # annotate <error|warning> <message>
  local msg="$2"
  msg="${msg//%/%25}"
  msg="${msg//$'\r'/%0D}"
  msg="${msg//$'\n'/%0A}"
  echo "::$1 title=stack guard::$msg"
}

# "<number> <STATE> <mergedAt|-> <baseRefName>" of the PR whose head is branch $1
# (an open one first, when a branch name has been reused), or "" when there is none
# or the lookup fails.
pr_for_branch() {
  gh pr list ${repo_args[@]+"${repo_args[@]}"} --head "$1" --state all --limit 20 \
    --json number,state,mergedAt,baseRefName \
    --jq 'sort_by(.state != "OPEN") | .[0] // empty
          | "\(.number) \(.state) \(.mergedAt // "-") \(.baseRefName // "-")"' 2>/dev/null || true
}

if [ "$MODE" = "merged" ]; then
  [ "$BASE_REF" = "$DEFAULT_BRANCH" ] && exit 0
  read -r num state b_merged_at b_base <<< "$(pr_for_branch "$BASE_REF")"
  recover="To get them there, open a pull request from \`$BASE_REF\` to \`$DEFAULT_BRANCH\` (or cherry-pick this PR's commits onto a branch from \`$DEFAULT_BRANCH\`)."
  if [ "$state" = MERGED ] && [ -n "${MERGED_AT:-}" ] && [ "$b_merged_at" != "-" ] \
     && [[ ! "$b_merged_at" < "$MERGED_AT" ]]; then
    # This runs seconds after the merge. The base's PR has merged since — at or after
    # this one (ISO-8601 UTC compares as text) — so it took these changes with it.
    [ "$b_base" = "$DEFAULT_BRANCH" ] && exit 0
    text="⚠️ **stack guard:** this PR was merged into \`$BASE_REF\`, and that branch's PR #$num then merged into \`$b_base\` — not \`$DEFAULT_BRANCH\`. These changes reach $DEFAULT_BRANCH only when \`$b_base\` does; check that it does."
    rc=0; level=warning
  else
    case "$state" in
      OPEN)
        text="⚠️ **stack guard:** this PR was merged into \`$BASE_REF\`, not \`$DEFAULT_BRANCH\`. Its changes will reach $DEFAULT_BRANCH only when #$num merges — merge #$num next, and don't close it or delete \`$BASE_REF\` first."
        rc=0; level=warning ;;
      MERGED)
        text="🚨 **stack guard: these changes are not on $DEFAULT_BRANCH.** This PR was merged into \`$BASE_REF\`, whose own PR #$num had already merged — so nothing will carry them to $DEFAULT_BRANCH. $recover"
        rc=1; level=error ;;
      CLOSED)
        text="🚨 **stack guard: these changes are not on $DEFAULT_BRANCH.** This PR was merged into \`$BASE_REF\`, whose PR #$num was closed without merging. $recover"
        rc=1; level=error ;;
      *)
        text="🚨 **stack guard: these changes are not on $DEFAULT_BRANCH.** This PR was merged into \`$BASE_REF\`, which has no pull request into $DEFAULT_BRANCH. $recover"
        rc=1; level=error ;;
    esac
  fi
  if [ -n "${PR_NUMBER:-}" ]; then
    marker="<!-- stack-guard -->"
    # A re-run of this (red) job must not post the same comment again.
    posted="$(gh pr view "$PR_NUMBER" ${repo_args[@]+"${repo_args[@]}"} --json comments \
      --jq "[.comments[].body | select(contains(\"$marker\"))] | length" 2>/dev/null || echo 0)"
    if [ "${posted:-0}" = "0" ]; then
      printf '%s\n%s\n' "$marker" "$text" | gh pr comment "$PR_NUMBER" ${repo_args[@]+"${repo_args[@]}"} --body-file - >/dev/null 2>&1 \
        || annotate warning "could not comment on #$PR_NUMBER (a fork PR's token is read-only)"
    fi
  fi
  annotate "$level" "${text//\`/}"
  exit "$rc"
fi

# MODE=check (default), on every open PR: fail (a red check) when the PR's base
#   isn't main, naming the PR below and what to do. The base is read LIVE from the
#   API, because a re-run replays the old event, whose payload still names the old
#   base; the payload is only the fallback.
# MODE=merged, after a PR merged into anything but main: comment on it. Its changes
#   are not on main; if the base branch's own PR already merged or closed (or there
#   is none), they are stranded — this is what happened in the F2 stack (#281).
#
# ADVISORY. A required check can't block a stacked merge: branch protection applies
# to a PR's base branch, and a stacked PR's base is an unprotected feature branch.
# The real guard is delete-branch-on-merge (on since 2026-09-25, off during F2):
# GitHub deletes a merged PR's branch and retargets the next PR to main. This script
# is the signal before a mistake (check) and the alarm right after one (merged).
#
# Why "base must be main" and not "the PR below has merged" (#282's first sketch):
# in F2, #274 was merged into #271's branch 36 s AFTER #271 had reached main — the
# sketch's rule would have been green for it.
#
# The pass/fail decision never depends on the API: the lookups only improve the
# message, so a failed lookup still fails. Guards against accidents between trusted
# collaborators, not a malicious PR (a `pull_request` check runs the PR's own copy).
#
# Env: MODE, BASE_REF, DEFAULT_BRANCH (required); REPO (owner/name), PR_NUMBER.
# Needs `gh` with GH_TOKEN (pull-requests: read; write for the merged-mode comment).
set -uo pipefail
: "${BASE_REF:?BASE_REF is required}" "${DEFAULT_BRANCH:?DEFAULT_BRANCH is required}"
MODE="${MODE:-check}"
case "$MODE" in
  check|merged) ;;
  *) echo "::error title=stack guard::unknown MODE '$MODE' (expected check or merged)"; exit 2 ;;
esac
repo_args=()
[ -n "${REPO:-}" ] && repo_args=(--repo "$REPO")

# Workflow-command escaping: % first, then CR and LF.
annotate() {  # annotate <error|warning> <message>
  local msg="$2"
  msg="${msg//%/%25}"
  msg="${msg//$'\r'/%0D}"
  msg="${msg//$'\n'/%0A}"
  echo "::$1 title=stack guard::$msg"
}

# "<number> <STATE> <mergedAt|-> <baseRefName>" of the PR whose head is branch $1
# (an open one first, when a branch name has been reused), or "" when there is none
# or the lookup fails.
pr_for_branch() {
  gh pr list ${repo_args[@]+"${repo_args[@]}"} --head "$1" --state all --limit 20 \
    --json number,state,mergedAt,baseRefName \
    --jq 'sort_by(.state != "OPEN") | .[0] // empty
          | "\(.number) \(.state) \(.mergedAt // "-") \(.baseRefName // "-")"' 2>/dev/null || true
}

if [ "$MODE" = "merged" ]; then
  [ "$BASE_REF" = "$DEFAULT_BRANCH" ] && exit 0
  below="$(pr_for_branch "$BASE_REF")"
  num="${below%% *}"
  state="${below#* }"
  recover="To get them there, open a pull request from \`$BASE_REF\` to \`$DEFAULT_BRANCH\` (or cherry-pick this PR's commits onto a branch from \`$DEFAULT_BRANCH\`)."
  case "$state" in
    OPEN)
      body="⚠️ **stack guard:** this PR was merged into \`$BASE_REF\`, not \`$DEFAULT_BRANCH\`. Its changes will reach $DEFAULT_BRANCH only when #$num merges — merge #$num next, and don't close it or delete \`$BASE_REF\` first."
      rc=0; level=warning ;;
    MERGED)
      body="🚨 **stack guard: these changes are not on $DEFAULT_BRANCH.** This PR was merged into \`$BASE_REF\`, whose own PR #$num had already merged — so nothing will carry them to $DEFAULT_BRANCH. $recover"
      rc=1; level=error ;;
    CLOSED)
      body="🚨 **stack guard: these changes are not on $DEFAULT_BRANCH.** This PR was merged into \`$BASE_REF\`, whose PR #$num was closed without merging. $recover"
      rc=1; level=error ;;
    *)
      body="🚨 **stack guard: these changes are not on $DEFAULT_BRANCH.** This PR was merged into \`$BASE_REF\`, which has no pull request into $DEFAULT_BRANCH. $recover"
      rc=1; level=error ;;
  esac
  if [ -n "${PR_NUMBER:-}" ]; then
    printf '%s\n' "$body" | gh pr comment "$PR_NUMBER" ${repo_args[@]+"${repo_args[@]}"} --body-file - >/dev/null 2>&1 \
      || annotate warning "could not comment on #$PR_NUMBER (a fork PR's token is read-only)"
  fi
  annotate "$level" "${body//\`/}"
  exit "$rc"
fi

# MODE=check
base="$BASE_REF"
if [ -n "${PR_NUMBER:-}" ] && [ -n "${REPO:-}" ]; then
  live="$(gh api "repos/$REPO/pulls/$PR_NUMBER" --jq .base.ref 2>/dev/null || true)"
  [ -n "$live" ] && base="$live"
fi

if [ "$base" = "$DEFAULT_BRANCH" ]; then
  echo "stack guard: this PR targets $DEFAULT_BRANCH — ok"
  exit 0
fi

read -r num state _ _ <<< "$(pr_for_branch "$base")"
case "$state" in
  OPEN)
    msg="This PR is stacked on #$num ($base), which hasn't merged yet. Merge #$num into $DEFAULT_BRANCH first; GitHub then retargets this PR to $DEFAULT_BRANCH and this check re-runs green." ;;
  MERGED)
    msg="#$num ($base) has already merged, but this PR still targets its branch — merging it now would leave these changes off $DEFAULT_BRANCH. Retarget this PR to $DEFAULT_BRANCH (Edit, next to the title) before merging." ;;
  CLOSED)
    msg="This PR targets $base, whose PR #$num was closed without merging. Retarget this PR to $DEFAULT_BRANCH before merging." ;;
  *)
    msg="This PR targets $base, not $DEFAULT_BRANCH. Retarget this PR to $DEFAULT_BRANCH before merging." ;;
esac
annotate error "$msg"
exit 1
