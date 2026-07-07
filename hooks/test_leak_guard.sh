#!/usr/bin/env bash
# End-to-end leak-guard tests: C7 (five rejections) + C8 (clean pass) + C9
# (fresh-clone bootstrap). Scratch repos only — never touches a real remote.
# Uses a SYNTHETIC CANARY (B3): no real marker appears in this file or any
# test payload.
#
# Usage: bash hooks/test_leak_guard.sh

set -u

HOOKS_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CANARY="canary wombat teacup"
PASS=0
FAIL=0

say()  { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS+1)); say "  PASS: $1"; }
bad()  { FAIL=$((FAIL+1)); say "  FAIL: $1"; }

# --- scratch "public" remote (bare) + working clone wired like the real repo
REMOTE="$WORK/remote.git"
git init -q --bare "$REMOTE"

new_repo() {
  # $1 = dir. Creates a repo with the guard installed and a clean base commit
  # pushed, remote URL disguised as the public GitHub URL via pushInsteadOf.
  local dir="$1"
  git init -q "$dir"
  (
    cd "$dir"
    git config user.name "Hana Mori"
    git config user.email "214302556+hanamorix@users.noreply.github.com"
    mkdir -p hooks
    cp "$HOOKS_DIR/leak_guard.py" "$HOOKS_DIR/pre-push" hooks/
    cp "$HOOKS_DIR/allowed-identities.txt" "$HOOKS_DIR/denied-paths.txt" \
       "$HOOKS_DIR/allowed-binary-paths.txt" hooks/
    printf 'substr:%s\n' "$CANARY" > hooks/leak-rules.local
    chmod +x hooks/pre-push
    git config core.hooksPath hooks
    # Real pushes get the URL AFTER insteadOf rewriting, so a github-URL match
    # can't be exercised against a local bare remote — use LEAK_GUARD_FORCE=1
    # (the hook's test/drill override) instead.
    git remote add origin "$REMOTE"
    echo clean > base.txt
    git add -A && git commit -qm "base"
    git push -q origin main 2>/dev/null || git push -q origin master:main 2>/dev/null
  )
}

expect_reject() {
  # $1 = label. Repo at $WORK/t is one commit ahead; push must FAIL *because
  # the guard fired* (red-team m5: a crashing hook also fails the push — only
  # a LEAK-GUARD line on stderr counts as a real rejection).
  err="$WORK/push-err.txt"
  if ( cd "$WORK/t" && LEAK_GUARD_FORCE=1 git push -q origin HEAD:main ) 2>"$err"; then
    bad "$1 (push was accepted — should have been rejected)"
  elif grep -q "LEAK-GUARD" "$err"; then
    ok "$1"
  else
    bad "$1 (push failed but guard did not report — crash?)"
    sed -n '1,5p' "$err" >&2
  fi
  ( cd "$WORK/t" && git reset -q --hard origin/main 2>/dev/null || git reset -q --hard HEAD~1 )
}

say "C7 — five rejection cases (synthetic canary only)"
new_repo "$WORK/t"

# (a) canary in file content
( cd "$WORK/t" && echo "note: $CANARY here" > leak.txt && git add -A && git commit -qm "innocent message" )
expect_reject "C7a content"

# (b) canary in commit message
( cd "$WORK/t" && echo clean2 > c.txt && git add -A && git commit -qm "mentions $CANARY" )
expect_reject "C7b message"

# (c) non-allowlisted author email
( cd "$WORK/t" && echo clean3 > d.txt && git add -A \
  && git -c user.name="Hana" -c user.email="hana@nanoclaw.example" commit -qm "identity test" )
expect_reject "C7c metadata"

# (d) file under a retired private tree
( cd "$WORK/t" && mkdir -p docs/superpowers && echo x > docs/superpowers/new.md \
  && git add -A && git commit -qm "adds private-tree file" )
expect_reject "C7d path"

# (e) image outside allowed paths
( cd "$WORK/t" && printf '\x89PNG\r\n' > photo.png && git add -A && git commit -qm "adds image" )
expect_reject "C7e binary"

say "C8 — clean pass (no false positives)"
(
  cd "$WORK/t"
  echo "see https://github.com/hanamorix/companion-emergence for the project" > org.txt
  mkdir -p expressions && printf '\x89PNG\r\n' > expressions/fine.png
  git add -A && git commit -qm "clean commit with org substring + allowed image"
)
( cd "$WORK/t" && LEAK_GUARD_FORCE=1 git push -q origin HEAD:main ) 2>/dev/null \
  && ok "C8 clean push accepted" || bad "C8 clean push was rejected (false positive)"

# ToT-gmail-authored commit passes the metadata arm
(
  cd "$WORK/t" && echo tot > tot.txt && git add -A \
  && git -c user.name="ThinkerOfThoughts" -c user.email="thinkerofthoughts42@gmail.com" \
       commit -qm "contributor commit"
)
( cd "$WORK/t" && LEAK_GUARD_FORCE=1 git push -q origin HEAD:main ) 2>/dev/null \
  && ok "C8 ToT-authored push accepted" || bad "C8 ToT-authored push rejected (false positive)"

say "C9 — fresh clone + bootstrap → guard fires on first push"
git clone -q "$REMOTE" "$WORK/fresh"
(
  cd "$WORK/fresh"
  git config user.name "Hana Mori"
  git config user.email "214302556+hanamorix@users.noreply.github.com"
  # the tracked hooks/ dir came with the clone; bootstrap = the documented one-liner
  printf 'substr:%s\n' "$CANARY" > hooks/leak-rules.local
  git config core.hooksPath hooks
  echo "$CANARY" > fresh-leak.txt
  git add -A && git commit -qm "fresh clone leak attempt"
)
( cd "$WORK/fresh" && LEAK_GUARD_FORCE=1 git push -q origin HEAD:main ) 2>/dev/null \
  && bad "C9 fresh-clone push accepted (guard did not fire)" \
  || ok "C9 fresh-clone guard fired"

say "B3 — fresh init with no origin/main → fail-closed full-history scan"
(
  git init -q "$WORK/noorigin"
  cd "$WORK/noorigin"
  git config user.name "Hana Mori"
  git config user.email "214302556+hanamorix@users.noreply.github.com"
  mkdir -p hooks
  cp "$HOOKS_DIR/leak_guard.py" "$HOOKS_DIR/pre-push" hooks/
  cp "$HOOKS_DIR/allowed-identities.txt" "$HOOKS_DIR/denied-paths.txt" \
     "$HOOKS_DIR/allowed-binary-paths.txt" hooks/
  printf 'substr:%s\n' "$CANARY" > hooks/leak-rules.local
  chmod +x hooks/pre-push
  git config core.hooksPath hooks
  git remote add origin "$REMOTE"
  echo "$CANARY" > early-leak.txt
  git add -A && git commit -qm "leak buried in first commit"
  echo clean > later.txt
  git add -A && git commit -qm "clean tip"
)
( cd "$WORK/noorigin" && LEAK_GUARD_FORCE=1 git push -q origin HEAD:refs/heads/scratch ) 2>"$WORK/b3-err.txt" \
  && bad "B3 no-origin push accepted (unguarded)" \
  || { grep -q "LEAK-GUARD" "$WORK/b3-err.txt" && ok "B3 fail-closed scan fired" || bad "B3 push failed without guard report"; }

say ""
say "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
