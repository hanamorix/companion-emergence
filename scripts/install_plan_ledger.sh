#!/usr/bin/env bash
# Install (or verify) the `plan-ledger` skill for THIS project — tier 1.
#
# Copies the skill's router, methodology and stage prompts from
# ThinkerOfThoughts/claude-code-skills (pinned commit below) into the
# project-scoped skills dir `.claude/skills/plan-ledger/`, which is
# gitignored: every checkout runs this once. The per-project config the
# skill reads is tracked at the repo root (`plan-ledger.companion.md`);
# the upstream example config is NOT copied because it targets the other
# dev's machine.
#
#   bash scripts/install_plan_ledger.sh          # install / refresh
#   bash scripts/install_plan_ledger.sh --check  # the skill's own self-check:
#                                                # installed copy == upstream copy
#
# Needs git + network for the shallow clone. No persona, no LLM, no docs
# mutation outside `.claude/skills/`.
set -euo pipefail

UPSTREAM_URL="https://github.com/ThinkerOfThoughts/claude-code-skills"
UPSTREAM_REF="89bc8fd31b5f6aeefda4b059af460731dc60a849"   # 2026-09 Plan_ledger
UPSTREAM_DIR="Plan_ledger"
SKILL_NAME="plan-ledger"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/.claude/skills/$SKILL_NAME"
MODE="${1:-install}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[plan-ledger] fetching $UPSTREAM_URL @ ${UPSTREAM_REF:0:12}"
git -C "$WORK" init -q
git -C "$WORK" remote add origin "$UPSTREAM_URL"
git -C "$WORK" fetch -q --depth 1 origin "$UPSTREAM_REF"
git -C "$WORK" checkout -q FETCH_HEAD -- "$UPSTREAM_DIR"

SRC="$WORK/$UPSTREAM_DIR"
for required in SKILL.md METHODOLOGY.md README.md stages; do
  [ -e "$SRC/$required" ] || { echo "[plan-ledger] ERROR: upstream is missing $required" >&2; exit 1; }
done

sync_tree() {
  # Everything the skill needs at runtime; the upstream example config is
  # left out on purpose (its paths belong to the other dev's machine).
  mkdir -p "$1/stages"
  cp "$SRC/SKILL.md" "$SRC/METHODOLOGY.md" "$SRC/README.md" "$1/"
  cp "$SRC"/stages/*.md "$1/stages/"
}

case "$MODE" in
  install)
    sync_tree "$DEST"
    echo "[plan-ledger] installed to $DEST"
    echo "[plan-ledger] config: $REPO_ROOT/plan-ledger.companion.md"
    echo "[plan-ledger] invoke with /plan-ledger at the start of a planning conversation"
    ;;
  --check)
    [ -d "$DEST" ] || { echo "[plan-ledger] not installed; run without --check" >&2; exit 1; }
    EXPECT="$WORK/expected"
    sync_tree "$EXPECT"
    if diff -r "$EXPECT" "$DEST"; then
      echo "[plan-ledger] OK: installed copy == upstream copy @ ${UPSTREAM_REF:0:12}"
    else
      echo "[plan-ledger] DRIFT: installed copy differs from upstream (see diff above)" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [--check]" >&2
    exit 2
    ;;
esac
