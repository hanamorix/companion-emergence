#!/usr/bin/env bash
# release_preflight.sh — reproduce the release CI's verify-version + validate
# gate LOCALLY, before the tag is cut.
#
# WHY: the CI `validate` job (release.yml) runs uv sync --locked, pytest, ruff,
# pnpm test, pnpm build (tsc+vite), cargo — but only AFTER the tag ref exists.
# Any of five cheap-to-check faults (six-file version drift, a pre-release
# identifier in a version file, a dirty lockfile, a tsc error vitest doesn't
# catch, a pytest failure hidden by a truncated tail) forces a
# delete-tag → re-tag → re-sync round-trip. This runs the identical checks
# up front so those become a pre-tag red/green.
#
# Usage:  bash scripts/release_preflight.sh <tag>     e.g. v0.0.42
#         bash scripts/release_preflight.sh           (skips the tag==files check)
#
# Exits non-zero on any failure. On success prints the exact
# `gh release edit` command to run AFTER the release workflow (CI's
# generate_release_notes overwrites the curated changelog every time).

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${1:-}"
FAIL=0
LOG_DIR="$(mktemp -d)"
step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  ✅ %s\n' "$1"; }
bad()  { printf '  ❌ %s\n' "$1" >&2; FAIL=1; }

# ---------------------------------------------------------------------------
# 1. Six-file version pin (mirrors release.yml verify-version) + MSI guard
# ---------------------------------------------------------------------------
step "version pin (six files agree; no pre-release identifier in files)"
v_py=$(grep -E '^version = "[0-9]' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
v_cargo=$(grep -E '^version = "[0-9]' app/src-tauri/Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
v_tauri=$(grep '"version"' app/src-tauri/tauri.conf.json | head -1 | sed 's/.*"\([0-9][^"]*\)".*/\1/')
v_pkg=$(grep '"version"' app/package.json | head -1 | sed 's/.*"\([0-9][^"]*\)".*/\1/')
v_lock_uv=$(awk '/name = "companion-emergence"/ {getline; print}' uv.lock | sed 's/.*"\(.*\)".*/\1/')
v_lock_cargo=$(awk '/^name = "nellface"$/ {getline; print}' app/src-tauri/Cargo.lock | sed 's/.*"\(.*\)".*/\1/')
printf '  pyproject=%s cargo=%s tauri=%s pkg=%s uv.lock=%s cargo.lock=%s\n' \
  "$v_py" "$v_cargo" "$v_tauri" "$v_pkg" "$v_lock_uv" "$v_lock_cargo"

for lv in "Cargo.toml:$v_cargo" "tauri.conf.json:$v_tauri" "package.json:$v_pkg" \
          "uv.lock:$v_lock_uv" "Cargo.lock:$v_lock_cargo"; do
  [[ "${lv#*:}" == "$v_py" ]] || bad "${lv%%:*} (${lv#*:}) disagrees with pyproject.toml ($v_py)"
done
# MSI rejects pre-release identifiers: every version FILE must be plain X.Y.Z
# (the git TAG may carry -alpha.N, but the six files stay at the base minor).
for lv in "pyproject:$v_py" "Cargo.toml:$v_cargo" "tauri.conf.json:$v_tauri" \
          "package.json:$v_pkg" "Cargo.lock:$v_lock_cargo"; do
  [[ "${lv#*:}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    bad "${lv%%:*} (${lv#*:}) has a pre-release identifier — MSI target will reject it"
done
if [[ -n "$TAG" ]]; then
  tag_base="${TAG#v}"; tag_base="${tag_base%%-*}"
  norm() { echo "$1" | sed 's/[-.]//g; s/alpha/a/g; s/beta/b/g'; }
  [[ "$(norm "$tag_base")" == "$(norm "$v_py")" ]] || \
    bad "tag $TAG (base $tag_base) does not match pyproject.toml ($v_py)"
fi
[[ "$FAIL" -eq 0 ]] && ok "version pin consistent"

# ---------------------------------------------------------------------------
# 2. Lockfile clean (CI runs uv sync --locked; a stale uv.lock fails it)
# ---------------------------------------------------------------------------
step "uv.lock up to date (uv lock must not dirty the tree)"
before=$(git status --porcelain uv.lock)
uv lock >/dev/null 2>&1 || bad "uv lock failed"
after=$(git status --porcelain uv.lock)
if [[ "$before" == "$after" ]]; then ok "uv.lock unchanged"; else
  bad "uv lock dirtied uv.lock — stage the diff before tagging (CI uv sync --locked will fail)"
fi

step "uv sync --all-extras --locked (strict, same as CI)"
if uv sync --all-extras --locked >"$LOG_DIR/sync.log" 2>&1; then ok "locked sync clean"
else bad "uv sync --locked failed (see $LOG_DIR/sync.log)"; fi

# ---------------------------------------------------------------------------
# 3. Backend: full pytest (grep FAILED — never judge from a truncated tail)
# ---------------------------------------------------------------------------
step "backend pytest (full run, checked by grep '^FAILED')"
uv run pytest -q -p no:randomly >"$LOG_DIR/pytest.log" 2>&1
if grep -q '^FAILED' "$LOG_DIR/pytest.log"; then
  bad "pytest failures:"; grep '^FAILED' "$LOG_DIR/pytest.log" | sed 's/^/     /' >&2
else
  tail -1 "$LOG_DIR/pytest.log" | grep -q "passed" && ok "$(tail -1 "$LOG_DIR/pytest.log")" \
    || bad "pytest produced no pass summary (see $LOG_DIR/pytest.log)"
fi

step "ruff (uv run ruff check .)"
if uv run ruff check . >"$LOG_DIR/ruff.log" 2>&1; then ok "ruff clean"
else bad "ruff findings (see $LOG_DIR/ruff.log)"; fi

# ---------------------------------------------------------------------------
# 4. Frontend: pnpm test + pnpm build (tsc catches what vitest can't)
# ---------------------------------------------------------------------------
step "frontend pnpm test + pnpm build (tsc && vite)"
( cd app && pnpm test >"$LOG_DIR/vitest.log" 2>&1 ) && ok "vitest clean" \
  || bad "vitest failures (see $LOG_DIR/vitest.log)"
if ( cd app && pnpm build >"$LOG_DIR/build.log" 2>&1 ); then ok "tsc+vite build clean"
else
  # Agent-shell fallback: approve-builds can block `pnpm build`; try .bin directly.
  if ( cd app && ./node_modules/.bin/tsc --noEmit && ./node_modules/.bin/vite build ) \
       >"$LOG_DIR/build2.log" 2>&1; then ok "tsc+vite build clean (.bin bypass)"
  else bad "pnpm build failed — tsc error will fail CI validate AFTER the tag (see $LOG_DIR/build.log)"; fi
fi

# ---------------------------------------------------------------------------
# 5. Rust: cargo check + test (CI validate runs both)
# ---------------------------------------------------------------------------
step "cargo check + test"
( cd app/src-tauri && cargo check >"$LOG_DIR/cargo_check.log" 2>&1 ) && ok "cargo check clean" \
  || bad "cargo check failed (see $LOG_DIR/cargo_check.log)"
( cd app/src-tauri && cargo test >"$LOG_DIR/cargo_test.log" 2>&1 ) && ok "cargo test clean" \
  || bad "cargo test failed (see $LOG_DIR/cargo_test.log)"

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo
if [[ "$FAIL" -ne 0 ]]; then
  echo "❌ PREFLIGHT FAILED — do NOT cut the tag. Logs in $LOG_DIR" >&2
  exit 1
fi
echo "✅ PREFLIGHT GREEN — safe to tag${TAG:+ $TAG}."
echo
echo "After the release workflow publishes the GitHub Release, CI's"
echo "generate_release_notes OVERWRITES the curated changelog. Restore it with:"
echo "    gh release edit ${TAG:-<tag>} --notes-file .public-sync/changelog-public.md"
rm -rf "$LOG_DIR"
