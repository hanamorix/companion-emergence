#!/usr/bin/env bash
# update.sh — apply the current git state over an existing install (#179).
#
# Safety tier: 3 (live persona / mutating). Stops the supervisor, replaces
# the installed brain, restarts the supervisor. Never touches persona data.
#
# Works for both install kinds reported by `nell paths install_kind`:
#   source  — the install IS a git checkout: git pull --ff-only + uv sync.
#   bundled — a desktop-app python-runtime: build a wheel from the source
#             tree and install it (plus locked deps) into that runtime,
#             mirroring app/build_python_runtime.sh steps 3-5. Keep the two
#             in sync by hand; bash cannot share the recipe safely.
#
# Windows: not supported (the bundled runtime ships no bash) — see #255.
#
# Usage:
#   scripts/update.sh [--persona NAME] [--ref REF] [--source DIR] [--nell PATH]
#                     [--no-restart] [--dry-run] [--allow-app-rewrite]
#
#   --dry-run prints the command plan (one `plan: ...` line per command) and
#   executes nothing past preflight.
set -euo pipefail

REPO_URL="https://github.com/hanamorix/companion-emergence"
PERSONA=""; REF="main"; SOURCE=""; NELL=""; RESTART=1; DRY=0; ALLOW_APP=0
ORIG_ARGS=("$@")   # kept for the sudo re-exec; the parse loop consumes $@

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --persona) PERSONA="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --nell) NELL="$2"; shift 2;;
    --no-restart) RESTART=0; shift;;
    --dry-run) DRY=1; shift;;
    --allow-app-rewrite) ALLOW_APP=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "update.sh: unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

# ---- preflight -------------------------------------------------------------
missing=""
for tool in git uv; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done
if [ -n "$missing" ]; then
  echo "update.sh: required tool(s) not on PATH:$missing" >&2
  echo "  git: https://git-scm.com   uv: https://docs.astral.sh/uv/" >&2
  exit 2
fi

if [ -z "$NELL" ]; then
  if command -v nell >/dev/null 2>&1; then NELL="$(command -v nell)"
  elif [ -x "$HOME/.local/bin/nell" ]; then NELL="$HOME/.local/bin/nell"
  else echo "update.sh: cannot find nell; pass --nell PATH" >&2; exit 2; fi
fi

# Word-split on purpose: empty when no persona was given.
PERSONA_ARGS=()
[ -n "$PERSONA" ] && PERSONA_ARGS=(--persona "$PERSONA")

nell_path() { "$NELL" paths "$1" "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"; }
INSTALL_ROOT="$(nell_path install_root)" || { echo "update.sh: nell paths install_root failed" >&2; exit 2; }
INSTALL_KIND="$(nell_path install_kind)" || { echo "update.sh: nell paths install_kind failed" >&2; exit 2; }
case "$INSTALL_KIND" in
  source|bundled) ;;
  *) echo "update.sh: unexpected install_kind '$INSTALL_KIND' — is this nell older than #179?" >&2; exit 2;;
esac

run() {
  if [ "$DRY" = 1 ]; then echo "plan: $*"; else echo "+ $*" >&2; "$@"; fi
}

# ---- source tree -----------------------------------------------------------
TMP=""
BRAIN_STOPPED=0
on_exit() {
  status=$?
  if [ "$BRAIN_STOPPED" = 1 ] && [ "$status" -ne 0 ] && [ "$DRY" = 0 ]; then
    echo "update.sh: failed (exit $status) — restarting the brain anyway" >&2
    "$NELL" service start "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}" || true
  fi
  [ -n "$TMP" ] && rm -rf "$TMP"
  exit "$status"
}
trap on_exit EXIT

if [ "$INSTALL_KIND" = "source" ]; then
  SRC_TREE="$(cd "$INSTALL_ROOT/.." && pwd)"   # the .venv sits inside the checkout
elif [ -n "$SOURCE" ]; then
  SRC_TREE="$SOURCE"
else
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/ce-update.XXXXXX")"
  SRC_TREE="$TMP/companion-emergence"
  run git clone --depth 1 --branch "$REF" "$REPO_URL" "$SRC_TREE"
fi
if [ "$DRY" = 0 ] || [ -z "$TMP" ]; then
  [ -f "$SRC_TREE/pyproject.toml" ] || { echo "update.sh: no pyproject.toml in $SRC_TREE" >&2; exit 2; }
fi

# ---- stop -------------------------------------------------------------------
if [ "$RESTART" = 1 ]; then
  run "$NELL" service stop "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"
  BRAIN_STOPPED=1
fi

# ---- apply ------------------------------------------------------------------
if [ "$INSTALL_KIND" = "source" ]; then
  [ -z "$SOURCE" ] && run git -C "$SRC_TREE" pull --ff-only
  run sh -c "cd '$SRC_TREE' && uv sync --all-extras"
else
  PY_BIN="$INSTALL_ROOT/bin/python3"
  NELL_BIN="$INSTALL_ROOT/bin/nell"
  case "$INSTALL_ROOT" in
    *.app/*)
      if [ "$ALLOW_APP" != 1 ]; then
        echo "update.sh: $INSTALL_ROOT is inside a .app bundle; rewriting it invalidates the bundle signature (Gatekeeper may re-prompt on next launch). Re-run with --allow-app-rewrite to proceed." >&2
        exit 2
      fi;;
  esac
  if [ ! -w "$INSTALL_ROOT" ]; then
    if [ "$DRY" = 1 ]; then
      echo "plan: sudo $0 ${ORIG_ARGS[*]} --nell $NELL --source $SRC_TREE --no-restart"
    else
      echo "update.sh: $INSTALL_ROOT is not writable; re-running under sudo" >&2
      # --no-restart: this unprivileged parent already stopped the brain and restarts it below.
      sudo -E "$0" "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}" --nell "$NELL" --source "$SRC_TREE" --no-restart
      BRAIN_STOPPED=0
      [ "$RESTART" = 1 ] && run "$NELL" service start "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"
      exit 0
    fi
  fi
  run sh -c "cd '$SRC_TREE' && rm -rf dist && uv build --wheel"
  run sh -c "cd '$SRC_TREE' && uv export --format requirements-txt --no-dev --no-emit-project --locked --quiet --output-file dist/requirements.txt"
  # pip regenerates bin/nell with a baked shebang; the shipped file is a
  # relocatable wrapper (app/build_python_runtime.sh step 5). Save and restore it.
  run cp "$NELL_BIN" "$NELL_BIN.orig"
  run uv pip install --python "$PY_BIN" --require-hashes --requirements "$SRC_TREE/dist/requirements.txt" --quiet
  run sh -c "uv pip install --python '$PY_BIN' --no-deps --quiet '$SRC_TREE'/dist/*.whl"
  run mv "$NELL_BIN.orig" "$NELL_BIN"
fi

# ---- verify -----------------------------------------------------------------
if [ "$DRY" = 0 ]; then
  want="$(sed -n 's/^version = "\(.*\)"/\1/p' "$SRC_TREE/pyproject.toml" | head -1)"
  got="$("$NELL" --version | awk '{print $NF}')"
  if [ "$got" != "$want" ]; then
    echo "update.sh: version mismatch after update: installed '$got', source '$want'" >&2
    exit 1
  fi
  echo "update.sh: installed $got"
else
  echo "plan: verify nell --version == pyproject version"
fi

# ---- start ------------------------------------------------------------------
if [ "$RESTART" = 1 ]; then
  run "$NELL" service start "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"
  BRAIN_STOPPED=0
  run "$NELL" service status "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"
fi
