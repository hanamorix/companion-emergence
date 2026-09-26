#!/usr/bin/env bash
# update.sh — apply the current git state over an existing install (#179).
#
# Safety tier: 3 (live persona / mutating). Stops the supervisor, replaces
# the installed brain, restarts the supervisor. Never touches persona data.
#
# Works for both install kinds reported by `nell paths install_kind` (or, on a
# nell older than that key, by the python3 beside it — #285):
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
#   --persona is required unless --no-restart (`nell supervisor` needs it).
#   --dry-run prints the command plan (one `plan: ...` line per command) and
#   executes nothing past preflight.
set -euo pipefail
unset CDPATH   # an exported CDPATH would redirect (and echo) the relative cd's below
umask 022      # runtime files must stay user-readable; sudo unions a hardened umask in

REPO_URL="https://github.com/hanamorix/companion-emergence"
PERSONA=""; REF="main"; SOURCE=""; NELL=""; RESTART=1; DRY=0; ALLOW_APP=0

usage() { sed -n '2,/^set /{/^#/p;}' "$0" | sed 's/^# \{0,1\}//'; }   # the header, however long

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
if [ "$RESTART" = 1 ] && [ -z "$PERSONA" ]; then
  echo "update.sh: --persona NAME is required to stop and restart the brain (or pass --no-restart)" >&2
  exit 2
fi
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

# Absolute paths, then leave the caller's cwd. The bundled nell runs `python3 -c`,
# which puts the cwd on sys.path: from a checkout (as the README runs this) it
# imports the checkout's brain/ and misreports install_kind and --version (#285).
abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
case "$NELL" in
  */*) ;;
  *) command -v "$NELL" >/dev/null 2>&1 || { echo "update.sh: '$NELL' is not on PATH" >&2; exit 2; }
     NELL="$(command -v "$NELL")";;   # a bare --nell name means "look it up", as before
esac
NELL="$(abspath "$NELL")"
[ -z "$SOURCE" ] || SOURCE="$(cd "$SOURCE" && pwd)"
cd /

# Word-split on purpose: empty when no persona was given.
PERSONA_ARGS=()
[ -n "$PERSONA" ] && PERSONA_ARGS=(--persona "$PERSONA")

nell_path() { "$NELL" paths "$1" "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"; }

# Ask the python that lives beside nell (both install kinds put python3 in the
# same bin/) for the same two facts brain/cli.py derives: sys.prefix, and
# whether brain/ sits in a checkout. Follows symlinks like the bundled wrapper.
probe_install() {
  local src="$NELL" dir
  while [ -L "$src" ]; do
    dir="$(cd "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    case "$src" in /*) ;; *) src="$dir/$src";; esac
  done
  "$(cd "$(dirname "$src")" && pwd)/python3" -c '
import pathlib, sys, brain
pkg = pathlib.Path(brain.__file__).resolve().parent
print(sys.prefix)
print("source" if (pkg.parent / "pyproject.toml").exists() else "bundled")'
}

# The install keys only exist from #179 on; older nells (every release up to
# v0.0.42) reject them, so fall back to the probe (#285).
if ! { INSTALL_ROOT="$(nell_path install_root 2>/dev/null)" && INSTALL_KIND="$(nell_path install_kind 2>/dev/null)"; }; then
  probe="$(probe_install)" || { echo "update.sh: cannot locate the install behind $NELL" >&2; exit 2; }
  { read -r INSTALL_ROOT; read -r INSTALL_KIND; } <<<"$probe"
fi
case "$INSTALL_KIND" in
  source|bundled) ;;
  *) echo "update.sh: unexpected install_kind '$INSTALL_KIND' — is this nell older than #179?" >&2; exit 2;;
esac
# `supervisor stop` exits 0 ("not running") for a persona with no state, so a typo
# would update the runtime under the live bridge. persona_dir predates #179.
if [ "$RESTART" = 1 ] && [ ! -d "$(nell_path persona_dir)" ]; then
  echo "update.sh: no persona '$PERSONA' (see \`nell personas\`)" >&2
  exit 2
fi

run() {
  if [ "$DRY" = 1 ]; then echo "plan: $*"; else echo "+ $*" >&2; "$@"; fi
}
# Exit 2 = already running: the app relaunched the bridge mid-update (app start or
# its Restart button), so it may be on old or mixed code. Restart it onto the new.
start_brain() {
  local rc=0
  run "$NELL" supervisor start "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}" || rc=$?
  [ "$rc" = 2 ] || return "$rc"
  # restart's own start can exit 2 too: the app is starting it right now, after the
  # install finished, so that bridge is on the new code as well.
  run "$NELL" supervisor restart "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}" || [ $? = 2 ]
}

# ---- source tree -----------------------------------------------------------
TMP=""
BRAIN_STOPPED=0
on_exit() {
  status=$?
  if [ "$BRAIN_STOPPED" = 1 ] && [ "$status" -ne 0 ] && [ "$DRY" = 0 ]; then
    echo "update.sh: failed (exit $status) — restarting the brain anyway" >&2
    "$NELL" supervisor start "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}" || true
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
src_version() { sed -n 's/^version = "\(.*\)"/\1/p' "$SRC_TREE/pyproject.toml" | head -1; }
if [ "$DRY" = 0 ] || [ -z "$TMP" ]; then
  [ -f "$SRC_TREE/pyproject.toml" ] || { echo "update.sh: no pyproject.toml in $SRC_TREE" >&2; exit 2; }
  # Refuse now, before stopping anything, rather than fail the verify step after.
  [ -n "$(src_version)" ] || { echo "update.sh: no version = \"...\" line in $SRC_TREE/pyproject.toml" >&2; exit 2; }
else
  echo "plan: check $SRC_TREE/pyproject.toml exists and has a version"   # a dry run clones nothing to check
fi

# ---- privileges (bundled) -------------------------------------------------------
UV="$(command -v uv)"
AS_ROOT=()
if [ "$INSTALL_KIND" = "bundled" ]; then
  case "$INSTALL_ROOT" in
    *.app/*)
      if [ "$ALLOW_APP" != 1 ]; then
        echo "update.sh: $INSTALL_ROOT is inside a .app bundle; rewriting it invalidates the bundle signature (Gatekeeper may re-prompt on next launch). Re-run with --allow-app-rewrite to proceed." >&2
        exit 2
      fi;;
    */.mount_*/*)
      echo "update.sh: $INSTALL_ROOT is inside a running AppImage (a read-only mount); download the new AppImage instead." >&2
      exit 2;;
  esac
  if [ ! -w "$INSTALL_ROOT" ]; then
    # Only the writes into the runtime run as root (build + export stay the user's).
    # Absolute uv: sudo's secure_path drops ~/.local/bin, uv's default home. -H:
    # root's own HOME, so uv's cache never leaves root-owned files in the user's.
    # sudo's env_reset drops the proxy/CA/UV_* settings the unprivileged steps just
    # used; hand them to root explicitly through `env`.
    ROOT_ENV=()
    for k in $(compgen -e); do
      case "$k" in
        HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|http_proxy|https_proxy|all_proxy|no_proxy|SSL_CERT_FILE|SSL_CERT_DIR|UV_*)
          ROOT_ENV+=("$k=${!k}");;
      esac
    done
    AS_ROOT=(sudo -H env "${ROOT_ENV[@]+"${ROOT_ENV[@]}"}")
    echo "update.sh: $INSTALL_ROOT is not writable; the install steps run under sudo" >&2
    # Ask for the password now, before anything is stopped.
    run sudo -v || { echo "update.sh: sudo failed; the update was NOT applied" >&2; exit 1; }
  fi
fi

# ---- stop -------------------------------------------------------------------
if [ "$RESTART" = 1 ]; then
  BRAIN_STOPPED=1   # before the call: a stop that times out (exit 1) may still take the bridge down
  run "$NELL" supervisor stop "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"
fi

# ---- apply ------------------------------------------------------------------
if [ "$INSTALL_KIND" = "source" ]; then
  [ -z "$SOURCE" ] && run git -C "$SRC_TREE" pull --ff-only
  run sh -c "cd '$SRC_TREE' && uv sync --all-extras"
else
  PY_BIN="$INSTALL_ROOT/bin/python3"
  NELL_BIN="$INSTALL_ROOT/bin/nell"
  run sh -c "cd '$SRC_TREE' && rm -rf dist && uv build --wheel"
  run sh -c "cd '$SRC_TREE' && uv export --format requirements-txt --no-dev --no-emit-project --locked --quiet --output-file dist/requirements.txt"
  # pip regenerates bin/nell with a baked shebang; the shipped file is a
  # relocatable wrapper (app/build_python_runtime.sh step 5). Save and restore it.
  run "${AS_ROOT[@]+"${AS_ROOT[@]}"}" cp "$NELL_BIN" "$NELL_BIN.orig"
  # Index flags: see app/build_python_runtime.sh step 4 (#287). Not --emit-index-url:
  # that needs uv >= 0.12, and this runs on the user's uv.
  run "${AS_ROOT[@]+"${AS_ROOT[@]}"}" "$UV" pip install --python "$PY_BIN" --require-hashes --requirements "$SRC_TREE/dist/requirements.txt" --quiet \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
  run "${AS_ROOT[@]+"${AS_ROOT[@]}"}" sh -c "'$UV' pip install --python '$PY_BIN' --no-deps --quiet '$SRC_TREE'/dist/*.whl"
  run "${AS_ROOT[@]+"${AS_ROOT[@]}"}" mv "$NELL_BIN.orig" "$NELL_BIN"
fi

# ---- verify -----------------------------------------------------------------
if [ "$DRY" = 0 ]; then
  want="$(src_version)"   # re-read: a source-kind pull may have changed it
  [ -n "$want" ] || { echo "update.sh: updated, but no version line left in $SRC_TREE/pyproject.toml to verify against" >&2; exit 1; }
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
  start_brain
  BRAIN_STOPPED=0
  run "$NELL" supervisor status "${PERSONA_ARGS[@]+"${PERSONA_ARGS[@]}"}"
fi
