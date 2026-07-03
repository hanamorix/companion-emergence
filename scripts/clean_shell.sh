#!/usr/bin/env bash
# clean_shell.sh — run a command in a "brain-faithful" shell that behaves like a
# real user's machine, not a Claude Code agent shell.
#
# WHY: the agent shell injects env that makes normal things fail as fake
# "product bugs", each re-diagnosed from scratch:
#   * ANTHROPIC_BASE_URL points at the Tamp proxy (:7778) → the brain's nested
#     `claude -p` provider subprocess 401s (the child CLI can't auth against it).
#     A real user's brain auths via keychain/file, with no base-url override.
#   * *.workers.dev is proxied → curl/httpx to a relay get 000/empty; relay +
#     easy-connect tests need the proxy bypassed.
# This wrapper drops those overrides and bypasses the proxy so live-inference,
# relay, and easy-connect commands behave as they would for a user.
#
# NOTE: for `pnpm tauri dev` also use scripts/dev_build.sh — that additionally
# removes the STALE bundled runtime (app/src-tauri/target/debug/python-runtime)
# that shadows source and 404s every new route.
#
# Usage:
#   scripts/clean_shell.sh <cmd> [args...]   # run one command in the clean env
#   scripts/clean_shell.sh                   # open an interactive clean subshell
#   eval "$(scripts/clean_shell.sh --export)"# apply the clean env to THIS shell

_clean_env_exports() {
  cat <<'EOF'
unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY
export NO_PROXY='*' no_proxy='*'
export NELL_HARNESS_CLEAN=1
EOF
}

if [[ "${1:-}" == "--export" ]]; then
  _clean_env_exports
  exit 0
fi

# Build the clean environment for the child.
unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY
export NO_PROXY='*' no_proxy='*'
export NELL_HARNESS_CLEAN=1

if [[ "$#" -eq 0 ]]; then
  echo "[clean_shell] ANTHROPIC_BASE_URL/API_KEY unset, NO_PROXY=* — opening a clean subshell." >&2
  echo "[clean_shell] exit to return." >&2
  exec "${SHELL:-/bin/bash}"
fi

exec "$@"
