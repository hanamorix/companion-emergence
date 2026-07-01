#!/usr/bin/env bash
# One-time setup: give Nell's brain its OWN clean `claude` login, isolated from
# your interactive coding config.
#
# WHY: the `claude` CLI injects your global interactive config into every call
# the brain makes on Nell's behalf — the superpowers block, the agent-types
# list, the enabled-skills catalogue, and your whole ~/.claude/CLAUDE.md. Nell
# receives all of it every turn and sometimes narrates it ("skill-loading noise
# mid-scene"). It can't be stripped by CLI flags without breaking her tools or
# auth. The clean fix is a dedicated CLAUDE_CONFIG_DIR with its own login and no
# plugins/hooks/CLAUDE.md.
#
# This script logs `claude` in under that dedicated dir and, only if the login
# verifies, writes the `.brain-authed` marker the brain looks for. Until the
# marker exists the brain uses your normal config unchanged (nothing breaks).
#
# Run it ONCE:  bash scripts/setup_brain_claude_login.sh
# Undo it:      rm -f "<KINDLED_HOME>/claude-config/.brain-authed"
#               (the brain instantly falls back to your normal config)

set -euo pipefail

# Resolve KINDLED_HOME the same way the brain does (platformdirs on macOS).
HOME_DIR="${KINDLED_HOME:-${NELLBRAIN_HOME:-$HOME/Library/Application Support/companion-emergence}}"
BRAIN_CFG="$HOME_DIR/claude-config"
MARKER="$BRAIN_CFG/.brain-authed"

echo "Brain claude config dir: $BRAIN_CFG"
mkdir -p "$BRAIN_CFG"

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' not found on PATH. Install Claude Code and sign in normally first." >&2
  exit 1
fi

echo
echo "A browser sign-in will open. Log in with the SAME Anthropic account you"
echo "use normally — this just gives the brain its own clean copy of the login."
echo

# Interactive OAuth, scoped to the dedicated dir (never touches ~/.claude).
CLAUDE_CONFIG_DIR="$BRAIN_CFG" claude auth login

echo
echo "Verifying..."
STATUS_JSON="$(CLAUDE_CONFIG_DIR="$BRAIN_CFG" claude auth status 2>/dev/null || true)"

if printf '%s' "$STATUS_JSON" | grep -q '"loggedIn": *true'; then
  printf 'ok' > "$MARKER"
  echo "✅ Brain login verified. Marker written: $MARKER"
  echo "   Restart the bridge (or the app) so the daemon picks it up."
else
  rm -f "$MARKER"
  echo "❌ Login did not verify as logged-in. Marker NOT written; the brain keeps" >&2
  echo "   using your normal config (nothing broke). Re-run to try again." >&2
  echo "   auth status was: $STATUS_JSON" >&2
  exit 1
fi
