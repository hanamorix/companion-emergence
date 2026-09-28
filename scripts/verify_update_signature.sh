#!/usr/bin/env bash
# Verify a `tauri signer` signature with the reference minisign (#286 slice 3).
# tauri wraps both the .sig file and the public key (tauri.conf.json's
# plugins.updater.pubkey) in one line of base64; minisign wants the plain files.
# Exit 0 only when the file's signature and its trusted comment verify.
#
# Usage: verify_update_signature.sh FILE SIG_FILE PUBKEY_BASE64
set -euo pipefail

[ $# -eq 3 ] || { echo "usage: $0 FILE SIG_FILE PUBKEY_BASE64" >&2; exit 2; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
base64 -d < "$2" > "$work/file.minisig"
printf '%s' "$3" | base64 -d > "$work/key.pub"
minisign -V -m "$1" -x "$work/file.minisig" -p "$work/key.pub"
