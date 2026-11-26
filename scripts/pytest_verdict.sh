#!/usr/bin/env bash
# pytest_verdict.sh LOG EXIT_CODE — judge a full pytest run the way CI does (#363).
#
# release_preflight.sh used to grep only '^FAILED' and accept any last line containing "passed",
# never reading pytest's exit status. An `ERROR tests/x.py::t - boom` run, whose summary still
# reads "5 passed, 1 error", came out green here and red in release.yml after the tag existed.
#
# Red when pytest's exit status is non-zero OR the log carries a FAILED/ERROR line. The exit
# status is authoritative for green: trailing interpreter-shutdown noise after the summary must
# not turn a clean run red, so this never looks at the last line.
#
# Exit 0 = green (prints the summary line). Exit 1 = red (prints the offending lines to stderr).

set -uo pipefail

log="${1:?usage: pytest_verdict.sh LOG EXIT_CODE}"
rc="${2:?usage: pytest_verdict.sh LOG EXIT_CODE}"

bad_lines="$(grep -E '^(FAILED|ERROR) ' "$log" || true)"

if [[ "$rc" != "0" || -n "$bad_lines" ]]; then
  echo "pytest exited $rc" >&2
  [[ -n "$bad_lines" ]] && printf '%s\n' "$bad_lines" >&2
  exit 1
fi

grep -E '[0-9]+ passed' "$log" | tail -1
