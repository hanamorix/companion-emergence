#!/usr/bin/env bash
# overlay_e2e.sh — #286 slice 2 through-path check against a BUILT runtime
# (app/build_python_runtime.sh). Installs this checkout's brain into a throwaway
# overlay, proves the real launchers pick it up, reverts, proves the bundle is
# back. Runs on macOS, Linux and Windows (Git Bash). Needs uv.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$REPO/app/src-tauri/python-runtime"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unset NELLBRAIN_HOME
case "$(uname -s)" in
  # Windows: call the bundled python.exe the way nell.bat does (Git Bash can't exec the .bat cleanly — see app/build_python_runtime.sh step 6a). What this proves is that the bundled python.exe and pythonw.exe process the overlay .pth; nell.bat itself is covered by tests/integration/test_nell_bat_wrapper.py and the build's verify step.
  MINGW*|MSYS*|CYGWIN*)
    PY="$RT/python.exe"; PYW="$RT/pythonw.exe"
    export KINDLED_HOME="$(cygpath -w "$WORK/home")"
    nell() { "$PY" -P -c "import sys; from brain.cli import main; sys.exit(main())" "$@"; };;
  *)
    PY="$RT/bin/python3"; PYW=""
    export KINDLED_HOME="$WORK/home"
    nell() { "$RT/bin/nell" "$@"; };;
esac
COMMIT=e2e0000000000000000000000000000000000000
where_brain() { "$1" -P -c "import brain, sys; sys.stdout.write(brain.__file__)"; }

cd "$REPO"
uv build --wheel --out-dir "$WORK/dist" >/dev/null
uv export --format requirements-txt --no-dev --no-emit-project --locked --quiet --output-file "$WORK/req.txt"
WHL="$(ls "$WORK"/dist/*.whl | head -n1)"
if [ "${PY%.exe}" != "$PY" ]; then WHL="$(cygpath -w "$WHL")"; REQ="$(cygpath -w "$WORK/req.txt")"; else REQ="$WORK/req.txt"; fi

cd /
nell update --status | tee "$WORK/status.json"
grep -q '"supported": true' "$WORK/status.json" || { echo "e2e: runtime has no overlay hook" >&2; exit 1; }
nell update --wheel "$WHL" --requirements "$REQ" --commit "$COMMIT"
case "$(where_brain "$PY")" in *brain-overlay*) echo "e2e: python sees the overlay brain";; *) echo "e2e: FAIL overlay not active" >&2; exit 1;; esac
"$PY" -P -c "from brain.update.overlay import loaded_overlay as l; o=l(); assert o and o['commit']=='$COMMIT' and o['bundle_match'], o; print('e2e: loaded_overlay OK')"
nell --version
if [ -n "$PYW" ]; then
  "$PYW" -P -c "import brain, pathlib; pathlib.Path(r'$(cygpath -w "$WORK")/pyw.txt').write_text(brain.__file__)"
  grep -q brain-overlay "$WORK/pyw.txt" || { echo "e2e: FAIL pythonw does not see the overlay" >&2; exit 1; }
  echo "e2e: pythonw sees the overlay brain"
fi
nell update --revert
case "$(where_brain "$PY")" in *brain-overlay*) echo "e2e: FAIL still on the overlay after revert" >&2; exit 1;; *) echo "e2e: reverted to the bundle brain";; esac
echo "e2e: PASS"
