#!/usr/bin/env bash
# overlay_e2e.sh — #286 slice 2 through-path check against a BUILT runtime
# (app/build_python_runtime.sh). Installs this checkout's brain into a throwaway
# overlay, proves the real launchers pick it up, reverts, proves the bundle is
# back. Runs on macOS, Linux and Windows (Git Bash). Needs uv.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$REPO/app/src-tauri/python-runtime"
WORK="$(mktemp -d)"
# The in-use holder (below) exits when $WORK/stop appears; let it go before cleaning up.
trap 'touch "$WORK/stop" 2>/dev/null; wait 2>/dev/null; rm -rf "$WORK"' EXIT
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
where_brain() { "$1" -P -c "import brain, sys; sys.stdout.write(brain.__file__)"; }

cd "$REPO"
uv build --wheel --out-dir "$WORK/dist" >/dev/null
uv export --format requirements-txt --no-dev --no-emit-project --locked --quiet --output-file "$WORK/req.txt"
WHL="$(ls "$WORK"/dist/*.whl | head -n1)"
if [ "${PY%.exe}" != "$PY" ]; then WHL="$(cygpath -w "$WHL")"; REQ="$(cygpath -w "$WORK/req.txt")"; else REQ="$WORK/req.txt"; fi

cd /
nell update --status | tee "$WORK/status.json"
grep -q '"supported": true' "$WORK/status.json" || { echo "e2e: runtime has no overlay hook" >&2; exit 1; }
nell update --wheel "$WHL" --requirements "$REQ" --commit e2e0000000000000000000000000000000000000  # = $A below
case "$(where_brain "$PY")" in *brain-overlay*) echo "e2e: python sees the overlay brain";; *) echo "e2e: FAIL overlay not active" >&2; exit 1;; esac
nell --version
if [ -n "$PYW" ]; then
  "$PYW" -P -c "import brain, pathlib; pathlib.Path(r'$(cygpath -w "$WORK")/pyw.txt').write_text(brain.__file__)"
  grep -q brain-overlay "$WORK/pyw.txt" || { echo "e2e: FAIL pythonw does not see the overlay" >&2; exit 1; }
  echo "e2e: pythonw sees the overlay brain"
fi

# #304: a second install whose requirements really differ from the bundle — one small
# pure-Python pin swapped for an older release, hashed by `uv pip compile` — so pip does
# a real --require-hashes download; more installs (rotation + prune); then --rollback.
# #302/#314: A is marked in use by a live process (this script's shell, as a bridge marks
# its folder at start), so prune must keep it and re-applying A must install beside it.
A=e2e0000000000000000000000000000000000000
B=e2e1111111111111111111111111111111111111
C=e2e2222222222222222222222222222222222222
D=e2e3333333333333333333333333333333333333
OV="$WORK/home/brain-overlay"
native() { if [ "${PY%.exe}" != "$PY" ]; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
status_is() {  # <active commit> <previous commit | None>
  nell update --status | "$PY" -P -c "import json, sys
s = json.load(sys.stdin)
got = ((s['active'] or {}).get('commit'), (s['previous'] or {}).get('commit'))
assert got == (sys.argv[1], None if sys.argv[2] == 'None' else sys.argv[2]), got" "$1" "$2"
}
certifi_is() { "$PY" -P -c "import importlib.metadata as m, sys; v = m.version('certifi'); assert v == sys.argv[1], v" "$1"; }
has_dir() { ls "$OV" | grep -Eqx "$1"; }  # an overlay folder name matches the regex
status_is $A None
A_DIR="$(ls "$OV" | grep -Ex 'e2e000000000-[0-9a-f]{8}')"
# A live process running A's brain marks it in use through the real mark_in_use() (what a
# bridge does at start), then lives until $WORK/stop appears. Its own pid, not a parent's:
# under Git Bash a native program's parent is a short-lived fork, not this script's shell.
"$PY" -P -c "import pathlib, sys, time
from brain.update.overlay import mark_in_use
mark_in_use()
stop, end = pathlib.Path(sys.argv[1]), time.time() + 900
while not stop.exists() and time.time() < end:
    time.sleep(0.2)" "$(native "$WORK/stop")" &
for _ in $(seq 100); do ls "$OV/$A_DIR/.in-use/"* >/dev/null 2>&1 && break; sleep 0.2; done
ls "$OV/$A_DIR/.in-use/"* >/dev/null 2>&1 || { echo "e2e: FAIL mark_in_use wrote no marker for A" >&2; exit 1; }
CERT_BUNDLE="$(sed -n -E 's/^certifi==([^ ;\\]+).*/\1/p' "$WORK/req.txt")"
[ -n "$CERT_BUNDLE" ] || { echo "e2e: FAIL no certifi pin to swap" >&2; exit 1; }
echo "certifi<$CERT_BUNDLE" | uv pip compile - --quiet --no-header --no-annotate --generate-hashes -o "$WORK/certifi.txt"
CERT_OLD="$(sed -n -E 's/^certifi==([^ ;\\]+).*/\1/p' "$WORK/certifi.txt")"
awk '/^certifi==/{skip=1} skip{if(!/\\$/)skip=0; next} 1' "$WORK/req.txt" > "$WORK/req2.txt"
cat "$WORK/certifi.txt" >> "$WORK/req2.txt"
REQ2="$(native "$WORK/req2.txt")"
nell update --wheel "$WHL" --requirements "$REQ2" --commit $B
certifi_is "$CERT_OLD"
status_is $B $A
echo "e2e: hashed download of certifi $CERT_OLD (bundle has $CERT_BUNDLE) is live"
nell update --wheel "$WHL" --requirements "$REQ" --commit $C
status_is $C $B
[ -f "$OV/$A_DIR/stamp.json" ] || { echo "e2e: FAIL prune deleted an overlay a live process uses" >&2; exit 1; }
echo "e2e: prune keeps an unnamed overlay a live process uses"
nell update --wheel "$WHL" --requirements "$REQ" --commit $A
status_is $A $C
has_dir "$A_DIR-r1" && [ -f "$OV/$A_DIR/stamp.json" ] || { echo "e2e: FAIL re-applying A replaced its in-use folder" >&2; exit 1; }
echo "e2e: re-applying an in-use overlay's commit installs beside it"
touch "$WORK/stop"; wait  # the holder exits; its marker stays behind, now naming a dead pid
nell update --wheel "$WHL" --requirements "$REQ2" --commit $D
status_is $D $A
for gone in "$A_DIR" 'e2e111111111-[0-9a-f]{8}' 'e2e222222222-[0-9a-f]{8}'; do
  if has_dir "$gone"; then echo "e2e: FAIL prune kept $gone" >&2; exit 1; fi
done
echo "e2e: rotation keeps two, prune removed the rest once no process uses them (a dead marker doesn't count)"
nell update --rollback >/dev/null
status_is $A None
certifi_is "$CERT_BUNDLE"
echo "e2e: rollback runs the previous overlay again"
PTH="$(find "$OV" -name '*.pth' | head -n1)"
[ -z "$PTH" ] || { echo "e2e: FAIL $PTH — the hook doesn't process .pth files inside an overlay (#302)" >&2; exit 1; }
nell update --revert
case "$(where_brain "$PY")" in *brain-overlay*) echo "e2e: FAIL still on the overlay after revert" >&2; exit 1;; *) echo "e2e: reverted to the bundle brain";; esac
echo "e2e: PASS"
