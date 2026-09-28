"""`scripts/verify_update_signature.sh` unwraps tauri's base64 files for minisign (#286 slice 3).

`tauri signer` wraps both the .sig file and the public key in one line of base64;
the reference minisign wants the plain files. The real signature check runs in the
brain-main workflow's pull_request job (valid + tampered); this pins the unwrap.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "verify_update_signature.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash + a PATH stub")

SIG = "untrusted comment: signature from tauri secret key\nRUS...sig\ntrusted comment: t\nGLOBAL\n"
PUB = "untrusted comment: minisign public key: ABC\nRWS...pub\n"


def _run(tmp_path, *args, status=0):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    rec = tmp_path / "rec"
    (stub / "minisign").write_text(
        "#!/usr/bin/env bash\n"
        f'rec="{rec}"; mkdir -p "$rec"; printf "%s\\n" "$@" > "$rec/argv"\n'
        'while [ $# -gt 0 ]; do case "$1" in\n'
        '  -x) cp "$2" "$rec/sig"; shift;; -p) cp "$2" "$rec/pub"; shift;; esac; shift; done\n'
        f"exit {status}\n", encoding="utf-8")
    (stub / "minisign").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)
    return r, rec


def test_hands_minisign_the_unwrapped_signature_and_key(tmp_path):
    f = tmp_path / "manifest.json"
    f.write_text("{}", encoding="utf-8")
    sig = tmp_path / "manifest.json.sig"
    sig.write_text(base64.b64encode(SIG.encode()).decode(), encoding="utf-8")
    r, rec = _run(tmp_path, str(f), str(sig), base64.b64encode(PUB.encode()).decode())
    assert r.returncode == 0, r.stderr
    assert (rec / "sig").read_text(encoding="utf-8") == SIG
    assert (rec / "pub").read_text(encoding="utf-8") == PUB
    argv = (rec / "argv").read_text(encoding="utf-8").split("\n")
    assert argv[0] == "-V" and argv[argv.index("-m") + 1] == str(f)


def test_a_failed_verification_fails_the_script(tmp_path):
    f = tmp_path / "m"
    f.write_text("{}", encoding="utf-8")
    sig = tmp_path / "m.sig"
    sig.write_text(base64.b64encode(SIG.encode()).decode(), encoding="utf-8")
    r, _ = _run(tmp_path, str(f), str(sig), base64.b64encode(PUB.encode()).decode(), status=1)
    assert r.returncode != 0


def test_wrong_argument_count_is_a_usage_error(tmp_path):
    r, _ = _run(tmp_path, "only-one")
    assert r.returncode == 2 and "usage" in r.stderr
