"""`scripts/brain_main_manifest.py` — the brain-main channel's manifest (#286 slice 3)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_SCRIPT = REPO / "scripts" / "brain_main_manifest.py"


def _load():
    spec = importlib.util.spec_from_file_location("brain_main_manifest", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load()

SHA = "0123456789abcdef0123456789abcdef01234567"
REQS = (
    "certifi==2025.1.31 \\\n"
    "    --hash=sha256:" + "a" * 64 + "\n"
    "torch==2.5.1+cpu \\\n"
    "    --hash=sha256:" + "b" * 64 + "\n"
)


def _files(tmp_path, reqs=REQS, wheel_name="companion_emergence-0.0.42-py3-none-any.whl"):
    wheel = tmp_path / wheel_name
    wheel.write_bytes(b"not really a wheel")
    req = tmp_path / "requirements.txt"
    req.write_text(reqs, encoding="utf-8")
    return wheel, req


def test_manifest_carries_the_spec_fields_and_real_hashes(tmp_path):
    wheel, req = _files(tmp_path)
    m = bm.build_manifest(wheel=wheel, requirements=req, commit=SHA,
                          built_at=datetime(2026, 9, 27, 12, 0, tzinfo=UTC))
    assert m == {
        "schema": 1,
        "commit": SHA,
        "brain_version": "0.0.42",
        "built_at": "2026-09-27T12:00:00Z",
        "python": "3.13",
        "min_bundle_version": bm.MIN_BUNDLE_VERSION,
        "wheel": {"name": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()},
        "requirements": {"name": "requirements.txt",
                         "sha256": hashlib.sha256(req.read_bytes()).hexdigest()},
    }


@pytest.mark.parametrize("commit", ["abc1234", "A" * 40, "g" * 40])
def test_rejects_anything_but_a_full_lowercase_sha(tmp_path, commit):
    wheel, req = _files(tmp_path)
    with pytest.raises(ValueError, match="40-hex"):
        bm.build_manifest(wheel=wheel, requirements=req, commit=commit, built_at=datetime.now(UTC))


def test_rejects_a_foreign_wheel(tmp_path):
    wheel, req = _files(tmp_path, wheel_name="other-1.0-py3-none-any.whl")
    with pytest.raises(ValueError, match="not a companion_emergence wheel"):
        bm.build_manifest(wheel=wheel, requirements=req, commit=SHA, built_at=datetime.now(UTC))


def test_rejects_a_line_nell_update_would_refuse(tmp_path):
    wheel, req = _files(tmp_path, reqs="-e .\n")
    with pytest.raises(RuntimeError, match="unrecognised requirement line"):
        bm.build_manifest(wheel=wheel, requirements=req, commit=SHA, built_at=datetime.now(UTC))


def test_rejects_a_pin_without_a_hash(tmp_path):
    wheel, req = _files(tmp_path, reqs="certifi==2025.1.31\n")
    with pytest.raises(ValueError, match="without hashes: certifi"):
        bm.build_manifest(wheel=wheel, requirements=req, commit=SHA, built_at=datetime.now(UTC))


def test_python_matches_the_bundled_runtime():
    script = (REPO / "app" / "build_python_runtime.sh").read_text(encoding="utf-8")
    full = re.search(r'^PY_VERSION="(\d+\.\d+)\.\d+"', script, re.M).group(1)
    assert bm.PYTHON == full


def test_min_bundle_version_never_predates_the_overlay_hook():
    # v0.0.42 has neither the tolerant readers (slice 1) nor the overlay hook (slice 2).
    assert tuple(int(p) for p in bm.MIN_BUNDLE_VERSION.split(".")) >= (0, 0, 43)


def test_write_cli_writes_and_prints_the_manifest(tmp_path, capsys):
    wheel, req = _files(tmp_path)
    out = tmp_path / "manifest.json"
    rc = bm.main(["write", "--wheel", str(wheel), "--requirements", str(req),
                  "--commit", SHA, "--out", str(out)])
    assert rc == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == json.loads(capsys.readouterr().out)
    assert written["commit"] == SHA


def test_write_cli_exits_1_with_a_reason(tmp_path, capsys):
    wheel, req = _files(tmp_path, reqs="-e .\n")
    rc = bm.main(["write", "--wheel", str(wheel), "--requirements", str(req),
                  "--commit", SHA, "--out", str(tmp_path / "m.json")])
    assert rc == 1
    assert "brain_main_manifest: unrecognised requirement line" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()
