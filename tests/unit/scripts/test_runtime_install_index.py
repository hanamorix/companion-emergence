"""Both runtime installers must name the pytorch-cpu index (#287, #285).

torch is locked from the `pytorch-cpu` index declared in pyproject.toml. The
installers export the lock with `uv export` (which drops index URLs) and install
it with `uv pip install --require-hashes` (which ignores [tool.uv.sources]), so
without the index named on the install line uv searches PyPI only: Linux and
Windows fail with "no version of torch==...+cpu", and macOS fails the hash check
because PyPI's torch wheel is a different file from the locked one.

The URL is duplicated into the scripts on purpose (`--emit-index-url` needs
uv >= 0.12, newer than many users' uv); this test is the drift guard.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = [REPO / "app" / "build_python_runtime.sh", REPO / "scripts" / "update.sh"]


def _torch_index_url() -> str:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    (source,) = data["tool"]["uv"]["sources"]["torch"]
    return next(i["url"] for i in data["tool"]["uv"]["index"] if i["name"] == source["index"])


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_hashed_install_names_the_torch_index(script):
    url = _torch_index_url()
    lines = script.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
    installs = [line for line in lines if "pip install" in line and "--require-hashes" in line]
    assert installs, f"no hashed `pip install` found in {script.name}"
    for line in installs:
        assert f"--index {url}" in line, line
        # PyTorch's index also lists common packages (certifi) at older versions;
        # first-index would pin those to it and fail. Safe: every file is hash-pinned.
        assert "--index-strategy unsafe-best-match" in line, line
