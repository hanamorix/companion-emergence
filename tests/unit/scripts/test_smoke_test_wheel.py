"""scripts/smoke_test_wheel.sh must install the CPU torch build (#290).

The smoke installs the wheel unlocked (no lockfile, no [tool.uv.sources]), so on
Linux uv resolves torch from PyPI: the CUDA build plus ~15 nvidia-* packages,
gigabytes per run. The project only ever ships CPU torch (pyproject.toml's
pytorch-cpu pin). UV_TORCH_BACKEND=cpu routes just the PyTorch-ecosystem
packages to PyTorch's CPU index; uv versions without it ignore the variable.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "smoke_test_wheel.sh"


def test_wheel_install_uses_the_cpu_torch_backend():
    lines = SCRIPT.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
    installs = [line for line in lines if "pip install" in line and "$WHEEL" in line]
    assert installs, "no wheel install found in smoke_test_wheel.sh"
    for line in installs:
        assert "UV_TORCH_BACKEND=cpu" in line, line
