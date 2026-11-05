"""cryptography must be a DIRECT dependency, not just transitive (crypto-decision doc §1)."""
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_cryptography_is_direct_dependency() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("cryptography") for d in deps), (
        "cryptography must be an explicit dependency in pyproject.toml"
    )
