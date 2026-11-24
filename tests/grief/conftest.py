"""The grief tests write memories under `sorrow`, a feeling her vocabulary holds beyond the framework's own."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _her_feelings(register_feelings) -> None:
    register_feelings("sorrow")
