"""The grief tests write memories under `sorrow`, a feeling her vocabulary holds beyond the framework's own."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _her_feelings(register_feelings, tmp_path) -> None:
    # In her file too: a process that is not the owner reads the persona folder's file and its copy follows it (#803 step 15).
    register_feelings("sorrow", persona_dir=tmp_path)
