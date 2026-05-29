"""Test that _run_initiate_review_tick computes and threads UserPresence."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.initiate.user_pattern import UserPresence


def test_run_initiate_review_tick_passes_user_presence(tmp_path: Path) -> None:
    """compute_user_presence result must be forwarded as user_presence to run_initiate_review_tick."""
    from brain.bridge.supervisor import _run_initiate_review_tick

    presence = UserPresence(
        silence_days=2.0, ignore_streak=0, likely_active=True, response_lag_p50=None
    )
    captured: list = []

    def _fake_run(persona_dir, *, provider, voice_template, cap_per_tick=3, user_presence=None):
        captured.append(user_presence)

    with (
        patch("brain.bridge.supervisor.compute_user_presence", return_value=presence) as mock_cpu,
        patch("brain.bridge.supervisor.run_initiate_review_tick", side_effect=_fake_run),
        patch("brain.bridge.supervisor.PersonaConfig") as mock_cfg,
    ):
        mock_cfg.load.return_value = MagicMock(initiate_review_cap_per_tick=3)
        event_bus = MagicMock()
        _run_initiate_review_tick(tmp_path, provider=MagicMock(), event_bus=event_bus)

    mock_cpu.assert_called_once_with(tmp_path)
    assert len(captured) == 1
    assert captured[0] is presence
