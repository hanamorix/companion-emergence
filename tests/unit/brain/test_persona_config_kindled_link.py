"""PersonaConfig kindled_link_enabled field: default OFF, round-trip through save→load.

D2 criteria (added one test at a time per TDD).
"""
from brain.persona_config import PersonaConfig


def test_kindled_link_enabled_defaults_false():
    c = PersonaConfig()
    assert c.kindled_link_enabled is False


def test_kindled_link_enabled_missing_key_loads_false(tmp_path):
    """A config dict with no kindled_link_enabled key loads as False."""
    c = PersonaConfig.load(tmp_path / "missing.json")
    assert c.kindled_link_enabled is False


def test_kindled_link_enabled_round_trip(tmp_path):
    """Round-trip through save→load preserves kindled_link_enabled=True."""
    path = tmp_path / "persona_config.json"
    PersonaConfig(kindled_link_enabled=True).save(path)
    loaded = PersonaConfig.load(path)
    assert loaded.kindled_link_enabled is True
