"""PersonaConfig kindled_relay_url field: defaults None, round-trips save→load.

Part A of Phase 7a T7 — supervisor wiring.
"""
from brain.persona_config import PersonaConfig


def test_kindled_relay_url_defaults_none():
    c = PersonaConfig()
    assert c.kindled_relay_url is None


def test_kindled_relay_url_round_trip(tmp_path):
    """Round-trip through save→load preserves kindled_relay_url."""
    path = tmp_path / "persona_config.json"
    PersonaConfig(kindled_relay_url="https://relay.example.com").save(path)
    loaded = PersonaConfig.load(path)
    assert loaded.kindled_relay_url == "https://relay.example.com"
