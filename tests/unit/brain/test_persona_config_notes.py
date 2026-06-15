"""PersonaConfig notes fields: default OFF, and round-trip through save→load
so notes_enabled / notes_folder survive a write to persona_config.json."""
from brain.persona_config import PersonaConfig


def test_notes_fields_default_off(tmp_path):
    c = PersonaConfig.load(tmp_path / "missing.json")
    assert c.notes_enabled is False
    assert c.notes_folder is None


def test_notes_fields_round_trip(tmp_path):
    path = tmp_path / "persona_config.json"
    PersonaConfig(notes_enabled=True, notes_folder="/Users/x/Documents/Nell Notes").save(path)
    loaded = PersonaConfig.load(path)
    assert loaded.notes_enabled is True
    assert loaded.notes_folder == "/Users/x/Documents/Nell Notes"
