import json

from brain.bridge.feed import build_monologue_entries


def _write_digest(persona_dir, lines):
    p = persona_dir / "monologue_digest.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in lines) + "\n")


def test_monologue_feed_hides_unsurfaced(tmp_path):
    _write_digest(
        tmp_path,
        [
            {"ts": "2026-06-01T10:00:00Z", "digest": "shown", "surfaced": True},
            {"ts": "2026-06-01T11:00:00Z", "digest": "hidden", "surfaced": False},
        ],
    )
    bodies = [e.body for e in build_monologue_entries(tmp_path, limit=10)]
    assert "shown" in bodies
    assert "hidden" not in bodies


def test_monologue_feed_missing_surfaced_treated_as_true(tmp_path):
    _write_digest(tmp_path, [{"ts": "2026-06-01T10:00:00Z", "digest": "legacy"}])
    bodies = [e.body for e in build_monologue_entries(tmp_path, limit=10)]
    assert "legacy" in bodies


def test_monologue_feed_dev_override_reveals_withheld(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_REVEAL_WITHHELD_MONOLOGUE", "1")
    _write_digest(
        tmp_path, [{"ts": "2026-06-01T11:00:00Z", "digest": "hidden", "surfaced": False}]
    )
    bodies = [e.body for e in build_monologue_entries(tmp_path, limit=10)]
    assert "hidden" in bodies


def test_build_feed_includes_surfaced_monologue(tmp_path):
    """A surfaced monologue survives the full multi-source build_feed merge
    (closes the gap where the merge test omitted monologue entirely)."""
    from brain.bridge.feed import build_feed

    _write_digest(
        tmp_path,
        [{"ts": "2026-06-01T12:00:00Z", "digest": "merged monologue", "surfaced": True}],
    )
    feed = build_feed(tmp_path, limit=50)
    assert any(e.type == "monologue" and e.body == "merged monologue" for e in feed)
