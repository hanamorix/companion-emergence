# tests/unit/brain/maker/test_wiring_emotion.py
from brain.maker.maker import Making
from brain.maker.wiring import making_emotion_delta


def test_emotion_delta_is_vocab_filtered_and_source_varied(monkeypatch):
    # registered vocab present
    import brain.chat.extractor as ext
    monkeypatch.setattr(ext, "_filter_to_registered", lambda d: {k: v for k, v in d.items() if k in {"tenderness", "satisfaction"}})
    d = making_emotion_delta(Making("elegy", "t", "c", "private", None), dominant_source="grief")
    assert set(d).issubset({"tenderness", "satisfaction"})
    assert all(v > 0 for v in d.values())
