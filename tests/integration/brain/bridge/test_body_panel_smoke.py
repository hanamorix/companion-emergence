"""Smoke test: BodyPanel transitions across the brain→bridge→UI seam.

This is the integration path UI actually reads:

    MemoryStore → build_persona_state() → result["body"]["body_emotions"]
                                          (dict[name → float, 0..10])

The BodyPanel.tsx then:
- renders "Body quiet." iff all six values are ≤ 0.4 (BodyPanel.tsx:38-42)
- renders bars for any emotion > 0.4, sorted desc, top 5 (BodyPanel.tsx:31-37)

This file walks three scenarios and asserts the dict shape the UI would see.
Run with ``-s -v`` to see the human-readable transition demo printed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.persona_state import build_persona_state
from brain.memory.store import Memory, MemoryStore

# The six body-class emotions tracked in brain/body/state.py:21-30
BODY_EMOTIONS = (
    "arousal",
    "desire",
    "climax",
    "touch_hunger",
    "comfort_seeking",
    "rest_need",
)
QUIET_THRESHOLD = 0.4  # mirrors BodyPanel.tsx:32, :38


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "smoke_persona"
    p.mkdir()
    return p


def _seed(persona_dir: Path, emotions: dict[str, float]) -> None:
    store = MemoryStore(persona_dir / "memories.db")
    try:
        mem = Memory.create_new(
            content="smoke seed",
            memory_type="conversation",
            domain="chat",
            emotions=emotions,
        )
        store.create(mem)
    finally:
        store.close()


def _ui_render(body_emotions: dict[str, float]) -> str:
    """Return what BodyPanel.tsx would render for this body_emotions dict."""
    above = [(n, v) for n, v in body_emotions.items() if v > QUIET_THRESHOLD]
    if not above:
        return "Body quiet."
    above.sort(key=lambda kv: kv[1], reverse=True)
    bars = [f"{n.replace('_', ' ')} {v:.1f}" for n, v in above[:5]]
    return " | ".join(bars)


def _read_body_emotions(persona_dir: Path) -> dict[str, float]:
    state = build_persona_state(persona_dir, now=datetime.now(UTC))
    assert state["body"] is not None, "build_persona_state returned no body block"
    return state["body"]["body_emotions"]


def test_quiet_when_no_emotion_bearing_memories(persona_dir, capsys):
    """Empty store → all six body emotions are 0 → UI shows 'Body quiet.'"""
    # Seed the store file (need at least the schema in place).
    MemoryStore(persona_dir / "memories.db").close()

    body_emotions = _read_body_emotions(persona_dir)
    ui = _ui_render(body_emotions)

    with capsys.disabled():
        print(f"\n[scenario 1: empty store] body_emotions={body_emotions}")
        print(f"  UI renders: {ui!r}")

    assert all(v <= QUIET_THRESHOLD for v in body_emotions.values())
    assert ui == "Body quiet."


def test_single_bar_desire_strong(persona_dir, capsys):
    """A memory with desire:8.0 → body_emotions.desire=8.0 → UI shows desire bar."""
    _seed(persona_dir, {"desire": 8.0})

    body_emotions = _read_body_emotions(persona_dir)
    ui = _ui_render(body_emotions)

    with capsys.disabled():
        print(f"\n[scenario 2: one memory, desire=8.0] body_emotions={body_emotions}")
        print(f"  UI renders: {ui!r}")

    assert body_emotions["desire"] == pytest.approx(8.0)
    # Other five stay quiet
    for name in BODY_EMOTIONS:
        if name != "desire":
            assert body_emotions[name] <= QUIET_THRESHOLD, f"{name}={body_emotions[name]}"
    assert ui != "Body quiet."
    assert "desire" in ui


def test_multi_bar_arousal_desire_touch_hunger(persona_dir, capsys):
    """One memory with arousal:7, desire:6, touch_hunger:5 →
    UI renders three bars sorted desc."""
    _seed(persona_dir, {"arousal": 7.0, "desire": 6.0, "touch_hunger": 5.0})

    body_emotions = _read_body_emotions(persona_dir)
    ui = _ui_render(body_emotions)

    with capsys.disabled():
        print(f"\n[scenario 3: arousal=7, desire=6, touch_hunger=5] body_emotions={body_emotions}")
        print(f"  UI renders: {ui!r}")

    assert body_emotions["arousal"] == pytest.approx(7.0)
    assert body_emotions["desire"] == pytest.approx(6.0)
    assert body_emotions["touch_hunger"] == pytest.approx(5.0)
    assert body_emotions["climax"] <= QUIET_THRESHOLD
    assert body_emotions["comfort_seeking"] <= QUIET_THRESHOLD
    assert body_emotions["rest_need"] <= QUIET_THRESHOLD
    # UI sorts desc: arousal first, then desire, then touch_hunger
    assert ui.startswith("arousal 7.0")
    assert "desire 6.0" in ui
    assert "touch hunger 5.0" in ui  # underscore → space in Bar component


def test_transition_quiet_to_stirring_to_quiet(persona_dir, capsys):
    """Walk the full transition: empty → seed desire → wipe → empty again."""
    MemoryStore(persona_dir / "memories.db").close()
    quiet_state = _read_body_emotions(persona_dir)
    quiet_ui = _ui_render(quiet_state)

    _seed(persona_dir, {"desire": 7.5, "comfort_seeking": 3.2})
    stirring_state = _read_body_emotions(persona_dir)
    stirring_ui = _ui_render(stirring_state)

    # Deactivate the memory (mirrors the brain's natural state churn).
    store = MemoryStore(persona_dir / "memories.db")
    try:
        # Set active=0 on every row.
        store._conn.execute("UPDATE memories SET active = 0")  # noqa: SLF001
        store._conn.commit()  # noqa: SLF001
    finally:
        store.close()
    back_quiet_state = _read_body_emotions(persona_dir)
    back_quiet_ui = _ui_render(back_quiet_state)

    with capsys.disabled():
        print("\n[scenario 4: transition walk]")
        print(f"  step 1 (empty):       {quiet_ui!r}")
        print(f"  step 2 (desire 7.5):  {stirring_ui!r}")
        print(f"  step 3 (deactivated): {back_quiet_ui!r}")

    assert quiet_ui == "Body quiet."
    assert "desire 7.5" in stirring_ui
    # comfort_seeking=3.2 > 0.4 threshold → it should also render as a bar.
    assert "comfort seeking 3.2" in stirring_ui
    assert back_quiet_ui == "Body quiet."
