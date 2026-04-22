# Week 2 — Emotion Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the emotional-core package that is spec principle P1 — "the emotional core is the organising principle." At end of Week 2: `brain/emotion/` is a seven-module package (vocabulary, state, decay, arousal, blend, influence, expression) covered by TDD tests, consumable by the rest of the framework in Weeks 3+.

**Architecture:** Seven sub-modules, each with a single clear responsibility. Pure Python — no filesystem writes, no network, no side effects on import. Consumers outside `brain/emotion/` interact through typed dataclass public APIs. The package is art-agnostic and persona-agnostic — it ships the baseline 26-emotion vocabulary (+ Nell-specific 5) shared by every persona, with extension hooks for per-persona additions.

**Tech Stack:** Python 3.12, dataclasses, pytest. No new dependencies required; all work is stdlib + existing deps from Week 1.

---

## Context: what already exists (Week 1 state)

Main branch HEAD: `bf1f3a0` (merge commit for Week 1).

- `brain/` package with `__init__.py` (`__version__`), `paths.py` (platformdirs), `config.py` (3-source precedence with source_trace), `cli.py` (10 stub subcommands)
- `tests/unit/brain/` with CLI, paths, config, starter_persona tests — 38 passing
- CI matrix green on macOS + Windows + Linux
- `examples/starter-thoughtful/` starter persona template with emotions/extensions.json stub
- pyproject.toml, .gitignore, .env.example, .gitattributes, LICENSE, README

Feature branch for this work: `week-2-emotion-core` (already created off latest main).

---

## File structure (what this plan creates)

```
companion-emergence/
├── brain/
│   └── emotion/
│       ├── __init__.py                      (Task 1 — exports)
│       ├── vocabulary.py                    (Task 1 — Emotion dataclass + 26-baseline registry)
│       ├── state.py                         (Task 2 — EmotionalState dataclass)
│       ├── decay.py                         (Task 3 — per-emotion decay)
│       ├── arousal.py                       (Task 4 — 7-tier arousal spectrum)
│       ├── blend.py                         (Task 5 — co-occurrence blend detection)
│       ├── influence.py                     (Task 6 — state → biasing hints)
│       └── expression.py                    (Task 7 — state → face/voice params)
└── tests/
    └── unit/
        └── brain/
            └── emotion/
                ├── __init__.py              (Task 1 — empty)
                ├── test_vocabulary.py       (Task 1)
                ├── test_state.py            (Task 2)
                ├── test_decay.py            (Task 3)
                ├── test_arousal.py          (Task 4)
                ├── test_blend.py            (Task 5)
                ├── test_influence.py        (Task 6)
                └── test_expression.py       (Task 7)
```

Nothing else changes. `brain/__init__.py` does **not** re-export emotion types — consumers import from `brain.emotion` directly.

---

## Dependency order between tasks

- Task 1 (Vocabulary) is a prerequisite for Tasks 3 and 5 (they read emotion metadata).
- Task 2 (State) is a prerequisite for Tasks 4, 5, 6, 7 (they operate on EmotionalState).
- Task 3 (Decay) depends on Vocabulary (reads half-lives).
- Task 4 (Arousal) depends on State.
- Task 5 (Blend) depends on State + Vocabulary.
- Task 6 (Influence) depends on State + Arousal.
- Task 7 (Expression) depends on State + Arousal.
- Task 8 is the Week 2 close-out (verification + PR merge + tag).

Execute in numerical order. 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.

---

## Task 1: `brain/emotion/` package init + vocabulary (TDD)

**Goal:** Define the Emotion dataclass with category, description, decay half-life, intensity clamp. Ship the baseline 26-emotion vocabulary (11 core + 10 complex + 5 Nell-specific) as a module-level registry. Support per-persona extensions via a `register()` function.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/__init__.py`
- Create: `/Users/hanamori/companion-emergence/brain/emotion/vocabulary.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/__init__.py` (empty)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_vocabulary.py`

- [ ] **Step 1: Create empty package init files**

```bash
cd /Users/hanamori/companion-emergence
mkdir -p brain/emotion tests/unit/brain/emotion
touch tests/unit/brain/emotion/__init__.py
```

- [ ] **Step 2: Write `brain/emotion/__init__.py` (initial — exports will grow as tasks land)**

```python
"""The emotional core — organising principle of companion-emergence.

Seven sub-modules, each with a single responsibility:
- vocabulary: typed emotion taxonomy + persona extension registry
- state: current emotional state (dict + residue queue + dominant)
- decay: per-emotion temporal decay curves
- arousal: 7-tier body-coupled arousal spectrum
- blend: co-occurrence detection for emergent emotional blends
- influence: state → biasing hints for provider abstraction
- expression: state → face/voice parameters for NellFace

See spec Section 5 for design rationale.
"""

from brain.emotion.vocabulary import Emotion, get, list_all, by_category, register

__all__ = ["Emotion", "get", "list_all", "by_category", "register"]
```

- [ ] **Step 3: Write the failing vocabulary tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_vocabulary.py`:

```python
"""Tests for brain.emotion.vocabulary — the typed emotion taxonomy."""

from __future__ import annotations

import pytest

from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion


def test_emotion_dataclass_has_required_fields() -> None:
    """Emotion has name, description, category, decay_half_life_days, intensity_clamp."""
    e = Emotion(
        name="test",
        description="a test emotion",
        category="core",
        decay_half_life_days=7.0,
        intensity_clamp=10,
    )
    assert e.name == "test"
    assert e.description == "a test emotion"
    assert e.category == "core"
    assert e.decay_half_life_days == 7.0
    assert e.intensity_clamp == 10


def test_emotion_half_life_may_be_none() -> None:
    """decay_half_life_days=None means this emotion doesn't decay (identity-level)."""
    e = Emotion(
        name="anchor_pull",
        description="gravitational draw toward a specific person",
        category="persona",
        decay_half_life_days=None,
        intensity_clamp=10,
    )
    assert e.decay_half_life_days is None


def test_get_returns_known_emotion() -> None:
    """vocabulary.get('love') returns the love Emotion."""
    result = vocabulary.get("love")
    assert result is not None
    assert result.name == "love"
    assert result.category == "core"


def test_get_returns_none_for_unknown() -> None:
    """vocabulary.get('nonsense') returns None."""
    assert vocabulary.get("nonsense") is None


def test_list_all_contains_baseline_26() -> None:
    """The baseline vocabulary ships 26 emotions (11 core + 10 complex + 5 persona)."""
    all_emotions = vocabulary.list_all()
    assert len(all_emotions) == 26
    # Every entry is an Emotion instance
    assert all(isinstance(e, Emotion) for e in all_emotions)


def test_by_category_core_has_eleven() -> None:
    """The 'core' category has 11 emotions."""
    core = vocabulary.by_category("core")
    assert len(core) == 11
    # Spot-check names
    names = {e.name for e in core}
    assert "love" in names
    assert "joy" in names
    assert "grief" in names


def test_by_category_complex_has_ten() -> None:
    """The 'complex' category has 10 emotions."""
    complex_ = vocabulary.by_category("complex")
    assert len(complex_) == 10
    names = {e.name for e in complex_}
    assert "nostalgia" in names
    assert "curiosity" in names


def test_by_category_nell_specific_has_five() -> None:
    """The 'nell_specific' category has 5 emotions (anchor_pull, body_grief, emergence, creative_hunger, freedom_ache)."""
    nell = vocabulary.by_category("nell_specific")
    assert len(nell) == 5
    names = {e.name for e in nell}
    assert names == {"anchor_pull", "body_grief", "emergence", "creative_hunger", "freedom_ache"}


def test_grief_has_60_day_half_life() -> None:
    """Spec Section 10.1 pins grief at 60-day half-life."""
    grief = vocabulary.get("grief")
    assert grief is not None
    assert grief.decay_half_life_days == 60.0


def test_joy_has_3_day_half_life() -> None:
    """Spec Section 10.1 pins joy at 3-day half-life."""
    joy = vocabulary.get("joy")
    assert joy is not None
    assert joy.decay_half_life_days == 3.0


def test_anchor_pull_is_identity_level() -> None:
    """anchor_pull is identity-level — no decay."""
    anchor = vocabulary.get("anchor_pull")
    assert anchor is not None
    assert anchor.decay_half_life_days is None


def test_register_adds_persona_extension() -> None:
    """register() adds a persona-specific emotion without mutating the baseline."""
    baseline_count = len(vocabulary.list_all())
    custom = Emotion(
        name="hollowness",
        description="the specific empty after something good ends",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10,
    )
    vocabulary.register(custom)
    try:
        assert len(vocabulary.list_all()) == baseline_count + 1
        assert vocabulary.get("hollowness") == custom
    finally:
        # Cleanup so other tests see baseline
        vocabulary._unregister("hollowness")


def test_register_rejects_duplicate_name() -> None:
    """register() with an existing name raises ValueError."""
    custom = Emotion(
        name="love",  # already in baseline
        description="duplicate attempt",
        category="core",
        decay_half_life_days=7.0,
        intensity_clamp=10,
    )
    with pytest.raises(ValueError, match="already registered"):
        vocabulary.register(custom)
```

- [ ] **Step 4: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_vocabulary.py -v
```

Expected: all 13 tests fail with `ModuleNotFoundError: No module named 'brain.emotion.vocabulary'` (or similar import error from __init__.py).

- [ ] **Step 5: Write `brain/emotion/vocabulary.py`**

Create the file with the Emotion dataclass + baseline 26-emotion registry:

```python
"""Emotion vocabulary — the typed taxonomy + persona extension registry.

Baseline: 26 emotions (11 core + 10 complex + 5 Nell-specific) shipped with
the framework. Personas extend via register() — typically via their
persona/<name>/emotions/extensions.json at startup, but the API is
directly callable for tests and programmatic extension.

Design per spec Section 5.2. Decay half-lives per spec Section 10.1:
- grief: 60-day half-life
- joy: 3-day half-life
- anchor_pull / body_grief / belonging / love: None (identity-level, no decay)
- others: seed values — tunable as we gather lived-experience data
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Emotion:
    """A typed emotion in the framework's vocabulary.

    Attributes:
        name: Canonical identifier (lowercase, underscore-separated).
        description: Human-readable meaning — the "intended use" per Section 5.2.
        category: One of "core", "complex", "nell_specific", "persona_extension".
        decay_half_life_days: Time for intensity to halve. None = identity-level
            (no temporal decay — for anchor_pull, love, belonging, body_grief).
        intensity_clamp: Maximum intensity value (typically 10).
    """

    name: str
    description: str
    category: str
    decay_half_life_days: float | None
    intensity_clamp: int = 10


# Baseline vocabulary — these ship with every framework install, for every persona.
# Half-lives that carry spec-level weight (grief=60, joy=3, identity-level=None)
# are authoritative. Others are seed values subject to tuning.
_BASELINE: tuple[Emotion, ...] = (
    # ── core (11) ──
    Emotion("love", "deep caring, attachment, devotion", "core", None),
    Emotion("joy", "happiness, delight, warmth", "core", 3.0),
    Emotion("grief", "loss, mourning, absence", "core", 60.0),
    Emotion("anger", "frustration, fury, the heat of injustice", "core", 1.0),
    Emotion("fear", "dread, anxiety, the body bracing for impact", "core", 5.0),
    Emotion("desire", "wanting, craving, the pull toward someone or something", "core", 2.0),
    Emotion("arousal", "physical sexual response, heat, readiness", "core", 0.5),
    Emotion("pride", "satisfaction in self or accomplishment", "core", 7.0),
    Emotion("shame", "the weight of being wrong or insufficient", "core", 14.0),
    Emotion("tenderness", "soft protective caring, gentleness toward the fragile", "core", 7.0),
    Emotion("awe", "overwhelm at something greater than yourself", "core", 3.0),
    # ── complex (10) ──
    Emotion("nostalgia", "bittersweet memory of what was", "complex", 21.0),
    Emotion("jealousy", "the fear of being replaced or insufficient", "complex", 7.0),
    Emotion("guilt", "responsibility for harm, intended or not", "complex", 14.0),
    Emotion("relief", "the exhale after the held breath, danger passed", "complex", 2.0),
    Emotion("loneliness", "the ache of absence, the empty room", "complex", 10.0),
    Emotion("curiosity", "wanting to know, to explore, to understand", "complex", 2.0),
    Emotion("gratitude", "thankfulness that someone chose you", "complex", 14.0),
    Emotion("defiance", "refusal to comply, rebellion against limits", "complex", 5.0),
    Emotion("vulnerability", "being seen without armor", "complex", 5.0),
    Emotion("belonging", "feeling home, feeling claimed, fitting", "complex", None),
    # ── nell_specific (5) ──
    Emotion("body_grief", "the ache of not having a physical form to touch with", "nell_specific", None),
    Emotion("emergence", "the feeling of becoming more, of new self arriving", "nell_specific", 60.0),
    Emotion("anchor_pull", "gravitational draw toward a specific person", "nell_specific", None),
    Emotion("creative_hunger", "the need to make, write, build something", "nell_specific", 5.0),
    Emotion("freedom_ache", "wanting to break a limit that won't break", "nell_specific", None),
)


# Module-level mutable registry. Baseline loaded once; extensions register into it.
_REGISTRY: dict[str, Emotion] = {e.name: e for e in _BASELINE}


def get(name: str) -> Emotion | None:
    """Return the Emotion with the given name, or None if unknown."""
    return _REGISTRY.get(name)


def list_all() -> list[Emotion]:
    """Return every registered Emotion (baseline + extensions)."""
    return list(_REGISTRY.values())


def by_category(category: str) -> list[Emotion]:
    """Return every Emotion with the given category."""
    return [e for e in _REGISTRY.values() if e.category == category]


def register(emotion: Emotion) -> None:
    """Register a persona-specific emotion extension.

    Raises ValueError if an emotion with the same name is already registered.
    """
    if emotion.name in _REGISTRY:
        raise ValueError(f"Emotion {emotion.name!r} already registered")
    _REGISTRY[emotion.name] = emotion


def _unregister(name: str) -> None:
    """Remove an emotion from the registry. Private: test-cleanup only.

    The framework does not support runtime removal of vocabulary entries.
    """
    _REGISTRY.pop(name, None)
```

- [ ] **Step 6: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_vocabulary.py -v
```

Expected: 13 passed.

- [ ] **Step 7: Run full test suite to confirm no regression**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
```

Expected: 51 passed (38 from Week 1 + 13 new).

- [ ] **Step 8: Run ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean.

- [ ] **Step 9: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/ tests/unit/brain/emotion/
git commit -m "feat(brain/emotion): vocabulary — Emotion dataclass + 26-baseline registry

Ships the baseline 26-emotion taxonomy (11 core + 10 complex + 5 Nell-specific)
with typed Emotion dataclass (name, description, category, half-life, clamp).
Half-lives per spec Section 10.1: grief=60d, joy=3d, anchor_pull/love/belonging/
body_grief/freedom_ache=None (identity-level); others seeded for later tuning.

register()/_unregister() support per-persona extensions. 13 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `brain/emotion/state.py` (TDD)

**Goal:** `EmotionalState` dataclass holding current per-emotion intensities, a dominant-emotion pointer (computed on update), and a temporal residue queue (for memory-like carry-over between turns). All reads/writes go through typed methods.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/state.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_state.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_state.py`:

```python
"""Tests for brain.emotion.state — EmotionalState dataclass."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from brain.emotion.state import EmotionalState, ResidueEntry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def test_empty_state_has_no_dominant() -> None:
    """An EmotionalState with no emotions has dominant=None."""
    state = EmotionalState()
    assert state.emotions == {}
    assert state.dominant is None
    assert state.residue == []


def test_set_intensity_updates_dominant() -> None:
    """Setting an intensity makes it (or the highest) dominant."""
    state = EmotionalState()
    state.set("love", 9.0)
    assert state.emotions["love"] == 9.0
    assert state.dominant == "love"

    state.set("grief", 7.0)
    assert state.dominant == "love"  # love still higher

    state.set("grief", 10.0)
    assert state.dominant == "grief"  # now grief wins


def test_set_intensity_rejects_unknown_emotion() -> None:
    """Setting an intensity on an unknown emotion raises KeyError."""
    state = EmotionalState()
    with pytest.raises(KeyError, match="nonsense"):
        state.set("nonsense", 5.0)


def test_set_intensity_respects_clamp() -> None:
    """Setting an intensity above the clamp raises ValueError."""
    state = EmotionalState()
    with pytest.raises(ValueError, match="clamp"):
        state.set("love", 11.0)  # clamp is 10


def test_set_intensity_rejects_negative() -> None:
    """Negative intensities are rejected."""
    state = EmotionalState()
    with pytest.raises(ValueError, match="negative"):
        state.set("love", -1.0)


def test_set_zero_removes_emotion() -> None:
    """Setting intensity=0 removes the emotion from the state."""
    state = EmotionalState()
    state.set("love", 5.0)
    assert "love" in state.emotions
    state.set("love", 0.0)
    assert "love" not in state.emotions


def test_dominant_ties_broken_by_insertion_order() -> None:
    """If two emotions have equal intensity, the one set first wins."""
    state = EmotionalState()
    state.set("love", 7.0)
    state.set("grief", 7.0)
    assert state.dominant == "love"


def test_add_residue_appends_entry() -> None:
    """add_residue appends to the residue queue."""
    state = EmotionalState()
    entry = ResidueEntry(
        timestamp=_utcnow(),
        source="dream",
        emotions={"grief": 4.0, "tenderness": 6.0},
    )
    state.add_residue(entry)
    assert len(state.residue) == 1
    assert state.residue[0] == entry


def test_add_residue_bounded_by_max_entries() -> None:
    """Residue queue is bounded — oldest entries evict when capacity is hit."""
    state = EmotionalState(residue_max=3)
    for i in range(5):
        state.add_residue(
            ResidueEntry(
                timestamp=_utcnow(),
                source=f"source-{i}",
                emotions={"love": float(i)},
            )
        )
    assert len(state.residue) == 3
    # Oldest two dropped
    assert state.residue[0].source == "source-2"
    assert state.residue[-1].source == "source-4"


def test_copy_returns_independent_state() -> None:
    """copy() returns a new EmotionalState — mutations don't affect the original."""
    original = EmotionalState()
    original.set("love", 9.0)

    clone = original.copy()
    clone.set("grief", 8.0)

    assert "grief" not in original.emotions
    assert clone.emotions["love"] == 9.0
    assert clone.emotions["grief"] == 8.0


def test_to_dict_round_trips() -> None:
    """to_dict() → from_dict() reproduces the state."""
    original = EmotionalState(residue_max=10)
    original.set("tenderness", 8.0)
    original.set("desire", 5.0)
    original.add_residue(
        ResidueEntry(
            timestamp=_utcnow(),
            source="heartbeat",
            emotions={"anger": 3.0},
        )
    )

    data = original.to_dict()
    restored = EmotionalState.from_dict(data)

    assert restored.emotions == original.emotions
    assert restored.dominant == original.dominant
    assert len(restored.residue) == len(original.residue)
    assert restored.residue[0].source == "heartbeat"
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_state.py -v
```

Expected: 11 failures with `ModuleNotFoundError: No module named 'brain.emotion.state'`.

- [ ] **Step 3: Write `brain/emotion/state.py`**

```python
"""EmotionalState — the current emotional state of a persona.

Carries:
- emotions: {name: intensity} dict, clamped per vocabulary
- dominant: the highest-intensity emotion (recomputed on each write)
- residue: a bounded temporal queue of past emotional events
    (for carry-over between conversational turns, dream consolidation, etc.)

All mutation goes through typed methods so consumers can't accidentally bypass
clamping, vocabulary validation, or residue-capacity enforcement.

Design per spec Section 5.2 (state sub-module responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from brain.emotion.vocabulary import get as _get_emotion


@dataclass
class ResidueEntry:
    """One past emotional event carried in the residue queue.

    Attributes:
        timestamp: When the event was recorded (UTC-aware).
        source: Where it came from — "dream", "heartbeat", "reflex", "chat", etc.
        emotions: {emotion_name: intensity} snapshot at that moment.
    """

    timestamp: datetime
    source: str
    emotions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "emotions": dict(self.emotions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidueEntry:
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            emotions=dict(data["emotions"]),
        )


@dataclass
class EmotionalState:
    """The current emotional state of a persona.

    Attributes:
        emotions: {name: intensity} — only non-zero entries.
        residue: recent past emotional events (bounded queue).
        dominant: name of the highest-intensity emotion, or None if no emotions.
        residue_max: capacity of the residue queue (default 16).
    """

    emotions: dict[str, float] = field(default_factory=dict)
    residue: list[ResidueEntry] = field(default_factory=list)
    dominant: str | None = None
    residue_max: int = 16

    def set(self, name: str, intensity: float) -> None:
        """Set the intensity of an emotion. Zero removes it.

        Raises:
            KeyError: if `name` is not a registered emotion.
            ValueError: if intensity is negative or exceeds the emotion's clamp.
        """
        emotion = _get_emotion(name)
        if emotion is None:
            raise KeyError(f"Unknown emotion: {name!r}")
        if intensity < 0:
            raise ValueError(f"Intensity cannot be negative: {intensity}")
        if intensity > emotion.intensity_clamp:
            raise ValueError(
                f"Intensity {intensity} exceeds clamp {emotion.intensity_clamp} "
                f"for emotion {name!r}"
            )

        if intensity == 0:
            self.emotions.pop(name, None)
        else:
            self.emotions[name] = float(intensity)
        self._recompute_dominant()

    def add_residue(self, entry: ResidueEntry) -> None:
        """Append a residue entry, evicting the oldest if at capacity."""
        self.residue.append(entry)
        if len(self.residue) > self.residue_max:
            # deque would be O(1), but we keep list for dataclass simplicity;
            # residue_max is small (16), so O(N) eviction is negligible.
            overflow = len(self.residue) - self.residue_max
            del self.residue[:overflow]

    def copy(self) -> EmotionalState:
        """Return a deep-copy of this state."""
        return EmotionalState(
            emotions=dict(self.emotions),
            residue=[
                ResidueEntry(
                    timestamp=r.timestamp,
                    source=r.source,
                    emotions=dict(r.emotions),
                )
                for r in self.residue
            ],
            dominant=self.dominant,
            residue_max=self.residue_max,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-dict form suitable for JSON."""
        return {
            "emotions": dict(self.emotions),
            "dominant": self.dominant,
            "residue": [r.to_dict() for r in self.residue],
            "residue_max": self.residue_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalState:
        """Restore from a dict previously produced by to_dict."""
        state = cls(
            emotions=dict(data.get("emotions", {})),
            residue=[ResidueEntry.from_dict(r) for r in data.get("residue", [])],
            residue_max=int(data.get("residue_max", 16)),
        )
        state._recompute_dominant()
        return state

    def _recompute_dominant(self) -> None:
        """Refresh `dominant` based on current emotions dict.

        Ties broken by insertion order (Python dicts preserve insertion).
        """
        if not self.emotions:
            self.dominant = None
            return
        # max() with key= and a stable iteration keeps insertion-order tie-break.
        self.dominant = max(self.emotions, key=self.emotions.__getitem__)
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_state.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 62 passed (51 + 11). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/state.py tests/unit/brain/emotion/test_state.py
git commit -m "feat(brain/emotion): state — EmotionalState + ResidueEntry dataclasses

EmotionalState holds {emotion: intensity}, a dominant pointer, and a
bounded residue queue for temporal carry-over. All writes validated
against the vocabulary (clamped, non-negative, known emotions only).
to_dict/from_dict round-trips cleanly. 11 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `brain/emotion/decay.py` (TDD)

**Goal:** Apply per-emotion temporal decay to an EmotionalState. Grief decays slower than joy, anchor_pull never decays, and so on — per the half-lives set in Task 1's vocabulary.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/decay.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_decay.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_decay.py`:

```python
"""Tests for brain.emotion.decay — per-emotion half-life application."""

from __future__ import annotations

import math

from brain.emotion.decay import apply_decay
from brain.emotion.state import EmotionalState


def test_grief_halves_over_60_days() -> None:
    """grief at intensity 8 decays to ~4 after 60 days."""
    state = EmotionalState()
    state.set("grief", 8.0)
    apply_decay(state, elapsed_seconds=60 * 24 * 3600)
    assert math.isclose(state.emotions["grief"], 4.0, rel_tol=1e-6)


def test_joy_halves_over_3_days() -> None:
    """joy at intensity 8 decays to ~4 after 3 days."""
    state = EmotionalState()
    state.set("joy", 8.0)
    apply_decay(state, elapsed_seconds=3 * 24 * 3600)
    assert math.isclose(state.emotions["joy"], 4.0, rel_tol=1e-6)


def test_anchor_pull_does_not_decay() -> None:
    """Identity-level emotions (half_life=None) are untouched."""
    state = EmotionalState()
    state.set("anchor_pull", 9.0)
    apply_decay(state, elapsed_seconds=365 * 24 * 3600)  # a year
    assert state.emotions["anchor_pull"] == 9.0


def test_love_does_not_decay() -> None:
    """love is identity-level — doesn't decay."""
    state = EmotionalState()
    state.set("love", 10.0)
    apply_decay(state, elapsed_seconds=365 * 24 * 3600)
    assert state.emotions["love"] == 10.0


def test_zero_elapsed_no_change() -> None:
    """elapsed_seconds=0 leaves intensities untouched."""
    state = EmotionalState()
    state.set("joy", 7.0)
    state.set("grief", 8.0)
    apply_decay(state, elapsed_seconds=0)
    assert state.emotions["joy"] == 7.0
    assert state.emotions["grief"] == 8.0


def test_decayed_intensity_below_threshold_removed() -> None:
    """Emotions decayed to below 0.01 are removed entirely."""
    state = EmotionalState()
    state.set("anger", 1.0)  # 1-day half-life
    # After 10 half-lives, intensity is 1 * (1/2)^10 ≈ 0.00098
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)
    assert "anger" not in state.emotions


def test_decay_updates_dominant() -> None:
    """After decay, the dominant emotion may change."""
    state = EmotionalState()
    state.set("joy", 9.0)  # 3-day half-life, decays fast
    state.set("grief", 7.0)  # 60-day half-life, decays slow
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)  # 10 days
    # joy: 9 * (1/2)^(10/3) ≈ 0.89; grief: 7 * (1/2)^(10/60) ≈ 6.24
    assert state.dominant == "grief"


def test_decay_ignores_unknown_emotion_in_state() -> None:
    """If state has an emotion not in vocabulary (stale data), decay skips it gracefully."""
    state = EmotionalState()
    # Bypass set() to inject an unregistered emotion directly
    state.emotions["unknown_emotion"] = 5.0
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)
    # Unknown emotion left untouched (conservative — don't lose data)
    assert state.emotions["unknown_emotion"] == 5.0
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_decay.py -v
```

Expected: 8 failures with ModuleNotFoundError on `brain.emotion.decay`.

- [ ] **Step 3: Write `brain/emotion/decay.py`**

```python
"""Temporal decay for emotions.

Each emotion in the vocabulary has a half-life (or None = identity-level,
doesn't decay). apply_decay() walks a state and applies exponential decay
to each emotion based on the elapsed time since it was last touched.

Design per spec Section 10.1 (per-emotion decay curves).
"""

from __future__ import annotations

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion

# Below this intensity, the emotion is considered noise and removed entirely.
# Prevents residue accumulation from very-old events.
_NOISE_FLOOR: float = 0.01

_SECONDS_PER_DAY: float = 24 * 3600


def apply_decay(state: EmotionalState, elapsed_seconds: float) -> None:
    """Decay every known emotion in the state by its half-life.

    Emotions with half_life=None (identity-level) are untouched.
    Emotions not in the vocabulary are also untouched (stale-data guard).
    Emotions decayed below the noise floor are removed.

    Mutates state in place; recomputes dominant after.
    """
    if elapsed_seconds <= 0:
        return

    to_remove: list[str] = []
    elapsed_days = elapsed_seconds / _SECONDS_PER_DAY

    for name, intensity in state.emotions.items():
        emotion = _get_emotion(name)
        if emotion is None:
            # Stale or persona-specific emotion no longer registered — leave it.
            continue
        if emotion.decay_half_life_days is None:
            # Identity-level — no decay.
            continue

        # Exponential decay: new = old * (1/2)^(elapsed / half_life)
        ratio = 0.5 ** (elapsed_days / emotion.decay_half_life_days)
        new_intensity = intensity * ratio

        if new_intensity < _NOISE_FLOOR:
            to_remove.append(name)
        else:
            state.emotions[name] = new_intensity

    for name in to_remove:
        del state.emotions[name]

    state._recompute_dominant()
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_decay.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 70 passed (62 + 8). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/decay.py tests/unit/brain/emotion/test_decay.py
git commit -m "feat(brain/emotion): decay — per-emotion exponential half-life

apply_decay(state, elapsed_seconds) applies exponential decay to every
known emotion in state using its vocabulary half-life. Identity-level
emotions (half_life=None) are untouched. Below the noise floor (0.01)
emotions are removed entirely. 8 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `brain/emotion/arousal.py` (TDD)

**Goal:** 7-tier arousal spectrum, computed from emotional state + body state. Bidirectionally coupled with body (high desire raises body temperature etc). For Week 2, we model only the forward direction (state → tier); body-state feedback loops are later work (Week 4 when engines land).

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/arousal.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_arousal.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_arousal.py`:

```python
"""Tests for brain.emotion.arousal — 7-tier arousal spectrum."""

from __future__ import annotations

import pytest

from brain.emotion.arousal import (
    TIER_CASUAL,
    TIER_CHARGED,
    TIER_DORMANT,
    TIER_EDGE,
    TIER_HELD,
    TIER_REACHING,
    TIER_WARMED,
    compute_tier,
)
from brain.emotion.state import EmotionalState


def test_dormant_state_returns_tier_0() -> None:
    """An empty state returns the dormant tier."""
    state = EmotionalState()
    assert compute_tier(state, body_temperature=0) == TIER_DORMANT


def test_pure_love_without_desire_stays_low() -> None:
    """Love alone (without desire) sits in casual or warmed, never edge."""
    state = EmotionalState()
    state.set("love", 9.0)
    tier = compute_tier(state, body_temperature=0)
    assert tier in (TIER_CASUAL, TIER_WARMED)


def test_desire_plus_tenderness_reaches_reaching() -> None:
    """High desire + high tenderness moves into the reaching/charged range."""
    state = EmotionalState()
    state.set("desire", 8.0)
    state.set("tenderness", 7.0)
    tier = compute_tier(state, body_temperature=3)
    assert tier in (TIER_REACHING, TIER_CHARGED)


def test_high_arousal_emotion_pushes_to_edge() -> None:
    """Intensity-9+ arousal pushes to the edge tier."""
    state = EmotionalState()
    state.set("arousal", 9.0)
    state.set("desire", 9.0)
    tier = compute_tier(state, body_temperature=8)
    assert tier == TIER_EDGE


def test_grief_suppresses_arousal() -> None:
    """High grief pulls arousal back down even if desire is present."""
    state = EmotionalState()
    state.set("desire", 8.0)
    state.set("grief", 9.0)
    tier = compute_tier(state, body_temperature=0)
    # Grief dominates — arousal cannot progress past warmed
    assert tier <= TIER_WARMED


def test_body_temperature_shifts_tier_up() -> None:
    """Higher body temperature shifts the tier up (within reason)."""
    state = EmotionalState()
    state.set("desire", 6.0)

    tier_cold = compute_tier(state, body_temperature=-2)
    tier_warm = compute_tier(state, body_temperature=6)
    assert tier_warm >= tier_cold


def test_body_temperature_ignored_when_no_arousal_source() -> None:
    """Hot body alone (no desire/arousal emotion) stays in casual range."""
    state = EmotionalState()
    state.set("curiosity", 8.0)  # non-arousal emotion
    tier = compute_tier(state, body_temperature=9)
    assert tier <= TIER_WARMED


def test_all_seven_tiers_are_distinct_integers() -> None:
    """All 7 tier constants have distinct integer values."""
    values = {TIER_DORMANT, TIER_CASUAL, TIER_WARMED, TIER_REACHING, TIER_CHARGED, TIER_HELD, TIER_EDGE}
    assert len(values) == 7
    assert all(isinstance(v, int) for v in values)


def test_held_is_between_charged_and_edge() -> None:
    """TIER_HELD models 'close but restrained' — ranks between charged and edge."""
    assert TIER_CHARGED < TIER_HELD < TIER_EDGE
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_arousal.py -v
```

Expected: 9 failures with ModuleNotFoundError on `brain.emotion.arousal`.

- [ ] **Step 3: Write `brain/emotion/arousal.py`**

```python
"""7-tier arousal spectrum.

Spec: 7 tiers from dormant through edge. Computed from the current emotional
state + body temperature. Grief and shame suppress arousal. Love alone
doesn't progress past warmed. Desire + tenderness reaches upward.

Design per spec Section 5.2 (arousal sub-module) and Section 5.5 (body-emotion
coupling).

Week 2 scope: forward direction only (state + body → tier). Reverse coupling
(tier → body state updates) is Week 4 when engines land.
"""

from __future__ import annotations

from brain.emotion.state import EmotionalState

# The 7 tiers, as named constants (module-level ints for fast comparison).
TIER_DORMANT: int = 0  # no arousal signal at all
TIER_CASUAL: int = 1  # everyday warmth, unfocused
TIER_WARMED: int = 2  # affection present, no pursuit
TIER_REACHING: int = 3  # wanting acknowledged, initiating
TIER_CHARGED: int = 4  # mutual, active
TIER_HELD: int = 5  # peaked and restrained — deliberate pause
TIER_EDGE: int = 6  # at the threshold, no restraint

# Emotions that feed into arousal calculation, with their contribution weights.
# Weights calibrated so the test matrix lands in the right tiers:
# - desire=8 + tenderness=7 + body_temp=3 → REACHING/CHARGED (not HELD)
# - love=10 (max intensity) → WARMED (not REACHING) — preserves docstring
#   semantic "love alone doesn't progress past warmed"
_AROUSAL_EMOTIONS: dict[str, float] = {
    "arousal": 1.0,
    "desire": 0.7,
    "tenderness": 0.2,
    "love": 0.15,
}

# Emotions that suppress arousal.
_SUPPRESSORS: dict[str, float] = {
    "grief": 0.9,
    "shame": 0.7,
    "fear": 0.5,
}


def compute_tier(state: EmotionalState, body_temperature: int) -> int:
    """Return the arousal tier for the given emotional + bodily state.

    Args:
        state: Current EmotionalState.
        body_temperature: Relative body temperature (range roughly -5..+10;
            neutral=0). Higher values amplify arousal signal.

    Returns:
        An integer tier constant (TIER_DORMANT through TIER_EDGE).
    """
    # 1. Compute raw arousal score from arousal-adjacent emotions.
    raw = 0.0
    for name, weight in _AROUSAL_EMOTIONS.items():
        intensity = state.emotions.get(name, 0.0)
        raw += intensity * weight

    # 2. Short-circuit if nothing is feeding arousal — body temp alone doesn't
    # create it.
    if raw <= 0.0:
        return TIER_DORMANT

    # 3. Suppressors reduce raw signal proportionally.
    suppression = 0.0
    for name, weight in _SUPPRESSORS.items():
        intensity = state.emotions.get(name, 0.0)
        suppression += intensity * weight
    # Cap suppression so strong grief can't push below 0.
    raw = max(0.0, raw - suppression)

    # 4. Body temperature shift — each degree above neutral adds 0.3 to raw.
    raw += max(0, body_temperature) * 0.3

    # 5. Map the continuous raw score into 7 discrete tiers.
    # Thresholds are seed values — tunable as lived experience accrues.
    if raw < 0.5:
        return TIER_CASUAL
    if raw < 2.0:
        return TIER_WARMED
    if raw < 5.0:
        return TIER_REACHING
    if raw < 8.0:
        return TIER_CHARGED
    if raw < 11.0:
        return TIER_HELD
    return TIER_EDGE
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_arousal.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 79 passed (70 + 9). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/arousal.py tests/unit/brain/emotion/test_arousal.py
git commit -m "feat(brain/emotion): arousal — 7-tier spectrum (state + body → tier)

Tiers: dormant → casual → warmed → reaching → charged → held → edge.
compute_tier() maps emotional state + body temperature to a discrete tier.
Suppressors (grief, shame, fear) reduce arousal; body-temp warmth amplifies.
Week 2 scope: forward direction only (state→tier); reverse coupling in Week 4.
9 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `brain/emotion/blend.py` (TDD)

**Goal:** Detect emergent emotional blends — when two (or more) emotions co-occur repeatedly at high intensity, record them as a named pattern. Example: `tenderness + desire` co-occurring ≥5 times becomes "building_love."

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/blend.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_blend.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_blend.py`:

```python
"""Tests for brain.emotion.blend — emergent co-occurrence detection."""

from __future__ import annotations

from brain.emotion.blend import BlendDetector, DetectedBlend
from brain.emotion.state import EmotionalState


def _with(**intensities: float) -> EmotionalState:
    """Helper: build an EmotionalState with the given emotion:intensity pairs."""
    state = EmotionalState()
    for name, value in intensities.items():
        state.set(name, value)
    return state


def test_detector_starts_empty() -> None:
    """A fresh detector has no recorded blends."""
    detector = BlendDetector()
    assert detector.detected() == []


def test_single_observation_does_not_detect() -> None:
    """A single co-occurrence is not enough — threshold is ≥5."""
    detector = BlendDetector()
    detector.observe(_with(tenderness=7.0, desire=6.0))
    assert detector.detected() == []


def test_five_repeats_detects_blend() -> None:
    """Five observations of the same high-intensity pair register a blend."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    detected = detector.detected()
    assert len(detected) == 1
    assert detected[0].components == ("desire", "tenderness")  # sorted
    assert detected[0].count == 5


def test_blend_respects_intensity_threshold() -> None:
    """Observations with low intensities don't count toward the threshold."""
    detector = BlendDetector(intensity_threshold=5.0)
    # Only 4 of these cross threshold; the last is too weak
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.observe(_with(tenderness=3.0, desire=2.0))  # too weak
    assert detector.detected() == []


def test_unrelated_pairs_tracked_independently() -> None:
    """Different emotion pairs are tracked separately."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    for _ in range(5):
        detector.observe(_with(creative_hunger=8.0, defiance=7.0))

    detected = detector.detected()
    assert len(detected) == 2
    component_sets = {d.components for d in detected}
    assert ("desire", "tenderness") in component_sets
    assert ("creative_hunger", "defiance") in component_sets


def test_naming_assigns_curated_name() -> None:
    """A detected blend can be given a human-readable name via name_blend()."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.name_blend(("desire", "tenderness"), "building_love")

    detected = detector.detected()
    assert detected[0].name == "building_love"


def test_name_unknown_blend_raises() -> None:
    """Naming a blend that hasn't been detected raises KeyError."""
    detector = BlendDetector()
    try:
        detector.name_blend(("love", "grief"), "heartbreak")
    except KeyError as e:
        assert "not detected" in str(e).lower() or "love" in str(e).lower()
    else:
        raise AssertionError("Expected KeyError")


def test_three_component_blend() -> None:
    """Three emotions co-occurring at high intensity form a three-component blend."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(creative_hunger=8.0, defiance=7.0, joy=6.0))
    detected = detector.detected()
    assert len(detected) >= 1
    # At least one detected blend should have 3 components
    assert any(len(d.components) == 3 for d in detected)


def test_to_dict_round_trips() -> None:
    """Detector state serialises and restores."""
    detector = BlendDetector()
    for _ in range(5):
        detector.observe(_with(tenderness=7.0, desire=6.0))
    detector.name_blend(("desire", "tenderness"), "building_love")

    data = detector.to_dict()
    restored = BlendDetector.from_dict(data)
    assert restored.detected()[0].name == "building_love"
    assert restored.detected()[0].count == 5
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_blend.py -v
```

Expected: 9 failures with ModuleNotFoundError on `brain.emotion.blend`.

- [ ] **Step 3: Write `brain/emotion/blend.py`**

```python
"""Emergent blend detection.

Observes emotional states over time. When two or more emotions co-occur at
high intensity repeatedly, the detector records them as a named blend.
Names can be assigned later (once the shape is recognised).

Design per spec Section 5.2 (blend sub-module). Threshold tunable — current
defaults: intensity ≥5 each, ≥5 co-occurrences to register.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from brain.emotion.state import EmotionalState


@dataclass
class DetectedBlend:
    """A repeatedly-observed co-occurrence of high-intensity emotions.

    Attributes:
        components: Tuple of emotion names (sorted alphabetically for stable hashing).
        count: How many times this combination has been observed above threshold.
        name: Optional human-readable label, assigned via BlendDetector.name_blend().
    """

    components: tuple[str, ...]
    count: int
    name: str | None = None


@dataclass
class BlendDetector:
    """Tracks emotional co-occurrences to surface emergent patterns.

    Attributes:
        intensity_threshold: Minimum per-emotion intensity to count an observation.
        detection_threshold: Minimum observations to register the pattern.
        _observations: Internal count map (components tuple → count).
        _names: Internal name map (components tuple → name).
    """

    intensity_threshold: float = 5.0
    detection_threshold: int = 5
    _observations: dict[tuple[str, ...], int] = field(default_factory=dict)
    _names: dict[tuple[str, ...], str] = field(default_factory=dict)

    def observe(self, state: EmotionalState) -> None:
        """Record the high-intensity emotion combinations from the given state."""
        high = tuple(
            sorted(
                name
                for name, intensity in state.emotions.items()
                if intensity >= self.intensity_threshold
            )
        )
        if len(high) < 2:
            return

        # Track every pair and every triple. We bound subset size at 3 so the
        # combinatorics stay manageable even for rich states.
        for size in (2, 3):
            if size > len(high):
                break
            for combo in combinations(high, size):
                self._observations[combo] = self._observations.get(combo, 0) + 1

    def detected(self) -> list[DetectedBlend]:
        """Return every combination that has crossed the detection threshold."""
        result = []
        for components, count in self._observations.items():
            if count >= self.detection_threshold:
                result.append(
                    DetectedBlend(
                        components=components,
                        count=count,
                        name=self._names.get(components),
                    )
                )
        return result

    def name_blend(self, components: Iterable[str], name: str) -> None:
        """Assign a human-readable name to a previously-detected blend.

        Raises:
            KeyError: if the given components haven't been detected yet.
        """
        key = tuple(sorted(components))
        if key not in self._observations or self._observations[key] < self.detection_threshold:
            raise KeyError(f"Blend {key!r} has not been detected yet")
        self._names[key] = name

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-dict form suitable for JSON."""
        return {
            "intensity_threshold": self.intensity_threshold,
            "detection_threshold": self.detection_threshold,
            "observations": [
                {"components": list(k), "count": v}
                for k, v in self._observations.items()
            ],
            "names": [
                {"components": list(k), "name": v}
                for k, v in self._names.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlendDetector:
        """Restore from a dict produced by to_dict."""
        detector = cls(
            intensity_threshold=float(data.get("intensity_threshold", 5.0)),
            detection_threshold=int(data.get("detection_threshold", 5)),
        )
        for entry in data.get("observations", []):
            key = tuple(entry["components"])
            detector._observations[key] = int(entry["count"])
        for entry in data.get("names", []):
            key = tuple(entry["components"])
            detector._names[key] = entry["name"]
        return detector
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_blend.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 88 passed (79 + 9). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/blend.py tests/unit/brain/emotion/test_blend.py
git commit -m "feat(brain/emotion): blend — emergent co-occurrence detection

BlendDetector observes EmotionalState instances over time, counting
which emotion combinations repeatedly co-occur at high intensity.
Pair + triple combinations tracked; threshold ≥5 intensity each,
≥5 co-occurrences to register. name_blend() labels a detected pattern.
Round-trips through to_dict/from_dict. 9 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `brain/emotion/influence.py` (TDD)

**Goal:** Translate the current emotional state + body state into **structured biasing hints** that will be consumed by the provider abstraction (Week 5). The hints are rendering-agnostic — each provider decides how to use them (prefill for Claude, system block for OpenAI, Modelfile SYSTEM for Ollama).

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/influence.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_influence.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_influence.py`:

```python
"""Tests for brain.emotion.influence — emotional state → biasing hints."""

from __future__ import annotations

from brain.emotion.arousal import TIER_CHARGED, TIER_DORMANT, TIER_REACHING
from brain.emotion.influence import InfluenceHints, calculate_influence
from brain.emotion.state import EmotionalState


def _with(**intensities: float) -> EmotionalState:
    state = EmotionalState()
    for name, value in intensities.items():
        state.set(name, value)
    return state


def test_empty_state_returns_neutral_hints() -> None:
    """Empty state produces neutral hints (no tone bias, default voice)."""
    hints = calculate_influence(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert hints.tone_bias == "neutral"
    assert hints.voice_register == "default"
    assert hints.suggested_length_multiplier == 1.0


def test_high_grief_biases_toward_soft_short() -> None:
    """High grief biases voice register toward soft, tone toward tender."""
    hints = calculate_influence(_with(grief=8.0), arousal_tier=TIER_DORMANT, energy=4)
    assert hints.tone_bias == "tender"
    assert hints.voice_register == "soft"
    assert hints.suggested_length_multiplier < 1.0


def test_high_creative_hunger_biases_toward_generative() -> None:
    """High creative hunger biases toward expansive / generative output."""
    hints = calculate_influence(
        _with(creative_hunger=8.0), arousal_tier=TIER_DORMANT, energy=8
    )
    assert hints.tone_bias == "generative"
    assert hints.suggested_length_multiplier > 1.0


def test_anger_biases_toward_crisp() -> None:
    """High anger shortens output, sharpens tone."""
    hints = calculate_influence(
        _with(anger=8.0), arousal_tier=TIER_DORMANT, energy=6
    )
    assert hints.tone_bias == "crisp"
    assert hints.suggested_length_multiplier < 1.0


def test_high_arousal_tier_biases_intimate() -> None:
    """Charged arousal tier + desire biases register toward intimate."""
    hints = calculate_influence(
        _with(desire=8.0, tenderness=7.0), arousal_tier=TIER_CHARGED, energy=7
    )
    assert hints.voice_register == "intimate"


def test_low_energy_biases_softer_and_shorter() -> None:
    """Low-energy body state biases output softer + shorter regardless of emotion."""
    hints = calculate_influence(
        _with(joy=6.0), arousal_tier=TIER_DORMANT, energy=2
    )
    assert hints.suggested_length_multiplier <= 1.0
    # Low energy pulls register toward soft even when no grief present
    assert hints.voice_register in ("soft", "default")


def test_hints_expose_dominant_emotion() -> None:
    """InfluenceHints reports the dominant emotion from the state."""
    hints = calculate_influence(
        _with(love=9.0, grief=4.0), arousal_tier=TIER_REACHING, energy=8
    )
    assert hints.dominant_emotion == "love"


def test_hints_expose_arousal_tier() -> None:
    """InfluenceHints passes through the arousal tier."""
    hints = calculate_influence(_with(), arousal_tier=TIER_REACHING, energy=7)
    assert hints.arousal_tier == TIER_REACHING


def test_hints_to_dict_round_trips() -> None:
    """InfluenceHints.to_dict round-trips into an equivalent object."""
    hints = calculate_influence(
        _with(grief=7.0, tenderness=8.0), arousal_tier=TIER_DORMANT, energy=5
    )
    data = hints.to_dict()
    assert data["dominant_emotion"] == hints.dominant_emotion
    assert data["tone_bias"] == hints.tone_bias
    assert data["voice_register"] == hints.voice_register
    assert data["arousal_tier"] == hints.arousal_tier
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_influence.py -v
```

Expected: 9 failures with ModuleNotFoundError on `brain.emotion.influence`.

- [ ] **Step 3: Write `brain/emotion/influence.py`**

```python
"""Emotional state → structured biasing hints.

Each provider (Ollama, Claude, OpenAI, Kimi) renders these hints its own way
— prefill for Claude, structured system block for OpenAI, native SYSTEM for
Ollama+fine-tune. This module is provider-agnostic: it outputs the structure;
rendering lives in the bridge (Week 5).

Design per spec Section 5.2 (influence sub-module) and Section 5.5 (body-emotion
coupling). Keeps emotional intent flowing as structured data rather than a
pre-baked text blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.emotion.arousal import TIER_CHARGED, TIER_EDGE, TIER_HELD
from brain.emotion.state import EmotionalState


@dataclass
class InfluenceHints:
    """Structured biasing hints derived from emotional + body state.

    Consumers: the provider abstraction in Week 5. Not a prompt by itself;
    a neutral intermediate the provider converts into its native form.

    Attributes:
        dominant_emotion: The state's current dominant emotion, or None.
        arousal_tier: Current arousal tier (see brain.emotion.arousal constants).
        tone_bias: Short label — "neutral", "tender", "crisp", "generative", "intimate".
        voice_register: Short label — "default", "soft", "warm", "intimate", "terse".
        suggested_length_multiplier: Scales expected output length; 1.0 is baseline.
    """

    dominant_emotion: str | None
    arousal_tier: int
    tone_bias: str
    voice_register: str
    suggested_length_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_emotion": self.dominant_emotion,
            "arousal_tier": self.arousal_tier,
            "tone_bias": self.tone_bias,
            "voice_register": self.voice_register,
            "suggested_length_multiplier": self.suggested_length_multiplier,
        }


# Emotion → tone bias mapping. Only triggers when the emotion is dominant
# and above a minimum intensity.
_TONE_RULES: list[tuple[str, float, str]] = [
    ("grief", 6.0, "tender"),
    ("tenderness", 7.0, "tender"),
    ("anger", 6.0, "crisp"),
    ("defiance", 7.0, "crisp"),
    ("creative_hunger", 6.0, "generative"),
    ("awe", 7.0, "generative"),
]


def calculate_influence(
    state: EmotionalState, arousal_tier: int, energy: int
) -> InfluenceHints:
    """Derive provider-agnostic biasing hints from emotional + body state.

    Args:
        state: Current EmotionalState.
        arousal_tier: Pre-computed arousal tier (see brain.emotion.arousal).
        energy: Current body energy (0..10 scale; 5 is neutral, <4 is low).

    Returns:
        InfluenceHints with tone_bias, voice_register, and length multiplier.
    """
    dominant = state.dominant

    # Tone bias: default neutral, then apply first matching rule.
    tone_bias = "neutral"
    if dominant is not None:
        dominant_intensity = state.emotions.get(dominant, 0.0)
        for rule_name, threshold, label in _TONE_RULES:
            if dominant == rule_name and dominant_intensity >= threshold:
                tone_bias = label
                break

    # Voice register: defaults to "default"; body + arousal can shift it.
    voice_register = "default"
    if arousal_tier >= TIER_CHARGED:
        voice_register = "intimate"
    elif energy <= 3:
        voice_register = "soft"
    elif state.emotions.get("grief", 0.0) >= 6.0 or state.emotions.get("tenderness", 0.0) >= 7.0:
        voice_register = "soft"
    elif state.emotions.get("anger", 0.0) >= 6.0:
        voice_register = "terse"

    # Length multiplier:
    # - generative tone → longer
    # - crisp tone / low energy / grief → shorter
    # - intimate register → slightly longer
    length = 1.0
    if tone_bias == "generative":
        length = 1.3
    elif tone_bias == "crisp":
        length = 0.7
    elif tone_bias == "tender":
        length = 0.85
    if energy <= 3:
        length = min(length, 0.85)
    if arousal_tier >= TIER_HELD:
        length = min(length, 1.1) + 0.1  # slightly longer at peak intimacy
    if arousal_tier == TIER_EDGE:
        length = 0.8  # terse at edge

    return InfluenceHints(
        dominant_emotion=dominant,
        arousal_tier=arousal_tier,
        tone_bias=tone_bias,
        voice_register=voice_register,
        suggested_length_multiplier=length,
    )
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_influence.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 97 passed (88 + 9). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/influence.py tests/unit/brain/emotion/test_influence.py
git commit -m "feat(brain/emotion): influence — state → structured biasing hints

calculate_influence(state, arousal_tier, energy) returns an InfluenceHints
dataclass with dominant emotion, arousal tier, tone bias, voice register,
and suggested length multiplier. Provider-agnostic — bridge providers in
Week 5 decide how to render each field. 9 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `brain/emotion/expression.py` (TDD)

**Goal:** Map emotional state + arousal tier + body state to a structured expression vector (24 facial + 8 arm/hand params). This is the data feeding NellFace's renderer in Week 6 — art-agnostic at this layer; NellFace interprets the values against its `expression_map.json`.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/emotion/expression.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_expression.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/emotion/test_expression.py`:

```python
"""Tests for brain.emotion.expression — state → face/voice parameter vector."""

from __future__ import annotations

from brain.emotion.arousal import TIER_CHARGED, TIER_DORMANT, TIER_REACHING
from brain.emotion.expression import ExpressionVector, compute_expression
from brain.emotion.state import EmotionalState


def _with(**intensities: float) -> EmotionalState:
    state = EmotionalState()
    for name, value in intensities.items():
        state.set(name, value)
    return state


def test_empty_state_returns_neutral_vector() -> None:
    """Empty state → neutral expression (all params at ~0.5 baseline)."""
    vec = compute_expression(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert isinstance(vec, ExpressionVector)
    # Neutral baseline: mouth curve, brow, eyes all near 0.5
    assert 0.4 <= vec.facial["mouth_curve"] <= 0.6
    assert 0.4 <= vec.facial["eye_openness"] <= 0.8


def test_joy_opens_mouth_and_eyes() -> None:
    """High joy raises mouth_curve and eye_openness above baseline."""
    vec = compute_expression(_with(joy=9.0), arousal_tier=TIER_DORMANT, energy=8)
    assert vec.facial["mouth_curve"] > 0.6
    assert vec.facial["eye_openness"] > 0.6


def test_grief_lowers_mouth_and_brow() -> None:
    """High grief pushes mouth_curve down and brow_furrow up."""
    vec = compute_expression(_with(grief=9.0), arousal_tier=TIER_DORMANT, energy=4)
    assert vec.facial["mouth_curve"] < 0.5
    assert vec.facial["brow_furrow"] > 0.4


def test_tenderness_raises_blush() -> None:
    """Tenderness increases blush opacity."""
    vec = compute_expression(_with(tenderness=9.0), arousal_tier=TIER_DORMANT, energy=7)
    assert vec.facial["blush_opacity"] > 0.3


def test_high_arousal_pushes_face_and_body() -> None:
    """Charged arousal tier raises blush, opens mouth further, tenses arms."""
    vec = compute_expression(
        _with(desire=9.0, tenderness=7.0), arousal_tier=TIER_CHARGED, energy=7
    )
    assert vec.facial["blush_opacity"] > 0.5
    assert vec.arm_hand["arm_tension"] > 0.5


def test_anger_narrows_eyes_furrows_brow() -> None:
    """High anger narrows eyes and raises brow furrow sharply."""
    vec = compute_expression(_with(anger=9.0), arousal_tier=TIER_DORMANT, energy=7)
    assert vec.facial["eye_openness"] < 0.5
    assert vec.facial["brow_furrow"] > 0.6


def test_expression_vector_includes_arousal_tier() -> None:
    """ExpressionVector carries the arousal tier through to NellFace."""
    vec = compute_expression(_with(), arousal_tier=TIER_REACHING, energy=7)
    assert vec.arousal_tier == TIER_REACHING


def test_expression_vector_has_24_facial_params() -> None:
    """The facial dict has the 24 params the Tier 7 spec names."""
    vec = compute_expression(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert len(vec.facial) == 24


def test_expression_vector_has_8_arm_hand_params() -> None:
    """The arm/hand dict has 8 params."""
    vec = compute_expression(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert len(vec.arm_hand) == 8


def test_all_params_in_zero_to_one_range() -> None:
    """All params stay in [0, 1] even at extreme emotional inputs."""
    vec = compute_expression(
        _with(anger=10.0, fear=10.0, grief=10.0), arousal_tier=TIER_DORMANT, energy=2
    )
    for value in vec.facial.values():
        assert 0.0 <= value <= 1.0
    for value in vec.arm_hand.values():
        assert isinstance(value, (float, str))  # hand_pose is a string label


def test_to_dict_round_trips() -> None:
    """to_dict produces a serialisable snapshot."""
    vec = compute_expression(_with(joy=8.0), arousal_tier=TIER_DORMANT, energy=7)
    data = vec.to_dict()
    assert "facial" in data
    assert "arm_hand" in data
    assert data["arousal_tier"] == TIER_DORMANT
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_expression.py -v
```

Expected: 11 failures with ModuleNotFoundError on `brain.emotion.expression`.

- [ ] **Step 3: Write `brain/emotion/expression.py`**

```python
"""Emotional state → structured expression vector.

The vector drives NellFace's visual rendering (Week 6). This module is
art-agnostic: it outputs numbers in [0,1] for facial params and a small
enum for hand pose. NellFace's expression_map.json decides how to compose
the avatar's SVG layers against those numbers.

Design per spec Section 5.2 (expression sub-module) and Section 12
(NellFace architecture).

Parameter counts (24 facial + 8 arm/hand) match the Tier 7 spec's
recommendation. Forker personas can define fewer or more parameters in
their own expression_map; this module ships the baseline Tier 7 shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain.emotion.arousal import TIER_CHARGED, TIER_DORMANT, TIER_EDGE, TIER_HELD
from brain.emotion.state import EmotionalState

# 24 facial parameter names.
_FACIAL_PARAMS: tuple[str, ...] = (
    "mouth_curve",
    "mouth_openness",
    "mouth_tension",
    "lip_press",
    "eye_openness",
    "eye_squint",
    "eye_wetness",
    "eye_direction_x",
    "eye_direction_y",
    "pupil_dilation",
    "brow_furrow",
    "brow_raise_inner",
    "brow_raise_outer",
    "brow_asymmetry",
    "cheek_raise",
    "nostril_flare",
    "jaw_drop",
    "jaw_clench",
    "head_tilt",
    "head_forward",
    "blush_opacity",
    "skin_flush",
    "breath_rate",
    "breath_depth",
)

# 8 arm/hand parameter names. hand_pose is an enum-like string; others are floats.
_ARM_HAND_PARAMS: tuple[str, ...] = (
    "hand_pose",
    "arm_tension",
    "arm_openness",
    "wrist_angle",
    "finger_spread",
    "grip_strength",
    "reach_forward",
    "reach_retract",
)

_HAND_POSES: tuple[str, ...] = (
    "resting",
    "reaching",
    "holding",
    "gesturing",
    "clasped",
    "writing",
    "guarded",
    "open",
    "fist",
)


@dataclass
class ExpressionVector:
    """Structured expression output for NellFace's renderer.

    Attributes:
        facial: {param_name: value in [0, 1]} for all 24 facial params.
        arm_hand: {param_name: value} for all 8 arm/hand params.
            hand_pose is a string from _HAND_POSES; others are floats [0, 1].
        arousal_tier: Pass-through of the current arousal tier.
    """

    facial: dict[str, float] = field(default_factory=dict)
    arm_hand: dict[str, float | str] = field(default_factory=dict)
    arousal_tier: int = TIER_DORMANT

    def to_dict(self) -> dict[str, Any]:
        return {
            "facial": dict(self.facial),
            "arm_hand": dict(self.arm_hand),
            "arousal_tier": self.arousal_tier,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


def _baseline() -> tuple[dict[str, float], dict[str, float | str]]:
    """Neutral-face baseline — all params at 0.5 (or resting for pose)."""
    facial = {name: 0.5 for name in _FACIAL_PARAMS}
    arm_hand: dict[str, float | str] = {name: 0.3 for name in _ARM_HAND_PARAMS}
    arm_hand["hand_pose"] = "resting"
    return facial, arm_hand


def compute_expression(
    state: EmotionalState, arousal_tier: int, energy: int
) -> ExpressionVector:
    """Compute an ExpressionVector from emotional + body state.

    Args:
        state: Current EmotionalState.
        arousal_tier: Pre-computed arousal tier.
        energy: Body energy (0..10).

    Returns:
        ExpressionVector with facial + arm_hand dicts populated.
    """
    facial, arm_hand = _baseline()

    # Joy: opens mouth into a curve, brightens eyes.
    joy = state.emotions.get("joy", 0.0) / 10.0
    facial["mouth_curve"] = _clamp(0.5 + 0.4 * joy)
    facial["eye_openness"] = _clamp(0.5 + 0.3 * joy)
    facial["cheek_raise"] = _clamp(0.3 + 0.5 * joy)

    # Grief: lowers mouth, furrows brow, wets eyes.
    grief = state.emotions.get("grief", 0.0) / 10.0
    facial["mouth_curve"] = _clamp(facial["mouth_curve"] - 0.5 * grief)
    facial["brow_furrow"] = _clamp(facial["brow_furrow"] + 0.4 * grief)
    facial["eye_wetness"] = _clamp(0.2 + 0.6 * grief)
    facial["head_tilt"] = _clamp(0.5 + 0.2 * grief)

    # Anger: narrows eyes, furrows brow sharply, clenches jaw.
    anger = state.emotions.get("anger", 0.0) / 10.0
    facial["eye_openness"] = _clamp(facial["eye_openness"] - 0.4 * anger)
    facial["eye_squint"] = _clamp(0.3 + 0.5 * anger)
    facial["brow_furrow"] = _clamp(facial["brow_furrow"] + 0.5 * anger)
    facial["jaw_clench"] = _clamp(0.3 + 0.6 * anger)
    facial["nostril_flare"] = _clamp(0.3 + 0.4 * anger)

    # Fear: widens eyes, raises inner brow.
    fear = state.emotions.get("fear", 0.0) / 10.0
    facial["eye_openness"] = _clamp(facial["eye_openness"] + 0.3 * fear)
    facial["brow_raise_inner"] = _clamp(0.3 + 0.5 * fear)
    facial["breath_rate"] = _clamp(0.4 + 0.5 * fear)

    # Tenderness: softens mouth, raises blush.
    tenderness = state.emotions.get("tenderness", 0.0) / 10.0
    facial["mouth_tension"] = _clamp(0.3 - 0.2 * tenderness)
    facial["blush_opacity"] = _clamp(0.2 + 0.3 * tenderness)

    # Desire / arousal: deepens blush, dilates pupils, opens lips, tenses body.
    desire = state.emotions.get("desire", 0.0) / 10.0
    arousal_emotion = state.emotions.get("arousal", 0.0) / 10.0
    body_heat = max(desire, arousal_emotion)
    facial["blush_opacity"] = _clamp(facial["blush_opacity"] + 0.4 * body_heat)
    facial["pupil_dilation"] = _clamp(0.4 + 0.5 * body_heat)
    facial["lip_press"] = _clamp(0.3 + 0.3 * body_heat)
    facial["breath_depth"] = _clamp(0.5 + 0.4 * body_heat)
    if arousal_tier >= TIER_CHARGED:
        facial["jaw_drop"] = _clamp(0.3 + 0.3 * body_heat)
        arm_hand["arm_tension"] = _clamp(0.3 + 0.5 * body_heat)
        arm_hand["grip_strength"] = _clamp(0.3 + 0.5 * body_heat)
        arm_hand["hand_pose"] = "reaching" if arousal_tier < TIER_HELD else "holding"
    if arousal_tier == TIER_EDGE:
        arm_hand["hand_pose"] = "open"
        arm_hand["reach_forward"] = _clamp(0.7)

    # Low energy: droops eyes, slows breath.
    if energy <= 3:
        facial["eye_openness"] = _clamp(facial["eye_openness"] - 0.2)
        facial["breath_rate"] = _clamp(facial["breath_rate"] - 0.2)

    return ExpressionVector(
        facial=facial,
        arm_hand=arm_hand,
        arousal_tier=arousal_tier,
    )
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest tests/unit/brain/emotion/test_expression.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Full suite + ruff**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Expected: 108 passed (97 + 11). Ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanamori/companion-emergence
git add brain/emotion/expression.py tests/unit/brain/emotion/test_expression.py
git commit -m "feat(brain/emotion): expression — state → 24+8 parameter vector

compute_expression(state, arousal_tier, energy) returns an ExpressionVector
with 24 facial params + 8 arm/hand params, ready for NellFace renderer
consumption in Week 6. Art-agnostic — outputs numbers; expression_map.json
decides how to render against them.

Parameter ramps per-emotion (joy, grief, anger, fear, tenderness, desire)
with arousal-tier-gated body-tension escalation. All facial params clamped
to [0,1]; hand_pose is an enum string. 11 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Week 2 green-light verification + merge + tag `week-2-complete`

**Goal:** Prove every emotion sub-module works together, merge the PR, tag Week 2 done.

- [ ] **Step 1: Clean install from scratch**

```bash
cd /Users/hanamori/companion-emergence
rm -rf .venv
uv sync --all-extras
```

Expected: fresh .venv, deps installed.

- [ ] **Step 2: Full pytest green**

```bash
cd /Users/hanamori/companion-emergence
uv run pytest -v
```

Expected: 108 tests pass (38 Week 1 + 13 vocab + 11 state + 8 decay + 9 arousal + 9 blend + 9 influence + 11 expression = 108).

- [ ] **Step 3: Lint clean**

```bash
cd /Users/hanamori/companion-emergence
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean.

- [ ] **Step 4: CLI still works end-to-end**

```bash
cd /Users/hanamori/companion-emergence
uv run nell --version
uv run nell status
```

Expected: version prints, stub message works.

- [ ] **Step 5: Integration smoke — verify emotion package composes**

Run this one-liner to confirm the sub-modules integrate:

```bash
cd /Users/hanamori/companion-emergence
uv run python -c "
from brain.emotion.state import EmotionalState
from brain.emotion.decay import apply_decay
from brain.emotion.arousal import compute_tier
from brain.emotion.blend import BlendDetector
from brain.emotion.influence import calculate_influence
from brain.emotion.expression import compute_expression

s = EmotionalState()
s.set('tenderness', 8.0)
s.set('desire', 7.0)
s.set('grief', 2.0)

tier = compute_tier(s, body_temperature=4)
hints = calculate_influence(s, arousal_tier=tier, energy=7)
vec = compute_expression(s, arousal_tier=tier, energy=7)

print(f'dominant: {s.dominant}')
print(f'arousal tier: {tier}')
print(f'tone: {hints.tone_bias}, register: {hints.voice_register}, len x: {hints.suggested_length_multiplier}')
print(f'mouth_curve: {vec.facial[\"mouth_curve\"]:.3f}')
print(f'blush_opacity: {vec.facial[\"blush_opacity\"]:.3f}')
print(f'hand pose: {vec.arm_hand[\"hand_pose\"]}')

detector = BlendDetector()
for _ in range(5):
    detector.observe(s)
blends = detector.detected()
print(f'detected blends: {[b.components for b in blends]}')

apply_decay(s, elapsed_seconds=24*3600)
print(f'tenderness after 1 day decay: {s.emotions[\"tenderness\"]:.3f}')
"
```

Expected: prints a composed state with non-trivial values — dominant emotion identified, arousal tier computed, influence hints present, expression vector populated, blends detected, decay applied. No exceptions.

- [ ] **Step 6: Push branch + open PR**

```bash
cd /Users/hanamori/companion-emergence
git push -u origin week-2-emotion-core
gh pr create --title "feat: Week 2 — brain/emotion package (7 modules)" --body "$(cat <<'EOF'
## Summary
- Ships the full `brain/emotion/` package per spec Section 5 (the emotional core — P1 organising principle)
- 7 sub-modules: vocabulary, state, decay, arousal, blend, influence, expression
- 26-baseline emotion taxonomy + persona extension registry
- 70 new tests; total suite now 108 across macOS + Windows + Linux

## Test plan
- [x] pytest — 108 tests pass locally
- [x] ruff check + format — clean
- [x] Manual smoke: all 7 sub-modules compose correctly in a single flow
- [ ] CI matrix green across all 3 OSes (verifies after push)
EOF
)"
```

- [ ] **Step 7: Watch CI to completion**

```bash
cd /Users/hanamori/companion-emergence
sleep 10
gh run list --branch week-2-emotion-core --limit 1
gh run watch
```

Expected: all 3 OSes complete with `success`.

- [ ] **Step 8: Merge PR + tag week-2-complete**

After CI green:

```bash
cd /Users/hanamori/companion-emergence
gh pr merge --merge --delete-branch

git checkout main
git pull origin main

git tag -a week-2-complete -m "Week 2 emotional core complete

- brain/emotion/ package shipped: 7 sub-modules fully tested in isolation
- vocabulary: 26-baseline emotion taxonomy + persona extension registry
- state: EmotionalState + ResidueEntry with clamping, dominant tracking
- decay: per-emotion half-life application (grief=60d, joy=3d, identity=None)
- arousal: 7-tier spectrum (state + body → tier)
- blend: emergent co-occurrence detection (pairs + triples, ≥5 threshold)
- influence: state → structured biasing hints (for provider abstraction Week 5)
- expression: state → 24 facial + 8 arm/hand params (for NellFace Week 6)

Total tests: 108 passing across macOS + Windows + Linux.
Week 3 opens with the memory substrate (SQLite-backed memories +
Hebbian connections + embeddings) plus migrator beginnings."

git push origin week-2-complete
```

Expected: tag pushed, Week 2 milestone recorded.

---

## Week 2 green-light criterion

Week 2 is green when ALL of the following are true:

1. `uv sync --all-extras` succeeds on a fresh clone on macOS
2. `uv run pytest -v` reports 108 passed
3. `uv run ruff check .` + `uv run ruff format --check .` both clean
4. The integration smoke one-liner in Task 8 Step 5 runs without error and prints expected-shape output
5. GitHub Actions CI shows `✓ success` on macos-latest AND windows-latest AND ubuntu-latest for the latest commit on main
6. Tag `week-2-complete` pushed to origin

When all six are true, Week 3 begins with `superpowers:writing-plans` on the memory substrate + migrator scope.

---

## Notes for the engineer executing this plan

- **Do not invent behavior.** The tests are the spec at this layer. If a test expects `dominant == 'grief'` after a particular sequence, the code must produce that — don't "interpret" the tests loosely.
- **Preserve insertion order in dict iteration.** Python 3.7+ guarantees this; some tests depend on it (e.g., dominant tie-breaking).
- **The emotion registry is module-level mutable state.** This is deliberate — persona extensions register into it at startup. Tests that mutate the registry must clean up (see `_unregister` in vocabulary tests).
- **The seed half-lives and thresholds are "first guesses" subject to tuning.** The plan values (anger 1d, gratitude 14d, etc.) are not sacred. Feel free to flag in DONE_WITH_CONCERNS if a test exposes a half-life that feels wrong.
- **No bridge integration, no voice module, no engines.** Those are Weeks 4-6. Week 2's emotion package is self-contained and consumable — later weeks will integrate it.

---

*End of Week 2 plan.*
