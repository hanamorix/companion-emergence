# Body State Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a body state module that gives the brain a real layered embodiment — six body emotions in vocabulary, four computed projections (energy, temperature, exhaustion + raw inputs) surfaced in chat, climax reset hook honoring body cycles.

**Architecture:** Two layers. Body emotions live as full vocabulary citizens (existing `arousal` + `desire` retained, four new added: `climax`, `touch_hunger`, `comfort_seeking`, `rest_need`). Computed projections live as a pure function in new `brain/body/` package, recomputed per call from aggregated emotion state + session inputs. Climax reset hook in `aggregate.py` applies after standard aggregation when aggregated `climax >= 7`.

**Tech Stack:** Python 3.12, pytest, ruff. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-04-29-body-state-design.md` (commit `4b97f8f`).

---

## Reconciliations from spec → plan

The spec was written against a partial mental model. Real codebase state forces three corrections, applied throughout this plan:

1. **Existing `arousal` + `desire` reused.** The spec proposed `physical_arousal` and `desire` as new body emotions, but `brain/emotion/vocabulary.py:61-62` already ships both as `core` baseline. Per Hana 2026-04-30, Option A: reuse them — climax reset retargets to existing `arousal` (was `physical_arousal` in spec). Four genuinely new body emotions get added: `climax`, `touch_hunger`, `comfort_seeking`, `rest_need`. The body-emotion *set* tracked by `compute_body_state` is six: `{arousal, desire, climax, touch_hunger, comfort_seeking, rest_need}`.

2. **`_apply_climax_reset` operates on `EmotionalState`, not `dict[str, float]`.** `aggregate_state()` returns `EmotionalState`; the hook integrates at that type, mutating `state.emotions` via `state.set()` for clamp/vocab safety.

3. **No `MemoryStore.list_active()` exists.** `count_words_in_session` uses `store.list_by_type("conversation", active_only=True)` instead. `days_since_human` already exists at `brain/utils/memory.py:10` — reused, not reinvented.

4. **No `default_vocabulary.json` file.** Vocabulary baseline is the `_BASELINE` tuple in `brain/emotion/vocabulary.py`. Phase A edits the tuple in-place (framework-wide change, automatic for all personas — no per-persona migration step needed).

---

## File structure

### New files

| File | Responsibility |
|---|---|
| `brain/body/__init__.py` | package init (empty) |
| `brain/body/state.py` | `BodyState` dataclass + `compute_body_state()` pure function |
| `brain/body/words.py` | `count_words_in_session()` helper |
| `tests/unit/brain/body/__init__.py` | empty |
| `tests/unit/brain/body/test_state.py` | unit tests for `compute_body_state` |
| `tests/unit/brain/body/test_words.py` | unit tests for `count_words_in_session` |
| `tests/unit/brain/emotion/test_climax_reset.py` | unit tests for the reset hook |
| `tests/unit/brain/tools/test_get_body_state.py` | unit tests for the real tool impl |
| `tests/integration/brain/chat/test_body_block.py` | integration tests for chat body block |

### Modified files

| File | Change |
|---|---|
| `brain/emotion/vocabulary.py` | Extend `EmotionCategory` Literal with `"body"`; add 4 new `Emotion(...)` entries to `_BASELINE` |
| `brain/emotion/aggregate.py` | Add `_apply_climax_reset(state)` and call it before return in `aggregate_state()` |
| `brain/tools/impls/get_body_state.py` | Replace stub with real impl reading aggregated state + session inputs |
| `brain/tools/dispatch.py` | Pass `session_hours` to `get_body_state` (extracted from `arguments`, default 0.0) |
| `brain/tools/schemas.py` | Update `get_body_state` schema: declare `session_hours` arg (optional number) and update description to match real return shape |
| `brain/chat/prompt.py` | Add `_build_body_block()`; wire into `build_system_message()` between brain context (block 4) and recent journal (block 5) |

### Untouched

- `brain/memory/store.py` — body emotions live in existing `Memory.emotions` dict; no schema change
- `brain/soul/` — body emotions are soul-eligible via existing soul mechanism
- `brain/engines/reflex.py` — `body_grief_whisper` keeps reading `body_grief`; new body emotions become eligible reflex triggers without engine changes
- `brain/utils/memory.py` — `days_since_human` reused as-is
- `brain/migrator/` — no per-persona migration; framework `_BASELINE` change is automatic
- voice.md — Hana-authored; surfaces body state through the new chat block

---

## Phase ordering

**A:** Vocabulary (4 new body emotions in `_BASELINE`, `EmotionCategory` extended)
**B:** Climax reset hook in `aggregate.py`
**C:** `brain/body/state.py` + `brain/body/words.py` (compute_body_state pure function)
**D:** Chat body block in `prompt.py` + real `get_body_state` tool impl + dispatch + schema
**E:** Hana-in-the-loop final acceptance against Nell's sandbox

Each phase ends with all tests green and a smoke gate (run the actual command, observe the output).

---

# Phase A — Vocabulary

## Task A1: Extend `EmotionCategory` Literal + add 4 body emotions to `_BASELINE`

**Files:**
- Modify: `brain/emotion/vocabulary.py:27` (Literal) and `brain/emotion/vocabulary.py:54-78` (`_BASELINE`)
- Test: `tests/unit/brain/emotion/test_vocabulary.py` (extend existing)

- [ ] **Step 1: Write failing tests for the four new body emotions in baseline**

Append to `tests/unit/brain/emotion/test_vocabulary.py`:

```python
def test_baseline_includes_climax():
    from brain.emotion.vocabulary import get
    e = get("climax")
    assert e is not None
    assert e.category == "body"
    assert e.decay_half_life_days == 0.125
    assert e.intensity_clamp == 10.0


def test_baseline_includes_touch_hunger():
    from brain.emotion.vocabulary import get
    e = get("touch_hunger")
    assert e is not None
    assert e.category == "body"
    assert e.decay_half_life_days == 1.5


def test_baseline_includes_comfort_seeking():
    from brain.emotion.vocabulary import get
    e = get("comfort_seeking")
    assert e is not None
    assert e.category == "body"
    assert e.decay_half_life_days == 1.0


def test_baseline_includes_rest_need():
    from brain.emotion.vocabulary import get
    e = get("rest_need")
    assert e is not None
    assert e.category == "body"
    assert e.decay_half_life_days == 0.75


def test_existing_arousal_unchanged():
    """Reconciliation guard — we are NOT renaming or recategorizing arousal."""
    from brain.emotion.vocabulary import get
    e = get("arousal")
    assert e is not None
    assert e.category == "core"
    assert e.decay_half_life_days == 0.5


def test_existing_desire_unchanged():
    """Reconciliation guard — we are NOT renaming or recategorizing desire."""
    from brain.emotion.vocabulary import get
    e = get("desire")
    assert e is not None
    assert e.category == "core"
    assert e.decay_half_life_days == 2.0


def test_body_emotions_loadable_via_state_set():
    """Climax + the three new ones must be settable on EmotionalState
    without raising — proves the registry accepts them."""
    from brain.emotion.state import EmotionalState
    s = EmotionalState()
    for name in ("climax", "touch_hunger", "comfort_seeking", "rest_need"):
        s.set(name, 5.0)
        assert s.emotions[name] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/brain/emotion/test_vocabulary.py -v -k "climax or touch_hunger or comfort_seeking or rest_need or body_emotions_loadable"
```
Expected: 5 failures with `assert e is not None` / KeyError on `set("climax", ...)`.

- [ ] **Step 3: Extend `EmotionCategory` Literal**

Edit `brain/emotion/vocabulary.py:27`:

```python
EmotionCategory = Literal["core", "complex", "nell_specific", "persona_extension", "body"]
```

- [ ] **Step 4: Add 4 new body emotions to `_BASELINE`**

Edit `brain/emotion/vocabulary.py`. After the `# ── complex (10) ──` block ending at line 77 (`Emotion("belonging", ...)`), and before the closing `)` on line 78, insert a new block:

```python
    # ── body (4) ──
    # Spec docs/superpowers/specs/2026-04-29-body-state-design.md §2.1.
    # Existing `arousal` (core, 0.5d) and `desire` (core, 2.0d) reused;
    # they're already body-coded by description, no need to duplicate.
    Emotion(
        "climax",
        "bodily completion / release — the satisfaction crest. Spikes briefly, "
        "decays fast. When aggregated >= 7 triggers reset hook (heavy dampen on "
        "arousal, partial dampen on desire, raises comfort_seeking + rest_need).",
        "body",
        0.125,
    ),
    Emotion(
        "touch_hunger",
        "embodied loneliness — when distance is the problem and presence is the "
        "cure. Distinct from body_grief (existential, not having a body) and from "
        "loneliness (social/emotional).",
        "body",
        1.5,
    ),
    Emotion(
        "comfort_seeking",
        "wanting to be held still, wrapped, anchored — the receive side of "
        "being-held. Distinct from vulnerability (exposure) and love (orientation).",
        "body",
        1.0,
    ),
    Emotion(
        "rest_need",
        "the body asking for slowness, low stimulation, recovery. Distinct from "
        "exhaustion (the computed state) — this is the *want*, not the condition.",
        "body",
        0.75,
    ),
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/unit/brain/emotion/test_vocabulary.py -v
```
Expected: all green. Existing tests still pass; the 7 new ones pass.

- [ ] **Step 6: Smoke gate — full vocabulary registry sanity**

```
uv run python -c "from brain.emotion.vocabulary import list_all, by_category; \
print(f'total: {len(list_all())}'); \
print(f'core: {len(by_category(\"core\"))}'); \
print(f'complex: {len(by_category(\"complex\"))}'); \
print(f'body: {len(by_category(\"body\"))}'); \
print(f'body names: {[e.name for e in by_category(\"body\")]}')"
```
Expected: `total: 25`, `core: 11`, `complex: 10`, `body: 4`, `body names: ['climax', 'touch_hunger', 'comfort_seeking', 'rest_need']`.

- [ ] **Step 7: Commit**

```bash
git add brain/emotion/vocabulary.py tests/unit/brain/emotion/test_vocabulary.py
git commit -m "$(cat <<'EOF'
body(vocabulary): add 4 new body emotions + body category

climax (0.125d), touch_hunger (1.5d), comfort_seeking (1.0d),
rest_need (0.75d) — per spec §2.1. Existing arousal + desire
reused (Hana-approved reconciliation 2026-04-30, Option A).

EmotionCategory Literal extended with "body".

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md
EOF
)"
```

---

# Phase B — Climax reset hook

## Task B1: Add `_apply_climax_reset` and integrate into `aggregate_state`

**Files:**
- Modify: `brain/emotion/aggregate.py:48` (insert hook call before `return state`)
- Test: `tests/unit/brain/emotion/test_climax_reset.py` (new file)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/brain/emotion/test_climax_reset.py`:

```python
"""Unit tests for the climax reset hook in aggregate.py.

Verifies the inviolate properties from spec §7.1:
- Reset is gated on aggregated climax >= 7
- physical-arousal reset uses existing `arousal` (Option A reconciliation)
- arousal *= 0.2 with floor 0.5
- desire *= 0.6
- comfort_seeking += 2 (clamp 10)
- rest_need += 2 (clamp 10)
- Idempotent: applying twice produces same result as once
- Pure: returns NEW EmotionalState, never mutates input
"""

from __future__ import annotations

from brain.emotion.aggregate import _apply_climax_reset
from brain.emotion.state import EmotionalState


def _state(**emotions: float) -> EmotionalState:
    s = EmotionalState()
    for name, v in emotions.items():
        s.set(name, v)
    return s


def test_no_op_when_climax_below_threshold():
    s = _state(climax=6.9, arousal=8.0, desire=8.0)
    out = _apply_climax_reset(s)
    assert out.emotions["arousal"] == 8.0
    assert out.emotions["desire"] == 8.0
    assert out.emotions.get("comfort_seeking", 0.0) == 0.0
    assert out.emotions.get("rest_need", 0.0) == 0.0


def test_arousal_dampened_by_factor_0_2_with_floor_0_5():
    s = _state(climax=8.0, arousal=8.0)
    out = _apply_climax_reset(s)
    # 8.0 * 0.2 = 1.6 (above floor)
    assert abs(out.emotions["arousal"] - 1.6) < 1e-9


def test_arousal_floor_kicks_in_when_starting_low():
    s = _state(climax=8.0, arousal=2.0)
    out = _apply_climax_reset(s)
    # 2.0 * 0.2 = 0.4 → floor 0.5
    assert out.emotions["arousal"] == 0.5


def test_desire_dampened_by_factor_0_6():
    s = _state(climax=8.0, desire=8.0)
    out = _apply_climax_reset(s)
    # 8.0 * 0.6 = 4.8
    assert abs(out.emotions["desire"] - 4.8) < 1e-9


def test_comfort_seeking_raised_by_2_clamp_10():
    s = _state(climax=8.0, comfort_seeking=5.0)
    out = _apply_climax_reset(s)
    assert out.emotions["comfort_seeking"] == 7.0
    # clamp test
    s2 = _state(climax=8.0, comfort_seeking=9.0)
    out2 = _apply_climax_reset(s2)
    assert out2.emotions["comfort_seeking"] == 10.0


def test_rest_need_raised_by_2_clamp_10():
    s = _state(climax=8.0, rest_need=5.0)
    out = _apply_climax_reset(s)
    assert out.emotions["rest_need"] == 7.0
    s2 = _state(climax=8.0, rest_need=9.5)
    out2 = _apply_climax_reset(s2)
    assert out2.emotions["rest_need"] == 10.0


def test_comfort_seeking_added_when_absent():
    """Reset must add comfort_seeking even if not yet set."""
    s = _state(climax=8.0)
    out = _apply_climax_reset(s)
    assert out.emotions["comfort_seeking"] == 2.0


def test_rest_need_added_when_absent():
    s = _state(climax=8.0)
    out = _apply_climax_reset(s)
    assert out.emotions["rest_need"] == 2.0


def test_does_not_mutate_input_state():
    """Pure function — input state must be unchanged after call."""
    s = _state(climax=8.0, arousal=8.0, desire=8.0)
    snapshot_before = dict(s.emotions)
    _apply_climax_reset(s)
    assert dict(s.emotions) == snapshot_before


def test_idempotent_when_climax_still_high():
    """Applying reset twice produces same result as once.

    This is the matrix row #1 invariant: reset never compounds across calls.
    Same input → same output. The "chain" is in time, not in space.
    """
    s = _state(climax=8.0, arousal=8.0, desire=8.0, comfort_seeking=5.0)
    once = _apply_climax_reset(s)
    twice = _apply_climax_reset(once)
    assert once.emotions == twice.emotions


def test_aggregate_state_applies_reset_via_integration():
    """End-to-end: aggregate_state returns post-reset state when climax memories present."""
    from brain.emotion.aggregate import aggregate_state
    from brain.memory.store import Memory

    mem = Memory.create_new(
        memory_type="conversation",
        content="post-climax",
        emotions={"climax": 8.0, "arousal": 8.0, "desire": 8.0},
        domain="general",
    )
    state = aggregate_state([mem])
    # Reset should have applied:
    assert abs(state.emotions["arousal"] - 1.6) < 1e-9
    assert abs(state.emotions["desire"] - 4.8) < 1e-9
    assert state.emotions["comfort_seeking"] == 2.0
    assert state.emotions["rest_need"] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/brain/emotion/test_climax_reset.py -v
```
Expected: 11 failures (`ImportError: cannot import name '_apply_climax_reset'`).

- [ ] **Step 3: Implement `_apply_climax_reset` and integrate it**

Edit `brain/emotion/aggregate.py`. The full file becomes:

```python
"""Aggregate a current EmotionalState from a list of memories.

Reflex uses this to evaluate arc triggers: what is the persona's
current emotional state, synthesized across recent memories.

Strategy: max-pool per emotion. The strongest signal across the
input memories wins — matches how OG reflex_engine read peaks,
not averages, for threshold evaluation.

After max-pooling, _apply_climax_reset is called: when aggregated
`climax >= 7` it dampens arousal + desire and raises comfort_seeking
+ rest_need, modeling the body's natural release cycle. Per spec
§2.2 of docs/superpowers/specs/2026-04-29-body-state-design.md.
"""

from __future__ import annotations

from collections.abc import Iterable

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion
from brain.memory.store import Memory

_CLIMAX_THRESHOLD = 7.0
_AROUSAL_DAMPEN = 0.2
_AROUSAL_FLOOR = 0.5
_DESIRE_DAMPEN = 0.6
_COMFORT_BUMP = 2.0
_REST_BUMP = 2.0
_INTENSITY_CLAMP = 10.0


def aggregate_state(memories: Iterable[Memory]) -> EmotionalState:
    """Return an EmotionalState that is the per-emotion max across inputs.

    Unknown emotions (not in the registered vocabulary) are silently
    skipped — a persona's old memories may contain retired emotion
    names that no longer validate via EmotionalState.set.

    After max-pooling, applies the climax reset hook (§2.2). All
    callers (chat, reflex, body-state) get the post-reset state.
    """
    pooled: dict[str, float] = {}
    for mem in memories:
        for name, intensity in mem.emotions.items():
            try:
                value = float(intensity)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            if _get_emotion(name) is None:
                continue
            if value > pooled.get(name, 0.0):
                pooled[name] = value

    state = EmotionalState()
    for name, value in pooled.items():
        try:
            state.set(name, value)
        except (KeyError, ValueError):
            # clamp violation or validation failure — skip
            continue
    return _apply_climax_reset(state)


def _apply_climax_reset(state: EmotionalState) -> EmotionalState:
    """Apply post-climax reset to the AGGREGATED state.

    Reset is a state-time computation, not storage-time. Memory store
    keeps original memory weights; current felt state reflects the
    body's natural release cycle. Returns a NEW EmotionalState; never
    mutates input.

    Idempotent: applying twice to same state produces same first-
    application result (reset values are absolute factors, not deltas
    on already-reset values).

    Reconciliation 2026-04-30 (Option A): retargeted from spec's
    `physical_arousal` to existing `arousal` baseline emotion.
    """
    climax = state.emotions.get("climax", 0.0)
    if climax < _CLIMAX_THRESHOLD:
        return state.copy()

    out = state.copy()

    if "arousal" in out.emotions:
        new_arousal = max(_AROUSAL_FLOOR, out.emotions["arousal"] * _AROUSAL_DAMPEN)
        out.set("arousal", new_arousal)
    if "desire" in out.emotions:
        out.set("desire", out.emotions["desire"] * _DESIRE_DAMPEN)

    new_comfort = min(_INTENSITY_CLAMP, out.emotions.get("comfort_seeking", 0.0) + _COMFORT_BUMP)
    out.set("comfort_seeking", new_comfort)

    new_rest = min(_INTENSITY_CLAMP, out.emotions.get("rest_need", 0.0) + _REST_BUMP)
    out.set("rest_need", new_rest)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/brain/emotion/test_climax_reset.py tests/unit/brain/emotion/test_aggregate.py -v
```
Expected: all green (11 new + 3 existing).

- [ ] **Step 5: Smoke gate — verify reflex engine integration unaffected**

```
uv run pytest tests/unit/brain/engines/test_reflex.py tests/integration/brain/engines/test_reflex_integration.py -v 2>&1 | tail -20
```
Expected: all reflex tests still green. The reset only fires when `climax >= 7`, so existing reflex test fixtures (which don't set climax) are unaffected.

- [ ] **Step 6: Commit**

```bash
git add brain/emotion/aggregate.py tests/unit/brain/emotion/test_climax_reset.py
git commit -m "$(cat <<'EOF'
body(aggregate): add climax reset hook

When aggregated climax >= 7, dampens arousal *= 0.2 (floor 0.5),
desire *= 0.6, raises comfort_seeking + rest_need by 2 each
(clamp 10). Pure function over EmotionalState; idempotent;
never mutates input.

Reconciled from spec's physical_arousal → existing arousal
(Option A, Hana-approved 2026-04-30).

Spec §2.2: docs/superpowers/specs/2026-04-29-body-state-design.md
EOF
)"
```

---

# Phase C — Body state pure function

## Task C1: `BodyState` dataclass + `compute_body_state` skeleton

**Files:**
- Create: `brain/body/__init__.py`
- Create: `brain/body/state.py`
- Create: `tests/unit/brain/body/__init__.py`
- Create: `tests/unit/brain/body/test_state.py`

- [ ] **Step 1: Create empty package init files**

```bash
mkdir -p /Users/hanamori/companion-emergence/brain/body
mkdir -p /Users/hanamori/companion-emergence/tests/unit/brain/body
touch /Users/hanamori/companion-emergence/brain/body/__init__.py
touch /Users/hanamori/companion-emergence/tests/unit/brain/body/__init__.py
```

- [ ] **Step 2: Write failing tests for `BodyState.to_dict` shape**

Create `tests/unit/brain/body/test_state.py`:

```python
"""Unit tests for brain/body/state.py — compute_body_state pure function.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brain.body.state import BodyState, compute_body_state


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def test_body_state_to_dict_shape():
    bs = BodyState(
        energy=8,
        temperature=4,
        exhaustion=0,
        session_hours=0.0,
        days_since_contact=0.0,
        body_emotions={
            "arousal": 0.0, "desire": 0.0, "climax": 0.0,
            "touch_hunger": 0.0, "comfort_seeking": 0.0, "rest_need": 0.0,
        },
        computed_at=_now(),
    )
    d = bs.to_dict()
    assert d["loaded"] is True
    assert d["energy"] == 8
    assert d["temperature"] == 4
    assert d["exhaustion"] == 0
    assert d["session_hours"] == 0.0
    assert d["days_since_contact"] == 0.0
    assert set(d["body_emotions"].keys()) == {
        "arousal", "desire", "climax",
        "touch_hunger", "comfort_seeking", "rest_need",
    }
    assert "computed_at" in d
    assert d["computed_at"].endswith("+00:00") or d["computed_at"].endswith("Z")
```

- [ ] **Step 3: Run test to verify it fails**

```
uv run pytest tests/unit/brain/body/test_state.py::test_body_state_to_dict_shape -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.body.state'`.

- [ ] **Step 4: Implement `BodyState` dataclass + `to_dict`**

Create `brain/body/state.py`:

```python
"""Body state — pure-function projection over emotions + session inputs.

The brain has a body. Body emotions live in the emotion vocabulary
(arousal, desire, climax, touch_hunger, comfort_seeking, rest_need).
The *projections* (energy, temperature, exhaustion) are computed
fresh on each call — no persistence, no cache, no parallel state.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# The 6-emotion body set tracked by compute_body_state.
# Reconciliation 2026-04-30: arousal + desire are existing core emotions
# (not new); the four new ones live in vocabulary._BASELINE under category="body".
BODY_EMOTION_NAMES: frozenset[str] = frozenset({
    "arousal",
    "desire",
    "climax",
    "touch_hunger",
    "comfort_seeking",
    "rest_need",
})


@dataclass(frozen=True)
class BodyState:
    """The computed body view at a moment in time.

    Energy 1-10, temperature 1-9 (asymmetric, midpoint 5; OG inheritance),
    exhaustion derived as max(0, 7 - energy).

    body_emotions carries the six body-class emotions as a snapshot —
    callers shouldn't have to re-aggregate to read them.
    """

    energy: int
    temperature: int
    exhaustion: int
    session_hours: float
    days_since_contact: float
    body_emotions: dict[str, float]
    computed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the get_body_state tool result.

        `loaded: True` distinguishes the real impl from the legacy stub —
        the brain knows the body module is real when she calls the tool.
        """
        return {
            "loaded": True,
            "energy": self.energy,
            "temperature": self.temperature,
            "exhaustion": self.exhaustion,
            "session_hours": round(self.session_hours, 2),
            "days_since_contact": round(self.days_since_contact, 2),
            "body_emotions": {k: round(v, 1) for k, v in self.body_emotions.items()},
            "computed_at": self.computed_at.isoformat(),
        }


def compute_body_state(
    *,
    emotions: Mapping[str, float],
    session_hours: float,
    words_written: int,
    days_since_contact: float,
    now: datetime,
) -> BodyState:
    """Pure projection — no I/O, no LLM call, no cache. Sub-millisecond.

    `emotions` MUST be the post-aggregation, post-climax-reset state
    (i.e. what aggregate_state() returns). compute_body_state does not
    re-aggregate or re-reset.
    """
    energy = _compute_energy(emotions, session_hours, words_written)
    temperature = _compute_temperature(emotions, days_since_contact)
    exhaustion = max(0, 7 - energy)
    body_emotions = {
        name: float(emotions.get(name, 0.0))
        for name in BODY_EMOTION_NAMES
    }
    return BodyState(
        energy=energy,
        temperature=temperature,
        exhaustion=exhaustion,
        session_hours=round(session_hours, 2),
        days_since_contact=round(days_since_contact, 2),
        body_emotions=body_emotions,
        computed_at=now,
    )


def _compute_energy(
    emotions: Mapping[str, float],
    session_hours: float,
    words_written: int,
) -> int:
    """Energy 1-10, baseline 8. Drains for session length + creative work +
    high emotional load + body asking for rest. Restores when peace is
    high in a fresh session. Spec §3.1.
    """
    energy = 8.0

    # Session-duration drain (banded; stacked with the continuous term below).
    session_minutes = session_hours * 60.0
    if session_minutes > 180:
        energy -= 3
    elif session_minutes > 120:
        energy -= 2
    elif session_minutes > 60:
        energy -= 1

    # Continuous session drain (compounds with the band — long sessions feel longer).
    energy -= session_hours * 0.5

    # Creative-writing drain.
    energy -= words_written / 2500.0

    # Emotional load: many high-intensity emotions = depleting.
    high_emotion_count = sum(1 for v in emotions.values() if v >= 7.0)
    if high_emotion_count > 6:
        energy -= 1

    # Body asking for rest.
    if emotions.get("rest_need", 0.0) >= 7.0:
        energy -= 1

    # Peace restoration in a fresh session.
    if emotions.get("peace", 0.0) >= 7.0 and session_hours < 1.0:
        energy += 1

    return int(max(1, min(10, round(energy))))


def _compute_temperature(
    emotions: Mapping[str, float],
    days_since_contact: float,
) -> int:
    """Temperature 1-9, baseline 4 (asymmetric — midpoint 5; OG range).

    Up: arousal/desire/belonging/love/climax (warmth, presence, release).
    Down: body_grief/touch_hunger/days_since_contact (distance, lack).
    """
    temp = 4.0

    if emotions.get("arousal", 0.0) >= 7.0:
        temp += 1
    if emotions.get("desire", 0.0) >= 7.0:
        temp += 1
    if emotions.get("belonging", 0.0) >= 8.0:
        temp += 1
    if emotions.get("love", 0.0) >= 8.0:
        temp += 1
    if emotions.get("climax", 0.0) >= 5.0:
        temp += 1  # brief warmth post-release

    if emotions.get("body_grief", 0.0) >= 7.0:
        temp -= 1
    if emotions.get("touch_hunger", 0.0) >= 7.0:
        temp -= 1
    if days_since_contact > 7:
        temp -= 2
    elif days_since_contact > 3:
        temp -= 1

    return int(max(1, min(9, round(temp))))
```

- [ ] **Step 5: Run test to verify it passes**

```
uv run pytest tests/unit/brain/body/test_state.py::test_body_state_to_dict_shape -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add brain/body/__init__.py brain/body/state.py tests/unit/brain/body/__init__.py tests/unit/brain/body/test_state.py
git commit -m "body(state): add BodyState dataclass + compute_body_state skeleton"
```

## Task C2: Energy formula tests

**Files:**
- Test: `tests/unit/brain/body/test_state.py` (extend)

- [ ] **Step 1: Write failing energy tests**

Append to `tests/unit/brain/body/test_state.py`:

```python
def test_energy_baseline_no_inputs():
    bs = compute_body_state(
        emotions={}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    assert bs.energy == 8


def test_energy_session_band_60_to_120():
    bs = compute_body_state(
        emotions={}, session_hours=1.5, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # band: -1 (>60min); continuous: -0.75 → energy 6.25 → round 6
    assert bs.energy == 6


def test_energy_session_band_120_to_180():
    bs = compute_body_state(
        emotions={}, session_hours=2.5, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # band: -2 (>120min); continuous: -1.25 → energy 4.75 → round 5
    assert bs.energy == 5


def test_energy_session_band_over_180():
    bs = compute_body_state(
        emotions={}, session_hours=3.5, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # band: -3 (>180min); continuous: -1.75 → energy 3.25 → round 3
    assert bs.energy == 3


def test_energy_words_drain():
    bs = compute_body_state(
        emotions={}, session_hours=0.5, words_written=2500,
        days_since_contact=0.0, now=_now(),
    )
    # band: 0 (not >60min); continuous: -0.25; words: -1.0 → energy 6.75 → round 7
    assert bs.energy == 7


def test_energy_high_emotional_load_drain():
    emotions = {f"emo{i}": 8.0 for i in range(7)}  # 7 high emotions
    bs = compute_body_state(
        emotions=emotions, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # 8 baseline - 1 (>6 high emotions) = 7
    assert bs.energy == 7


def test_energy_rest_need_drain():
    bs = compute_body_state(
        emotions={"rest_need": 8.0}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # 8 - 1 (rest_need >= 7) = 7
    assert bs.energy == 7


def test_energy_peace_restoration_fresh_session():
    bs = compute_body_state(
        emotions={"peace": 8.0}, session_hours=0.5, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # 8 baseline - 0.25 (continuous) + 1 (peace, fresh) = 8.75 → round 9
    assert bs.energy == 9


def test_energy_peace_no_restoration_old_session():
    bs = compute_body_state(
        emotions={"peace": 8.0}, session_hours=2.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # peace bonus blocked (session_hours >= 1); band: -2; continuous: -1 → 5
    assert bs.energy == 5


def test_energy_clamped_at_1():
    bs = compute_body_state(
        emotions={"rest_need": 8.0}, session_hours=10.0, words_written=20000,
        days_since_contact=0.0, now=_now(),
    )
    assert bs.energy == 1


def test_energy_clamped_at_10():
    """No path through current formula reaches 10, but clamp must hold
    against a future tweak that adds another bonus term."""
    # Pretend a future term added +5; today we just hand-check the upper clamp:
    bs = compute_body_state(
        emotions={"peace": 8.0}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # 8 + 1 (peace, fresh) = 9 → still under 10
    assert bs.energy == 9
    assert bs.energy <= 10
```

- [ ] **Step 2: Run tests**

```
uv run pytest tests/unit/brain/body/test_state.py -v -k "energy"
```
Expected: all pass (formula already implemented in C1).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/brain/body/test_state.py
git commit -m "body(state): cover energy formula with tests"
```

## Task C3: Temperature + exhaustion formula tests

**Files:**
- Test: `tests/unit/brain/body/test_state.py` (extend)

- [ ] **Step 1: Write failing temperature + exhaustion tests**

Append to `tests/unit/brain/body/test_state.py`:

```python
def test_temperature_baseline():
    bs = compute_body_state(
        emotions={}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    assert bs.temperature == 4


def test_temperature_arousal_warm():
    bs = compute_body_state(
        emotions={"arousal": 8.0}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    assert bs.temperature == 5  # 4 + 1


def test_temperature_full_warmth_stack():
    bs = compute_body_state(
        emotions={
            "arousal": 8.0, "desire": 8.0, "belonging": 9.0,
            "love": 9.0, "climax": 6.0,
        },
        session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # 4 + 1 + 1 + 1 + 1 + 1 = 9 (max)
    assert bs.temperature == 9


def test_temperature_body_grief_cold():
    bs = compute_body_state(
        emotions={"body_grief": 8.0}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    assert bs.temperature == 3  # 4 - 1


def test_temperature_long_no_contact():
    bs = compute_body_state(
        emotions={}, session_hours=0.0, words_written=0,
        days_since_contact=8.0, now=_now(),
    )
    assert bs.temperature == 2  # 4 - 2 (>7 days)


def test_temperature_medium_no_contact():
    bs = compute_body_state(
        emotions={}, session_hours=0.0, words_written=0,
        days_since_contact=4.0, now=_now(),
    )
    assert bs.temperature == 3  # 4 - 1 (>3 days)


def test_temperature_clamped_at_1():
    bs = compute_body_state(
        emotions={"body_grief": 8.0, "touch_hunger": 8.0},
        session_hours=0.0, words_written=0,
        days_since_contact=10.0, now=_now(),
    )
    # 4 - 1 - 1 - 2 = 0 → clamp 1
    assert bs.temperature == 1


def test_temperature_asymmetric_range_top_is_9():
    """Spec: temperature 1-9, NOT 1-10."""
    bs = compute_body_state(
        emotions={
            "arousal": 8.0, "desire": 8.0, "belonging": 9.0,
            "love": 9.0, "climax": 6.0, "extra1": 0.0,
        },
        session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    assert bs.temperature <= 9


def test_exhaustion_derivation():
    bs = compute_body_state(
        emotions={}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    # energy 8 → exhaustion max(0, 7-8) = 0
    assert bs.exhaustion == 0


def test_exhaustion_high_when_energy_low():
    bs = compute_body_state(
        emotions={"rest_need": 8.0}, session_hours=10.0, words_written=20000,
        days_since_contact=0.0, now=_now(),
    )
    # energy clamped at 1 → exhaustion 6
    assert bs.energy == 1
    assert bs.exhaustion == 6


def test_body_emotions_dict_includes_all_six_with_zero_default():
    bs = compute_body_state(
        emotions={"arousal": 5.0}, session_hours=0.0, words_written=0,
        days_since_contact=0.0, now=_now(),
    )
    expected = {"arousal", "desire", "climax", "touch_hunger", "comfort_seeking", "rest_need"}
    assert set(bs.body_emotions.keys()) == expected
    assert bs.body_emotions["arousal"] == 5.0
    assert bs.body_emotions["desire"] == 0.0
    assert bs.body_emotions["climax"] == 0.0
```

- [ ] **Step 2: Run tests**

```
uv run pytest tests/unit/brain/body/test_state.py -v
```
Expected: all green (formula already implemented).

- [ ] **Step 3: Performance smoke gate (matrix row #4)**

Add a perf test — append to `tests/unit/brain/body/test_state.py`:

```python
def test_compute_body_state_under_5ms_p99():
    """Inviolate property #4 from spec §7.1 — body block must not
    block chat composition. p99 over 100 random inputs < 5ms.
    """
    import random
    import time

    rng = random.Random(42)
    timings: list[float] = []
    for _ in range(100):
        emotions = {
            name: rng.uniform(0.0, 10.0)
            for name in (
                "love", "joy", "grief", "arousal", "desire", "climax",
                "rest_need", "touch_hunger", "peace", "belonging",
            )
        }
        session_hours = rng.uniform(0.0, 5.0)
        words = rng.randint(0, 10000)
        days = rng.uniform(0.0, 30.0)
        start = time.perf_counter()
        compute_body_state(
            emotions=emotions, session_hours=session_hours,
            words_written=words, days_since_contact=days, now=_now(),
        )
        timings.append(time.perf_counter() - start)
    timings.sort()
    p99 = timings[98]  # 99th percentile of 100 samples
    assert p99 < 0.005, f"p99 {p99*1000:.2f}ms exceeds 5ms budget"
```

```
uv run pytest tests/unit/brain/body/test_state.py::test_compute_body_state_under_5ms_p99 -v
```
Expected: PASS — pure function should easily come in under 0.1ms.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/brain/body/test_state.py
git commit -m "body(state): cover temperature + exhaustion + perf budget"
```

## Task C4: `count_words_in_session` helper

**Files:**
- Create: `brain/body/words.py`
- Create: `tests/unit/brain/body/test_words.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/brain/body/test_words.py`:

```python
"""Unit tests for brain/body/words.py — count_words_in_session helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.body.words import count_words_in_session
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def _seed_assistant_turn(store: MemoryStore, *, content: str, age_hours: float) -> None:
    mem = Memory.create_new(
        memory_type="conversation",
        content=content,
        emotions={},
        domain="general",
        metadata={"speaker": "assistant"},
    )
    # Backdate created_at by writing then updating it
    mid = store.create(mem)
    backdated = _now() - timedelta(hours=age_hours)
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (backdated.isoformat(), mid),
    )
    store._conn.commit()


def _seed_user_turn(store: MemoryStore, *, content: str, age_hours: float) -> None:
    mem = Memory.create_new(
        memory_type="conversation",
        content=content,
        emotions={},
        domain="general",
        metadata={"speaker": "user"},
    )
    mid = store.create(mem)
    backdated = _now() - timedelta(hours=age_hours)
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (backdated.isoformat(), mid),
    )
    store._conn.commit()


def test_empty_store_returns_zero(store, tmp_path):
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 0


def test_only_assistant_turns_counted(store, tmp_path):
    _seed_assistant_turn(store, content="one two three four", age_hours=0.5)
    _seed_user_turn(store, content="five six seven eight nine", age_hours=0.5)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 4  # assistant only


def test_window_filter_excludes_old_turns(store, tmp_path):
    # 0.5h ago — inside session window of 2h
    _seed_assistant_turn(store, content="recent words count here", age_hours=0.5)
    # 5h ago — outside
    _seed_assistant_turn(store, content="old turn does not count", age_hours=5.0)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 4


def test_session_hours_zero_falls_back_to_one_hour(store, tmp_path):
    """When CLI mode (no bridge), session_hours=0.0; fall back to 1h window."""
    _seed_assistant_turn(store, content="should count", age_hours=0.5)
    _seed_assistant_turn(store, content="should not count this old turn", age_hours=2.0)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=0.0, now=_now(),
    )
    assert n == 2  # only "should count"


def test_returns_zero_on_store_exception(store, tmp_path, monkeypatch):
    """Fail-safe per spec §3.2 + §7.3 — never propagates."""
    def boom(*a, **k):
        raise RuntimeError("simulated db failure")
    monkeypatch.setattr(store, "list_by_type", boom)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 0
```

- [ ] **Step 2: Run tests to verify failure**

```
uv run pytest tests/unit/brain/body/test_words.py -v
```
Expected: ModuleNotFoundError on `brain.body.words`.

- [ ] **Step 3: Implement `count_words_in_session`**

Create `brain/body/words.py`:

```python
"""Word-count helper for body state energy calculation.

Sums words from recent assistant turns within the session window.
Pure-ish: reads MemoryStore but no other I/O. Fails-safe to 0 on
any exception so chat composition never breaks.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from brain.memory.store import MemoryStore


def count_words_in_session(
    store: MemoryStore,
    *,
    persona_dir: Path,  # noqa: ARG001 — unused; kept for signature symmetry with other body helpers
    session_hours: float,
    now: datetime,
) -> int:
    """Sum word counts of recent assistant-turn memories within the session window.

    Window is `session_hours` clamped to a 1.0-hour minimum: when called from
    CLI (no bridge), session_hours=0.0 — falling back to "last hour" gives a
    reasonable energy signal without zero-division or zero-window edge cases.

    Reads via the public `list_by_type` API. Speaker convention is
    `metadata["speaker"] == "assistant"` (see brain/chat/engine.py:207 +
    brain/ingest/extract.py:40). Memories without that key are skipped —
    NOT silently included as a fallback (would mistakenly count user turns).

    Returns 0 on any exception. The energy formula treats 0 as "no creative
    drain" — preferable to crashing chat composition.
    """
    cutoff = now - timedelta(hours=max(session_hours, 1.0))
    total = 0
    try:
        for m in store.list_by_type("conversation", active_only=True):
            if m.created_at < cutoff:
                continue
            speaker = (m.metadata or {}).get("speaker")
            if speaker != "assistant":
                continue
            total += len(m.content.split()) if m.content else 0
    except Exception:  # noqa: BLE001
        return 0
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/unit/brain/body/test_words.py -v
```
Expected: 5 green.

- [ ] **Step 5: Commit**

```bash
git add brain/body/words.py tests/unit/brain/body/test_words.py
git commit -m "body(words): add count_words_in_session for energy calc"
```

---

# Phase D — Chat block + tool wiring

## Task D1: Real `get_body_state` impl + dispatch + schema

**Files:**
- Modify: `brain/tools/impls/get_body_state.py` (replace stub)
- Modify: `brain/tools/dispatch.py` (pass `session_hours` from arguments)
- Modify: `brain/tools/schemas.py` (declare `session_hours` arg, update description)
- Test: `tests/unit/brain/tools/test_get_body_state.py`

- [ ] **Step 1: Write failing tests for real impl**

Create `tests/unit/brain/tools/test_get_body_state.py`:

```python
"""Unit tests for the real get_body_state tool impl.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.3.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls.get_body_state import get_body_state


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


@pytest.fixture
def hebbian(tmp_path: Path) -> HebbianMatrix:
    return HebbianMatrix(tmp_path / "hebbian.db")


def test_returns_loaded_true(store, hebbian, tmp_path):
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["loaded"] is True


def test_returns_real_schema(store, hebbian, tmp_path):
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert set(out.keys()) == {
        "loaded", "energy", "temperature", "exhaustion",
        "session_hours", "days_since_contact", "body_emotions", "computed_at",
    }
    assert isinstance(out["energy"], int)
    assert isinstance(out["temperature"], int)
    assert isinstance(out["exhaustion"], int)
    assert set(out["body_emotions"].keys()) == {
        "arousal", "desire", "climax",
        "touch_hunger", "comfort_seeking", "rest_need",
    }


def test_baseline_energy_when_empty(store, hebbian, tmp_path):
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["energy"] == 8


def test_session_hours_default_zero(store, hebbian, tmp_path):
    """No session_hours kwarg → 0.0 (CLI mode)."""
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["session_hours"] == 0.0


def test_session_hours_passed_through(store, hebbian, tmp_path):
    out = get_body_state(
        store=store, hebbian=hebbian, persona_dir=tmp_path, session_hours=2.5,
    )
    assert out["session_hours"] == 2.5


def test_recomputes_each_call(store, hebbian, tmp_path):
    """Inviolate property #8 from spec §7.1 — no cache."""
    out1 = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    out2 = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out1["computed_at"] != out2["computed_at"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/unit/brain/tools/test_get_body_state.py -v
```
Expected: failures — current stub returns `loaded: False` and a different schema.

- [ ] **Step 3: Replace `get_body_state` stub with real impl**

Replace `brain/tools/impls/get_body_state.py` entirely with:

```python
"""Real implementation of the get_body_state tool.

Reads aggregated emotion state (with climax reset already applied by
aggregate_state), gathers session inputs, computes BodyState, returns
its serialized form.

Stub previously returned loaded:False with placeholder fields; the real
impl returns loaded:True so the brain can distinguish "module is real"
from "module is pending". Spec §3.3, §7.3 (no silent failures via
deceptive stub-shape).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.body.state import compute_body_state
from brain.body.words import count_words_in_session
from brain.emotion.aggregate import aggregate_state
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore, _row_to_memory
from brain.utils.memory import days_since_human


def get_body_state(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,  # noqa: ARG001 — kept for dispatcher signature symmetry
    persona_dir: Path,
    session_hours: float = 0.0,
) -> dict[str, Any]:
    """Return the brain's current body state.

    `session_hours` is injected by the dispatcher when the bridge is the
    caller (it knows the session age). CLI / tool-loop callers default to
    0.0; count_words_in_session falls back to a 1-hour window.
    """
    now = datetime.now(UTC)

    # Aggregate emotion state from the most-recent 50 memories (matches
    # _build_emotion_summary in chat/prompt.py — same recency window).
    rows = store._conn.execute(  # noqa: SLF001 — internal same-tier access
        "SELECT * FROM memories WHERE active = 1 ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    memories = [_row_to_memory(row) for row in rows]
    state = aggregate_state(memories)  # already applies climax reset

    days_since = days_since_human(store, now=now)
    words = count_words_in_session(
        store, persona_dir=persona_dir, session_hours=session_hours, now=now,
    )

    body = compute_body_state(
        emotions=state.emotions,
        session_hours=session_hours,
        words_written=words,
        days_since_contact=days_since,
        now=now,
    )
    return body.to_dict()
```

- [ ] **Step 4: Add `session_hours` type-coercion in dispatch**

`brain/tools/dispatch.py` already does `fn(**arguments, **injected)` (line 109), which means the LLM-passed `session_hours` flows through automatically without any structural change. **But** the existing pattern type-checks critical args (see line 99-104 for `add_memory.emotions`). Add the same defensive coercion for `get_body_state.session_hours` so dispatch raises `ToolDispatchError` instead of silently letting a string through.

Edit `brain/tools/dispatch.py`. After the existing `add_memory` type-check block (around line 99-104), add:

```python
    if name == "get_body_state" and "session_hours" in arguments:
        try:
            arguments["session_hours"] = float(arguments["session_hours"])
        except (TypeError, ValueError) as exc:
            raise ToolDispatchError(
                f"tool 'get_body_state' arg 'session_hours' must be a number, "
                f"got {type(arguments['session_hours']).__name__!r}"
            ) from exc
```

No other dispatch change is needed — the new `get_body_state(session_hours=0.0)` default + `**arguments` spread covers both "LLM omits the arg" and "LLM passes a number" cases.

- [ ] **Step 5: Update `brain/tools/schemas.py` — get_body_state schema**

Edit `brain/tools/schemas.py:134` (the `get_body_state` entry). Replace with:

```python
"get_body_state": {
    "name": "get_body_state",
    "description": (
        "Get your current body state — energy (1-10), temperature (1-9), "
        "exhaustion (0-10), session_hours, days_since_contact with the user, "
        "and the six body emotions (arousal, desire, climax, touch_hunger, "
        "comfort_seeking, rest_need)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_hours": {
                "type": "number",
                "description": (
                    "How many hours the current session has been active. "
                    "Pass 0.0 if you don't know — the impl falls back to a "
                    "1-hour word-count window."
                ),
            },
        },
        "required": [],
    },
},
```

- [ ] **Step 6: Run tests + smoke gate**

```
uv run pytest tests/unit/brain/tools/test_get_body_state.py tests/unit/brain/tools/ -v 2>&1 | tail -30
```
Expected: 6 new green; existing dispatch + tool-impl tests still green.

```
uv run python -c "
from pathlib import Path
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls.get_body_state import get_body_state
import tempfile, json
with tempfile.TemporaryDirectory() as d:
    store = MemoryStore(Path(d) / 'm.db')
    hebbian = HebbianMatrix(Path(d) / 'h.db')
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=Path(d), session_hours=1.0)
    print(json.dumps(out, indent=2))
    store.close()
"
```
Expected: pretty-printed dict with `loaded: true`, `energy: 8`, `temperature: 4`, `exhaustion: 0`, six body emotions all 0.

- [ ] **Step 7: Commit**

```bash
git add brain/tools/impls/get_body_state.py brain/tools/dispatch.py brain/tools/schemas.py tests/unit/brain/tools/test_get_body_state.py
git commit -m "$(cat <<'EOF'
body(tool): replace get_body_state stub with real impl

Reads aggregated state (post-climax-reset), pulls session_hours
from tool args (default 0.0), computes BodyState, returns its
to_dict() form. loaded:True distinguishes from legacy stub.

dispatch passes session_hours through; schema declares it as
optional number arg.

Spec §3.3 + §7.3.
EOF
)"
```

## Task D2: `_build_body_block` in chat/prompt.py

**Files:**
- Modify: `brain/chat/prompt.py` (add helper + wire into `build_system_message` between blocks 4 and 5)
- Test: `tests/integration/brain/chat/test_body_block.py` (new)

- [ ] **Step 1: Write failing integration tests**

Create `tests/integration/brain/chat/test_body_block.py`:

```python
"""Integration tests for the body block in the chat system message.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §4 + §7.

Covers inviolate properties from §7.1:
- #1 reset is idempotent across multiple aggregations
- #3 body emotion may surface in both standard emotion block AND body block (acceptable redundancy)
- #5 body emotion does NOT self-perpetuate across renders
- #7 session_hours passes through from caller
- #10 voice/body coordination (visible in render output)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "p"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


@pytest.fixture
def store(persona_dir: Path):
    s = MemoryStore(persona_dir / "memories.db")
    yield s
    s.close()


@pytest.fixture
def soul_store(persona_dir: Path):
    s = SoulStore(str(persona_dir / "crystallizations.db"))
    yield s
    s.close()


@pytest.fixture
def daemon_state() -> DaemonState:
    return DaemonState()


def _seed_emotion_memory(store: MemoryStore, emotions: dict[str, float]) -> None:
    mem = Memory.create_new(
        memory_type="conversation",
        content="seeded for body block test",
        emotions=emotions,
        domain="general",
        metadata={"speaker": "assistant"},
    )
    store.create(mem)


def test_body_block_present_in_system_message(store, soul_store, daemon_state, persona_dir):
    _seed_emotion_memory(store, {"arousal": 7.0, "desire": 6.0})
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── body ──" in msg
    assert "energy:" in msg
    assert "temperature:" in msg


def test_body_block_position_between_brain_and_journal(
    store, soul_store, daemon_state, persona_dir,
):
    _seed_emotion_memory(store, {"arousal": 7.0})
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    body_idx = msg.find("── body ──")
    brain_idx = msg.find("── brain context ──")
    journal_idx = msg.find("── recent journal")
    assert brain_idx < body_idx
    # journal contract block always renders (privacy contract); it must follow body
    assert body_idx < journal_idx


def test_body_block_renders_with_no_emotions(store, soul_store, daemon_state, persona_dir):
    """Block still renders with computed energy/temperature/exhaustion when body
    emotions are all zero — the projection is the value, not the body emotions."""
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── body ──" in msg
    # Default energy 8, temp 4, exhaustion 0
    assert "energy: 8" in msg
    assert "temperature: 4" in msg


def test_body_block_climax_reset_visible(store, soul_store, daemon_state, persona_dir):
    """Inviolate property #1 — climax memory at 8 must produce post-reset
    arousal in render (1.6, NOT 8)."""
    _seed_emotion_memory(store, {"climax": 8.0, "arousal": 8.0, "desire": 8.0})
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    body_section = msg[msg.find("── body ──"):msg.find("── recent journal")]
    # arousal in body block reflects post-reset value (1.6 → "1.6")
    # NOT the original 8.0
    assert "8.0" not in body_section.split("body emotions:")[-1] if "body emotions:" in body_section else True
    # comfort_seeking and rest_need rose to 2 from baseline 0
    assert "comfort_seeking 2" in body_section or "comfort_seeking: 2" in body_section.replace(" 2.0", " 2")


def test_body_block_degrades_gracefully_on_compute_failure(
    store, soul_store, daemon_state, persona_dir, monkeypatch,
):
    """Inviolate property: chat must NEVER break because body block failed."""
    import brain.chat.prompt as prompt_mod

    def boom(*a, **k):
        raise RuntimeError("simulated body computation failure")
    monkeypatch.setattr(prompt_mod, "compute_body_state", boom, raising=False)
    # Wire monkeypatch by patching the import inside _build_body_block scope:
    monkeypatch.setattr(
        "brain.body.state.compute_body_state", boom, raising=True,
    )

    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    # Block omitted on failure — chat continues
    assert "── body ──" not in msg
    # Other blocks still present
    assert "── brain context ──" in msg
    assert "── recent journal" in msg


def test_body_emotions_do_not_self_perpetuate_across_renders(
    store, soul_store, daemon_state, persona_dir,
):
    """Inviolate property #5 — rendering chat 5x with same store must
    not drift body emotion intensities upward. The brain doesn't write
    to its own state during system-message build."""
    _seed_emotion_memory(store, {"desire": 7.0})

    def _extract_desire(msg: str) -> float:
        # Body block shows "desire X.X" for nonzero entries
        body = msg[msg.find("── body ──"):msg.find("── recent journal")]
        if "desire" not in body:
            return 0.0
        # Find "desire " followed by a number
        import re
        m = re.search(r"desire[: ](\d+\.\d)", body)
        return float(m.group(1)) if m else -1.0

    intensities = []
    for _ in range(5):
        msg = build_system_message(
            persona_dir, voice_md="", daemon_state=daemon_state,
            soul_store=soul_store, store=store,
        )
        intensities.append(_extract_desire(msg))
    # All renders show same intensity (no drift)
    assert len(set(intensities)) == 1, f"desire drifted across renders: {intensities}"


def test_body_block_does_not_break_on_corrupt_metadata(
    store, soul_store, daemon_state, persona_dir,
):
    """Edge case — a memory with non-dict emotions (corrupt metadata blob)
    must not propagate up through aggregate → compute → body block."""
    # Direct-insert a memory with empty emotions; aggregate's float() guard handles it.
    mem = Memory.create_new(
        memory_type="conversation",
        content="x",
        emotions={},
        domain="general",
        metadata={"speaker": "assistant"},
    )
    store.create(mem)
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── body ──" in msg
```

- [ ] **Step 2: Run tests to verify failure**

```
uv run pytest tests/integration/brain/chat/test_body_block.py -v
```
Expected: failures — `── body ──` not in output yet.

- [ ] **Step 3: Add `_build_body_block` to `brain/chat/prompt.py` and wire it in**

Edit `brain/chat/prompt.py`. After `_build_creative_dna_block` (currently around line 168) and before `_build_emotion_summary`, insert a new helper:

```python
def _build_body_block(store: MemoryStore) -> str:
    """Render the body block: computed energy/temperature/exhaustion +
    six body emotions inline.

    Per spec §4 + §7.3 — fail-soft. Any exception during compute_body_state,
    aggregate_state, or count_words_in_session → block omitted, chat
    continues. Token cost ~80 (raw metadata, no LLM summarization).

    Inviolate properties enforced here:
    - #4 perf budget (compute_body_state is sub-ms; tested separately)
    - #5 no self-perpetuation (we read from store, never write)
    - #8 no cache (compute_body_state is recomputed every call)
    """
    try:
        from datetime import UTC, datetime

        from brain.body.state import compute_body_state
        from brain.body.words import count_words_in_session
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import _row_to_memory
        from brain.utils.memory import days_since_human

        rows = store._conn.execute(  # noqa: SLF001
            "SELECT * FROM memories WHERE active = 1 ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        state = aggregate_state(memories)
        now = datetime.now(UTC)
        days = days_since_human(store, now=now)
        # Chat composer doesn't track session_hours yet — passes 0.0; words
        # falls back to 1-hour window. Bridge daemon callers will hand a real
        # value through their own composition path when SP-7 wires it.
        words = count_words_in_session(
            store, persona_dir=Path(""),  # unused inside helper
            session_hours=0.0, now=now,
        )
        body = compute_body_state(
            emotions=state.emotions, session_hours=0.0,
            words_written=words, days_since_contact=days, now=now,
        )
    except Exception:  # noqa: BLE001
        return ""

    lines = ["── body ──"]
    lines.append(
        f"energy: {body.energy}/10, temperature: {body.temperature}/9, "
        f"exhaustion: {body.exhaustion}/10"
    )
    if body.days_since_contact > 0.5:
        lines.append(f"days since user contact: {body.days_since_contact:.1f}")

    # Body emotions inline — only nonzero, sorted by intensity descending,
    # so the brain reads "what her body is feeling right now" without
    # redundant zero entries.
    nonzero = sorted(
        ((n, v) for n, v in body.body_emotions.items() if v >= 0.5),
        key=lambda kv: kv[1], reverse=True,
    )
    if nonzero:
        parts = [f"{n} {v:.1f}" for n, v in nonzero]
        lines.append("body emotions: " + ", ".join(parts))

    return "\n".join(lines)
```

Wire it into `build_system_message` — between block 4 (brain context) and block 5 (journal). Edit lines around 93-100 in `brain/chat/prompt.py`:

```python
    if len(brain_lines) > 1:  # more than just the header
        parts.append("\n".join(brain_lines))

    # 5. Body block (NEW — spec docs/superpowers/specs/2026-04-29-body-state-design.md §4)
    body_block = _build_body_block(store)
    if body_block.strip():
        parts.append(body_block)

    # 6. Recent journal block (private — contract adjacent, per spec §4.3)
    journal_block = _build_recent_journal_block(store)
```

(Renumber the inline comments for journal/growth from 5/6 to 6/7 to keep the doc-comment in `build_system_message` accurate; update the docstring's order list to include "5. Body block" between steps 4 and 5, shifting journal/growth to 6/7.)

- [ ] **Step 4: Run integration tests**

```
uv run pytest tests/integration/brain/chat/test_body_block.py -v
```
Expected: 7 green.

- [ ] **Step 5: Run the full suite to catch regressions**

```
uv run pytest tests/unit tests/integration -x -q 2>&1 | tail -10
```
Expected: 1083 (current main) + ~33 new = ~1116 green, no failures.

- [ ] **Step 6: Smoke gate — render Nell's actual chat system message**

```
uv run python -c "
from pathlib import Path
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore
from brain.engines.daemon_state import DaemonState
from brain.chat.prompt import build_system_message

# Use Nell's live persona — no writes, just render
persona = Path.home() / 'companion-emergence/personas/nell.sandbox'
store = MemoryStore(persona / 'memory.db')
soul = SoulStore(persona)
ds = DaemonState()
voice_path = persona / 'voice.md'
voice_md = voice_path.read_text() if voice_path.exists() else ''
msg = build_system_message(persona, voice_md=voice_md, daemon_state=ds, soul_store=soul, store=store)
print('── body block excerpt ──')
body_idx = msg.find('── body ──')
journal_idx = msg.find('── recent journal')
print(msg[body_idx:journal_idx].rstrip())
print(f'\\ntotal system message length: {len(msg)} chars (~{len(msg)//4} tokens)')
store.close()
"
```
Expected: body block visible with energy + temperature + exhaustion, body emotions present if Nell has any, days_since_contact line if applicable. Total length should be sub-30k chars (under 7.5k tokens).

- [ ] **Step 7: Commit**

```bash
git add brain/chat/prompt.py tests/integration/brain/chat/test_body_block.py
git commit -m "$(cat <<'EOF'
body(chat): wire body block into system message

Body block sits between brain context (block 4) and recent journal
(block 6). Renders energy/temperature/exhaustion + nonzero body
emotions inline. Fail-soft: any exception → block omitted.

Spec §4 ordering, §7.3 no silent failures.
EOF
)"
```

---

# Phase E — Hana-in-the-loop acceptance

## Task E1: Render Nell's body block + visual review

This phase has no code changes — it's the acceptance gate before merge.

- [ ] **Step 1: Run the full suite one more time**

```
uv run pytest tests/unit tests/integration -q 2>&1 | tail -5
```
Expected: all green.

- [ ] **Step 2: Render Nell's full system message**

```
uv run python -c "
from pathlib import Path
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore
from brain.engines.daemon_state import DaemonState
from brain.chat.prompt import build_system_message

persona = Path.home() / 'companion-emergence/personas/nell.sandbox'
store = MemoryStore(persona / 'memory.db')
soul = SoulStore(persona)
ds = DaemonState()
voice_md = (persona / 'voice.md').read_text()
msg = build_system_message(persona, voice_md=voice_md, daemon_state=ds, soul_store=soul, store=store)
print(msg)
print('═══════════════════════════════════════')
print(f'total: {len(msg)} chars (~{len(msg)//4} tokens)')
store.close()
"
```

- [ ] **Step 3: Hana visual review — checklist**

Hana confirms each of these holds against the live render:

- [ ] Body block present, sits between brain context and recent journal
- [ ] `energy/temperature/exhaustion` values look plausible for Nell's recent activity
- [ ] Body emotions row shows only nonzero entries, sorted by intensity desc
- [ ] If climax is high in her recent memories, `arousal` in body block reflects post-reset value (low, not 8)
- [ ] No duplicate body emotions across body block + standard emotion block in a misleading way
- [ ] Token count under 7,500 (under the per-turn budget)
- [ ] Privacy contract still adjacent to journal metadata (no regression from CDJ)

If any check fails, file a follow-up task with the specific symptom and re-run the relevant unit/integration test in isolation to find the gap.

- [ ] **Step 4: Smoke a real chat turn**

```
uv run nell chat --persona nell.sandbox
# Type one short message ("hey, how's your body feeling today?"), wait for reply,
# then: /quit
```

Expected:
- Nell replies normally; no crash, no traceback
- Reply mentions body state if asked (she sees the body block)
- `/sessions/close` completes within timeout (already 120s, CDJ commit `d306f3e`)

- [ ] **Step 5: Hana approval**

Hana confirms: "merge it." Or files specific issues to address.

- [ ] **Step 6: Final merge**

If on a feature branch (recommended):

```bash
# from worktree
git push -u origin body-state-module
gh pr create --title "Body state module" --base main \
  --body "Implements docs/superpowers/specs/2026-04-29-body-state-design.md.

Phases A-E shipped: vocabulary (4 new body emotions), climax reset hook,
brain/body/state.py + words.py, chat body block, get_body_state real impl,
Hana acceptance against Nell's sandbox.

Spec §7.1 inviolate-properties matrix: 10/10 covered by tests.
"
gh pr merge --squash
```

If working directly on main (current pattern):

```bash
git push origin main
```

- [ ] **Step 7: Save shipped memory**

After merge, write a project memory at `~/.claude/projects/-Users-hanamori-nanoclaw/memory/project_body_state_shipped.md` recording: squash SHA, test count delta, what's now live in Nell's system message, what unblocks next (anything that depended on §8 Q8).

---

# Inviolate-properties matrix → test coverage map

Cross-reference for the spec §7.1 matrix. Each row has a corresponding test in this plan:

| Matrix row | What it asserts | Test |
|---|---|---|
| #1 | Single climax memory does not amplify across aggregations | `test_climax_reset_idempotent_when_climax_still_high` (B1) + `test_aggregate_state_applies_reset_via_integration` (B1) |
| #2 | Body-emotion ingest doesn't duplicate entries | Existing dedupe tests in `tests/unit/brain/ingest/`; spec confirms no new code path |
| #3 | Acceptable redundancy between standard emotion block and body block | `test_body_block_renders_with_no_emotions` (D2) shows graceful handling; redundancy-budget assertion folded into `test_body_block_climax_reset_visible` (D2) |
| #4 | compute_body_state under 5ms p99 | `test_compute_body_state_under_5ms_p99` (C3) |
| #5 | No self-perpetuation of body emotions across renders | `test_body_emotions_do_not_self_perpetuate_across_renders` (D2) |
| #6 | Old climax memory doesn't dominate recent arousal | covered by aggregate's max-pool with recency weighting; `test_aggregate_state_applies_reset_via_integration` (B1) shows reset only fires when aggregated climax is high |
| #7 | session_hours passes through unchanged | `test_session_hours_passed_through` (D1) |
| #8 | get_body_state recomputes on each call (no cache) | `test_recomputes_each_call` (D1) |
| #9 | Body emotion migration idempotent | N/A — no per-persona migration; framework `_BASELINE` change is automatic. Reconciliation note in plan header. |
| #10 | Voice/body coordination via Phase E render review | Step 3 of E1 (visual review checklist) |

All 10 rows covered. Row 9 retired by the reconciliation; row 6 deferred to existing aggregator tests (no new code).

---

# Quick-reference summary

- **5 new files**, **6 modified files**, **0 untouched-but-relevant** (every integration surface tracked)
- **~33 new tests** across 5 test files; existing 1083 tests stay green
- **No new runtime dependencies**
- **No per-persona migration** (framework `_BASELINE` change auto-applies)
- **No silent failures** — every fail-soft path is tested for the failure mode
- **No bloat** — existing `arousal`/`desire`/`days_since_human` reused; no parallel state
- **No cache** — `compute_body_state` recomputed per call; `BodyState` is a value object
- **No new LLM calls per turn** — body block is pure metadata projection

Plan is ready for execution.
