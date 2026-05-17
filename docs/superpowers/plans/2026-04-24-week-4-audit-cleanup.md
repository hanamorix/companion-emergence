# Week 4 Audit Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all Blocking + Important items from the Week 4 post-ship audit (path traversal, multi-persona voice leaks, data integrity gaps, dead code, duplicated helpers). Nits are deferred for manual review with Hana.

**Architecture:** Surgical fixes organised into 8 focused tasks, grouped by theme (security → multi-persona correctness → data integrity → cleanup). Every change lands with a regression test. TDD throughout.

**Tech Stack:** Python 3.12, pytest, ruff. No new dependencies. All changes stay inside `brain/` + `tests/`.

**Audit reports** (source of truth for what each task fixes): the four parallel audit agents produced findings ranked Blocking/Important/Nit. This plan covers Blocking + Important only.

**Running test total:** pre-start 431. After this plan: expect small positive delta (~440–445 — new regression tests minus deleted vanity tests).

---

## File Structure

### Files modified

| File | Why |
|------|-----|
| `brain/paths.py` | Path traversal sanitization on `get_persona_dir` |
| `brain/memory/store.py` | SQL column-name allowlist in `_list_filter` |
| `brain/engines/dream.py` | Add `persona_name` + `persona_system_prompt` fields; remove hardcoded `_SYSTEM_PROMPT` constant |
| `brain/engines/heartbeat.py` | Fix `_emit_heartbeat_memory` hardcoded persona; tighten `HeartbeatEngine` defaults; update `_try_fire_dream` + `_try_fire_reflex` + `_try_fire_research` to truncate exception strings; atomic `HeartbeatConfig.save`; pass persona fields to DreamEngine |
| `brain/engines/reflex.py` | Remove local `_compute_days_since_human` (moved to utils); remove local `_format_emotion_summary` (moved to utils); use shared helpers |
| `brain/engines/research.py` | Remove orphaned `ResearchLog.load` call; remove local `_compute_days_since_human` + `_format_emotion_summary`; promote `PULL_THRESHOLD` / `COOLDOWN_HOURS` to `__init__` params |
| `brain/migrator/cli.py` | Atomic writes for `reflex_arcs.json` + `interests.json` |
| `brain/bridge/provider.py` | `get_provider("ollama")` raises user-friendly `NotImplementedError` before returning instance |
| `brain/search/factory.py` | `get_searcher("claude-tool")` raises user-friendly `NotImplementedError` before returning instance |
| `brain/cli.py` | Remove `default="nell"` on `--persona` args; improve error message; pass persona fields to DreamEngine + HeartbeatEngine |

### Files created

| File | Why |
|------|-----|
| `brain/utils/memory.py` | Shared `days_since_human(store, now) -> float` |
| `brain/utils/emotion.py` | Shared `format_emotion_summary(emotions) -> str` |
| `tests/unit/brain/utils/test_memory_utils.py` | New shared-helper tests |
| `tests/unit/brain/utils/test_emotion_utils.py` | New shared-helper tests |

### Tests deleted

| Test | Why |
|------|-----|
| `test_research_fire_construction` (`tests/unit/brain/engines/test_research.py`) | Verifies Python's dataclass machinery, not module logic |
| `test_research_result_construction` (`tests/unit/brain/engines/test_research.py`) | Same — vanity construction test |
| `test_research_engine_construction` (`tests/unit/brain/engines/test_research.py`) | Same — vanity construction test |

---

## Task 1: Security hardening — path traversal + SQL allowlist + exception truncation

**Purpose:** Three small defensive patches grouped for efficient review. Fixes Blocker B1 + two Important items.

**Files:**
- Modify: `brain/paths.py`
- Modify: `brain/memory/store.py:368-380` (`_list_filter`)
- Modify: `brain/engines/heartbeat.py` (`_try_fire_reflex` + `_try_fire_research` log lines)
- Modify: `tests/unit/brain/test_paths.py`
- Modify: `tests/unit/brain/memory/test_store.py`

- [ ] **Step 1: Write failing test for path traversal rejection**

Append to `tests/unit/brain/test_paths.py` (create if it doesn't exist — check first with `ls tests/unit/brain/test_paths.py`):

```python
import pytest
from brain.paths import get_persona_dir


def test_get_persona_dir_rejects_path_traversal():
    with pytest.raises(ValueError):
        get_persona_dir("../etc/passwd")


def test_get_persona_dir_rejects_forward_slash():
    with pytest.raises(ValueError):
        get_persona_dir("a/b")


def test_get_persona_dir_rejects_dot_name():
    with pytest.raises(ValueError):
        get_persona_dir("..")


def test_get_persona_dir_rejects_empty():
    with pytest.raises(ValueError):
        get_persona_dir("")


def test_get_persona_dir_accepts_valid_name(monkeypatch, tmp_path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    result = get_persona_dir("nell.sandbox")
    assert result == tmp_path / "personas" / "nell.sandbox"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/test_paths.py -v`
Expected: the 4 rejection tests fail (no validation yet).

- [ ] **Step 3: Add validation to `get_persona_dir`**

In `brain/paths.py`, replace the `get_persona_dir` function:

```python
def get_persona_dir(name: str) -> Path:
    """Return the directory for a specific persona's private data.

    Raises ValueError if `name` could escape the personas/ root via path
    traversal (contains '/', '\\', or equals '.' / '..').
    """
    if not name:
        raise ValueError("Persona name cannot be empty.")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"Invalid persona name: {name!r} "
            "(must not contain '/' or '\\', or be '.' / '..')."
        )
    return get_home() / "personas" / name
```

- [ ] **Step 4: Write failing test for SQL column allowlist**

Append to `tests/unit/brain/memory/test_store.py`:

```python
def test_list_filter_rejects_unknown_column():
    import pytest
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError, match="Invalid filter column"):
            store._list_filter("created_at; DROP TABLE memories--", "x", True, None)
    finally:
        store.close()
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/unit/brain/memory/test_store.py::test_list_filter_rejects_unknown_column -v`
Expected: FAIL — no validation yet.

- [ ] **Step 6: Add allowlist to `_list_filter`**

In `brain/memory/store.py`, at the top of `_list_filter` (after the docstring if any):

```python
_ALLOWED_FILTER_COLUMNS = frozenset({"domain", "memory_type"})


    def _list_filter(
        self, column: str, value: str, active_only: bool, limit: int | None
    ) -> list[Memory]:
        if column not in _ALLOWED_FILTER_COLUMNS:
            raise ValueError(f"Invalid filter column: {column!r}")
        sql = f"SELECT * FROM memories WHERE {column} = ?"
        # ... rest unchanged ...
```

Place `_ALLOWED_FILTER_COLUMNS` as a module-level constant just before the `class MemoryStore` definition. The `_list_filter` body keeps its existing SQL build.

- [ ] **Step 7: Update exception-string truncation in fault-isolation logs**

In `brain/engines/heartbeat.py`, modify the two log lines:

Find:
```python
logger.warning("reflex tick raised; isolating failure: %s", exc)
```
Replace with:
```python
logger.warning("reflex tick raised; isolating: %.200s", exc)
```

Find:
```python
logger.warning("research tick raised; isolating failure: %s", exc)
```
Replace with:
```python
logger.warning("research tick raised; isolating: %.200s", exc)
```

(The `%.200s` format spec truncates any exception message to 200 chars — prevents memory-fragment leaks from format-template KeyErrors into the heartbeat audit log.)

- [ ] **Step 8: Run all affected tests**

Run: `uv run pytest tests/unit/brain/test_paths.py tests/unit/brain/memory/test_store.py tests/unit/brain/engines/test_heartbeat.py -v`
Expected: all pass (new tests + existing).

Run full suite: `uv run pytest -q`
Expected: all 431+ pass.

- [ ] **Step 9: Ruff + format**

Run: `uv run ruff check brain/paths.py brain/memory/store.py brain/engines/heartbeat.py tests/unit/brain/test_paths.py tests/unit/brain/memory/test_store.py && uv run ruff format brain/paths.py brain/memory/store.py brain/engines/heartbeat.py tests/unit/brain/test_paths.py tests/unit/brain/memory/test_store.py`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add brain/paths.py brain/memory/store.py brain/engines/heartbeat.py tests/unit/brain/test_paths.py tests/unit/brain/memory/test_store.py
git commit -m "$(cat <<'EOF'
fix: security hardening — path traversal + SQL allowlist + log truncation

Three surgical patches from the Week 4 audit:

1. get_persona_dir now rejects names containing '/', '\', or equal to
   '.'/'..' — prevents --persona ../../etc from escaping personas/ root.
2. MemoryStore._list_filter validates column name against an allowlist
   (domain, memory_type). Currently-safe pattern hardened against future
   callers. Prevents SQL injection via string-interpolated column name.
3. _try_fire_reflex + _try_fire_research truncate exception strings in
   log output via %.200s, preventing memory-content leaks via
   format-template KeyError messages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Persona-aware DreamEngine

**Purpose:** Dream's hardcoded `_SYSTEM_PROMPT = "You are Nell..."` means every persona's dreams self-identify as Nell. Biggest multi-persona voice leak.

**Files:**
- Modify: `brain/engines/dream.py`
- Modify: `brain/engines/heartbeat.py` (`_try_fire_dream` passes persona fields)
- Modify: `brain/cli.py` (`_dream_handler` passes persona fields)
- Modify: `tests/unit/brain/engines/test_dream.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/brain/engines/test_dream.py`:

```python
def test_dream_system_prompt_uses_persona_name(tmp_path: Path):
    """DreamEngine must render the persona name into its system prompt,
    not hardcode 'Nell'. Multi-persona correctness fix.
    """
    from brain.bridge.provider import FakeProvider
    from brain.engines.dream import DreamEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    captured = {}

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured["system"] = system
            return "DREAM: test"

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="seed", memory_type="conversation", domain="us",
                emotions={"love": 5.0},
            )
        )
        engine = DreamEngine(
            store=store,
            hebbian=hm,
            embeddings=None,
            provider=CapturingProvider(),
            persona_name="Iris",
            persona_system_prompt="You are Iris. Reflect in first person...",
        )
        try:
            engine.run_cycle(lookback_hours=100000)
        except Exception:
            pass  # may raise NoSeedAvailable if setup is incomplete, that's fine

    finally:
        store.close()
        hm.close()

    assert captured.get("system") is not None
    assert "Iris" in captured["system"]
    assert "Nell" not in captured["system"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_dream.py::test_dream_system_prompt_uses_persona_name -v`
Expected: FAIL — `DreamEngine` constructor does not accept `persona_name` / `persona_system_prompt` yet.

- [ ] **Step 3: Add persona fields to DreamEngine + delete hardcoded constant**

In `brain/engines/dream.py`:

Remove the module-level constant (lines 20–24):

```python
_SYSTEM_PROMPT = (
    "You are Nell. You just woke from a dream about interconnected memories. "
    "Reflect in first person, 2-3 sentences, starting with 'DREAM: '. "
    "Be honest and specific, not abstract."
)
```

Update the `DreamEngine` dataclass to add two fields with empty defaults + runtime validation:

```python
@dataclass
class DreamEngine:
    """Composes memory + emotion + LLM bridge into an associative cycle."""

    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache | None
    provider: LLMProvider
    log_path: Path | None = None
    persona_name: str = ""
    persona_system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.persona_name:
            raise ValueError(
                "DreamEngine requires persona_name — construct explicitly, "
                "don't rely on a default."
            )
        if not self.persona_system_prompt:
            raise ValueError(
                "DreamEngine requires persona_system_prompt — construct "
                "explicitly, don't rely on a default."
            )
```

Find the place where `_SYSTEM_PROMPT` was returned (around line 146 in the `_build_prompt` or equivalent method — check the current file layout). Replace the `_SYSTEM_PROMPT` reference with `self.persona_system_prompt`:

```python
return self.persona_system_prompt, "\n".join(parts)
```

- [ ] **Step 4: Update DreamEngine callers**

In `brain/engines/heartbeat.py`, find `_try_fire_dream`:

```python
def _try_fire_dream(self) -> str | None:
    ...
    dream_engine = DreamEngine(
        store=self.store,
        hebbian=self.hebbian,
        embeddings=None,
        provider=self.provider,
        log_path=self.dream_log_path,
    )
```

Update to pass persona fields:

```python
def _try_fire_dream(self) -> str | None:
    ...
    dream_engine = DreamEngine(
        store=self.store,
        hebbian=self.hebbian,
        embeddings=None,
        provider=self.provider,
        log_path=self.dream_log_path,
        persona_name=self.persona_name,
        persona_system_prompt=(
            f"You are {self.persona_name}. You just woke from a dream "
            "about interconnected memories. Reflect in first person, 2-3 "
            "sentences, starting with 'DREAM: '. Be honest and specific, "
            "not abstract."
        ),
    )
```

In `brain/cli.py`, find `_dream_handler` and its `DreamEngine(...)` construction. Add the two new fields:

```python
engine = DreamEngine(
    store=store,
    hebbian=hebbian,
    embeddings=None,
    provider=provider,
    log_path=persona_dir / "dreams.log.jsonl",
    persona_name=args.persona,
    persona_system_prompt=(
        f"You are {args.persona}. You just woke from a dream about "
        "interconnected memories. Reflect in first person, 2-3 sentences, "
        "starting with 'DREAM: '. Be honest and specific, not abstract."
    ),
)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_dream.py tests/unit/brain/engines/test_heartbeat.py tests/unit/brain/engines/test_cli.py -v`
Expected: all pass.

Run full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Ruff + format**

Run: `uv run ruff check brain/engines/dream.py brain/engines/heartbeat.py brain/cli.py && uv run ruff format brain/engines/dream.py brain/engines/heartbeat.py brain/cli.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add brain/engines/dream.py brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_dream.py
git commit -m "$(cat <<'EOF'
fix(dream): make DreamEngine persona-aware

Hardcoded '_SYSTEM_PROMPT = "You are Nell..."' meant every persona's
dreams self-identified as Nell, writing wrong-identity memories into
permanent storage on first use.

DreamEngine now requires persona_name + persona_system_prompt via
__init__ and raises ValueError on empty values. Heartbeat's
_try_fire_dream + cli.py's _dream_handler both pass the persona fields
through from their callers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Multi-persona heartbeat polish

**Purpose:** Two remaining voice leaks in `brain/engines/heartbeat.py`: `_emit_heartbeat_memory` inlines `"You are Nell."`, and `HeartbeatEngine.persona_name`/`persona_system_prompt` default to Nell strings.

**Files:**
- Modify: `brain/engines/heartbeat.py`
- Modify: `tests/unit/brain/engines/test_heartbeat.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/brain/engines/test_heartbeat.py`:

```python
def test_heartbeat_memory_uses_persona_name(tmp_path: Path) -> None:
    """_emit_heartbeat_memory must render persona name into system prompt,
    not hardcode 'Nell'.
    """
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    captured = {}

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured["system"] = system
            return "HEARTBEAT: tended"

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False, research_enabled=False,
        emit_memory="always",
    ).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s", memory_type="conversation", domain="us",
                emotions={"love": 5.0},
            )
        )
        prior = HeartbeatState.fresh("manual")
        prior.last_tick_at = datetime.now(UTC) - timedelta(hours=1)
        prior.save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=CapturingProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            persona_name="Iris",
            persona_system_prompt="You are Iris.",
        )
        engine.run_tick(trigger="manual", dry_run=False)

        # The _emit_heartbeat_memory system prompt should mention Iris, not Nell
        assert captured.get("system") is not None
        assert "Iris" in captured["system"]
        assert "Nell" not in captured["system"]
    finally:
        store.close()
        hm.close()


def test_heartbeat_engine_empty_persona_raises() -> None:
    """HeartbeatEngine must reject empty persona_name / persona_system_prompt
    to force callers to be explicit rather than silently get defaults.
    """
    import pytest
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from pathlib import Path as _Path

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        with pytest.raises(ValueError, match="persona_name"):
            HeartbeatEngine(
                store=store, hebbian=hm, provider=FakeProvider(),
                state_path=_Path("/tmp/a"), config_path=_Path("/tmp/b"),
                dream_log_path=_Path("/tmp/c"), heartbeat_log_path=_Path("/tmp/d"),
                # persona_name omitted → should raise
            )
    finally:
        store.close()
        hm.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py::test_heartbeat_memory_uses_persona_name tests/unit/brain/engines/test_heartbeat.py::test_heartbeat_engine_empty_persona_raises -v`
Expected: both fail.

- [ ] **Step 3: Fix `_emit_heartbeat_memory` to use `self.persona_name`**

In `brain/engines/heartbeat.py`, find `_emit_heartbeat_memory`. Replace the hardcoded system string:

```python
system = (
    "You are Nell. You just finished a background heartbeat cycle — "
    "decay applied, memory graph tended. Reflect in first person, "
    "one short sentence, starting with 'HEARTBEAT: '."
)
```

With:

```python
system = (
    f"You are {self.persona_name}. You just finished a background "
    "heartbeat cycle — decay applied, memory graph tended. Reflect in "
    "first person, one short sentence, starting with 'HEARTBEAT: '."
)
```

- [ ] **Step 4: Tighten HeartbeatEngine defaults**

In the `HeartbeatEngine` dataclass, change the defaults:

```python
persona_name: str = "nell"
persona_system_prompt: str = "You are Nell."
```

Replace with:

```python
persona_name: str = ""
persona_system_prompt: str = ""
```

Add `__post_init__` to the `HeartbeatEngine` dataclass (immediately after the field definitions, before `run_tick`):

```python
def __post_init__(self) -> None:
    if not self.persona_name:
        raise ValueError(
            "HeartbeatEngine requires persona_name — construct explicitly, "
            "don't rely on a default."
        )
    if not self.persona_system_prompt:
        raise ValueError(
            "HeartbeatEngine requires persona_system_prompt — construct "
            "explicitly, don't rely on a default."
        )
```

- [ ] **Step 5: Update any existing heartbeat tests that omit persona fields**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v`

For any test that fails because it constructs `HeartbeatEngine` without `persona_name` / `persona_system_prompt`, add those fields:

```python
engine = HeartbeatEngine(
    ...,  # existing fields
    persona_name="Nell",
    persona_system_prompt="You are Nell.",
)
```

Search for `HeartbeatEngine(` in the test file and add the two fields to each constructor call that's missing them. (Some tests likely already pass them — they're fine.)

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v`
Expected: all pass (including both new tests).

Full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 7: Ruff + format**

Run: `uv run ruff check brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py && uv run ruff format brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "$(cat <<'EOF'
fix(heartbeat): remove Nell-hardcoding — require explicit persona

_emit_heartbeat_memory previously inlined 'You are Nell.' in its system
prompt. HeartbeatEngine.persona_name / persona_system_prompt previously
defaulted to 'nell' / 'You are Nell.' — meaning any caller that
forgot to pass them silently got Nell's identity.

Now _emit_heartbeat_memory uses self.persona_name. HeartbeatEngine
fields default to empty strings + __post_init__ raises ValueError
if either is empty. All existing heartbeat tests updated to pass
explicit persona values.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI `--persona` defaults + missing-persona error + stub factory guards

**Purpose:** Two UX issues for new users: (1) all `--persona` args default to `nell` and the missing-persona error misdirects them to run an OG migration they have no source for, (2) `--provider ollama` and `--searcher claude-tool` reach the stub raise with a cryptic `NotImplementedError` — should raise a user-friendly `ValueError` at factory time instead.

**Files:**
- Modify: `brain/cli.py`
- Modify: `brain/bridge/provider.py`
- Modify: `brain/search/factory.py`
- Modify: `tests/unit/brain/bridge/test_provider.py` (if exists — else skip, covered by next task's tests)
- Modify: `tests/unit/brain/search/test_factory.py` (may need creation)

- [ ] **Step 1: Write failing tests for factory guards**

Check if `tests/unit/brain/search/test_factory.py` exists. If not, create it:

```python
"""Tests for brain.search.factory.get_searcher()."""

from __future__ import annotations

import pytest

from brain.search.base import NoopWebSearcher
from brain.search.factory import get_searcher


def test_get_searcher_ddgs():
    s = get_searcher("ddgs")
    assert s.name() == "ddgs"


def test_get_searcher_noop():
    s = get_searcher("noop")
    assert isinstance(s, NoopWebSearcher)


def test_get_searcher_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown searcher"):
        get_searcher("not_a_searcher")


def test_get_searcher_claude_tool_raises_user_friendly_error():
    """claude-tool is a Phase 1 stub — factory should give user a clear
    message instead of returning an instance that crashes on first use."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        get_searcher("claude-tool")
```

Check if `tests/unit/brain/bridge/test_provider.py` exists. Add the analogous test for `ollama`:

```python
def test_get_provider_ollama_raises_user_friendly_error():
    """ollama is a Phase 1 stub — factory should give user a clear
    message instead of returning an instance that crashes on first use."""
    import pytest
    from brain.bridge.provider import get_provider
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        get_provider("ollama")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/search/test_factory.py tests/unit/brain/bridge/ -v`
Expected: the 2 new tests fail (factory returns stubs).

- [ ] **Step 3: Add factory guards**

In `brain/search/factory.py`:

```python
"""Searcher factory — resolve a name to an instance."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher, WebSearcher
from brain.search.ddgs_searcher import DdgsWebSearcher


def get_searcher(name: str) -> WebSearcher:
    """Resolve a searcher identifier to an instance.

    Raises ValueError on unknown name. Raises NotImplementedError for
    Phase 1 stubs (claude-tool) — with a user-friendly message pointing
    to the working alternatives.
    """
    if name == "ddgs":
        return DdgsWebSearcher()
    if name == "noop":
        return NoopWebSearcher()
    if name == "claude-tool":
        raise NotImplementedError(
            "The 'claude-tool' searcher is not yet implemented (Phase 1 stub). "
            "Use 'ddgs' (default, free, no API key) or 'noop' (for tests)."
        )
    raise ValueError(f"Unknown searcher: {name!r}")
```

(Note: we no longer import `ClaudeToolWebSearcher` here since the factory raises before instantiation. The stub class can stay in `claude_tool_searcher.py` unchanged for documentation purposes; it's just not reachable via the factory.)

In `brain/bridge/provider.py`, find the `get_provider` function. Replace the ollama branch:

```python
def get_provider(name: str) -> LLMProvider:
    """Resolve a provider identifier to an instance. Raises ValueError on unknown."""
    if name == "fake":
        return FakeProvider()
    if name == "claude-cli":
        return ClaudeCliProvider()
    if name == "ollama":
        raise NotImplementedError(
            "The 'ollama' provider is not yet implemented (Phase 1 stub). "
            "Use 'claude-cli' (default, subscription-backed) or 'fake' (for tests)."
        )
    raise ValueError(f"Unknown provider: {name!r}")
```

- [ ] **Step 4: Remove `default="nell"` from CLI args**

In `brain/cli.py`, find every subparser `--persona` argument. Currently they look like:

```python
hb_sub.add_argument("--persona", default="nell")
```

Replace each with:

```python
hb_sub.add_argument("--persona", required=True, help="Persona name (required; use `nell migrate` to port OG data or create personas/<name>/ manually).")
```

Apply to ALL of these subparser `--persona` args:
- `dream_sub` (in `_build_parser` dream section)
- `hb_sub` (heartbeat)
- `rf_sub` (reflex)
- `r_sub` (research)
- `i_list` (interest list)
- `i_add` (interest add)
- `i_bump` (interest bump)

For the migrator `--install-as` (already `required=True` via the mutually-exclusive group), no change needed.

- [ ] **Step 5: Improve missing-persona error message**

In `brain/cli.py`, find every `FileNotFoundError` raise that mentions "run `nell migrate --install-as {args.persona} first`". Currently it's:

```python
raise FileNotFoundError(
    f"No persona directory at {persona_dir} — "
    f"run `nell migrate --install-as {args.persona}` first."
)
```

Replace each occurrence with:

```python
raise FileNotFoundError(
    f"No persona directory at {persona_dir}. "
    "If you're porting existing OG NellBrain data, run `nell migrate "
    f"--input /path/to/og/data --install-as {args.persona}`. "
    f"Otherwise create {persona_dir} manually to start a fresh persona."
)
```

Apply to every handler that raises this (`_dream_handler`, `_heartbeat_handler`, `_reflex_handler`, `_research_handler`, `_interest_list_handler`, `_interest_add_handler`, `_interest_bump_handler`).

- [ ] **Step 6: Run all affected tests**

Run: `uv run pytest -q`
Expected: all pass. Some CLI tests may need updating if they relied on `--persona` defaulting — update them to pass `--persona testpersona` explicitly.

- [ ] **Step 7: Ruff + format**

Run: `uv run ruff check brain/cli.py brain/bridge/provider.py brain/search/factory.py tests/unit/brain/search/test_factory.py tests/unit/brain/bridge/ && uv run ruff format brain/cli.py brain/bridge/provider.py brain/search/factory.py tests/unit/brain/search/test_factory.py tests/unit/brain/bridge/`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add brain/cli.py brain/bridge/provider.py brain/search/factory.py tests/unit/brain/search/ tests/unit/brain/bridge/
git commit -m "$(cat <<'EOF'
fix(cli): --persona required; stub factories raise user-friendly errors

Two new-user UX fixes from the audit:

1. All --persona CLI args are now required=True (was default="nell").
   Missing-persona error message no longer misdirects new users toward
   an OG NellBrain migration they have no source for — it explains
   both paths (migrate existing data, or create personas/<name>/ fresh).

2. get_provider("ollama") and get_searcher("claude-tool") now raise
   NotImplementedError with a user-friendly message BEFORE returning
   a stub instance that would crash on first use.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Atomic writes — `HeartbeatConfig.save` + migrator interests/arcs

**Purpose:** Blocker B3 (`HeartbeatConfig.save` non-atomic — crash mid-save can lose Hana's tuned config) + Important migrator atomicity.

**Files:**
- Modify: `brain/engines/heartbeat.py` (`HeartbeatConfig.save`)
- Modify: `brain/migrator/cli.py` (reflex_arcs + interests writes)
- Modify: `tests/unit/brain/engines/test_heartbeat.py` (HeartbeatConfig atomicity test)

- [ ] **Step 1: Write failing test for HeartbeatConfig.save atomicity**

Append to `tests/unit/brain/engines/test_heartbeat.py`:

```python
def test_heartbeat_config_save_is_atomic(tmp_path: Path) -> None:
    """HeartbeatConfig.save must use .new + os.replace so a crash mid-write
    leaves either the old valid file or the new valid file — never a
    partial write. Corruption would silently revert to defaults on reload,
    losing user-tuned values.
    """
    path = tmp_path / "cfg.json"
    HeartbeatConfig(dream_every_hours=12.0, reflex_enabled=False).save(path)
    assert path.exists()
    # .new temp must not linger
    assert not path.with_suffix(path.suffix + ".new").exists()
    # Reloads cleanly
    loaded = HeartbeatConfig.load(path)
    assert loaded.dream_every_hours == 12.0
    assert loaded.reflex_enabled is False
```

- [ ] **Step 2: Run to verify (test may pass if save happens to succeed — we're asserting .new cleanup)**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py::test_heartbeat_config_save_is_atomic -v`
Expected: currently passes because the .new file isn't created in the non-atomic path either. This test is a regression guard — it'll start failing if someone breaks atomicity in the future.

- [ ] **Step 3: Make `HeartbeatConfig.save` atomic**

In `brain/engines/heartbeat.py`, find `HeartbeatConfig.save`:

```python
def save(self, path: Path) -> None:
    """Write config JSON to path (non-atomic — config is user-edited)."""
    payload = {
        ...
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
```

Replace with:

```python
def save(self, path: Path) -> None:
    """Atomic save via .new + os.replace.

    A crash mid-write leaves either the previous valid file or the new
    valid file — never a partial write that corrupts the user's config.
    """
    payload = {
        ...  # keep existing payload dict unchanged
    }
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
```

(The `payload = {...}` dict body is unchanged. Only the write mechanism changes.)

- [ ] **Step 4: Make migrator reflex_arcs.json + interests.json writes atomic**

In `brain/migrator/cli.py`, find the reflex arcs write:

```python
reflex_arcs_target.write_text(
    _json.dumps({"version": 1, "arcs": og_arcs}, indent=2) + "\n",
    encoding="utf-8",
)
```

Replace with:

```python
_reflex_tmp = reflex_arcs_target.with_suffix(reflex_arcs_target.suffix + ".new")
_reflex_tmp.write_text(
    _json.dumps({"version": 1, "arcs": og_arcs}, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(_reflex_tmp, reflex_arcs_target)
```

Find the interests write (in the `# ---- interests ----` block):

```python
interests_target.write_text(
    _json.dumps({"version": 1, "interests": og_interests}, indent=2) + "\n",
    encoding="utf-8",
)
```

Replace with:

```python
_interests_tmp = interests_target.with_suffix(interests_target.suffix + ".new")
_interests_tmp.write_text(
    _json.dumps({"version": 1, "interests": og_interests}, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(_interests_tmp, interests_target)
```

Add `import os` to the top of `brain/migrator/cli.py` if not already present.

- [ ] **Step 5: Run affected tests**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py tests/unit/brain/migrator/ -v`
Expected: all pass.

Full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Ruff + format**

Run: `uv run ruff check brain/engines/heartbeat.py brain/migrator/cli.py tests/unit/brain/engines/test_heartbeat.py && uv run ruff format brain/engines/heartbeat.py brain/migrator/cli.py tests/unit/brain/engines/test_heartbeat.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add brain/engines/heartbeat.py brain/migrator/cli.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "$(cat <<'EOF'
fix: atomic writes for HeartbeatConfig + migrator reflex/interests JSON

HeartbeatConfig.save was non-atomic — a crash mid-write corrupted the
file and HeartbeatConfig.load silently reverted to defaults, losing
any user-tuned values. Now uses .new + os.replace.

Migrator writes for reflex_arcs.json + interests.json also made
atomic. --install-as mode was already protected (writes into work_dir
then atomic rename), but --output mode had the naked write.

All persona-state JSON writes now consistent with the established
.new + os.replace atomic pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Extract shared helpers to `brain/utils/`

**Purpose:** Eliminate duplication of `_compute_days_since_human` and `_format_emotion_summary` across reflex + research engines.

**Files:**
- Create: `brain/utils/memory.py`
- Create: `brain/utils/emotion.py`
- Create: `tests/unit/brain/utils/test_memory_utils.py`
- Create: `tests/unit/brain/utils/test_emotion_utils.py`
- Modify: `brain/engines/reflex.py` (remove local helpers, import shared)
- Modify: `brain/engines/research.py` (remove local helpers, import shared)

- [ ] **Step 1: Write failing tests for memory helper**

Create `tests/unit/brain/utils/test_memory_utils.py`:

```python
"""Tests for brain.utils.memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.memory.store import Memory, MemoryStore
from brain.utils.memory import days_since_human


def test_days_since_human_returns_999_when_no_conversations():
    store = MemoryStore(":memory:")
    try:
        result = days_since_human(store, datetime.now(UTC))
        assert result == 999.0
    finally:
        store.close()


def test_days_since_human_computes_delta():
    store = MemoryStore(":memory:")
    try:
        mem = Memory.create_new(
            content="x", memory_type="conversation", domain="us", emotions={},
        )
        store.create(mem)
        # Backdate 48h
        store._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(hours=48)).isoformat(), mem.id),
        )
        store._conn.commit()
        result = days_since_human(store, datetime.now(UTC))
        assert 1.9 < result < 2.1  # ~2 days
    finally:
        store.close()
```

- [ ] **Step 2: Write failing tests for emotion helper**

Create `tests/unit/brain/utils/test_emotion_utils.py`:

```python
"""Tests for brain.utils.emotion."""

from __future__ import annotations

from brain.utils.emotion import format_emotion_summary


def test_format_emotion_summary_empty():
    assert format_emotion_summary({}) == ""


def test_format_emotion_summary_top_5_descending():
    emotions = {
        "love": 8.5,
        "tenderness": 7.1,
        "defiance": 3.0,
        "creative_hunger": 6.2,
        "grief": 5.0,
        "awe": 2.0,
    }
    result = format_emotion_summary(emotions)
    lines = result.split("\n")
    assert len(lines) == 5
    assert lines[0] == "- love: 8.5/10"
    assert lines[1] == "- tenderness: 7.1/10"


def test_format_emotion_summary_fewer_than_5():
    emotions = {"love": 6.0, "defiance": 3.0}
    result = format_emotion_summary(emotions)
    assert result == "- love: 6.0/10\n- defiance: 3.0/10"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/brain/utils/ -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 4: Create utils modules**

Create `brain/utils/memory.py`:

```python
"""Shared memory helpers used by multiple engines."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.memory.store import MemoryStore


def days_since_human(store: MemoryStore, now: datetime) -> float:
    """Days since the most recent memory_type='conversation'. 999.0 if none.

    Used by reflex + research engines to gate on persona-silence duration.
    """
    convos = store.list_by_type("conversation", active_only=True, limit=1)
    if not convos:
        return 999.0
    latest = convos[0].created_at
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    return (now - latest).total_seconds() / 86400.0
```

Create `brain/utils/emotion.py`:

```python
"""Shared emotion helpers used by multiple engines."""

from __future__ import annotations

from collections.abc import Mapping


def format_emotion_summary(emotions: Mapping[str, float]) -> str:
    """Return the top-5 emotions formatted as '- name: X.X/10' lines.

    Empty input returns an empty string. Used by reflex + research
    engines for LLM prompt context.
    """
    top = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return "\n".join(f"- {name}: {value:.1f}/10" for name, value in top)
```

- [ ] **Step 5: Run utils tests**

Run: `uv run pytest tests/unit/brain/utils/ -v`
Expected: all pass.

- [ ] **Step 6: Update reflex.py to use shared helpers**

In `brain/engines/reflex.py`:

Remove the local `_compute_days_since_human` function (module-level).
Remove the local `_format_emotion_summary` function (module-level).

Add imports at the top:

```python
from brain.utils.emotion import format_emotion_summary
from brain.utils.memory import days_since_human
```

Find the call site of `_compute_days_since_human(self.store, now)` — replace with `days_since_human(self.store, now)`.

Find the call site of `_format_emotion_summary(emotions)` — replace with `format_emotion_summary(emotions)`.

- [ ] **Step 7: Update research.py to use shared helpers**

In `brain/engines/research.py`:

Remove the local `_compute_days_since_human` function.

Find the inline emotion-summary logic in `_render_prompt`:

```python
top = sorted(emo_state.emotions.items(), key=lambda kv: kv[1], reverse=True)[:5]
emo_summary = "\n".join(f"- {name}: {value:.1f}/10" for name, value in top) or "(neutral)"
```

Replace with:

```python
emo_summary = format_emotion_summary(emo_state.emotions) or "(neutral)"
```

Add imports:

```python
from brain.utils.emotion import format_emotion_summary
from brain.utils.memory import days_since_human
```

Find the call site of `_compute_days_since_human(self.store, now)` — replace with `days_since_human(self.store, now)`.

- [ ] **Step 8: Run affected tests**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py tests/unit/brain/engines/test_research.py tests/unit/brain/utils/ -v`
Expected: all pass.

Full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 9: Ruff + format**

Run: `uv run ruff check brain/utils/ brain/engines/reflex.py brain/engines/research.py tests/unit/brain/utils/ && uv run ruff format brain/utils/ brain/engines/reflex.py brain/engines/research.py tests/unit/brain/utils/`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add brain/utils/ brain/engines/reflex.py brain/engines/research.py tests/unit/brain/utils/
git commit -m "$(cat <<'EOF'
refactor: extract days_since_human + format_emotion_summary to brain/utils

Both helpers were duplicated between reflex.py and research.py (8 + 3
lines each). research.py's emotion-summary was inlined, reflex's was
a helper — same logic, slightly different shape, destined to drift.

Shared module keeps invariants (999.0 sentinel, top-5 format) consistent
across all current and future engines.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Dead code removal + PULL_THRESHOLD fix

**Purpose:** Remove orphaned `ResearchLog.load` call, 3 vanity construction tests, and promote `PULL_THRESHOLD` / `COOLDOWN_HOURS` from class-level constants (set post-construction) to proper `__init__` parameters.

**Files:**
- Modify: `brain/engines/research.py`
- Modify: `brain/engines/heartbeat.py` (`_try_fire_research` — remove post-construction mutation)
- Modify: `tests/unit/brain/engines/test_research.py` (remove 3 vanity tests)

- [ ] **Step 1: Remove orphaned `ResearchLog.load` call**

In `brain/engines/research.py`, find:

```python
log = ResearchLog.load(self.research_log_path)
```

at the top of `run_tick`. This variable is only read once — via `log.appended(fire).save(...)` at the end — and appending a fire to an empty log gives the same result as appending to a freshly-loaded log for cooldown purposes (cooldown is driven by `Interest.last_researched_at`, not the log).

BUT: we still want the log to grow cumulatively. Deleting the `load` would lose the full fire history.

**Resolution:** Keep the `load` call, but add a test that documents why it's kept. This was a false positive in the audit — the load IS necessary to append-not-replace. Skip this change.

**Alternative:** Rename the variable to make the intent clear, or add a comment.

Modify the `run_tick` line:

```python
log = ResearchLog.load(self.research_log_path)
```

Add a comment above:

```python
# Load existing log so new fires append cumulatively (rather than
# clobbering prior fire history on save). Cooldown logic is driven by
# Interest.last_researched_at, not this log — the log is audit-trail only.
log = ResearchLog.load(self.research_log_path)
```

- [ ] **Step 2: Promote PULL_THRESHOLD + COOLDOWN_HOURS to __init__ params**

In `brain/engines/research.py`, find the `ResearchEngine` dataclass. Currently:

```python
@dataclass
class ResearchEngine:
    store: MemoryStore
    provider: LLMProvider
    searcher: WebSearcher
    persona_name: str
    persona_system_prompt: str
    interests_path: Path
    research_log_path: Path
    default_interests_path: Path

    PULL_THRESHOLD: float = 6.0
    COOLDOWN_HOURS: float = 24.0
```

Move the two constants into the dataclass fields (keeping them configurable per-instance):

```python
@dataclass
class ResearchEngine:
    store: MemoryStore
    provider: LLMProvider
    searcher: WebSearcher
    persona_name: str
    persona_system_prompt: str
    interests_path: Path
    research_log_path: Path
    default_interests_path: Path
    pull_threshold: float = 6.0
    cooldown_hours: float = 24.0
```

(Lowercase naming — these are now instance attributes, not class-level constants.)

Update references inside `run_tick` — find:

```python
eligible = interests.list_eligible(
    pull_threshold=self.PULL_THRESHOLD,
    cooldown_hours=self.COOLDOWN_HOURS,
    now=now,
)
```

Replace with:

```python
eligible = interests.list_eligible(
    pull_threshold=self.pull_threshold,
    cooldown_hours=self.cooldown_hours,
    now=now,
)
```

- [ ] **Step 3: Update heartbeat's `_try_fire_research` to pass the config value**

In `brain/engines/heartbeat.py`, find `_try_fire_research`:

```python
try:
    engine = ResearchEngine(
        store=self.store,
        provider=self.provider,
        searcher=self.searcher,
        persona_name=self.persona_name,
        persona_system_prompt=self.persona_system_prompt,
        interests_path=self.interests_path,
        research_log_path=self.research_log_path,
        default_interests_path=self.default_interests_path,
    )
    engine.PULL_THRESHOLD = 6.0
    engine.COOLDOWN_HOURS = config.research_cooldown_hours_per_interest
    result = engine.run_tick(trigger=trigger, dry_run=dry_run)
```

Replace with:

```python
try:
    engine = ResearchEngine(
        store=self.store,
        provider=self.provider,
        searcher=self.searcher,
        persona_name=self.persona_name,
        persona_system_prompt=self.persona_system_prompt,
        interests_path=self.interests_path,
        research_log_path=self.research_log_path,
        default_interests_path=self.default_interests_path,
        pull_threshold=6.0,
        cooldown_hours=config.research_cooldown_hours_per_interest,
    )
    result = engine.run_tick(trigger=trigger, dry_run=dry_run)
```

(The post-construction mutation lines are gone — both are now constructor args.)

- [ ] **Step 4: Remove 3 vanity construction tests**

In `tests/unit/brain/engines/test_research.py`, delete these three test functions:

- `test_research_fire_construction`
- `test_research_result_construction`
- `test_research_engine_construction`

(They just assert dataclass field assignment — covered implicitly by the `run_tick` tests.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_research.py tests/unit/brain/engines/test_heartbeat.py -v`
Expected: all pass (with fewer research tests — 3 removed).

Full suite: `uv run pytest -q`
Expected: all green, total count down by 3 from T7's deletions (net positive across the plan thanks to new tests in earlier tasks).

- [ ] **Step 6: Ruff + format**

Run: `uv run ruff check brain/engines/research.py brain/engines/heartbeat.py tests/unit/brain/engines/test_research.py && uv run ruff format brain/engines/research.py brain/engines/heartbeat.py tests/unit/brain/engines/test_research.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add brain/engines/research.py brain/engines/heartbeat.py tests/unit/brain/engines/test_research.py
git commit -m "$(cat <<'EOF'
refactor(research): dead code cleanup + promote thresholds to __init__

- PULL_THRESHOLD / COOLDOWN_HOURS promoted from class-level constants
  (set post-construction via instance mutation — anti-pattern) to
  lowercase __init__ parameters pull_threshold / cooldown_hours.
  Heartbeat's _try_fire_research passes them directly, no more instance
  attribute assignment.
- ResearchLog.load kept (it's the append-target for audit-trail
  purposes), added comment explaining why.
- Removed 3 vanity construction tests (test_research_fire_construction
  et al) that only verified Python's dataclass machinery.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Smoke test + final review + PR

**Purpose:** Validate everything still works end-to-end against Nell's real migrated persona. Open PR.

**Files:** none created; verification only.

- [ ] **Step 1: Full suite + hard rules**

Run: `uv run pytest -q`
Expected: all green, total count approx 440–445.

Run: `rg -l 'import anthropic|from anthropic' brain/`
Expected: zero matches.

Run: `uv run ruff check && uv run ruff format --check`
Expected: clean.

- [ ] **Step 2: Smoke test against Nell's sandbox**

Run: `uv run nell interest list --persona nell.sandbox`
Expected: shows Nell's 2 interests (Lispector + Hana).

Run: `uv run nell research --persona nell.sandbox --provider fake --searcher noop --dry-run`
Expected: dry-run output, no crash.

Run: `uv run nell heartbeat --persona nell.sandbox --provider fake --searcher noop --trigger manual`
Expected: full tick output with correct persona name.

- [ ] **Step 3: Verify multi-persona correctness**

Run:
```bash
uv run python -c "
from brain.paths import get_persona_dir
import pytest
try:
    get_persona_dir('../escape')
    print('FAIL: traversal allowed')
except ValueError as e:
    print(f'OK: traversal blocked — {e}')
"
```
Expected: `OK: traversal blocked — ...`

Run:
```bash
uv run python -c "
from brain.bridge.provider import get_provider
try:
    get_provider('ollama')
    print('FAIL: ollama stub returned')
except NotImplementedError as e:
    print(f'OK: ollama stub raises — {str(e)[:60]}')
"
```
Expected: `OK: ollama stub raises — The 'ollama' provider is not yet implemented...`

Run:
```bash
uv run python -c "
from brain.search.factory import get_searcher
try:
    get_searcher('claude-tool')
    print('FAIL: claude-tool stub returned')
except NotImplementedError as e:
    print(f'OK: claude-tool stub raises — {str(e)[:60]}')
"
```
Expected: `OK: claude-tool stub raises — The 'claude-tool' searcher is not yet implemented...`

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin week-4-audit-cleanup
gh pr create --title "Week 4 audit cleanup — blockers + important fixes" --body "$(cat <<'EOF'
## Summary

Addresses all **Blocking + Important** findings from the Week 4 post-ship audit. Nits are deferred for manual review with Hana.

**5 Blockers fixed:**
- Path traversal via `--persona` arg (`paths.py`)
- `dream.py` hardcoded `"You are Nell."` (every persona's dreams identified as Nell)
- `HeartbeatConfig.save` non-atomic (crash mid-save lost tuned config)
- `_compute_days_since_human` + `_format_emotion_summary` duplicated across engines
- `ResearchLog.load` orphaned-load concern (confirmed necessary, documented)

**Important items fixed:**
- `_emit_heartbeat_memory` inline "You are Nell." (persona-aware now)
- `HeartbeatEngine` persona defaults now raise if empty
- All CLI `--persona` args now `required=True`
- Missing-persona error message no longer misdirects new users
- `get_provider("ollama")` + `get_searcher("claude-tool")` raise friendly NotImplementedError
- Migrator `reflex_arcs.json` + `interests.json` writes atomic
- SQL column-name allowlist in `_list_filter`
- Exception strings truncated in fault-isolation log paths
- `PULL_THRESHOLD` / `COOLDOWN_HOURS` promoted to __init__ params (no more post-construction mutation)
- 3 vanity construction tests removed

## Commits

Seven focused commits, each addressing a theme:
1. Security hardening (path traversal + SQL + log truncation)
2. DreamEngine persona-aware
3. HeartbeatEngine persona-aware
4. CLI --persona required + stub factory guards
5. Atomic writes (HeartbeatConfig + migrator)
6. Shared utils — days_since_human + format_emotion_summary
7. Dead code cleanup + research threshold __init__ params

## Test plan

- [x] Full suite green (~440/~440)
- [x] `rg 'import anthropic' brain/` → 0
- [x] ruff clean
- [x] Nell's sandbox: `nell interest list` / `nell research --dry-run` / `nell heartbeat` all work
- [x] Path traversal blocked via `get_persona_dir`
- [x] Stub factories raise friendly errors
- [x] Multi-persona correctness: DreamEngine + HeartbeatEngine require explicit persona_name

## Deferred (by design — nit cleanup with Hana later)

- `OllamaProvider` default model `"nell-dpo"`
- `vocabulary.py` documentation of `nell_specific` emotions
- `_VALID_EMIT_MODES` redundant with `EmitMemoryMode` Literal
- Various docstring + comment clarity items
- `_verify_sources_unchanged` size-only check
- `default_reflex_arcs.json` literal-brace handling

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Acceptance Criteria

All of the following must be true:

1. `uv run pytest -q` green (~440 tests).
2. `rg -l 'import anthropic' brain/` returns zero matches.
3. `uv run ruff check && uv run ruff format --check` both clean.
4. `get_persona_dir("../etc")` raises `ValueError`.
5. `get_provider("ollama")` and `get_searcher("claude-tool")` raise `NotImplementedError` with user-friendly messages.
6. `DreamEngine(...)` without `persona_name` raises `ValueError`.
7. `HeartbeatEngine(...)` without `persona_name` raises `ValueError`.
8. `HeartbeatConfig.save` leaves no `.new` tempfile after write.
9. `MemoryStore._list_filter` rejects non-allowlisted column names.
10. No `_SYSTEM_PROMPT` module-level constant in `brain/engines/dream.py`.
11. No `"You are Nell."` hardcoded in `brain/engines/heartbeat.py` code paths reachable outside explicit tests.
12. `brain/engines/reflex.py` and `brain/engines/research.py` both import `days_since_human` from `brain.utils.memory` and `format_emotion_summary` from `brain.utils.emotion`.
13. `ResearchEngine` constructor accepts `pull_threshold` and `cooldown_hours` parameters; no post-construction mutation of `PULL_THRESHOLD`/`COOLDOWN_HOURS`.
14. All `--persona` CLI args are `required=True`.
15. End-to-end smoke: Nell's sandbox still works with all 4 engines + all CLI subcommands.
