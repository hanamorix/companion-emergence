# Week 4 — Dream Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `brain/bridge/provider.py` (LLM provider abstraction with Claude CLI default) + `brain/engines/dream.py` (associative dream cycle) + `nell dream` subcommand. First Week 4 engine; template for heartbeat/reflex/research to follow.

**Architecture:** Three layers. Bridge layer is LLM-agnostic (ABC + concrete providers + factory). Engine layer composes memory + emotion + bridge into a single `run_cycle()`. CLI layer parses args and dispatches.

**Tech Stack:** Python 3.12 stdlib (subprocess, hashlib, json, argparse), numpy (already a Week 3 dep), pytest. No new third-party dependencies.

---

## Context: what already exists (Week 3.5 state)

Main branch HEAD: `68fdd43` (Week 4 spec commit). Last code merge: `ea38701` (Week 3.5 migrator).

- 254 tests passing. Ruff clean.
- `brain/emotion/` (Week 2), `brain/memory/` (Week 3), `brain/migrator/` (Week 3.5) all live.
- Nell persona migrated at `~/Library/Application Support/companion-emergence/personas/nell/` — 1,142 memories + 4,404 Hebbian edges.
- `brain/cli.py` has `_STUB_COMMANDS` with `"dream"` as a stub — this task wires it to the real engine.

Feature branch: `week-4-dream` (to be created off `main`).

---

## File structure

```
brain/
├── bridge/                             (NEW)
│   ├── __init__.py                     (T1 — exports)
│   └── provider.py                     (T1 — ABC + 3 providers + factory)
├── engines/                            (NEW)
│   ├── __init__.py                     (T2 — exports)
│   └── dream.py                        (T2 — DreamEngine, run_cycle, DreamResult)
└── cli.py                              (T3 — wire nell dream subcommand)

tests/unit/brain/bridge/
├── __init__.py                         (T1)
└── test_provider.py                    (T1)
tests/unit/brain/engines/
├── __init__.py                         (T2)
└── test_dream.py                       (T2)
tests/unit/brain/test_cli.py            (T3 — remove 'dream' from stubs expected list)
```

---

## Dependency order

T1 (bridge) → T2 (dream engine uses `LLMProvider`) → T3 (CLI wires both) → T4 (close-out).

Execute in order. 1 → 2 → 3 → 4.

---

## Task 1: `brain/bridge/provider.py` — LLM provider abstraction

**Goal:** Ship `LLMProvider` ABC with three concrete implementations: `FakeProvider` (deterministic hash-echo for tests), `ClaudeCliProvider` (subprocess wrapper around `claude -p`), `OllamaProvider` (stub raising `NotImplementedError`). Plus `get_provider(name)` factory.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/bridge/__init__.py`
- Create: `/Users/hanamori/companion-emergence/brain/bridge/provider.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/bridge/__init__.py` (empty)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/bridge/test_provider.py`

- [ ] **Step 1: Create feature branch + empty package dirs**

```bash
cd /Users/hanamori/companion-emergence
git checkout main
git pull origin main
git checkout -b week-4-dream
mkdir -p brain/bridge tests/unit/brain/bridge
touch tests/unit/brain/bridge/__init__.py
```

- [ ] **Step 2: Write `brain/bridge/__init__.py`**

```python
"""LLM provider abstraction for companion-emergence engines.

Exports LLMProvider ABC + three concrete providers + factory.
See docs/superpowers/specs/2026-04-23-week-4-dream-engine-design.md.
"""

from brain.bridge.provider import (
    ClaudeCliProvider,
    FakeProvider,
    LLMProvider,
    OllamaProvider,
    get_provider,
)

__all__ = [
    "ClaudeCliProvider",
    "FakeProvider",
    "LLMProvider",
    "OllamaProvider",
    "get_provider",
]
```

- [ ] **Step 3: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/bridge/test_provider.py`:

```python
"""Tests for brain.bridge.provider — LLM provider abstraction."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.provider import (
    ClaudeCliProvider,
    FakeProvider,
    LLMProvider,
    OllamaProvider,
    get_provider,
)


def test_llm_provider_is_abstract() -> None:
    """LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_fake_provider_is_deterministic() -> None:
    """Same (prompt, system) produces the same output every call."""
    p = FakeProvider()
    a = p.generate("hello", system="be helpful")
    b = p.generate("hello", system="be helpful")
    assert a == b


def test_fake_provider_different_prompts_differ() -> None:
    """Different prompts → different outputs."""
    p = FakeProvider()
    assert p.generate("a") != p.generate("b")


def test_fake_provider_name() -> None:
    """FakeProvider.name() returns 'fake'."""
    assert FakeProvider().name() == "fake"


def test_fake_provider_output_has_dream_prefix() -> None:
    """Fake output starts with 'DREAM:' so downstream dream engine logic works."""
    assert FakeProvider().generate("anything").startswith("DREAM:")


def test_ollama_provider_raises_not_implemented() -> None:
    """OllamaProvider.generate raises NotImplementedError with a clear message."""
    p = OllamaProvider()
    with pytest.raises(NotImplementedError, match="stub"):
        p.generate("anything")


def test_ollama_provider_name_includes_model() -> None:
    """OllamaProvider.name() includes the model identifier."""
    assert OllamaProvider(model="nell-dpo").name() == "ollama:nell-dpo"


def test_claude_cli_provider_builds_expected_command() -> None:
    """ClaudeCliProvider spawns `claude -p <prompt> --output-format json`."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"result": "DREAM: test output"})
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        p = ClaudeCliProvider(model="sonnet")
        out = p.generate("test prompt", system="you are helpful")

    assert out == "DREAM: test output"
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "test prompt" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd


def test_claude_cli_provider_forwards_system_prompt() -> None:
    """ClaudeCliProvider passes the system prompt via --system-prompt."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"result": "ok"})

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ClaudeCliProvider().generate("p", system="you are nell")

    cmd = mock_run.call_args[0][0]
    assert "--system-prompt" in cmd
    assert "you are nell" in cmd


def test_claude_cli_provider_raises_on_nonzero_exit() -> None:
    """Non-zero exit code surfaces a RuntimeError that includes stderr."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "auth failed"

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider()
        with pytest.raises(RuntimeError, match="auth failed"):
            p.generate("p")


def test_claude_cli_provider_name() -> None:
    """Name includes the model identifier."""
    assert ClaudeCliProvider(model="sonnet").name() == "claude-cli:sonnet"


def test_claude_cli_provider_subprocess_timeout_surfaced() -> None:
    """subprocess.TimeoutExpired surfaces as TimeoutError with context."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)):
        p = ClaudeCliProvider()
        with pytest.raises(TimeoutError, match="timed out"):
            p.generate("p")


def test_get_provider_resolves_known_names() -> None:
    """get_provider returns the right class for each known name."""
    assert isinstance(get_provider("fake"), FakeProvider)
    assert isinstance(get_provider("claude-cli"), ClaudeCliProvider)
    assert isinstance(get_provider("ollama"), OllamaProvider)


def test_get_provider_unknown_name_raises() -> None:
    """Unknown provider name raises ValueError with a clear message."""
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonsense")
```

- [ ] **Step 4: Run tests — expect 14 failures**

```bash
uv run pytest tests/unit/brain/bridge/test_provider.py -v
```
Expected: 14 failures on `ModuleNotFoundError`.

- [ ] **Step 5: Write `brain/bridge/provider.py`**

```python
"""LLM provider abstraction — ABC + concrete providers + factory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from abc import ABC, abstractmethod


_DEFAULT_TIMEOUT_SECONDS = 300


class LLMProvider(ABC):
    """Abstract LLM provider. Subclasses implement `generate` and `name`."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the LLM's completion for the given prompt."""

    @abstractmethod
    def name(self) -> str:
        """Return a short provider name (e.g. 'fake', 'claude-cli:sonnet')."""


class FakeProvider(LLMProvider):
    """Deterministic hash-based echo provider for tests — zero network calls."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        seed_input = (system or "").encode("utf-8") + b"||" + prompt.encode("utf-8")
        h = hashlib.sha256(seed_input).hexdigest()[:16]
        return f"DREAM: test dream {h} — an associative thread"

    def name(self) -> str:
        return "fake"


class ClaudeCliProvider(LLMProvider):
    """Shells out to `claude -p <prompt> --output-format json`.

    Uses Hana's Claude Code subscription — no per-token API billing.
    Per the feedback memory: this is the default Claude path for
    companion-emergence and Hana's other projects.
    """

    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", self._model]
        if system is not None:
            cmd.extend(["--system-prompt", system])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ClaudeCliProvider: subprocess timed out after {self._timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ClaudeCliProvider failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        payload = json.loads(result.stdout)
        return str(payload["result"])

    def name(self) -> str:
        return f"claude-cli:{self._model}"


class OllamaProvider(LLMProvider):
    """Placeholder for local Ollama integration.

    Stub until Hana's local Ollama stack is back up. Fill in by:
    1. Replacing raise with an httpx POST to {host}/api/generate
    2. Parsing the streamed/non-streamed response
    3. Adding the httpx dep to pyproject
    """

    def __init__(self, model: str = "nell-dpo", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError(
            "OllamaProvider is a stub; fill in when the local Ollama stack is available."
        )

    def name(self) -> str:
        return f"ollama:{self._model}"


def get_provider(name: str) -> LLMProvider:
    """Resolve a provider identifier to an instance. Raises ValueError on unknown."""
    if name == "fake":
        return FakeProvider()
    if name == "claude-cli":
        return ClaudeCliProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown provider: {name!r}")
```

- [ ] **Step 6: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/bridge/test_provider.py -v
```
Expected: 14 passed.

- [ ] **Step 7: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```
Expected: 268 passed (254 + 14). Ruff clean. If format fails, run `uv run ruff format .`.

- [ ] **Step 8: Commit**

```bash
git add brain/bridge/ tests/unit/brain/bridge/
git commit -m "feat(brain/bridge): LLM provider abstraction

LLMProvider ABC with three concrete implementations:
- FakeProvider: deterministic SHA-256-based echo. Used in tests; zero
  network calls. Output starts with 'DREAM:' so downstream dream
  engine contract holds on both fake and real providers.
- ClaudeCliProvider: shells out to 'claude -p <prompt> --output-format
  json'. Uses Hana's subscription entitlement (no per-token billing).
  Default Claude path per the feedback-memory preference.
- OllamaProvider: placeholder raising NotImplementedError. Filled in
  when local Ollama stack returns.

get_provider(name) factory resolves names; unknown → ValueError.

14 tests green; 268 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `brain/engines/dream.py` — associative dream cycle

**Goal:** `DreamEngine` class with `run_cycle()` that selects a seed, spread-activates via Hebbian, builds a prompt, calls the LLM, writes a dream memory, and strengthens edges. Plus `DreamResult` dataclass and `NoSeedAvailable` exception.

**Files:**
- Create: `/Users/hanamori/companion-emergence/brain/engines/__init__.py`
- Create: `/Users/hanamori/companion-emergence/brain/engines/dream.py`
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/engines/__init__.py` (empty)
- Create: `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_dream.py`

- [ ] **Step 1: Create package dirs**

```bash
cd /Users/hanamori/companion-emergence
mkdir -p brain/engines tests/unit/brain/engines
touch tests/unit/brain/engines/__init__.py
```

- [ ] **Step 2: Write `brain/engines/__init__.py`**

```python
"""Cognitive engines for companion-emergence.

Dreams consolidate associative patterns; heartbeat/reflex/research
follow in later weeks. See:
docs/superpowers/specs/2026-04-23-week-4-dream-engine-design.md
"""

from brain.engines.dream import DreamEngine, DreamResult, NoSeedAvailable

__all__ = ["DreamEngine", "DreamResult", "NoSeedAvailable"]
```

- [ ] **Step 3: Write the failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_dream.py`:

```python
"""Tests for brain.engines.dream — associative dream cycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider, LLMProvider
from brain.engines.dream import DreamEngine, DreamResult, NoSeedAvailable
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


@pytest.fixture
def hebbian() -> HebbianMatrix:
    return HebbianMatrix(db_path=":memory:")


@pytest.fixture
def engine(store: MemoryStore, hebbian: HebbianMatrix, tmp_path: Path) -> DreamEngine:
    return DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=FakeProvider(),
        log_path=tmp_path / "dreams.log.jsonl",
    )


def _mem(content: str, importance: float = 5.0, **kw: object) -> Memory:
    defaults: dict[str, object] = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    m = Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]
    m.importance = importance
    return m


def test_run_cycle_raises_if_no_seed_candidates(engine: DreamEngine) -> None:
    """Empty lookback window → NoSeedAvailable."""
    with pytest.raises(NoSeedAvailable):
        engine.run_cycle(lookback_hours=24)


def test_run_cycle_picks_highest_importance_seed_in_window(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Seed selection: highest-importance conversation within lookback window."""
    low = _mem("low", importance=2.0)
    high = _mem("high", importance=8.0)
    store.create(low)
    store.create(high)

    result = engine.run_cycle()
    assert result.seed.id == high.id


def test_run_cycle_explicit_seed_overrides_autoselect(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """When seed_id is provided, auto-select is skipped."""
    m1 = _mem("low", importance=2.0)
    m2 = _mem("explicit", importance=1.0)
    store.create(m1)
    store.create(m2)

    result = engine.run_cycle(seed_id=m2.id)
    assert result.seed.id == m2.id


def test_run_cycle_writes_dream_memory_to_store(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Successful cycle creates a new memory_type='dream' record."""
    seed = _mem("seed content", importance=8.0)
    store.create(seed)

    result = engine.run_cycle()
    assert result.memory is not None
    assert result.memory.memory_type == "dream"

    restored = store.get(result.memory.id)
    assert restored is not None
    assert restored.memory_type == "dream"


def test_run_cycle_dream_content_starts_with_dream_prefix(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Dream memory content starts with 'DREAM: ' even if LLM omits the prefix."""
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle()
    assert result.memory is not None
    assert result.memory.content.startswith("DREAM:")


def test_run_cycle_metadata_includes_seed_and_activated_ids(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Dream memory's metadata records seed_id + activated ids + provider name."""
    seed = _mem("seed", importance=8.0)
    neighbour = _mem("neighbour", importance=4.0)
    store.create(seed)
    store.create(neighbour)
    hebbian.strengthen(seed.id, neighbour.id, delta=0.7)

    result = engine.run_cycle()
    assert result.memory is not None
    md = result.memory.metadata
    assert md["seed_id"] == seed.id
    assert neighbour.id in md["activated"]
    assert md["provider"] == "fake"


def test_run_cycle_strengthens_edges_to_each_activated_neighbour(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Each (seed, activated) pair gets its Hebbian weight reinforced."""
    seed = _mem("seed", importance=8.0)
    n1 = _mem("n1", importance=4.0)
    n2 = _mem("n2", importance=4.0)
    for m in (seed, n1, n2):
        store.create(m)
    hebbian.strengthen(seed.id, n1.id, delta=0.5)
    hebbian.strengthen(seed.id, n2.id, delta=0.5)

    before_n1 = hebbian.weight(seed.id, n1.id)
    before_n2 = hebbian.weight(seed.id, n2.id)

    result = engine.run_cycle()
    assert result.strengthened_edges == 2

    assert hebbian.weight(seed.id, n1.id) > before_n1
    assert hebbian.weight(seed.id, n2.id) > before_n2


def test_run_cycle_dry_run_returns_result_without_writes(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """dry_run=True returns DreamResult with memory=None, no store/hebbian mutation."""
    seed = _mem("seed", importance=8.0)
    neighbour = _mem("neighbour", importance=4.0)
    store.create(seed)
    store.create(neighbour)
    hebbian.strengthen(seed.id, neighbour.id, delta=0.5)

    count_before = store.count()
    weight_before = hebbian.weight(seed.id, neighbour.id)

    result = engine.run_cycle(dry_run=True)

    assert result.memory is None
    assert result.dream_text is None
    assert result.strengthened_edges == 0
    assert store.count() == count_before
    assert hebbian.weight(seed.id, neighbour.id) == weight_before


def test_run_cycle_dry_run_populates_seed_neighbours_and_prompt(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Dry run still runs seed selection, neighbourhood assembly, prompt build."""
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle(dry_run=True)
    assert result.seed.id == seed.id
    assert result.prompt != ""
    assert result.system_prompt != ""


def test_run_cycle_respects_lookback_window(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Memories older than lookback_hours are excluded from seed candidates."""
    old = _mem("old", importance=9.0)
    old.created_at = datetime.now(UTC) - timedelta(hours=48)
    store.create(old)
    recent = _mem("recent", importance=2.0)
    store.create(recent)

    result = engine.run_cycle(lookback_hours=24)
    assert result.seed.id == recent.id


def test_run_cycle_appends_to_dreams_log(
    engine: DreamEngine, store: MemoryStore, tmp_path: Path
) -> None:
    """Each successful cycle appends one JSONL line to dreams.log.jsonl."""
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    engine.run_cycle()

    log_path = tmp_path / "dreams.log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["seed_id"] == seed.id
    assert entry["provider"] == "fake"
    assert "timestamp" in entry
    assert "dream_id" in entry


def test_run_cycle_dry_run_does_not_log(
    engine: DreamEngine, store: MemoryStore, tmp_path: Path
) -> None:
    """Dry run does not write to the dream log."""
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    engine.run_cycle(dry_run=True)
    log_path = tmp_path / "dreams.log.jsonl"
    assert not log_path.exists() or log_path.read_text() == ""


def test_run_cycle_prompt_contains_seed_and_neighbours(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """User prompt references the seed content and each activated neighbour."""
    seed = _mem("the seed thought", importance=8.0)
    neighbour = _mem("the neighbour thought", importance=4.0)
    store.create(seed)
    store.create(neighbour)
    hebbian.strengthen(seed.id, neighbour.id, delta=0.5)

    result = engine.run_cycle(dry_run=True)
    assert "the seed thought" in result.prompt
    assert "the neighbour thought" in result.prompt


def test_run_cycle_system_prompt_mentions_nell_and_dream_prefix(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """System prompt establishes the Nell persona and DREAM: convention."""
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle(dry_run=True)
    assert "Nell" in result.system_prompt
    assert "DREAM:" in result.system_prompt


def test_run_cycle_respects_neighbour_limit(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """neighbour_limit caps how many activated memories feed the prompt."""
    seed = _mem("seed", importance=8.0)
    store.create(seed)
    for i in range(10):
        n = _mem(f"n{i}")
        store.create(n)
        hebbian.strengthen(seed.id, n.id, delta=0.5)

    result = engine.run_cycle(dry_run=True, neighbour_limit=3)
    assert len(result.neighbours) <= 3
```

- [ ] **Step 4: Run tests — expect 15 failures**

```bash
uv run pytest tests/unit/brain/engines/test_dream.py -v
```
Expected: 15 failures on ModuleNotFoundError.

- [ ] **Step 5: Write `brain/engines/dream.py`**

```python
"""Associative dream cycle.

Produces first-person meta-reflection memories by threading associated
experiences together. See spec Section 5 for the cycle step-by-step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain.bridge.provider import LLMProvider
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


_SYSTEM_PROMPT = (
    "You are Nell. You just woke from a dream about interconnected memories. "
    "Reflect in first person, 2-3 sentences, starting with 'DREAM: '. "
    "Be honest and specific, not abstract."
)


class NoSeedAvailable(Exception):
    """Raised when there are no conversation memories within the lookback window."""


@dataclass(frozen=True)
class DreamResult:
    """Outcome of a single dream cycle."""

    seed: Memory
    neighbours: list[tuple[Memory, float]]
    system_prompt: str
    prompt: str
    dream_text: str | None
    memory: Memory | None
    strengthened_edges: int


@dataclass
class DreamEngine:
    """Composes memory + emotion + LLM bridge into an associative cycle."""

    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache | None
    provider: LLMProvider
    log_path: Path | None = None

    def run_cycle(
        self,
        *,
        seed_id: str | None = None,
        lookback_hours: int = 24,
        depth: int = 2,
        decay_per_hop: float = 0.5,
        neighbour_limit: int = 8,
        strengthen_delta: float = 0.1,
        dry_run: bool = False,
    ) -> DreamResult:
        seed = self._select_seed(seed_id=seed_id, lookback_hours=lookback_hours)
        neighbours = self._spread_activate(
            seed, depth=depth, decay_per_hop=decay_per_hop, limit=neighbour_limit
        )
        system_prompt, user_prompt = self._build_prompt(seed, neighbours)

        if dry_run:
            return DreamResult(
                seed=seed,
                neighbours=neighbours,
                system_prompt=system_prompt,
                prompt=user_prompt,
                dream_text=None,
                memory=None,
                strengthened_edges=0,
            )

        raw_text = self.provider.generate(user_prompt, system=system_prompt)
        dream_text = raw_text if raw_text.startswith("DREAM:") else f"DREAM: {raw_text}"

        dream_memory = self._write_dream_memory(seed, neighbours, dream_text)
        edges = self._strengthen_edges(seed, neighbours, strengthen_delta)
        self._log(seed, neighbours, dream_memory)

        return DreamResult(
            seed=seed,
            neighbours=neighbours,
            system_prompt=system_prompt,
            prompt=user_prompt,
            dream_text=dream_text,
            memory=dream_memory,
            strengthened_edges=edges,
        )

    # --- internals ---

    def _select_seed(self, *, seed_id: str | None, lookback_hours: int) -> Memory:
        if seed_id is not None:
            mem = self.store.get(seed_id)
            if mem is None:
                raise NoSeedAvailable(f"Seed id {seed_id!r} not found in store.")
            return mem

        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        candidates = self.store.list_by_type("conversation", active_only=True)
        in_window = [m for m in candidates if m.created_at >= cutoff]
        if not in_window:
            raise NoSeedAvailable(
                f"No conversation memories within the last {lookback_hours} hours."
            )
        in_window.sort(key=lambda m: m.importance, reverse=True)
        return in_window[0]

    def _spread_activate(
        self, seed: Memory, *, depth: int, decay_per_hop: float, limit: int
    ) -> list[tuple[Memory, float]]:
        activation = self.hebbian.spreading_activation(
            [seed.id], depth=depth, decay_per_hop=decay_per_hop
        )
        activation.pop(seed.id, None)  # seed not a neighbour of itself
        results: list[tuple[Memory, float]] = []
        for mid, act in sorted(activation.items(), key=lambda p: p[1], reverse=True):
            mem = self.store.get(mid)
            if mem is not None:
                results.append((mem, act))
                if len(results) >= limit:
                    break
        return results

    def _build_prompt(
        self, seed: Memory, neighbours: list[tuple[Memory, float]]
    ) -> tuple[str, str]:
        parts = [
            f"Seed memory (domain={seed.domain}):",
            f"  {seed.content}",
            "",
        ]
        if neighbours:
            parts.append("Also present:")
            for mem, _ in neighbours:
                parts.append(f"  - {mem.content[:120]}")
        else:
            parts.append("No other memories resonated with this one yet.")
        return _SYSTEM_PROMPT, "\n".join(parts)

    def _write_dream_memory(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        dream_text: str,
    ) -> Memory:
        aggregated_emotions: dict[str, float] = dict(seed.emotions)
        for mem, _ in neighbours:
            for k, v in mem.emotions.items():
                aggregated_emotions[k] = aggregated_emotions.get(k, 0.0) + v

        dream = Memory.create_new(
            content=dream_text,
            memory_type="dream",
            domain=seed.domain,
            emotions=aggregated_emotions,
            metadata={
                "seed_id": seed.id,
                "activated": [m.id for m, _ in neighbours],
                "provider": self.provider.name(),
            },
        )
        self.store.create(dream)
        return dream

    def _strengthen_edges(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        delta: float,
    ) -> int:
        count = 0
        for mem, activation in neighbours:
            weighted = delta * activation
            if weighted <= 0.0:
                continue
            self.hebbian.strengthen(seed.id, mem.id, delta=weighted)
            count += 1
        return count

    def _log(
        self,
        seed: Memory,
        neighbours: list[tuple[Memory, float]],
        dream_memory: Memory,
    ) -> None:
        if self.log_path is None:
            return
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "seed_id": seed.id,
            "neighbour_ids": [m.id for m, _ in neighbours],
            "dream_id": dream_memory.id,
            "provider": self.provider.name(),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
```

Note on `field` import: the `from dataclasses import dataclass, field` line is there for future extension; if ruff complains about unused import, change to `from dataclasses import dataclass`.

- [ ] **Step 6: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/engines/test_dream.py -v
```
Expected: 15 passed.

- [ ] **Step 7: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```
Expected: 283 passed (268 + 15). Ruff clean.

- [ ] **Step 8: Commit**

```bash
git add brain/engines/ tests/unit/brain/engines/
git commit -m "feat(brain/engines/dream): associative dream cycle

DreamEngine.run_cycle() executes the associative pattern: select
highest-importance recent conversation memory as seed → spread-activate
via Hebbian BFS → build prompt (system: Nell persona + DREAM: prefix
convention; user: seed + activated neighbours) → call LLM via bridge
→ write new memory_type='dream' memory → strengthen each (seed, N)
edge proportional to N's activation.

DreamResult carries the seed, neighbours, prompt, dream text, new
memory, and strengthened-edge count for downstream inspection.

Dry-run mode populates seed + neighbours + prompt without any LLM
call or store mutation — useful for 'what would this cycle do?'.

Dream log appended as JSONL when log_path is configured.

15 tests green (all against FakeProvider — zero network in CI);
283 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `brain/cli.py` wiring — `nell dream` subcommand

**Goal:** Replace the `dream` stub in `brain/cli.py` with a real subparser that instantiates stores + bridge provider from a persona dir and dispatches to `DreamEngine.run_cycle()`. 5 new tests.

**Files:**
- Modify: `/Users/hanamori/companion-emergence/brain/cli.py`
- Modify: `/Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py` (remove `"dream"` from the stubs-expected list)
- Create or extend: `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/hanamori/companion-emergence/tests/unit/brain/engines/test_cli.py`:

```python
"""Tests for the `nell dream` CLI subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def nell_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal nell persona dir with one recent memory + empty hebbian."""
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))

    from brain.paths import get_persona_dir

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)

    store = MemoryStore(db_path=persona / "memories.db")
    seed = Memory.create_new(
        content="first meeting test seed",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
    )
    seed.importance = 8.0
    store.create(seed)
    store.close()

    # hebbian.db is created on first access by DreamEngine's own call,
    # but init it here so the CLI opens a real file.
    from brain.memory.hebbian import HebbianMatrix

    h = HebbianMatrix(db_path=persona / "hebbian.db")
    h.close()
    return persona


def test_nell_dream_dry_run_with_fake_provider(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """nell dream --dry-run --provider fake runs without writes and prints a summary."""
    from brain.cli import main

    rc = main(["dream", "--dry-run", "--provider", "fake"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "first meeting test seed" in out or "DREAM" in out or "dry-run" in out.lower()


def test_nell_dream_real_cycle_with_fake_provider(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """nell dream --provider fake writes a new dream memory."""
    from brain.cli import main

    rc = main(["dream", "--provider", "fake"])
    assert rc == 0

    store = MemoryStore(db_path=nell_persona / "memories.db")
    dreams = store.list_by_type("dream")
    store.close()
    assert len(dreams) == 1


def test_nell_dream_ollama_surfaces_not_implemented(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--provider ollama fails cleanly with a NotImplementedError message."""
    from brain.cli import main

    with pytest.raises(NotImplementedError):
        main(["dream", "--provider", "ollama"])


def test_nell_dream_unknown_provider_fails(nell_persona: Path) -> None:
    """Unknown provider name raises ValueError."""
    from brain.cli import main

    with pytest.raises(ValueError, match="Unknown provider"):
        main(["dream", "--provider", "nonsense"])


def test_nell_dream_unknown_persona_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persona dir missing → clear error (no silent no-op)."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    from brain.cli import main

    with pytest.raises(FileNotFoundError, match="persona"):
        main(["dream", "--persona", "ghost", "--provider", "fake"])
```

- [ ] **Step 2: Update `tests/unit/brain/test_cli.py` to drop `"dream"` from the stubs list**

Find the line that lists expected stub names and remove `"dream"`:

```bash
grep -n '"dream"' /Users/hanamori/companion-emergence/tests/unit/brain/test_cli.py
```

Edit the line to remove the string. The test that parametrizes stub subcommands should no longer include `dream`.

- [ ] **Step 3: Run tests — expect failures**

```bash
uv run pytest tests/unit/brain/engines/test_cli.py -v
```
Expected: 5 failures — `nell dream` is still a stub so it prints "not implemented" rather than calling the engine.

- [ ] **Step 4: Wire `nell dream` in `brain/cli.py`**

Read `/Users/hanamori/companion-emergence/brain/cli.py`. Apply these changes:

1. Remove `"dream"` from `_STUB_COMMANDS`:
```python
_STUB_COMMANDS: tuple[str, ...] = (
    "supervisor",
    "heartbeat",
    "reflex",
    "status",
    "rest",
    "soul",
    "memory",
    "works",
)
```
(Note: `migrate` was removed in Week 3.5, `dream` removed here.)

2. Add imports at the top (alongside other `from brain...` imports):
```python
from brain.bridge.provider import get_provider
from brain.engines.dream import DreamEngine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.paths import get_persona_dir
```

3. Add a `_dream_handler(args)` function alongside the existing `_make_stub`:

```python
def _dream_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell dream` to the DreamEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — run `nell migrate --install-as {args.persona}` first."
        )
    store = MemoryStore(db_path=persona_dir / "memories.db")
    hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
    provider = get_provider(args.provider)
    engine = DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=provider,
        log_path=persona_dir / "dreams.log.jsonl",
    )
    try:
        result = engine.run_cycle(
            seed_id=args.seed,
            lookback_hours=args.lookback,
            depth=args.depth,
            decay_per_hop=args.decay,
            neighbour_limit=args.limit,
            dry_run=args.dry_run,
        )
    finally:
        store.close()
        hebbian.close()

    if args.dry_run:
        print("Dry run — no writes.")
        print(f"Seed: {result.seed.id}  ({result.seed.content[:80]})")
        print(f"Neighbours: {len(result.neighbours)}")
        print(f"Prompt preview:\n{result.prompt[:400]}")
    else:
        print(result.dream_text or "")
    return 0
```

4. In `_build_parser()`, AFTER the `for name in _STUB_COMMANDS` loop and AFTER the `_build_migrate_parser(subparsers)` call, add:

```python
    dream_sub = subparsers.add_parser(
        "dream",
        help="Run one dream cycle against a persona's memory store.",
    )
    dream_sub.add_argument("--persona", default="nell", help="Persona name (default: nell).")
    dream_sub.add_argument("--seed", default=None, help="Explicit seed memory id (default: auto-select).")
    dream_sub.add_argument(
        "--provider", default="claude-cli",
        help="LLM provider: claude-cli (default), fake, ollama.",
    )
    dream_sub.add_argument("--dry-run", action="store_true", help="Skip LLM call and store writes.")
    dream_sub.add_argument("--lookback", type=int, default=24, help="Hours of history to consider (default: 24).")
    dream_sub.add_argument("--depth", type=int, default=2, help="Spreading-activation depth (default: 2).")
    dream_sub.add_argument("--decay", type=float, default=0.5, help="Per-hop decay (default: 0.5).")
    dream_sub.add_argument("--limit", type=int, default=8, help="Max neighbours in prompt (default: 8).")
    dream_sub.set_defaults(func=_dream_handler)
```

- [ ] **Step 5: Run tests — expect green**

```bash
uv run pytest tests/unit/brain/engines/test_cli.py -v
uv run pytest tests/unit/brain/test_cli.py -v
```
Expected: 5 new tests pass; existing test_cli.py still green after removing `"dream"` from the stubs list.

- [ ] **Step 6: Manual CLI smoke**

```bash
uv run nell dream --help
```
Expected: help text with --persona, --seed, --provider, --dry-run, --lookback, --depth, --decay, --limit.

- [ ] **Step 7: Full suite + ruff**

```bash
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
```
Expected: 288 passed (283 + 5). Ruff clean.

- [ ] **Step 8: Commit**

```bash
git add brain/cli.py tests/unit/brain/engines/test_cli.py tests/unit/brain/test_cli.py
git commit -m "feat(brain/cli): wire nell dream subcommand to the DreamEngine

_dream_handler opens the persona's memories.db + hebbian.db via
get_persona_dir(), resolves the provider via get_provider(), runs one
cycle, and prints either the dry-run summary or the dream text.

try/finally around run_cycle closes both DB connections cleanly
whether the LLM call succeeds or the cycle raises.

dream moves out of _STUB_COMMANDS; the corresponding stub test
expectation removed.

5 new CLI tests; 288 total.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Close-out — CI + merge

**Goal:** Fresh-install verify, CI green on 3 OSes, merge PR. No tag (Week 4 tag waits until all four engines ship; this is engine #1 of 4).

- [ ] **Step 1: Clean install + verify**

```bash
cd /Users/hanamori/companion-emergence
rm -rf .venv
uv sync --all-extras
uv run pytest 2>&1 | tail -3
uv run ruff check .
uv run ruff format --check .
uv run nell dream --help
```
Expected: 288 passed, ruff clean, `dream --help` prints real usage.

- [ ] **Step 2: Dry-run smoke against real nell persona**

```bash
uv run nell dream --persona nell --provider fake --dry-run
```
Expected: prints seed + neighbours + prompt preview. No writes (the integration test covers writes with FakeProvider on an ephemeral persona).

- [ ] **Step 3: Push branch + open PR**

```bash
git push -u origin week-4-dream
gh pr create --title "feat: Week 4 — dream engine + LLM bridge (Claude CLI default)" --body "$(cat <<'EOF'
## Summary
- Adds `brain/bridge/` (LLMProvider ABC + ClaudeCliProvider default + FakeProvider for tests + OllamaProvider stub)
- Adds `brain/engines/dream.py` (associative cycle: seed → spread-activate → LLM reflect → new memory + Hebbian strengthen)
- Wires `nell dream` subcommand
- 34 new tests (14 bridge + 15 dream + 5 CLI); suite reaches 288 across macOS + Windows + Linux

## Per-task
| Task | Commits | Tests |
|---|---|---|
| 1. brain/bridge/provider.py | feat + cleanup | 14 |
| 2. brain/engines/dream.py | feat + cleanup | 15 |
| 3. brain/cli.py wiring | feat + cleanup | 5 |

## Design decisions worth noting
- Claude CLI is the default provider — uses Hana's subscription, not the Anthropic API (feedback memory).
- FakeProvider always emits `DREAM:` prefix so engine auto-prefix logic stays exercised on both real and fake paths.
- `--dry-run` populates seed + neighbours + prompt without LLM call or writes — lets you preview a cycle safely.
- Dream cycles default to the canonical nell persona; `--persona nell.sandbox` recommended for the first few real LLM runs.

## Test plan
- [x] Fresh `uv sync --all-extras` succeeds
- [x] pytest — 288 tests pass locally
- [x] ruff check + format — clean
- [x] `uv run nell dream --help` prints real usage
- [x] `uv run nell dream --dry-run --provider fake` against real nell persona runs to completion
- [ ] CI matrix green across macOS + Ubuntu + Windows
- [ ] Hana runs a real `nell dream --persona nell.sandbox --provider claude-cli` cycle and inspects
EOF
)"
```

- [ ] **Step 4: Watch CI to completion**

```bash
sleep 15
gh run list --branch week-4-dream --limit 1
gh run watch
```
Expected: success on all 3 OSes.

- [ ] **Step 5: Merge + sync main**

```bash
gh pr merge --merge --delete-branch
git checkout main
git pull origin main
```

No tag. Week 4 tag waits until heartbeat/reflex/research also ship.

---

## Week 4 green-light criterion

1. `uv sync --all-extras` fresh install succeeds.
2. `uv run pytest` → 288 passed.
3. `uv run ruff check .` + `uv run ruff format --check .` both clean.
4. `uv run nell dream --help` prints real subcommand usage.
5. `uv run nell dream --dry-run --provider fake --persona nell` runs without error.
6. CI matrix green on all 3 OSes.
7. PR merged to main.

**User-side (not part of automated criterion):**
- Hana clones nell to nell.sandbox, runs `nell dream --persona nell.sandbox --provider claude-cli`, inspects the resulting dream memory.
- Once trusted, she runs against the canonical nell persona.

---

## Notes for the engineer executing this plan

- **Claude CLI default, not the API.** The feedback memory `feedback_claude_cli_over_api.md` is binding. Do not propose an `AnthropicApiProvider` unless Hana explicitly asks. ClaudeCliProvider uses her subscription.
- **Tests use FakeProvider only.** Zero network calls in CI. `subprocess.run` is mocked in the ClaudeCliProvider tests — no real `claude` binary invoked.
- **Dream engine is side-effecting.** `store.create()` and `hebbian.strengthen()` mutate persistent state. The `dry_run=True` path is the only safe way to inspect a cycle without writes.
- **`DREAM:` prefix contract.** Every dream memory's content starts with `"DREAM: "`. The engine auto-adds it if the LLM response doesn't have it. Downstream consumers (future analytics, supervisor) rely on this convention.
- **Hebbian strengthening is activation-weighted.** `delta * activation` means strongly-activated neighbours get more reinforcement than weakly-activated ones. Tunable via `strengthen_delta` parameter.
- **No scheduler, no batching.** One `run_cycle()` per invocation. Batching flags and the cron daemon land in later weeks.

---

*End of Week 4 plan.*
