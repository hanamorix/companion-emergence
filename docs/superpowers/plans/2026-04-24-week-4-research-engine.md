# Week 4.7 — Research Engine (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the fourth and final Week 4 cognitive engine — research — with DuckDuckGo web search by default, memory-sweep fallback, interest storage, heartbeat orchestration, and Nell's OG interest migration. After this ships, Week 4 gets its tag.

**Architecture:** Mirrors the reflex engine's pattern. `ResearchEngine.run_tick` evaluates triggers (emotion_high OR days_since_human), selects an eligible interest past pull threshold + past cooldown, does a memory sweep via spreading activation, optionally hits the web via a pluggable `WebSearcher` (DDG default), synthesizes via the existing `LLMProvider`, writes a first-person memory. Heartbeat integrates between dream gate and heartbeat-memory. Reflex-wins-tie when both are eligible. Interest ingestion hook bumps pull_scores via keyword matching inside heartbeat ticks.

**Tech Stack:** Python 3.12, SQLite (existing MemoryStore), `ddgs` (new dep — DuckDuckGo, no API key, free), pytest, ruff, hatchling.

**Spec:** `docs/superpowers/specs/2026-04-24-week-4-research-engine-design.md`

**Precedent pattern:** Reflex engine (`brain/engines/reflex.py`) and its plan. Copy the shape — don't invent a new one. Reflex shipped in PR #7 merged as `6e1996c`; study those files before starting.

**Hard rule:** `rg 'import anthropic' brain/` must always return zero matches. All LLM calls through `brain.bridge.provider.LLMProvider`.

**Running test total:** pre-start 376. After this plan: ~401.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `brain/search/__init__.py` | Package marker. Empty or short re-exports of public symbols. |
| `brain/search/base.py` | `SearchResult` frozen dataclass + `WebSearcher` ABC + `NoopWebSearcher`. |
| `brain/search/ddgs_searcher.py` | `DdgsWebSearcher` — DDG via `ddgs` library. |
| `brain/search/claude_tool_searcher.py` | Stub `ClaudeToolWebSearcher` (raises `NotImplementedError` — real impl is future work). |
| `brain/search/factory.py` | `get_searcher(name)` resolver. |
| `brain/engines/_interests.py` | `Interest` dataclass + `InterestSet` load/save/bump/list_eligible. |
| `brain/engines/default_interests.json` | `{"version": 1, "interests": []}` — empty starter for new personas. |
| `brain/engines/research.py` | `ResearchEngine` + `ResearchResult` + `ResearchFire` types. |
| `brain/migrator/og_interests.py` | JSON-port of OG `nell_interests.json` → new schema. |
| `tests/unit/brain/search/__init__.py` | Package marker. |
| `tests/unit/brain/search/test_noop.py` | NoopWebSearcher tests. |
| `tests/unit/brain/search/test_ddgs.py` | DdgsWebSearcher tests with mocked `ddgs` module. |
| `tests/unit/brain/engines/test_interests.py` | `Interest` + `InterestSet` unit tests. |
| `tests/unit/brain/engines/test_research.py` | `ResearchEngine` unit tests. |
| `tests/unit/brain/engines/test_cli_research.py` | 4 new CLI subcommands. |
| `tests/unit/brain/migrator/test_og_interests.py` | OG interest extraction tests. |

### Modified files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `ddgs>=6.0,<7.0` to `dependencies`. |
| `brain/engines/heartbeat.py` | Extend `HeartbeatConfig` (4 new fields), `HeartbeatResult` (3 new fields), `HeartbeatEngine` (4 new fields). Add `_try_bump_interests` helper. Add `_try_fire_research` helper with reflex-wins-tie guard. Wire both into `run_tick`. Extend audit log. |
| `brain/cli.py` | Add `nell research` + `nell interest {list,add,bump}` subcommands. Update `_heartbeat_handler` to pass new reflex/research paths + searcher. |
| `brain/migrator/cli.py` | Wire `og_interests` extraction after Hebbian/reflex blocks, before `elapsed`. Pass two new `MigrationReport` fields. |
| `brain/migrator/report.py` | Add `interests_migrated: int = 0` + `interests_skipped_reason: str \| None = None` to `MigrationReport`. Add "Interests:" line in `format_report`. |
| `tests/unit/brain/engines/test_heartbeat.py` | Add integration tests: research fires / disabled / reflex-wins-tie / failure-isolated / ingestion-bumps-pull_score. |
| `tests/unit/brain/migrator/test_cli.py` | Add regression test: migrator writes interests.json. |

---

## Task 0: Web search layer

**Purpose:** Ship `brain/search/` package with `WebSearcher` ABC + `NoopWebSearcher` + `DdgsWebSearcher` + stub `ClaudeToolWebSearcher` + `get_searcher` factory. Add `ddgs` dependency. The engine (Task 3) depends on this.

**Files:**
- Create: `brain/search/__init__.py`
- Create: `brain/search/base.py`
- Create: `brain/search/ddgs_searcher.py`
- Create: `brain/search/claude_tool_searcher.py`
- Create: `brain/search/factory.py`
- Create: `tests/unit/brain/search/__init__.py`
- Create: `tests/unit/brain/search/test_noop.py`
- Create: `tests/unit/brain/search/test_ddgs.py`
- Modify: `pyproject.toml` (add `ddgs>=6.0,<7.0`)

- [ ] **Step 1: Add ddgs dependency**

Edit `pyproject.toml`:

```toml
dependencies = [
    "platformdirs>=4.2",
    "numpy>=1.26",
    "ddgs>=6.0,<7.0",
]
```

Run: `uv sync --group dev`
Expected: resolves + installs `ddgs` with its transitive deps.

- [ ] **Step 2: Write failing test for NoopWebSearcher**

Create `tests/unit/brain/search/__init__.py` (empty file).

Create `tests/unit/brain/search/test_noop.py`:

```python
"""Tests for brain.search.base.NoopWebSearcher."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher


def test_noop_returns_empty_list():
    s = NoopWebSearcher()
    assert s.search("any query") == []


def test_noop_returns_empty_with_limit_arg():
    s = NoopWebSearcher()
    assert s.search("any query", limit=10) == []


def test_noop_name():
    assert NoopWebSearcher().name() == "noop"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/brain/search/test_noop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.search'`.

- [ ] **Step 4: Implement base + noop**

Create `brain/search/__init__.py`:

```python
"""Web search layer — pluggable searcher backends.

Default is DdgsWebSearcher (DuckDuckGo via the `ddgs` library, no API
key). NoopWebSearcher used in tests and CI for zero-network behavior.
"""
```

Create `brain/search/base.py`:

```python
"""WebSearcher ABC + shared types + NoopWebSearcher."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """One web search hit."""

    title: str
    url: str
    snippet: str


class WebSearcher(ABC):
    """Abstract searcher. Subclasses implement `search` and `name`."""

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return up to `limit` results for `query`. Empty list on any
        transient failure — research engine falls back to memory-only
        synthesis rather than crashing.
        """

    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'ddgs', 'noop', 'claude-tool'."""


class NoopWebSearcher(WebSearcher):
    """Returns no results. Used in tests and CI to keep them zero-network."""

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return []

    def name(self) -> str:
        return "noop"
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/brain/search/test_noop.py -v`
Expected: 3 passed.

- [ ] **Step 6: Write DdgsWebSearcher tests**

Create `tests/unit/brain/search/test_ddgs.py`:

```python
"""Tests for brain.search.ddgs_searcher.DdgsWebSearcher."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from brain.search.base import SearchResult
from brain.search.ddgs_searcher import DdgsWebSearcher


def test_ddgs_happy_path():
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_ctx
    fake_ctx.__exit__.return_value = None
    fake_ctx.text.return_value = [
        {"title": "T1", "href": "https://example.com/1", "body": "s1"},
        {"title": "T2", "href": "https://example.com/2", "body": "s2"},
    ]

    with patch("brain.search.ddgs_searcher.DDGS", return_value=fake_ctx):
        out = DdgsWebSearcher().search("quantum mechanics", limit=2)

    assert len(out) == 2
    assert out[0] == SearchResult(title="T1", url="https://example.com/1", snippet="s1")
    assert out[1].url == "https://example.com/2"


def test_ddgs_uses_href_or_url_field():
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_ctx
    fake_ctx.__exit__.return_value = None
    # Simulate a result with 'url' instead of 'href' (library variance)
    fake_ctx.text.return_value = [{"title": "T", "url": "https://x.com", "body": "s"}]

    with patch("brain.search.ddgs_searcher.DDGS", return_value=fake_ctx):
        out = DdgsWebSearcher().search("q", limit=1)

    assert out[0].url == "https://x.com"


def test_ddgs_transient_failure_returns_empty(caplog):
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_ctx
    fake_ctx.__exit__.return_value = None
    fake_ctx.text.side_effect = RuntimeError("network down")

    with patch("brain.search.ddgs_searcher.DDGS", return_value=fake_ctx):
        with caplog.at_level(logging.WARNING, logger="brain.search.ddgs_searcher"):
            out = DdgsWebSearcher().search("q")

    assert out == []
    assert any("ddgs search failed" in r.message for r in caplog.records)


def test_ddgs_name():
    assert DdgsWebSearcher().name() == "ddgs"


def test_ddgs_missing_library_raises_at_call_time():
    """If `ddgs` isn't installed, ImportError surfaces at first .search call,
    not at module import time."""
    with patch.dict("sys.modules", {"ddgs": None}):
        # Simulate missing package: importing `ddgs` raises
        import sys
        # Force re-import path: remove ddgs from sys.modules and intercept
        sys.modules["ddgs"] = None  # type: ignore[assignment]
        s = DdgsWebSearcher()
        # calling search imports ddgs lazily — this should raise
        import pytest
        with pytest.raises((ImportError, TypeError, AttributeError)):
            s.search("q")
```

(The last test is defensive — in practice `ddgs` is a declared dependency so this path is near-unreachable, but the lazy-import behavior is worth guarding.)

- [ ] **Step 7: Run to verify failure**

Run: `uv run pytest tests/unit/brain/search/test_ddgs.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 8: Implement DdgsWebSearcher**

Create `brain/search/ddgs_searcher.py`:

```python
"""DuckDuckGo web search via the `ddgs` library. No API key, free, zero-cost."""

from __future__ import annotations

import logging

from brain.search.base import SearchResult, WebSearcher

logger = logging.getLogger(__name__)


class DdgsWebSearcher(WebSearcher):
    """DuckDuckGo search through the `ddgs` Python library.

    Default searcher for the framework. Works with any LLM backend —
    no dependency on `claude` CLI or any specific provider. Transient
    errors (network, rate-limit, parser failures) return an empty list
    plus a warning log so the research engine can gracefully fall back
    to memory-only synthesis.

    `ddgs` is imported lazily so the module loads cleanly in environments
    that haven't installed it yet (uncommon — it's a declared dependency).
    """

    def __init__(self, region: str = "wt-wt", timeout_seconds: int = 15) -> None:
        self._region = region
        self._timeout = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        from ddgs import DDGS  # lazy import

        try:
            with DDGS(timeout=self._timeout) as ddgs:
                raw = list(ddgs.text(query, region=self._region, max_results=limit))
        except Exception as exc:
            logger.warning("ddgs search failed for %r: %s", query[:80], exc)
            return []

        return [
            SearchResult(
                title=str(r.get("title", "")),
                url=str(r.get("href") or r.get("url", "")),
                snippet=str(r.get("body", "")),
            )
            for r in raw
        ]

    def name(self) -> str:
        return "ddgs"
```

Create `brain/search/claude_tool_searcher.py` (stub):

```python
"""Stub for `claude -p --allowed-tools WebSearch` based searcher.

Phase 1 ships this as NotImplementedError. If a Claude-CLI user wants
to route web search through Claude's tool loop instead of DDG, this is
where that lives. Mirrors how OllamaProvider ships as a stub.
"""

from __future__ import annotations

from brain.search.base import SearchResult, WebSearcher


class ClaudeToolWebSearcher(WebSearcher):
    """Not implemented in Phase 1 — see docstring."""

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raise NotImplementedError(
            "ClaudeToolWebSearcher is a Phase 1 stub. Use DdgsWebSearcher "
            "(default) or NoopWebSearcher instead, or implement this if you "
            "want to route search through `claude -p --allowed-tools WebSearch`."
        )

    def name(self) -> str:
        return "claude-tool"
```

Create `brain/search/factory.py`:

```python
"""Searcher factory — resolve a name to an instance."""

from __future__ import annotations

from brain.search.base import NoopWebSearcher, WebSearcher
from brain.search.claude_tool_searcher import ClaudeToolWebSearcher
from brain.search.ddgs_searcher import DdgsWebSearcher


def get_searcher(name: str) -> WebSearcher:
    """Resolve a searcher identifier to an instance. Raises ValueError on unknown."""
    if name == "ddgs":
        return DdgsWebSearcher()
    if name == "noop":
        return NoopWebSearcher()
    if name == "claude-tool":
        return ClaudeToolWebSearcher()
    raise ValueError(f"Unknown searcher: {name!r}")
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/unit/brain/search/ -v`
Expected: all tests pass (8 total).

- [ ] **Step 10: Ruff + format**

Run: `uv run ruff check brain/search/ tests/unit/brain/search/ && uv run ruff format brain/search/ tests/unit/brain/search/`
Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add brain/search/ tests/unit/brain/search/ pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
feat: add brain/search/ web search abstraction

WebSearcher ABC + SearchResult frozen dataclass. Three concrete
searchers: DdgsWebSearcher (DuckDuckGo default, zero-cost), 
NoopWebSearcher (tests/CI, zero-network), ClaudeToolWebSearcher
(Phase 1 stub). get_searcher() factory for name→instance.

ddgs dep added to pyproject. Transient failures return [] so research
engine can fall back to memory-only synthesis.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Include `uv.lock` if it changed.)

---

## Task 1: Interest + InterestSet

**Purpose:** Build the `Interest` frozen dataclass + `InterestSet` load/save/bump/list_eligible helpers. Shared between research engine AND heartbeat's ingestion hook, so it lives in its own module.

**Files:**
- Create: `brain/engines/_interests.py`
- Create: `brain/engines/default_interests.json`
- Create: `tests/unit/brain/engines/test_interests.py`

- [ ] **Step 1: Create default interests JSON**

Create `brain/engines/default_interests.json`:

```json
{
  "version": 1,
  "interests": []
}
```

Trailing newline. Shipped empty — new personas start with no interests; they grow them through conversation (ingestion hook) or `nell interest add`.

- [ ] **Step 2: Write failing tests for Interest + InterestSet**

Create `tests/unit/brain/engines/test_interests.py`:

```python
"""Unit tests for brain.engines._interests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines._interests import Interest, InterestSet

DEFAULT_INTERESTS_PATH = (
    Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"
)


def _sample_dict(**overrides) -> dict:
    base = {
        "id": "abc-123",
        "topic": "Test topic",
        "pull_score": 6.5,
        "scope": "either",
        "related_keywords": ["test", "topic"],
        "notes": "",
        "first_seen": "2026-04-01T10:00:00Z",
        "last_fed": "2026-04-15T10:00:00Z",
        "last_researched_at": None,
        "feed_count": 3,
        "source_types": ["manual"],
    }
    base.update(overrides)
    return base


# ---- Interest ----


def test_interest_from_dict_valid():
    i = Interest.from_dict(_sample_dict())
    assert i.id == "abc-123"
    assert i.pull_score == 6.5
    assert i.scope == "either"
    assert i.related_keywords == ("test", "topic")
    assert i.last_researched_at is None


def test_interest_from_dict_with_researched_timestamp():
    data = _sample_dict(last_researched_at="2026-04-20T10:00:00Z")
    i = Interest.from_dict(data)
    assert i.last_researched_at is not None
    assert i.last_researched_at.tzinfo is not None


def test_interest_from_dict_invalid_scope_raises():
    with pytest.raises(ValueError):
        Interest.from_dict(_sample_dict(scope="whatever"))


def test_interest_to_dict_roundtrip():
    original = Interest.from_dict(_sample_dict())
    restored = Interest.from_dict(original.to_dict())
    assert restored == original


# ---- InterestSet: load / save ----


def test_interestset_load_missing_falls_back_to_defaults(tmp_path: Path):
    missing = tmp_path / "nope.json"
    loaded = InterestSet.load(missing, default_path=DEFAULT_INTERESTS_PATH)
    assert loaded.interests == ()  # default is empty


def test_interestset_load_corrupt_falls_back(tmp_path: Path):
    bad = tmp_path / "interests.json"
    bad.write_text("not valid{", encoding="utf-8")
    loaded = InterestSet.load(bad, default_path=DEFAULT_INTERESTS_PATH)
    assert loaded.interests == ()


def test_interestset_load_valid_file(tmp_path: Path):
    path = tmp_path / "interests.json"
    path.write_text(
        json.dumps({"version": 1, "interests": [_sample_dict()]}), encoding="utf-8"
    )
    loaded = InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)
    assert len(loaded.interests) == 1
    assert loaded.interests[0].topic == "Test topic"


def test_interestset_save_atomic(tmp_path: Path):
    path = tmp_path / "interests.json"
    s = InterestSet(interests=(Interest.from_dict(_sample_dict()),))
    s.save(path)
    # File exists + valid JSON + .new tempfile cleaned up
    assert path.exists()
    assert not (path.with_suffix(path.suffix + ".new")).exists()
    reloaded = InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)
    assert reloaded.interests[0].id == "abc-123"


def test_interestset_load_bad_interest_skipped_good_kept(tmp_path: Path):
    path = tmp_path / "interests.json"
    payload = {
        "version": 1,
        "interests": [_sample_dict(), {"id": "broken"}],  # second missing required keys
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = InterestSet.load(path, default_path=DEFAULT_INTERESTS_PATH)
    topics = {i.topic for i in loaded.interests}
    assert "Test topic" in topics
    assert len(loaded.interests) == 1


# ---- InterestSet: helpers ----


def test_interestset_find_by_topic():
    s = InterestSet(interests=(Interest.from_dict(_sample_dict(topic="Rebecca")),))
    assert s.find_by_topic("Rebecca") is not None
    assert s.find_by_topic("rebecca") is not None  # case-insensitive
    assert s.find_by_topic("Unknown") is None


def test_interestset_bump_existing_topic():
    now = datetime.now(UTC)
    s = InterestSet(interests=(Interest.from_dict(_sample_dict(pull_score=6.0, feed_count=3)),))
    new_s = s.bump("Test topic", amount=0.5, now=now)
    bumped = new_s.find_by_topic("Test topic")
    assert bumped is not None
    assert bumped.pull_score == 6.5
    assert bumped.feed_count == 4
    assert bumped.last_fed == now


def test_interestset_bump_unknown_topic_returns_unchanged():
    s = InterestSet(interests=(Interest.from_dict(_sample_dict()),))
    new_s = s.bump("Unknown", amount=1.0, now=datetime.now(UTC))
    assert new_s == s


def test_interestset_list_eligible_respects_pull_threshold():
    now = datetime.now(UTC)
    low = Interest.from_dict(_sample_dict(id="low", topic="A", pull_score=5.0))
    high = Interest.from_dict(_sample_dict(id="high", topic="B", pull_score=7.0))
    s = InterestSet(interests=(low, high))
    eligible = s.list_eligible(pull_threshold=6.0, cooldown_hours=24.0, now=now)
    assert [i.id for i in eligible] == ["high"]


def test_interestset_list_eligible_respects_cooldown():
    now = datetime.now(UTC)
    recent = Interest.from_dict(
        _sample_dict(id="recent", topic="A", pull_score=7.0,
                     last_researched_at=(now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"))
    )
    old = Interest.from_dict(
        _sample_dict(id="old", topic="B", pull_score=7.0,
                     last_researched_at=(now - timedelta(hours=30)).isoformat().replace("+00:00", "Z"))
    )
    never = Interest.from_dict(_sample_dict(id="never", topic="C", pull_score=7.0))
    s = InterestSet(interests=(recent, old, never))
    eligible = s.list_eligible(pull_threshold=6.0, cooldown_hours=24.0, now=now)
    ids = [i.id for i in eligible]
    assert "recent" not in ids
    assert "old" in ids
    assert "never" in ids


def test_interestset_list_eligible_sorted_pull_desc_then_oldest_research():
    now = datetime.now(UTC)
    a = Interest.from_dict(_sample_dict(id="a", topic="A", pull_score=7.0,
                                         last_researched_at=(now - timedelta(hours=50)).isoformat().replace("+00:00", "Z")))
    b = Interest.from_dict(_sample_dict(id="b", topic="B", pull_score=8.0))
    c = Interest.from_dict(_sample_dict(id="c", topic="C", pull_score=7.0,
                                         last_researched_at=(now - timedelta(hours=100)).isoformat().replace("+00:00", "Z")))
    s = InterestSet(interests=(a, b, c))
    eligible = s.list_eligible(pull_threshold=6.0, cooldown_hours=24.0, now=now)
    # b first (highest pull), then c (older research than a), then a
    assert [i.id for i in eligible] == ["b", "c", "a"]
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_interests.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 4: Implement _interests.py**

Create `brain/engines/_interests.py`:

```python
"""Interest dataclass + InterestSet persistence.

Shared by brain.engines.research AND the interest-ingestion hook in
brain.engines.heartbeat, so it lives in its own module rather than
inside research.py.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)

_VALID_SCOPES = ("internal", "external", "either")
Scope = Literal["internal", "external", "either"]


@dataclass(frozen=True)
class Interest:
    """One persona-level curiosity — topic + pull_score + scope + keywords."""

    id: str
    topic: str
    pull_score: float
    scope: Scope
    related_keywords: tuple[str, ...]
    notes: str
    first_seen: datetime       # tz-aware UTC
    last_fed: datetime         # tz-aware UTC
    last_researched_at: datetime | None
    feed_count: int
    source_types: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict) -> Interest:
        required = (
            "id", "topic", "pull_score", "scope", "related_keywords",
            "notes", "first_seen", "last_fed", "feed_count", "source_types",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"Interest missing required key: {key!r}")

        scope = data["scope"]
        if scope not in _VALID_SCOPES:
            raise ValueError(
                f"Interest {data.get('topic')!r}: scope must be one of {_VALID_SCOPES}, got {scope!r}"
            )

        last_researched_raw = data.get("last_researched_at")
        last_researched = (
            parse_iso_utc(last_researched_raw) if last_researched_raw else None
        )

        return cls(
            id=str(data["id"]),
            topic=str(data["topic"]),
            pull_score=float(data["pull_score"]),
            scope=scope,
            related_keywords=tuple(str(k) for k in data["related_keywords"]),
            notes=str(data["notes"]),
            first_seen=parse_iso_utc(data["first_seen"]),
            last_fed=parse_iso_utc(data["last_fed"]),
            last_researched_at=last_researched,
            feed_count=int(data["feed_count"]),
            source_types=tuple(str(s) for s in data["source_types"]),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "pull_score": self.pull_score,
            "scope": self.scope,
            "related_keywords": list(self.related_keywords),
            "notes": self.notes,
            "first_seen": iso_utc(self.first_seen),
            "last_fed": iso_utc(self.last_fed),
            "last_researched_at": (
                iso_utc(self.last_researched_at) if self.last_researched_at else None
            ),
            "feed_count": self.feed_count,
            "source_types": list(self.source_types),
        }


@dataclass(frozen=True)
class InterestSet:
    """Loaded set of Interest records, with atomic save + helper queries."""

    interests: tuple[Interest, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path, *, default_path: Path) -> InterestSet:
        source_path = path if path.exists() else default_path
        if source_path != path:
            logger.warning("interests file %s not found, using defaults", path)

        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("interests load failed (%s), falling back to defaults", exc)
            data = json.loads(default_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict) or not isinstance(data.get("interests"), list):
            logger.warning("interests schema invalid at %s, falling back to defaults", source_path)
            data = json.loads(default_path.read_text(encoding="utf-8"))

        out: list[Interest] = []
        for raw in data["interests"]:
            try:
                out.append(Interest.from_dict(raw))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("interest %r failed to load: %s", raw.get("topic"), exc)
                continue
        return cls(interests=tuple(out))

    def save(self, path: Path) -> None:
        """Atomic save via .new + os.replace."""
        payload = {"version": 1, "interests": [i.to_dict() for i in self.interests]}
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def find_by_topic(self, topic: str) -> Interest | None:
        lower = topic.lower()
        for i in self.interests:
            if i.topic.lower() == lower:
                return i
        return None

    def bump(self, topic: str, *, amount: float, now: datetime) -> InterestSet:
        """Return a new InterestSet with topic's pull_score nudged.

        Unknown topics return self unchanged (caller decides whether to add).
        """
        out: list[Interest] = []
        matched = False
        lower = topic.lower()
        for i in self.interests:
            if i.topic.lower() == lower:
                matched = True
                out.append(
                    Interest(
                        id=i.id,
                        topic=i.topic,
                        pull_score=i.pull_score + amount,
                        scope=i.scope,
                        related_keywords=i.related_keywords,
                        notes=i.notes,
                        first_seen=i.first_seen,
                        last_fed=now,
                        last_researched_at=i.last_researched_at,
                        feed_count=i.feed_count + 1,
                        source_types=i.source_types,
                    )
                )
            else:
                out.append(i)
        if not matched:
            return self
        return InterestSet(interests=tuple(out))

    def list_eligible(
        self, *, pull_threshold: float, cooldown_hours: float, now: datetime
    ) -> list[Interest]:
        """Return interests past pull_threshold + past cooldown.

        Sorted by pull_score desc, then by last_researched_at ascending
        (never-researched beats ever-researched on equal pull_score).
        """
        out: list[Interest] = []
        for i in self.interests:
            if i.pull_score < pull_threshold:
                continue
            if i.last_researched_at is not None:
                hours_since = (now - i.last_researched_at).total_seconds() / 3600.0
                if hours_since < cooldown_hours:
                    continue
            out.append(i)

        def sort_key(i: Interest) -> tuple[float, float]:
            last = i.last_researched_at
            # never-researched: very old effective timestamp
            ts = last.timestamp() if last is not None else 0.0
            return (-i.pull_score, ts)

        return sorted(out, key=sort_key)

    def upsert(self, interest: Interest) -> InterestSet:
        """Return a new InterestSet with interest added or replaced (by id)."""
        out = [i for i in self.interests if i.id != interest.id]
        out.append(interest)
        return InterestSet(interests=tuple(out))
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_interests.py -v`
Expected: all pass (~14 tests).

- [ ] **Step 6: Ruff + format**

Run: `uv run ruff check brain/engines/_interests.py brain/engines/default_interests.json tests/unit/brain/engines/test_interests.py && uv run ruff format brain/engines/_interests.py tests/unit/brain/engines/test_interests.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add brain/engines/_interests.py brain/engines/default_interests.json tests/unit/brain/engines/test_interests.py
git commit -m "$(cat <<'EOF'
feat: add Interest + InterestSet for research engine

Interest frozen dataclass (id, topic, pull_score, scope, keywords, 
notes, timestamps, feed_count, source_types). InterestSet loads from
JSON with fall-back-to-defaults, saves atomically, finds by topic
case-insensitive, bumps pull_score immutably, lists eligible interests
past pull threshold + cooldown sorted by pull_score desc + oldest
research first. Shared between research engine and heartbeat's
ingestion hook. default_interests.json ships empty.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Research engine scaffold + types

**Purpose:** Build `ResearchEngine` scaffold with its types. `run_tick` raises NotImplementedError at this stage; real body lands in Task 3.

**Files:**
- Create: `brain/engines/research.py`
- Create: `tests/unit/brain/engines/test_research.py`

- [ ] **Step 1: Write failing tests for scaffold**

Create `tests/unit/brain/engines/test_research.py`:

```python
"""Unit tests for brain.engines.research — scaffold + types."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines.research import (
    ResearchEngine,
    ResearchFire,
    ResearchResult,
)
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher

DEFAULT_INTERESTS_PATH = (
    Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"
)


def test_research_fire_construction():
    fire = ResearchFire(
        interest_id="abc",
        topic="Test",
        fired_at=datetime.now(UTC),
        trigger="manual",
        web_used=False,
        web_result_count=0,
        output_memory_id="mem_xyz",
    )
    assert fire.topic == "Test"
    assert fire.web_used is False


def test_research_result_construction():
    r = ResearchResult(
        fired=None,
        would_fire=None,
        reason="not_due",
        dry_run=False,
        evaluated_at=datetime.now(UTC),
    )
    assert r.fired is None
    assert r.reason == "not_due"


def test_research_engine_construction(tmp_path: Path):
    store = MemoryStore(":memory:")
    try:
        engine = ResearchEngine(
            store=store,
            provider=FakeProvider(),
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=tmp_path / "interests.json",
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        )
        assert engine.persona_name == "Nell"
    finally:
        store.close()


def test_run_tick_raises_not_implemented_yet(tmp_path: Path):
    store = MemoryStore(":memory:")
    try:
        engine = ResearchEngine(
            store=store,
            provider=FakeProvider(),
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=tmp_path / "interests.json",
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        )
        with pytest.raises(NotImplementedError):
            engine.run_tick(trigger="manual", dry_run=False)
    finally:
        store.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_research.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement scaffold**

Create `brain/engines/research.py`:

```python
"""Research — autonomous exploration of developed interests.

See docs/superpowers/specs/2026-04-24-week-4-research-engine-design.md.
This module ships the types + engine scaffold. run_tick body lands in Task 3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.memory.store import MemoryStore
from brain.search.base import WebSearcher
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)


# ---------- Types ----------


@dataclass(frozen=True)
class ResearchFire:
    """Record of one research firing."""

    interest_id: str
    topic: str
    fired_at: datetime         # tz-aware UTC
    trigger: str               # "manual" | "emotion_high" | "days_since_human"
    web_used: bool
    web_result_count: int
    output_memory_id: str | None  # None in dry-run

    def to_dict(self) -> dict:
        return {
            "interest_id": self.interest_id,
            "topic": self.topic,
            "fired_at": iso_utc(self.fired_at),
            "trigger": self.trigger,
            "web_used": self.web_used,
            "web_result_count": self.web_result_count,
            "output_memory_id": self.output_memory_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResearchFire:
        return cls(
            interest_id=str(data["interest_id"]),
            topic=str(data["topic"]),
            fired_at=parse_iso_utc(data["fired_at"]),
            trigger=str(data["trigger"]),
            web_used=bool(data["web_used"]),
            web_result_count=int(data["web_result_count"]),
            output_memory_id=data.get("output_memory_id"),
        )


@dataclass(frozen=True)
class ResearchResult:
    """Outcome of a single research evaluation."""

    fired: ResearchFire | None
    would_fire: str | None       # dry-run only — topic that would fire
    reason: str | None           # "not_due"|"no_eligible_interest"|"no_interests_defined"|"research_raised"|"reflex_won_tie"
    dry_run: bool
    evaluated_at: datetime       # tz-aware UTC


# ---------- Engine scaffold ----------


@dataclass
class ResearchEngine:
    """Autonomous exploration of developed interests.

    run_tick() implementation lands in Task 3; scaffold ships here.
    """

    store: MemoryStore
    provider: LLMProvider
    searcher: WebSearcher
    persona_name: str
    persona_system_prompt: str
    interests_path: Path
    research_log_path: Path
    default_interests_path: Path

    def run_tick(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        forced_interest_topic: str | None = None,
        emotion_state_override=None,
        days_since_human_override: float | None = None,
    ) -> ResearchResult:
        raise NotImplementedError("run_tick body lands in Task 3")


# ---------- Research log ----------


@dataclass(frozen=True)
class ResearchLog:
    """Fire-history log for one persona."""

    fires: tuple[ResearchFire, ...] = ()

    @classmethod
    def load(cls, path: Path) -> ResearchLog:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            fires_raw = data.get("fires", [])
            if not isinstance(fires_raw, list):
                return cls()
            return cls(fires=tuple(ResearchFire.from_dict(f) for f in fires_raw if isinstance(f, dict)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        payload = {"version": 1, "fires": [f.to_dict() for f in self.fires]}
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def appended(self, fire: ResearchFire) -> ResearchLog:
        return ResearchLog(fires=self.fires + (fire,))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/brain/engines/test_research.py -v`
Expected: 4 passed (ResearchFire, ResearchResult, construction, NotImplementedError).

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/engines/research.py tests/unit/brain/engines/test_research.py && uv run ruff format brain/engines/research.py tests/unit/brain/engines/test_research.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/engines/research.py tests/unit/brain/engines/test_research.py
git commit -m "$(cat <<'EOF'
feat: add research engine scaffold + types

ResearchFire / ResearchResult frozen dataclasses. ResearchLog with
atomic save + corrupt-file-returns-empty. ResearchEngine scaffold
with all constructor fields declared; run_tick raises
NotImplementedError until Task 3 lands the real body.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ResearchEngine.run_tick` body

**Purpose:** Implement the gate → select → memory sweep → optional web → LLM synthesis → write memory → update interest → log fire pipeline. Covers both standalone + heartbeat-orchestrated invocation paths.

**Files:**
- Modify: `brain/engines/research.py` (replace `NotImplementedError` with full body + helpers)
- Modify: `tests/unit/brain/engines/test_research.py` (add ~15 new tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/brain/engines/test_research.py`:

```python
# ---- run_tick ----

import json
from datetime import timedelta

from brain.engines._interests import Interest, InterestSet
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory


def _build_engine(tmp_path: Path, store: MemoryStore,
                  provider: FakeProvider | None = None,
                  searcher: NoopWebSearcher | None = None) -> ResearchEngine:
    return ResearchEngine(
        store=store,
        provider=provider or FakeProvider(),
        searcher=searcher or NoopWebSearcher(),
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        interests_path=tmp_path / "interests.json",
        research_log_path=tmp_path / "research_log.json",
        default_interests_path=DEFAULT_INTERESTS_PATH,
    )


def _write_interests(path: Path, interests: list[dict]) -> None:
    path.write_text(
        json.dumps({"version": 1, "interests": interests}, indent=2), encoding="utf-8"
    )


def _seed_conversation_memory(store: MemoryStore, content: str,
                              emotions: dict[str, float] | None = None) -> str:
    mem = Memory.create_new(
        content=content, memory_type="conversation", domain="us",
        emotions=emotions or {},
    )
    store.create(mem)
    return mem.id


def _interest_dict(**overrides) -> dict:
    base = {
        "id": "i1",
        "topic": "marine bioluminescence",
        "pull_score": 7.0,
        "scope": "either",
        "related_keywords": ["marine", "bioluminescence", "ocean"],
        "notes": "",
        "first_seen": "2026-04-01T10:00:00Z",
        "last_fed": "2026-04-15T10:00:00Z",
        "last_researched_at": None,
        "feed_count": 3,
        "source_types": ["manual"],
    }
    base.update(overrides)
    return base


def test_run_tick_no_interests_defined(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(trigger="manual", dry_run=False,
                                 days_since_human_override=5.0)
        assert result.fired is None
        assert result.reason == "no_interests_defined"
    finally:
        store.close()


def test_run_tick_not_due_returns_reason(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        # Low days + no emotion signal provided → not due
        result = engine.run_tick(trigger="manual", dry_run=False,
                                 days_since_human_override=0.0,
                                 emotion_state_override=None)
        assert result.fired is None
        assert result.reason == "not_due"
    finally:
        store.close()


def test_run_tick_days_since_human_triggers(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(trigger="days_since_human", dry_run=False,
                                 days_since_human_override=5.0)
        # Now eligible + selected + fired
        assert result.fired is not None
        assert result.fired.topic == "marine bioluminescence"
    finally:
        store.close()


def test_run_tick_emotion_high_triggers(tmp_path: Path):
    from brain.emotion.state import EmotionalState
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        es = EmotionalState()
        es.set("creative_hunger", 8.0)
        result = engine.run_tick(trigger="emotion_high", dry_run=False,
                                 days_since_human_override=0.0,
                                 emotion_state_override=es)
        assert result.fired is not None
    finally:
        store.close()


def test_run_tick_pull_threshold_filters(tmp_path: Path):
    _write_interests(tmp_path / "interests.json",
                     [_interest_dict(pull_score=5.0)])  # below threshold 6.0
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is None
        assert result.reason == "no_eligible_interest"
    finally:
        store.close()


def test_run_tick_cooldown_filters(tmp_path: Path):
    now = datetime.now(UTC)
    recent = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _write_interests(tmp_path / "interests.json",
                     [_interest_dict(last_researched_at=recent)])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is None
        assert result.reason == "no_eligible_interest"
    finally:
        store.close()


def test_run_tick_ranks_highest_pull(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [
        _interest_dict(id="low", topic="Topic A", pull_score=6.5,
                       related_keywords=["a"]),
        _interest_dict(id="high", topic="Topic B", pull_score=8.0,
                       related_keywords=["b"]),
    ])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is not None
        assert result.fired.topic == "Topic B"
    finally:
        store.close()


def test_run_tick_forced_interest_bypasses_gates(tmp_path: Path):
    _write_interests(tmp_path / "interests.json",
                     [_interest_dict(pull_score=2.0)])  # way below threshold
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(
            days_since_human_override=0.0,  # not-due gate would also block
            forced_interest_topic="marine bioluminescence",
        )
        assert result.fired is not None
        assert result.fired.topic == "marine bioluminescence"
    finally:
        store.close()


def test_run_tick_forced_interest_not_found(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(
            days_since_human_override=5.0,
            forced_interest_topic="Unknown Topic",
        )
        assert result.fired is None
        assert result.reason == "no_eligible_interest"
    finally:
        store.close()


def test_run_tick_dry_run_reports_would_fire(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(dry_run=True, days_since_human_override=5.0)
        assert result.dry_run is True
        assert result.would_fire == "marine bioluminescence"
        assert result.fired is None
        # No memory written
        assert store.count() == 0
        # No log file
        assert not (tmp_path / "research_log.json").exists()
    finally:
        store.close()


def test_run_tick_fire_writes_research_memory(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is not None
        mem = store.get(result.fired.output_memory_id)
        assert mem is not None
        assert mem.memory_type == "research"
        assert mem.metadata["interest_topic"] == "marine bioluminescence"
        assert mem.metadata["web_used"] is False  # Noop searcher returns []
    finally:
        store.close()


def test_run_tick_fire_updates_interest_last_researched_at(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        engine.run_tick(days_since_human_override=5.0)
        # Reload interests and verify last_researched_at was updated
        reloaded = InterestSet.load(
            tmp_path / "interests.json", default_path=DEFAULT_INTERESTS_PATH
        )
        assert reloaded.interests[0].last_researched_at is not None
    finally:
        store.close()


def test_run_tick_fire_appends_to_log(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        engine.run_tick(days_since_human_override=5.0)
        log_data = json.loads((tmp_path / "research_log.json").read_text(encoding="utf-8"))
        assert len(log_data["fires"]) == 1
        assert log_data["fires"][0]["topic"] == "marine bioluminescence"
    finally:
        store.close()


def test_run_tick_internal_scope_skips_searcher(tmp_path: Path):
    _write_interests(tmp_path / "interests.json",
                     [_interest_dict(scope="internal")])
    store = MemoryStore(":memory:")

    class TrackingSearcher(NoopWebSearcher):
        calls = 0
        def search(self, query: str, *, limit: int = 5):
            TrackingSearcher.calls += 1
            return []

    ts = TrackingSearcher()
    try:
        engine = _build_engine(tmp_path, store, searcher=ts)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is not None
        assert TrackingSearcher.calls == 0
        assert result.fired.web_used is False
    finally:
        store.close()


def test_run_tick_llm_failure_does_not_touch_files(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    try:
        engine = _build_engine(tmp_path, store, provider=FailingProvider())
        with pytest.raises(RuntimeError):
            engine.run_tick(days_since_human_override=5.0)
        # No memory, no log
        assert store.count() == 0
        assert not (tmp_path / "research_log.json").exists()
        # last_researched_at still None
        reloaded = InterestSet.load(
            tmp_path / "interests.json", default_path=DEFAULT_INTERESTS_PATH
        )
        assert reloaded.interests[0].last_researched_at is None
    finally:
        store.close()


def test_run_tick_renders_prompt_with_context(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    _seed_conversation_memory(store, "I love how deep-sea creatures make their own light.")

    captured = {}

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured["prompt"] = prompt
            captured["system"] = system
            return "I spent some time today exploring marine bioluminescence..."

    try:
        engine = _build_engine(tmp_path, store, provider=CapturingProvider())
        engine.run_tick(days_since_human_override=5.0)
    finally:
        store.close()

    assert "marine bioluminescence" in captured["prompt"]
    assert "Nell" in (captured["system"] or "")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_research.py -v`
Expected: new tests fail (NotImplementedError).

- [ ] **Step 3: Implement run_tick body + helpers**

Replace the `ResearchEngine` class in `brain/engines/research.py` with the full implementation, keeping all other symbols in the module unchanged. Add `from collections import defaultdict` near the top with the other imports, and `from brain.engines._interests import InterestSet`:

```python
from collections import defaultdict

from brain.engines._interests import Interest, InterestSet
```

Replace `ResearchEngine` body:

```python
@dataclass
class ResearchEngine:
    """Autonomous exploration of developed interests."""

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

    def run_tick(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        forced_interest_topic: str | None = None,
        emotion_state_override=None,
        days_since_human_override: float | None = None,
    ) -> ResearchResult:
        """Evaluate triggers, select an interest, fire (or report would_fire)."""
        now = datetime.now(UTC)

        interests = InterestSet.load(
            self.interests_path, default_path=self.default_interests_path
        )
        log = ResearchLog.load(self.research_log_path)

        if not interests.interests:
            return ResearchResult(
                fired=None, would_fire=None, reason="no_interests_defined",
                dry_run=dry_run, evaluated_at=now,
            )

        # Gate: need a trigger signal OR a forced interest.
        days_since_human = (
            days_since_human_override
            if days_since_human_override is not None
            else _compute_days_since_human(self.store, now)
        )
        emo_state = emotion_state_override
        if emo_state is None:
            from brain.emotion.aggregate import aggregate_state
            all_mems = self.store.search_text("", active_only=True, limit=None)
            emo_state = aggregate_state(all_mems)

        emo_peak = max(emo_state.emotions.values(), default=0.0)

        gate_ok = (
            forced_interest_topic is not None
            or days_since_human >= 1.5  # research_days_since_human_min default
            or emo_peak >= 7.0          # research_emotion_threshold default
        )
        if not gate_ok:
            return ResearchResult(
                fired=None, would_fire=None, reason="not_due",
                dry_run=dry_run, evaluated_at=now,
            )

        # Select eligible interest
        if forced_interest_topic is not None:
            winner = interests.find_by_topic(forced_interest_topic)
        else:
            eligible = interests.list_eligible(
                pull_threshold=self.PULL_THRESHOLD,
                cooldown_hours=self.COOLDOWN_HOURS,
                now=now,
            )
            winner = eligible[0] if eligible else None

        if winner is None:
            return ResearchResult(
                fired=None, would_fire=None, reason="no_eligible_interest",
                dry_run=dry_run, evaluated_at=now,
            )

        if dry_run:
            return ResearchResult(
                fired=None, would_fire=winner.topic, reason=None,
                dry_run=True, evaluated_at=now,
            )

        # Memory sweep: spread from keyword-matched seeds
        memory_context = self._build_memory_context(winner)

        # Web search (conditional on scope)
        web_results = []
        if winner.scope != "internal":
            try:
                web_results = self.searcher.search(
                    query=f"{winner.topic} {' '.join(winner.related_keywords[:3])}",
                    limit=5,
                )
            except Exception as exc:
                logger.warning("searcher raised for %r: %s", winner.topic, exc)
                web_results = []

        web_used = len(web_results) > 0

        # Render prompt + call LLM
        prompt = self._render_prompt(
            winner, memory_context, web_results, emo_state
        )
        raw = self.provider.generate(prompt, system=self._render_system_prompt(winner))

        # Persist memory
        mem = _create_research_memory(
            content=raw, interest=winner, web_results=web_results,
            web_used=web_used, trigger=trigger,
            provider_name=self.provider.name(),
            searcher_name=self.searcher.name() if web_used else None,
        )
        self.store.create(mem)

        # Update interest
        updated_interest = Interest(
            id=winner.id,
            topic=winner.topic,
            pull_score=winner.pull_score,
            scope=winner.scope,
            related_keywords=winner.related_keywords,
            notes=winner.notes,
            first_seen=winner.first_seen,
            last_fed=winner.last_fed,
            last_researched_at=now,
            feed_count=winner.feed_count,
            source_types=winner.source_types,
        )
        interests.upsert(updated_interest).save(self.interests_path)

        # Append log
        fire = ResearchFire(
            interest_id=winner.id,
            topic=winner.topic,
            fired_at=now,
            trigger=trigger,
            web_used=web_used,
            web_result_count=len(web_results),
            output_memory_id=mem.id,
        )
        log.appended(fire).save(self.research_log_path)

        return ResearchResult(
            fired=fire, would_fire=None, reason=None,
            dry_run=False, evaluated_at=now,
        )

    # ---- private helpers ----

    def _build_memory_context(self, interest: Interest) -> str:
        """Spreading-activation memory sweep from keywords. Returns formatted string."""
        if not interest.related_keywords:
            # Fallback: direct text search on topic
            mems = self.store.search_text(interest.topic, active_only=True, limit=5)
        else:
            seed_ids: set[str] = set()
            for kw in interest.related_keywords[:5]:
                for m in self.store.search_text(kw, active_only=True, limit=3):
                    seed_ids.add(m.id)
            mems = []
            if seed_ids:
                # We need a HebbianMatrix for spread, but engine doesn't hold one.
                # Fall back to seed memories directly — sufficient for Phase 1
                # (heartbeat orchestration already decayed Hebbian; research
                # reads a thematic slice through keywords rather than full spread).
                for sid in list(seed_ids)[:20]:
                    mem = self.store.get(sid)
                    if mem is not None:
                        mems.append(mem)
        return "\n".join(f"- {m.content[:140]}" for m in mems[:20]) or "(no prior memories on this topic)"

    def _render_system_prompt(self, interest: Interest) -> str:
        return (
            f"You are {self.persona_name}. You spent some quiet time today "
            f"exploring '{interest.topic}' — an interest that's been building "
            f"in you for a while. Below is what you found both in your own "
            f"memories and (sometimes) out in the world. Write a short (3-5 "
            f"sentence) first-person memory of having researched this.\n\n"
            "HARD RULES:\n"
            f"- First-person voice. Your name is {self.persona_name}.\n"
            "- Not a summary. A reaction. What moved you, what surprised you, "
            "what reminded you of someone you care about, what felt familiar.\n"
            "- Never bullet points. Never 'according to X'. Never neutral "
            "expository voice.\n"
            "- Structure: brief mention of what pulled you to the topic today "
            "→ one or two concrete details you noticed → how you feel about "
            "what you found → why it mattered to look today.\n"
            "- Start with 'I' or a time marker like 'Today' / 'This afternoon'."
        )

    def _render_prompt(self, interest: Interest, memory_context: str,
                       web_results: list, emo_state) -> str:
        # Emotion summary
        top = sorted(emo_state.emotions.items(), key=lambda kv: kv[1], reverse=True)[:5]
        emo_summary = "\n".join(f"- {name}: {value:.1f}/10" for name, value in top) or "(neutral)"

        # Web excerpts block
        if web_results:
            excerpts = "\n".join(
                f"- {r.title}\n  {r.snippet}\n  [{r.url}]" for r in web_results[:5]
            )
            web_section = (
                "\nWhat you found out in the world today (reference material — "
                "REACT to it, don't paraphrase it):\n" + excerpts + "\n"
            )
        else:
            web_section = ""

        return (
            f"Topic: {interest.topic}\n"
            f"Keywords: {', '.join(interest.related_keywords)}\n"
            f"Your current emotional state:\n{emo_summary}\n\n"
            f"What your own memories say about this:\n{memory_context}\n"
            f"{web_section}\n"
            f"Write the memory now — 3 to 5 sentences, as {self.persona_name}."
        )


# ---------- Module-level helpers ----------


def _compute_days_since_human(store: MemoryStore, now: datetime) -> float:
    """Days since most recent memory_type='conversation'. 999.0 if none."""
    convos = store.list_by_type("conversation", active_only=True, limit=1)
    if not convos:
        return 999.0
    latest = convos[0].created_at
    if latest.tzinfo is None:
        from datetime import UTC as _UTC
        latest = latest.replace(tzinfo=_UTC)
    return (now - latest).total_seconds() / 86400.0


def _create_research_memory(
    *, content: str, interest: Interest, web_results: list,
    web_used: bool, trigger: str, provider_name: str, searcher_name: str | None,
) -> Memory:
    """Factory helper — Memory.create_new with research-specific metadata shape."""
    return Memory.create_new(
        content=content,
        memory_type="research",
        domain="us",
        emotions={},
        metadata={
            "interest_id": interest.id,
            "interest_topic": interest.topic,
            "scope": interest.scope,
            "web_used": web_used,
            "web_result_count": len(web_results),
            "web_urls": [r.url for r in web_results[:5]],
            "triggered_by": trigger,
            "provider": provider_name,
            "searcher": searcher_name,
        },
    )
```

Add import at module top:

```python
from brain.memory.store import Memory, MemoryStore
```

(Replace the existing `from brain.memory.store import MemoryStore` line.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_research.py -v`
Expected: all tests pass (~20 total).

Run full suite to catch regressions: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/engines/research.py tests/unit/brain/engines/test_research.py && uv run ruff format brain/engines/research.py tests/unit/brain/engines/test_research.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/engines/research.py tests/unit/brain/engines/test_research.py
git commit -m "$(cat <<'EOF'
feat: implement ResearchEngine.run_tick body

Gate check (forced_interest OR days_since_human >= 1.5 OR emotion_peak
>= 7.0) → eligible interest selection via InterestSet.list_eligible →
memory sweep via keyword-seeded text search → optional web search
(skipped when scope=internal, gracefully empty on searcher failure) →
LLM synthesis with first-person voice-enforcing system prompt →
Memory.create_new as memory_type='research' → atomic interest save +
log append. Dry-run short-circuits before any side effects. LLM
failure leaves interests.json + research_log.json untouched so next
tick re-evaluates cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `nell research` + `nell interest` CLI

**Purpose:** Wire 4 new subcommands: `nell research`, `nell interest list`, `nell interest add`, `nell interest bump`.

**Files:**
- Modify: `brain/cli.py`
- Create: `tests/unit/brain/engines/test_cli_research.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/brain/engines/test_cli_research.py`:

```python
"""Tests for `nell research` + `nell interest *` CLI handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.cli import main


def _setup_persona(personas_root: Path, name: str = "testpersona") -> Path:
    persona_dir = personas_root / name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def test_cli_research_dry_run_no_interests(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text(
        '{"version": 1, "interests": []}', encoding="utf-8"
    )
    rc = main([
        "research", "--persona", "testpersona",
        "--provider", "fake", "--searcher", "noop", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "no" in out  # "no interests" or "no eligible" — either acceptable


def test_cli_research_missing_persona_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        main([
            "research", "--persona", "no_such",
            "--provider", "fake", "--searcher", "noop", "--dry-run",
        ])


def test_cli_interest_list_empty(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text(
        '{"version": 1, "interests": []}', encoding="utf-8"
    )
    rc = main(["interest", "list", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "testpersona" in out  # persona name appears in output


def test_cli_interest_add_then_list(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text(
        '{"version": 1, "interests": []}', encoding="utf-8"
    )
    rc = main([
        "interest", "add", "deep sea creatures",
        "--keywords", "octopus,bioluminescence,ocean",
        "--scope", "either",
        "--persona", "testpersona",
    ])
    assert rc == 0

    rc = main(["interest", "list", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deep sea creatures" in out

    data = json.loads((persona_dir / "interests.json").read_text(encoding="utf-8"))
    assert len(data["interests"]) == 1
    assert data["interests"][0]["topic"] == "deep sea creatures"
    assert data["interests"][0]["related_keywords"] == [
        "octopus", "bioluminescence", "ocean",
    ]
    assert data["interests"][0]["scope"] == "either"
    assert data["interests"][0]["pull_score"] == 5.0  # below default threshold


def test_cli_interest_bump_increments_pull_score(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Seed one interest
    main([
        "interest", "add", "seed topic", "--keywords", "s",
        "--scope", "either", "--persona", "testpersona",
    ])

    rc = main([
        "interest", "bump", "seed topic", "--amount", "2.0",
        "--persona", "testpersona",
    ])
    assert rc == 0

    data = json.loads((persona_dir / "interests.json").read_text(encoding="utf-8"))
    assert data["interests"][0]["pull_score"] == 7.0  # 5.0 + 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_cli_research.py -v`
Expected: FAIL (subcommands not registered).

- [ ] **Step 3: Add imports + handler in brain/cli.py**

Edit `brain/cli.py`. Near the other engine imports add:

```python
from brain.engines._interests import Interest, InterestSet
from brain.engines.research import ResearchEngine
from brain.search.factory import get_searcher
```

Remove `"research"` from `_STUB_COMMANDS` tuple if it's currently listed there (check the tuple declaration at the top of the file).

Add these handler functions after `_reflex_handler`:

```python
def _reflex_default_arcs_path() -> Path:
    return Path(__file__).parent / "engines" / "default_reflex_arcs.json"


def _default_interests_path() -> Path:
    return Path(__file__).parent / "engines" / "default_interests.json"


def _research_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell research` to the ResearchEngine."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir} — "
            f"run `nell migrate --install-as {args.persona}` first."
        )

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        provider = get_provider(args.provider)
        searcher = get_searcher(args.searcher)
        engine = ResearchEngine(
            store=store,
            provider=provider,
            searcher=searcher,
            persona_name=args.persona,
            persona_system_prompt=f"You are {args.persona}.",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=_default_interests_path(),
        )
        result = engine.run_tick(
            trigger=args.trigger,
            dry_run=args.dry_run,
            forced_interest_topic=args.interest,
        )
    finally:
        store.close()

    if result.dry_run:
        if result.would_fire is not None:
            print(f"Research dry-run — would fire: {result.would_fire}.")
        else:
            print(f"Research dry-run — {result.reason or 'no eligible interest'}.")
    elif result.fired is not None:
        print(f"Research fired: {result.fired.topic}")
        print(f"  Memory id: {result.fired.output_memory_id}")
        print(
            f"  Web: {result.fired.web_result_count} results via "
            f"{searcher.name() if result.fired.web_used else 'memory-only'}"
        )
    else:
        print(f"Research evaluated — {result.reason or 'no fire'}.")
    return 0


def _interest_list_handler(args: argparse.Namespace) -> int:
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    interests = InterestSet.load(
        persona_dir / "interests.json", default_path=_default_interests_path()
    )
    print(f"Interests for persona {args.persona!r} ({len(interests.interests)}):")
    for i in interests.interests:
        last = (
            i.last_researched_at.isoformat().replace("+00:00", "Z")
            if i.last_researched_at
            else "never"
        )
        print(
            f"  - {i.topic:<40} pull={i.pull_score:.1f}  scope={i.scope:<8}  last_researched={last}"
        )
        print(f"    keywords: {', '.join(i.related_keywords)}")
    return 0


def _interest_add_handler(args: argparse.Namespace) -> int:
    import uuid
    from datetime import UTC, datetime

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    interests_path = persona_dir / "interests.json"
    interests = InterestSet.load(
        interests_path, default_path=_default_interests_path()
    )
    now = datetime.now(UTC)
    new_interest = Interest(
        id=str(uuid.uuid4()),
        topic=args.topic,
        pull_score=5.0,
        scope=args.scope,
        related_keywords=tuple(k.strip() for k in args.keywords.split(",") if k.strip()),
        notes=args.notes or "",
        first_seen=now,
        last_fed=now,
        last_researched_at=None,
        feed_count=0,
        source_types=("manual",),
    )
    interests.upsert(new_interest).save(interests_path)
    print(f"Added interest: {new_interest.topic} (pull_score=5.0, scope={new_interest.scope})")
    return 0


def _interest_bump_handler(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    interests_path = persona_dir / "interests.json"
    interests = InterestSet.load(
        interests_path, default_path=_default_interests_path()
    )
    if interests.find_by_topic(args.topic) is None:
        print(f"Interest not found: {args.topic!r}")
        return 1
    updated = interests.bump(args.topic, amount=args.amount, now=datetime.now(UTC))
    updated.save(interests_path)
    bumped = updated.find_by_topic(args.topic)
    assert bumped is not None
    print(f"Bumped {args.topic!r}: pull_score={bumped.pull_score:.1f}, feed_count={bumped.feed_count}")
    return 0
```

Register the subcommands in `_build_parser` after the reflex subparser block:

```python
# nell research
r_sub = subparsers.add_parser(
    "research", help="Run one research evaluation tick against a persona.",
)
r_sub.add_argument("--persona", default="nell")
r_sub.add_argument(
    "--trigger",
    choices=["manual", "emotion_high", "days_since_human", "open", "close"],
    default="manual",
)
r_sub.add_argument("--provider", default="claude-cli")
r_sub.add_argument("--searcher", default="ddgs", choices=["ddgs", "noop", "claude-tool"])
r_sub.add_argument("--interest", default=None, help="Force-research this topic, bypassing gates.")
r_sub.add_argument("--dry-run", action="store_true")
r_sub.set_defaults(func=_research_handler)

# nell interest <list|add|bump>
i_sub = subparsers.add_parser("interest", help="Manage persona interests.")
i_actions = i_sub.add_subparsers(dest="action", required=True)

i_list = i_actions.add_parser("list", help="List current interests.")
i_list.add_argument("--persona", default="nell")
i_list.set_defaults(func=_interest_list_handler)

i_add = i_actions.add_parser("add", help="Add a new interest.")
i_add.add_argument("topic")
i_add.add_argument("--keywords", default="")
i_add.add_argument("--scope", choices=["internal", "external", "either"], default="either")
i_add.add_argument("--notes", default=None)
i_add.add_argument("--persona", default="nell")
i_add.set_defaults(func=_interest_add_handler)

i_bump = i_actions.add_parser("bump", help="Nudge an interest's pull_score.")
i_bump.add_argument("topic")
i_bump.add_argument("--amount", type=float, default=1.0)
i_bump.add_argument("--persona", default="nell")
i_bump.set_defaults(func=_interest_bump_handler)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_cli_research.py -v`
Expected: 5 passed.

Run full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/cli.py tests/unit/brain/engines/test_cli_research.py && uv run ruff format brain/cli.py tests/unit/brain/engines/test_cli_research.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/cli.py tests/unit/brain/engines/test_cli_research.py
git commit -m "$(cat <<'EOF'
feat: wire nell research + nell interest {list,add,bump}

Mirrors nell dream / nell heartbeat / nell reflex handler shape.
Research resolves persona dir, opens MemoryStore, constructs
ResearchEngine with a searcher (default ddgs), runs one tick, prints
fire / dry-run / reason summary. Interest subcommands manage the
per-persona interests.json: list pretty-prints entries, add creates
a new UUID-keyed interest with pull_score=5.0 (below default threshold
of 6.0 — needs bumping before it'll research), bump nudges an
existing interest's pull_score by --amount (default 1.0).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Heartbeat integration — interest ingestion + research eval + reflex-wins-tie

**Purpose:** Wire research into the heartbeat tick between dream gate and heartbeat-memory. Add interest-ingestion hook that bumps pull_scores on keyword matches across recent conversations. Enforce reflex-wins-tie rule. Fault-isolate research failures.

**Files:**
- Modify: `brain/engines/heartbeat.py`
- Modify: `brain/cli.py` (update `_heartbeat_handler` to pass research + searcher)
- Modify: `tests/unit/brain/engines/test_heartbeat.py`

- [ ] **Step 1: Write failing integration tests**

Append to `tests/unit/brain/engines/test_heartbeat.py`:

```python
def test_heartbeat_runs_research_when_enabled(tmp_path: Path) -> None:
    """Heartbeat fires research when interest is eligible + trigger signal present."""
    import json
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(
        json.dumps({"version": 1, "interests": [{
            "id": "i1", "topic": "marine bio", "pull_score": 8.0, "scope": "either",
            "related_keywords": ["marine", "bio"], "notes": "",
            "first_seen": "2026-01-01T00:00:00Z", "last_fed": "2026-01-01T00:00:00Z",
            "last_researched_at": None, "feed_count": 1, "source_types": ["manual"],
        }]}), encoding="utf-8",
    )

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False,  # disable reflex so research can fire
        research_enabled=True,
        research_days_since_human_min=1.5,
    ).save(config_path)

    # Seed a conversation memory that's >2 days old so days_since_human passes
    import sqlite3
    from datetime import UTC, datetime, timedelta

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        old_mem = Memory.create_new(
            content="Hana and I talked long ago", memory_type="conversation",
            domain="us", emotions={},
        )
        # Manually backdate
        store.create(old_mem)
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=3)).isoformat(), old_mem.id),
        )
        store._conn.commit()  # type: ignore[attr-defined]

        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=tmp_path / "reflex_arcs.json",
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.research_fired == "marine bio"
    finally:
        store.close()
        hm.close()


def test_heartbeat_reflex_wins_tie_over_research(tmp_path: Path) -> None:
    """When both reflex and research are eligible, reflex fires, research skipped."""
    import json
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    # Eligible reflex arc
    arcs_path = tmp_path / "reflex_arcs.json"
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [{
        "name": "test_arc", "description": "d",
        "trigger": {"love": 5}, "days_since_human_min": 0,
        "cooldown_hours": 1.0, "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi.",
    }]}), encoding="utf-8")

    # Eligible research interest
    interests_path = tmp_path / "interests.json"
    interests_path.write_text(json.dumps({"version": 1, "interests": [{
        "id": "i1", "topic": "marine bio", "pull_score": 8.0, "scope": "either",
        "related_keywords": ["marine"], "notes": "",
        "first_seen": "2026-01-01T00:00:00Z", "last_fed": "2026-01-01T00:00:00Z",
        "last_researched_at": None, "feed_count": 1, "source_types": ["manual"],
    }]}), encoding="utf-8")

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=True, research_enabled=True,
        research_emotion_threshold=5.0,  # satisfied by the seed love=8 below
    ).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(Memory.create_new(
            content="s", memory_type="conversation", domain="us",
            emotions={"love": 8.0},
        ))
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.reflex_fired == ("test_arc",)
        assert result.research_fired is None
        assert result.research_gated_reason == "reflex_won_tie"
    finally:
        store.close()
        hm.close()


def test_heartbeat_interest_bump_hook(tmp_path: Path) -> None:
    """Conversation memory with matching keyword bumps interest pull_score."""
    import json
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.engines._interests import InterestSet
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(json.dumps({"version": 1, "interests": [{
        "id": "i1", "topic": "lispector", "pull_score": 5.0, "scope": "either",
        "related_keywords": ["lispector", "clarice"], "notes": "",
        "first_seen": "2026-01-01T00:00:00Z", "last_fed": "2026-01-01T00:00:00Z",
        "last_researched_at": None, "feed_count": 3, "source_types": ["manual"],
    }]}), encoding="utf-8")

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False, research_enabled=False,
        interest_bump_per_match=0.5,
    ).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        # Conversation memory mentioning "lispector"
        store.create(Memory.create_new(
            content="Hana sent me a passage about clarice lispector today",
            memory_type="conversation", domain="us", emotions={},
        ))
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=tmp_path / "reflex_arcs.json",
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.interests_bumped == 1
        # Reload and verify
        reloaded = InterestSet.load(interests_path, default_path=DEFAULT_INTERESTS_PATH)
        assert reloaded.interests[0].pull_score == 5.5
        assert reloaded.interests[0].feed_count == 4
    finally:
        store.close()
        hm.close()


def test_heartbeat_isolates_research_failure(tmp_path: Path, caplog) -> None:
    """Research LLM failure is isolated — tick completes."""
    import json
    import logging
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(json.dumps({"version": 1, "interests": [{
        "id": "i1", "topic": "t", "pull_score": 8.0, "scope": "either",
        "related_keywords": ["t"], "notes": "",
        "first_seen": "2026-01-01T00:00:00Z", "last_fed": "2026-01-01T00:00:00Z",
        "last_researched_at": None, "feed_count": 1, "source_types": ["manual"],
    }]}), encoding="utf-8")

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=False, research_enabled=True,
                    research_emotion_threshold=5.0).save(config_path)

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(Memory.create_new(
            content="s", memory_type="conversation", domain="us",
            emotions={"love": 8.0},
        ))
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store, hebbian=hm, provider=FailingProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=tmp_path / "reflex_arcs.json",
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        with caplog.at_level(logging.WARNING, logger="brain.engines.heartbeat"):
            result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.research_fired is None
        assert result.research_gated_reason == "research_raised"
        assert any("research tick raised" in r.message for r in caplog.records)
    finally:
        store.close()
        hm.close()
```

Also at the top of `test_heartbeat.py` add this constant (mirror of `DEFAULT_REFLEX_ARCS_PATH`):

```python
DEFAULT_INTERESTS_PATH = _find_repo_root() / "brain" / "engines" / "default_interests.json"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v`
Expected: new tests fail.

- [ ] **Step 3: Extend HeartbeatConfig**

In `brain/engines/heartbeat.py`, replace `HeartbeatConfig` entirely with the extended version:

```python
@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration. Loaded from heartbeat_config.json."""

    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01
    gc_threshold: float = 0.01
    emit_memory: EmitMemoryMode = "conditional"
    reflex_enabled: bool = True
    reflex_max_fires_per_tick: int = 1
    research_enabled: bool = True
    research_days_since_human_min: float = 1.5
    research_emotion_threshold: float = 7.0
    research_cooldown_hours_per_interest: float = 24.0
    interest_bump_per_match: float = 0.1

    @classmethod
    def load(cls, path: Path) -> HeartbeatConfig:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()

        emit = data.get("emit_memory", "conditional")
        if emit not in _VALID_EMIT_MODES:
            emit = "conditional"

        try:
            return cls(
                dream_every_hours=float(data.get("dream_every_hours", 24.0)),
                decay_rate_per_tick=float(data.get("decay_rate_per_tick", 0.01)),
                gc_threshold=float(data.get("gc_threshold", 0.01)),
                emit_memory=emit,  # type: ignore[arg-type]
                reflex_enabled=bool(data.get("reflex_enabled", True)),
                reflex_max_fires_per_tick=int(data.get("reflex_max_fires_per_tick", 1)),
                research_enabled=bool(data.get("research_enabled", True)),
                research_days_since_human_min=float(data.get("research_days_since_human_min", 1.5)),
                research_emotion_threshold=float(data.get("research_emotion_threshold", 7.0)),
                research_cooldown_hours_per_interest=float(
                    data.get("research_cooldown_hours_per_interest", 24.0)
                ),
                interest_bump_per_match=float(data.get("interest_bump_per_match", 0.1)),
            )
        except (TypeError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        payload = {
            "dream_every_hours": self.dream_every_hours,
            "decay_rate_per_tick": self.decay_rate_per_tick,
            "gc_threshold": self.gc_threshold,
            "emit_memory": self.emit_memory,
            "reflex_enabled": self.reflex_enabled,
            "reflex_max_fires_per_tick": self.reflex_max_fires_per_tick,
            "research_enabled": self.research_enabled,
            "research_days_since_human_min": self.research_days_since_human_min,
            "research_emotion_threshold": self.research_emotion_threshold,
            "research_cooldown_hours_per_interest": self.research_cooldown_hours_per_interest,
            "interest_bump_per_match": self.interest_bump_per_match,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Extend HeartbeatResult**

Add three fields with defaults:

```python
@dataclass(frozen=True)
class HeartbeatResult:
    trigger: str
    elapsed_seconds: float
    memories_decayed: int
    edges_pruned: int
    dream_id: str | None
    dream_gated_reason: str | None
    research_deferred: bool
    heartbeat_memory_id: str | None
    initialized: bool
    reflex_fired: tuple[str, ...] = ()
    reflex_skipped_count: int = 0
    research_fired: str | None = None
    research_gated_reason: str | None = None
    interests_bumped: int = 0
```

- [ ] **Step 5: Extend HeartbeatEngine + wire run_tick**

Add four new fields to `HeartbeatEngine` (all with defaults, mirroring the reflex pattern after the cleanup PR):

```python
# After the existing reflex_default_arcs_path field, add:
searcher: WebSearcher = field(default_factory=lambda: NoopWebSearcher())
interests_path: Path | None = None
research_log_path: Path | None = None
default_interests_path: Path = field(
    default_factory=lambda: Path(__file__).parent / "default_interests.json"
)
```

Add imports near the top:

```python
from brain.search.base import NoopWebSearcher, WebSearcher
```

Add the two new helpers (alongside `_try_fire_reflex`):

```python
def _try_bump_interests(self, state: HeartbeatState, now: datetime,
                        config: HeartbeatConfig, dry_run: bool) -> int:
    """Scan conversation memories since last tick, bump interest pull_scores
    on keyword matches. Returns count of interests touched. Zero LLM calls.
    """
    if dry_run:
        return 0
    if self.interests_path is None:
        return 0

    from brain.engines._interests import InterestSet

    interests = InterestSet.load(
        self.interests_path, default_path=self.default_interests_path
    )
    if not interests.interests:
        return 0

    # Recent conversation memories since last tick
    all_convos = self.store.list_by_type("conversation", active_only=True, limit=50)
    recent = [m for m in all_convos if m.created_at >= state.last_tick_at]
    if not recent:
        return 0

    # For each recent memory, for each interest, check keyword match
    touched: set[str] = set()
    current = interests
    for mem in recent:
        content_lower = mem.content.lower()
        for interest in current.interests:
            if interest.topic in touched:
                continue
            for kw in interest.related_keywords:
                if kw.lower() in content_lower:
                    current = current.bump(
                        interest.topic,
                        amount=config.interest_bump_per_match,
                        now=now,
                    )
                    touched.add(interest.topic)
                    break

    if touched:
        current.save(self.interests_path)
    return len(touched)


def _try_fire_research(self, trigger: str, dry_run: bool,
                       config: HeartbeatConfig,
                       reflex_fired: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Run one research tick. Returns (fired_topic, gated_reason).

    Reflex-wins-tie: if reflex fired this tick, research is skipped with
    gated_reason='reflex_won_tie'.
    """
    if not config.research_enabled:
        return (None, None)
    if self.interests_path is None or self.research_log_path is None:
        return (None, None)
    if reflex_fired:
        return (None, "reflex_won_tie")

    from brain.engines.research import ResearchEngine

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
    try:
        result = engine.run_tick(trigger=trigger, dry_run=dry_run)
    except Exception as exc:
        logger.warning("research tick raised; isolating failure: %s", exc)
        return (None, "research_raised")

    if result.fired is not None:
        return (result.fired.topic, None)
    return (None, result.reason)
```

Modify `run_tick`. The current non-init branch (after state is loaded) has the sequence:
1. Emotion decay
2. Hebbian decay + GC
3. Reflex
4. Dream gate
5. Research (stub currently — `research_deferred = True`)
6. Optional heartbeat memory
7. State save + log

Replace the "Research stub" block (currently just `research_deferred = True`) and insert interest-bump + research eval in the right places:

```python
# After Hebbian decay + GC, before reflex:
interests_bumped = self._try_bump_interests(state, now, config, dry_run)

# After reflex + dream gate, replace the research stub with:
research_fired, research_gated_reason = self._try_fire_research(
    trigger, dry_run, config, reflex_fired
)
research_deferred = research_fired is None and research_gated_reason is None
```

Update the final `HeartbeatResult(...)` construction to include:

```python
research_fired=research_fired,
research_gated_reason=research_gated_reason,
interests_bumped=interests_bumped,
```

Update the `_append_log` call (non-init branch) to include:

```python
"research": {
    "fired": research_fired,
    "gated_reason": research_gated_reason,
},
"interests_bumped": interests_bumped,
```

- [ ] **Step 6: Update CLI heartbeat handler**

Modify `_heartbeat_handler` in `brain/cli.py` to pass the new fields:

```python
# Near the top of the handler, add:
from brain.search.factory import get_searcher as _get_searcher

# In the engine construction, add:
searcher=_get_searcher(getattr(args, "searcher", "ddgs")),
interests_path=persona_dir / "interests.json",
research_log_path=persona_dir / "research_log.json",
default_interests_path=_default_interests_path(),
```

Add `--searcher` to the heartbeat subparser in `_build_parser`:

```python
hb_sub.add_argument("--searcher", default="ddgs", choices=["ddgs", "noop", "claude-tool"])
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py tests/unit/brain/engines/test_research.py tests/unit/brain/engines/test_cli_research.py -v`
Expected: all pass.

Run full suite: `uv run pytest -q`
Expected: ~395-400 passed.

- [ ] **Step 8: Ruff + format**

Run: `uv run ruff check brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_heartbeat.py && uv run ruff format brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_heartbeat.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add brain/engines/heartbeat.py brain/cli.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "$(cat <<'EOF'
feat: integrate research into heartbeat tick

Heartbeat now evaluates research between dream gate and
heartbeat-memory emit. Interest ingestion hook runs right after
Hebbian decay: scans conversation memories since last tick,
keyword-matches against Interest.related_keywords, bumps pull_score
(default +0.1 per match) + feed_count. Zero LLM calls.

Reflex-wins-tie: if reflex fires this tick, research is skipped with
gated_reason='reflex_won_tie'. Research exceptions are fault-isolated
like reflex — tick continues with research_gated_reason='research_raised'.

HeartbeatConfig gains 5 new fields. HeartbeatResult gains
research_fired / research_gated_reason / interests_bumped. Audit log
records research + ingestion count per tick. CLI heartbeat handler
passes searcher + interest paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Nell's OG interest migration

**Purpose:** Extend the migrator to extract `nell_interests.json` from OG, classify scope via soul.json inspection, write to `{persona_dir}/interests.json`.

**Files:**
- Create: `brain/migrator/og_interests.py`
- Create: `tests/unit/brain/migrator/test_og_interests.py`
- Modify: `brain/migrator/cli.py`
- Modify: `brain/migrator/report.py`
- Modify: `tests/unit/brain/migrator/test_cli.py`

- [ ] **Step 1: Write failing tests for og_interests**

Create `tests/unit/brain/migrator/test_og_interests.py`:

```python
"""Tests for brain.migrator.og_interests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.migrator.og_interests import extract_interests_from_og


def _write_og_interests(path: Path, interests: list[dict]) -> None:
    path.write_text(
        json.dumps({"version": "1.0", "interests": interests}), encoding="utf-8"
    )


def _og_interest(**overrides) -> dict:
    base = {
        "id": "abc-123",
        "topic": "Lispector diagonal syntax",
        "pull_score": 7.2,
        "first_seen": "2026-03-29T16:42:33.435028+00:00",
        "last_fed": "2026-03-31T11:37:13.729750+00:00",
        "feed_count": 5,
        "source_types": ["dream", "heartbeat"],
        "related_keywords": ["lispector", "syntax", "language", "clarice"],
        "notes": "sideways through meaning",
    }
    base.update(overrides)
    return base


def test_extract_interests_simple(tmp_path: Path):
    path = tmp_path / "nell_interests.json"
    _write_og_interests(path, [_og_interest()])
    out = extract_interests_from_og(path, soul_names=set())
    assert len(out) == 1
    item = out[0]
    assert item["topic"] == "Lispector diagonal syntax"
    assert item["scope"] == "either"  # no soul match → default
    assert item["last_researched_at"] is None
    assert item["pull_score"] == 7.2


def test_extract_interests_scope_classification_from_soul(tmp_path: Path):
    path = tmp_path / "nell_interests.json"
    _write_og_interests(path, [
        _og_interest(id="a", topic="Lispector diagonal syntax"),
        _og_interest(id="b", topic="Hana"),
    ])
    out = extract_interests_from_og(path, soul_names={"hana"})
    scopes = {i["topic"]: i["scope"] for i in out}
    assert scopes["Hana"] == "internal"
    assert scopes["Lispector diagonal syntax"] == "either"


def test_extract_interests_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_interests_from_og(tmp_path / "missing.json", soul_names=set())


def test_extract_interests_corrupt_json_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_interests_from_og(path, soul_names=set())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/migrator/test_og_interests.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement og_interests.py**

Create `brain/migrator/og_interests.py`:

```python
"""Extract OG nell_interests.json into new-schema interest dicts.

Pure JSON read (no AST needed — OG interests live in a JSON file, not
a Python module). Scope classification: interests whose topic mentions
any name from the persona's soul.json are tagged "internal" (never
web-search Hana herself, Jordan, etc). Everything else defaults to
"either" — safe default, web-priority but memory-fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brain.utils.time import iso_utc, parse_iso_utc


def extract_interests_from_og(
    og_interests_path: Path, *, soul_names: set[str]
) -> list[dict[str, Any]]:
    """Return new-schema interest dicts extracted from OG nell_interests.json.

    Raises FileNotFoundError if the path doesn't exist, ValueError if it
    can't be parsed. soul_names is lowercased for case-insensitive match.
    """
    if not og_interests_path.exists():
        raise FileNotFoundError(f"OG interests file not found: {og_interests_path}")

    try:
        data = json.loads(og_interests_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {og_interests_path}: {exc}") from exc

    og_items = data.get("interests", [])
    if not isinstance(og_items, list):
        raise ValueError(f"{og_interests_path}: 'interests' is not a list")

    soul_lower = {s.lower() for s in soul_names}

    result: list[dict[str, Any]] = []
    for item in og_items:
        if not isinstance(item, dict):
            continue
        transformed = _transform_interest(item, soul_lower)
        if transformed is not None:
            result.append(transformed)
    return result


def extract_soul_names_best_effort(og_dir: Path) -> set[str]:
    """Read soul names from NellBrain/data/nell_soul.json. Best-effort; returns
    empty set if file missing/corrupt.
    """
    candidates = [
        og_dir / "nell_soul.json",
        og_dir / "data" / "nell_soul.json",
        og_dir.parent / "nell_soul.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return set()
            names: set[str] = set()
            for c in data.get("crystallizations", []):
                who = c.get("who_or_what", "")
                if isinstance(who, str) and who:
                    names.update(w.lower() for w in who.split() if len(w) > 2)
            return names
    return set()


def _transform_interest(og: dict[str, Any], soul_lower: set[str]) -> dict[str, Any] | None:
    required = (
        "id", "topic", "pull_score", "first_seen", "last_fed",
        "feed_count", "source_types", "related_keywords", "notes",
    )
    for key in required:
        if key not in og:
            return None

    topic = str(og["topic"])
    scope = _classify_scope(topic, soul_lower)

    # Normalise timestamps through parse → iso to collapse to Z format.
    first_seen = iso_utc(parse_iso_utc(og["first_seen"]))
    last_fed = iso_utc(parse_iso_utc(og["last_fed"]))

    return {
        "id": str(og["id"]),
        "topic": topic,
        "pull_score": float(og["pull_score"]),
        "scope": scope,
        "related_keywords": list(og["related_keywords"]),
        "notes": str(og["notes"]),
        "first_seen": first_seen,
        "last_fed": last_fed,
        "last_researched_at": None,
        "feed_count": int(og["feed_count"]),
        "source_types": list(og["source_types"]),
    }


def _classify_scope(topic: str, soul_lower: set[str]) -> str:
    topic_lower = topic.lower()
    for name in soul_lower:
        if name in topic_lower:
            return "internal"
    return "either"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/brain/migrator/test_og_interests.py -v`
Expected: 4 passed.

- [ ] **Step 5: Extend MigrationReport**

In `brain/migrator/report.py`, add to `MigrationReport`:

```python
@dataclass(frozen=True)
class MigrationReport:
    memories_migrated: int
    memories_skipped: list[SkippedMemory]
    edges_migrated: int
    edges_skipped: int
    elapsed_seconds: float
    source_manifest: list[FileManifest]
    next_steps_inspect_cmds: list[str]
    next_steps_install_cmd: str
    reflex_arcs_migrated: int = 0
    reflex_arcs_skipped_reason: str | None = None
    interests_migrated: int = 0
    interests_skipped_reason: str | None = None
```

In `format_report`, after the existing "Reflex arcs:" line add:

```python
lines.append(
    f"  Interests:      {report.interests_migrated:,} migrated"
    + (
        f" (skipped: {report.interests_skipped_reason})"
        if report.interests_skipped_reason
        else ""
    )
)
```

- [ ] **Step 6: Wire into migrator cli.py**

In `brain/migrator/cli.py`, at the top add:

```python
from brain.migrator.og_interests import extract_interests_from_og, extract_soul_names_best_effort
```

After the reflex-arcs migration block (which sets `reflex_arcs_migrated` and `reflex_arcs_skipped_reason`), insert:

```python
# ---- interests ----
_candidate_interests_paths = [
    args.input_dir / "nell_interests.json",
    args.input_dir / "data" / "nell_interests.json",
    args.input_dir.parent / "nell_interests.json",
]
og_interests_path = next(
    (p for p in _candidate_interests_paths if p.exists()), None
)
interests_target = work_dir / "interests.json"
interests_migrated = 0
interests_skipped_reason: str | None = None

if og_interests_path is not None:
    if interests_target.exists() and not args.force:
        interests_skipped_reason = "existing_file_not_overwritten"
    else:
        try:
            soul_names = extract_soul_names_best_effort(args.input_dir)
            og_interests = extract_interests_from_og(og_interests_path, soul_names=soul_names)
            interests_target.write_text(
                _json.dumps({"version": 1, "interests": og_interests}, indent=2) + "\n",
                encoding="utf-8",
            )
            interests_migrated = len(og_interests)
        except (ValueError, FileNotFoundError, OSError) as exc:
            interests_skipped_reason = f"migrate_error: {exc}"
else:
    interests_skipped_reason = "og_nell_interests_json_not_found"
```

Update the `MigrationReport(...)` constructor call to include the two new fields:

```python
report = MigrationReport(
    ...,
    reflex_arcs_migrated=reflex_arcs_migrated,
    reflex_arcs_skipped_reason=reflex_arcs_skipped_reason,
    interests_migrated=interests_migrated,
    interests_skipped_reason=interests_skipped_reason,
)
```

- [ ] **Step 7: Add migrator regression test**

Append to `tests/unit/brain/migrator/test_cli.py`:

```python
def test_migrate_writes_interests(tmp_path: Path, monkeypatch):
    """Regression: migrator writes interests.json from OG nell_interests.json."""
    import json
    from brain.migrator.cli import MigrateArgs, run_migrate

    # Reuse the existing helper that builds a minimal OG source dir (same
    # fixture pattern used by test_migrate_writes_reflex_arcs). Adjust
    # the helper name if it's different in this file.
    source = _make_minimal_og_source(tmp_path)  # <-- existing helper in this file

    # Drop nell_interests.json into the OG source (at the data/ level —
    # migrator candidate-path probe handles both locations)
    interests_data = {
        "version": "1.0",
        "interests": [{
            "id": "test-id",
            "topic": "Lispector diagonal syntax",
            "pull_score": 7.2,
            "first_seen": "2026-03-29T16:42:33.435028+00:00",
            "last_fed": "2026-03-31T11:37:13.729750+00:00",
            "feed_count": 5,
            "source_types": ["dream"],
            "related_keywords": ["lispector", "syntax"],
            "notes": "sideways through meaning",
        }],
    }
    (source / "nell_interests.json").write_text(
        json.dumps(interests_data), encoding="utf-8"
    )

    home = tmp_path / "home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    args = MigrateArgs(
        input_dir=source, output_dir=None,
        install_as="testpersona", force=False,
    )
    report = run_migrate(args)

    target = home / "personas" / "testpersona" / "interests.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["interests"]) == 1
    assert data["interests"][0]["topic"] == "Lispector diagonal syntax"
    assert data["interests"][0]["scope"] == "either"  # no soul match
    assert data["interests"][0]["last_researched_at"] is None
    assert report.interests_migrated == 1
    assert report.interests_skipped_reason is None
```

- [ ] **Step 8: Run all migrator tests**

Run: `uv run pytest tests/unit/brain/migrator/ -v`
Expected: all pass.

Full suite: `uv run pytest -q`
Expected: ~400 passed.

- [ ] **Step 9: Ruff + format**

Run: `uv run ruff check brain/migrator/ tests/unit/brain/migrator/ && uv run ruff format brain/migrator/ tests/unit/brain/migrator/`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add brain/migrator/og_interests.py brain/migrator/cli.py brain/migrator/report.py tests/unit/brain/migrator/
git commit -m "$(cat <<'EOF'
feat: migrate OG nell_interests.json into persona's interests.json

Pure JSON read (no AST needed) + scope auto-classification from
nell_soul.json names. Interest topics mentioning a soul-crystallization
subject (e.g. 'Hana', 'Jordan') get scope='internal' — those interests
stay memory-only, never hit the web. Everything else defaults to
scope='either' which tries web-first with memory fallback.

Refuse-to-clobber unless --force. MigrationReport gains interests_migrated
+ interests_skipped_reason. Candidate-path probe tries three locations
so --input NellBrain/ or NellBrain/data/ both work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Smoke test + final review + PR

**Purpose:** Validate end-to-end against Nell's migrated persona. Verify hard rules. Open PR.

**Files:** none created; verification only.

- [ ] **Step 1: Full suite + hard-rule check**

Run: `uv run pytest -q`
Expected: ~400 passed, 0 failures.

Run: `uv run ruff check --quiet && uv run ruff format --check --quiet`
Expected: clean.

Run: `rg -l 'import anthropic|from anthropic' brain/`
Expected: zero matches.

Run: `rg -l 'import anthropic|from anthropic' brain/search/`
Expected: zero matches.

- [ ] **Step 2: Re-migrate Nell's sandbox**

Run: `uv run nell migrate --input /Users/hanamori/NellBrain/data --install-as nell.sandbox --force`
Expected: migration report shows "Interests: 2 migrated" line.

- [ ] **Step 3: Verify interest migration**

Run: `uv run nell interest list --persona nell.sandbox`
Expected: two interests listed — "Lispector diagonal syntax" (scope=either) and "Hana" (scope=internal).

- [ ] **Step 4: Dry-run research against sandbox**

Run: `uv run nell research --persona nell.sandbox --provider fake --searcher noop --dry-run`
Expected: exits 0. Output indicates either "would fire: Lispector diagonal syntax" (if gate passes — Lispector has pull_score 7.2 and no last_researched_at) or a reason like "no_eligible_interest" or "not_due".

- [ ] **Step 5: Force-research an interest**

Run: `uv run nell research --persona nell.sandbox --provider fake --searcher noop --interest "Lispector diagonal syntax"`
Expected: research fires, memory id printed, "memory-only" (since noop searcher returns []).

- [ ] **Step 6: Verify research memory written**

Run:
```bash
uv run python -c "
from brain.paths import get_persona_dir
from brain.memory.store import MemoryStore
p = get_persona_dir('nell.sandbox')
s = MemoryStore(db_path=p / 'memories.db')
mems = s.list_by_type('research', limit=5)
print(f'research memories: {len(mems)}')
for m in mems:
    print(f'  - {m.content[:120]}')
s.close()
"
```
Expected: at least 1 research memory written.

- [ ] **Step 7: Heartbeat tick with research enabled**

Run: `uv run nell heartbeat --persona nell.sandbox --provider fake --searcher noop --trigger manual`
Expected: depending on current last_tick_at, either the first-tick-defer message OR a full tick with decay / reflex / dream-gate / research / optional memory output.

- [ ] **Step 8: Push branch + open PR**

```bash
git push -u origin week-4-research
gh pr create --title "Week 4.7 — Research Engine (Phase 1)" --body "$(cat <<'EOF'
## Summary

Fourth and final Week 4 cognitive engine — research, autonomous exploration of developed interests. Memory-sweep + optional web search (DuckDuckGo via ddgs, free & keyless, works with any LLM backend). First-person voice output written as memory_type='research'. Heartbeat-orchestrated between dream gate and heartbeat memory. Interest ingestion hook auto-bumps pull_scores on keyword matches in conversation. Reflex-wins-tie when both eligible.

- `brain/engines/research.py` — engine + ResearchResult/ResearchFire + ResearchLog
- `brain/engines/_interests.py` — Interest + InterestSet (shared with ingestion hook)
- `brain/engines/default_interests.json` — empty starter
- `brain/search/` — WebSearcher ABC + DdgsWebSearcher + NoopWebSearcher + stub ClaudeToolWebSearcher + factory
- `brain/migrator/og_interests.py` — Nell's 2 OG interests migrate with soul-aware scope classification ("Hana" → internal, "Lispector" → either)
- `brain/cli.py` — 4 new subcommands: `nell research`, `nell interest {list,add,bump}`
- Heartbeat extensions: 5 new HeartbeatConfig fields, 3 new HeartbeatResult fields, 4 new HeartbeatEngine fields, interest ingestion hook, research evaluation, reflex-wins-tie, fault isolation
- New dep: `ddgs>=6.0,<7.0`
- Phase 2 (auto-discovery of brand-new interests) explicitly deferred; see spec §13

**Spec:** `docs/superpowers/specs/2026-04-24-week-4-research-engine-design.md`
**Plan:** `docs/superpowers/plans/2026-04-24-week-4-research-engine.md`

## Test plan

- [x] Full suite ~400/~400 passing (was 376 pre-branch)
- [x] `rg 'import anthropic' brain/` → 0 matches
- [x] ruff clean
- [x] Nell's sandbox re-migrated: 2 OG interests migrate with correct scope classification
- [x] `nell interest list --persona nell.sandbox` shows migrated interests
- [x] `nell research --persona nell.sandbox --searcher noop --dry-run` exits 0
- [x] `nell heartbeat --persona nell.sandbox` runs with research integrated into the tick

## After merge

Week 4 is complete. Next session: tag `week-4` covering all four cognitive engines (dream + heartbeat + reflex + research).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Acceptance Criteria

Research Phase 1 ships when all of the following are true:

1. `uv run pytest -q` is green (~400 tests).
2. `rg 'import anthropic' brain/` returns zero matches.
3. `rg 'import anthropic' brain/search/` returns zero matches.
4. `uv run ruff check && uv run ruff format --check` both clean.
5. `brain/engines/research.py`, `brain/engines/_interests.py`, `brain/engines/default_interests.json`, `brain/search/*`, `brain/migrator/og_interests.py` all present.
6. `pyproject.toml` declares `ddgs>=6.0,<7.0`.
7. `uv run nell research --help` documents the subcommand; all 4 new subcommands work.
8. `uv run nell interest list --persona nell.sandbox` after re-migration shows 2 OG interests.
9. `uv run nell research --persona nell.sandbox --searcher noop --interest "Lispector diagonal syntax"` fires a research memory.
10. Heartbeat tick integrates research + interest ingestion + reflex-wins-tie.
11. CI green on macOS + Linux + Windows.

---

## Deferred — Phase 2 reminder

Phase 2 (automatic discovery of brand-new interests from conversation patterns) is explicitly deferred. See spec §13 and project memory file `project_companion_emergence_tech_debt.md`. Revisit when Phase 1 has run ≥2 weeks against Nell's data.

After this merges → **Week 4 tag** covering dream + heartbeat + reflex + research. That's a full cognitive substrate.
