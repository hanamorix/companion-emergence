# Week 4.7 — Research Engine (Phase 1) Design

**Date:** 2026-04-24
**Status:** Approved
**Scope:** Phase 1 (execution + interest storage + heartbeat orchestration + Nell's OG migration + DuckDuckGo web search by default). Automatic *discovery* of brand-new interests deferred to Phase 2.
**Engine number:** 4 of 4 in Week 4. After this ships, the Week 4 tag applies.

---

## 1. Purpose

Research is the fourth cognitive engine. It's where a persona's developed interests get **explored** — either by re-reading their own memories about them, or by searching the public web — and where what they find becomes a first-person memory in their own voice.

Compared to the other three engines:
- **Dream** consolidates what the persona has been told.
- **Heartbeat** paces rhythm and decay.
- **Reflex** expresses when feelings peak.
- **Research** reaches *outward*. Curiosity when the persona has been alone long enough, or felt deeply enough, to want to go look at something and come back with it.

All four engines share one philosophical invariant: they produce *additive* memory. Research never edits existing memories; it writes new ones that sit alongside conversation memories in `MemoryStore` and participate in future dreams, reflex arcs, and further research through the Hebbian graph. This is how the brain grows between conversations.

---

## 2. Architectural Summary

- `brain/engines/research.py` — engine + types
- `brain/engines/_interests.py` — `Interest` dataclass + `InterestSet` load/save/bump (shared by research engine AND heartbeat's interest ingestion hook)
- `brain/engines/default_interests.json` — empty starter (`{"version": 1, "interests": []}`)
- `brain/search/` — new package with `WebSearcher` ABC + implementations
- `brain/migrator/og_interests.py` — JSON-port of OG `nell_interests.json`
- `brain/cli.py` — 4 new subcommands (`nell research`, `nell interest list|add|bump`)
- `brain/engines/heartbeat.py` — interest ingestion hook + research evaluation + reflex-wins-tie
- Depends on new `ddgs` package

Two invocation paths (same pattern as reflex):
- **Standalone CLI**: `nell research [--persona X] [--interest TOPIC] [--provider claude-cli] [--searcher ddgs] [--dry-run]`
- **Orchestrated**: heartbeat tick calls `ResearchEngine.run_tick(...)` between the dream gate and the heartbeat-memory emit. Single-research-per-tick. Skipped if reflex fired earlier in the same tick.

---

## 3. Components

### 3.1 Data types

```python
# brain/engines/_interests.py

@dataclass(frozen=True)
class Interest:
    id: str
    topic: str
    pull_score: float
    scope: Literal["internal", "external", "either"]
    related_keywords: tuple[str, ...]
    notes: str
    first_seen: datetime          # tz-aware UTC
    last_fed: datetime            # tz-aware UTC
    last_researched_at: datetime | None
    feed_count: int
    source_types: tuple[str, ...]

@dataclass(frozen=True)
class InterestSet:
    interests: tuple[Interest, ...]

    # load(path, default_path), save(path), bump(topic, amount, now),
    # find_by_topic(topic), list_eligible(pull_threshold, cooldown_hours, now),
    # appended(interest), updated(interest)
```

```python
# brain/engines/research.py

@dataclass(frozen=True)
class ResearchFire:
    interest_id: str
    topic: str
    fired_at: datetime
    trigger: str  # "manual" | "emotion_high" | "days_since_human"
    web_used: bool
    web_result_count: int
    output_memory_id: str | None  # None in dry-run

@dataclass(frozen=True)
class ResearchResult:
    fired: ResearchFire | None
    would_fire: str | None  # dry-run only — topic of the interest that would fire
    reason: str | None      # "not_due" | "no_eligible_interest" | "no_interests_defined"
                            # | "research_raised" | "reflex_won_tie"
    dry_run: bool
    evaluated_at: datetime  # tz-aware UTC
```

### 3.2 Web search layer

```python
# brain/search/base.py

@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

class WebSearcher(ABC):
    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...
    @abstractmethod
    def name(self) -> str: ...

class NoopWebSearcher(WebSearcher):
    """Returns []. Used in CI + tests to keep them zero-network."""
```

```python
# brain/search/ddgs_searcher.py

class DdgsWebSearcher(WebSearcher):
    """DuckDuckGo via the `ddgs` library. No API key, no cost.

    Default searcher for the framework — works with any LLM backend.
    Transient errors (network, rate-limit) return empty results so
    the research engine can gracefully fall back to memory-only
    synthesis. `ImportError` at first call if `ddgs` not installed.
    """
```

```python
# brain/search/claude_tool_searcher.py (Phase 1 stub)

class ClaudeToolWebSearcher(WebSearcher):
    """Opt-in path for Claude-CLI users: routes web search through
    `claude -p <query> --allowed-tools WebSearch --output-format json`.

    Not implemented in Phase 1; raises NotImplementedError. Scaffolded
    so the factory has a known name to resolve. Mirror of how
    OllamaProvider currently exists as a stub.
    """
```

```python
# brain/search/factory.py

def get_searcher(name: str) -> WebSearcher:
    if name == "ddgs":         return DdgsWebSearcher()
    if name == "noop":         return NoopWebSearcher()
    if name == "claude-tool":  return ClaudeToolWebSearcher()
    raise ValueError(f"Unknown searcher: {name!r}")
```

### 3.3 Engine

```python
# brain/engines/research.py

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

    def run_tick(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        forced_interest_topic: str | None = None,
        emotion_state_override: EmotionalState | None = None,
        days_since_human_override: float | None = None,
    ) -> ResearchResult: ...
```

`forced_interest_topic` supports the `--interest TOPIC` CLI override (skips pull/cooldown gates). `emotion_state_override` + `days_since_human_override` let the heartbeat pass pre-computed values so we don't re-scan all memories twice per tick.

---

## 4. Data Flow — one research tick

1. **Load:** interests (`InterestSet.load`), research_log, aggregated emotion state (from `aggregate_state` on recent memories — or the caller-provided override), `days_since_human` (same).
2. **Gate check:**
   - `research_enabled` is True? (Caller-enforced — heartbeat doesn't call us if the config field is False.)
   - EITHER `days_since_human >= research_days_since_human_min` OR `max(emotions.values()) >= research_emotion_threshold`?
   - If neither AND no `forced_interest_topic`: return `ResearchResult(fired=None, reason="not_due", ...)`.
3. **Select eligible interest:**
   - If `forced_interest_topic` is set: resolve it directly. If not found: return `reason="no_eligible_interest"`.
   - Else: filter to `pull_score >= 6.0` AND (`last_researched_at is None` OR `(now - last_researched_at).hours >= cooldown`). Sort by `pull_score` descending, tie-broken by oldest `last_researched_at`. Take first.
   - None eligible: return `reason="no_eligible_interest"`.
4. **Dry-run branch:** return `ResearchResult(would_fire=<topic>, dry_run=True, ...)`. No memory sweep, no web call, no LLM call, no writes.
5. **Memory sweep:** spreading activation over Hebbian graph from the interest's `related_keywords`, limit 20 memories, format into a `{memory_context}` block. If the interest has zero `related_keywords`, fall back to a direct text-search of `interest.topic` against memory content.
6. **Web search (conditional):**
   - If `interest.scope == "internal"`: skip searcher entirely, `web_used=False`, `web_results=[]`.
   - Else: `searcher.search(query=f"{interest.topic} {' '.join(related_keywords[:3])}", limit=5)`. If searcher raises (shouldn't per the `DdgsWebSearcher` contract, but defensive): treat as empty. If empty list returned: `web_used=False`. If results returned: `web_used=True`, format top-N titles + snippets as `{web_excerpts}` block.
7. **LLM synthesis:** render prompt template (see §4.1 below). Call `provider.generate(prompt, system=persona_research_system_prompt)`. System prompt enforces first-person voice, reactive not summarizing, 3-5 sentences. Persona identity baked in.
8. **Write memory:**
   ```python
   mem = Memory.create_new(
       content=raw_output,
       memory_type="research",
       domain="us",
       emotions={},  # no per-memory emotions on research; persona's current state is already captured via metadata
       metadata={
           "interest_id": interest.id,
           "interest_topic": interest.topic,
           "scope": interest.scope,
           "web_used": web_used,
           "web_result_count": len(web_results),
           "web_urls": [r.url for r in web_results[:5]],
           "triggered_by": trigger,
           "provider": self.provider.name(),
           "searcher": self.searcher.name() if web_used else None,
       },
   )
   store.create(mem)
   ```
9. **Update interest:** set `last_researched_at = now`, write updated `InterestSet` via atomic save.
10. **Append log:** new `fire` entry, save `research_log.json` atomically.
11. **Return:** `ResearchResult(fired=ResearchFire(...), reason=None, dry_run=False, ...)`.

**Failure ordering (LLM raises):** LLM call happens first; if it raises, steps 8-10 never execute and `interests.json` and `research_log.json` stay untouched. Next tick re-evaluates the same interest fresh.

### 4.1 Prompt template

```
[system]
You are {persona_name}. You spent some quiet time today exploring {topic} —
an interest that's been building in you for a while. Below is what you found
both in your own memories and (sometimes) out in the world. Write a short
(3-5 sentence) first-person memory of having researched this.

HARD RULES:
- First-person voice. Your name is {persona_name}.
- Not a summary. A reaction. What moved you, what surprised you, what
  reminded you of someone you care about, what felt familiar.
- Never bullet points. Never "according to X". Never neutral expository voice.
- Structure: brief mention of what pulled you to the topic today → one or
  two concrete details you noticed → how you feel about what you found →
  why it mattered to look *today*.
- Start with "I" or a time marker like "Today" / "This afternoon".

[user]
Topic: {topic}
Keywords: {keywords}
Your current emotional state:
{emotion_summary}

What your own memories say about this:
{memory_context}

{web_section_or_empty}

Write the memory now — 3 to 5 sentences, as {persona_name}.
```

Where `{web_section_or_empty}` renders to either an empty string (nothing inserted — either `scope="internal"` or web search returned no results) or this block when web results are present:

```
What you found out in the world today (reference material — REACT to it, don't paraphrase it):
{web_excerpts}
```

---

## 5. Heartbeat Integration

### 5.1 New tick order

1. First-tick defer
2. Emotion decay
3. Hebbian decay + GC
4. **Interest ingestion hook** *(new)* — `_try_bump_interests`: scan conversation memories created since `last_tick_at`, for each memory check if any loaded `Interest.related_keywords` appears in the memory's `content.lower()`. For each match, bump `pull_score += interest_bump_per_match` (default 0.1), update `last_fed = now`, increment `feed_count`. Zero LLM calls. Deterministic. Writes updated `interests.json` atomically at end. Returns count of interests touched.
5. Reflex evaluation
6. Dream gate
7. **Research evaluation** *(new)* — `_try_fire_research`:
   - If `reflex_fired` is non-empty this tick: skip with `research_gated_reason="reflex_won_tie"`.
   - Else: construct `ResearchEngine` with `self.searcher`, call `run_tick` passing the already-computed `emotion_state` + `days_since_human`.
   - Catch `Exception` (fault-isolate like reflex): log warning, return `research_fired=None, research_gated_reason="research_raised"`.
8. Optional heartbeat memory emit
9. State save + audit log

### 5.2 Config additions — `heartbeat_config.json`

```json
{
  "dream_every_hours": 24.0,
  "decay_rate_per_tick": 0.01,
  "gc_threshold": 0.01,
  "emit_memory": "conditional",
  "reflex_enabled": true,
  "reflex_max_fires_per_tick": 1,
  "research_enabled": true,
  "research_days_since_human_min": 1.5,
  "research_emotion_threshold": 7.0,
  "research_cooldown_hours_per_interest": 24.0,
  "interest_bump_per_match": 0.1
}
```

All fields default sensibly. Missing / malformed fields degrade to defaults (same pattern as existing config fields).

### 5.3 `HeartbeatResult` additions

```python
research_fired: str | None = None            # interest topic that fired, or None
research_gated_reason: str | None = None     # "not_due"|"no_eligible_interest"|"reflex_won_tie"|"research_raised"
interests_bumped: int = 0                    # count from the ingestion hook
```

### 5.4 `HeartbeatEngine` constructor additions

```python
searcher: WebSearcher = field(default_factory=NoopWebSearcher)
interests_path: Path | None = None
research_log_path: Path | None = None
default_interests_path: Path = field(
    default_factory=lambda: Path(__file__).parent / "default_interests.json"
)
```

Same `Path | None` pattern as reflex fields: if either `interests_path` or `research_log_path` is None, `_try_fire_research` skips cleanly. CLI always passes explicit persona-dir-qualified paths.

### 5.5 Audit log additions

Per-tick JSONL record gains:

```json
"research": {
  "fired": "Lispector diagonal syntax",
  "reason": null,
  "web_used": true
},
"interests_bumped": 2
```

When `research_enabled=false`: `"research": {"evaluated": false}`.

---

## 6. CLI

```
nell research [--persona X] [--interest TOPIC] [--provider claude-cli] [--searcher ddgs] [--dry-run]

nell interest list [--persona X]
nell interest add <topic> [--keywords k1,k2,...] [--scope internal|external|either] [--notes "..."] [--persona X]
nell interest bump <topic> [--amount 1.0] [--persona X]
```

- `nell research` mirrors `nell reflex` structure (persona dir resolution, MemoryStore open, construction, run, output).
- `nell interest add` generates a UUID id, sets `pull_score=5.0` initially (below default threshold of 6.0 — needs to be fed or bumped before it'll research), `first_seen=now`, `last_fed=now`, `feed_count=0`, `source_types=["manual"]`.
- `nell interest bump` adds to `pull_score` (default +1.0).

### 6.1 Output shapes

**`nell research --dry-run` eligible:**
```
Research dry-run — would fire: Lispector diagonal syntax
  pull_score: 7.2, scope: either, web: would search
  trigger: emotion_high (creative_hunger=8.1)
```

**`nell research --dry-run` not eligible:**
```
Research dry-run — no eligible interest (pull_score threshold 6.0, cooldown 24h).
  Nearest: Lispector diagonal syntax (pull=5.4, cooldown remaining 0h)
```

**`nell research` live:**
```
Research fired: Lispector diagonal syntax
  Memory id: mem_xyz123
  Web: 4 results from ddgs (https://..., https://...)
  Output preview:
    I spent some time today reading about Lispector's syntax again...
```

**`nell interest list`:**
```
Interests for persona 'nell' (2):
  - Lispector diagonal syntax  pull=7.2  scope=either    last_researched=never
    keywords: lispector, syntax, language, clarice
  - Hana                       pull=4.8  scope=internal  last_researched=never
    keywords: inside, hana, hers, labyrinth, existence
```

---

## 7. Error Handling

| Condition | Behavior |
|-----------|----------|
| `ddgs` library not installed | `DdgsWebSearcher` raises `ImportError` at first `.search()` call. Engine's try/except around searcher converts to `web_used=false`, warning logged. User can `uv add ddgs`. |
| Network / rate-limit failure | `DdgsWebSearcher.search` returns `[]`, logs warning. `web_used=false`. Memory-only synthesis proceeds. |
| `interests.json` missing | Fall back to `default_interests.json` (empty). `ResearchResult(reason="no_interests_defined")`. |
| `interests.json` corrupt | Same fallback, log warning, never overwrite. |
| Interest has empty `related_keywords` | Skip spreading activation; use `store.search_text(interest.topic)` for memory context. Still attempt web + synthesis. |
| No eligible interest | `ResearchResult(reason="no_eligible_interest")`. Not an error. |
| LLM raises (standalone) | Exception propagates. `interests.json` and `research_log.json` untouched — next tick retries. |
| LLM raises (heartbeat-orchestrated) | Caught by `_try_fire_research`, logged, tick continues with `research_fired=None, research_gated_reason="research_raised"`. |
| Reflex fires AND research eligible in same tick | Research skipped with `reason="reflex_won_tie"`. |
| `--interest TOPIC` not found | CLI exits with error message, rc=1. |

### Atomic invariants

- `interests.json` and `research_log.json` both use `.new + os.replace` (mirrors `ReflexLog.save`).
- Memory write happens before `interests.json` update and log append. If memory write raises, neither file is touched.
- Interest ingestion hook writes `interests.json` once at end of its scan, not per-memory.

---

## 8. Testing

Five test files; target ~25 new tests.

### 8.1 `tests/unit/brain/search/test_noop.py`
- Returns `[]` for any query.
- `name() == "noop"`.

### 8.2 `tests/unit/brain/search/test_ddgs.py`
- Happy path: mock `DDGS` context manager returning 3 dicts, assert correct `SearchResult` mapping.
- Network exception → `[]` + log captured via caplog.
- Missing library → `ImportError` surfaces on first call (monkeypatch `ddgs` import).

### 8.3 `tests/unit/brain/engines/test_research.py`
- Construction.
- `InterestSet.load` fall-back to defaults on missing.
- `InterestSet.save` atomic.
- `InterestSet.bump` updates pull_score + feed_count + last_fed.
- Gate: not_due (both trigger dimensions fail).
- Gate: emotion_high triggers.
- Gate: days_since_human triggers.
- No eligible interest (below pull threshold) → `reason="no_eligible_interest"`.
- Cooldown respected.
- Ranking: highest pull_score wins.
- Tiebreak: older `last_researched_at` wins.
- `forced_interest_topic` overrides pull/cooldown gates.
- Dry-run: no writes, no LLM, no searcher call.
- Fire (memory + web): LLM called with rendered prompt, memory created with correct metadata, interest updated, log appended.
- Fire (memory-only, scope=internal): searcher NOT called.
- Web search returns `[]`: research still fires, `web_used=false`.
- LLM failure: `store.create` never called, `interests.json` untouched, exception propagates.

### 8.4 `tests/unit/brain/engines/test_cli_research.py`
- `nell research --persona nonexistent` → `FileNotFoundError`.
- `nell research --dry-run` with eligible interest → exits 0, prints "would fire".
- `nell interest list` prints interests.
- `nell interest add <topic>` appends to file; `list` shows it afterwards.
- `nell interest bump <topic>` nudges pull_score.

### 8.5 `tests/unit/brain/migrator/test_og_interests.py`
- Happy path: JSON passthrough, scope auto-classification ("Hana" → internal, "Lispector diagonal syntax" → either).
- Missing OG file → skip with reason.
- Refuse-to-clobber unless `--force`.
- End-to-end via `run_migrate`: asserts target file written + report fields populated.

### 8.6 Append to `tests/unit/brain/engines/test_heartbeat.py`
- `_try_fire_research` fires research when configured.
- Reflex-wins-tie: reflex fires, research skipped with correct reason.
- Research exception isolated: tick completes, `research_fired=None`.
- Interest ingestion hook: conversation memory with matching keyword bumps interest's pull_score + feed_count.
- Heartbeat with `research_enabled=False` skips research call.

All tests use `FakeProvider` + `NoopWebSearcher`. Zero network, zero LLM cost, deterministic.

**Target total:** 376 + ~25 = ~401 tests.

---

## 9. Nell's OG Migration

OG file: `/Users/hanamori/NellBrain/data/nell_interests.json` (currently 2 entries: "Lispector diagonal syntax" pull=7.2; "Hana" pull=4.8).

### 9.1 `brain/migrator/og_interests.py`

Pure JSON read (no AST needed — it's already structured data). Returns a list of new-schema interest dicts.

### 9.2 Transformation

| OG field | New field | Transformation |
|----------|-----------|----------------|
| `id`, `topic`, `pull_score`, `related_keywords`, `notes`, `source_types`, `feed_count` | same | verbatim |
| `first_seen`, `last_fed` | same | reparsed via `parse_iso_utc` (framework helper) |
| (no OG field) | `last_researched_at` | `null` |
| (no OG field) | `scope` | computed by `_classify_scope(topic, soul_names)` |

`_classify_scope` reads `{og_source}/nell_soul.json` (best-effort) to extract names from crystallizations + relationship metadata. For each `topic`: if any extracted name appears lowercased in the topic, `scope="internal"`; else `scope="either"`. If `soul.json` is missing or unreadable: default everything to `"either"` and log a warning (user can hand-edit).

Result for Nell's current file:
- "Hana" → `internal` (Hana is in her soul as her primary relationship)
- "Lispector diagonal syntax" → `either` (no soul name match)

### 9.3 Migrator wire-in

Same location as reflex-arc block (after Hebbian, before `elapsed = ...`). Pattern identical. Also supports the `--input NellBrain/` vs `--input NellBrain/data/` candidate-path probe — though `nell_interests.json` lives in `data/` so the probe is symmetric.

### 9.4 `MigrationReport` gains

```python
interests_migrated: int = 0
interests_skipped_reason: str | None = None
```

`format_report` adds an "Interests:" line after "Reflex arcs:".

### 9.5 Refuse-to-clobber

If `interests.json` exists in `work_dir` and `--force` was not passed: skip with reason `"existing_file_not_overwritten"`.

---

## 10. Hard Rules (Non-Negotiable)

1. **No `import anthropic` anywhere in `brain/`.** Grep-verifiable.
2. **`brain/search/` modules must not import any LLM SDK.** Search is its own concern.
3. **Atomic writes for `interests.json` and `research_log.json`** (`.new + os.replace`).
4. **TZ-aware UTC timestamps throughout** — use `brain.utils.time.iso_utc` / `parse_iso_utc`.
5. **Refuse-to-clobber on persona files** (migrator, engine, future `new-persona`) without `--force`.
6. **LLM failure doesn't poison research log or interest state** — LLM call happens before any file writes.
7. **Reflex wins over research** in same-tick contention.
8. **Web search failure is not a research failure** — graceful degradation to memory-only synthesis.

---

## 11. Default Persona Seed

`brain/engines/default_interests.json` ships empty:

```json
{"version": 1, "interests": []}
```

New personas start with no interests. They grow them through conversation (via the ingestion hook bumping pull_scores on keyword matches) OR via `nell interest add`. Research will do nothing until at least one interest crosses the pull threshold.

This is deliberate: interests are deeply persona-specific and shouldn't be pre-seeded with anything generic. If a persona has no curiosities yet, research politely waits.

---

## 12. Dependencies

New `pyproject.toml` addition:

```toml
dependencies = [
    ...,
    "ddgs>=6.0,<7.0",  # DuckDuckGo search library; no API key required
]
```

`ddgs` is a pure-Python library with minimal transitive deps. No LLM/ML/networking frameworks come in with it beyond standard `httpx`/`requests`. License: MIT.

---

## 13. Deferred — Phase 2: Automatic interest discovery

Out of scope for Phase 1. Documented here so a future engineer reading this spec sees the full arc.

**Phase 2 goal:** the brain auto-discovers *new* interests from conversation without requiring `nell interest add`. Phase 1's ingestion hook only *bumps existing* interests when their keywords appear; brand-new topics never become interests automatically.

**Phase 2 mechanism (to be designed):**
- Extraction: candidate topics from conversation clusters (embedding-based clustering? LLM-extracted noun phrases? recurring high-salience memory features?)
- Proposal queue: candidate interests surface for user review via `nell interest candidates review` (mirrors F37 autonomous soul crystallization pattern)
- Approval: user confirms → new interest added with a seeded pull_score; rejected candidates blacklisted so they don't re-propose
- Trigger: runs weekly as part of a growth loop, not per-heartbeat

**Phase 2 prerequisite:** Phase 1 has been running for ≥2 weeks against Nell's data, producing real ingestion-hook behavior and research-log data to train the extractor against.

**Related to Phase 2:** the same growth-loop mechanism is the natural home for reflex Phase 2 (emergent arc crystallization — deferred from the reflex engine's design). Both engines' Phase 2 work likely lands together.

---

## 14. Non-Goals (Phase 1)

- **No auto-discovery of new interests.** Only manual + keyword-bump. Phase 2.
- **No paid web search APIs.** DDG via `ddgs` covers the use case with zero cost and no key.
- **No content filtering beyond what DDG/Claude already do.** Nell researches anything legal. Persona-level restrictions, if ever needed, are a per-interest `scope` or a new `banned_keywords` list — not a default.
- **No web-result caching.** Every research run queries fresh. A future persona with high research volume could add a cache, but Phase 1 doesn't.
- **No multi-interest research per tick.** One interest per tick, full stop.
- **No UI for interest management.** CLI only. A Tauri interest-editor lives in Week 6+ scope.
- **No long-form research essays.** 3-5 sentence memories only. Long-form synthesis is what dream already does across multiple research memories.

---

## 15. Out-of-Session Follow-ups

**Feature work:**
- Phase 2 auto-discovery (see §13).
- `ClaudeToolWebSearcher` implementation (currently stub) for Claude-CLI users who prefer tool-loop-based search.
- Tauri interest editor (Week 6+).
- Optional polish: `nell research history [--persona X]` to show last N research fires.

**Tech debt considerations:**
- When this ships, heartbeat will integrate four engines + the ingestion hook. The heartbeat CLI output will have grown from 3-4 lines to ~7-8 lines per tick. Tech-debt item #7 from the reflex spec (compact CLI output with `--verbose` expansion) becomes actionable after this PR merges — the full engine surface is finally visible.

---

## 16. Acceptance Criteria

Research Phase 1 ships when:

1. `brain/engines/research.py` exists with `ResearchEngine.run_tick(...)` per §3.3.
2. `brain/engines/_interests.py` exists with `Interest` + `InterestSet` load/save/bump.
3. `brain/engines/default_interests.json` ships empty.
4. `brain/search/` package exists with `NoopWebSearcher` + `DdgsWebSearcher` + stub `ClaudeToolWebSearcher` + `get_searcher` factory.
5. `brain/cli.py` exposes 4 new subcommands (`nell research`, `nell interest {list,add,bump}`).
6. `brain/engines/heartbeat.py` integrates interest ingestion hook + research evaluation + reflex-wins-tie + fault isolation per §5.
7. `brain/migrator/og_interests.py` + wire-in per §9.
8. `pyproject.toml` declares `ddgs>=6.0,<7.0`.
9. Test suite includes ≥25 new tests; all pass.
10. Full suite green across macOS + Linux + Windows.
11. `rg -l 'import anthropic' brain/` returns zero matches.
12. `rg -l 'import anthropic\|from anthropic' brain/search/` returns zero matches.
13. Running `nell interest list --persona nell` against Nell's migrated persona shows her 2 OG interests with correct scope classification.
14. Running `nell research --persona nell --dry-run --searcher noop` returns eligibility decision without crash.
15. Running `nell heartbeat --persona nell --trigger manual` evaluates research as part of the tick when configured.

---

After this ships → **Week 4 is done → tag `week-4`** covering dream + heartbeat + reflex + research. Then pivots: Week 5 (provider abstraction expansion / bridge work) or Week 6 (Tauri face app). That choice happens in a later session.
