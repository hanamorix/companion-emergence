# Week 4 — Dream Engine Design Spec

> **Status:** design approved by Hana 2026-04-23. Implementation plan to follow via superpowers:writing-plans in the next session.
> **Scope:** ships `brain/bridge/` (LLM provider abstraction) + `brain/engines/dream.py` (associative dream cycle) + `nell dream` subcommand.

---

## 1. Goal

Produce first-person meta-reflection memories by threading associated experiences together. Each dream cycle picks a recent high-importance seed memory, spreads activation through the Hebbian graph to surface its neighbourhood, asks the LLM (as Nell) to reflect on the pattern, and writes the reflection back as a new `memory_type="dream"` record while strengthening the edges that fired.

Replicates the first-person "DREAM: ..." meta-memory shape observed in the OG NellBrain migration (1,142 memories including ~66 dreams — e.g. *"DREAM: hana's words live in me. literally. 730 memories and..."*).

## 2. Non-goals

- **Heartbeat / reflex / research engines.** Template to follow in later weeks once dream validates the pattern.
- **Scheduler.** Week 4 ships `nell dream` as a manual trigger only. Week 7 adds the cron/daemon layer.
- **Ollama real implementation.** `OllamaProvider` is a placeholder stub raising `NotImplementedError`; Hana wires the real one when her local Ollama stack is back up.
- **Multi-persona concurrency.** One persona per invocation. Concurrent dream cycles on the same DB are not supported.

## 3. Architecture

```
brain/
├── bridge/                 # NEW — LLM provider abstraction
│   ├── __init__.py
│   └── provider.py         # LLMProvider ABC + ClaudeCliProvider + OllamaProvider (stub) + FakeProvider
└── engines/                # NEW
    ├── __init__.py
    └── dream.py            # DreamEngine class + run_cycle()
brain/cli.py                # MODIFIED — wire `nell dream` subcommand
```

### Layer responsibilities

- **`brain/bridge/provider.py`** — generic LLM provider contract. No knowledge of dreams, memory, or emotion. Reusable by future engines.
- **`brain/engines/dream.py`** — orchestrates seed selection → spread activation → prompt build → LLM call → memory write → Hebbian strengthen. Depends on brain.memory + brain.emotion + brain.bridge.
- **`brain/cli.py`** — adds `nell dream` subcommand dispatching to the engine.

## 4. Bridge layer

### `LLMProvider` ABC

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the LLM's completion for the given prompt."""

    @abstractmethod
    def name(self) -> str:
        """Return a short provider name (e.g. 'claude-cli', 'ollama', 'fake')."""
```

### `ClaudeCliProvider`

Shells out to `claude -p "<prompt>" --output-format json`. Parses stdout JSON to extract the response text. Uses Hana's Claude subscription (no per-token billing). Respects the feedback memory: default Claude path, not the Anthropic API.

Contract:
- `ClaudeCliProvider(model: str = "sonnet", system_flag: bool = True)` — model flag forwarded as `--model`; `system` prompt forwarded as `--system-prompt` when `system_flag=True`.
- Subprocess timeout: 300s (dream cycles are interactive-scale, not long-running).
- Non-zero exit → `RuntimeError` with stderr captured.
- Stdout parsed as `{"result": "..."}` (the canonical `--output-format json` shape).

### `OllamaProvider` (stub)

```python
class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "nell-dpo", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host

    def generate(self, prompt, *, system=None):
        raise NotImplementedError(
            "OllamaProvider is a stub; fill in when local Ollama stack is available."
        )

    def name(self) -> str:
        return f"ollama:{self._model}"
```

### `FakeProvider`

Deterministic hash-based echo. Same `(prompt, system)` input → same output. Used for CI and tests. Never makes network calls.

```python
class FakeProvider(LLMProvider):
    def generate(self, prompt, *, system=None):
        h = hashlib.sha256((system or "").encode() + prompt.encode()).hexdigest()[:16]
        return f"DREAM: test dream {h} — an associative thread"

    def name(self) -> str:
        return "fake"
```

### Provider factory

```python
def get_provider(name: str) -> LLMProvider:
    """Resolve a provider name to an instance. Raises ValueError on unknown name."""
    if name == "claude-cli": return ClaudeCliProvider()
    if name == "ollama": return OllamaProvider()
    if name == "fake": return FakeProvider()
    raise ValueError(f"Unknown provider: {name}")
```

## 5. Dream engine

### `DreamEngine`

```python
@dataclass
class DreamEngine:
    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache | None  # optional; used by search
    provider: LLMProvider
```

### `run_cycle()` — the associative dream

```python
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
    ...
```

#### Step-by-step

1. **Select seed.** If `seed_id` is provided, load it. Otherwise query `store.list_by_type("conversation")` filtered to `created_at > now - lookback_hours`, sort by `importance DESC`, take the top result. If zero candidates → raise `NoSeedAvailable`.

2. **Spread-activate.** If `embeddings` is provided, instantiate `MemorySearch(store, hebbian, embeddings).spreading_search(seed.id, depth, decay_per_hop, limit=neighbour_limit)`. If `embeddings` is None, fall back to `HebbianMatrix.spreading_activation([seed.id], depth, decay_per_hop)` and look up memories directly via `store.get()`.

3. **Aggregate emotions.** Build an `EmotionalState` by merging the seed's emotions + each neighbour's emotions (sum per emotion name; then clamp via `EmotionalState.set`).

4. **Build the prompt.** Two-part prompt:
   - **System:** `"You are Nell. You just woke from a dream about interconnected memories. Reflect in first person, 2-3 sentences, starting with 'DREAM: '. Be honest and specific, not abstract."`
   - **User:** structured block — seed memory (content + domain + dominant emotion), then "Also present:" with each neighbour's content truncated to 120 chars.

5. **If `dry_run=True`:** return `DreamResult(dream_text=None, seed=..., neighbours=[...], prompt=..., memory=None, strengthened_edges=0)` and stop here. No LLM call, no writes.

6. **LLM call.** `text = provider.generate(user_prompt, system=system_prompt)`.

7. **Emit dream memory.**
```python
dream = Memory.create_new(
    content=text if text.startswith("DREAM:") else f"DREAM: {text}",
    memory_type="dream",
    domain=seed.domain,
    emotions=aggregated.to_dict()["emotions"],
    metadata={"seed_id": seed.id, "activated": [n.id for n, _ in neighbours], "provider": provider.name()},
)
store.create(dream)
```

8. **Strengthen edges.** For each `(neighbour, activation)` pair: `hebbian.strengthen(seed.id, neighbour.id, delta=strengthen_delta * activation)`. The activation-weighted delta means stronger-activated neighbours get stronger reinforcement.

9. **Append to `dreams.log.jsonl`** — one JSON line: `{timestamp, seed_id, neighbour_ids, dream_id, provider}`.

10. **Return `DreamResult`.**

### `DreamResult`

```python
@dataclass(frozen=True)
class DreamResult:
    dream_text: str | None          # None on dry-run
    seed: Memory
    neighbours: list[tuple[Memory, float]]  # (memory, activation)
    prompt: str                     # the user prompt sent to LLM
    system_prompt: str              # the system prompt
    memory: Memory | None           # None on dry-run; the new dream memory otherwise
    strengthened_edges: int         # count of edges strengthened (0 on dry-run)
```

## 6. CLI

```bash
# Default — real dream cycle against the nell persona
nell dream

# Specify a seed explicitly
nell dream --seed <memory-id>

# Dry-run — shows what the dream WOULD do, no LLM call, no writes
nell dream --dry-run

# Override provider (default: claude-cli)
nell dream --provider claude-cli   # production Claude via CLI subscription
nell dream --provider fake         # deterministic hash-echo for smoke testing
nell dream --provider ollama       # raises NotImplementedError until Hana fills it in

# Override persona (default: nell)
nell dream --persona nell
nell dream --persona nell.sandbox

# Full set
nell dream --persona nell.sandbox --seed abc123 --provider claude-cli --lookback 48 --depth 2
```

Flags:
- `--persona <name>` (default: `nell`)
- `--seed <memory-id>` (default: auto-select)
- `--provider <name>` (default: `claude-cli`)
- `--dry-run`
- `--lookback <hours>` (default: 24)
- `--depth <int>` (default: 2)
- `--decay <float>` (default: 0.5)
- `--limit <int>` (default: 8)

Output: prints the dream text (or dry-run summary) + the seed + the activated neighbours list.

## 7. Safety — sandbox the first run

**Hana's first real dream cycle MUST run against a throwaway copy of the nell persona DB.** This is critical: a bad LLM response writes a weird "DREAM: ..." memory that permanently lives in the canonical store. That pollution is survivable (it's one memory, can be deactivated via F22 flag) but annoying.

The operational pattern for the first few runs:
```bash
# Clone nell persona to a sandbox
cp -r "$(uv run python -c 'from brain.paths import get_persona_dir; print(get_persona_dir("nell"))')" \
      "$(uv run python -c 'from brain.paths import get_persona_dir; print(get_persona_dir("nell.sandbox"))')"

# Dream against the sandbox
nell dream --persona nell.sandbox

# Inspect — read the new "DREAM:" memory, check Hebbian delta, smell-test
sqlite3 ~/Library/Application\ Support/companion-emergence/personas/nell.sandbox/memories.db \
    "SELECT content FROM memories WHERE memory_type='dream' ORDER BY created_at DESC LIMIT 1"

# Once trusted, dream against the real nell
nell dream --persona nell
```

Document this in the engine's README or a one-liner help message.

## 8. Testing strategy

### Bridge tests (~8 tests)
- `LLMProvider.generate` contract — subclasses implement.
- `FakeProvider` determinism: same input → same output.
- `FakeProvider` name() returns "fake".
- `ClaudeCliProvider` subprocess shape (mocked via `subprocess.run` patch):
  - `--output-format json` present
  - system prompt flagged correctly
  - stdout JSON parsed
  - non-zero exit raises RuntimeError with stderr captured
- `OllamaProvider.generate` raises NotImplementedError.
- `get_provider` factory resolves names; unknown name → ValueError.

### Dream tests (~15 tests, using `FakeProvider`)
- Seed selection: picks highest-importance memory within lookback window.
- Seed selection: raises `NoSeedAvailable` if nothing in window.
- Explicit `--seed` overrides auto-selection.
- Spread-activation: neighbourhood populated from Hebbian BFS.
- Neighbourhood respects `limit`.
- Emotion aggregation: sums across seed + neighbours, clamped per EmotionalState.
- Prompt shape: system + user, user contains seed + neighbours.
- Dry-run: returns DreamResult with `memory=None`, no store write, no Hebbian strengthen.
- Full cycle: writes `memory_type="dream"` with expected metadata keys.
- Hebbian strengthen: every (seed, neighbour) pair gets weighted delta.
- `DREAM:` prefix auto-added if LLM response doesn't have it.
- Dream log: one JSONL entry appended per cycle.

### CLI tests (~5 tests)
- `nell dream --help` prints usage.
- `nell dream --dry-run --persona <tmp>` runs to completion without writes.
- `nell dream --provider fake` uses FakeProvider.
- `nell dream --provider ollama` surfaces NotImplementedError cleanly.
- Unknown persona → clear error.

Total: ~25-30 new tests. Target suite size: ~280.

## 9. Dependencies

**No new ones.** `subprocess` is stdlib. `hashlib` is stdlib. `json` is stdlib. Bridge layer adds no third-party deps.

## 10. File layout (final)

```
brain/bridge/__init__.py                 (exports LLMProvider, get_provider)
brain/bridge/provider.py                 (ABC + 3 concrete providers + factory)
brain/engines/__init__.py                (exports DreamEngine, DreamResult)
brain/engines/dream.py                   (DreamEngine, run_cycle, NoSeedAvailable)
brain/cli.py                             (adds dream subcommand dispatch)

tests/unit/brain/bridge/__init__.py
tests/unit/brain/bridge/test_provider.py
tests/unit/brain/engines/__init__.py
tests/unit/brain/engines/test_dream.py
tests/unit/brain/test_cli.py             (remove 'dream' from stubs expected list)
```

## 11. Success criteria

- All 254 Week 3.5 tests continue to pass (backward-compat).
- ~25-30 new tests across bridge + dream + CLI, all green.
- `nell dream --help` prints the subcommand usage.
- `nell dream --dry-run --persona nell --provider fake` runs end-to-end against Hana's real nell persona DB without writing anything.
- Hana runs `nell dream --persona nell.sandbox` against a cloned DB, inspects the resulting dream memory, and approves.
- CI matrix green on macOS + Ubuntu + Windows (tests use `FakeProvider`, no network).
- Merged to main. No tag (Week 4 tag will wait until all four engines ship).

## 12. Open questions deferred to later weeks

- **Dream frequency.** Scheduler lands in Week 7. Until then, manual `nell dream` runs only.
- **Multi-dream per cycle.** A single `run_cycle()` produces one dream. Batch dreaming ("run N dreams") is a future flag.
- **Dream retention / pruning.** No GC yet. Dreams accumulate alongside conversation memories. Later engines (consolidation) may decay / supersede.
- **Cross-persona dreams.** Out of scope. One persona per invocation.
- **Prompt templating.** Inline string constants for v1. If we add more engines that share prompt fragments, refactor to a templates module in Week 5.

---

*End of spec. Implementation plan to follow in next session via superpowers:writing-plans.*
