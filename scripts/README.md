# Scripts — operator and contributor utilities

A handful of one-off and ongoing scripts that don't belong in the
canonical `nell` CLI. Each script is rated by **safety tier**: read
the tier before running and treat anything above tier 1 as
explicitly opt-in.

## Safety tiers

| Tier | Meaning | Examples |
| --- | --- | --- |
| **1 — Safe / CI** | Read-only or pure-temp. No live persona, no live LLM, no docs mutation. Safe to run in CI and on a contributor laptop unattended. | `smoke_test_wheel.sh`, `backfill_soul_for_persona.py --dry-run` |
| **2 — Safe with quota** | Talks to a real LLM provider via `claude-cli`. Spends real Anthropic quota but operates on a temp persona that gets torn down. | `stress_test_image.py` (set `RUN_LIVE_CLAUDE_STRESS=1`) |
| **3 — Live persona / mutating** | Reads or writes a real persona on this machine, or mutates tracked docs. Requires explicit `--live` opt-in. | `stress_test_voice.py --live`, `backfill_soul_for_persona.py` (no `--dry-run`) |

A script's docstring or `--help` is the source of truth for its tier;
this table is a quick map.

## Inventory

### `smoke_test_wheel.sh` — tier 1

Builds the wheel + sdist, installs the wheel into a fresh `uv venv`
under `mktemp`, and runs read-only `nell` invocations against the
installed package to confirm metadata / entry points are honest.
Asserts `brain.__file__` lives inside the temp venv so a working-dir
shadow can't false-positive.

```bash
bash scripts/smoke_test_wheel.sh
```

Cost: tens of seconds, network for `uv build`. No live persona. No
LLM calls. Adds the wheel + sdist to `dist/` (cleaned up on next
build).

### `stress_test_image.py` — tier 2

Builds a temp persona in `tempfile.mkdtemp`, starts an in-process
bridge with the real `ClaudeCliProvider`, uploads a 4×4 PNG, sends a
chat with `image_shas`, and asserts Nell's reply mentions visual
content. Tears down the temp persona on exit.

```bash
RUN_LIVE_CLAUDE_STRESS=1 uv run python scripts/stress_test_image.py
```

Cost: one live `claude --print` subprocess call (one Anthropic
API-billing event under your Claude Code subscription). The
`RUN_LIVE_CLAUDE_STRESS=1` gate exists so a contributor can't burn
quota by accident.

### `stress_test_voice.py` — tier 3 (when `--live`)

Multi-turn voice / persona stress harness. Defaults to refusing
contact with a real persona — pass `--persona-dir <path>` for an
isolated temp persona, or `--live --persona <name>` to target the
user's installed companion-emergence persona.

```bash
# tier 1 — pure dry run, no LLM, no live state
uv run python scripts/stress_test_voice.py --persona-dir /tmp/test-home/personas/test --no-write

# tier 3 — live persona, real provider, writes report
uv run python scripts/stress_test_voice.py --live --persona nell --output /tmp/voice-report.md
```

Cost (tier 3): many live provider calls; mutates the live persona's
memory store. Reports default to stdout; pass `--output <path>` for
a Markdown report. The previous version hardcoded the live Nell
persona dir + an `/audits/` doc path and overwrote a tracked file
on every run; the rewrite preserves no such defaults.

### `backfill_soul_for_persona.py` — tier 3 (when not `--dry-run`)

Re-derives soul candidate entries from the active memory store for a
persona that's missing them (e.g. a partial migration). Read-only
under `--dry-run`; writes to `<persona>/soul_candidates.jsonl` and
`<persona>/soul.json` otherwise.

```bash
# tier 1 — dry run, prints what it would do
uv run python scripts/backfill_soul_for_persona.py --persona nell --dry-run

# tier 3 — actual backfill
uv run python scripts/backfill_soul_for_persona.py --persona nell
```

The supervisor must be stopped before a non-dry-run backfill so the
JSONL append and a live soul-review pass don't race.

## Adding a new script

If you add anything here:

1. Lead the docstring with a tier line: `# Safety tier: <N>`.
2. Default to safe behavior; require an explicit flag (or env var)
   to escalate.
3. Document the cost (LLM quota, mutation surface, runtime).
4. Add an entry to this README under **Inventory**.
