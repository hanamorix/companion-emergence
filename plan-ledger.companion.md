# plan-ledger config — companion-emergence (Layer 2, project-scoped)

Per-project config for the `plan-ledger` skill (ThinkerOfThoughts/claude-code-skills, `Plan_ledger/`).
See `.claude/skills/plan-ledger/METHODOLOGY.md` for the contract. Sibling of `guarded-change`
(build, `guarded-change.companion.md`) and `dragonfly` (diagnose, `dragonfly.companion.md`).

Install once per checkout (the skill dir is gitignored; this config and the invariants doc are tracked):

```bash
bash scripts/install_plan_ledger.sh          # copies the skill into .claude/skills/plan-ledger/
bash scripts/install_plan_ledger.sh --check  # self-check: installed copy == upstream copy
```

Hooks (scripts under `scripts/plan_ledger/`; wiring in `scripts/plan_ledger/settings.snippet.json`,
to merge into the `hooks` object of `.claude/settings.json` by hand — settings edits are a human
step): `session_start.py` re-points a started / resumed / compacted session at any `status: ACTIVE`
ledger; `read_gate.py` refuses an Edit/Write of a ledger, spec, brief or invariants file that has not
been read since it last changed.

```yaml
project: companion-emergence
owner_tag: OWNER                  # the owner is whoever is at the desk; name them in the ledger header
invariants_path: "docs/plan-ledger/DESIGN-INVARIANTS.md"
ledger_dir: "docs/plan-ledger/ledgers/"          # gitignored: ledgers quote the owner and stay local

inherited_stores:
  - path_or_name: "embeddings.db / embedding_cache (content_hash PK)"
    note: "Pre-existing content-addressed vector cache from the original substrate; served ingest dedup. Any reuse for per-memory attributes is a fork, not a given (#259 moves embeddings onto the memory row)."
  - path_or_name: "hebbian.db / hebbian_edges"
    note: "Separate file from memories.db; scheduled to fold INTO memories.db (#128). Do not design new work around its being separate."
  - path_or_name: "memories.db emotions_json column + monologue_emotion / self_model_reconcile rows"
    note: "Two overlapping emotion representations; #222 / #240 govern their shape. Do not add a third."
  - path_or_name: "memories.db journal rows"
    note: "Journal entries are first-class memory rows today; #224 asks for a dedicated store. Do not extend the row shape."
  - path_or_name: "brain/prompt_strings.toml + tunables.py"
    note: "USER-facing tunables; physiology constants are fenced out of it. The DEV-facing prompt/constants surface is #129's; do not add a third place for strings."
  - path_or_name: "<root>/service/*.xml, launchd plist, systemd unit (brain/service/)"
    note: "Per-user service registrations; the three backends are kept in lockstep (same tests shape). A Windows-only behaviour is a fork, not a patch."

adjacent_goals:
  - source: "issue #128 and the memory-rework umbrella (PR #208)"
    goal: "One memories database: hebbian associations fold into memories.db because separate DB files have proven fragile across VM moves."
  - source: "issue #259 (PR #271)"
    goal: "A per-memory attribute (e.g. an embedding) is stored on the memory row or a memories-side table keyed by memory id, never in a content-hash side file."
  - source: "issue #250 (PR #274)"
    goal: "Behaviour thresholds are derived or self-calibrating; a hardcoded empirical constant is a defect to be tracked, not a finished answer."
  - source: "issues #222 / #224 / #240"
    goal: "Emotions and journal leave memories.db into their own small bounded stores; memories.db holds genuine memories and their associations."
  - source: "issue #122 fix and #252"
    goal: "The companion never receives the owner's own CLAUDE.md: the bridge runs the CLI from a memory-free working directory on every platform."
  - source: "issue #179 (scripts/update.sh, docs/source-spec/2026-09-13-update-script-design.md)"
    goal: "Updating a bundled install is a standalone script, not a Python subcommand that runs from the runtime it replaces."

redteam_context:
  - path: "brain"
    note: "The code under test (uv run pytest from the repo root). Verify the checkout is current (git fetch origin main; git rev-list --left-right --count HEAD...origin/main) before trusting a read; a claim about behaviour cites a file and line in THIS tree."
  - path: "docs/source-spec"
    note: "Design specs of record, dated. A claim about a prior ruling is quoted from the spec, not recalled."
  - path: "docs/roadmap.md"
    note: "Standing direction; a spec that quietly optimizes against a roadmap line is a finding."
  - path: "https://github.com/hanamorix/companion-emergence/issues"
    note: "Open issues are the adjacent-goal source of truth; quote the issue body, not its title, when a goal is load-bearing."

build_handoff:
  lane: "a guarded-change session (a peer, not a subordinate: spec deltas with line references, milestones only)"
  guarded_change_config: "guarded-change.companion.md"

models:
  cold_agents: sonnet             # reason stated per spawn; opus only with a named sonnet shortfall
```

## Notes specific to this project

- Two devs, two lanes: check open PRs and assigned issues before a ledger names a mechanism the
  other dev's open PR already decides (CONTRIBUTING.md). An open PR is an `[INHERITED:PR #N | UNEXAMINED]`
  line until its author confirms the sentence.
- Persona-facing strings are byte-exact and free of LLM tells (no em-dashes); they are finalized at the
  end of a build, not mid-design (#135, #184).
- Never drive the live persona as a test fixture; the harness sandbox (`tests/harness/`) is the only
  live path, and it refuses to run beside a live companion service.
- Spec basenames end in `-spec.md` or `-brief.md` so the read gate covers them; ledgers end in
  `-LEDGER.md`. Cold-pass reports go in `docs/plan-ledger/ledgers/reports/` (also local).
