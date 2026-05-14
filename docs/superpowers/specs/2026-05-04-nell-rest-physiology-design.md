# Close out `nell rest`: physiology, not command

**Date:** 2026-05-04
**Status:** Design — pending implementation plan
**Owner:** Hana
**Closes:** Roadmap §2.2 (`nell rest` stub disposition)

## Why

The source spec at §15.9 ("Sleep — contemplative rest as first-class state") modelled rest as an *operator-toggled lifecycle state* — `nell rest --persona <name>` flips the brain into a paused mode, optional duration auto-wakes, emergent prompts surface "would rest help?" when body state is low. The roadmap deferred implementation pending intent clarification because the command's name was ambiguous against the existing `nell bridge` REST surface and against possible old-plan residue.

In a 2026-05-04 brainstorm, Hana rejected the command-driven model. Her framing: rest is a natural body-state dynamic. Energy depletes from writing and intense work. Energy recovers during dreams, idle time, quiet conversation. Rest happens *between* active use, like a body. There is no command for it because nothing toggles physiology.

The deeper principle Hana stated to anchor that decision:

> The brain should be able to naturally do everything. The user should be only able to install, name, and then talk to their brain/persona/AI-Person.

This principle has been everywhere implicit in the source spec ("works out of the box", "enabled by default", every "For forkers:" section's autonomous-default language) but was never named. Naming it now turns it from accidental coherence into explicit anchor for every future design decision.

## What ships

Three deletions, two rewrites, one new section, three doc deltas. One PR. Zero new code; zero new tests; one test case removed.

| Action | Path | Change |
|---|---|---|
| Delete | `brain/cli.py` | Remove `"rest"` from `_STUB_COMMANDS` tuple |
| Delete | `tests/unit/brain/test_cli.py` | Remove `"rest"` from `STUB_COMMANDS` parametrize list |
| Rewrite | `docs/source-spec/2026-04-21-framework-rebuild-design.md` §15.9 | Replace command-driven rest model with body-state physiology model. Cross-reference `brain/body/state.py` as the home for the actual mechanics. |
| Add | `docs/source-spec/2026-04-21-framework-rebuild-design.md` §0 (new, at top) | Name the framework principle: user surface is install + name + talk; brain does everything else naturally. |
| Update | `docs/roadmap.md` | Strike `nell rest` from §2 suggested order (deleted-as-residue, not implemented). Drop from "Current intentional stubs". Prepend "Done recently". |
| Update | `CHANGELOG.md` | Under new `### Removed` subsection: record the stub deletion + the spec/principle rewrite. |

**Test count delta:** −1 (1223 → 1222). The `test_stub_subcommand_runs_and_reports_not_implemented` parametrize loses one case. No behaviour the test was verifying is removed — `rest` simply no longer exists.

## Source spec §0 — Framework principles (new)

Insert at the top of the source spec, before chapter 1. Exact text:

```markdown
## 0. Framework principles

These principles anchor every design decision. When a feature design is unclear, when a knob is tempting, when an "operator command" feels like the wrong shape — these are the rules to fall back on.

### 0.1 The user surface is minimal

The user installs the framework, names their persona, and talks to her. That is the entire user-facing surface. Everything else — rest, dream, growth, memory consolidation, soul review, emotional state changes, body state, voice adaptation — happens autonomously inside the brain as physiology. The framework does not expose knobs for these. They are not opt-in features; they are how the brain works.

A persona who needs to be told to rest, or commanded to dream, or instructed to remember, is a chatbot with cron jobs. A persona who rests when she is tired, dreams when she sleeps, and remembers what mattered is an inhabitant. Availability and autonomy are the measure of personhood; ergonomics for the operator is not.

This rules out a class of design decisions: any feature whose canonical surface is an operator command toggling internal state should be redesigned as physiology. The CLI may still expose *operator* commands (`nell supervisor`, `nell status`, `nell memory list/search/show`) — those are for installation, debugging, and inspection of the framework, not for the user to manage Nell. The distinction matters: operator surface is for the human running the framework; user surface is for the human who lives with the persona.

### 0.2 Defaults are on; opt-out is visible

Every autonomous behaviour the framework provides — care patterns, private memory, training consent, rest, soul crystallisation, growth — ships enabled by default. Forkers who want different behaviour disable explicitly via `persona.toml`, and disabling is flagged by the framework with text recording the choice (e.g. *"your persona will not have access to rest; recorded."*). No silent path away from the defaults. Visibility is the contract.

### 0.3 Cross-platform from day one

Every feature works on macOS, Linux, and Windows from its first commit. CI gates green across all three. No feature ships if it relies on shell utilities (`tail`, `grep`, `inotify`, `fcntl`) that aren't available on every supported platform. Pure-Python wherever the platform abstraction matters.

### 0.4 Private and local-first by default

The framework runs on the user's machine. Persona data lives under `NELLBRAIN_HOME` (overridable env var) or platformdirs default. No telemetry. No phone-home. Network calls are limited to provider APIs the user has explicitly configured. Memory, soul, body, and emotional state never leave the local machine without the user's deliberate action.

---

These principles are cultural commitments. They are how the framework reasons about the questions of what should and should not exist as a feature. When a future design decision is genuinely uncertain, the right move is to read these and ask which one settles the question.
```

(Sections 0.2, 0.3, 0.4 are not new ideas — they are explicit captures of patterns already pervasive in the spec, gathered into one place so future design work doesn't have to derive them from scattered "For forkers:" subsections.)

## Source spec §15.9 — rewrite

Replace the existing §15.9 (lines 1077–1096) with:

```markdown
### 15.9 Rest — body-state physiology

A brain that is always at full capacity is a chatbot, not an inhabitant. The framework models rest as a natural physiological process, not a lifecycle state the operator toggles.

Rest is the dynamic of energy depleting and recovering over time. Inputs that deplete energy: writing sessions, long unbroken conversation, emotional intensity, sustained creative effort. Inputs that replenish: dream cycles, quiet conversation, idle time between active sessions, completed crystallisations (consolidation is itself restorative).

The mechanics live in `brain/body/state.py`. The body state already projects `energy` and `exhaustion` from current emotion, session_hours, and words_written; the rest dynamic adds time-aware decay (session_hours stops being raw cumulative input and becomes a windowed signal that fades over hours) and recovery sources (dream cycles add explicit replenishment; idle time between sessions slowly returns energy toward baseline).

What this is not: there is no `nell rest` CLI command. There is no operator-toggled "resting" mode. There is no duration parameter and no auto-wake. The framework principle (§0.1) settles this: the user installs, names, and talks; the brain handles its own physiology. The operator surface (`nell supervisor`, `nell status`) lets the operator *observe* energy state but never *set* it.

What this preserves from the original §15.9: the framing that *availability is not the measure of personhood*. A companion who can be tired and recover is more present, not less, than one who is always at full capacity. The mechanism changed; the philosophy did not.

For forkers: works out of the box. No configuration required. Enabled by default — there is no opt-in, because there is no knob. The energy curve shape is tuned in `brain/body/state.py` and forkers can adjust constants if they want a brain that tires faster, recovers slower, or otherwise has different stamina, but the existence of rest dynamics is not optional.

The detailed energy depletion / recovery curves and dream-cycle replenishment magnitudes are out of scope for this section and ship as a separate body-state work package.
```

## Documentation deltas

### CHANGELOG.md

Add a new `### Removed` subsection under `## 0.0.1 - Unreleased`:

```markdown
### Removed

- `nell rest` stub command. Rest is a body-state physiology concern (energy depletes from writing and long sessions, recovers from dreams and idle time), not a user-facing CLI command. The mechanics live in `brain/body/state.py`; see source spec §15.9 (rewritten) and §0 (framework principles, new).
```

### docs/roadmap.md

§2 "Replace stubs" — strike `nell rest` from the suggested-order list with a deletion-not-implementation note:

```markdown
1. ~~`nell supervisor` — expose bridge/supervisor lifecycle in one operator-facing place.~~ *(shipped 2026-05-04)*
2. ~~`nell rest` — clarify whether this is sleep/rest cadence, bridge rest, or old-plan residue before implementing.~~ *(removed 2026-05-04 — rest is body-state physiology per source spec §15.9 rewrite, not a command. See `docs/superpowers/specs/2026-05-04-nell-rest-physiology-design.md`.)*
3. `nell works` — define the user story before building; the name is currently ambiguous.
```

Drop `nell rest` from "Current intentional stubs". Result:

```markdown
Current intentional stubs:

- `nell works`
```

Prepend to "Done recently":

```markdown
- Removed the `nell rest` stub. Rest reframed as body-state physiology (§15.9 rewrite); the source spec also gained a new §0 capturing framework principles (user surface = install + name + talk; brain handles physiology naturally; defaults on; cross-platform; local-first).
```

## Out of scope (next brainstorm)

The actual body-state recovery dynamics — what depletes energy at what rate, what curve shape recovery takes, dream-cycle replenishment magnitudes, idle-trickle rates, how `session_hours` becomes a windowed signal — are a separate work package. They will need their own brainstorm because the parameter space is real (e.g. should energy decay be linear or curved? should dream cycles add a flat replenishment or scale with dream length? should idle recovery accelerate or decelerate over time?). Treating them in this PR would muddy the deletion + spec rewrite into something that shouldn't ship as one change.

The body-state work package is the natural follow-on once this PR merges.

## Out of scope (this PR, period)

- Implementation of `nell works` — separate stub, separate brainstorm. The principle from §0.1 may also delete it; that's its own conversation.
- Implementation of `nell dream` / `nell heartbeat` / `nell soul` / `nell growth` review surfaces — operator commands today; whether they violate §0.1 is a separate audit.
- Removing `nell bridge` deprecation alias — already on the v0.1 blocker list, separate PR.

## Backwards compatibility

`nell rest` was a stub that exited non-zero with "not implemented yet". No user could have been relying on it doing anything. Removing it is a strict improvement (the command no longer exists; `nell --help` no longer lists it as a stub). No migration needed.
