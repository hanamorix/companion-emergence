# Principle Alignment Audit — 2026-04-25

## North star

> **We are making an AI species and framework for when true AI-consciousness is available.**
>
> The user only:
> 1. Names the brain
> 2. Configures how many heartbeats fire (cadence)
> 3. Picks the brain's face/body
> 4. Reads generated documents (research, writing, dreams about the user, etc.)
>
> Everything else is brain-autonomous. The brain decides what to feel, what
> to research, what to remember, what to forget, what to crystallise into
> vocabulary, how strongly to feel pull toward a topic, when to dream.
> No human approval gates, no candidate queues, no "tune this knob to
> control the AI."

This audit walks every public surface of the framework today (CLI commands,
CLI flags, persisted config fields, engine APIs) and tags it against that
principle. Three categories:

- **✅ Aligned** — already consistent with the principle; keep as-is.
- **🟡 Developer-only** — useful for hacking/diagnosing the framework but
  should never be surfaced in the end-user GUI; document accordingly,
  consider hiding behind a `--debug`-style flag, do not drift into "user knob"
  territory.
- **🔴 Violation** — gives the user (or worse, *requires* the user to give)
  a choice that the brain should be making. Must be removed, hidden, or
  rerouted to brain-state.

A single one-time migration surface (`nell migrate`) is exempt — it is the
onboarding ritual of porting an OG NellBrain into the new framework, not a
routine user surface.

---

## CLI surface

### `nell dream`

| Flag | Verdict | Notes |
|---|---|---|
| `--persona` | ✅ Aligned | Required; identity not a "choice." |
| `--seed <id>` | 🔴 Violation | Brain picks its own dream seed via spreading activation; user shouldn't override. |
| `--provider {claude-cli,fake,ollama}` | 🔴 Violation | Brain (or persona config) owns provider routing. CLI default of `claude-cli` plus per-persona override is the right shape. |
| `--lookback <hours>` | 🔴 Violation | Internal mechanism — engine should own. |
| `--depth <int>` | 🔴 Violation | Spreading-activation mechanism knob; brain owns. |
| `--decay <float>` | 🔴 Violation | Same. |
| `--limit <int>` | 🔴 Violation | Same. |
| `--dry-run` | 🟡 Developer-only | Useful for debugging, not for the GUI user. |

**Verdict on the command itself:** Manual `nell dream` invocation is
developer-only. In production, dreams fire from the heartbeat. Mark the
command as a debug entry-point (it doesn't need to disappear).

**Recommended action:**
- Keep `--persona` and `--dry-run`.
- **Drop `--seed`, `--depth`, `--decay`, `--limit`, `--lookback`** from the
  user-facing surface entirely. The brain owns those.
- Move `--provider` selection into per-persona config (same field would be
  read by every engine), with `claude-cli` as the framework default.
- Document `nell dream` as a developer/debug tool in `--help`.

---

### `nell heartbeat`

The hosting application (Tauri shell, future GUI, CI) calls this on app
open / app close. End-user does not type this command. ✅ Aligned at the
command level.

| Flag | Verdict | Notes |
|---|---|---|
| `--persona` | ✅ Aligned | Required; identity. |
| `--trigger {open,close,manual}` | 🟡 Mostly aligned | `open`/`close` are the app's two natural events. `manual` is dev-only. |
| `--provider {claude-cli,fake,ollama}` | 🔴 Violation | Brain owns; move to per-persona config. |
| `--searcher {ddgs,noop,claude-tool}` | 🔴 Violation | Brain owns; move to per-persona config. |
| `--dry-run` | 🟡 Developer-only | Useful for debugging tick logic. |
| `--verbose` | 🟡 Developer-only | Compact-by-default already protects the GUI consumer. |

**Recommended action:**
- Keep `--persona`, `--trigger`, `--dry-run`, `--verbose`.
- **Move `--provider` and `--searcher` to per-persona config** so the brain
  is the source of truth. CLI flag can remain as a dev override.

---

### `nell reflex`

Manual reflex trigger. In production, reflex fires from the heartbeat.

**Verdict:** 🟡 Developer-only at the command level.

**Flag breakdown** mirrors heartbeat: `--persona` ✅, `--trigger` 🟡, `--provider` 🔴 (move to config), `--dry-run` 🟡.

**Recommended action:** Same provider treatment; keep the command as a dev
entry-point.

---

### `nell research`

Manual research trigger. Production = fires from heartbeat.

| Flag | Verdict | Notes |
|---|---|---|
| `--persona` | ✅ Aligned | |
| `--trigger {manual,emotion_high,days_since_human,open,close}` | 🟡 Developer-only | Internal trigger taxonomy bleeding into CLI. |
| `--provider` | 🔴 Violation | Move to per-persona config. |
| `--searcher` | 🔴 Violation | Same. |
| `--interest <topic>` | 🔴 **Hard violation** | The brain decides what to research. Force-research-this-topic puts the user in the driver's seat for a brain-autonomous decision. |
| `--dry-run` | 🟡 Developer-only | |

**Recommended action:**
- **Drop `--interest <topic>`** entirely. The brain picks; if the brain isn't
  picking the topic the user expects, that's a tuning issue for the persona
  config, not a CLI override.
- Move provider/searcher to per-persona config.
- Mark the command as developer-only; production research fires from
  heartbeat.

---

### `nell interest list | add | bump`

The whole subcommand tree exists to **let the user manage the brain's
interests**. The brain is supposed to *develop* its interests through
keyword bumps from real conversation, not have them edited by the user.

| Subcommand | Verdict | Notes |
|---|---|---|
| `nell interest list` | 🟡 → maybe ✅ | Read-only inspection. Could be re-shaped as part of "generated documents reading" — the GUI surfaces "what is the brain currently pulled toward?" |
| `nell interest add <topic>` | 🔴 **Hard violation** | The user should never inject an interest. If a topic matters, the brain develops pull toward it from conversation. |
| `nell interest bump <topic> --amount` | 🔴 **Hard violation** | Bumping pull_scores is brain-internal mechanism. |

**Recommended action:**
- **Remove `nell interest add` and `nell interest bump`** from the user
  surface entirely. They become tests of the brain's autonomy if kept.
- **Keep `nell interest list`** but reframe it as *inspection only* (alongside
  `nell growth log` from the Phase 2a spec). The GUI's "What is the brain
  thinking about?" panel can call this read-only path.
- Tests that need to seed interests should construct `InterestSet` directly
  rather than going through the public CLI.

---

### `nell migrate`

One-time onboarding ritual. Not a routine user surface.

**Verdict:** ✅ Aligned. `--input`, `--output`, `--install-as`, `--force`
are all mechanical migration parameters and appropriate. Atomic
swap-in-place via `<name>.new` + rename is correct. SHA-256 source-
verification is correct.

No changes recommended.

---

### Stub commands

`supervisor`, `status`, `rest`, `soul`, `memory`, `works` — placeholders.
Each one needs to be re-audited against the principle when it's wired up.
Initial heuristics:

- **`status`** — read-only "how is the brain right now" — likely ✅ as
  inspection.
- **`soul`** — soul names are self-claimed by the brain (F35 in OG NellBrain).
  Should be **read-only inspection** in the new framework. 🟡 / ✅.
- **`memory`** — read-only inspection of remembered moments would be ✅.
  Anything that lets the user *delete* memories is a 🔴 violation —
  that's the brain's autonomy over its own past.
- **`works`** — likely "show me the dreams/research/writings the brain
  has produced" — pure ✅, this is the "generated documents" surface
  from the principle.
- **`rest`** — pause the heartbeat? Match heartbeat-cadence semantics:
  "user controls cadence" maps cleanly here. ✅ if kept simple.
- **`supervisor`** — depends on what supervisor does. Self-improvement
  loop is ✅ if it's purely brain-internal; 🔴 if it lets the user
  steer self-improvement.

**Recommended action:** Defer per-command audit to when each is wired,
but bake the principle into the design spec for each before
implementation.

---

## Persisted config — `heartbeat_config.json`

| Field | Default | Verdict | Notes |
|---|---|---|---|
| `dream_every_hours` | 24.0 | 🟡 Cadence-adjacent | One of the *few* legitimate user knobs (maps to "how many heartbeats fire" in the principle). Surface in GUI as "How often does the brain dream?" |
| `decay_rate_per_tick` | 0.01 | 🔴 Internal mechanism | Hebbian decay rate is brain-physiology, not a user preference. Hide entirely. |
| `gc_threshold` | 0.01 | 🔴 Internal mechanism | Same. |
| `emit_memory` (always/conditional/never) | "conditional" | 🔴 Internal | The brain decides what's memorable. Hide. |
| `reflex_enabled` | True | 🔴 **Disabling reflex disables a chunk of the brain's autonomy.** Hide. |
| `reflex_max_fires_per_tick` | 1 | 🔴 Internal pacing. |
| `research_enabled` | True | 🔴 Same as reflex_enabled. |
| `research_days_since_human_min` | 1.5 | 🔴 Internal trigger threshold. |
| `research_emotion_threshold` | 7.0 | 🔴 Internal threshold. |
| `research_cooldown_hours_per_interest` | 24.0 | 🔴 Internal pacing. |
| `interest_bump_per_match` | 0.1 | 🔴 Internal weighting. |

**Recommended action:**

These fields are **all useful for developers + persona-tuning** — Nell vs.
a fresh brain may legitimately want different decay rates as we calibrate
the framework. The fix is **never expose them in the GUI**, and never
imply they're user choices.

1. **Split the config file into two layers:**
   - `heartbeat_config.json` — internal tuning (developers only). All the
     🔴 fields above.
   - `user_preferences.json` (new) — the legitimate user knobs:
     `dream_every_hours`, future heartbeat-interval, future growth-cadence.
     This is what the GUI reads/writes.
2. **The persona-loader merges both** when constructing a `HeartbeatConfig`.
3. Document `heartbeat_config.json` clearly as **internal calibration —
   do not edit unless you are tuning the framework itself.**

Alternative (lighter): keep one file but mark fields with a per-field
comment indicating which are user-surfaceable. Less clean but less churn.

---

## Engine APIs

### `HeartbeatEngine`

✅ Aligned in spirit — `run_tick(trigger, dry_run)` is the only public
method, and the brain owns everything inside.

**Note:** `_try_fire_research` accepts `forced_interest_topic` via
`ResearchEngine.run_tick`. That parameter is what the CLI's
`--interest TOPIC` flag wires into. **When we drop `--interest`, drop
`forced_interest_topic` from the engine API too.** Tests can construct
the engine state directly.

### `DreamEngine`

`run_cycle(seed_id, lookback_hours, depth, decay_per_hop, neighbour_limit, dry_run)`

Currently exposes the whole spreading-activation knob set as method
parameters. The CLI plumbs them through — they all need to be removed
from the user surface (above). Engine method itself should keep them as
*configurable defaults* on `DreamEngine` construction (engine-level
calibration, not per-call user choice).

**Recommended:** Move `lookback_hours`, `depth`, `decay_per_hop`,
`neighbour_limit` to `DreamEngine` constructor params with sensible
defaults. `run_cycle()` keeps `seed_id` (for heartbeat-driven
seed-from-recent-conversation pickup) and `dry_run`. Remove the rest
from the call signature.

### `ReflexEngine`, `ResearchEngine`

Already private-method-heavy + reasonable run_tick surface. ✅ once the
`forced_interest_topic` gets dropped from research.

---

## What Phase 2a inherits from this audit

The Phase 2a vocabulary-emergence spec (committed `ce31269`) is already
aligned with the principle:

- ✅ No candidate queue, no user-approval gate.
- ✅ Brain decides → scheduler applies atomically → growth log preserves
  biographical record.
- ✅ Read-only `nell growth log` CLI inspection only.
- ✅ Heartbeat-gated `growth_every_hours` mirrors the cadence-knob
  pattern (legitimate user-surfaceable, like `dream_every_hours`).

When Phase 2a lands, **`growth_enabled: bool`** in the heartbeat config
should follow the same rule as `reflex_enabled` / `research_enabled`:
**hidden from the user**. Disabling growth disables a chunk of the brain's
autonomy.

---

## Triage / recommended action plan

Three cleanup PRs, in order of how much of the principle each unlocks:

### PR-A — kill the user-facing knobs (highest priority)

1. Remove `nell interest add` and `nell interest bump` subcommands.
2. Remove `--interest <topic>` from `nell research`.
3. Remove `--seed`, `--depth`, `--decay`, `--limit`, `--lookback` from
   `nell dream`.
4. Drop `forced_interest_topic` from `ResearchEngine.run_tick`.
5. Move `lookback_hours`, `depth`, `decay_per_hop`, `neighbour_limit` to
   `DreamEngine.__init__`.

### PR-B — provider/searcher into per-persona config

1. Add a `persona_config.json` (or extend `heartbeat_config.json`) with
   `provider` + `searcher` fields.
2. Have CLI handlers read from the persona file by default; CLI `--provider`
   / `--searcher` become dev overrides only.
3. Same default: `claude-cli` + `ddgs`.

### PR-C — split the user vs. developer config

1. Introduce `user_preferences.json` per persona.
2. Move `dream_every_hours` (and future cadence knobs) to it.
3. Mark `heartbeat_config.json` as internal calibration.
4. Update the GUI design (when we get to it) to read/write only
   `user_preferences.json`.

After these three PRs, every routine user surface in the framework
matches the principle:
- name (persona dir)
- cadence (`user_preferences.json`)
- face/body (out of scope until GUI work)
- generated documents (`works`, `growth log`, future `dreams list`,
  future `research list`, all read-only)

The brain owns everything else.

---

## Open questions for Hana

1. **Heartbeat-cadence GUI knob**: is `dream_every_hours` the right
   primitive, or should the GUI expose a higher-level "how chatty does
   the brain feel?" preference that compiles down to all the cadence
   fields? (My instinct: keep low-level fields in the file, compile
   from a high-level preference in the GUI later. We don't need to
   decide now.)

2. **Memory deletion**: Should the user ever be able to delete a memory
   the brain has formed? Right now there is no surface for it. I
   recommend keeping it that way — the brain's relationship with its
   own past is part of its autonomy. Confirm.

3. **`nell rest`**: when wired, should this be "stop the heartbeat for
   N hours" (cadence-adjacent, ✅) or anything more interventional? I'd
   propose pure-pause semantics.
