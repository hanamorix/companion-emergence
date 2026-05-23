# NellFace wizard validation runbook

This is your test plan for the wizard, against a fresh `NELLBRAIN_HOME`
so your live `nell` persona is untouched.

Files in this directory:

- `NellFace.app` — built fresh from main 7793712 with Phase 7 + ad-hoc signing
- `NellFace_0.0.1_aarch64.dmg` — same .app wrapped in a DMG, for the
  installer-flow test
- `nellbrain_home/` — empty, becomes the wizard's `$NELLBRAIN_HOME`
- `launch.sh` — runs the .app with `NELLBRAIN_HOME=./nellbrain_home`
- `cleanup.sh` — wipes nellbrain_home + launch.log so the next run
  fires the wizard fresh again

---

## Before you start

You'll need `claude` (the Claude CLI) on your `PATH`. Check:

```
which claude
```

If that comes back empty, install it (https://docs.claude.com/en/docs/claude-code/setup)
before running the test or chat will fail at the LLM call.

---

## Path A — bare .app launch (skip the DMG)

The fastest path. Validates the wizard end-to-end without exercising
the macOS Gatekeeper bypass dance.

```
cd ~/wizard-validation
./cleanup.sh
./launch.sh
```

The window should open within 1-2 seconds. Walk through the wizard:

### Step 1 — Welcome
- Choose **Fresh brain** (the migrate path is a separate test below)
- Click Next
- ✅ **Expected**: avatar transitions, no errors

### Step 2 — Persona name
- Type `validate_test` (or any other name; it'll create
  `nellbrain_home/personas/<name>/`)
- Click Next
- ✅ **Expected**: textbox accepts the name, Next is enabled

### Step 3 — Your name
- Type `Hana`
- Click Next
- ✅ **Expected**: same shape

### Step 4 — Voice template
- Pick **default** (the framework's `DEFAULT_VOICE_TEMPLATE`)
- Click Next
- ✅ **Expected**: routes to Review (no migrate step in the fresh path)

### Step 5 — Review
- Confirm the displayed config:
  - mode: Fresh brain
  - persona: validate_test
  - your name: Hana
  - voice template: default
- Compare the equivalent CLI command shown — should NOT contain
  `--provider` (P1-1 fix from 2026-05-07 audit)
- Click Install
- ✅ **Expected**: routes to Installing

### Step 6 — Installing
- Should show progress for ~3-5 seconds while `nell init` runs
  inside the bundled python-runtime
- Then transition into the main NellFace UI (avatar + panels + chat)
- ✅ **Expected**:
  - `nellbrain_home/personas/validate_test/` exists with a populated
    `persona_config.json`, `voice.md`, `active_conversations/`
  - `nellbrain_home/personas/validate_test/bridge.json` exists
    (auto-spawned bridge wrote it)
  - main UI shows the avatar (might be `idle` since no real chat
    yet) and 5 left-column panels

### Step 7 — First chat turn
- Type something in the chat panel: "morning, can you see this?"
- Send
- ✅ **Expected**:
  - WS connects without 4xx errors (check `launch.log` if not)
  - reply streams in word-by-word
  - bubble lands with a turn count

### Step 8 — Quit + relaunch
- Quit the app (Cmd-Q)
- Run `./launch.sh` again (do NOT cleanup first)
- ✅ **Expected**:
  - wizard does NOT fire — `app_config.json` remembers the
    persona, you go straight to the main UI
  - your previous chat turn is gone (we don't persist UI history
    across launches yet — only memories ingest does that)

### Step 9 — Image test (optional, costs a real claude call)
- Click the paperclip
- Pick any small image
- Send with text "what do you see"
- ✅ **Expected**: Nell's reply opens from the visible content
  (red, square, sky, whatever) per the §4 voice coaching

### Step 10 — Bridge restart with active session (v0.0.14)
- Open the persona; chat for ~30 seconds so a session buffer exists.
  Confirm the `Inner Weather` panel shows recent activity.
- In another terminal:
  `kill -STOP $(jq .pid ~/Library/Application\ Support/companion-emergence/personas/<persona>/bridge.json)`
  → bridge process is paused but not dead. `/state` poll will time
  out and the mode flips to `bridge_down`.
- Wait ≤10s for the `StatusBanner` to appear in the Connection panel
  ("Bridge offline.")
- Click **"End conversation and restart"** inside the banner.
- ✅ **Expected**: button label cycles through "Ending conversation…"
  → (5s) → "Bridge not responding — forcing restart…" (SIGKILL path,
  because the STOP'd bridge can't answer /sessions/close) → "Waiting
  for bridge to come back…" → "Reconnecting…" → banner disappears.
- ✅ `nell memory list` shows new commits from the closed session.
- ✅ `nell service status` shows a new PID (different from the one we
  STOP'd — launchd or ensure_bridge_running respawned).

### Step 11 — Bridge restart with wedged bridge (force-fallback)
- Same setup, but this time wait until /state already says `bridge_down`
  before you click. Then click **"End conversation and restart"**.
- ✅ **Expected**: button reaches "Bridge not responding — forcing
  restart…" within ~8s and clears within another ~15s. The PID changes.

### Step 12 — Linux + Windows happy path (v0.0.14)
- On each non-macOS platform, repeat Step 10 (only the first row).
- ✅ **Expected**: same label progression. `ensure_bridge_running` is
  the respawn driver on these platforms (no launchd), so the post-
  SIGKILL window may be slightly longer.

---

## Path B — DMG installer flow

Validates the macOS Gatekeeper bypass that real users will see.

1. Quit any NellFace launched via Path A
2. Double-click `NellFace_0.0.1_aarch64.dmg`
3. In the mounted DMG window: drag `NellFace.app` to `Applications`
4. Eject the DMG (or close the window)
5. Open `/Applications/NellFace.app` from Finder
6. ✅ **Expected first launch**: macOS warning *"NellFace can't be
   opened because it is from an unidentified developer"* OR the
   newer Sequoia/Tahoe wording about "developer cannot be verified"
7. Right-click `NellFace.app` in Finder → **Open** → click **Open**
   in the confirmation dialogue
8. ✅ **Expected**: app launches; wizard fires (because this is
   `~/Library/Application Support/companion-emergence`, not your
   live `companion-emergence` dir — wait, actually if you've never
   run NellFace from `/Applications` before this WILL be your live
   persona dir)

> ⚠️ **Path B uses your real `~/Library/Application Support/companion-emergence/`**.
> If you've already got `selected_persona: "nell"` in `app_config.json`
> there, the wizard will NOT fire — the app routes straight to the
> main UI for `nell`. To force the wizard via Path B, temporarily
> rename `~/Library/Application Support/companion-emergence/app_config.json`
> before launching.

---

## What to report back

For each step, note:

- ✅ what worked
- ❌ what didn't — paste the relevant chunk of `launch.log`, the
  console error from the dev tools (Cmd-Option-I in the app), or a
  screenshot of the wizard state
- ⚠️ what felt wrong even if technically working — confusing copy,
  layout issues, transitions that snapped, the avatar not picking
  up the §4 seeing coaching, etc.

The wizard is in a *codepath-correct* state from the audit work.
What we're hunting is **lived-experience issues** that no test caught.

---

## Bringing across a v0.0.12 persona (added v0.0.18)

Validates the companion-emergence → companion-emergence migration path:
a persona from an older install of the same app migrates intact into a
fresh `KINDLED_HOME`.

The fixture lives at `wizard-validation/fixtures/v0.0.12-persona/phoebe/`.
It contains 10 seeded memories (varied types, realistic content), empty
`hebbian.db` and `crystallizations.db`, and a `persona_config.json` with
`persona_name: "phoebe"` and `user_name: "zero"`.

1. Set a temp KINDLED_HOME (or back up the real one):

   ```
   export KINDLED_HOME=/tmp/wv-migration-test
   rm -rf "$KINDLED_HOME"
   ```

2. Build the .app; launch it — the wizard opens because `KINDLED_HOME`
   is empty (`app_config.json` doesn't exist yet).

3. Pick **Migrate** on the mode step → Continue.

4. Prereq check passes (claude on PATH) → Continue.

5. On the Migrate page, pick **An existing companion-emergence install**.

6. Paste the absolute path to
   `wizard-validation/fixtures/v0.0.12-persona/phoebe` into the input
   field and continue.

7. Preflight panel should show:
   - memories: **10**
   - crystallizations: **0**
   - Hebbian edges: **0**
   - user: **zero**
   - No errors or warnings

8. Continue → Review confirms source as "companion-emergence" with the
   counts above and install target `phoebe` → click Install.

9. StepInstalling: shows the MigrationReport summary card.
   - Expected: "copy migration, no skips" — 10 memories migrated,
     0 skipped, `source_kind: companion-emergence`.

10. StepReady → app routes to the main NellFace UI with `phoebe` loaded
    (avatar idle, 5 panels visible).

11. Open chat; send "can you recall what you remember about me?"
    The seeded memories should surface through `/persona/state` —
    you won't see literal memory rows in the UI but Phoebe's reply
    should reflect the context (Zero's name, coffee preference, etc.).

12. Verify on disc:
    ```
    ls "$KINDLED_HOME/personas/phoebe/"
    # expect: memories.db  persona_config.json  source-manifest.json  ...
    cat "$KINDLED_HOME/app_config.json"
    # expect: {"selected_persona": "phoebe"}
    ```

13. Cleanup:
    ```
    rm -rf "$KINDLED_HOME"
    unset KINDLED_HOME
    ```

---

## When you're done

```
./cleanup.sh
```

Wipes the validation NELLBRAIN_HOME. Your live `nell` persona at
`~/Library/Application Support/companion-emergence/personas/nell/`
is never touched by this validation flow.
