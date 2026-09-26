# Brain updates from `main` via a signed build and a user-writable overlay (#286)

**Date:** 2026-09-26 · **Status:** draft for review (Hana) · **Issue:** #286 · **Supersedes the
premise of:** #255, #256 (see §10)

## 1. Problem

NellFace's "Check for updates" (ConnectionPanel) only runs Tauri's signed release updater. A fix
merged to `main` reaches app users only when a release is tagged. #286 asks the app to also
update the brain (the Python backend) from the current `main` state, dependencies included,
and restart the bridge — with no script for anyone to find and run.

Constraints found while designing (all verified 2026-09-26, §12):

- The bundle ships no pip, uv or git. `build_python_runtime.sh` deletes pip, setuptools and
  wheel ("Runtime should not self-mutate").
- Every caller runs the bundled runtime by absolute path: the app (`bundled_nell_path`), the
  launchd / systemd / Task Scheduler services, and the `~/.local/bin/nell` symlink.
- Rewriting the bundle in place (what `scripts/update.sh` does) breaks the macOS signature,
  needs sudo on `.deb` (and mixes files with dpkg, #289), and cannot work on an AppImage
  (read-only mount).
- `bridgeVersionCheck.ts` requires the bridge's major.minor.patch to equal the app's; any
  difference forces a restart and then the "version mismatch" banner.

## 2. Decisions (Hana, 2026-09-26)

| Question | Decision |
|---|---|
| Audience | Everyone, **CI-gated**: only `main` commits whose CI passed can reach users. |
| Trigger | A button. "Check for updates" shows brain-update availability; the user clicks Update. |
| Recovery | Automatic rollback when the updated bridge is unhealthy, plus one visible "Use the release brain" action while an overlay is active. |
| Approach | A: CI publishes a signed brain build; the app verifies it and installs only the changed packages into a user-writable overlay. (B, build from source on the user's machine, and C, rewrite the bundle, rejected.) |

## 3. Architecture

Five units. Each can be understood and tested alone.

### 3.1 CI publisher — `.github/workflows/brain-main.yml`

- Trigger: `workflow_run` of `test` completing successfully on a push to `main` (plus
  `workflow_dispatch` for a dry run to a test tag).
- Builds the wheel (`uv build --wheel`), exports the lock
  (`uv export --format requirements-txt --no-dev --no-emit-project --locked --emit-index-url`;
  CI's uv is new enough for `--emit-index-url`, so the file carries the pytorch-cpu index line
  itself), and runs `scripts/smoke_test_wheel.sh`.
- Writes `manifest.json`:

  ```json
  {
    "schema": 1,
    "commit": "<40-hex sha>",
    "brain_version": "0.0.42",
    "built_at": "<UTC ISO-8601>",
    "python": "3.13",
    "min_bundle_version": "0.0.43",
    "wheel": {"name": "companion_emergence-0.0.42-py3-none-any.whl", "sha256": "<hex>"},
    "requirements": {"name": "requirements.txt", "sha256": "<hex>"}
  }
  ```

  `min_bundle_version` is a constant in the workflow, set in slice 3 to the first tagged
  release containing slice 1 (0.0.43 above is illustrative). Raising it is how a future
  non-additive data change would stop older apps from taking `main` builds.
- Signs it with the existing updater key (`pnpm tauri signer sign`, secrets
  `TAURI_UPDATER_PRIVATE_KEY` / `TAURI_UPDATER_KEY_PASSWORD`, as `release.yml` already does), then
  **verifies its own signature** against the public key embedded in `tauri.conf.json`
  (`minisign -V`) before publishing.
- Replaces the four assets (`manifest.json`, `manifest.json.sig`, the wheel,
  `requirements.txt`) on a rolling pre-release tag `brain-main`.

### 3.2 App commands (Rust, `app/src-tauri`)

- `check_brain_update` → fetch `manifest.json` + `.sig`, verify with the pubkey the app already
  ships (`minisign-verify`, already in `Cargo.lock` via the updater plugin). Returns
  `{available, commit, brain_version, reason}`. A bad signature is "no update" and is logged as
  a security event, never installed.
- `apply_brain_update` → download the wheel and requirements to a temp dir, check both SHA-256s
  against the verified manifest, run the bundled `nell update --wheel … --requirements …
  --commit …` (the running bridge keeps its loaded code until it restarts). The restart then
  reuses the Restart button's graceful flow (`useRestartBridge`: `/sessions/snapshot` →
  `/supervisor/shutdown` → `ensureBridgeRunning` → `/health` poll), which ends the conversation
  through the safe path; `force_restart_bridge` (SIGKILL) only when a graceful step times out,
  exactly as that flow does today. Unhealthy → `nell update --rollback` and the same restart.
- `revert_brain` → `nell update --revert`, then restart.
- The availability decision is a pure function (unit-tested): available iff the signature is
  valid, `manifest.commit` ≠ active overlay commit, `brain_version` ≥ the bundle's version (never
  downgrade a newer app), and the bundle version ≥ `min_bundle_version`.

### 3.3 `nell update` — `brain/update/` (Python)

Behaviour depends on `install_kind`:

- **bundled** (the new path):
  1. Diff: for each exact pin in the requirements, keep it only if the bundle's installed
     version (read from the bundle's own `*.dist-info` records, never the overlay's) differs.
     Always drop `pip`, `setuptools`, `wheel` — the build strips them on purpose.
  2. Install with pip run **directly from `ensurepip`'s bundled wheel**
     (`python -P <ensurepip/_bundled/pip-*.whl>/pip install --target <overlay>/<commit>
     --no-deps --require-hashes -r <diff>`), then the wheel itself (`--no-deps`). The bundle is
     never written to.
  3. Import smoke with the overlay first on `sys.path`: `brain.cli`, `brain.bridge.server`,
     `brain.chat.engine`. Failure → delete `<commit>/`, leave `current.json` untouched.
  4. Write `<commit>/stamp.json` (`commit`, `brain_version`, `bundle_id`, installed-at) and
     swap `current.json` atomically (write temp, `os.replace`). The old `current` becomes
     `previous`; folders other than those two are pruned.
- **source** (a checkout): exec the checkout's `scripts/update.sh` with the given arguments.
  This is #256's request (a thin entry point to the script) and keeps one source-install
  updater.
- `--revert`: clear `current` (no overlay → the release brain). `--rollback`: make `previous`
  current (or clear it if there is no previous).
- A lock file in the overlay root refuses a second concurrent run.
- Overlay root: `<KINDLED_HOME>/brain-overlay/` via `brain.paths` (`get_home()`), shared by all
  personas (it holds code, not persona data).

### 3.4 Activation hook — shipped in the bundle

`build_python_runtime.sh` writes into the bundled site-packages:

- `_ce_overlay.pth` containing one line: `import _ce_overlay`.
- `_ce_overlay.py`: resolves the home directory the same way `brain.paths.get_home()` does
  (`KINDLED_HOME`, then deprecated `NELLBRAIN_HOME`, then platformdirs), reads
  `brain-overlay/current.json`, and if its `bundle_id` equals this runtime's `bundle-id` file and
  the folder exists, inserts the overlay folder at the front of `sys.path`.
- **Fails open**: any exception → do nothing → the release brain runs. It must stay cheap
  (runs at every interpreter start) and import nothing heavy.
- `bundle-id`: a file in the runtime root, written at build time: the SHA-256 of the exported
  requirements plus the brain wheel's version.

Because every launch path (wrapper, services, `pythonw -c` on Windows, the `-P -m` bridge
spawn) starts the bundled interpreter, the `.pth` activates the overlay for all of them without
changing any caller.

### 3.5 Version handshake and UI

- `/health` gains `overlay: {commit, brain_version} | null` (and `nell paths` gains
  `overlay_dir` / `overlay_active`).
- `bridgeVersionCheck.ts`: when `/health.overlay` is non-null and the bridge version ≥ the app
  version, the result is `ok`. Without an overlay the existing exact-match rule is unchanged.
- ConnectionPanel: a brain row under the app-update row — "Brain update available:
  `main @abc1234` (0.0.42)" + **Update**; while an overlay is active, "Brain: `main @abc1234`"
  + **Use the release brain**. Also fix the stale hard-coded "Current: v0.0.11" to
  `getVersion()`.

## 4. Flow

1. **Publish:** `test` green on `main` → `brain-main.yml` builds, smokes, signs, self-verifies,
   publishes.
2. **Check:** the user clicks "Check for updates" → the release check (unchanged) and
   `check_brain_update` run in parallel → the panel shows what is available.
3. **Apply:** Update → download + SHA-256 → `nell update` (diff, pip into a new folder, import
   smoke, atomic swap) → the Restart button's graceful restart (ends the conversation via the
   session snapshot) → `/health`. The button is labelled like Restart's ("End conversation and
   update the brain") so ending the conversation is never a surprise.
4. **Revert:** "Use the release brain" → clear `current` → restart. Folders are kept until
   pruned, so re-applying the same commit is instant.

## 5. When an overlay stops being valid

| Situation | Result |
|---|---|
| The app updates to a new release | New `bundle-id` → the hook ignores the old overlay → the release brain runs. If `main` is still ahead, Check offers it again. A stale overlay can never downgrade a newer bundle. |
| The app is moved or reinstalled at the same version | Same `bundle-id` → the overlay stays valid (plain files, no absolute paths — why this is a pointer file and not a venv). |
| `current.json` corrupt, folder missing | The hook does nothing → the release brain. |

## 6. Errors

The release brain is the floor; nothing leaves the user with no brain.

| Failure | Behaviour |
|---|---|
| Network, bad signature, bad manifest (Check) | "Couldn't check for a brain update"; logged (a bad signature as a security event). No install. |
| SHA-256 mismatch on download | Stop; delete the temp dir; overlay untouched. |
| pip fails (network, hash) | Delete the partial `<commit>/`; `current.json` untouched. |
| Import smoke fails | Same; "this brain build doesn't load on your machine". |
| Bridge unhealthy after the swap | `--rollback` + restart. If that is also unhealthy: clear the overlay, restart on the release brain, hand over to the existing restart / recovery UI. |
| Update during a chat turn | Like the Restart button: the conversation is ended through `/sessions/snapshot` (the safe path, memories committed) before shutdown; SIGKILL only if a graceful step times out. |
| Concurrent updates | The lock refuses the second: "an update is already running". |

## 7. Data safety — the rollback invariant

Code rolls back; persona data does not. Whatever a `main` brain writes, the release brain must
still read after a revert or rollback.

- **Rule:** persisted-state changes are **additive only** (new fields / columns). Renames,
  removals and changed meanings need their own migration spec and are not rollback-safe. The
  rule goes into CLAUDE.local.md Gotchas.
- **Root fix (slice 1):** readers that build objects with `Cls(**record)` reject unknown keys.
  Found by `git grep` (2026-09-26): `attunement/backfill.py:138` (`BackfillState`),
  `attunement/store.py:53` (`CurrentRead`), `attunement/store.py:129` (`LearnedPattern` — also
  *silently skips* the record via `except (TypeError, ValueError): continue`),
  `bridge/state_file.py:131` (`BridgeState` — on the rollback path itself),
  `ingest/emotion_backfill.py:91` (`EmotionBackfillState`). The plan's first task widens the
  sweep (`cls(**…)`, `**json.loads`, dataclass/pydantic constructors from dicts). Each reader
  keeps known fields and ignores unknown ones. SQLite stores are already tolerant (they read
  `sqlite3.Row` by name; migrations are `ADD COLUMN`).
- **Canary:** one test per persisted-state reader feeding a record with an extra unknown field;
  it must load with the known fields intact.
- **Gate:** a release brain tolerates newer data only once it contains the tolerant readers, so
  `manifest.min_bundle_version` = the first release with slice 1. Older apps are not offered
  `main` updates; they get there through a normal app update first.

## 8. Security

- The signature is checked in Rust before anything else is downloaded; every file's hash then
  comes from the verified manifest (signature → manifest → requirement hashes, enforced by
  pip's `--require-hashes`).
- A bad signature never falls back to installing.
- The overlay lives in the user's own data directory; a local attacker who can write there
  already controls the account.
- The design keeps the build script's "Runtime should not self-mutate" rule: pip runs from
  `ensurepip`'s wheel against the overlay only.

## 9. Testing

TDD throughout; temp `KINDLED_HOME`, no mocked state files; through-path tests where a path
exists.

| Unit | Tests |
|---|---|
| `brain/update/` | Diff against a fake bundle's dist-info (changed pins only; pip/setuptools/wheel dropped). Swap: current/previous rotate, atomic, prune keeps two. Lock. Import-smoke failure deletes the folder and leaves `current.json` byte-identical. Revert and rollback. Source kind execs `scripts/update.sh`. |
| Activation hook | Through-path with a real interpreter: subprocess with `KINDLED_HOME=tmp` and an overlay holding a decoy module → the overlay wins. Mismatched `bundle-id`, corrupt JSON, missing folder → bundle. Start-up cost check. **Home-resolution parity:** the hook's copy of the home lookup must equal `brain.paths.get_home()` for `KINDLED_HOME`, `NELLBRAIN_HOME` and the platformdirs default (the Rust `nellbrain_home()` drift bug, CLAUDE.local.md Gotchas, is the precedent). |
| Tolerant readers | Canary per reader (§7). |
| Rust | Manifest verification with a throwaway key pair (valid; tampered manifest, tampered sig, wrong key rejected). Availability decision table (§3.2). |
| Frontend (vitest) | Brain row in every state; `bridgeVersionCheck` with and without an overlay. |
| Overlay end-to-end in CI | Extend `runtime-build` on all three runners: build the runtime, sign a local manifest with a test key, run `nell update` from local files, confirm the overlay is active through the real launchers (`nell --version`; `pythonw -c` on Windows), revert, confirm the bundle is back. This is the Windows proof. |
| Publisher | Self-verify after signing (§3.1). |

Manual before merge of slice 4: `pnpm tauri dev` on macOS with a throwaway persona — Check →
Update → Revert. The full gate (pytest, ruff, `pnpm test`, `pnpm build`, cargo) applies to every
slice.

## 10. Effects on other issues

| Issue | Effect | Action |
|---|---|---|
| **#256** `nell update` wrapping `update.sh` | Its premise ("a Python subcommand runs from the runtime it replaces") does not hold for an overlay, which never replaces the running runtime. `nell update` becomes the bundled overlay installer and, on a source install, execs `scripts/update.sh` — which is what #256 asked for. | Closed by slice 2. |
| **#255** Windows updater (PowerShell twin) | #255 is about bundled Windows installs. The overlay path covers them (in-app and via `nell update`; pip from `ensurepip`, no bash). The build prunes nothing from `ensurepip` on Windows (verified by reading the prune list); slice 2's Windows CI run proves it. | Superseded; close when slice 4 ships. `docs/releases/cross-platform-validation.md`'s Windows-updater row is updated in slice 4. |
| **#284** opt-in training dependencies | The overlay provides the install mechanism (install the training extras into the overlay on enable). It is *not* a drop-in: today a missing `peft`/`datasets` is "a broken install, surfaced plainly" (`judge_lora.py`), so #284 must also make the self-tune tier fall back to knob-refit when those packages are absent, and move them out of the base dependencies. | Separate spec after slice 4. |
| **#289** `.deb` pip-over-dpkg mixing | The app path never writes to the bundle, so app users are unaffected. `scripts/update.sh`'s `.deb` branch still mixes files. | Stays open for the script path. |
| **#179** update-script spec | `update.sh` stays the source/dev updater; `nell update` delegates to it on source installs. | Cross-reference only. |
| **#288** (merged 2026-09-26) | Provides the `-P` wrapper and bridge spawn, the torch index fix, and the `runtime-build` workflow slice 2 extends. | Dependency met. |
| **#291** wheel smoke on PRs (open) | `brain-main.yml` reuses `scripts/smoke_test_wheel.sh`, including its `UV_TORCH_BACKEND=cpu` change. | Slice 3 depends on #291 merging. |

## 11. Wiring

- **Reads from:** the supervisor lifecycle (`nell supervisor stop/start/restart`,
  `bridge.json`), `/health`, `/sessions/snapshot` and `/supervisor/shutdown` (via the
  Restart button's graceful flow), the bundle's dist-info records and `bundle-id`.
- **Feeds into:** `/health.overlay` (read by `bridgeVersionCheck` and ConnectionPanel), the app
  log and launch-failures log (rollback events), `nell paths` (`overlay_dir`, `overlay_active`).
- **Scope:** an ops surface, not an organ — Nell's emotional and memory loops do not consume it
  (as with `update.sh`, #179). Letting Nell know she runs a `main` build is deferred (§13), not
  built half-wired.
- **Maturity:** lands EXPERIMENTAL in `docs/maturity-manifest.md`; promoted after the Linux
  validator's real-machine run (§13).

## 12. Verified facts (2026-09-26)

Spiked against a real `build_python_runtime.sh` runtime (macOS, python-build-standalone 3.13.1):

- A `--system-site-packages` venv + `uv pip install` reinstalls everything (1141 MB): uv ignores
  system site-packages. → rejected.
- `uv pip install --dry-run --python <bundle>` against the lock lists exactly the differing pins
  (0 for the same lock; the moved pin when one changes).
- A `--target` overlay holding only the diff plus the brain wheel: 8 MB; Python loads the
  changed packages and `brain` from the overlay and torch from the bundle.
- `ensurepip` survives the prune and carries pip 24.3.1; pip runs directly from that wheel,
  installs into `--target`, follows the index line inside the requirements file, and rejects a
  tampered hash ("THESE PACKAGES DO NOT MATCH THE HASHES").
- `minisign-verify` 0.2.5 is already in `app/src-tauri/Cargo.lock`; `release.yml` already signs
  with `tauri signer sign`.
- Readers: `sqlite3.Row` stores read by name; the five strict `Cls(**record)` readers are listed
  in §7.

Not yet verified (proved in the named slice):

- `ensurepip` present and pip-from-wheel working in the Windows and Linux runtimes (slice 2 CI).
- `minisign -V` accepting a `tauri signer` signature with the app's pubkey format (slice 3).

## 13. Delivery and deferred

**Slices** (order matters for safety):

1. Tolerant readers + canary + the additive-only rule. Ships in a release before any overlay is
   offered.
2. `brain/update/` + activation hook + `nell update` + the overlay end-to-end in `runtime-build`.
   Closes #256.
3. `brain-main.yml` (build, smoke, sign, self-verify, publish). Needs #291.
4. Rust commands + ConnectionPanel + version-handshake change. Closes #286 (and #255).

**Deferred** (also recorded in `project_companion_emergence_deferred.md` and the next brainstorm):

| Item | Why | Revisit |
|---|---|---|
| #284 training extras into the overlay | Needs its own tier-fallback work (§10). | After slice 4 |
| Automatic / background brain updates | Decision: button-driven. | If users forget to check |
| Update channels (release vs `main`, betas) | One channel is enough; a picker would be a knob. | If `main` proves too unstable |
| Nell's awareness of running a `main` build | Unrequested; avoid a half-wired organ. | A self-model / feed brainstorm |
| Moving `update.sh`'s bundled branch onto `nell update` | Cleanup, not need. | After slice 2 is stable |
| Offline or delta updates; keeping more than two versions | Unneeded at ~8 MB overlays. | If overlays grow |
| Linux real-machine run (Kubuntu validator) of Check → Update → Revert | No Linux host here. | Before promoting slice 4 out of EXPERIMENTAL |

**Out of scope:** updating the app shell (the signed Tauri updater keeps doing that); persona
data changes beyond the tolerant-reader fix.
