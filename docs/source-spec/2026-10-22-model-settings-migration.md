# Model and settings migration: what is renamed or removed, and what a revert does (#529)

Migration spec required by `2026-09-26-brain-update-overlay-design.md` section 7: "Renames, removals and changed
meanings need their own migration spec and are not rollback-safe." This change renames one file and removes three
persona keys, so it is not rollback-safe, and this document says what that costs.

## 1. What changes

| Before | After |
|---|---|
| `$KINDLED_HOME/tunables.json` (an `overrides` object) holds the install's settings | `$KINDLED_HOME/tunables.toml` holds them. The first start that finds a `tunables.json` folds it into the TOML and renames it `tunables.json.migrated-<UTC stamp>` (never deleted by the app). A `tunables.json` found later (a restored backup, a copied install) is folded the same way. |
| Each persona's `persona_config.json` holds `model`, `tier_models`, `derived_unavailable` | The model is one install-wide setting per job in `tunables.toml` (`model.<job>`). The three keys are removed from a persona's file by the migration. |

Folding rules (code: `brain/tunables_migration.py`, `brain/model_migration.py`):

- A JSON value goes into the TOML only where the TOML value still equals the shipped default; a setting the user already
  changed keeps its value and the old one is logged and discarded.
- The persona model of the most recently opened Claude persona is written once (the decision), and the marker
  `model_migration.done` records every persona that decision considered. A persona in the marker is only cleared on later
  starts, never applied. A persona that is not in the marker (imported later) is applied only where the install's value
  is still the shipped default.

## 2. What an older brain does after an in-app revert (`nell update --revert`, or a rollback to the release brain)

- The older brain does not read `tunables.toml` or `model_migration.done`. It runs on its own defaults: its chat model is
  the bare `sonnet` alias, and every other tunable is at its shipped default. The user's model picks and tuned settings
  are not in effect while it runs.
- Nothing is lost. `tunables.toml` is a file the older brain never opens, so it stays as it was.
- Changes made while downgraded are written the old way: a `tunables.json` (settings) and a `model` key in a persona's
  file (the picker).
- On the next upgrade the new brain uses `tunables.toml` again, so the user's values are back in effect, and then:
  - a fresh `tunables.json` is folded by the rule above: a setting still at its default takes the value the older brain
    saved, a setting the user changed since keeps the TOML value;
  - a persona `model` the older brain saved in a persona the decision considered is cleared and not applied (a pick made
    in the older brain is not carried over; the user picks the model again if they want a different one).
- A user who wants the older brain to run with their values re-sets them there; there is no automatic copy back (the
  migration is one way).

Verified by `tests/unit/brain/test_model_migration.py::test_an_upgrade_after_a_downgrade_keeps_the_settings_and_folds_what_the_older_brain_wrote`.

## 3. Decision

Owner ruling (2026-10-22), on whether the file rename and the persona-key removal are acceptable given the rollback rule:

> "for #1, this risk there being an ever increasing number of stale files. given that a user's settings aren't actually lost on a downgrade, they're just not used, and the next time they switch to an upgrade they'd get used again, I call it acceptable. keep the removal."

So: the `tunables.json` rename and the removal of the persona keys stay as built. The cost accepted is that settings are
not in effect while the older brain runs.
