# Troubleshooting

## The "GDK popup" warning on Linux

If you launch NellFace from a terminal you may see:

```
(nellface:29789): Gdk-WARNING **: Tried to map a popup with a non-top most parent
```

This is **benign**. It comes from WebKitGTK's interaction with the KDE / GNOME
compositor and doesn't affect functionality. We don't suppress it because
filtering Tauri's stderr could swallow real errors. The upstream issue is
tracked at <https://gitlab.gnome.org/GNOME/gtk/-/issues/> — search for
"non-top most parent".

## Auto-update isn't working on Linux

Auto-update is **AppImage-only**. If you installed the `.deb` package on
Debian / Ubuntu / Kubuntu, you'll see a "Visit releases page" link instead of
the "Download & Install" button in the Connection panel — click it to grab
the newer `.deb` and install it manually (`sudo dpkg -i companion-emergence_*.deb`).

If you're running the AppImage and update still hangs:
- Ensure the AppImage is executable: `chmod +x Companion.Emergence_*.AppImage`
- Move it out of read-only locations (e.g. anywhere under `/opt/` set up by
  another package manager)
- Run from a writable location (typically `~/Downloads/` or `~/Applications/`)

## Where are the logs?

On macOS: `~/Library/Logs/companion-emergence/`
On Linux: `~/.local/state/companion-emergence/log/`
On Windows: `%LOCALAPPDATA%\hanamorix\companion-emergence\Logs\`

Run `nell paths` to see the exact resolved values on your machine, or set the `KINDLED_HOME` environment variable to override the root.

Key files to look at:

- `<persona>/heartbeats.log.jsonl` — every heartbeat tick (autonomous body loop)
- `<persona>/dreams.log.jsonl` — every dream (offline reflection)
- `<persona>/soul_audit.jsonl` — soul-review decisions (kept forever)
- `bridge-<persona>.log` — bridge daemon stdout / stderr (in the log dir above)

## Memories forgotten before the #362 fix

Until #369, the forgetting pass treated a memory that soul review had accepted like any other. It could fade the memory, and later drop it, even though a crystallisation was built on it. #369 stops that going forward. It does not bring back what was already lost. In a read-only check of one long-lived persona, about 40% of its soul-accepted memories were already gone and none were mid-fade. The crystallisations themselves are intact. The episodic memories behind them are what went.

What you can get back depends on how far the memory got:

| State when the bug hit | Original text kept? | How to get it back |
|---|---|---|
| `fading` | Yes, in `content_snapshot`. | With a copy of the persona directory from before the loss, `nell recover --persona <name> --from <that copy>` un-fades a memory that is faded in the live database and present in the copy. Without a copy there is no command: `MemoryStore.unfade(memory_id)` restores it. The forgetting pass will not do it either way: exempt memories are skipped before any state change, so a faded exempt memory stays faded. Quit the app and stop the bridge first (`nell supervisor stop --persona <name>`), for the reason `nell recover` does below. |
| `lost` (hard-deleted) | Usually not. The graveyard (`forgotten_memories.jsonl`) keeps only the summary written when the memory faded, its emotions and its hebbian neighbours. The exception is a memory the dedup sweep removed: its full record is in `consolidation_archive.jsonl` (the `target` field), written before the row was deleted. | `nell recover --persona <name>` restores summary stubs (see below). To get the original text back you need either that archive entry or a copy of the persona directory from before the loss: `nell recover --persona <name> --from <that copy>`. |

### What `nell recover` does here

The app's Connection panel has a **Recover memories…** button. Opening it runs `nell recover --dry-run` and shows "N memories to restore, N to un-fade"; **Restore now** then runs the real command (`--from` is added if you gave it a folder; leave the folder blank to restore from the graveyard). The button never passes `--force`, and the bridge is running while the app is open, so the real run exits with "Bridge appears to be running". A terminal run refuses for the same reason. To restore: quit the app, run `nell supervisor stop --persona <name>`, then run `nell recover` from a terminal. Opening the app again starts the bridge.

- With no `--from`, it works from the graveyard. It restores a summary stub for **every** graveyard entry that is not in `memories.db` and has a creation timestamp. It cannot be limited to soul-linked memories.
- Run `--dry-run` first to see the counts. It writes nothing.
- It refuses to run while the bridge looks live, because it must not race the forgetting pass. `--force` overrides that.
- Before it writes, it copies `memories.db`, `hebbian.db` and `emotion_vocabulary.json` into `recover-backup-<UTC timestamp>/` in the persona directory.
- It rewrites `source-manifest.json`: `migrated_at_utc` and `lived_age_hours_at_migration` are set to now. That restarts the import grace, so every memory created before the recovery is exempt from fading for 7 lived days (`IMPORT_GRACE_LIVED_HOURS` in `brain/forgetting/policy.py`), not only the ones it restored. Know that before running it on a live persona.
- Restoring summary stubs into a live persona changes what she remembers. That is the persona owner's call, so nothing runs it automatically.

### What does not help

The accepted candidate in `soul_candidates.jsonl` keeps the text the soul review saw. In the persona that was checked, that text was longer than the graveyard stub in only about 5% of cases, so it rarely recovers more than the stub does.

### Known unknown

Some soul-accepted memory ids are in neither `memories.db` nor the graveyard. `consolidation_archive.jsonl`, which the dedup sweep and the consolidation gate write, did not account for them in the sample checked. The cause has not been identified. It is tracked in #394.
