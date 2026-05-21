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
