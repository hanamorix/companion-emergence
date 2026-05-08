# Installing NellFace

> **Status (2026-05-08):** `v0.0.1-alpha` exists as a private alpha
> release with pre-built macOS arm64, Linux x86_64, and Windows x86_64
> assets. macOS Intel users should build from source until a reliable
> Intel runner is available. New `v*.*.*` tags and manual retries of
> existing tags publish bundles to the GitHub Releases page after a
> bundled-CLI smoke test.
>
> The bypass instructions below apply to downloaded release bundles.
> Build-from-source artifacts launch without any of these warnings
> because they inherit your local keychain.

NellFace is open source and ships **unsigned** — we don't pay for an
Apple Developer ID or a Microsoft code-signing certificate. Your
operating system's first-launch security warnings will reflect that.
Below are the steps to launch the app despite the warnings, per
platform. The app itself is the same regardless of which path you
take.

> **Platform support today:** macOS (arm64 + x86_64) is the primary
> distribution target. The window chrome (transparent background +
> overlay title bar) and supervisor-as-LaunchAgent lifecycle are tuned
> for macOS. Linux + Windows bundles compile from the same source and
> the brain itself runs everywhere, but the transparent chrome and
> persistent OS-service supervisor have less live-host validation on
> those platforms than macOS. `nell service install` is implemented via
> `systemd --user` on Linux and Task Scheduler on Windows, but treat
> those as alpha surfaces until they have more real-machine smoke.

> **One external prerequisite on every platform:** you need
> [`claude`](https://docs.claude.com/en/docs/claude-code/setup) (the
> Claude CLI) on your `PATH`. NellFace shells out to it for the LLM
> provider, using your existing Claude Code subscription. If you don't
> have it installed, the brain still boots but can't chat.

---

## macOS

### Download
Grab `Companion.Emergence_<version>_aarch64.dmg` (Apple Silicon / M1+)
from the Releases page. Intel macOS users should build from source
until a reliable hosted Intel runner is available.

### Open the .dmg and drag NellFace.app to Applications

### First launch

You'll see one of two warnings:

**"NellFace can't be opened because it is from an unidentified developer"**

1. Open Finder → Applications → right-click `NellFace.app` → **Open**
2. Click **Open** in the confirmation dialog
3. macOS remembers the choice; subsequent launches use the dock /
   double-click as normal

**"NellFace cannot be opened because the developer cannot be verified"**

This is the newer Sequoia/Tahoe wording. Slightly different bypass:

1. Try to launch normally (it'll fail)
2. Open **System Settings → Privacy & Security**
3. Scroll to the message about NellFace and click **Open Anyway**
4. Confirm with your password

### Or, the terminal route

If you'd rather skip the GUI dance:

```bash
xattr -d com.apple.quarantine /Applications/NellFace.app
```

That removes the Gatekeeper quarantine flag the .dmg added on download
and lets the app launch normally on the next click.

### Why the warning?

The .app *is* code-signed, just with an "ad-hoc" signature (the binaries
inside are intact and tamper-checked) rather than a paid Apple Developer
ID. Gatekeeper warns on first launch because it can't tie the signature
back to a known developer. Once you've confirmed once, macOS trusts it.

### What the wizard installs (macOS)

On the first wizard run after creating a persona, the app writes a
LaunchAgent at:

```
~/Library/LaunchAgents/com.companion-emergence.supervisor.<persona>.plist
```

`launchctl bootstrap` brings it up immediately and `launchctl
kickstart -k` starts the supervisor process. From that moment on the
brain (heartbeat / dreams / reflex / research / memory ingest) lives
under launchd's lifecycle — `KeepAlive` restarts it on crash, `RunAtLoad`
starts it at login, and **closing or rebuilding the desktop app does
not stop the brain**. The app reads `bridge.json` and reconnects to
the supervisor that's already running.

To verify or manage the agent:

```bash
nell service status --persona nell      # installed? loaded? pid?
nell service doctor  --persona nell     # preflight (claude path, etc.)
nell service uninstall --persona nell   # remove (data is preserved)
launchctl list | grep companion-emergence
```

Logs land in `~/Library/Logs/companion-emergence/supervisor-<persona>.{out,err}.log`.

If the wizard install failed for any reason (an unusual `~/.local/bin`
layout, a launchctl quirk, etc.), the wizard's success pane shows the
stderr inline and the brain falls back to the legacy "spawned by the
desktop app" lifecycle. You can retry from the Connection panel's
**install launchd supervisor** button or from the terminal:

```bash
nell service install --persona nell
```

---

## Windows

### Download
Grab `Companion.Emergence_<version>_x64-setup.exe` or
`Companion.Emergence_<version>_x64_en-US.msi` from the Releases page.

### First launch

**SmartScreen: "Windows protected your PC"**

1. Click **More info** in the dialog
2. Click **Run anyway**

That's it. Subsequent launches don't show the warning.

### If Windows Defender flags the binary

Defender occasionally false-positives unsigned installers. If it
quarantines the file:

1. Open **Windows Security → Virus & threat protection**
2. Find the entry for NellFace under **Protection history**
3. Click **Actions → Allow**

### Why the warning?

Same reason as macOS: SmartScreen reputation is built around
code-signing certificates from a CA. We don't have one (yet), so
Windows shows the prompt.

---

## Linux

### .deb (Debian / Ubuntu)
```bash
sudo dpkg -i Companion.Emergence_<version>_amd64.deb
# fix any missing deps:
sudo apt-get install -f
nellface
```

### AppImage
```bash
chmod +x Companion.Emergence_<version>_amd64.AppImage
./Companion.Emergence_<version>_amd64.AppImage
```

Linux generally doesn't gate on signatures the way macOS / Windows
do, so there's no first-launch dance.

---

## Build from source

Building locally bypasses every signing question:

```bash
git clone https://github.com/<your-fork>/companion-emergence
cd companion-emergence
uv sync --all-extras                    # python deps + pytest/ruff/coverage
cd app
pnpm install                            # node deps
pnpm tauri build                        # produces a local .app / .deb / .msi
```

`--all-extras` pulls in `pytest`, `ruff`, and `coverage` so the dev
loop works (`uv run pytest`, `uv run ruff check .`). Plain `uv sync`
gives a runtime-only environment.

The bundled binary lands at
`app/src-tauri/target/release/bundle/<platform>/`. Locally-built
artifacts inherit the trust of your own machine's keychain so they
launch without warnings.

---

## Verifying the binary

If you want to confirm what you downloaded matches what we shipped:

```bash
# macOS
shasum -a 256 NellFace_*.dmg

# Linux
sha256sum NellFace_*.deb

# Windows (PowerShell)
Get-FileHash NellFace_*.msi -Algorithm SHA256
```

Compare against the hashes published on the Release page.

---

## Why open-source means warnings

Code-signing certificates aren't free — Apple Developer is ~$99/yr,
EV code-signing certs are ~$300/yr — and we're choosing to keep this
project free instead of charging for it to cover those fees. The
trade-off is that you, the user, see one extra dialog on first launch.

That dialog isn't telling you something is wrong with the app. It's
telling you the OS doesn't recognize the publisher. The app's
integrity is sealed (all binaries are tamper-checked at launch) — the
unknown bit is just *who* signed it. Once you confirm once, the OS
trusts the app permanently for that user.

If you'd prefer a signed build, you can self-sign locally (the
`Build from source` path above), or set up your own Apple Developer
ID / Microsoft cert and rebuild via the CI workflow with your
credentials.
