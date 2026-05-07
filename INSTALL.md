# Installing NellFace

NellFace is open source and unsigned — we don't pay for an Apple
Developer ID or a Microsoft code-signing certificate. Your operating
system's first-launch security warnings will reflect that. Below are
the steps to launch the app despite the warnings, per platform. The
app itself is the same regardless of which path you take.

> **One external prerequisite on every platform:** you need
> [`claude`](https://docs.claude.com/en/docs/claude-code/setup) (the
> Claude CLI) on your `PATH`. NellFace shells out to it for the LLM
> provider, using your existing Claude Code subscription. If you don't
> have it installed, the brain still boots but can't chat.

---

## macOS

### Download
Grab `NellFace_<version>_aarch64.dmg` (Apple Silicon / M1+) or
`NellFace_<version>_x64.dmg` (Intel) from the Releases page.

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

---

## Windows

### Download
Grab `NellFace_<version>_x64-setup.exe` or
`NellFace_<version>_x64_en-US.msi` from the Releases page.

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
sudo dpkg -i nellface_<version>_amd64.deb
# fix any missing deps:
sudo apt-get install -f
nellface
```

### AppImage
```bash
chmod +x NellFace_<version>_amd64.AppImage
./NellFace_<version>_amd64.AppImage
```

Linux generally doesn't gate on signatures the way macOS / Windows
do, so there's no first-launch dance.

---

## Build from source

Building locally bypasses every signing question:

```bash
git clone https://github.com/<your-fork>/companion-emergence
cd companion-emergence
uv sync                                 # python deps
cd app
pnpm install                            # node deps
pnpm tauri build                        # produces a local .app / .deb / .msi
```

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
