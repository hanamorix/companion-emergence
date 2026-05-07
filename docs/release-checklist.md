# Release checklist

This project is private/local-first during development. Before sharing a public build or tagged release, do this checklist instead of relying on ad-hoc local state.

## Pre-release verification

- Run the full test suite: `python3 -m pytest -q`
- Run lint: `python3 -m ruff check .`
- Confirm the bridge starts locally and writes `bridge.json` with owner-only file permissions.
- Smoke-test a complete companion loop:
  - create a session
  - send a chat turn
  - close the session
  - confirm memories ingest
  - run soul review/growth paths that are meant to be active for this release
- Review `docs/audits/` and confirm no unresolved P1/P2 findings apply to the release candidate.

## Privacy and local-data checks

- Confirm no bearer tokens, local persona data, memory databases, audit logs, or `.env` files are committed.
- Confirm MCP tool invocation logs use the intended privacy mode/redaction behavior.
- Confirm WebSocket authentication uses `Sec-WebSocket-Protocol: bearer, <token>` rather than URL query-string tokens.

## Packaging/versioning

- Set the intended version in `pyproject.toml`.
- Add or update `CHANGELOG.md` before tagging or sharing a build.
- Review `docs/roadmap.md` and confirm known stubs/incomplete surfaces are documented.
- Build the wheel/sdist only after tests and lint pass.
- **Run `bash scripts/smoke_test_wheel.sh`** — builds the wheel + sdist,
  installs into a fresh `uv venv`, exercises `nell --version` /
  `nell init` / `nell status` against a temp `NELLBRAIN_HOME` to
  confirm the package metadata + entry points are honest. The script
  exits non-zero on the first failure; passing means an outside
  installer can use the wheel without falling back to the source tree.
- Do not add release automation until the CLI/API surface is stable enough to version.

## Phase 7 — NellFace.app cross-platform release

The Phase 7 bundled Python runtime ships inside the Tauri .app, so a
release means producing platform-specific bundles for distribution.

### Building locally

- macOS arm64 / macOS x86_64 / Linux x86_64 / Linux arm64: from `app/`
  run `pnpm tauri build`. The `beforeBuildCommand` in `tauri.conf.json`
  invokes `bash build_python_runtime.sh` which downloads
  `python-build-standalone` for the host arch and installs the brain
  wheel into the bundled site-packages. Bundle output:
  `app/src-tauri/target/release/bundle/<platform>/`.
- Windows x86_64: same command, but the build script must run via
  Git Bash (ships with Git for Windows) — the `find` / `tar` /
  `curl` invocations rely on POSIX tooling.

### Building via CI

`.github/workflows/release.yml` matrices across macOS arm64, macOS
x86_64, Linux x86_64, and Windows x86_64. Triggered by pushing a
`v*.*.*` tag. Bundles upload as workflow artifacts (`.app`, `.dmg`,
`.deb`, `.AppImage`, `.msi`, `.exe`). Signing/notarization is NOT
in CI — see below.

### Signing — open-source default

This project is open source and ships **without paid signing**. The
default state is:

- **macOS**: ad-hoc signed (`bundle.macOS.signingIdentity = "-"` in
  `tauri.conf.json`). All binaries, including the embedded
  `python-runtime/` tree, get sealed into the .app's signature
  via `codesign --deep`. `codesign --verify --deep --strict` passes.
  Gatekeeper still warns "unidentified developer" on first launch
  because there's no paid Apple Developer ID, but right-click →
  Open works as expected. Users get the steps in `INSTALL.md`.
- **Windows**: unsigned. SmartScreen warns "Windows protected your
  PC"; the user clicks "More info → Run anyway." Documented in
  `INSTALL.md`.
- **Linux**: unsigned. `.deb` and `.AppImage` install without a
  signing dance.

Confirm before any tagged release:

- Run `pnpm tauri build` locally on each target host
- macOS: `codesign --verify --deep --strict NellFace.app` exits 0
- macOS: `codesign -dv --verbose=2 NellFace.app | grep "Signature=adhoc"`
- All three platforms: launch the bundle from a clean user
  directory and verify a chat turn round-trips with `claude` on PATH

### Optional — paid signing (if/when budget permits)

If you decide to pay for signing later, the manual steps:

**macOS** (Apple Developer ID, ~$99/yr):
1. Get a Developer ID Application certificate via Xcode →
   Preferences → Accounts → Manage Certificates.
2. Replace `signingIdentity = "-"` in `tauri.conf.json` with
   `"Developer ID Application: YOUR NAME (TEAM_ID)"`.
3. Add notarization keys to your keychain (`xcrun notarytool
   store-credentials`).
4. Build: `pnpm tauri build`. Tauri handles signing inline.
5. Notarize the resulting .dmg: `xcrun notarytool submit
   NellFace.dmg --keychain-profile <stored-name> --wait`.
6. Staple: `xcrun stapler staple NellFace.dmg`.

**Windows** (OV or EV code-signing cert):
1. Get an OV or EV cert from a CA (DigiCert, Sectigo, etc.).
2. Sign: `signtool sign /tr http://timestamp.digicert.com /td
   sha256 /fd sha256 /a NellFace.msi`.
3. Verify: `signtool verify /pa NellFace.msi`.

**Linux** (.deb / AppImage):
1. .deb signing via dpkg-sig is optional but recommended for
   public repos that want to be added to a third-party APT source.
2. AppImage zsync + GPG signing for delta updates.

### Auto-update

Tauri ships `tauri-plugin-updater` for in-app updates. Not yet
wired — needs an update server (S3 + signed manifest, or a managed
service like updately.app). Defer until the first public release
gets feedback on actual demand.

## Known incomplete surfaces

- Growth modules that remain incomplete must be called out in release notes.
