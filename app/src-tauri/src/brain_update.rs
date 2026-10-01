//! Brain updates from `main` (#286 slice 4). Pure helpers — manifest
//! verification, the availability decision, asset checks — are unit-tested
//! here; the Tauri commands (Task 3) are thin I/O around them.

use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub(crate) const REPO_RELEASES: &str =
    "https://github.com/hanamorix/companion-emergence/releases/download";
pub(crate) const DEFAULT_TAG: &str = "brain-main";
pub(crate) const TAG_ENV: &str = "KINDLED_BRAIN_UPDATE_TAG";
pub(crate) const MANIFEST_SCHEMA: u32 = 1;

fn b64_text(s: &str, what: &str) -> Result<String, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| format!("{what} base64: {e}"))?;
    String::from_utf8(bytes).map_err(|e| format!("{what} utf-8: {e}"))
}

/// Verify `manifest` against tauri-signer's base64-wrapped minisign signature,
/// with the base64-wrapped public key the app ships (tauri.conf.json
/// plugins.updater.pubkey) — the same unwrap tauri-plugin-updater does. The
/// trusted comment must name manifest.json, so no other file signed with the
/// shared updater key can pass as a manifest.
pub(crate) fn verify_manifest(manifest: &[u8], sig_b64: &str, pubkey_b64: &str) -> Result<(), String> {
    let pk = minisign_verify::PublicKey::decode(&b64_text(pubkey_b64, "public key")?)
        .map_err(|e| format!("public key: {e}"))?;
    let sig = minisign_verify::Signature::decode(&b64_text(sig_b64, "signature")?)
        .map_err(|e| format!("signature: {e}"))?;
    pk.verify(manifest, &sig, false)
        .map_err(|e| format!("signature does not verify: {e}"))?;
    if !sig.trusted_comment().split('\t').any(|f| f == "file:manifest.json") {
        return Err(format!("signature is for another file: {}", sig.trusted_comment()));
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub(crate) struct Asset {
    pub name: String,
    pub sha256: String,
}

/// slice 3's manifest.json. Unknown fields are ignored (serde default), so a
/// later field (e.g. a signed channel) stays compatible with this app.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub(crate) struct Manifest {
    pub schema: u32,
    pub commit: String,
    pub brain_version: String,
    pub built_at: String,
    pub python: String,
    pub min_bundle_version: String,
    pub wheel: Asset,
    pub requirements: Asset,
}

#[derive(Debug)]
pub(crate) enum ManifestError {
    /// A schema this app doesn't know: a newer app is needed, whatever the rest says.
    Unsupported,
    Invalid(String),
}

/// Read the schema before the struct, so a future manifest whose fields changed
/// shape reads as "needs a newer app", not as a broken download.
pub(crate) fn parse_manifest(bytes: &[u8]) -> Result<Manifest, ManifestError> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| ManifestError::Invalid(format!("manifest.json: {e}")))?;
    if v.get("schema").and_then(|s| s.as_u64()) != Some(u64::from(MANIFEST_SCHEMA)) {
        return Err(ManifestError::Unsupported);
    }
    serde_json::from_value(v).map_err(|e| ManifestError::Invalid(format!("manifest.json: {e}")))
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct BrainUpdateCheck {
    pub available: bool,
    pub commit: Option<String>,
    pub brain_version: Option<String>,
    /// Machine reason the frontend matches on (see Task 2 Interfaces).
    pub reason: String,
    pub detail: Option<String>,
}

impl BrainUpdateCheck {
    pub(crate) fn none(reason: &str, detail: Option<String>) -> Self {
        Self { available: false, commit: None, brain_version: None, reason: reason.into(), detail }
    }
}

/// Leading major.minor.patch (like the frontend's _parseSemver); "0.0.28a1" → (0, 0, 28).
pub(crate) fn parse_triple(v: &str) -> Option<(u64, u64, u64)> {
    let mut parts = v.trim().trim_start_matches('v').splitn(3, '.');
    let mut next = || -> Option<u64> {
        let digits: String = parts.next()?.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    };
    Some((next()?, next()?, next()?))
}

/// Available iff the manifest schema is known, this app is new enough for the
/// build (min_bundle_version — the rollback invariant gate), the build is not
/// older than the app's own brain (never downgrade), and it isn't already active.
/// The signature was checked before this is called.
pub(crate) fn decide(m: &Manifest, active_commit: Option<&str>, bundle_version: &str) -> BrainUpdateCheck {
    let out = |available: bool, reason: &str| BrainUpdateCheck {
        available,
        commit: Some(m.commit.clone()),
        brain_version: Some(m.brain_version.clone()),
        reason: reason.into(),
        detail: None,
    };
    if m.schema != MANIFEST_SCHEMA {
        return out(false, "unsupported_manifest");
    }
    let (Some(brain), Some(bundle), Some(min)) = (
        parse_triple(&m.brain_version),
        parse_triple(bundle_version),
        parse_triple(&m.min_bundle_version),
    ) else {
        return out(false, "unparseable_version");
    };
    if bundle < min {
        return out(false, "app_too_old");
    }
    if brain < bundle {
        return out(false, "older_than_app");
    }
    if active_commit == Some(m.commit.as_str()) {
        return out(false, "already_active");
    }
    out(true, "available")
}

/// The downloaded files are written under these names: no path separators or
/// `..`, and only the two shapes slice 3 publishes.
pub(crate) fn check_asset_names(m: &Manifest) -> Result<(), String> {
    let w = &m.wheel.name;
    let plain = |s: &str| !s.contains('/') && !s.contains('\\') && !s.contains("..");
    if !(plain(w) && w.starts_with("companion_emergence-") && w.ends_with("-py3-none-any.whl")) {
        return Err(format!("unexpected wheel name {w:?}"));
    }
    if m.requirements.name != "requirements.txt" {
        return Err(format!("unexpected requirements name {:?}", m.requirements.name));
    }
    Ok(())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// KINDLED_BRAIN_UPDATE_TAG may pick a brain-main dry-run tag (testing); anything
/// else falls back to brain-main. Assets are still verified against the app's key.
pub(crate) fn resolve_tag(env: Option<&str>) -> String {
    match env {
        Some(t) if t == DEFAULT_TAG => t.to_string(),
        Some(t) if t.strip_prefix("brain-main-").is_some_and(|rest| {
            !rest.is_empty() && rest.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        }) => t.to_string(),
        _ => DEFAULT_TAG.to_string(),
    }
}

pub(crate) fn asset_url(tag: &str, name: &str) -> String {
    format!("{REPO_RELEASES}/{tag}/{name}")
}

fn status_json(status: &str) -> Option<serde_json::Value> {
    serde_json::from_str(status).ok()
}

/// `nell update --status` → does this install support overlay updates?
pub(crate) fn status_supported(status: &str) -> bool {
    status_json(status).and_then(|v| v.get("supported")?.as_bool()).unwrap_or(false)
}

/// `nell update --status` → the active overlay's commit, if any. An overlay built
/// for another bundle (the app was reinstalled under it) never loads, so it
/// doesn't count as active.
pub(crate) fn active_commit_from_status(status: &str) -> Option<String> {
    let v = status_json(status)?;
    let active = v.get("active")?;
    let bundle = v.get("bundle_id")?.as_str()?;
    if active.get("bundle_id")?.as_str()? != bundle {
        return None;
    }
    active.get("commit")?.as_str().map(str::to_string)
}

/// `nell update --rollback`'s printed state → the overlay it landed on (None = the
/// release brain), so the app can say which brain is back (#335).
pub(crate) fn active_commit_after_flip(stdout: &str) -> Option<String> {
    status_json(stdout)?.get("active")?.get("commit")?.as_str().map(str::to_string)
}

/// What the launch path needs to know about the active overlay (#335).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct BrainOverlayStatus {
    pub active_commit: Option<String>,
    /// false: installed but never seen running; missing/odd → true (never auto-drop on a guess)
    pub confirmed: bool,
    /// "rollback" | "revert"; missing/odd → "revert" (the release brain, the floor)
    pub undo: String,
}

pub(crate) fn overlay_status_from(status: &str) -> BrainOverlayStatus {
    let Some(commit) = active_commit_from_status(status) else {
        return BrainOverlayStatus { active_commit: None, confirmed: true, undo: "revert".into() };
    };
    let active = status_json(status).and_then(|v| v.get("active").cloned()).unwrap_or_default();
    BrainOverlayStatus {
        active_commit: Some(commit),
        confirmed: active.get("confirmed").and_then(|c| c.as_bool()).unwrap_or(true),
        undo: match active.get("undo").and_then(|u| u.as_str()) {
            Some("rollback") => "rollback",
            _ => "revert",
        }
        .into(),
    }
}

/// The base64-wrapped minisign public key from tauri.conf.json's updater plugin.
pub(crate) fn updater_pubkey(plugins: &HashMap<String, serde_json::Value>) -> Result<String, String> {
    plugins
        .get("updater")
        .and_then(|u| u.get("pubkey"))
        .and_then(|p| p.as_str())
        .map(str::to_string)
        .ok_or_else(|| "no updater pubkey in tauri.conf.json".to_string())
}

use std::process::Stdio;
use std::time::{Duration, Instant};
// tauri::AppHandle::config()/package_info() are inherent methods in tauri
// 2.11 — no `tauri::Manager` import needed (unlike bundled_nell_path's
// resource_dir(), which does require it; see lib.rs).

const MANIFEST_MAX: usize = 64 * 1024;
const SIG_MAX: usize = 4 * 1024;
const WHEEL_MAX: usize = 64 * 1024 * 1024;
const REQUIREMENTS_MAX: usize = 1024 * 1024;
const STATUS_TIMEOUT_S: u64 = 30;
const INSTALL_TIMEOUT_S: u64 = 900; // pip may fetch a changed wheel or two
const FLIP_TIMEOUT_S: u64 = 60;

/// The updater's own nell runs on the release brain (KINDLED_NO_OVERLAY=1): a
/// broken overlay must not break the tool that replaces it. Never used for the
/// bridge restart, which must load the new overlay.
fn floor(std_cmd: std::process::Command, args: &[&str]) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::from(std_cmd);
    cmd.args(args)
        .env("KINDLED_NO_OVERLAY", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Run `nell <args>` on the release brain; a spawn error, timeout or non-zero
/// exit is logged to launch-failures.log and returned as Err(stderr tail).
async fn run_nell(app: &tauri::AppHandle, args: &[&str], timeout_s: u64) -> Result<String, String> {
    let label = format!("nell {}", args.join(" "));
    let started = Instant::now();
    let mut cmd = floor(crate::nell_command(app)?, args);
    let out = match tokio::time::timeout(Duration::from_secs(timeout_s), cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            let msg = format!("spawn {label}: {e}");
            crate::record_spawn_failure(app, &label, None, Some(&msg), started);
            return Err(msg);
        }
        Err(_) => {
            crate::record_spawn_failure(app, &label, None, Some("timeout"), started);
            return Err(format!("{label} timed out"));
        }
    };
    if !out.status.success() {
        crate::record_spawn_failure(app, &label, Some(&out), None, started);
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(crate::tail(stderr.trim(), 1000));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Append a non-spawn event (bad signature, rollback) to launch-failures.log.
fn log_event(app: &tauri::AppHandle, command: &str, detail: &str) {
    let Ok(home) = crate::nellbrain_home() else { return };
    let runtime = crate::bundled_nell_path(app).ok().flatten();
    let _ = crate::launch_log::append_failure(
        &home,
        &crate::launch_log::LaunchFailure {
            command: command.to_string(),
            exit_code: None,
            stdout_tail: String::new(),
            stderr_tail: detail.to_string(),
            platform: std::env::consts::OS.to_string(),
            platform_version: std::env::consts::ARCH.to_string(),
            runtime_path_exists: runtime.as_ref().is_some_and(|p| p.exists()),
            bundled_runtime_path: runtime.map(|p| p.to_string_lossy().into_owned()),
            duration_ms: 0,
        },
    );
}

/// The downloaded bytes must match the verified manifest's sha256. A mismatch is
/// usually a publish in progress (new manifest, old file for a moment), so the
/// message says to check again; nothing is written or installed either way.
fn check_digest(name: &str, expected: &str, bytes: &[u8]) -> Result<(), String> {
    let got = sha256_hex(bytes);
    if got != expected {
        return Err(format!(
            "{name} didn't match its checksum (expected {expected}, got {got}) — if a new build was just published, check for updates again"
        ));
    }
    Ok(())
}

/// A declared Content-Length over the cap is refused before any body is read.
fn declared_too_big(content_length: Option<u64>, max: usize) -> bool {
    content_length.is_some_and(|n| n > max as u64)
}

/// Append `chunk` unless that would take `body` past `max` (then leave it as is).
fn append_capped(body: &mut Vec<u8>, chunk: &[u8], max: usize) -> bool {
    if body.len() + chunk.len() > max {
        return false;
    }
    body.extend_from_slice(chunk);
    true
}

/// GET `url` into memory, never holding more than `max` bytes: the declared
/// length is checked first, then the body is streamed with the cap enforced.
async fn fetch(client: &reqwest::Client, url: &str, max: usize) -> Result<Vec<u8>, String> {
    let too_big = || format!("{url}: larger than {max} bytes");
    let mut resp = client
        .get(url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("{url}: {e}"))?;
    if declared_too_big(resp.content_length(), max) {
        return Err(too_big());
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| format!("{url}: {e}"))? {
        if !append_capped(&mut body, &chunk, max) {
            return Err(too_big());
        }
    }
    Ok(body)
}

enum FetchError {
    Unreachable(String),
    BadSignature(String),
    Unsupported,
}

/// Fetch + verify manifest.json. A bad signature is retried once after 3 s —
/// manifest and .sig upload in parallel, so they can briefly disagree — then
/// logged as a security event. Never falls back to installing.
async fn fetch_verified_manifest(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
    tag: &str,
) -> Result<Manifest, FetchError> {
    let pubkey = updater_pubkey(&app.config().plugins.0).map_err(FetchError::Unreachable)?;
    let mut last = String::new();
    for attempt in 0..2 {
        if attempt == 1 {
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
        let manifest = fetch(client, &asset_url(tag, "manifest.json"), MANIFEST_MAX)
            .await
            .map_err(FetchError::Unreachable)?;
        let sig = fetch(client, &asset_url(tag, "manifest.json.sig"), SIG_MAX)
            .await
            .map_err(FetchError::Unreachable)?;
        let sig = String::from_utf8_lossy(&sig).into_owned();
        match verify_manifest(&manifest, &sig, &pubkey) {
            Ok(()) => {
                return parse_manifest(&manifest).map_err(|e| match e {
                    ManifestError::Unsupported => FetchError::Unsupported,
                    ManifestError::Invalid(e) => FetchError::Unreachable(e),
                });
            }
            Err(e) => last = e,
        }
    }
    log_event(app, "brain update: manifest signature rejected (security)", &last);
    Err(FetchError::BadSignature(last))
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| format!("http client: {e}"))
}

fn tag() -> String {
    resolve_tag(std::env::var(TAG_ENV).ok().as_deref())
}

#[tauri::command]
pub(crate) async fn check_brain_update(app: tauri::AppHandle) -> Result<BrainUpdateCheck, String> {
    if crate::bundled_nell_path(&app)?.is_none() {
        return Ok(BrainUpdateCheck::none("dev_build", None));
    }
    let client = http_client()?;
    let manifest = match fetch_verified_manifest(&app, &client, &tag()).await {
        Ok(m) => m,
        Err(FetchError::Unreachable(e)) => return Ok(BrainUpdateCheck::none("unreachable", Some(e))),
        Err(FetchError::BadSignature(e)) => return Ok(BrainUpdateCheck::none("bad_signature", Some(e))),
        Err(FetchError::Unsupported) => return Ok(BrainUpdateCheck::none("unsupported_manifest", None)),
    };
    let status = run_nell(&app, &["update", "--status"], STATUS_TIMEOUT_S).await?;
    if !status_supported(&status) {
        return Ok(BrainUpdateCheck::none("unsupported_install", None));
    }
    let bundle_version = app.package_info().version.to_string();
    Ok(decide(&manifest, active_commit_from_status(&status).as_deref(), &bundle_version))
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct BrainUpdateApplied {
    pub commit: String,
    pub brain_version: String,
    /// An overlay was active before this install: undo = rollback to it;
    /// otherwise undo = the release brain.
    pub had_active: bool,
}

/// Removed on drop, success or failure.
struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Re-fetch and re-verify (never trust the earlier check), download both assets,
/// check their sha256 against the verified manifest, then `nell update` installs
/// them into the overlay. The running bridge keeps its code until the frontend
/// restarts it.
#[tauri::command]
pub(crate) async fn apply_brain_update(app: tauri::AppHandle) -> Result<BrainUpdateApplied, String> {
    if crate::bundled_nell_path(&app)?.is_none() {
        return Err("brain updates need the installed app (this is a dev build)".into());
    }
    let client = http_client()?;
    let tag = tag();
    let m = match fetch_verified_manifest(&app, &client, &tag).await {
        Ok(m) => m,
        Err(FetchError::Unreachable(e)) => return Err(format!("couldn't reach the brain update: {e}")),
        Err(FetchError::BadSignature(_)) => return Err("the brain update's signature didn't verify".into()),
        Err(FetchError::Unsupported) => return Err("this brain update needs a newer app".into()),
    };
    let status = run_nell(&app, &["update", "--status"], STATUS_TIMEOUT_S).await?;
    let active = active_commit_from_status(&status);
    let verdict = decide(&m, active.as_deref(), &app.package_info().version.to_string());
    if !verdict.available {
        return Err(format!("no brain update to apply ({})", verdict.reason));
    }
    check_asset_names(&m)?;
    let dir = TempDir(std::env::temp_dir().join(format!(
        "ce-brain-update-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    )));
    std::fs::create_dir_all(&dir.0).map_err(|e| format!("temp dir: {e}"))?;
    for (asset, max) in [(&m.wheel, WHEEL_MAX), (&m.requirements, REQUIREMENTS_MAX)] {
        let bytes = fetch(&client, &asset_url(&tag, &asset.name), max).await?;
        check_digest(&asset.name, &asset.sha256, &bytes)?;
        std::fs::write(dir.0.join(&asset.name), &bytes).map_err(|e| format!("write {}: {e}", asset.name))?;
    }
    let wheel = dir.0.join(&m.wheel.name).to_string_lossy().into_owned();
    let req = dir.0.join(&m.requirements.name).to_string_lossy().into_owned();
    run_nell(
        &app,
        &["update", "--wheel", &wheel, "--requirements", &req, "--commit", &m.commit],
        INSTALL_TIMEOUT_S,
    )
    .await?;
    Ok(BrainUpdateApplied { commit: m.commit, brain_version: m.brain_version, had_active: active.is_some() })
}

/// The updated bridge was unhealthy (spec §6): make the previous overlay current
/// (or the release brain when there is none). The frontend restarts afterwards.
/// Returns the commit it landed on; None = the release brain (#335).
#[tauri::command]
pub(crate) async fn rollback_brain(app: tauri::AppHandle, reason: String) -> Result<Option<String>, String> {
    log_event(&app, "brain update: rolled back", &reason);
    let out = run_nell(&app, &["update", "--rollback"], FLIP_TIMEOUT_S).await?;
    Ok(active_commit_after_flip(&out))
}

/// "Use the release brain": clear the active overlay. The frontend restarts afterwards.
#[tauri::command]
pub(crate) async fn revert_brain(app: tauri::AppHandle) -> Result<(), String> {
    log_event(&app, "brain update: switched to the release brain", "");
    run_nell(&app, &["update", "--revert"], FLIP_TIMEOUT_S).await.map(|_| ())
}

/// The launch check (#335): is an overlay active, has it proven itself, how to undo it.
/// A dev build has no bundled nell and no overlay, so don't spawn anything.
#[tauri::command]
pub(crate) async fn brain_overlay_status(app: tauri::AppHandle) -> Result<BrainOverlayStatus, String> {
    if crate::bundled_nell_path(&app)?.is_none() {
        return Ok(overlay_status_from(""));
    }
    let status = run_nell(&app, &["update", "--status"], STATUS_TIMEOUT_S).await?;
    Ok(overlay_status_from(&status))
}

/// The bridge came back healthy on this commit: it has proven itself (#335).
#[tauri::command]
pub(crate) async fn confirm_brain_update(app: tauri::AppHandle, commit: String) -> Result<(), String> {
    run_nell(&app, &["update", "--confirm", &commit], FLIP_TIMEOUT_S).await.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &[u8] = include_bytes!("brain_update_fixtures/manifest.json");
    const SIG: &str = include_str!("brain_update_fixtures/manifest.json.sig");
    const PUB: &str = include_str!("brain_update_fixtures/fixture.pub");

    #[test]
    fn verifies_a_tauri_signed_manifest() {
        verify_manifest(MANIFEST, SIG, PUB).unwrap();
    }

    #[test]
    fn rejects_a_tampered_manifest() {
        let mut m = MANIFEST.to_vec();
        m.push(b' ');
        assert!(verify_manifest(&m, SIG, PUB).is_err());
    }

    #[test]
    fn rejects_the_apps_real_key_for_a_fixture_signature() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let real = conf["plugins"]["updater"]["pubkey"].as_str().unwrap();
        assert!(verify_manifest(MANIFEST, SIG, real).is_err());
    }

    const OTHER: &[u8] = include_bytes!("brain_update_fixtures/other.json");
    const OTHER_SIG: &str = include_str!("brain_update_fixtures/other.json.sig");

    #[test]
    fn rejects_a_signature_for_another_file() {
        let err = verify_manifest(OTHER, OTHER_SIG, PUB).unwrap_err();
        assert!(err.contains("another file"), "{err}");
        assert!(verify_manifest(MANIFEST, OTHER_SIG, PUB).is_err());
    }

    #[test]
    fn parse_manifest_checks_the_schema_first() {
        assert_eq!(parse_manifest(MANIFEST).unwrap().brain_version, "0.0.43");
        assert!(matches!(parse_manifest(br#"{"schema": 2, "commit": 7, "wheel": "x"}"#),
                         Err(ManifestError::Unsupported)));
        assert!(matches!(parse_manifest(br#"{"commit": "c"}"#), Err(ManifestError::Unsupported)));
        assert!(matches!(parse_manifest(br#"{"schema": "1"}"#), Err(ManifestError::Unsupported)));
        assert!(matches!(parse_manifest(br#"{"schema": 1, "commit": "c"}"#), Err(ManifestError::Invalid(_))));
        assert!(matches!(parse_manifest(b"not json"), Err(ManifestError::Invalid(_))));
    }

    #[test]
    fn rejects_garbage_encodings() {
        assert!(verify_manifest(MANIFEST, "not base64!", PUB).is_err());
        assert!(verify_manifest(MANIFEST, SIG, "not base64!").is_err());
    }

    fn manifest(brain: &str, min: &str) -> Manifest {
        Manifest {
            schema: 1,
            commit: "c".repeat(40),
            brain_version: brain.into(),
            built_at: "2026-09-28T12:00:00Z".into(),
            python: "3.13".into(),
            min_bundle_version: min.into(),
            wheel: Asset { name: format!("companion_emergence-{brain}-py3-none-any.whl"), sha256: "aa".into() },
            requirements: Asset { name: "requirements.txt".into(), sha256: "bb".into() },
        }
    }

    #[test]
    fn decision_table() {
        let c = "c".repeat(40);
        let r = |m: &Manifest, active: Option<&str>, bundle: &str| decide(m, active, bundle).reason;
        assert_eq!(r(&manifest("0.0.43", "0.0.43"), None, "0.0.43"), "available");
        assert_eq!(r(&manifest("0.0.44", "0.0.43"), None, "0.0.43"), "available");
        assert_eq!(r(&manifest("0.0.43", "0.0.43"), Some(&c), "0.0.43"), "already_active");
        assert_eq!(r(&manifest("0.0.42", "0.0.42"), None, "0.0.43"), "older_than_app");
        assert_eq!(r(&manifest("0.0.44", "0.0.44"), None, "0.0.43"), "app_too_old");
        assert_eq!(r(&manifest("x", "0.0.43"), None, "0.0.43"), "unparseable_version");
        let mut future = manifest("0.0.43", "0.0.43");
        future.schema = 2;
        assert_eq!(r(&future, None, "0.0.43"), "unsupported_manifest");
        assert!(decide(&manifest("0.0.43", "0.0.43"), None, "0.0.43").available);
        assert!(!decide(&manifest("0.0.42", "0.0.42"), None, "0.0.43").available);
    }

    #[test]
    fn parse_triple_takes_the_leading_numbers() {
        assert_eq!(parse_triple("0.0.43"), Some((0, 0, 43)));
        assert_eq!(parse_triple("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_triple("0.0.28a1"), Some((0, 0, 28)));
        assert_eq!(parse_triple("0.0"), None);
        assert_eq!(parse_triple("x.y.z"), None);
    }

    #[test]
    fn manifest_ignores_unknown_fields() {
        let mut v: serde_json::Value = serde_json::from_slice(MANIFEST).unwrap();
        v["channel"] = "brain-main".into();
        let m: Manifest = serde_json::from_value(v).unwrap();
        assert_eq!(m.brain_version, "0.0.43");
    }

    #[test]
    fn asset_names_are_checked() {
        let good = manifest("0.0.43", "0.0.43");
        check_asset_names(&good).unwrap();
        for bad in ["../evil-py3-none-any.whl", "companion_emergence-0.0.43/../../x-py3-none-any.whl",
                    "companion_emergence-0.0.43-py3-none-any.exe", "other-0.0.43-py3-none-any.whl"] {
            let mut m = good.clone();
            m.wheel.name = bad.into();
            assert!(check_asset_names(&m).is_err(), "{bad}");
        }
        let mut m = good.clone();
        m.requirements.name = "../requirements.txt".into();
        assert!(check_asset_names(&m).is_err());
    }

    #[test]
    fn sha256_hex_is_lowercase_hex() {
        assert_eq!(sha256_hex(b"abc"),
                   "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn tag_override_is_limited_to_the_brain_main_family() {
        assert_eq!(resolve_tag(None), "brain-main");
        assert_eq!(resolve_tag(Some("brain-main-dryrun")), "brain-main-dryrun");
        assert_eq!(resolve_tag(Some("brain-main")), "brain-main");
        for bad in ["v0.0.42", "brain-main-", "brain-main-X", "brain-main-a/b", "latest-release"] {
            assert_eq!(resolve_tag(Some(bad)), "brain-main", "{bad}");
        }
        assert_eq!(asset_url("brain-main", "manifest.json"),
                   "https://github.com/hanamorix/companion-emergence/releases/download/brain-main/manifest.json");
    }

    #[test]
    fn reads_nell_update_status() {
        let s = r#"{"supported": true, "install_kind": "bundled", "bundle_id": "b",
                    "active": {"dir": "d", "commit": "abc", "brain_version": "0.0.43", "bundle_id": "b"},
                    "previous": null}"#;
        assert!(status_supported(s));
        assert_eq!(active_commit_from_status(s).as_deref(), Some("abc"));
        // an overlay built for another bundle never loads, so it isn't active
        let other = s.replace(r#""bundle_id": "b","#, r#""bundle_id": "other","#);
        assert_eq!(active_commit_from_status(&other), None);
        let no_top = s.replace(r#""bundle_id": "b","#, "");
        assert_eq!(active_commit_from_status(&no_top), None);
        let none = r#"{"supported": false, "install_kind": "source", "active": null, "previous": null}"#;
        assert!(!status_supported(none));
        assert_eq!(active_commit_from_status(none), None);
        assert!(!status_supported("garbage"));
    }

    #[test]
    fn reads_where_a_rollback_landed() {
        // `nell update --rollback` prints the new state: an overlay, or none (the release brain)
        let landed = r#"{"active": {"dir": "d", "commit": "abc", "bundle_id": "b"}, "previous": null}"#;
        assert_eq!(active_commit_after_flip(landed).as_deref(), Some("abc"));
        assert_eq!(active_commit_after_flip(r#"{"active": null, "previous": {"commit": "x"}}"#), None);
        assert_eq!(active_commit_after_flip("garbage"), None);
    }

    #[test]
    fn reads_the_overlay_status_for_launch() {
        // confirmed/undo sit before bundle_id so the bundle swap below only hits the top level
        let s = r#"{"supported": true, "install_kind": "bundled", "bundle_id": "b",
                    "active": {"dir": "d", "commit": "abc", "confirmed": false, "undo": "rollback",
                               "brain_version": "0.0.43", "bundle_id": "b"},
                    "previous": null}"#;
        let none = BrainOverlayStatus { active_commit: None, confirmed: true, undo: "revert".into() };
        assert_eq!(
            overlay_status_from(s),
            BrainOverlayStatus { active_commit: Some("abc".into()), confirmed: false, undo: "rollback".into() }
        );
        // written before #335: no flag → proven; no undo → the release brain (the floor)
        let old = r#"{"bundle_id": "b", "active": {"commit": "abc", "bundle_id": "b"}, "previous": null}"#;
        assert_eq!(
            overlay_status_from(old),
            BrainOverlayStatus { active_commit: Some("abc".into()), confirmed: true, undo: "revert".into() }
        );
        // odd values never auto-drop a build: treated as proven, undo → revert
        let odd = r#"{"bundle_id": "b", "active": {"commit": "abc", "confirmed": "no", "undo": "sideways", "bundle_id": "b"}, "previous": null}"#;
        assert_eq!(
            overlay_status_from(odd),
            BrainOverlayStatus { active_commit: Some("abc".into()), confirmed: true, undo: "revert".into() }
        );
        // another bundle's overlay never loads; garbage / dev builds: no overlay
        let other = s.replacen(r#""bundle_id": "b","#, r#""bundle_id": "other","#, 1);
        assert_eq!(overlay_status_from(&other), none);
        assert_eq!(overlay_status_from(""), none);
    }

    #[test]
    fn reads_the_updater_pubkey_from_plugin_config() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let plugins: HashMap<String, serde_json::Value> =
            serde_json::from_value(conf["plugins"].clone()).unwrap();
        let key = updater_pubkey(&plugins).unwrap();
        assert!(minisign_verify::PublicKey::decode(&b64_text(&key, "k").unwrap()).is_ok());
        assert!(updater_pubkey(&HashMap::new()).is_err());
    }

    #[test]
    fn floor_command_disables_the_overlay() {
        let cmd = floor(std::process::Command::new("nell"), &["update", "--status"]);
        let std_cmd = cmd.as_std();
        let envs: Vec<_> = std_cmd.get_envs().collect();
        assert!(envs.contains(&(std::ffi::OsStr::new("KINDLED_NO_OVERLAY"), Some(std::ffi::OsStr::new("1")))));
        let args: Vec<_> = std_cmd.get_args().collect();
        assert_eq!(args, ["update", "--status"]);
    }

    #[test]
    fn a_declared_oversize_body_is_refused_before_reading() {
        assert!(declared_too_big(Some(11), 10));
        assert!(!declared_too_big(Some(10), 10));
        assert!(!declared_too_big(None, 10)); // no Content-Length: the streamed cap applies
    }

    #[test]
    fn the_streamed_cap_stops_at_max() {
        let mut body = Vec::new();
        assert!(append_capped(&mut body, b"12345", 10));
        assert!(append_capped(&mut body, b"67890", 10));
        assert!(!append_capped(&mut body, b"x", 10));
        assert_eq!(body, b"1234567890"); // the chunk that would overflow is not appended
    }

    #[test]
    fn a_checksum_mismatch_says_to_check_again() {
        let good = sha256_hex(b"wheel");
        assert!(check_digest("w.whl", &good, b"wheel").is_ok());
        let err = check_digest("w.whl", &good, b"other").unwrap_err();
        assert!(err.contains("w.whl") && err.contains("check for updates again"), "{err}");
    }
}
