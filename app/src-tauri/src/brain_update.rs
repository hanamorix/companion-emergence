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

/// `nell update --status` → the active overlay's commit, if any.
pub(crate) fn active_commit_from_status(status: &str) -> Option<String> {
    status_json(status)?.get("active")?.get("commit")?.as_str().map(str::to_string)
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
        let s = r#"{"supported": true, "install_kind": "bundled",
                    "active": {"dir": "d", "commit": "abc", "brain_version": "0.0.43", "bundle_id": "b"},
                    "previous": null}"#;
        assert!(status_supported(s));
        assert_eq!(active_commit_from_status(s).as_deref(), Some("abc"));
        let none = r#"{"supported": false, "install_kind": "source", "active": null, "previous": null}"#;
        assert!(!status_supported(none));
        assert_eq!(active_commit_from_status(none), None);
        assert!(!status_supported("garbage"));
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
}
