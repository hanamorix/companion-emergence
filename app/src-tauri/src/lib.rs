//! NellFace — Tauri 2 backend.
//!
//! Single command surface: `get_bridge_credentials` reads
//! `<NELLBRAIN_HOME>/personas/<persona>/bridge.json` and returns
//! the port + auth_token. Everything else (state polling, chat) is
//! plain HTTP from the frontend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BridgeCredentials {
    pub port: u16,
    pub auth_token: Option<String>,
}

/// Resolve the persona dir using the same logic as `brain.paths.get_persona_dir`:
/// honor `NELLBRAIN_HOME` env var, else fall back to platformdirs.
fn persona_dir(persona: &str) -> Result<PathBuf, String> {
    if let Ok(home) = std::env::var("NELLBRAIN_HOME") {
        return Ok(PathBuf::from(home).join("personas").join(persona));
    }
    // platformdirs default — matches brain.paths on macOS
    let base = dirs::data_dir()
        .ok_or_else(|| "could not resolve user data dir".to_string())?;
    Ok(base.join("companion-emergence").join("personas").join(persona))
}

#[tauri::command]
fn get_bridge_credentials(persona: String) -> Result<BridgeCredentials, String> {
    let dir = persona_dir(&persona)?;
    let bridge_json = dir.join("bridge.json");
    let raw = std::fs::read_to_string(&bridge_json)
        .map_err(|e| format!("read {}: {}", bridge_json.display(), e))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse bridge.json: {}", e))?;
    let port_u64 = parsed
        .get("port")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "bridge.json missing port".to_string())?;
    let port = port_u64 as u16;
    let auth_token = parsed
        .get("auth_token")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok(BridgeCredentials { port, auth_token })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        
        .invoke_handler(tauri::generate_handler![get_bridge_credentials])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
