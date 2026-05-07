//! NellFace — Tauri 2 backend.
//!
//! Commands the frontend invokes:
//!   - get_bridge_credentials(persona)  — read <persona>/bridge.json
//!   - read_app_config()                — read <NELLBRAIN_HOME>/app_config.json
//!   - write_app_config(config)         — write same
//!   - ensure_bridge_running(persona)   — spawn `nell supervisor start` if not live
//!   - run_init(args)                   — shell out to `uv run nell init` for the wizard
//!   - list_personas()                  — list <NELLBRAIN_HOME>/personas/* dirs
//!
//! Everything else (state polling, chat) is plain HTTP from the frontend
//! and goes via the bridge daemon (see brain/bridge/server.py).

use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BridgeCredentials {
    pub port: u16,
    pub auth_token: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AppConfig {
    /// Currently-selected persona; null on first launch (wizard fires).
    pub selected_persona: Option<String>,
    pub always_on_top: bool,
    pub reduced_motion: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitArgs {
    pub persona: String,
    pub user_name: Option<String>,
    pub voice_template: String,
    pub migrate_from: Option<String>,
    pub force: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Resolve <NELLBRAIN_HOME>, honoring the env var first, then
/// platformdirs (matches brain.paths).
fn nellbrain_home() -> Result<PathBuf, String> {
    if let Ok(home) = std::env::var("NELLBRAIN_HOME") {
        return Ok(PathBuf::from(home));
    }
    let base = dirs::data_dir()
        .ok_or_else(|| "could not resolve user data dir".to_string())?;
    Ok(base.join("companion-emergence"))
}

/// Mirror of brain.setup's persona-name validation. Anything outside
/// `[A-Za-z0-9_-]{1,40}` is rejected — defends against renderer
/// compromise constructing path-traversal segments before any
/// std::fs::read_to_string sees them.
fn validate_persona_name(persona: &str) -> Result<(), String> {
    if persona.is_empty() || persona.len() > 40 {
        return Err(format!(
            "persona name must be 1..=40 chars (got {} chars)",
            persona.len()
        ));
    }
    for c in persona.chars() {
        if !(c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(format!(
                "persona name {:?} contains invalid character {:?} \
                 (allowed: A-Za-z0-9_-)",
                persona, c
            ));
        }
    }
    Ok(())
}

fn persona_dir(persona: &str) -> Result<PathBuf, String> {
    validate_persona_name(persona)?;
    Ok(nellbrain_home()?.join("personas").join(persona))
}

fn app_config_path() -> Result<PathBuf, String> {
    Ok(nellbrain_home()?.join("app_config.json"))
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

#[tauri::command]
fn read_app_config() -> Result<AppConfig, String> {
    let path = app_config_path()?;
    if !path.exists() {
        return Ok(AppConfig::default());
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut cfg: AppConfig = serde_json::from_str(&raw)
        .map_err(|e| format!("parse app_config.json: {}", e))?;
    // A hand-edited or stale app_config.json with an invalid persona
    // name would otherwise feed straight into persona_dir() at the next
    // command call. Heal to None here so the wizard fires instead.
    if let Some(persona) = &cfg.selected_persona {
        if validate_persona_name(persona).is_err() {
            cfg.selected_persona = None;
        }
    }
    Ok(cfg)
}

#[tauri::command]
fn write_app_config(config: AppConfig) -> Result<(), String> {
    if let Some(persona) = &config.selected_persona {
        validate_persona_name(persona)?;
    }
    let path = app_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }
    let serialized = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("serialize: {}", e))?;
    std::fs::write(&path, serialized)
        .map_err(|e| format!("write {}: {}", path.display(), e))
}

#[tauri::command]
fn list_personas() -> Result<Vec<String>, String> {
    let dir = nellbrain_home()?.join("personas");
    if !dir.exists() {
        return Ok(vec![]);
    }
    let entries = std::fs::read_dir(&dir)
        .map_err(|e| format!("read {}: {}", dir.display(), e))?;
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        // Skip the `<name>.new` working dirs the migrator uses
        .filter(|n| !n.ends_with(".new"))
        // Skip `<name>.backup-*` archives the install-as flow leaves behind
        .filter(|n| !n.contains(".backup-"))
        // Skip dotfiles
        .filter(|n| !n.starts_with('.'))
        .collect();
    names.sort();
    Ok(names)
}

/// Resolve a Command that runs the `nell` CLI.
///
/// Production (Phase 7 bundle): app/src-tauri/python-runtime ships
/// inside Resources/. Path differs per OS because python-build-standalone
/// uses different layouts:
///   * macOS / Linux: Resources/python-runtime/bin/nell
///   * Windows:       Resources/python-runtime/Scripts/nell.exe
///
/// Dev (`pnpm tauri dev`): the bundled runtime usually isn't built,
/// so fall back to `uv run nell` against the source tree. Keeps the
/// dev iteration loop fast.
///
/// Resolution order:
///   1. Bundled per-OS path (production)
///   2. uv run nell (dev fallback)
fn nell_command(app: &tauri::AppHandle) -> Result<Command, String> {
    use tauri::Manager;
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resolve resource_dir: {}", e))?;
    let runtime_dir = resource_dir.join("python-runtime");
    let bundled = if cfg!(windows) {
        runtime_dir.join("Scripts").join("nell.exe")
    } else {
        runtime_dir.join("bin").join("nell")
    };
    if bundled.exists() {
        return Ok(Command::new(bundled));
    }
    // Dev fallback — uv on PATH against the source tree.
    let mut cmd = Command::new("uv");
    cmd.arg("run").arg("nell");
    Ok(cmd)
}

/// Quick liveness check for the bridge — calls /health with the persona's
/// own auth token. Returns true on 200.
async fn bridge_healthy(persona: &str) -> bool {
    let creds = match get_bridge_credentials(persona.to_string()) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let url = format!("http://127.0.0.1:{}/health", creds.port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let client = match client {
        Ok(c) => c,
        Err(_) => return false,
    };
    let mut req = client.get(&url);
    if let Some(token) = &creds.auth_token {
        req = req.header("Authorization", format!("Bearer {}", token));
    }
    matches!(req.send().await, Ok(r) if r.status().is_success())
}

#[tauri::command]
async fn ensure_bridge_running(
    app: tauri::AppHandle,
    persona: String,
) -> Result<(), String> {
    validate_persona_name(&persona)?;
    if bridge_healthy(&persona).await {
        return Ok(());
    }
    // Resolve nell via the bundled Python runtime (Phase 7) or the
    // uv-on-PATH dev fallback. Either way we spawn the supervisor's
    // start command — it blocks until /health returns 200, so when
    // the subprocess exits we can trust the bridge is up.
    let mut cmd = nell_command(&app)?;
    let output = cmd
        .args(["supervisor", "start", "--persona", &persona])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn nell supervisor start: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "supervisor start failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }
    Ok(())
}

#[tauri::command]
async fn run_init(app: tauri::AppHandle, args: InitArgs) -> Result<InitResult, String> {
    validate_persona_name(&args.persona)?;
    let mut cmd = nell_command(&app)?;
    cmd.arg("init");
    cmd.args(["--persona", &args.persona]);
    if let Some(name) = &args.user_name {
        cmd.args(["--user-name", name]);
    }
    cmd.args(["--voice-template", &args.voice_template]);
    if let Some(path) = &args.migrate_from {
        cmd.args(["--migrate-from", path]);
    } else {
        cmd.arg("--fresh");
    }
    if args.force {
        cmd.arg("--force");
    }
    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn nell init: {}", e))?;
    Ok(InitResult {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Apply the always-on-top toggle to the main window. Hits the Tauri
/// Manager API directly — the previous implementation only persisted
/// the bool to app_config.json without ever calling the window API,
/// so the UI showed a working toggle that didn't actually do anything
/// (audit 2026-05-07 P3).
#[tauri::command]
fn set_always_on_top(app: tauri::AppHandle, value: bool) -> Result<(), String> {
    use tauri::Manager;
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window not found".to_string())?;
    window
        .set_always_on_top(value)
        .map_err(|e| format!("set_always_on_top: {}", e))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_bridge_credentials,
            read_app_config,
            write_app_config,
            list_personas,
            ensure_bridge_running,
            run_init,
            set_always_on_top,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
