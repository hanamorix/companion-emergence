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
    let base = dirs::data_dir().ok_or_else(|| "could not resolve user data dir".to_string())?;
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
    // Audit 2026-05-07 P2-8: was `port_u64 as u16` which silently
    // wraps anything > 65535. A corrupt or hand-edited bridge.json
    // could then make the renderer hand the bridge token to an
    // unrelated local service. try_from rejects out-of-range and 0.
    let port_u64 = parsed
        .get("port")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "bridge.json missing port".to_string())?;
    if port_u64 == 0 {
        return Err("bridge.json port is 0".to_string());
    }
    let port = u16::try_from(port_u64)
        .map_err(|_| format!("bridge.json port {} out of u16 range", port_u64))?;
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
    let raw =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut cfg: AppConfig =
        serde_json::from_str(&raw).map_err(|e| format!("parse app_config.json: {}", e))?;
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
    let serialized =
        serde_json::to_string_pretty(&config).map_err(|e| format!("serialize: {}", e))?;
    std::fs::write(&path, serialized).map_err(|e| format!("write {}: {}", path.display(), e))
}

#[tauri::command]
fn list_personas() -> Result<Vec<String>, String> {
    let dir = nellbrain_home()?.join("personas");
    if !dir.exists() {
        return Ok(vec![]);
    }
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("read {}: {}", dir.display(), e))?;
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
/// Return the bundled production `nell` entry point when it exists.
fn bundled_nell_path(app: &tauri::AppHandle) -> Result<Option<PathBuf>, String> {
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
    Ok(if bundled.exists() {
        Some(bundled)
    } else {
        None
    })
}

/// Resolution order:
///   1. Bundled per-OS path (production)
///   2. uv run nell (dev fallback)
fn nell_command(app: &tauri::AppHandle) -> Result<Command, String> {
    if let Some(bundled) = bundled_nell_path(app)? {
        return Ok(Command::new(bundled));
    }
    // Dev fallback — uv on PATH against the source tree.
    let mut cmd = Command::new("uv");
    cmd.arg("run").arg("nell");
    Ok(cmd)
}

fn unstable_macos_app_path_reason(path: &std::path::Path) -> Option<String> {
    let text = path.to_string_lossy();
    if text.starts_with("/Volumes/") {
        return Some(
            "Move Companion Emergence to /Applications before installing the launchd service; the bundled runtime is currently under /Volumes.".to_string(),
        );
    }
    if text.contains("/AppTranslocation/") {
        return Some(
            "Move Companion Emergence to /Applications and relaunch before installing the launchd service; macOS is running this app from an AppTranslocation path.".to_string(),
        );
    }
    None
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
async fn ensure_bridge_running(app: tauri::AppHandle, persona: String) -> Result<(), String> {
    validate_persona_name(&persona)?;
    if bridge_healthy(&persona).await {
        return Ok(());
    }
    // Resolve nell via the bundled Python runtime (Phase 7) or the
    // uv-on-PATH dev fallback. Either way we spawn the supervisor's
    // start command — it blocks until /health returns 200, so when
    // the subprocess exits we can trust the bridge is up.
    //
    // Audit 2026-05-07 P2-9: 60s timeout. Without a bound, a hung
    // Python init / supervisor stall would freeze the Tauri renderer
    // through `await ensureBridgeRunning(...)` indefinitely. 60s is
    // generous against typical 5-15s startup; if it triggers, the
    // user gets `supervisor_start_timeout` instead of a stuck UI.
    let std_cmd = nell_command(&app)?;
    let mut cmd = tokio::process::Command::from(std_cmd);
    cmd.args(["supervisor", "start", "--persona", &persona])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let output = match tokio::time::timeout(std::time::Duration::from_secs(60), cmd.output()).await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(format!("spawn nell supervisor start: {}", e)),
        Err(_) => {
            return Err("supervisor_start_timeout".to_string());
        }
    };
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
    let std_cmd = nell_command(&app)?;
    let mut cmd = tokio::process::Command::from(std_cmd);
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
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Audit 2026-05-07 P2-9: 30s timeout — nell init creates a
    // persona dir, writes config + voice.md. Typical run is <2s.
    // First-run deadlocks would leave the wizard's installing step
    // forever-spinning; the timeout surfaces `init_timeout`.
    let output = match tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output()).await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(format!("spawn nell init: {}", e)),
        Err(_) => return Err("init_timeout".to_string()),
    };
    Ok(InitResult {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Install the launchd LaunchAgent for the persona's supervisor.
///
/// Runs ``nell service install --persona <name>``. Idempotent — if the
/// service is already installed, the CLI rewrites the plist and
/// kickstarts. Designed to be called from the wizard's StepInstalling
/// after ``nell init`` succeeds, so first-launch users get the
/// supervisor under launchd's lifecycle from the very first run rather
/// than the legacy Tauri-spawn-then-detach model.
///
/// Returns the same shape as ``run_init`` (success bool, stdout, stderr,
/// exit_code) so the wizard can surface failures inline. Non-zero exit
/// is reported but does NOT block the wizard transition: install can be
/// retried from the connection panel later if it failed for a transient
/// reason. macOS-only: non-Darwin platforms get an explicit unsupported
/// result so the UI can avoid implying that a persistent service was installed.
#[tauri::command]
async fn install_supervisor_service(
    app: tauri::AppHandle,
    persona: String,
) -> Result<InitResult, String> {
    validate_persona_name(&persona)?;
    if !cfg!(target_os = "macos") {
        return Ok(InitResult {
            success: false,
            stdout: String::new(),
            stderr: "persistent supervisor service install is currently macOS-only".to_string(),
            exit_code: 78,
        });
    }
    let bundled = bundled_nell_path(&app)?;
    if let Some(path) = &bundled {
        if let Some(reason) = unstable_macos_app_path_reason(path) {
            return Ok(InitResult {
                success: false,
                stdout: String::new(),
                stderr: reason,
                exit_code: 78,
            });
        }
    }
    let std_cmd = if let Some(path) = &bundled {
        Command::new(path)
    } else {
        nell_command(&app)?
    };
    let mut cmd = tokio::process::Command::from(std_cmd);
    cmd.args(["service", "install", "--persona", &persona]);
    if let Some(path) = &bundled {
        cmd.arg("--nell-path").arg(path);
    }
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // 30s budget — install writes a plist (cheap) + bootstraps via
    // launchctl (~1-3s). The hard cap is here so a wedged launchctl
    // can't hang the wizard indefinitely.
    let output = match tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output()).await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(format!("spawn nell service install: {}", e)),
        Err(_) => return Err("service_install_timeout".to_string()),
    };
    Ok(InitResult {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Result of probing the host for Anthropic's ``claude`` CLI.
///
/// Powers the wizard's prerequisites step: when ``found`` is false the
/// wizard surfaces an install-instructions panel and blocks ``Continue``
/// until a re-check passes. ``path`` is the resolved absolute path on
/// success so the user can see exactly which binary was picked
/// (helpful when multiple installs collide).
#[derive(Debug, Serialize)]
pub struct ClaudeCliCheck {
    pub found: bool,
    pub path: Option<String>,
    pub version: Option<String>,
}

#[tauri::command]
async fn check_claude_cli() -> Result<ClaudeCliCheck, String> {
    // Anthropic's installer puts the binary at ``~/.local/bin/claude``,
    // which launchd / Finder-launched processes don't see by default
    // because they inherit a stripped PATH. Probe a small set of known
    // locations PLUS try a bare ``claude`` lookup so Homebrew /
    // ``/usr/local`` installs are still found when the user's shell
    // PATH happens to be inherited.
    let candidates: Vec<PathBuf> = {
        let mut v: Vec<PathBuf> = Vec::new();
        if let Ok(home) = std::env::var("HOME") {
            v.push(PathBuf::from(&home).join(".local/bin/claude"));
        }
        v.push(PathBuf::from("/opt/homebrew/bin/claude"));
        v.push(PathBuf::from("/usr/local/bin/claude"));
        if cfg!(target_os = "windows") {
            if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
                v.push(
                    PathBuf::from(local_app_data)
                        .join("Programs")
                        .join("claude")
                        .join("claude.exe"),
                );
            }
        }
        v
    };
    let mut resolved: Option<PathBuf> = None;
    for cand in &candidates {
        if cand.is_file() {
            resolved = Some(cand.clone());
            break;
        }
    }
    // Fallback: ask the OS to resolve ``claude`` directly. tokio uses
    // the inherited PATH; if the shell rc happened to seed it (the
    // app was launched from terminal), this can succeed even when
    // the path probe missed.
    if resolved.is_none() {
        if let Ok(out) = tokio::process::Command::new("claude")
            .arg("--version")
            .output()
            .await
        {
            if out.status.success() {
                let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
                return Ok(ClaudeCliCheck {
                    found: true,
                    path: None,
                    version: if version.is_empty() {
                        None
                    } else {
                        Some(version)
                    },
                });
            }
        }
        return Ok(ClaudeCliCheck {
            found: false,
            path: None,
            version: None,
        });
    }
    let path = resolved.unwrap();
    let version = match tokio::process::Command::new(&path)
        .arg("--version")
        .output()
        .await
    {
        Ok(out) if out.status.success() => {
            let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    };
    Ok(ClaudeCliCheck {
        found: true,
        path: Some(path.to_string_lossy().to_string()),
        version,
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
            install_supervisor_service,
            check_claude_cli,
            set_always_on_top,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ────────────────────────────────────────────────────────────────────────────
// Tests — Audit 2026-05-07 P4-3
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_persona_name_accepts_canonical() {
        assert!(validate_persona_name("nell").is_ok());
        assert!(validate_persona_name("test_persona").is_ok());
        assert!(validate_persona_name("test-persona-2").is_ok());
        assert!(validate_persona_name("a").is_ok()); // single char ok
        assert!(validate_persona_name(&"x".repeat(40)).is_ok()); // exact upper bound
    }

    #[test]
    fn validate_persona_name_rejects_traversal() {
        assert!(validate_persona_name("../etc").is_err());
        assert!(validate_persona_name("../../escape").is_err());
        assert!(validate_persona_name("..").is_err());
        assert!(validate_persona_name("nell/sub").is_err());
        assert!(validate_persona_name("nell\\back").is_err());
    }

    #[test]
    fn validate_persona_name_rejects_size_bounds() {
        assert!(validate_persona_name("").is_err());
        assert!(validate_persona_name(&"x".repeat(41)).is_err());
        assert!(validate_persona_name(&"x".repeat(100)).is_err());
    }

    #[test]
    fn validate_persona_name_rejects_special_chars() {
        assert!(validate_persona_name("nell.dot").is_err());
        assert!(validate_persona_name("nell space").is_err());
        assert!(validate_persona_name("nell;semi").is_err());
        assert!(validate_persona_name("nell$dollar").is_err());
        assert!(validate_persona_name("nell!bang").is_err());
    }

    // Audit P2-8: port parsing must reject out-of-range and zero.
    // u16::try_from is what get_bridge_credentials uses; we test the
    // boundary semantics here so a future refactor doesn't regress.
    #[test]
    fn port_parse_accepts_valid_u16() {
        assert!(u16::try_from(1u64).is_ok());
        assert!(u16::try_from(8080u64).is_ok());
        assert!(u16::try_from(65535u64).is_ok());
    }

    #[test]
    fn port_parse_rejects_above_u16_max() {
        assert!(u16::try_from(65536u64).is_err());
        assert!(u16::try_from(100_000u64).is_err());
    }

    #[test]
    fn unstable_macos_app_path_detects_dmg_and_translocation() {
        let dmg = std::path::Path::new(
            "/Volumes/Companion Emergence/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell",
        );
        assert!(unstable_macos_app_path_reason(dmg)
            .unwrap()
            .contains("/Applications"));

        let translocated = std::path::Path::new(
            "/private/var/folders/xx/AppTranslocation/ABC/d/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell",
        );
        assert!(unstable_macos_app_path_reason(translocated)
            .unwrap()
            .contains("AppTranslocation"));

        let stable = std::path::Path::new(
            "/Applications/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell",
        );
        assert!(unstable_macos_app_path_reason(stable).is_none());
    }
}
