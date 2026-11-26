//! Windows glue: write `nell.cmd` and add its folder to the user PATH (HKCU, no admin).

use std::path::Path;
use std::time::Duration;

use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE, REG_EXPAND_SZ, REG_SZ};
use winreg::{RegKey, RegValue};

use crate::{
    atomic_write, classify_existing, merge_user_path, read_existing, refuse_outcome,
    render_windows_cmd, Action, Changed, Desired, Outcome,
};

/// Append `dir` to the user PATH value under `HKCU\<subkey>` unless an entry already exists.
/// Production passes `"Environment"`; tests pass a scratch key. The original text stays an exact
/// prefix and the value type is preserved (a missing value is created as REG_EXPAND_SZ).
pub fn ensure_user_path(subkey: &str, dir: &str) -> Result<Changed, String> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let key = hkcu
        .open_subkey_with_flags(subkey, KEY_READ | KEY_WRITE)
        .map_err(|e| format!("cannot open HKCU\\{}: {}", subkey, e))?;
    let existing = match key.get_raw_value("Path") {
        Ok(v) => Some(v),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("cannot read the user PATH: {}", e)),
    };
    let (vtype, existing_text) = match existing {
        None => (REG_EXPAND_SZ, None),
        Some(v) => {
            let vtype = v.vtype.clone();
            if vtype != REG_SZ && vtype != REG_EXPAND_SZ {
                return Err("the user PATH is not a text value; leaving it untouched".to_string());
            }
            let bytes = v.bytes.to_vec();
            if bytes.len() % 2 != 0 {
                return Err("the user PATH value is not valid UTF-16; leaving it untouched".to_string());
            }
            let units: Vec<u16> = bytes.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let text = String::from_utf16(&units)
                .map_err(|_| "the user PATH value is not valid UTF-16; leaving it untouched".to_string())?;
            (vtype, Some(text.trim_end_matches('\0').to_string()))
        }
    };
    let lookup = |name: &str| std::env::var(name).ok();
    match merge_user_path(existing_text.as_deref(), dir, &lookup) {
        None => Ok(Changed::No),
        Some(new_text) => {
            let mut bytes = Vec::new();
            for unit in new_text.encode_utf16().chain(std::iter::once(0u16)) {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            key.set_raw_value("Path", &RegValue { bytes: bytes.into(), vtype })
                .map_err(|e| format!("cannot write the user PATH: {}", e))?;
            Ok(Changed::Yes)
        }
    }
}

/// Tell running programs the environment changed. The API timeout is per window, so the call runs
/// on its own thread and we wait a bounded time overall. Returns whether it completed in time.
fn broadcast_env_change() -> bool {
    use windows::Win32::Foundation::{LPARAM, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{
        SendMessageTimeoutW, HWND_BROADCAST, SMTO_ABORTIFHUNG, WM_SETTINGCHANGE,
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let wide: Vec<u16> = "Environment\0".encode_utf16().collect();
        let result = unsafe {
            SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                WPARAM(0),
                LPARAM(wide.as_ptr() as isize),
                SMTO_ABORTIFHUNG,
                2000,
                None,
            )
        };
        let _ = tx.send(result.0 != 0);
    });
    rx.recv_timeout(Duration::from_secs(5)).unwrap_or(false)
}

/// Write `nell.cmd` (never over a file that is not ours) and add its folder to the user PATH.
/// `subkey` is `"Environment"` in production.
pub fn install_windows_cmd(
    bundled: &Path,
    cmd_dir: &Path,
    cmd_path: &Path,
    win_prefixes: &[(String, String)],
    subkey: &str,
) -> Outcome {
    let text = match render_windows_cmd(&bundled.to_string_lossy(), win_prefixes) {
        Ok(t) => t,
        Err(e) => return Outcome::fail(78, e),
    };
    if let Err(e) = std::fs::create_dir_all(cmd_dir) {
        return Outcome::fail(1, format!("create {}: {}", cmd_dir.display(), e));
    }
    let action = classify_existing(&read_existing(cmd_path), &Desired::Cmd { text: text.clone() });
    if let Action::Refuse(_) = action {
        return refuse_outcome(cmd_path);
    }
    if action != Action::AlreadyCurrent {
        if let Err(e) = atomic_write(cmd_path, &text, false) {
            return Outcome::fail(1, format!("write {}: {}", cmd_path.display(), e));
        }
    }
    let dir_text = cmd_dir.to_string_lossy().into_owned();
    match ensure_user_path(subkey, &dir_text) {
        Err(e) => Outcome::fail(
            1,
            format!(
                "The shortcut file was written to {} but your user PATH could not be updated ({}). Add this folder to your PATH by hand: {}",
                cmd_path.display(),
                e,
                cmd_dir.display()
            ),
        ),
        Ok(changed) => {
            let mut out = format!(
                "{} runs {}\nOpen a new Command Prompt or PowerShell window and run `nell --version`.",
                cmd_path.display(),
                bundled.display()
            );
            if changed == Changed::Yes && !broadcast_env_change() {
                out.push_str("\nIf nell is not found, sign out and back in.");
            }
            Outcome::ok(out)
        }
    }
}

/// Launch-time refresh of `nell.cmd`: rewrite OUR file when stale; never create, never touch a
/// foreign file, never edit the PATH.
pub fn refresh_cmd_at(cmd_path: &Path, desired_text: &str) -> Outcome {
    let existing = read_existing(cmd_path);
    if !crate::refresh_decision(&existing, desired_text) {
        return Outcome::ok("unchanged".to_string());
    }
    match atomic_write(cmd_path, desired_text, false) {
        Ok(()) => Outcome::ok("refreshed".to_string()),
        Err(e) => Outcome::fail(1, format!("refresh {}: {}", cmd_path.display(), e)),
    }
}
