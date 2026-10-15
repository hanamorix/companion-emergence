//! Decision logic and file/registry glue for the `nell` CLI shortcut (#408).
//!
//! This crate deliberately has NO Tauri dependency: a `cargo test` binary of a
//! Tauri-linked crate does not start on Windows (upstream tauri #13419), and the
//! shortcut logic needs tests on every OS.
//!
//! Layout: pure decision and text functions live here and compile everywhere;
//! file, registry and exec glue is split into `glue_unix` and `glue_windows`.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

mod cli_mode;
#[cfg(unix)]
mod glue_unix;
#[cfg(windows)]
mod glue_windows;

pub use cli_mode::{
    build_nell_command, cwd_hint_needed, derive_cli_nell_path, parse_cli_mode, resolve_cli_cwd,
    scrub_apprun_env, try_run_nell_cli, valid_appdir, CliMode,
};
#[cfg(unix)]
pub use glue_unix::{install_script_shim, prepare_linux_symlink, refresh_shim_at};
#[cfg(windows)]
pub use glue_windows::{ensure_user_path, install_windows_cmd, refresh_cmd_at};

pub const CLI_FLAG: &str = "--companion-nell-cli";
pub const CWD_ENV: &str = "COMPANION_NELL_CWD";
pub const SHIM_MARKER: &str = "companion-emergence-nell-shim";
pub const PRODUCT_NAME: &str = "Companion Emergence";

/// Shown when the running AppImage has no FUSE (extract-and-run mode). Owner-ruled: the installer
/// must say what the problem is and what to do. Plain words, no dashes. Pinned by tests.
pub const NO_FUSE_MESSAGE: &str = "This AppImage is running without FUSE (in extract-and-run mode), so the nell shortcut was not installed. A shortcut would re-extract the whole app on every nell command and delete the running app's files when the command ends. To get the shortcut, install FUSE for your distribution (on Debian and Ubuntu the package is libfuse2, called libfuse2t64 on Ubuntu 24.04), start the AppImage normally, and try again. Or install the .deb package instead, which does not need FUSE.";

/// Shown when APPIMAGE and APPDIR are set but the bundled files are not under APPDIR although the
/// path is a temporary mount (a spelling mismatch or a foreign layout).
pub const SAFE_LOCATION_MESSAGE: &str = "The app's files are not inside the AppImage's own folder, so a nell shortcut cannot be made safely. Start the AppImage again in the normal way and retry. If this keeps happening, install the .deb package instead.";

const TEMPLATE: &str = include_str!("nell_cmd.tmpl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    MacOs,
    Linux,
    Windows,
    Other,
}

pub fn current_os() -> Os {
    if cfg!(target_os = "macos") {
        Os::MacOs
    } else if cfg!(target_os = "linux") {
        Os::Linux
    } else if cfg!(windows) {
        Os::Windows
    } else {
        Os::Other
    }
}

/// What the lib.rs glue knows about the running app, as plain data.
#[derive(Clone, Debug, PartialEq)]
pub struct ShortcutEnv {
    pub os: Os,
    pub bundled: Option<PathBuf>,
    pub home: Option<PathBuf>,
    pub local_app_data: Option<PathBuf>,
    pub appimage: Option<PathBuf>,
    pub appdir: Option<PathBuf>,
    pub macos_unstable_reason: Option<String>,
    /// (variable name, value) pairs for LOCALAPPDATA, ProgramW6432, ProgramFiles, in that order.
    pub win_prefixes: Vec<(String, String)>,
}

impl ShortcutEnv {
    pub fn from_lookup(
        os: Os,
        lookup: &dyn Fn(&str) -> Option<OsString>,
        bundled: Option<PathBuf>,
        macos_unstable_reason: Option<String>,
    ) -> ShortcutEnv {
        let nonempty = |key: &str| lookup(key).filter(|v| !v.is_empty());
        // macOS keeps the old `std::env::var("HOME")` behaviour exactly, an empty HOME included.
        let home = if os == Os::MacOs {
            lookup("HOME").map(PathBuf::from)
        } else {
            nonempty("HOME").map(PathBuf::from)
        };
        let mut win_prefixes = Vec::new();
        for name in ["LOCALAPPDATA", "ProgramW6432", "ProgramFiles"] {
            if let Some(v) = nonempty(name) {
                win_prefixes.push((name.to_string(), v.to_string_lossy().into_owned()));
            }
        }
        ShortcutEnv {
            os,
            bundled,
            home,
            local_app_data: nonempty("LOCALAPPDATA").map(PathBuf::from),
            appimage: nonempty("APPIMAGE").map(PathBuf::from),
            appdir: valid_appdir(nonempty("APPDIR").as_deref()),
            macos_unstable_reason,
            win_prefixes,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ShortcutPlan {
    Symlink { bundled: PathBuf, target_dir: PathBuf, target: PathBuf },
    AppImageScript { appimage: PathBuf, target_dir: PathBuf, target: PathBuf },
    WindowsCmd { bundled: PathBuf, cmd_dir: PathBuf, cmd_path: PathBuf, win_prefixes: Vec<(String, String)> },
    Unavailable { reason: String },
}

fn unavailable(reason: &str) -> ShortcutPlan {
    ShortcutPlan::Unavailable { reason: reason.to_string() }
}

/// A component of the path that marks a temporary AppImage location: a FUSE mount
/// (`.mount_*`) or an extract-and-run directory (`appimage_extracted_*`).
pub fn has_temp_mount_component(p: &Path) -> bool {
    p.components().any(|c| match c {
        std::path::Component::Normal(s) => {
            let s = s.to_string_lossy();
            s.starts_with(".mount_") || s.starts_with("appimage_extracted_")
        }
        _ => false,
    })
}

fn symlink_plan(bundled: &Path, home: &Path) -> ShortcutPlan {
    let target_dir = home.join(".local").join("bin");
    let target = target_dir.join("nell");
    ShortcutPlan::Symlink { bundled: bundled.to_path_buf(), target_dir, target }
}

pub fn plan_shortcut(env: &ShortcutEnv) -> ShortcutPlan {
    if env.os == Os::Other {
        return unavailable("The nell shortcut is not supported on this platform.");
    }
    let bundled = match &env.bundled {
        Some(b) => b,
        None => {
            return unavailable(match env.os {
                Os::MacOs => "Bundled nell CLI not found in this build. Install from a release DMG.",
                Os::Linux => "Bundled nell CLI not found in this build. Install from the .deb or AppImage release.",
                _ => "Bundled nell CLI not found in this build. Install from the release installer.",
            })
        }
    };
    match env.os {
        Os::MacOs => {
            if let Some(reason) = &env.macos_unstable_reason {
                return ShortcutPlan::Unavailable { reason: reason.clone() };
            }
            match &env.home {
                Some(home) => symlink_plan(bundled, home),
                None => unavailable("HOME env var not set; cannot resolve ~/.local/bin"),
            }
        }
        Os::Linux => plan_linux(env, bundled),
        Os::Windows => match &env.local_app_data {
            Some(lad) => {
                let cmd_dir = lad.join("CompanionEmergence").join("bin");
                let cmd_path = cmd_dir.join("nell.cmd");
                ShortcutPlan::WindowsCmd {
                    bundled: bundled.clone(),
                    cmd_dir,
                    cmd_path,
                    win_prefixes: env.win_prefixes.clone(),
                }
            }
            None => unavailable("LOCALAPPDATA is not set, so the nell shortcut folder cannot be placed."),
        },
        Os::Other => unavailable("The nell shortcut is not supported on this platform."),
    }
}

fn plan_linux(env: &ShortcutEnv, bundled: &Path) -> ShortcutPlan {
    let under_appdir = env.appdir.as_ref().map_or(false, |d| bundled.starts_with(d));
    if let (Some(appimage), Some(appdir)) = (&env.appimage, &env.appdir) {
        if under_appdir {
            if !appimage.is_absolute() {
                return unavailable("The AppImage file path is not available, so a nell shortcut cannot point at it.");
            }
            let extracted = appdir
                .file_name()
                .map_or(false, |n| n.to_string_lossy().starts_with("appimage_extracted_"));
            if extracted {
                return unavailable(NO_FUSE_MESSAGE);
            }
            return match &env.home {
                Some(home) => {
                    let target_dir = home.join(".local").join("bin");
                    let target = target_dir.join("nell");
                    ShortcutPlan::AppImageScript { appimage: appimage.clone(), target_dir, target }
                }
                None => unavailable("HOME env var not set; cannot resolve ~/.local/bin"),
            };
        }
        if has_temp_mount_component(bundled) {
            return unavailable(SAFE_LOCATION_MESSAGE);
        }
    }
    if has_temp_mount_component(bundled) {
        return unavailable("The app is running from a temporary AppImage location, so a nell shortcut would point at files that disappear when the app closes. Start the AppImage normally, or install the .deb package.");
    }
    match &env.home {
        Some(home) => symlink_plan(bundled, home),
        None => unavailable("HOME env var not set; cannot resolve ~/.local/bin"),
    }
}

// ---------------------------------------------------------------------------- shim texts

/// POSIX sh script for `~/.local/bin/nell` that runs the AppImage in CLI mode.
pub fn render_appimage_script(appimage: &Path) -> Result<String, String> {
    let s = appimage
        .to_str()
        .ok_or_else(|| "the AppImage path is not valid UTF-8".to_string())?;
    if s.contains('\n') || s.contains('\0') {
        return Err("the AppImage path contains a line break".to_string());
    }
    let quoted = s.replace('\'', "'\\''");
    Ok(format!(
        "#!/bin/sh\n\
         # {marker} v1: written by Companion Emergence, refreshed on every app launch.\n\
         APPIMAGE_FILE='{quoted}'\n\
         [ -f \"$APPIMAGE_FILE\" ] || {{ echo \"nell: $APPIMAGE_FILE not found. Open Companion Emergence once to refresh this shortcut.\" >&2; exit 127; }}\n\
         COMPANION_NELL_CWD=\"$PWD\"; export COMPANION_NELL_CWD\n\
         exec \"$APPIMAGE_FILE\" --companion-nell-cli \"$@\"\n",
        marker = SHIM_MARKER,
        quoted = quoted
    ))
}

/// Fill the `{NELL_BAT}` placeholder and normalise every line ending to exactly CRLF.
pub fn fill_cmd_template(template: &str, nell_bat_expr: &str) -> String {
    template
        .replace("\r\n", "\n")
        .replace("{NELL_BAT}", nell_bat_expr)
        .replace('\n', "\r\n")
}

fn escape_percent(s: &str) -> String {
    s.replace('%', "%%")
}

/// `Some(rest)` when `s` starts with `prefix` ignoring case, and `rest` begins at a backslash.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let prefix = prefix.trim_end_matches('\\');
    let mut chars = s.char_indices();
    for pc in prefix.chars() {
        let (_, c) = chars.next()?;
        if !c.to_lowercase().eq(pc.to_lowercase()) {
            return None;
        }
    }
    let rest = match chars.next() {
        Some((i, _)) => &s[i..],
        None => return None,
    };
    if rest.starts_with('\\') {
        Some(rest)
    } else {
        None
    }
}

/// Text of `nell.cmd`. ASCII only (cmd reads a script in the OEM code page, #261): a non-ASCII
/// install prefix is written as a `%VAR%` reference, which cmd expands as Unicode at run time.
pub fn render_windows_cmd(nell_bat: &str, prefixes: &[(String, String)]) -> Result<String, String> {
    if nell_bat.contains('"') || nell_bat.chars().any(|c| c.is_control()) {
        return Err("the install path contains a character that cannot be used in nell.cmd".to_string());
    }
    let expr = if nell_bat.is_ascii() {
        escape_percent(nell_bat)
    } else {
        let mut found = None;
        for (name, value) in prefixes {
            if let Some(rest) = strip_prefix_ci(nell_bat, value) {
                if !rest.is_ascii() {
                    return Err(non_ascii_error());
                }
                found = Some(format!("%{}%{}", name, escape_percent(rest)));
                break;
            }
        }
        found.ok_or_else(non_ascii_error)?
    };
    Ok(fill_cmd_template(TEMPLATE, &expr))
}

fn non_ascii_error() -> String {
    "The install path contains characters that cmd.exe cannot read from a script, and it is not under a standard folder. Reinstall Companion Emergence to a folder with a plain name, or run nell from its install folder.".to_string()
}

// ---------------------------------------------------------------------------- user PATH

fn expand_vars(s: &str, lookup: &dyn Fn(&str) -> Option<String>) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(start) = rest.find('%') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('%') {
            Some(end) => {
                let name = &after[..end];
                match lookup(name) {
                    Some(v) if !name.is_empty() => out.push_str(&v),
                    _ => {
                        out.push('%');
                        out.push_str(name);
                        out.push('%');
                    }
                }
                rest = &after[end + 1..];
            }
            None => {
                out.push('%');
                out.push_str(after);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

fn normalize_path_entry(seg: &str, lookup: &dyn Fn(&str) -> Option<String>) -> String {
    let t = seg.trim().trim_matches('"').trim();
    let expanded = expand_vars(t, lookup).replace('/', "\\");
    expanded.trim_end_matches('\\').to_lowercase()
}

/// `Some(new_value)` when `dir` must be appended, `None` when an entry for it already exists.
/// The original value is never re-serialised: it stays an exact prefix of the new one.
pub fn merge_user_path(
    existing: Option<&str>,
    dir: &str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    let target = normalize_path_entry(dir, lookup);
    if let Some(e) = existing {
        if e.split(';').any(|seg| normalize_path_entry(seg, lookup) == target) {
            return None;
        }
    }
    Some(match existing {
        None => dir.to_string(),
        Some("") => dir.to_string(),
        Some(e) if e.ends_with(';') => format!("{}{}", e, dir),
        Some(e) => format!("{};{}", e, dir),
    })
}

// ---------------------------------------------------------------------------- ownership

#[derive(Clone, Debug, PartialEq)]
pub enum Existing {
    Missing,
    Symlink(Option<PathBuf>),
    /// `text` is None when the file is unreadable or not UTF-8.
    File { text: Option<String>, exec_bit: bool },
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Desired {
    Symlink { bundled: PathBuf },
    Script { text: String },
    Cmd { text: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Create,
    AlreadyCurrent,
    Replace,
    Refuse(String),
}

fn has_marker(text: &str) -> bool {
    text.lines().take(3).any(|l| l.contains(SHIM_MARKER))
}

/// A symlink is ours when it points at a bundled runtime's `python-runtime/bin/nell`.
fn is_our_symlink(target: &Path) -> bool {
    target.ends_with(Path::new("python-runtime").join("bin").join("nell"))
}

fn refuse() -> Action {
    Action::Refuse("it is not a Companion Emergence shortcut".to_string())
}

pub fn classify_existing(existing: &Existing, desired: &Desired) -> Action {
    match (desired, existing) {
        (_, Existing::Missing) => Action::Create,
        (_, Existing::Other) => refuse(),
        (_, Existing::File { text: None, .. }) => refuse(),
        (_, Existing::Symlink(None)) => refuse(),
        (Desired::Symlink { bundled }, Existing::Symlink(Some(t))) => {
            if t == bundled {
                Action::AlreadyCurrent
            } else if is_our_symlink(t) {
                Action::Replace
            } else {
                refuse()
            }
        }
        (Desired::Symlink { .. }, Existing::File { text: Some(t), .. }) => {
            if has_marker(t) {
                Action::Replace
            } else {
                refuse()
            }
        }
        (Desired::Script { .. }, Existing::Symlink(Some(t))) => {
            if is_our_symlink(t) {
                Action::Replace
            } else {
                refuse()
            }
        }
        (Desired::Script { text }, Existing::File { text: Some(t), exec_bit }) => {
            if !has_marker(t) {
                refuse()
            } else if t == text && *exec_bit {
                Action::AlreadyCurrent
            } else {
                Action::Replace
            }
        }
        (Desired::Cmd { .. }, Existing::Symlink(_)) => refuse(),
        (Desired::Cmd { text }, Existing::File { text: Some(t), .. }) => {
            if !has_marker(t) {
                refuse()
            } else if t == text {
                Action::AlreadyCurrent
            } else {
                Action::Replace
            }
        }
    }
}

/// Refresh on launch only when our own shim exists and is stale or lost its exec bit.
pub fn refresh_decision(existing: &Existing, desired_text: &str) -> bool {
    match existing {
        Existing::File { text: Some(t), exec_bit } => has_marker(t) && (t != desired_text || !*exec_bit),
        _ => false,
    }
}

/// Hint printed after a successful install. The macOS strings are byte-identical to the original.
pub fn path_hint_text(os: Os, on_path: bool) -> String {
    match (os, on_path) {
        (Os::MacOs, true) => "Open a new Terminal window and run `nell --version`.".to_string(),
        (Os::MacOs, false) => "If `nell` isn't found in Terminal, add `export PATH=\"$HOME/.local/bin:$PATH\"` to your shell rc (~/.zshrc) and reload.".to_string(),
        (_, true) => "Open a new terminal and run `nell --version`.".to_string(),
        (_, false) => "If `nell` isn't found, add `export PATH=\"$HOME/.local/bin:$PATH\"` to ~/.profile (or ~/.bashrc) and sign out and back in. Some distributions only add ~/.local/bin to PATH at login when the folder already existed.".to_string(),
    }
}

// ---------------------------------------------------------------------------- shared glue helpers

#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl Outcome {
    pub fn ok(stdout: String) -> Outcome {
        Outcome { success: true, stdout, stderr: String::new(), exit_code: 0 }
    }
    pub fn fail(exit_code: i32, stderr: String) -> Outcome {
        Outcome { success: false, stdout: String::new(), stderr, exit_code }
    }
}

pub(crate) fn refuse_outcome(target: &Path) -> Outcome {
    Outcome::fail(
        73,
        format!(
            "Refusing to overwrite {}: it is not a Companion Emergence shortcut. Move or delete it (likely a pip/uv-installed nell) and try again.",
            target.display()
        ),
    )
}

/// What is at `path` right now, without following symlinks.
pub(crate) fn read_existing(path: &Path) -> Existing {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(_) => return Existing::Missing,
    };
    if meta.file_type().is_symlink() {
        return Existing::Symlink(std::fs::read_link(path).ok());
    }
    if meta.is_file() {
        let text = if meta.len() > 1_048_576 {
            None
        } else {
            std::fs::read(path).ok().and_then(|b| String::from_utf8(b).ok())
        };
        #[cfg(unix)]
        let exec_bit = {
            use std::os::unix::fs::PermissionsExt;
            meta.permissions().mode() & 0o111 != 0
        };
        #[cfg(not(unix))]
        let exec_bit = true;
        return Existing::File { text, exec_bit };
    }
    Existing::Other
}

/// Write `text` to `target` through a uniquely named temp file in the same directory, then rename.
pub(crate) fn atomic_write(target: &Path, text: &str, executable: bool) -> std::io::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let name = target.file_name().unwrap_or_else(|| OsStr::new("nell")).to_string_lossy().into_owned();
    let tmp = dir.join(format!(".{}.tmp.{}.{}", name, std::process::id(), COUNTER.fetch_add(1, Ordering::SeqCst)));
    let result = (|| {
        std::fs::write(&tmp, text)?;
        if executable {
            set_executable(&tmp)?;
        }
        std::fs::rename(&tmp, target)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
fn set_executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

pub(crate) fn path_contains_dir(dir: &Path) -> bool {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|e| e == dir))
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Changed {
    Yes,
    No,
}
