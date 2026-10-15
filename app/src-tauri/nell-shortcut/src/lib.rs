//! Decision logic and file/registry glue for the `nell` CLI shortcut (#408).
//!
//! This crate deliberately has NO Tauri dependency: a `cargo test` binary of a
//! Tauri-linked crate does not start on Windows (upstream tauri #13419), and the
//! shortcut logic needs tests on every OS.
//!
//! TESTS-FIRST STATE: every function below is a stub (`unimplemented!()` or a
//! placeholder constant) so the test suite compiles and fails for the right
//! reason. The implementation lands in the next commit.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

pub const CLI_FLAG: &str = "--companion-nell-cli";
pub const CWD_ENV: &str = "COMPANION_NELL_CWD";
pub const SHIM_MARKER: &str = "companion-emergence-nell-shim";
pub const PRODUCT_NAME: &str = "Companion Emergence";
pub const NO_FUSE_MESSAGE: &str = "TODO";
pub const SAFE_LOCATION_MESSAGE: &str = "TODO";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    MacOs,
    Linux,
    Windows,
    Other,
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
        _os: Os,
        _lookup: &dyn Fn(&str) -> Option<OsString>,
        _bundled: Option<PathBuf>,
        _macos_unstable_reason: Option<String>,
    ) -> ShortcutEnv {
        unimplemented!()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ShortcutPlan {
    Symlink { bundled: PathBuf, target_dir: PathBuf, target: PathBuf },
    AppImageScript { appimage: PathBuf, target_dir: PathBuf, target: PathBuf },
    WindowsCmd { bundled: PathBuf, cmd_dir: PathBuf, cmd_path: PathBuf, win_prefixes: Vec<(String, String)> },
    Unavailable { reason: String },
}

pub fn plan_shortcut(_env: &ShortcutEnv) -> ShortcutPlan {
    unimplemented!()
}

pub fn has_temp_mount_component(_p: &Path) -> bool {
    unimplemented!()
}

/// None unless `raw` is an absolute path with at least one normal component.
pub fn valid_appdir(_raw: Option<&OsStr>) -> Option<PathBuf> {
    unimplemented!()
}

pub fn render_appimage_script(_appimage: &Path) -> Result<String, String> {
    unimplemented!()
}

/// Fill the `{NELL_BAT}` placeholder of the template and normalise to CRLF.
pub fn fill_cmd_template(_template: &str, _nell_bat_expr: &str) -> String {
    unimplemented!()
}

pub fn render_windows_cmd(_nell_bat: &str, _prefixes: &[(String, String)]) -> Result<String, String> {
    unimplemented!()
}

/// `Some(new_path_value)` when `dir` must be appended, `None` when already present.
pub fn merge_user_path(
    _existing: Option<&str>,
    _dir: &str,
    _lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    unimplemented!()
}

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

pub fn classify_existing(_existing: &Existing, _desired: &Desired) -> Action {
    unimplemented!()
}

pub fn refresh_decision(_existing: &Existing, _desired_text: &str) -> bool {
    unimplemented!()
}

pub fn path_hint_text(_os: Os, _on_path: bool) -> String {
    unimplemented!()
}

#[derive(Clone, Debug, PartialEq)]
pub enum CliMode {
    Gui,
    Nell(Vec<OsString>),
}

/// `args` are the process arguments AFTER the program name.
pub fn parse_cli_mode(_args: &[OsString]) -> CliMode {
    unimplemented!()
}

/// `<exe>/../../lib/<PRODUCT_NAME>/python-runtime/bin/nell`, lexically.
pub fn derive_cli_nell_path(_exe: &Path) -> PathBuf {
    unimplemented!()
}

pub fn resolve_cli_cwd(
    _cwd_env: Option<&OsStr>,
    _owd: Option<&OsStr>,
    _appdir_set: bool,
) -> Option<PathBuf> {
    unimplemented!()
}

/// (name, Some(new value)) = set, (name, None) = remove.
pub fn scrub_apprun_env(
    _vars: &[(OsString, OsString)],
    _appdir: &Path,
) -> Vec<(OsString, Option<OsString>)> {
    unimplemented!()
}

pub fn cwd_hint_needed(_cwd: &Path, _appdir: &Path) -> bool {
    unimplemented!()
}

pub fn build_nell_command(
    _nell: &Path,
    _rest: &[OsString],
    _cwd: Option<&Path>,
    _env_changes: &[(OsString, Option<OsString>)],
) -> std::process::Command {
    unimplemented!()
}

/// Entry point called first thing in `main()`. `None` = not CLI mode, start the GUI.
pub fn try_run_nell_cli() -> Option<i32> {
    None
}

#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

#[cfg(unix)]
pub fn install_script_shim(
    _target_dir: &Path,
    _target: &Path,
    _appimage: &Path,
    _appimage_exists: bool,
    _nell_path: &Path,
) -> Outcome {
    unimplemented!()
}

/// Rewrite an existing OUR-shim at `target` when stale or missing its exec bit. Never creates.
#[cfg(unix)]
pub fn refresh_shim_at(_target: &Path, _desired_text: &str) -> Outcome {
    unimplemented!()
}

/// Linux deb path: refuse foreign files/symlinks (exit 73), remove an "ours" shim file.
#[cfg(unix)]
pub fn prepare_linux_symlink(_target: &Path, _bundled: &Path) -> Result<(), Outcome> {
    unimplemented!()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Changed {
    Yes,
    No,
}

#[cfg(windows)]
pub fn ensure_user_path(_subkey: &str, _dir: &str) -> Result<Changed, String> {
    unimplemented!()
}

#[cfg(windows)]
pub fn install_windows_cmd(
    _bundled: &Path,
    _cmd_dir: &Path,
    _cmd_path: &Path,
    _win_prefixes: &[(String, String)],
    _subkey: &str,
) -> Outcome {
    unimplemented!()
}
