//! CLI mode: the AppImage shortcut runs the app binary with `--companion-nell-cli`, and `main()`
//! hands off to the bundled `nell` before any GUI code runs.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{CLI_FLAG, CWD_ENV, PRODUCT_NAME};

#[derive(Clone, Debug, PartialEq)]
pub enum CliMode {
    Gui,
    Nell(Vec<OsString>),
}

/// `args` are the process arguments AFTER the program name. Only a flag in FIRST position counts.
pub fn parse_cli_mode(args: &[OsString]) -> CliMode {
    match args.first() {
        Some(first) if first == OsStr::new(CLI_FLAG) => CliMode::Nell(args[1..].to_vec()),
        _ => CliMode::Gui,
    }
}

/// `<root>/usr/lib/<PRODUCT_NAME>/python-runtime/bin/nell` for an exe at `<root>/usr/bin/<name>`.
pub fn derive_cli_nell_path(exe: &Path) -> PathBuf {
    let usr = exe
        .parent()
        .and_then(|bin| bin.parent())
        .unwrap_or_else(|| Path::new("/"));
    usr.join("lib")
        .join(PRODUCT_NAME)
        .join("python-runtime")
        .join("bin")
        .join("nell")
}

/// `Some` only for an absolute path with at least one normal component (not "", not "/").
pub fn valid_appdir(raw: Option<&OsStr>) -> Option<PathBuf> {
    let raw = raw?;
    if raw.is_empty() {
        return None;
    }
    let p = PathBuf::from(raw);
    let normal = p
        .components()
        .filter(|c| matches!(c, std::path::Component::Normal(_)))
        .count();
    if p.is_absolute() && normal >= 1 {
        Some(p)
    } else {
        None
    }
}

/// Where the user was when they typed `nell`: the shim exports it, because AppRun changes the
/// working directory (and only sets OWD in FUSE mode).
pub fn resolve_cli_cwd(
    cwd_env: Option<&OsStr>,
    owd: Option<&OsStr>,
    appdir_set: bool,
) -> Option<PathBuf> {
    let usable = |s: &OsStr| {
        let p = PathBuf::from(s);
        if p.is_absolute() && p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    if let Some(p) = cwd_env.and_then(usable) {
        return Some(p);
    }
    if appdir_set {
        return owd.and_then(usable);
    }
    None
}

pub fn cwd_hint_needed(cwd: &Path, appdir: &Path) -> bool {
    cwd.starts_with(appdir)
}

/// Environment changes that undo what AppRun exported: entries under `appdir` are dropped from every
/// variable (the variable is removed when nothing is left, except PATH which keeps its original),
/// and AppRun's own bookkeeping variables are removed. `(name, None)` removes, `(name, Some(v))` sets.
pub fn scrub_apprun_env(
    vars: &[(OsString, OsString)],
    appdir: &Path,
) -> Vec<(OsString, Option<OsString>)> {
    if valid_appdir(Some(appdir.as_os_str())).is_none() {
        return Vec::new();
    }
    let always = ["APPDIR", "APPIMAGE", "ARGV0", "OWD"];
    let mut out = Vec::new();
    for (key, value) in vars {
        if always.iter().any(|a| OsStr::new(a) == key.as_os_str()) {
            out.push((key.clone(), None));
            continue;
        }
        let entries: Vec<PathBuf> = std::env::split_paths(value).collect();
        if !entries.iter().any(|e| e.starts_with(appdir)) {
            continue;
        }
        let kept: Vec<PathBuf> = entries.into_iter().filter(|e| !e.starts_with(appdir)).collect();
        if kept.is_empty() {
            if key.as_os_str() != OsStr::new("PATH") {
                out.push((key.clone(), None));
            }
        } else if let Ok(joined) = std::env::join_paths(kept) {
            out.push((key.clone(), Some(joined)));
        }
    }
    out
}

pub fn build_nell_command(
    nell: &Path,
    rest: &[OsString],
    cwd: Option<&Path>,
    env_changes: &[(OsString, Option<OsString>)],
) -> Command {
    let mut cmd = Command::new(nell);
    cmd.args(rest);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    for (key, value) in env_changes {
        match value {
            Some(v) => {
                cmd.env(key, v);
            }
            None => {
                cmd.env_remove(key);
            }
        }
    }
    cmd
}

/// First thing in `main()`: `None` means "not CLI mode, start the GUI". In CLI mode this never
/// returns on success (the bundled nell replaces this process); the value is an exit code on failure.
pub fn try_run_nell_cli() -> Option<i32> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match parse_cli_mode(&args) {
        CliMode::Gui => None,
        CliMode::Nell(rest) => Some(run_nell(rest)),
    }
}

#[cfg(unix)]
fn run_nell(rest: Vec<OsString>) -> i32 {
    use std::os::unix::process::CommandExt;

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("nell: cannot locate the Companion Emergence app: {}", e);
            return 127;
        }
    };
    let nell = derive_cli_nell_path(&exe);
    if !nell.is_file() {
        eprintln!(
            "nell: the bundled nell was not found at {} (unsupported app layout)",
            nell.display()
        );
        return 127;
    }
    let appdir = valid_appdir(std::env::var_os("APPDIR").as_deref());
    let cwd = resolve_cli_cwd(
        std::env::var_os(CWD_ENV).as_deref(),
        std::env::var_os("OWD").as_deref(),
        appdir.is_some(),
    );
    let mut changes: Vec<(OsString, Option<OsString>)> = vec![(OsString::from(CWD_ENV), None)];
    if let Some(dir) = &appdir {
        let vars: Vec<(OsString, OsString)> = std::env::vars_os().collect();
        changes.extend(scrub_apprun_env(&vars, dir));
        if cwd.is_none() {
            if let Ok(cur) = std::env::current_dir() {
                let canonical = dir.canonicalize().unwrap_or_else(|_| dir.clone());
                if cwd_hint_needed(&cur, &canonical) {
                    eprintln!("nell: working directory could not be restored; run nell through ~/.local/bin/nell");
                }
            }
        }
    }
    let err = build_nell_command(&nell, &rest, cwd.as_deref(), &changes).exec();
    eprintln!("nell: could not start {}: {}", nell.display(), err);
    126
}

#[cfg(not(unix))]
fn run_nell(_rest: Vec<OsString>) -> i32 {
    eprintln!("nell: the app's CLI mode is not available on this platform");
    2
}
