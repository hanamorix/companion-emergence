//! Unix-style path rows: C4 (symlink rows), C10 (non-UTF-8), C11, C12, C24, C26, C2/C2b.
#![cfg(unix)]

mod common;

use common::*;
use nell_shortcut::*;
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn p(s: &str) -> PathBuf {
    PathBuf::from(s)
}

// ------------------------------------------------------------ C4: ownership with symlinks

const OURS: &str = "/usr/lib/Companion Emergence/python-runtime/bin/nell";

#[test]
fn symlink_mechanism_rows() {
    let d = Desired::Symlink { bundled: p(OURS) };
    assert_eq!(classify_existing(&Existing::Missing, &d), Action::Create);
    assert_eq!(classify_existing(&Existing::Symlink(Some(p(OURS))), &d), Action::AlreadyCurrent);
    assert_eq!(
        classify_existing(&Existing::Symlink(Some(p("/opt/old/python-runtime/bin/nell"))), &d),
        Action::Replace,
        "an older bundled location is ours"
    );
    match classify_existing(&Existing::Symlink(Some(p("/home/u/venv/bin/nell"))), &d) {
        Action::Refuse(_) => {}
        other => panic!("a user's own symlink must be refused, got {other:?}"),
    }
    match classify_existing(&Existing::Symlink(None), &d) {
        Action::Refuse(_) => {}
        other => panic!("unreadable symlink must be refused, got {other:?}"),
    }
    let shim = format!("#!/bin/sh\n# {} v1\n", SHIM_MARKER);
    assert_eq!(
        classify_existing(&Existing::File { text: Some(shim), exec_bit: true }, &d),
        Action::Replace,
        "AppImage shim -> deb symlink switch"
    );
    match classify_existing(&Existing::File { text: Some("#!/bin/sh\necho mine\n".to_string()), exec_bit: true }, &d) {
        Action::Refuse(_) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn script_mechanism_replaces_our_symlink_but_not_a_foreign_one() {
    let d = Desired::Script { text: format!("#!/bin/sh\n# {} v1\n", SHIM_MARKER) };
    assert_eq!(classify_existing(&Existing::Symlink(Some(p(OURS))), &d), Action::Replace);
    match classify_existing(&Existing::Symlink(Some(p("/home/u/venv/bin/nell"))), &d) {
        Action::Refuse(_) => {}
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------ C10: non-UTF-8 arguments

#[test]
fn cli_mode_passes_non_utf8_arguments_verbatim() {
    let odd = OsStr::from_bytes(&[0x66, 0x6f, 0x80, 0x6f]).to_os_string();
    let args = vec![OsString::from("--companion-nell-cli"), odd.clone()];
    assert_eq!(parse_cli_mode(&args), CliMode::Nell(vec![odd]));
}

// ------------------------------------------------------------ C12: layout derivation

#[test]
fn derive_cli_nell_path_follows_the_tauri_layout() {
    assert_eq!(
        derive_cli_nell_path(Path::new("/tmp/.mount_x/usr/bin/nellface")),
        p("/tmp/.mount_x/usr/lib/Companion Emergence/python-runtime/bin/nell")
    );
    assert_eq!(
        derive_cli_nell_path(Path::new("/usr/bin/nellface")),
        p("/usr/lib/Companion Emergence/python-runtime/bin/nell")
    );
}

// ------------------------------------------------------------ C26: cwd

#[test]
fn resolve_cli_cwd_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let d = dir.as_os_str();
    assert_eq!(resolve_cli_cwd(Some(d), None, false), Some(dir.clone()));
    assert_eq!(resolve_cli_cwd(Some(OsStr::new("relative")), Some(d), true), Some(dir.clone()), "falls to OWD when APPDIR is set");
    assert_eq!(resolve_cli_cwd(Some(OsStr::new("relative")), Some(d), false), None, "OWD only counts under an AppImage");
    assert_eq!(resolve_cli_cwd(Some(OsStr::new("/definitely/not/here")), None, true), None);
    assert_eq!(resolve_cli_cwd(None, None, true), None);
    assert_eq!(resolve_cli_cwd(None, None, false), None);
}

#[test]
fn cwd_hint_fires_only_when_cwd_is_inside_the_appdir() {
    assert!(cwd_hint_needed(Path::new("/tmp/.mount_X/usr"), Path::new("/tmp/.mount_X")));
    assert!(!cwd_hint_needed(Path::new("/home/u"), Path::new("/tmp/.mount_X")));
    assert!(!cwd_hint_needed(Path::new("/tmp/.mount_XY"), Path::new("/tmp/.mount_X")));
}

// ------------------------------------------------------------ C24: environment scrub

fn vars(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
    pairs.iter().map(|(k, v)| (OsString::from(*k), OsString::from(*v))).collect()
}

fn changes(v: Vec<(OsString, Option<OsString>)>) -> BTreeMap<String, Option<String>> {
    v.into_iter()
        .map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|s| s.to_string_lossy().into_owned())))
        .collect()
}

#[test]
fn scrub_removes_apprun_values_and_keeps_the_users_own_entries() {
    let a = "/tmp/.mount_X";
    let env = vars(&[
        ("APPDIR", a),
        ("PYTHONHOME", "/tmp/.mount_X/usr/"),
        ("PYTHONPATH", "/tmp/.mount_X/usr/share/pyshared/:/home/u/mine"),
        ("PATH", "/tmp/.mount_X/usr/bin:/tmp/.mount_X/bin:/usr/bin:/bin"),
        ("LD_LIBRARY_PATH", "/tmp/.mount_X/usr/lib:/opt/x"),
        ("GST_PLUGIN_SYSTEM_PATH", "/tmp/.mount_X/usr/lib/gstreamer"),
        ("GSETTINGS_SCHEMA_DIR", "/tmp/.mount_X/usr/share/glib-2.0/schemas"),
        ("GTK_PATH", "/tmp/.mount_X/usr/lib/gtk"),
        ("GIO_EXTRA_MODULES", "/tmp/.mount_X/usr/lib/gio/modules"),
        ("GTK_IM_MODULE_FILE", "/tmp/.mount_X/usr/lib/immodules.cache"),
        ("GDK_BACKEND", "x11"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("OWD", "/home/u"),
        ("ARGV0", "x"),
        ("APPIMAGE", "/home/u/CE.AppImage"),
        ("HOME", "/home/u"),
    ]);
    let got = changes(scrub_apprun_env(&env, Path::new(a)));
    let mut want: BTreeMap<String, Option<String>> = BTreeMap::new();
    for k in [
        "APPDIR", "PYTHONHOME", "GST_PLUGIN_SYSTEM_PATH", "GSETTINGS_SCHEMA_DIR", "GTK_PATH",
        "GIO_EXTRA_MODULES", "GTK_IM_MODULE_FILE", "OWD", "ARGV0", "APPIMAGE",
    ] {
        want.insert(k.to_string(), None);
    }
    want.insert("PYTHONPATH".to_string(), Some("/home/u/mine".to_string()));
    want.insert("PATH".to_string(), Some("/usr/bin:/bin".to_string()));
    want.insert("LD_LIBRARY_PATH".to_string(), Some("/opt/x".to_string()));
    assert_eq!(got, want);
}

#[test]
fn scrub_never_removes_path_and_keeps_values_outside_the_appdir() {
    let env = vars(&[
        ("PATH", "/tmp/.mount_X/usr/bin:/tmp/.mount_X/bin"),
        ("PYTHONHOME", "/opt/pyhome"),
        ("XDG_DATA_DIRS", "/tmp/.mount_XY/share:/usr/share"),
    ]);
    let got = changes(scrub_apprun_env(&env, Path::new("/tmp/.mount_X")));
    assert!(!got.contains_key("PATH"), "a PATH with only AppImage entries keeps its original value");
    assert!(!got.contains_key("PYTHONHOME"), "a PYTHONHOME outside the appdir is untouched");
    assert!(!got.contains_key("XDG_DATA_DIRS"), "/tmp/.mount_XY is not under /tmp/.mount_X (component-wise)");
}

#[test]
fn scrub_with_a_degenerate_appdir_changes_nothing() {
    let env = vars(&[("PATH", "/usr/bin"), ("PYTHONHOME", "/usr")]);
    assert_eq!(valid_appdir(Some(OsStr::new(""))), None);
    assert_eq!(valid_appdir(Some(OsStr::new("/"))), None);
    // the dispatcher only scrubs with a valid appdir; the pure fn must also be inert for "/"
    assert!(scrub_apprun_env(&env, Path::new("/")).is_empty());
    assert!(scrub_apprun_env(&env, Path::new("")).is_empty());
}

// ------------------------------------------------------------ C11: hand-off command

#[test]
fn build_nell_command_carries_args_cwd_and_env_changes() {
    let rest = vec![OsString::from("a"), OsString::from("b c"), OsString::from("")];
    let envc = vec![
        (OsString::from("KEEP"), Some(OsString::from("1"))),
        (OsString::from("DROP"), None),
    ];
    let cmd = build_nell_command(Path::new("/x/nell"), &rest, Some(Path::new("/tmp")), &envc);
    assert_eq!(cmd.get_program(), OsStr::new("/x/nell"));
    let args: Vec<&OsStr> = cmd.get_args().collect();
    assert_eq!(args, vec![OsStr::new("a"), OsStr::new("b c"), OsStr::new("")]);
    assert_eq!(cmd.get_current_dir(), Some(Path::new("/tmp")));
    let envs: BTreeMap<String, Option<String>> = cmd
        .get_envs()
        .map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|s| s.to_string_lossy().into_owned())))
        .collect();
    assert_eq!(envs.get("KEEP"), Some(&Some("1".to_string())));
    assert_eq!(envs.get("DROP"), Some(&None));
}

#[test]
fn build_nell_command_runs_a_fake_nell_and_passes_the_exit_code() {
    let _g = EXEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let nell = tmp.path().join("nell");
    write_script(
        &nell,
        "#!/bin/sh\nfor a in \"$@\"; do printf 'ARG=%s\\n' \"$a\"; done\nprintf 'PWD=%s\\n' \"$(pwd -P)\"\nprintf 'KEEP=%s\\n' \"${KEEP-unset}\"\nexit 9\n",
    );
    let work = tempfile::tempdir().unwrap();
    let rest = vec![OsString::from("a"), OsString::from("b c"), OsString::from("")];
    let envc = vec![(OsString::from("KEEP"), Some(OsString::from("1")))];
    let out = run_retry_busy(&|| build_nell_command(&nell, &rest, Some(work.path()), &envc));
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(out.status.code(), Some(9), "{stdout}");
    let want_pwd = work.path().canonicalize().unwrap();
    assert_eq!(
        stdout,
        format!("ARG=a\nARG=b c\nARG=\nPWD={}\nKEEP=1\n", want_pwd.display())
    );
}

// ------------------------------------------------------------ C2 / C2b: AppImage shim under real sh

fn fake_appimage_body() -> &'static str {
    "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done\nprintf 'CWDENV=%s\\n' \"${COMPANION_NELL_CWD-unset}\"\nprintf 'EXTRACT=%s\\n' \"${APPIMAGE_EXTRACT_AND_RUN-unset}\"\nexit 7\n"
}

#[test]
fn shim_text_basics() {
    let t = render_appimage_script(Path::new("/home/u/Apps/CE.AppImage")).expect("render");
    assert!(t.starts_with("#!/bin/sh\n"), "{t}");
    assert!(t.contains(SHIM_MARKER));
    assert!(t.contains("--companion-nell-cli \"$@\""));
    assert!(t.contains("COMPANION_NELL_CWD"));
    assert!(!t.contains("APPIMAGE_EXTRACT_AND_RUN"), "no-FUSE runs never get a shim (R2-1)");
    assert!(render_appimage_script(Path::new("/a\nb/CE.AppImage")).is_err());
}

#[test]
fn shim_runs_the_appimage_with_args_cwd_and_exit_code() {
    let _g = EXEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let apps = tmp.path().join("it's a $dir");
    let app = apps.join("CE.AppImage");
    write_script(&app, fake_appimage_body());
    let shim_text = render_appimage_script(&app).expect("render");
    let shim = tmp.path().join("nell");
    write_script(&shim, &shim_text);
    let work = tempfile::tempdir().unwrap();
    let out = run_retry_busy(&|| {
        let mut c = Command::new("sh");
        c.arg(&shim).args(["a", "b c", "--appimage-help", ""]).current_dir(work.path());
        c
    });
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(out.status.code(), Some(7), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let want_cwd = work.path().canonicalize().unwrap();
    assert_eq!(
        stdout,
        format!(
            "--companion-nell-cli\na\nb c\n--appimage-help\n\nCWDENV={}\nEXTRACT=unset\n",
            want_cwd.display()
        )
    );
}

#[test]
fn shim_reports_a_missing_appimage_with_exit_127() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("gone.AppImage");
    let shim = tmp.path().join("nell");
    write_script(&shim, &render_appimage_script(&missing).expect("render"));
    let out = Command::new("sh").arg(&shim).output().unwrap();
    assert_eq!(out.status.code(), Some(127));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stderr.contains("gone.AppImage"), "{stderr}");
}
