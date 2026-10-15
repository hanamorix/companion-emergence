//! C25 / C26 / C11: the REAL `nellface` binary in CLI mode (unix only).
//!
//! Chain under test: main -> dispatcher -> current_exe -> derive layout -> restore cwd -> scrub
//! the AppRun environment -> exec the bundled nell. The fake tree is built from the same layout
//! rule the dispatcher uses, so this guards the CHAIN, not the layout itself (the layout is
//! evidenced by upstream Tauri source and, for real, by the release-time AppImage smoke).
//!
//! One sequential #[test]: no concurrent forks, no ETXTBSY between cases. If the flag ever
//! reaches the GUI, the child is killed and reaped on timeout instead of leaking a window.
#![cfg(unix)]

use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Guard(Option<Child>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn put_binary(dest: &Path) {
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    let src = PathBuf::from(env!("CARGO_BIN_EXE_nellface"));
    if std::fs::hard_link(&src, dest).is_err() {
        std::fs::copy(&src, dest).unwrap();
    }
}

fn fake_nell(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let body = r#"#!/bin/sh
printf 'PWD=%s\n' "$(pwd -P)"
for a in "$@"; do printf 'ARG=%s\n' "$a"; done
printf 'PYTHONHOME=%s\n' "${PYTHONHOME-unset}"
printf 'APPDIR=%s\n' "${APPDIR-unset}"
printf 'CWDENV=%s\n' "${COMPANION_NELL_CWD-unset}"
case ":$PATH:" in *":/tmp/.mount_testX/usr/bin:"*) echo PATH_LEAK ;; *) echo PATH_CLEAN ;; esac
exit "${FAKE_EXIT:-9}"
"#;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn run_with_timeout(mut cmd: Command, secs: u64) -> (Option<i32>, String) {
    cmd.stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null());
    let child = cmd.spawn().expect("spawn nellface copy");
    let mut guard = Guard(Some(child));
    let start = Instant::now();
    loop {
        let done = guard.0.as_mut().unwrap().try_wait().unwrap();
        if let Some(status) = done {
            let mut out = String::new();
            if let Some(mut so) = guard.0.as_mut().unwrap().stdout.take() {
                let _ = so.read_to_string(&mut out);
            }
            return (status.code(), out);
        }
        if start.elapsed() > Duration::from_secs(secs) {
            panic!("nellface did not exit within {secs}s in CLI mode: the flag reached the GUI");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn cli_mode_hands_off_to_the_bundled_nell_with_args_cwd_scrubbed_env_and_exit_code() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let exe = root.join("usr").join("bin").join("nellface");
    put_binary(&exe);
    fake_nell(&root.join("usr/lib/Companion Emergence/python-runtime/bin/nell"));
    let work = tempfile::tempdir().unwrap();
    let work_dir = work.path().canonicalize().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();

    let path_env = format!("/tmp/.mount_testX/usr/bin:{}", std::env::var("PATH").unwrap_or_default());
    let mut cmd = Command::new(&exe);
    cmd.args(["--companion-nell-cli", "a", "b c", ""])
        .current_dir(elsewhere.path())
        .env("COMPANION_NELL_CWD", &work_dir)
        .env("APPDIR", "/tmp/.mount_testX")
        .env("PYTHONHOME", "/tmp/.mount_testX/usr/")
        .env("PATH", path_env);
    let (code, stdout) = run_with_timeout(cmd, 20);

    assert_eq!(code, Some(9), "stdout: {stdout}");
    assert_eq!(
        stdout,
        format!(
            "PWD={}\nARG=a\nARG=b c\nARG=\nPYTHONHOME=unset\nAPPDIR=unset\nCWDENV=unset\nPATH_CLEAN\n",
            work_dir.display()
        )
    );

    // exit code passthrough with another status
    let mut cmd = Command::new(&exe);
    cmd.args(["--companion-nell-cli"]).current_dir(elsewhere.path()).env("FAKE_EXIT", "3");
    let (code, _) = run_with_timeout(cmd, 20);
    assert_eq!(code, Some(3));
}
