//! Shared helpers for unix-only tests (exec-bit handling, ETXTBSY retry).
#![cfg(unix)]
#![allow(dead_code)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};
use std::sync::Mutex;

/// Serialises tests whose oracle is the exec bit (a fork while another thread holds a
/// just-written script open can make exec fail with ETXTBSY, "Text file busy").
pub static EXEC_LOCK: Mutex<()> = Mutex::new(());

pub fn write_script(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

pub fn mode_of(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

fn is_text_busy(out: &Output) -> bool {
    String::from_utf8_lossy(&out.stderr).contains("Text file busy")
}

/// Run a command; retry up to 3 times ONLY when the failure is ETXTBSY (errno 26 or the shell's
/// "Text file busy" message). Exit 126 alone (permission denied) is never retried.
pub fn run_retry_busy(make: &dyn Fn() -> Command) -> Output {
    let mut last = None;
    for _ in 0..3 {
        match make().output() {
            Ok(out) if is_text_busy(&out) => last = Some(Ok(out)),
            Ok(out) => return out,
            Err(e) if e.raw_os_error() == Some(26) => last = Some(Err(e)),
            Err(e) => panic!("spawn failed: {e}"),
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    match last.unwrap() {
        Ok(out) => out,
        Err(e) => panic!("still ETXTBSY after retries: {e}"),
    }
}
