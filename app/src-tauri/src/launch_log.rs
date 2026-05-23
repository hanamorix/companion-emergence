//! Structured spawn-failure diagnostics.
//!
//! When a Tauri-spawned CLI call (nell init / migrate / supervisor start /
//! service install) fails, we append a JSON line to
//! `$KINDLED_HOME/launch-failures.log`. This survives the UI dropping the
//! error (the v0.0.18 frontend now surfaces Tauri rejection strings, but a
//! durable on-disk record lets users attach context to bug reports — the
//! Lord Grim "undefined on Windows" case).
//!
//! Rotation: when the live log reaches 200 KB it's renamed to
//! `launch-failures.1.log` (one depth — last ~100 failures of context),
//! and a fresh live log starts.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const ROTATION_THRESHOLD: u64 = 200 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchFailure {
    pub command: String,
    pub exit_code: Option<i32>,
    pub stdout_tail: String,
    pub stderr_tail: String,
    pub platform: String,
    pub platform_version: String,
    pub bundled_runtime_path: Option<String>,
    pub runtime_path_exists: bool,
    pub duration_ms: u64,
}

pub fn log_path(kindled_home: &Path) -> PathBuf {
    kindled_home.join("launch-failures.log")
}

/// Append one failure as a JSON line. Best-effort: returns an io::Error on
/// disk problems so callers can choose to ignore it (a read-only home must
/// never crash a spawn path).
pub fn append_failure(kindled_home: &Path, failure: &LaunchFailure) -> std::io::Result<()> {
    fs::create_dir_all(kindled_home)?;
    let log = log_path(kindled_home);

    // Rotate if the live log has grown past the threshold.
    if let Ok(meta) = fs::metadata(&log) {
        if meta.len() >= ROTATION_THRESHOLD {
            let prev = kindled_home.join("launch-failures.1.log");
            let _ = fs::remove_file(&prev); // drop the oldest rotation
            fs::rename(&log, &prev)?;
        }
    }

    let mut entry = serde_json::to_value(failure).unwrap_or(serde_json::Value::Null);
    if let serde_json::Value::Object(ref mut map) = entry {
        map.insert(
            "ts".into(),
            serde_json::Value::String(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        );
        map.insert(
            "kindled_home".into(),
            serde_json::Value::String(kindled_home.to_string_lossy().into_owned()),
        );
    }

    let mut f = OpenOptions::new().create(true).append(true).open(&log)?;
    writeln!(f, "{}", serde_json::to_string(&entry).unwrap_or_default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_home() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ce-launchlog-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample(stdout: &str) -> LaunchFailure {
        LaunchFailure {
            command: "nell init --persona phoebe".into(),
            exit_code: Some(1),
            stdout_tail: stdout.into(),
            stderr_tail: "boom".into(),
            platform: "linux".into(),
            platform_version: "x86_64".into(),
            bundled_runtime_path: Some("/x/nell".into()),
            runtime_path_exists: true,
            duration_ms: 42,
        }
    }

    #[test]
    fn append_creates_file_with_entry() {
        let home = tmp_home();
        append_failure(&home, &sample("")).unwrap();
        let log = fs::read_to_string(home.join("launch-failures.log")).unwrap();
        assert!(log.contains("\"command\":\"nell init --persona phoebe\""));
        assert!(log.contains("\"stderr_tail\":\"boom\""));
        assert!(log.contains("\"ts\":"));
        assert!(log.contains("\"kindled_home\":"));
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn append_is_one_json_line_per_entry() {
        let home = tmp_home();
        append_failure(&home, &sample("a")).unwrap();
        append_failure(&home, &sample("b")).unwrap();
        let log = fs::read_to_string(home.join("launch-failures.log")).unwrap();
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2);
        // Each line must be valid standalone JSON.
        for line in lines {
            serde_json::from_str::<serde_json::Value>(line).unwrap();
        }
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn rotates_at_threshold() {
        let home = tmp_home();
        let big = "x".repeat(190 * 1024);
        for _ in 0..3 {
            append_failure(&home, &sample(&big)).unwrap();
        }
        assert!(home.join("launch-failures.1.log").is_file());
        let current_size = fs::metadata(home.join("launch-failures.log")).unwrap().len();
        assert!(
            current_size < 220 * 1024,
            "live log should have rotated; was {current_size} bytes"
        );
        fs::remove_dir_all(&home).ok();
    }
}
