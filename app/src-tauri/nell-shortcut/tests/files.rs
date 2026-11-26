//! C5 / C13: file glue on real temp directories (unix only). Injected paths, no env access.
#![cfg(unix)]

mod common;

use common::*;
use nell_shortcut::*;
use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};

struct Fixture {
    _tmp: tempfile::TempDir,
    bin_dir: PathBuf,
    target: PathBuf,
    appimage: PathBuf,
    nell: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let bin_dir = root.join("home").join(".local").join("bin");
    let target = bin_dir.join("nell");
    let appimage = root.join("Apps").join("CE.AppImage");
    write_script(&appimage, "#!/bin/sh\nexit 0\n");
    let nell = root.join("mount/usr/lib/Companion Emergence/python-runtime/bin/nell");
    write_script(&nell, "#!/bin/sh\nexit 0\n");
    Fixture { _tmp: tmp, bin_dir, target, appimage, nell }
}

fn install(f: &Fixture) -> Outcome {
    install_script_shim(&f.bin_dir, &f.target, &f.appimage, true, &f.nell)
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .map(|rd| rd.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

#[test]
fn install_creates_an_executable_shim_and_leaves_no_temp_files() {
    let f = fixture();
    let out = install(&f);
    assert!(out.success, "{out:?}");
    assert_eq!(out.exit_code, 0);
    let text = fs::read_to_string(&f.target).unwrap();
    assert_eq!(text, render_appimage_script(&f.appimage).unwrap());
    assert_eq!(mode_of(&f.target), 0o755);
    assert_eq!(entries(&f.bin_dir), vec!["nell".to_string()]);
}

#[test]
fn rerun_is_idempotent() {
    let f = fixture();
    assert!(install(&f).success);
    let before = fs::read(&f.target).unwrap();
    let out = install(&f);
    assert!(out.success, "{out:?}");
    assert!(out.stdout.contains("already"), "{}", out.stdout);
    assert_eq!(fs::read(&f.target).unwrap(), before);
    assert_eq!(entries(&f.bin_dir), vec!["nell".to_string()]);
}

#[test]
fn stale_shim_is_rewritten_for_the_new_appimage() {
    let f = fixture();
    assert!(install(&f).success);
    let moved = f._tmp.path().join("Apps").join("Moved.AppImage");
    write_script(&moved, "#!/bin/sh\nexit 0\n");
    let out = install_script_shim(&f.bin_dir, &f.target, &moved, true, &f.nell);
    assert!(out.success, "{out:?}");
    assert_eq!(fs::read_to_string(&f.target).unwrap(), render_appimage_script(&moved).unwrap());
}

#[test]
fn lost_exec_bit_is_repaired_and_still_succeeds() {
    let f = fixture();
    assert!(install(&f).success);
    fs::set_permissions(&f.target, fs::Permissions::from_mode(0o644)).unwrap();
    let out = install(&f);
    assert!(out.success, "{out:?}");
    assert_eq!(mode_of(&f.target), 0o755);
}

#[test]
fn foreign_regular_file_is_never_overwritten() {
    let f = fixture();
    fs::create_dir_all(&f.bin_dir).unwrap();
    fs::write(&f.target, "#!/bin/sh\necho my own nell\n").unwrap();
    let out = install(&f);
    assert!(!out.success);
    assert_eq!(out.exit_code, 73);
    assert_eq!(fs::read_to_string(&f.target).unwrap(), "#!/bin/sh\necho my own nell\n");
}

#[test]
fn binary_file_is_refused_with_73_and_left_byte_identical() {
    let f = fixture();
    fs::create_dir_all(&f.bin_dir).unwrap();
    let elf: Vec<u8> = vec![0x7f, b'E', b'L', b'F', 0xff, 0xfe, 0x00, 0x80];
    fs::write(&f.target, &elf).unwrap();
    let out = install(&f);
    assert!(!out.success);
    assert_eq!(out.exit_code, 73);
    assert_eq!(fs::read(&f.target).unwrap(), elf);
}

#[test]
fn foreign_symlink_is_never_replaced() {
    let f = fixture();
    fs::create_dir_all(&f.bin_dir).unwrap();
    let venv_nell = f._tmp.path().join("venv/bin/nell");
    write_script(&venv_nell, "#!/bin/sh\nexit 0\n");
    symlink(&venv_nell, &f.target).unwrap();
    let out = install(&f);
    assert!(!out.success);
    assert_eq!(out.exit_code, 73);
    assert_eq!(fs::read_link(&f.target).unwrap(), venv_nell);
}

#[test]
fn our_old_symlink_is_replaced_by_the_shim() {
    let f = fixture();
    fs::create_dir_all(&f.bin_dir).unwrap();
    symlink(&f.nell, &f.target).unwrap(); // ends python-runtime/bin/nell: ours
    let out = install(&f);
    assert!(out.success, "{out:?}");
    assert!(!fs::symlink_metadata(&f.target).unwrap().file_type().is_symlink());
    assert!(fs::read_to_string(&f.target).unwrap().contains(SHIM_MARKER));
}

#[test]
fn missing_resolved_nell_path_writes_nothing_and_names_the_path() {
    let f = fixture();
    let ghost = f._tmp.path().join("ghost/usr/lib/Companion Emergence/python-runtime/bin/nell");
    let out = install_script_shim(&f.bin_dir, &f.target, &f.appimage, true, &ghost);
    assert!(!out.success);
    assert_eq!(out.exit_code, 78);
    assert!(out.stderr.contains("ghost"), "{}", out.stderr);
    assert!(!f.target.exists());
    assert!(entries(&f.bin_dir).is_empty());
}

#[test]
fn missing_appimage_file_writes_nothing() {
    let f = fixture();
    let out = install_script_shim(&f.bin_dir, &f.target, &f.appimage, false, &f.nell);
    assert!(!out.success);
    assert_eq!(out.exit_code, 78);
    assert!(!f.target.exists());
}

// ------------------------------------------------------------ C13: refresh on launch

fn desired(f: &Fixture) -> String {
    render_appimage_script(&f.appimage).unwrap()
}

#[test]
fn refresh_never_creates_a_missing_shim() {
    let f = fixture();
    let _ = refresh_shim_at(&f.target, &desired(&f));
    assert!(!f.target.exists());
    assert!(!f.bin_dir.exists());
}

#[test]
fn refresh_leaves_foreign_files_and_symlinks_alone() {
    let f = fixture();
    fs::create_dir_all(&f.bin_dir).unwrap();
    fs::write(&f.target, "echo mine\n").unwrap();
    let _ = refresh_shim_at(&f.target, &desired(&f));
    assert_eq!(fs::read_to_string(&f.target).unwrap(), "echo mine\n");
    fs::remove_file(&f.target).unwrap();
    symlink(&f.nell, &f.target).unwrap();
    let _ = refresh_shim_at(&f.target, &desired(&f));
    assert_eq!(fs::read_link(&f.target).unwrap(), f.nell);
}

#[test]
fn refresh_rewrites_a_stale_shim_with_mode_0755_and_no_temp_files() {
    let f = fixture();
    fs::create_dir_all(&f.bin_dir).unwrap();
    fs::write(&f.target, format!("#!/bin/sh\n# {} v1\nexec /old/CE.AppImage\n", SHIM_MARKER)).unwrap();
    fs::set_permissions(&f.target, fs::Permissions::from_mode(0o755)).unwrap();
    let out = refresh_shim_at(&f.target, &desired(&f));
    assert!(out.success, "{out:?}");
    assert_eq!(fs::read_to_string(&f.target).unwrap(), desired(&f));
    assert_eq!(mode_of(&f.target), 0o755);
    assert_eq!(entries(&f.bin_dir), vec!["nell".to_string()]);
}

#[test]
fn refresh_repairs_a_lost_exec_bit_and_skips_a_fully_current_shim() {
    let f = fixture();
    assert!(install(&f).success);
    fs::set_permissions(&f.target, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(refresh_shim_at(&f.target, &desired(&f)).success);
    assert_eq!(mode_of(&f.target), 0o755);
    let before = fs::metadata(&f.target).unwrap().modified().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));
    assert!(refresh_shim_at(&f.target, &desired(&f)).success);
    assert_eq!(fs::metadata(&f.target).unwrap().modified().unwrap(), before, "a current shim is not rewritten");
}

// ------------------------------------------------------------ Linux symlink pre-check

#[test]
fn prepare_linux_symlink_rows() {
    let f = fixture();
    let bundled = f.nell.clone();
    // missing: nothing to do
    assert!(prepare_linux_symlink(&f.target, &bundled).is_ok());
    // our shim file: removed so the symlink can be created
    fs::create_dir_all(&f.bin_dir).unwrap();
    fs::write(&f.target, format!("#!/bin/sh\n# {} v1\n", SHIM_MARKER)).unwrap();
    assert!(prepare_linux_symlink(&f.target, &bundled).is_ok());
    assert!(!f.target.exists());
    // foreign regular file: refused
    fs::write(&f.target, "echo mine\n").unwrap();
    let err = prepare_linux_symlink(&f.target, &bundled).unwrap_err();
    assert_eq!(err.exit_code, 73);
    assert_eq!(fs::read_to_string(&f.target).unwrap(), "echo mine\n");
    // foreign symlink: refused
    fs::remove_file(&f.target).unwrap();
    let venv_nell = f._tmp.path().join("venv/bin/nell");
    write_script(&venv_nell, "#!/bin/sh\nexit 0\n");
    symlink(&venv_nell, &f.target).unwrap();
    let err = prepare_linux_symlink(&f.target, &bundled).unwrap_err();
    assert_eq!(err.exit_code, 73);
    assert_eq!(fs::read_link(&f.target).unwrap(), venv_nell);
}
