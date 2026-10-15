//! C8 / C3x-adjacent: registry and cmd-file glue on a scratch HKCU key (Windows only).
//! Every test uses its OWN unique subkey (cargo test is multithreaded) and a drop guard that
//! deletes it even if an assertion panics. The real `Environment` key is never touched.
#![cfg(windows)]

use nell_shortcut::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use winreg::enums::*;
use winreg::{RegKey, RegValue};

static COUNTER: AtomicU32 = AtomicU32::new(0);
const PARENT: &str = r"Software\CompanionEmergenceTest";

struct Scratch {
    subkey: String,
}

impl Scratch {
    fn new() -> Scratch {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let subkey = format!(r"{}\t{}_{}", PARENT, std::process::id(), n);
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        hkcu.create_subkey(&subkey).expect("create scratch key");
        Scratch { subkey }
    }

    fn set(&self, vtype: RegType, text: &str) {
        let key = RegKey::predef(HKEY_CURRENT_USER)
            .open_subkey_with_flags(&self.subkey, KEY_READ | KEY_WRITE)
            .unwrap();
        key.set_raw_value("Path", &RegValue { bytes: utf16(text).into(), vtype }).unwrap();
    }

    fn set_raw(&self, vtype: RegType, bytes: Vec<u8>) {
        let key = RegKey::predef(HKEY_CURRENT_USER)
            .open_subkey_with_flags(&self.subkey, KEY_READ | KEY_WRITE)
            .unwrap();
        key.set_raw_value("Path", &RegValue { bytes: bytes.into(), vtype }).unwrap();
    }

    fn get(&self) -> Option<(RegType, Vec<u8>)> {
        let key = RegKey::predef(HKEY_CURRENT_USER).open_subkey(&self.subkey).unwrap();
        key.get_raw_value("Path").ok().map(|v| (v.vtype, v.bytes.to_vec()))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let _ = hkcu.delete_subkey_all(&self.subkey);
        // best effort only: another test may be using a sibling right now
        let _ = hkcu.delete_subkey(PARENT);
    }
}

fn utf16(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for u in s.encode_utf16().chain(std::iter::once(0u16)) {
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

fn decode(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    String::from_utf16(&units).unwrap().trim_end_matches('\0').to_string()
}

const DIR: &str = r"C:\Users\u\AppData\Local\CompanionEmergence\bin";

#[test]
fn expand_sz_value_keeps_its_type_and_original_text() {
    let s = Scratch::new();
    s.set(REG_EXPAND_SZ, r"%SystemRoot%\system32;C:\a");
    assert_eq!(ensure_user_path(&s.subkey, DIR), Ok(Changed::Yes));
    let (t, b) = s.get().unwrap();
    assert_eq!(t, REG_EXPAND_SZ);
    assert_eq!(decode(&b), format!(r"%SystemRoot%\system32;C:\a;{}", DIR));
    assert_eq!(ensure_user_path(&s.subkey, DIR), Ok(Changed::No));
}

#[test]
fn plain_sz_value_stays_sz() {
    let s = Scratch::new();
    s.set(REG_SZ, r"C:\a");
    assert_eq!(ensure_user_path(&s.subkey, DIR), Ok(Changed::Yes));
    let (t, b) = s.get().unwrap();
    assert_eq!(t, REG_SZ);
    assert_eq!(decode(&b), format!(r"C:\a;{}", DIR));
}

#[test]
fn missing_value_is_created_as_expand_sz() {
    let s = Scratch::new();
    assert_eq!(ensure_user_path(&s.subkey, DIR), Ok(Changed::Yes));
    let (t, b) = s.get().unwrap();
    assert_eq!(t, REG_EXPAND_SZ);
    assert_eq!(decode(&b), DIR);
}

#[test]
fn undecodable_value_is_an_error_and_left_byte_identical() {
    let s = Scratch::new();
    // Compare with what the registry actually stored: it does not hand back an odd-length string
    // byte for byte (CI: [67, 0, 0] stored for [67, 0, 58]). The product assertion is unchanged:
    // ensure_user_path fails and does not modify the stored value.
    let odd = vec![0x43, 0x00, 0x3a]; // odd length
    s.set_raw(REG_EXPAND_SZ, odd);
    let stored_odd = s.get().unwrap().1;
    assert!(ensure_user_path(&s.subkey, DIR).is_err());
    assert_eq!(s.get().unwrap().1, stored_odd);
    let lone_surrogate = vec![0x00, 0xd8, 0x41, 0x00, 0x00, 0x00];
    s.set_raw(REG_EXPAND_SZ, lone_surrogate);
    let stored_surrogate = s.get().unwrap().1;
    assert!(ensure_user_path(&s.subkey, DIR).is_err());
    assert_eq!(s.get().unwrap().1, stored_surrogate);
}

#[test]
fn non_string_value_is_an_error() {
    let s = Scratch::new();
    s.set_raw(REG_DWORD, vec![1, 0, 0, 0]);
    assert!(ensure_user_path(&s.subkey, DIR).is_err());
}

fn bat_in(tmp: &tempfile::TempDir) -> PathBuf {
    let bat = tmp.path().join("Companion Emergence").join("python-runtime").join("Scripts").join("nell.bat");
    std::fs::create_dir_all(bat.parent().unwrap()).unwrap();
    std::fs::write(&bat, "@echo off\r\n").unwrap();
    bat
}

#[test]
fn install_writes_the_cmd_and_extends_the_scratch_path_once() {
    let s = Scratch::new();
    s.set(REG_EXPAND_SZ, r"C:\a");
    let tmp = tempfile::tempdir().unwrap();
    let bat = bat_in(&tmp);
    let dir = tmp.path().join("CompanionEmergence").join("bin");
    let cmd = dir.join("nell.cmd");
    let out = install_windows_cmd(&bat, &dir, &cmd, &[], &s.subkey);
    assert!(out.success, "{out:?}");
    let want = render_windows_cmd(&bat.to_string_lossy(), &[]).unwrap();
    assert_eq!(std::fs::read_to_string(&cmd).unwrap(), want);
    let path_after = decode(&s.get().unwrap().1);
    assert_eq!(path_after, format!(r"C:\a;{}", dir.display()));
    let again = install_windows_cmd(&bat, &dir, &cmd, &[], &s.subkey);
    assert!(again.success, "{again:?}");
    assert_eq!(decode(&s.get().unwrap().1), path_after, "PATH is extended once");
}

#[test]
fn a_foreign_nell_cmd_is_never_overwritten() {
    let s = Scratch::new();
    let tmp = tempfile::tempdir().unwrap();
    let bat = bat_in(&tmp);
    let dir = tmp.path().join("bin");
    std::fs::create_dir_all(&dir).unwrap();
    let cmd = dir.join("nell.cmd");
    std::fs::write(&cmd, "@echo mine\r\n").unwrap();
    let out = install_windows_cmd(&bat, &dir, &cmd, &[], &s.subkey);
    assert!(!out.success);
    assert_eq!(out.exit_code, 73);
    assert_eq!(std::fs::read_to_string(&cmd).unwrap(), "@echo mine\r\n");
}

#[test]
fn path_write_failure_reports_the_file_and_the_folder_to_add_by_hand() {
    let tmp = tempfile::tempdir().unwrap();
    let bat = bat_in(&tmp);
    let dir = tmp.path().join("CompanionEmergence").join("bin");
    let cmd = dir.join("nell.cmd");
    let missing_subkey = format!(r"{}\does_not_exist_{}", PARENT, std::process::id());
    let out = install_windows_cmd(&bat, &dir, &cmd, &[], &missing_subkey);
    assert!(!out.success);
    assert!(cmd.exists(), "the file stays; a re-run completes the step");
    assert!(out.stderr.contains(&cmd.display().to_string()), "{}", out.stderr);
    assert!(out.stderr.contains(&dir.display().to_string()), "{}", out.stderr);
}
