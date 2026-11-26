//! The launch-time "is it our shim" gate (decides whether a refresh problem is worth logging).
//! Host-independent: only temp-dir paths, no path literals.

use nell_shortcut::{is_our_shim_file, is_our_shim_file_capped, SHIM_MARKER};
use std::fs;

fn shim_text() -> String {
    format!("#!/bin/sh\n# {} v1: written by Companion Emergence\nexec x\n", SHIM_MARKER)
}

#[test]
fn our_shim_is_recognised() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("nell");
    fs::write(&f, shim_text()).unwrap();
    assert!(is_our_shim_file(&f));
}

#[test]
fn a_users_own_text_file_is_not_ours() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("nell");
    fs::write(&f, "#!/bin/sh\necho my own nell\n").unwrap();
    assert!(!is_our_shim_file(&f));
}

#[test]
fn a_non_utf8_file_is_not_ours() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("nell");
    fs::write(&f, [0x7f, b'E', b'L', b'F', 0xff, 0xfe, 0x00, 0x80]).unwrap();
    assert!(!is_our_shim_file(&f));
}

#[test]
fn invalid_utf8_before_the_marker_is_not_ours() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("nell");
    let mut bytes = vec![0xff, 0xfe, b'\n'];
    bytes.extend_from_slice(shim_text().as_bytes());
    fs::write(&f, bytes).unwrap();
    assert!(!is_our_shim_file(&f));
}

#[test]
fn a_directory_and_a_missing_path_are_not_ours() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(!is_our_shim_file(tmp.path()));
    assert!(!is_our_shim_file(&tmp.path().join("nothing-here")));
}

#[test]
fn a_large_file_is_not_read_in_full() {
    // One huge first line whose last bytes are the marker. A full read would find the marker on
    // "line 1"; a capped read must stop before it and answer false.
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("nell");
    let mut bytes = vec![b'a'; 8 * 1024 * 1024];
    bytes.extend_from_slice(SHIM_MARKER.as_bytes());
    fs::write(&f, bytes).unwrap();
    assert!(!is_our_shim_file(&f));
    assert!(!is_our_shim_file_capped(&f, 64 * 1024));
    // control: with a cap that covers the whole file the same file IS recognised, so the answer
    // above came from the cap and not from the content
    assert!(is_our_shim_file_capped(&f, 16 * 1024 * 1024));
}

#[test]
fn a_multibyte_character_cut_by_the_cap_does_not_hide_our_shim() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("nell");
    let text = format!("# {}\n\u{e9}\u{e9}\u{e9}\u{e9}\n", SHIM_MARKER);
    fs::write(&f, &text).unwrap();
    // cap lands in the middle of an e-acute (2 bytes)
    let marker_end = 2 + SHIM_MARKER.len() + 1;
    assert!(is_our_shim_file_capped(&f, marker_end + 1));
}
