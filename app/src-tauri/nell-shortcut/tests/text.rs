//! String-only, host-independent tests: C3 (cmd text), C7 (PATH merge), C4 file rows,
//! C13 decision, C10 argument parsing, C6 macOS golden hint strings.

use nell_shortcut::*;
use std::ffi::OsString;

const MARKER: &str = SHIM_MARKER;

// ---------------------------------------------------------------- C3: nell.cmd text

const PF_BAT: &str = r"C:\Program Files\Companion Emergence\python-runtime\Scripts\nell.bat";

fn assert_crlf_only(text: &str) {
    assert!(!text.contains("\r\r"), "double CR in {:?}", text);
    for (i, ch) in text.char_indices() {
        if ch == '\n' {
            assert!(i > 0 && text.as_bytes()[i - 1] == b'\r', "bare LF at {i} in {:?}", text);
        }
    }
    assert!(text.ends_with("\r\n"), "text must end with CRLF");
}

#[test]
fn cmd_ascii_path_form() {
    let t = render_windows_cmd(PF_BAT, &[]).expect("render");
    assert!(t.is_ascii());
    assert_crlf_only(&t);
    assert!(t.contains(MARKER));
    assert!(t.contains(&format!("\"{}\" %*", PF_BAT)), "{t:?}");
    assert!(!t.contains("%1"));
    assert!(!t.contains("chcp"));
    assert!(!t.to_lowercase().contains("call "));
}

#[test]
fn cmd_percent_in_path_is_doubled() {
    let t = render_windows_cmd(r"C:\100%\x\nell.bat", &[]).expect("render");
    assert!(t.contains(r#""C:\100%%\x\nell.bat" %*"#), "{t:?}");
}

#[test]
fn cmd_rejects_quote_and_control_characters() {
    assert!(render_windows_cmd("C:\\a\"b\\nell.bat", &[]).is_err());
    assert!(render_windows_cmd("C:\\a\nb\\nell.bat", &[]).is_err());
    assert!(render_windows_cmd("C:\\a\rb\\nell.bat", &[]).is_err());
}

fn jose_local() -> String {
    format!("C:\\Users\\Jos\u{e9}\\AppData\\Local")
}

#[test]
fn cmd_non_ascii_prefix_becomes_env_reference_and_file_stays_ascii() {
    let bat = format!("{}\\Companion Emergence\\python-runtime\\Scripts\\nell.bat", jose_local());
    let prefixes = vec![("LOCALAPPDATA".to_string(), jose_local())];
    let t = render_windows_cmd(&bat, &prefixes).expect("render");
    assert!(t.is_ascii(), "{t:?}");
    assert!(
        t.contains("\"%LOCALAPPDATA%\\Companion Emergence\\python-runtime\\Scripts\\nell.bat\" %*"),
        "{t:?}"
    );
}

#[test]
fn cmd_prefix_match_is_case_insensitive() {
    let upper = jose_local().to_uppercase();
    let lower_bat = format!("{}\\Companion Emergence\\x\\nell.bat", jose_local().to_lowercase());
    let t = render_windows_cmd(&lower_bat, &[("LOCALAPPDATA".to_string(), upper)]).expect("render");
    assert!(t.contains("%LOCALAPPDATA%\\Companion Emergence\\x\\nell.bat"), "{t:?}");
}

#[test]
fn cmd_prefix_requires_a_backslash_boundary() {
    let bat = format!("{}X\\Companion Emergence\\x\\nell.bat", jose_local());
    let r = render_windows_cmd(&bat, &[("LOCALAPPDATA".to_string(), jose_local())]);
    assert!(r.is_err(), "LocalX must not match the Local prefix: {r:?}");
}

#[test]
fn cmd_non_ascii_without_matching_prefix_is_an_error() {
    let bat = format!("D:\\Programme\u{65E5}\\Companion Emergence\\x\\nell.bat");
    assert!(render_windows_cmd(&bat, &[("LOCALAPPDATA".to_string(), jose_local())]).is_err());
}

#[test]
fn cmd_template_fill_normalises_line_endings() {
    let lf = "@echo off\nrem x\n\"{NELL_BAT}\" %*\n";
    let crlf = "@echo off\r\nrem x\r\n\"{NELL_BAT}\" %*\r\n";
    for tmpl in [lf, crlf] {
        let t = fill_cmd_template(tmpl, r"C:\a b\nell.bat");
        assert_crlf_only(&t);
        assert!(t.contains("\"C:\\a b\\nell.bat\" %*"));
    }
}

#[test]
fn cmd_text_equals_the_shared_template() {
    let tmpl = include_str!("../src/nell_cmd.tmpl");
    let t = render_windows_cmd(PF_BAT, &[]).expect("render");
    assert_eq!(t, fill_cmd_template(tmpl, PF_BAT));
}

// ---------------------------------------------------------------- C7: PATH merge

const DIR: &str = r"C:\Users\u\AppData\Local\CompanionEmergence\bin";

fn lookup(name: &str) -> Option<String> {
    match name.to_ascii_uppercase().as_str() {
        "LOCALAPPDATA" => Some(r"C:\Users\u\AppData\Local".to_string()),
        _ => None,
    }
}

#[test]
fn path_merge_appends_and_preserves_the_original_bytes() {
    let cases: Vec<(Option<&str>, &str)> = vec![
        (None, DIR),
        (Some(""), DIR),
        (Some(r"C:\a"), r"C:\a;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r"C:\a;C:\b"), r"C:\a;C:\b;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r"C:\a;"), r"C:\a;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r"%SystemRoot%\system32;C:\a"), r"%SystemRoot%\system32;C:\a;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r"C:\a; "), r"C:\a; ;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r";;C:\a"), r";;C:\a;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r"C:\Program Files\x;C:\y z"), r"C:\Program Files\x;C:\y z;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some(r"C:\a;C:\Users\u\AppData\Local\CompanionEmergenceX\bin"), r"C:\a;C:\Users\u\AppData\Local\CompanionEmergenceX\bin;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
        (Some("\u{65E5}\u{672C};C:\\a"), "\u{65E5}\u{672C};C:\\a;C:\\Users\\u\\AppData\\Local\\CompanionEmergence\\bin"),
        (Some(r"C:\a;D:\b;E:\c"), r"C:\a;D:\b;E:\c;C:\Users\u\AppData\Local\CompanionEmergence\bin"),
    ];
    for (existing, expected) in cases {
        let got = merge_user_path(existing, DIR, &lookup).expect("must append");
        assert_eq!(got, expected, "existing {:?}", existing);
        if let Some(e) = existing {
            assert!(got.starts_with(e), "original must be an exact prefix: {:?}", existing);
        }
        // applying twice is a no-op
        assert_eq!(merge_user_path(Some(&got), DIR, &lookup), None, "second run must not write");
    }
}

#[test]
fn path_merge_recognises_an_existing_entry_in_any_spelling() {
    let present = [
        r"C:\a;C:\Users\u\AppData\Local\CompanionEmergence\bin",
        r"C:\a;c:\users\u\appdata\local\companionemergence\bin",
        r"C:\a;C:\Users\u\AppData\Local\CompanionEmergence\bin\",
        r#"C:\a;"C:\Users\u\AppData\Local\CompanionEmergence\bin""#,
        r"C:\a;%LOCALAPPDATA%\CompanionEmergence\bin",
        r"C:\a;C:/Users/u/AppData/Local/CompanionEmergence/bin",
        r"C:\a;  C:\Users\u\AppData\Local\CompanionEmergence\bin  ;C:\b",
    ];
    for existing in present {
        assert_eq!(merge_user_path(Some(existing), DIR, &lookup), None, "{existing:?}");
    }
}

// ---------------------------------------------------------------- C4 / C13: ownership (path-free rows)

fn script_text() -> String {
    format!("#!/bin/sh\n# {MARKER} v1\nexec x\n")
}

fn file<S: AsRef<str>>(text: Option<S>, exec: bool) -> Existing {
    Existing::File { text: text.map(|s| s.as_ref().to_string()), exec_bit: exec }
}

#[test]
fn script_rows_without_symlinks() {
    let d = Desired::Script { text: script_text() };
    assert_eq!(classify_existing(&Existing::Missing, &d), Action::Create);
    assert_eq!(classify_existing(&file(Some(&script_text()), true), &d), Action::AlreadyCurrent);
    assert_eq!(classify_existing(&file(Some(&script_text()), false), &d), Action::Replace, "lost exec bit is repaired");
    let stale = format!("#!/bin/sh\n# {MARKER} v1\nexec old\n");
    assert_eq!(classify_existing(&file(Some(&stale), true), &d), Action::Replace);
    for foreign in [Some("#!/bin/sh\necho mine\n"), None] {
        match classify_existing(&file(foreign, true), &d) {
            Action::Refuse(m) => assert!(!m.is_empty()),
            other => panic!("foreign {foreign:?}: expected Refuse, got {other:?}"),
        }
    }
    match classify_existing(&Existing::Other, &d) {
        Action::Refuse(_) => {}
        other => panic!("directory: expected Refuse, got {other:?}"),
    }
}

#[test]
fn marker_must_be_in_the_first_three_lines() {
    let d = Desired::Script { text: script_text() };
    let late = format!("a\nb\nc\n# {MARKER}\n");
    match classify_existing(&file(Some(&late), true), &d) {
        Action::Refuse(_) => {}
        other => panic!("marker on line 4 must not count as ours, got {other:?}"),
    }
}

#[test]
fn cmd_rows() {
    let text = fill_cmd_template(include_str!("../src/nell_cmd.tmpl"), PF_BAT);
    let d = Desired::Cmd { text: text.clone() };
    assert_eq!(classify_existing(&Existing::Missing, &d), Action::Create);
    assert_eq!(classify_existing(&file(Some(&text), false), &d), Action::AlreadyCurrent, "no exec bit on Windows");
    let old = text.replace("Program Files", "Old Place");
    assert_eq!(classify_existing(&file(Some(&old), false), &d), Action::Replace);
    match classify_existing(&file(Some("@echo hi\r\n"), false), &d) {
        Action::Refuse(_) => {}
        other => panic!("foreign nell.cmd must be refused, got {other:?}"),
    }
}

#[test]
fn refresh_decision_rows() {
    let t = script_text();
    let stale = format!("#!/bin/sh\n# {MARKER} v1\nexec old\n");
    assert!(refresh_decision(&file(Some(&stale), true), &t));
    assert!(!refresh_decision(&file(Some(&t), true), &t));
    assert!(refresh_decision(&file(Some(&t), false), &t), "lost exec bit");
    assert!(!refresh_decision(&file(Some("echo mine"), true), &t), "foreign file");
    assert!(!refresh_decision(&file::<&str>(None, true), &t), "unreadable");
    assert!(!refresh_decision(&Existing::Missing, &t), "never creates");
    assert!(!refresh_decision(&Existing::Other, &t));
}

// ---------------------------------------------------------------- C10: CLI argument parsing

fn os(v: &[&str]) -> Vec<OsString> {
    v.iter().map(|s| OsString::from(*s)).collect()
}

#[test]
fn cli_mode_only_when_flag_is_first() {
    assert_eq!(parse_cli_mode(&[]), CliMode::Gui);
    assert_eq!(
        parse_cli_mode(&os(&["--companion-nell-cli", "a", "b c", "", "--help", "-h", "--version", "--appimage-x"])),
        CliMode::Nell(os(&["a", "b c", "", "--help", "-h", "--version", "--appimage-x"]))
    );
    assert_eq!(parse_cli_mode(&os(&["--companion-nell-cli"])), CliMode::Nell(vec![]));
    assert_eq!(parse_cli_mode(&os(&["status", "--companion-nell-cli"])), CliMode::Gui);
    assert_eq!(parse_cli_mode(&os(&["--companion-nell-cli=x"])), CliMode::Gui);
    assert_eq!(parse_cli_mode(&os(&["--gapplication-service"])), CliMode::Gui);
    assert_eq!(parse_cli_mode(&os(&["--display=:0"])), CliMode::Gui);
}

// ---------------------------------------------------------------- C6: macOS golden strings
// Expected values were extracted from `git show a3f3fa7b:app/src-tauri/src/lib.rs` (path_hint),
// not retyped: the string continuation in the original joins the two literals with one space.

#[test]
fn macos_path_hint_strings_are_byte_identical_to_the_original() {
    assert_eq!(path_hint_text(Os::MacOs, true), "Open a new Terminal window and run `nell --version`.");
    assert_eq!(
        path_hint_text(Os::MacOs, false),
        "If `nell` isn't found in Terminal, add `export PATH=\"$HOME/.local/bin:$PATH\"` to your shell rc (~/.zshrc) and reload."
    );
}

#[test]
fn linux_path_hint_strings() {
    assert_eq!(path_hint_text(Os::Linux, true), "Open a new terminal and run `nell --version`.");
    assert_eq!(
        path_hint_text(Os::Linux, false),
        "If `nell` isn't found, add `export PATH=\"$HOME/.local/bin:$PATH\"` to ~/.profile (or ~/.bashrc) and sign out and back in. Some distributions only add ~/.local/bin to PATH at login when the folder already existed."
    );
}
