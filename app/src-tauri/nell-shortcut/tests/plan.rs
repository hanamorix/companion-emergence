//! C1, C29 (messages), C12 (product name): the mechanism decision and the user-facing texts.
//! Host rule (criteria C1, R4-2): tests with Linux/macOS-style path literals are cfg(unix);
//! tests with Windows-style path literals are cfg(windows); string-only tests are ungated.

use nell_shortcut::*;
use std::path::PathBuf;

const NO_FUSE_EXPECTED: &str = "This AppImage is running without FUSE (in extract-and-run mode), so the nell shortcut was not installed. A shortcut would re-extract the whole app on every nell command and delete the running app's files when the command ends. To get the shortcut, install FUSE for your distribution (on Debian and Ubuntu the package is libfuse2, called libfuse2t64 on Ubuntu 24.04), start the AppImage normally, and try again. Or install the .deb package instead, which does not need FUSE.";

const SAFE_EXPECTED: &str = "The app's files are not inside the AppImage's own folder, so a nell shortcut cannot be made safely. Start the AppImage again in the normal way and retry. If this keeps happening, install the .deb package instead.";

fn p(s: &str) -> PathBuf {
    PathBuf::from(s)
}

fn env(os: Os) -> ShortcutEnv {
    ShortcutEnv {
        os,
        bundled: None,
        home: None,
        local_app_data: None,
        appimage: None,
        appdir: None,
        macos_unstable_reason: None,
        win_prefixes: vec![],
    }
}

fn unavailable(plan: ShortcutPlan) -> String {
    match plan {
        ShortcutPlan::Unavailable { reason } => reason,
        other => panic!("expected Unavailable, got {:?}", other),
    }
}

// ---- string-only, host independent --------------------------------------------------------

#[test]
fn no_fuse_message_is_pinned_exactly() {
    assert_eq!(NO_FUSE_MESSAGE, NO_FUSE_EXPECTED);
}

#[test]
fn safe_location_message_is_pinned_exactly() {
    assert_eq!(SAFE_LOCATION_MESSAGE, SAFE_EXPECTED);
}

#[test]
fn user_facing_messages_have_no_dashes() {
    for m in [NO_FUSE_MESSAGE, SAFE_LOCATION_MESSAGE] {
        assert!(!m.contains('\u{2014}'), "em-dash in {m}");
        assert!(!m.contains('\u{2013}'), "en-dash in {m}");
    }
}

#[test]
fn no_fuse_message_names_the_problem_and_both_remedies() {
    let m = NO_FUSE_MESSAGE;
    assert!(m.contains("without FUSE"));
    assert!(m.contains("delete the running app's files"));
    assert!(m.contains("libfuse2"));
    assert!(m.contains(".deb"));
}

#[test]
fn product_name_matches_tauri_conf() {
    let conf = include_str!("../../tauri.conf.json");
    let needle = format!("\"productName\": \"{}\"", PRODUCT_NAME);
    assert!(conf.contains(&needle), "tauri.conf.json productName != {}", PRODUCT_NAME);
}

// ---- unix-style rows ------------------------------------------------------------------------

#[cfg(unix)]
mod unix_rows {
    use super::*;
    use std::ffi::{OsStr, OsString};

    const DEB_NELL: &str = "/usr/lib/Companion Emergence/python-runtime/bin/nell";

    #[test]
    fn row1_macos_symlink() {
        let mut e = env(Os::MacOs);
        e.bundled = Some(p("/Applications/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell"));
        e.home = Some(p("/Users/u"));
        assert_eq!(
            plan_shortcut(&e),
            ShortcutPlan::Symlink {
                bundled: p("/Applications/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell"),
                target_dir: p("/Users/u/.local/bin"),
                target: p("/Users/u/.local/bin/nell"),
            }
        );
    }

    #[test]
    fn row2_macos_unstable_reason_is_passed_through() {
        let mut e = env(Os::MacOs);
        e.bundled = Some(p("/Volumes/x/nell"));
        e.home = Some(p("/Users/u"));
        e.macos_unstable_reason = Some("moved it".to_string());
        assert_eq!(plan_shortcut(&e), ShortcutPlan::Unavailable { reason: "moved it".to_string() });
    }

    #[test]
    fn row3_linux_deb_symlink() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p(DEB_NELL));
        e.home = Some(p("/home/u"));
        assert_eq!(
            plan_shortcut(&e),
            ShortcutPlan::Symlink {
                bundled: p(DEB_NELL),
                target_dir: p("/home/u/.local/bin"),
                target: p("/home/u/.local/bin/nell"),
            }
        );
    }

    #[test]
    fn row4_linux_appimage_script() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p("/tmp/.mount_CEabc/usr/lib/Companion Emergence/python-runtime/bin/nell"));
        e.home = Some(p("/home/u"));
        e.appimage = Some(p("/home/u/Apps/CE.AppImage"));
        e.appdir = Some(p("/tmp/.mount_CEabc"));
        assert_eq!(
            plan_shortcut(&e),
            ShortcutPlan::AppImageScript {
                appimage: p("/home/u/Apps/CE.AppImage"),
                target_dir: p("/home/u/.local/bin"),
                target: p("/home/u/.local/bin/nell"),
            }
        );
    }

    #[test]
    fn row5_temp_mount_without_appimage_env_is_refused_naming_the_mount() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p("/tmp/.mount_X/usr/lib/Companion Emergence/python-runtime/bin/nell"));
        e.home = Some(p("/home/u"));
        let reason = unavailable(plan_shortcut(&e));
        assert!(reason.contains("temporary AppImage"), "{reason}");
    }

    #[test]
    fn row6_appimage_leaked_into_deb_run_still_symlinks() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p(DEB_NELL));
        e.home = Some(p("/home/u"));
        e.appimage = Some(p("/home/u/Apps/Other.AppImage"));
        assert_eq!(
            plan_shortcut(&e),
            ShortcutPlan::Symlink {
                bundled: p(DEB_NELL),
                target_dir: p("/home/u/.local/bin"),
                target: p("/home/u/.local/bin/nell"),
            }
        );
    }

    #[test]
    fn row6b_appimage_and_appdir_both_leaked_into_deb_run_still_symlinks() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p(DEB_NELL));
        e.home = Some(p("/home/u"));
        e.appimage = Some(p("/home/u/Apps/Other.AppImage"));
        e.appdir = Some(p("/tmp/.mount_OTHER"));
        assert_eq!(
            plan_shortcut(&e),
            ShortcutPlan::Symlink {
                bundled: p(DEB_NELL),
                target_dir: p("/home/u/.local/bin"),
                target: p("/home/u/.local/bin/nell"),
            }
        );
    }

    #[test]
    fn row7_relative_or_empty_appimage_is_unavailable() {
        for appimage in ["relative.AppImage", ""] {
            let mut e = env(Os::Linux);
            e.bundled = Some(p("/tmp/.mount_X/usr/lib/Companion Emergence/python-runtime/bin/nell"));
            e.home = Some(p("/home/u"));
            e.appimage = Some(p(appimage));
            e.appdir = Some(p("/tmp/.mount_X"));
            match plan_shortcut(&e) {
                ShortcutPlan::Unavailable { .. } => {}
                other => panic!("appimage {appimage:?}: expected Unavailable, got {:?}", other),
            }
        }
    }

    #[test]
    fn row8_home_unset_text() {
        for os in [Os::MacOs, Os::Linux] {
            let mut e = env(os);
            e.bundled = Some(p(DEB_NELL));
            assert_eq!(
                plan_shortcut(&e),
                ShortcutPlan::Unavailable { reason: "HOME env var not set; cannot resolve ~/.local/bin".to_string() },
                "{:?}",
                os
            );
        }
    }

    #[test]
    fn row11_bundled_missing_texts_per_os() {
        assert_eq!(
            plan_shortcut(&env(Os::MacOs)),
            ShortcutPlan::Unavailable {
                reason: "Bundled nell CLI not found in this build. Install from a release DMG.".to_string()
            }
        );
        assert_eq!(
            plan_shortcut(&env(Os::Linux)),
            ShortcutPlan::Unavailable {
                reason: "Bundled nell CLI not found in this build. Install from the .deb or AppImage release.".to_string()
            }
        );
    }

    #[test]
    fn row12_other_os_is_unsupported() {
        let mut e = env(Os::Other);
        e.bundled = Some(p("/x/nell"));
        e.home = Some(p("/home/u"));
        let reason = unavailable(plan_shortcut(&e));
        assert!(reason.contains("not supported on this platform"), "{reason}");
    }

    #[test]
    fn row13_no_fuse_appimage_gets_the_exact_message() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p("/tmp/appimage_extracted_0123abcd/usr/lib/Companion Emergence/python-runtime/bin/nell"));
        e.home = Some(p("/home/u"));
        e.appimage = Some(p("/home/u/CE.AppImage"));
        e.appdir = Some(p("/tmp/appimage_extracted_0123abcd"));
        assert_eq!(plan_shortcut(&e), ShortcutPlan::Unavailable { reason: NO_FUSE_EXPECTED.to_string() });
    }

    #[test]
    fn row13_extracted_path_without_appimage_env_is_a_temporary_location() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p("/tmp/appimage_extracted_0123abcd/usr/lib/Companion Emergence/python-runtime/bin/nell"));
        e.home = Some(p("/home/u"));
        let reason = unavailable(plan_shortcut(&e));
        assert!(reason.contains("temporary AppImage"), "{reason}");
    }

    #[test]
    fn row13b_mismatched_spelling_of_a_mount_path_gets_the_safe_location_message() {
        let mut e = env(Os::Linux);
        e.bundled = Some(p("/private/tmp/.mount_X/usr/lib/Companion Emergence/python-runtime/bin/nell"));
        e.home = Some(p("/home/u"));
        e.appimage = Some(p("/home/u/CE.AppImage"));
        e.appdir = Some(p("/tmp/.mount_X"));
        assert_eq!(plan_shortcut(&e), ShortcutPlan::Unavailable { reason: SAFE_EXPECTED.to_string() });
    }

    #[test]
    fn row14_from_lookup_maps_empty_to_none_except_macos_home() {
        let lookup = |k: &str| -> Option<OsString> {
            match k {
                "APPIMAGE" => Some(OsString::from("")),
                "APPDIR" => Some(OsString::from("/tmp/.mount_X")),
                "HOME" => Some(OsString::from("")),
                "LOCALAPPDATA" => Some(OsString::from("")),
                _ => None,
            }
        };
        let linux = ShortcutEnv::from_lookup(Os::Linux, &lookup, None, None);
        assert_eq!(linux.appimage, None);
        assert_eq!(linux.appdir, Some(p("/tmp/.mount_X")));
        assert_eq!(linux.home, None);
        assert_eq!(linux.local_app_data, None);
        let mac = ShortcutEnv::from_lookup(Os::MacOs, &lookup, None, None);
        assert_eq!(mac.home, Some(p("")));
    }

    #[test]
    fn row14_from_lookup_collects_windows_prefixes_in_order() {
        let lookup = |k: &str| -> Option<OsString> {
            match k {
                "LOCALAPPDATA" => Some(OsString::from("/l")),
                "ProgramW6432" => Some(OsString::from("/w")),
                "ProgramFiles" => Some(OsString::from("/f")),
                _ => None,
            }
        };
        let e = ShortcutEnv::from_lookup(Os::Windows, &lookup, None, None);
        assert_eq!(
            e.win_prefixes,
            vec![
                ("LOCALAPPDATA".to_string(), "/l".to_string()),
                ("ProgramW6432".to_string(), "/w".to_string()),
                ("ProgramFiles".to_string(), "/f".to_string()),
            ]
        );
    }

    #[test]
    fn valid_appdir_rows() {
        assert_eq!(valid_appdir(None), None);
        assert_eq!(valid_appdir(Some(OsStr::new(""))), None);
        assert_eq!(valid_appdir(Some(OsStr::new("/"))), None);
        assert_eq!(valid_appdir(Some(OsStr::new("relative/dir"))), None);
        assert_eq!(valid_appdir(Some(OsStr::new("/tmp/.mount_X"))), Some(p("/tmp/.mount_X")));
    }

    #[test]
    fn temp_mount_component_detection() {
        assert!(has_temp_mount_component(&p("/tmp/.mount_X/usr/lib/x")));
        assert!(has_temp_mount_component(&p("/run/user/1000/.mount_abc/usr")));
        assert!(has_temp_mount_component(&p("/tmp/appimage_extracted_9f/usr")));
        assert!(!has_temp_mount_component(&p("/usr/lib/Companion Emergence/x")));
        assert!(!has_temp_mount_component(&p("/home/u/mount_notes/x")));
    }
}

// ---- windows-style rows ----------------------------------------------------------------------

#[cfg(windows)]
mod windows_rows {
    use super::*;

    #[test]
    fn row9_windows_cmd_plan() {
        let mut e = env(Os::Windows);
        e.bundled = Some(p(r"C:\Program Files\Companion Emergence\python-runtime\Scripts\nell.bat"));
        e.local_app_data = Some(p(r"C:\Users\u\AppData\Local"));
        e.win_prefixes = vec![("LOCALAPPDATA".to_string(), r"C:\Users\u\AppData\Local".to_string())];
        assert_eq!(
            plan_shortcut(&e),
            ShortcutPlan::WindowsCmd {
                bundled: p(r"C:\Program Files\Companion Emergence\python-runtime\Scripts\nell.bat"),
                cmd_dir: p(r"C:\Users\u\AppData\Local\CompanionEmergence\bin"),
                cmd_path: p(r"C:\Users\u\AppData\Local\CompanionEmergence\bin\nell.cmd"),
                win_prefixes: vec![("LOCALAPPDATA".to_string(), r"C:\Users\u\AppData\Local".to_string())],
            }
        );
    }

    #[test]
    fn row10_windows_without_local_app_data_is_unavailable() {
        let mut e = env(Os::Windows);
        e.bundled = Some(p(r"C:\x\nell.bat"));
        let reason = unavailable(plan_shortcut(&e));
        assert!(!reason.is_empty());
    }

    #[test]
    fn row11_windows_bundled_missing_text() {
        assert_eq!(
            plan_shortcut(&env(Os::Windows)),
            ShortcutPlan::Unavailable {
                reason: "Bundled nell CLI not found in this build. Install from the release installer.".to_string()
            }
        );
    }
}
