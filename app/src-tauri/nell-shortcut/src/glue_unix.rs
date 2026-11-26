//! File glue for the Linux AppImage shim and the Linux symlink pre-check. Every input is a
//! parameter (no env access) so tests can inject temp directories.

use std::path::Path;

use crate::{
    atomic_write, classify_existing, current_os, path_contains_dir, path_hint_text, read_existing,
    refresh_decision, refuse_outcome, render_appimage_script, Action, Desired, Existing, Outcome,
};

/// Write (or refresh) `~/.local/bin/nell` as the AppImage shim.
///
/// `nell_path` is the bundled nell resolved in the CURRENT mount; the shim is only written when it
/// exists there, so an unknown layout yields a clear refusal instead of a broken script.
pub fn install_script_shim(
    target_dir: &Path,
    target: &Path,
    appimage: &Path,
    appimage_exists: bool,
    nell_path: &Path,
) -> Outcome {
    if !appimage_exists {
        return Outcome::fail(
            78,
            format!("The AppImage file {} could not be found, so a nell shortcut cannot point at it.", appimage.display()),
        );
    }
    if !nell_path.exists() {
        return Outcome::fail(
            78,
            format!(
                "The bundled nell was not found at {} inside this AppImage (unsupported app layout), so no nell shortcut was written.",
                nell_path.display()
            ),
        );
    }
    let text = match render_appimage_script(appimage) {
        Ok(t) => t,
        Err(e) => return Outcome::fail(78, e),
    };
    let existing = read_existing(target);
    match classify_existing(&existing, &Desired::Script { text: text.clone() }) {
        Action::Refuse(_) => refuse_outcome(target),
        action => {
            if let Err(e) = std::fs::create_dir_all(target_dir) {
                return Outcome::fail(1, format!("create {}: {}", target_dir.display(), e));
            }
            if action != Action::AlreadyCurrent {
                if let Err(e) = atomic_write(target, &text, true) {
                    return Outcome::fail(1, format!("write {}: {}", target.display(), e));
                }
            }
            let hint = path_hint_text(current_os(), path_contains_dir(target_dir));
            let note = if action == Action::AlreadyCurrent { " (already installed)" } else { "" };
            Outcome::ok(format!("{} runs {}{}\n{}", target.display(), appimage.display(), note, hint))
        }
    }
}

/// Launch-time refresh: rewrite OUR shim when it is stale or lost its exec bit. Never creates one,
/// never touches a foreign file or a symlink.
pub fn refresh_shim_at(target: &Path, desired_text: &str) -> Outcome {
    let existing = read_existing(target);
    if !refresh_decision(&existing, desired_text) {
        return Outcome::ok("unchanged".to_string());
    }
    match atomic_write(target, desired_text, true) {
        Ok(()) => Outcome::ok("refreshed".to_string()),
        Err(e) => Outcome::fail(1, format!("refresh {}: {}", target.display(), e)),
    }
}

/// Linux .deb path, before the unchanged `install_symlink_to`: refuse a foreign file or symlink
/// (exit 73) and remove one of our own shim files so the symlink can take its place.
pub fn prepare_linux_symlink(target: &Path, bundled: &Path) -> Result<(), Outcome> {
    let existing = read_existing(target);
    match classify_existing(&existing, &Desired::Symlink { bundled: bundled.to_path_buf() }) {
        Action::Refuse(_) => Err(refuse_outcome(target)),
        _ => {
            if let Existing::File { .. } = existing {
                if let Err(e) = std::fs::remove_file(target) {
                    return Err(Outcome::fail(1, format!("remove {}: {}", target.display(), e)));
                }
            }
            Ok(())
        }
    }
}
