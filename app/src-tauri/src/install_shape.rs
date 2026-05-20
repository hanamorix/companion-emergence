//! Detect how the current binary was installed on Linux.
//!
//! `appimage`  → running from an AppImage (APPIMAGE env var present)
//! `deb`       → /proc/self/exe under /usr/, /opt/
//! `native`    → macOS / Windows (no detection needed)
//! `unknown`   → Linux but exe path under /home/, /tmp/, etc.

use std::path::Path;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InstallShape {
    AppImage,
    Deb,
    Native,
    Unknown,
}

impl InstallShape {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AppImage => "appimage",
            Self::Deb => "deb",
            Self::Native => "native",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(target_os = "linux")]
pub fn detect() -> InstallShape {
    if std::env::var("APPIMAGE").is_ok() {
        return InstallShape::AppImage;
    }
    match std::fs::read_link("/proc/self/exe") {
        Ok(target) => classify_exe_path(&target),
        Err(_) => InstallShape::Unknown,
    }
}

#[cfg(not(target_os = "linux"))]
pub fn detect() -> InstallShape {
    InstallShape::Native
}

fn classify_exe_path(p: &Path) -> InstallShape {
    let s = p.to_string_lossy();
    if s.starts_with("/usr/") || s.starts_with("/opt/") {
        return InstallShape::Deb;
    }
    if s.starts_with("/tmp/.mount_") {
        return InstallShape::AppImage;
    }
    InstallShape::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_usr_bin_is_deb() {
        assert_eq!(classify_exe_path(Path::new("/usr/bin/companion-emergence")), InstallShape::Deb);
    }

    #[test]
    fn classify_usr_lib_is_deb() {
        assert_eq!(classify_exe_path(Path::new("/usr/lib/companion-emergence/bin/app")), InstallShape::Deb);
    }

    #[test]
    fn classify_opt_is_deb() {
        assert_eq!(classify_exe_path(Path::new("/opt/companion-emergence/app")), InstallShape::Deb);
    }

    #[test]
    fn classify_appimage_mount_is_appimage() {
        assert_eq!(classify_exe_path(Path::new("/tmp/.mount_compXYZ/AppRun")), InstallShape::AppImage);
    }

    #[test]
    fn classify_home_is_unknown() {
        assert_eq!(classify_exe_path(Path::new("/home/user/Downloads/Companion.AppImage")), InstallShape::Unknown);
    }
}
