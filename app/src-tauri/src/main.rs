// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // CLI mode (`--companion-nell-cli`, used by the Linux AppImage `nell` shortcut) must be handled
    // before ANY GUI or plugin code runs, so no Tauri argument handling can see the flag.
    if let Some(code) = nell_shortcut::try_run_nell_cli() {
        std::process::exit(code);
    }
    nellface_lib::run();
}
