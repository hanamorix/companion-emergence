/**
 * App-level configuration — selected persona, window prefs.
 *
 * Lives at <NELLBRAIN_HOME>/app_config.json so it sits alongside the
 * personas/ directory the brain owns. Read on app boot to decide
 * whether to show the wizard (no persona selected → first launch)
 * or the main Companion Emergence UI.
 */

import { invoke } from "@tauri-apps/api/core";

export interface AppConfig {
  selected_persona: string | null;
  always_on_top: boolean;
  reduced_motion: boolean;
}

const DEFAULT: AppConfig = {
  selected_persona: null,
  always_on_top: false,
  reduced_motion: false,
};

export async function readAppConfig(): Promise<AppConfig> {
  try {
    return await invoke<AppConfig>("read_app_config");
  } catch (e) {
    // Browser dev mode (no Tauri) — fall back to defaults.
    // Persona defaults to "nell" so the dev surface still works.
    console.warn("[appConfig] read_app_config failed, using defaults:", e);
    return { ...DEFAULT, selected_persona: import.meta.env.DEV ? "nell" : null };
  }
}

export async function writeAppConfig(config: AppConfig): Promise<void> {
  try {
    await invoke("write_app_config", { config });
  } catch (e) {
    console.warn("[appConfig] write_app_config failed (browser mode?):", e);
  }
}

export async function listPersonas(): Promise<string[]> {
  try {
    return await invoke<string[]>("list_personas");
  } catch (e) {
    console.warn("[appConfig] list_personas failed:", e);
    return [];
  }
}

export async function ensureBridgeRunning(persona: string): Promise<void> {
  try {
    await invoke("ensure_bridge_running", { persona });
  } catch (e) {
    // Browser dev mode — Tauri's `invoke` is undefined. Treat as a no-op
    // and assume the bridge is already running externally; the next
    // fetch through bridge.ts will surface a real failure if it isn't.
    if (import.meta.env.DEV) {
      console.warn("[appConfig] ensure_bridge_running unavailable in browser dev; assuming external bridge:", e);
      return;
    }
    throw e;
  }
}

/** Apply always-on-top to the main Tauri window. No-op in browser dev. */
export async function setAlwaysOnTop(value: boolean): Promise<void> {
  try {
    await invoke("set_always_on_top", { value });
  } catch (e) {
    if (import.meta.env.DEV) return;
    throw e;
  }
}

export interface InitArgs {
  persona: string;
  user_name: string | null;
  voice_template: "default" | "nell-example" | "skip";
  migrate_from: string | null;
  force: boolean;
}

export interface InitResult {
  success: boolean;
  stdout: string;
  stderr: string;
  exit_code: number;
}

export async function runInit(args: InitArgs): Promise<InitResult> {
  return await invoke<InitResult>("run_init", { args });
}

export interface ClaudeCliCheck {
  found: boolean;
  path: string | null;
  version: string | null;
}

/**
 * Probe the host for Anthropic's ``claude`` CLI — the LLM provider
 * the framework shells out to. Powers the wizard's prerequisites
 * step: when ``found`` is false the user gets install instructions
 * and re-checks until it passes. The Tauri side checks several
 * common install paths AND tries a bare ``claude --version`` so
 * Homebrew, ``~/.local/bin``, and ``/usr/local/bin`` installs all
 * resolve.
 */
export async function checkClaudeCli(): Promise<ClaudeCliCheck> {
  return await invoke<ClaudeCliCheck>("check_claude_cli");
}

/**
 * Install the launchd LaunchAgent for the persona's supervisor.
 *
 * Run this after a successful ``runInit`` so first-launch users land
 * on the launchd-managed-supervisor model directly. The supervisor
 * then survives .app quit/relaunch cycles via launchd's KeepAlive,
 * which solves the "talk to Nell, close the app, brain dies" issue
 * the audit cycle motivated.
 *
 * macOS-only — non-darwin platforms get a synthetic success and the
 * legacy Tauri-spawn-supervisor path keeps working unchanged.
 */
export async function installSupervisorService(persona: string): Promise<InitResult> {
  return await invoke<InitResult>("install_supervisor_service", { persona });
}
