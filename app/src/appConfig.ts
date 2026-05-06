/**
 * App-level configuration — selected persona, window prefs.
 *
 * Lives at <NELLBRAIN_HOME>/app_config.json so it sits alongside the
 * personas/ directory the brain owns. Read on app boot to decide
 * whether to show the wizard (no persona selected → first launch)
 * or the main NellFace UI.
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
  await invoke("ensure_bridge_running", { persona });
}

export interface InitArgs {
  persona: string;
  user_name: string | null;
  provider: string | null;
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
