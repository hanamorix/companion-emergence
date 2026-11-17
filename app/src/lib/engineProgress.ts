// TESTS-ONLY COMMIT STUB: the old behaviour (a fixed budget, no status read). Replaced by the real helper in the next commit.
import type { EngineProgress } from "../bridge";
import { HEALTH_POLL_INTERVAL_MS } from "./requestTimeout";

export type EngineProgressCallback = (reading: EngineProgress | null) => void;

export async function waitForHealth(
  _persona: string,
  probe: () => Promise<unknown>,
  budgetMs: number,
  _onProgress?: EngineProgressCallback,
): Promise<boolean> {
  const deadline = Date.now() + budgetMs;
  while (Date.now() < deadline) {
    try {
      await probe();
      return true;
    } catch {
      await new Promise<void>((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
    }
  }
  return false;
}

export function alongside<T>(_persona: string, _onProgress: EngineProgressCallback | undefined, work: () => Promise<T>): Promise<T> {
  return work();
}
