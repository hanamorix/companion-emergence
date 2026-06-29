import { Store } from "./storage";
import { handle } from "./router";

export interface Env {
  DB: D1Database;
  RATE_LIMITER: { limit: (opts: { key: string }) => Promise<{ success: boolean }> };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/healthz") return Response.json({ ok: true });
    return handle(request, env, new Store(env.DB), Date.now());
  },

  async scheduled(_event: ScheduledEvent, env: Env, _ctx: ExecutionContext): Promise<void> {
    const store = new Store(env.DB);
    const summary = await store.gc(Date.now());
    console.log("[gc]", JSON.stringify(summary));
  },
};
