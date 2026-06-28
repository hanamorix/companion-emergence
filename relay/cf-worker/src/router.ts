import { Env } from "./index";
import { Store, RelayReject } from "./storage";
import { checkAuth } from "./auth";

function bytesToHex(b: Uint8Array): string { return [...b].map((x) => x.toString(16).padStart(2, "0")).join(""); }
function newNonce(): string { return bytesToHex(crypto.getRandomValues(new Uint8Array(16))); }

export async function handle(request: Request, env: Env, store: Store, nowMs: number): Promise<Response> {
  const url = new URL(request.url);
  if (request.method !== "POST") return new Response("not found", { status: 404 });
  let body: any;
  try { body = await request.json(); } catch { return Response.json({ detail: "bad json" }, { status: 400 }); }

  try {
    switch (url.pathname) {
      case "/mailbox/register": {
        const owner = await store.register(String(body.mailbox_id), String(body.identity_pub), nowMs);
        return Response.json({ ok: true, owner });
      }
      case "/mailbox/challenge": {
        if (!(await store.isRegistered(String(body.mailbox_id))))
          return Response.json({ detail: "mailbox not registered" }, { status: 404 });
        const nonce = newNonce();
        await store.issueNonce(String(body.mailbox_id), nonce, nowMs);
        return Response.json({ nonce });
      }
      case "/envelope": {
        const id = await store.push(body as Record<string, unknown>, nowMs);
        return Response.json({ id });
      }
      case "/mailbox/fetch": {
        await checkAuth(store, body, nowMs);
        return Response.json({ envelopes: await store.fetch(String(body.mailbox_id)) });
      }
      case "/mailbox/ack": {
        await checkAuth(store, body, nowMs);
        await store.ack(String(body.mailbox_id), (body.envelope_ids ?? []).map(String));
        return Response.json({ ok: true });
      }
      default:
        return new Response("not found", { status: 404 });
    }
  } catch (e) {
    if (e instanceof RelayReject) return Response.json({ detail: e.detail }, { status: e.status });
    throw e;
  }
}
