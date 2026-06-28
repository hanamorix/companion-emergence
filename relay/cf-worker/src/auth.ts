import { canonicalJson } from "./canonical";
import { verifyEd25519 } from "./crypto";
import { Store, RelayReject } from "./storage";

export async function checkAuth(store: Store, body: any, nowMs: number): Promise<void> {
  const { mailbox_id, nonce, signature, identity_pub } = body ?? {};
  const owner = await store.ownerOf(String(mailbox_id ?? ""));
  if (!owner || !nonce || !signature || !identity_pub) throw new RelayReject(401, "auth required");
  if (identity_pub !== owner) throw new RelayReject(401, "not mailbox owner");
  if (!(await store.nonceLive(mailbox_id, nonce, nowMs))) throw new RelayReject(401, "bad nonce");
  const msg = new TextEncoder().encode(canonicalJson({ purpose: "kindled-relay-auth/1", mailbox: mailbox_id, nonce }));
  if (!(await verifyEd25519(identity_pub, signature, msg))) throw new RelayReject(401, "bad signature");
  await store.discardNonce(mailbox_id, nonce); // single-use
}
