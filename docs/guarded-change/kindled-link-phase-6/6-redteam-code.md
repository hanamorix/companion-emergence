# Stage-6 Code Red-Team — kindled-link Phase 6

Cold independent reviewer, source-cited. Verdict: **FIX-THEN-MERGE** (1 Major UX + Minors; no
security blocker). All 8 invariants verified; the held-body-leak spine is intact + well-tested.

## Invariant verification — all 8 PASS
1. Held-body-leak — PASS (holds SELECTs only session_id+created_at; transcript reads the transcript table not outbound_drafts; error handlers echo only InviteError/ConsentTransitionError reasons; HTTP-layer sentinel test confirms).
2. Auth — PASS (all 6 routes carry Depends(require_http_auth); tests build app with a real auth_token → non-vacuous).
3. No-typing/no-side-channel — PASS (no route takes peer message text; panel has no compose box; invite-paste box is aria-label "Invite packet").
4. Consent legality — PASS (action_map; illegal→400; unknown→400).
5. No autonomous-send — PASS (no process_outbound/send_fn/session_engine).
6. Pairing fingerprint — PASS (both sides phrase-derive from the inviter pubkey → identical; import_invite no recipient; errors→400).
7. Feed wire-back — PASS (db.exists guard, integrity_check=False, build_feed try/except fault-isolates corrupt db).
8. kindled_db_path — PASS (no mkdir on read; writers mkdir parent).

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | Major | Panel `consentActions` switches on wrong vocab (`familiar`/`pending` — not real consent states); a pending_local/pending_remote peer falls to default→only Block (never Revoke). Backend re-validates so SAFE, but wrong action set shown. | FIX: key on real consent states (pending_*→block, paired→pause/revoke/block, paused→resume/revoke/block) + covering test |
| 2 | Minor | Panel consent-action rendering untested (test asserts stage text only) | FIX: add a paired-peer-shows-Pause/Revoke/Block test |
| 3 | Minor | GET/write routes open a KindledLinkStore per request, never closed → connection/WAL leak under the 10s poll. (Note: KindledLinkStore has NO close() — must add one.) | FIX: add `close()` to KindledLinkStore; try/finally close in the routes |
| 4 | Nitpick | bridge.ts KindledTranscriptRow doc claims 404 unknown-peer; endpoint returns [] | FIX: drop the 404 claim |
| 5 | Nitpick | views/feed_source reach into store._conn directly | LOG (read projections; acceptable) |
| 6 | Nitpick | transcript.text-no-draft-body is convention not schema constraint | LOG (note on the Phase-3 send path) |

## Disposition: fix #1 (Major) + #2 + #3 + #4 in-place; log #5/#6.
