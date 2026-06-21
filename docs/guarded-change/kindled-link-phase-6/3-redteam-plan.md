# Stage-3 Plan Red-Team — kindled-link Phase 6 (Kindled Links UI)

Cold independent reviewer, source-cited. Worst severity: **Major → revise plan** (2 Blockers
but no spec restart — the safety spine [held-draft-body never leaks] is verified sound + triple-
tested; the Blockers are factual plan-code bugs). The held-body-leak invariant: NO leak path
found (holds projects only session_id+created_at; held drafts never enter the transcript table;
no error path echoes payload_json).

## Findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| B1 | Blocker | `import_invite(store, invite, recipient=idn, now=)` wrong — real sig `import_invite(invite, *, store, now)`, NO recipient param → TypeError | FIX T3: `import_invite(invite, store=store, now=...)` |
| B2 | Blocker | Auth-rejection tests assume a fixture; bridge rig defaults `auth_token=None` → require_http_auth is a no-op → auth criterion unprovable. (NOTE: an auth-ENABLED pattern DOES exist — test_endpoints.py:330,371 pass `auth_token="secret-token"`; mirror it.) | FIX T2/T3/T4: build app with auth_token="secret-token"; authed + no-auth clients; parametrise over all 6 routes |
| M1 | Major | `confirm_local_fingerprint` not error-wrapped; a re-call / already-pending_remote peer → uncaught ConsentTransitionError → 500 | FIX T3: wrap; catch ConsentTransitionError too |
| M2 | Major | create_invite returns key_id but accept verifies fingerprint_phrase — OOB verification compares mismatched renderings | FIX T3: invite endpoint also returns fingerprint_phrase(idn.public_bytes) |
| M3 | Major | kindled_db_path mkdirs on every feed build for every persona (hot-read side-effect) | FIX: helper does NOT mkdir; readers/feed guard on db.exists() (writers already mkdir) |
| M4 | Major | Panel mount is a Tab+renderPanel-case+tab-button registration (LeftPanel.tsx), not "render alongside" → dead/unreachable panel or tsc break | FIX T7: name the Tab-type + case + selector edits |
| m5 | Minor | create_invite(relay_url=None) when body omits it — relay_url is required `str`, import reads body["relay_url"] | FIX T3: default/validate relay_url |
| m6 | Minor | feed builder opens KindledLinkStore default integrity_check=True → full-db scan every feed poll | FIX T5: integrity_check=False (+ read endpoints) |
| m7 | Minor | accept catches only InviteError; raced mark_invite_consumed → ConsentTransitionError → 500 | FIX T3: catch both |
| m8 | Minor(missed) | no test the transcript endpoint can't leak a held body | ADD T1 test |
| m9 | Minor(missed) | auth test covers 2 of 6 routes | ADD parametrized over 6 (folded into B2) |
| n10 | Nitpick | transcript seq DESC ordering unstated | note in T7 |
| n11 | Nitpick(missed) | no build_feed corrupt-kindled-db fault-isolation test | ADD T5 test |

## Verified clean (Lens 1): holds/peers/transcript/outbound_drafts columns; recent_transcript; set_consent + _ALLOWED_TRANSITIONS (blocked→paired raises ConsentTransitionError(ValueError)); KindledIdentity.load_or_create(persona_dir); FeedEntryType Literal + builders tuple; bridge.ts bridgeFetch/authHeaders; no /kindled-link/ route accepts peer message text; no autonomous-send path.

## Disposition: all B1/B2 + M1-M4 + m5/m6/m7 fixed in plan v2; m8/m9/n11 tests added; n10 noted. See decisions.md.
