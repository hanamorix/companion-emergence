# decisions.md — kindled-link Phase 6 (append-only gate log)

## Gate: stage-3 plan red-team (2026-06-21)

- **Worst finding severity:** Blocker ×2 (but route = revise plan, no spec restart — safety spine sound)
- **Route taken:** revise plan (stage 2) → plan v2
- **Bounce count at this gate:** 1
- **Findings:** 2 Blocker, 4 Major, 4 Minor, 2 Nitpick — see `3-redteam-plan.md`

### Resolutions in plan revision v2

| # | Resolution |
|---|---|
| B1 | T3 fixed: `import_invite(invite, store=store, now=...)` (no recipient param) |
| B2 | T2/T3/T4 auth tests build the app with `auth_token="secret-token"` (mirroring the existing test_endpoints.py:330/371 pattern); a no-auth/wrong-token client asserts 401/403; a parametrized test covers all 6 routes |
| M1 | T3 wraps `confirm_local_fingerprint` + catches `ConsentTransitionError` → 400 |
| M2 | T3 invite endpoint also returns `fingerprint_phrase(idn.public_bytes)` so both panel sides verify the SAME phrase |
| M3 | `kindled_db_path` does NOT mkdir (writers already create the dir); read endpoints + feed builder guard on `db.exists()` |
| M4 | T7 names the explicit mount edits: extend the `Tab` union, add a `renderPanel` case, add a tab-strip button (LeftPanel.tsx) |
| m5 | T3 defaults relay_url to "" (or validates) before create_invite (relay_url is a required str) |
| m6 | T5 feed builder (+ read endpoints) open KindledLinkStore with `integrity_check=False` (hot-path) |
| m7 | T3 accept catches both InviteError and ConsentTransitionError → 400 |
| m8 | T1 adds a transcript-held-body-absence test |
| m9 | folded into B2 (parametrized auth over 6 routes) |
| n11 | T5 adds a corrupt-kindled-db build_feed fault-isolation test |
| n10 | T7 notes transcript seq-DESC ordering |

Next: build via subagent-driven-development (stage 5), then stage-6 cold code red-team.
