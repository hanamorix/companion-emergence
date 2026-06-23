# 1 — Spec: Kindled privacy-gate hardening (m9 + m10)

## Problem

Two deferred safety-spine hardening items on `brain/kindled_link/privacy_gate.py`
(the gate is the load-bearing privacy spine; parent design §5/§12). Both are
defence-in-depth: the LLM reflection remains the primary filter, the deterministic
layers are the backstops.

### m9 — pre-filter credential-family breadth (defence-in-depth)

`_prefilter` (`privacy_gate.py:46`) deterministically catches hard structural
leaks before any LLM call. Today `_PATTERNS` (`privacy_gate.py:25-37`) covers:
POSIX/Windows/home/`file://` paths, email, OpenAI `sk-` keys, a generic
`bearer|api_key|token|secret = <16+ char value>` assignment, and PEM key headers.

Gap: several **high-signal, vendor-prefixed credential families** are not caught
deterministically — they reach the gate only via the LLM reflection. If the LLM
reflection ever declines/defers (throttle, provider error → fail-closed hold, so
not a *leak*, but a missed deterministic catch) or mis-classifies a draft that
embeds a literal credential, the deterministic backstop is thinner than it could
be. The missing families are structurally unambiguous (vendor prefix + fixed
shape), so catching them adds near-zero false-positive risk.

Missing families (all credential/secret, NOT general PII — PII stays the LLM's job):
- AWS access-key IDs (`AKIA…`/`ASIA…` + 16 base32). **Only these two prefixes** —
  they are the access-key (credential) prefixes. AWS resource-id prefixes
  (`AGPA/AIDA/AROA/ANPA/AIPA/ANVA` = group/user/role/etc IDs) are NOT credentials
  and are pronounceable enough to collide with prose, so they are excluded (stage-3
  red-team minor).
- GitHub tokens (`ghp_/gho_/ghu_/ghs_/ghr_…`, `github_pat_…`)
- Slack tokens (`xoxb-/xoxp-/xoxa-/xoxr-/xoxs-…`)
- Google API keys (`AIza…`)
- Stripe keys (`sk_live_/sk_test_/rk_live_…`). Stripe `whsec_` webhook secrets are
  a real family but omitted this pass (lower signal; revisit if needed).
- JWTs (`eyJ….eyJ….<sig>`)

**JWT false-positive profile (stage-3 red-team major, design call made).** `eyJ` is
literally base64 of `{"`, so a `relationship_hint` value that is itself base64-encoded
JSON (the one structured-data surface `_payload_text` folds into the scanned text,
`privacy_gate.py:162`) can begin `eyJ…` and trip the JWT pattern as a *false* hit.
**This is accepted, not a defect:** a false hit on `_prefilter` fails **toward
hold/revise** — it never loosens a decision toward send (the founding invariant). For
a privacy gate, an over-cautious hold on a base64 blob is a non-event (the Kindled
simply doesn't send that turn). The earlier "near-zero FP for JWT" framing was
overstated; corrected here to "JWT may false-positive on base64-of-JSON; accepted
because it fails toward hold." A C-m9-2 corpus line pins this as accepted behaviour.

Out of scope (deliberately): generic high-entropy hex/base64, phone/SSN/credit-card/
IP. Those are PII-family (not credentials) and FP-prone on interior prose; the LLM
reflection owns them. Keeping m9 to vendor-prefixed credential shapes preserves the
M4 lesson (no tripping on ordinary introspective prose).

### m10 — budget depletion → hold (currently only send→revise)

`_apply_budget` (`privacy_gate.py:138`) tightens a model `send` to `revise` when
the per-peer disclosure budget is below `BUDGET_TIGHTEN_THRESHOLD` (0.25). It never
hardens to `hold`, even at a fully-exhausted budget.

The gap is a crumb-extraction vector over a long correspondence: at depleted budget
(≈0.0), a model `send` becomes `revise`; the session engine then `_regenerate`s a
"say less" revision and **re-gates** it (`session_engine.py:198-219`). If the re-gate
returns `send`, `_act_on_decision` **sends and debits** (`session_engine.py:188-191`,
with the `MIN_SEND_DEBIT=0.02` floor). So an exhausted budget still permits one more
revised, low-texture send per outbound attempt — exactly the slow-leak the budget
exists to bound. A peer who keeps the correspondence going indefinitely keeps
extracting `MIN_SEND_DEBIT`-floored crumbs.

Fix intent: below a hard **depletion floor** (`budget < BUDGET_DEPLETED_THRESHOLD`,
i.e. can't even afford the minimum debit), `_apply_budget` returns `hold` — a full
stop, no revised send. Between the floor and the tighten threshold, keep current
`revise` behaviour. Above tighten, keep `send`.

## The invariant this must preserve

**No peer-derived content may flip a hold/revise into a send** (parent §5,
`privacy_gate.py:4-5`). Both changes only *tighten* (add catches / harden send→hold);
neither adds any path that loosens a decision toward send, and neither introduces
peer-transcript text into the deterministic layers (`_prefilter` and `_apply_budget`
never see `transcript_summary`). m10 acts only on `decision.action == "send"`.

## Constraints

- `privacy_gate.py` may import ONLY stdlib + `{gate, limits, cli_throttle}` (the AST
  conformance oracle enforces this — `privacy_gate.py:6-8`). m9/m10 add no imports.
- Off-chat-path, DORMANT, off-by-default: zero standing-persona impact; the Layer-2
  cost/cache regression metrics do not apply (not a comparable workload).
- Pure-function changes (`_prefilter`, `_apply_budget` + one `limits` constant) —
  directly unit-assertable, no new instrumentation needed.

## Prior art

- M4 red-team lesson (credential-shaped value required): `privacy_gate.py:32-35`,
  `test_privacy_gate_prefilter.py:29-36`.
- m8 (missing/malformed texture_score → 1.0 safe default): `privacy_gate.py:120-126`.
- Existing budget behaviour + test: `_apply_budget`, `test_privacy_gate_budget.py`.
- Deferred ledger items m9/m10: `project_companion_emergence_kindled_link.md`,
  `project_companion_emergence_deferred.md` (Phase-4 §10 carryover).
