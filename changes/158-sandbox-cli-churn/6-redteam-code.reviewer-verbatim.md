# Stage-6 Code Red-Team — #158 (sandbox CLI-churn false positive)

Cold, independent reviewer. No shared context with the author. Adversarial read of the built code
against the plan + criteria, reading real source and running the suite.

## 1. Files read + sha256 + reviewer identity

- Agent type: **general-purpose**  |  Model: **opus**

| File | sha256 |
|---|---|
| `changes/158-sandbox-cli-churn/6-code-diff.patch` | `f51ceaa380739b8f101ac52f2dbc66d75c6b3efb4fba911aa445b92b1c0447e6` |
| `tests/harness/sandbox.py` (full, restored to pristine after a revert-probe) | `1a0bf2632c420fc4d2d600c91da9736afbb9d29f0605101f3cec848158ef11fc` |
| `tests/unit/harness/test_sandbox_isolation.py` (full) | `cc11c4ed568ebfc212b441a27c7d1816061133ac140b5933da3a6583656cf40e` |
| `changes/158-sandbox-cli-churn/2-plan.md` | `b54c3762794b1a9c2c2de5bebf099eaacc205c326a9528aec617428d8bb31d6d` |
| `changes/158-sandbox-cli-churn/1.5-criteria.md` | `b022d1000cf7061bba4e3e231d7773edfddd47822ff9d4a244fd2dd8f7a70a0e` |
| `brain/bridge/provider.py` (redteam_context, spot-read) | `3e1146f2a9a208db7e4784153b3865e0f2e1322f345b390baf1ea5df6bfdd39b` |

Base commit for the diff: `cde2eb468554fcf672d820150fecfbf48072acc5` (confirmed a valid commit;
`git diff --name-only` vs base returns exactly the two harness files — see §5).

**Discrimination probe run (and cleaned up):** I temporarily neutralized the fix branch
(`if False and mtime_ins_names ...` at sandbox.py:480), ran the new tests, confirmed the intended
tests fail, then restored the file. Post-restore sha256 matches the pristine value above, so the
working tree is unchanged by my review.

---

## 2. The five lenses

### Lens 1 — Factual (code vs plan + criteria)

**No blocking issue. Clean, with citations.**

- New constant `_CLAUDE_CLI_HOUSEKEEPING_FILES = ("policy-limits.json", "remote-settings.json")` at
  sandbox.py:293-296 — matches plan step 1 and G5 verbatim.
- Helper `_claude_cli_housekeeping_files()` at sandbox.py:299-310 derives `Path.home()/".claude"/name`
  for each — matches plan step 1 and G5.
- New `mtime_insensitive=` param on `_fingerprint` at sandbox.py:412-418; honored **only** when
  `hash_content` is True (sandbox.py:446-448: `if (mtime_insensitive and hash_content)`) — matches
  plan step 2 and the docstring at 431-438.
- Entry-building: mtime-insensitive files get `(st.st_size, None, _content_hash(f))`
  (sandbox.py:480-481); all others keep `(st.st_size, st.st_mtime_ns, _content_hash(f))`
  (sandbox.py:482-483) — matches plan step 2 exactly (nulled mtime slot, kept sha256, 3-tuple arity).
- Wire-in in `_snapshot` at sandbox.py:906-915 passes `mtime_insensitive=_claude_cli_housekeeping_files()`
  **only** when `gr == claude_root`, else `None` — matches plan step 3.
- Comment (A1) at sandbox.py:272-292 states the real safety argument (server-pushed config, per-run
  churn, Canary CAN write an absolute `~/.claude` path so a full prune would lose the warn floor),
  not the borrowed F4 "provably-cannot-write" claim — satisfies advisory A1.

The safety argument in the comment is **verified against source**: `brain/bridge/provider.py`
`generate()` (provider.py:518-536) and the text-`chat()` path build the `claude -p` command with
`--dangerously-skip-permissions` and **never** call `_apply_lean_flags` (provider.py:263-268), so
`--disallowedTools` is not applied on those paths — Bash/Write/Edit remain callable, meaning the
Canary genuinely can write an absolute `~/.claude` path. `_apply_lean_flags` (the disallow-list) is
applied only on the MCP-routed paths. This is exactly the premise the "do NOT full-prune" decision
rests on, and it holds.

### Lens 2 — Logical

Scrutinized the four charter sub-points:

**(a) Entry-tuple shape confusion / masked change — NO ISSUE.**
Three shapes exist: `(size, mtime_ns)` (2-tuple, no-hash root), `(size, mtime_ns, sha256)` (normal
hashed), `(size, None, sha256)` (mtime-insensitive). A given root's `before`/`after` snapshots both
pass the *same* `mtime_insensitive` list and the same `hash_content` value (both derive from
`_snapshot()` closure state), so a given key is consistently the same shape on both sides — there is
no before/after cross-shape comparison. `st.st_mtime_ns` is always an int, never `None`, so the
`None` sentinel is unambiguous and cannot collide with a real mtime. A content change with unchanged
size still flips the `sha256` slot → detected; a size change flips the size slot → detected. Nothing
is masked except the exact targeted case (identical bytes, bumped mtime).

**(b) Name-prefilter exactness — NO ISSUE, and test-pinned.**
Match is `mtime_ins_names and name in mtime_ins_names and f.resolve() in mtime_ins` (sandbox.py:480).
The basename gate is a cheap prefilter; the load-bearing check is `f.resolve() in mtime_ins`, an
exact resolved-path set membership. A same-basename file in a subdir (`~/.claude/plugins/policy-limits.json`)
resolves to a different absolute path, is not in `mtime_ins`, and keeps its mtime-sensitive entry.
Directly proven by `test_same_basename_in_subdir_stays_mtime_guarded` (see §3, G4b) and by my revert
probe, where that test fails once the branch is neutralized.

**(c) Honored only with `hash_content=True` — CORRECT condition.**
sandbox.py:446-448 gates `mtime_ins` on `(mtime_insensitive and hash_content)`. This is the right
condition: without a content hash, dropping mtime would leave a size-only entry (too weak). Since the
only caller passes `mtime_insensitive` exactly when `gr == claude_root` (sandbox.py:912-913), and
`_hash_critical(claude_root)` is True (sandbox.py:406-409), the two always co-occur in production, so
the gate never silently no-ops the intended path. Belt-and-suspenders, correct.

**(d) Creation/deletion of a housekeeping file — STILL DETECTED.**
The before/after comparison is at the *root* granularity (`before[g] != after.get(g)`,
sandbox.py:975), comparing whole fingerprint dicts. Absent→present adds a new dict key → dicts differ
→ detected (then, being a `~/.claude`-only diff, downgraded to a warn per option-(c), which is within
the owner-accepted envelope). Deletion removes a key → detected. Only a pure mtime bump on an existing
identical-byte file is silenced. Correct and consistent with the plan's stated intent.

### Lens 3 — Missed opportunity

**NITPICK only.** The `mtime_ins_names and` leading conjunct at sandbox.py:480 is a redundant
micro-guard: if `mtime_ins_names` is empty, `name in mtime_ins_names` is already False. It's a
harmless short-circuit optimization mirroring the `exclude` prefilter style, not a defect. No change
warranted.

No cleaner mechanism was left on the table. The chosen approach (content-aware mtime-null in the
existing per-file loop) reuses the already-computed content hash and the existing name-prefilter,
adds no new walk, and preserves the warn floor — which a full name-prune (the rejected alternative)
could not.

### Lens 4 — Unstated assumptions & risks (incl. position/concurrency)

- **Concurrency / shared mutable state — NO NEW ACCESSOR.** The change is a pure function extension of
  `_fingerprint` plus one keyword-arg at the single `_snapshot` call site. It introduces no new
  read-modify-write over shared state. `sandbox()` is already documented not-thread-safe / not-nestable
  (sandbox.py:938-939); this change does not alter that surface.
- **Position sensitivity — NOT position-sensitive.** `_fingerprint` builds a dict keyed by relpath;
  entry order is irrelevant to the equality compare. The `mtime_insensitive` set is order-independent.
- **Residual detection blind spot (owner-accepted).** The one genuine risk — a real Canary write to
  these two exact paths that changes *only* mtime, not bytes — is impossible in practice (a real write
  changes bytes → sha256 flips → detected). A write that produces byte-identical content is not a
  meaningful escape. This is strictly narrower than the pre-existing option-(c) downgrade already
  accepted by the owner. No new risk beyond what's documented.

### Lens 5 — Fidelity (mechanism, not proxy)

The code implements the **content-aware mtime-insensitivity** the plan specifies, not a convenient
proxy. Named loaded terms and where each is honored:

- "mtime-insensitive but content-sensitive" → `(size, None, sha256)` at sandbox.py:481. ✔ real.
- "targeted to two exact top-level paths, resolved-path match" → `f.resolve() in mtime_ins`
  (sandbox.py:480), set built from `_claude_cli_housekeeping_files()`. ✔ real, test-pinned (G4b).
- "same warn floor" → the fix touches only `_fingerprint`'s recorded entry; the post-run raise/warn
  block is untouched (see §4). ✔ real.

**HEADLINE FIDELITY CHECK — raise-vs-warn semantics untouched: CONFIRMED.** The diff contains no
`+`/`-` hunk anywhere in the post-run block (sandbox.py:976-1021). The `changed == [str(claude_root)]`
option-(c) condition (sandbox.py:1009), the `warnings.warn(... "DOWNGRADED to a warning" ...)`
(sandbox.py:1010-1019), and the `raise SandboxLeak(msg)` (sandbox.py:1021) are byte-for-byte as at
base. The two files are made mtime-insensitive via the `_fingerprint` path, **not** added to any
exclude/allowlist — see §4.

---

## 3. Per-gating-criterion test audit

I ran the new tests (6 pass), the full file (31 pass, ruff clean), and a **revert probe** (neutralized
the fix branch, re-ran) to confirm which tests actually discriminate.

| Crit | Test | Exercises governed path? | Discriminates (fails if fix reverted)? |
|---|---|---|---|
| **G1** | `test_cli_housekeeping_mtime_bump_is_silent` | YES — real `sandbox()` run, records warnings, asserts zero guard warning + clean exit. Has a genuine oracle-can-fail preface (test:185-189) asserting a mtime-SENSITIVE `_fingerprint` DOES flip. | **YES — FAILS when reverted** (confirmed). |
| **G1b** | `test_cli_housekeeping_content_change_still_warns` | YES — real `sandbox()` + `pytest.warns(match="DOWNGRADED to a warning")` on a content rewrite; plus a direct-`_fingerprint` oracle on the content axis (test:223-227). `match` string is the real code path (sandbox.py:1012). | Floor-preservation property — passes with OR without the fix (correct: content change warns either way). Not a fix-discriminator by design; it guards the floor. |
| **G2** | `test_non_housekeeping_claude_file_still_surfaces` (body 1, `settings.json`) | YES — real run, warns. | Passes both states (correct — non-housekeeping always mtime-sensitive). |
| **G3** | `test_housekeeping_churn_alongside_real_escape_still_hard_raises` | YES — real `sandbox(extra_guard_roots=[guarded])`, churns the two files AND mutates a second guarded root, asserts `pytest.raises(SandboxLeak)`. Proves the non-`~/.claude` escape still hard-raises. | Passes both states (correct — safety invariant that must hold regardless). |
| **G4** | `test_non_housekeeping_claude_file_still_surfaces` (body 2, `some-other.json`) | YES — a THIRD non-named top-level `.json` still warns → no blanket `~/.claude`/`*.json` prune. | Passes both states (correct). |
| **G4b** | `test_same_basename_in_subdir_stays_mtime_guarded` | YES — direct-`_fingerprint` scope pin (top-level entry unchanged on mtime bump, nested `plugins/policy-limits.json` entry changed) AND a real `sandbox()` run where the nested bump warns. | **YES — FAILS when reverted** (confirmed; the "top-level entry must ignore mtime" assert fires). |
| **G5** | `test_cli_housekeeping_constant_and_helper_pinned` | YES — pins the constant to exactly the two names and the helper to exactly those two under `~/.claude`. | Catches silent narrowing/widening of the set. |
| **G6** | `test_cli_housekeeping_constant_and_helper_pinned` (same test) | YES — asserts the two names are NOT in `_CLAUDE_SESSION_LOG_FILES`, NOT in `_CLAUDE_SESSION_LOG_DIRS`, NOT in `_claude_session_log_excludes()` names → they ride the mtime-insensitive path, not the exclude list. Plus `test_af2_exclusion_set_unchanged` (existing) stays green. | Guards the semantic split. |
| **G7** | (diff review + existing downgrade/scope tests green) | The post-run block is untouched by the diff (§4); existing option-(c) tests remain in the 31 passing. | N/A — verified by diff, not a new test. |

**Green-for-the-wrong-reason check.** None of the fix-critical tests (G1, G4b) pass vacuously — both
fail under revert. G1b/G2/G3/G4 are *floor/safety-preservation* tests that legitimately hold in both
states; that is their intended role (prove the fix didn't break existing behavior), not a discrimination
gap. The suite as a whole has a real discriminator for the mechanism (G1 + G4b) and real floor guards
(G1b/G2/G3/G4) — the correct division.

**Oracle-can-fail quality.**
- G1 preface (test:185-189): a mtime-sensitive `_fingerprint(hash_content=True)` before/after the
  bump is asserted to differ — a genuine can-fail oracle proving the churn WOULD trip pre-fix.
- G1b (test:223-227): a direct `_fingerprint(..., mtime_insensitive=ins)` before/after a *content*
  change is asserted to differ — the content-axis oracle. `pytest.warns(match="DOWNGRADED to a
  warning")` matches the real warning text at sandbox.py:1012 — not a proxy string.
- G4b (test:278-285): pins that the top-level entry ignores mtime while the nested same-basename entry
  changes — the exact resolved-path-scope property. This is the half that fails under revert.

**Existing tests broken/weakened by the diff — NONE.** The only non-append changes to the test file
are additive imports (`import warnings`; `_CLAUDE_CLI_HOUSEKEEPING_FILES`, `_claude_cli_housekeeping_files`
added to the existing `from tests.harness.sandbox import (...)` block). No existing test body was
touched. Full file: 31 passed, 0 failed, ruff clean.

---

## 4. Raise-vs-warn semantics + F4 exclude sets untouched

- **Raise/warn block:** the diff has zero hunks in sandbox.py:976-1021. `changed == [str(claude_root)]`,
  the `"DOWNGRADED to a warning"` `warnings.warn`, and `raise SandboxLeak(msg)` are byte-identical to
  base. **CONFIRMED untouched.**
- **F4 exclude sets:** `_CLAUDE_SESSION_LOG_DIRS` (sandbox.py ~219-249), `_CLAUDE_SESSION_LOG_FILES`
  (sandbox.py:250-255), and `_claude_session_log_excludes()` (sandbox.py:258-269) are all above the
  first diff hunk (which begins *after* line 269 and only ADDS a new constant/helper). No entries added
  or removed. The two housekeeping files are made mtime-insensitive via `_fingerprint`'s new param,
  **not** added to `_CLAUDE_SESSION_LOG_FILES` or the exclude list — explicitly asserted by G6.
  **CONFIRMED unchanged.**
- **Diff file set:** exactly `tests/harness/sandbox.py` + `tests/unit/harness/test_sandbox_isolation.py`.
  `git diff --name-only cde2eb46...` returns those two only. **No `brain/**` change** (harness-only, as
  required).

---

## 5. Verdict

**CLEAN** (worst severity: NITPICK — a single redundant short-circuit conjunct at sandbox.py:480, no
action warranted). The code implements the plan's content-aware, mtime-insensitive mechanism exactly;
the raise-vs-warn semantics and F4 exclude sets are provably untouched; the safety argument for not
full-pruning is verified against `brain/bridge/provider.py`; and the tests genuinely discriminate (G1
and G4b fail on a revert probe) while the floor/safety tests (G1b/G2/G3/G4) correctly hold in both
states — 31/31 pass, ruff clean.
