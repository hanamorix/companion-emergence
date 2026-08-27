# 3 — Cold red-team of the PLAN (#158)

Reviewer: **general-purpose**, model **opus**. Cold/independent — no shared context with the author.

## 1. Files read + sha256

| File | sha256 | How read |
|---|---|---|
| `changes/158-sandbox-cli-churn/1-spec.md` | `a12bc0cf94323aa74ea89ef9bac76ef52ce188c1a990969b4e2843950d49dec0` | full |
| `changes/158-sandbox-cli-churn/1.5-criteria.md` | `3ec242ef76d1deacd0dee247c21dee54ee1d0ea8fca27fefe214c5b03337ddc5` | full |
| `changes/158-sandbox-cli-churn/2-plan.md` | `505e5db3077bb6dfa655351f679bfceec61f62aa5f27b51e473d0e64714fb39f` | full |
| `tests/harness/sandbox.py` | `2d9766d93990d2e5337901cec58d03cf6f87359652826a1ee9eba3104e10b7ca` | full |
| `tests/unit/harness/test_sandbox_isolation.py` | `d547a2350ddffc919fc94cb7fb5b787d73400782624614d0096052bb0d1eb744` | full |
| `brain/bridge/provider.py` (redteam_context) | `3e1146f2a9a208db7e4784153b3865e0f2e1322f345b390baf1ea5df6bfdd39b` | targeted (lines 124–180, 233–267, 531/671/803/862/1151/1216/1383/1391) to verify the safety-argument citations |

---

## 2. The five lenses

### Lens 1 — Factual (clean, with source evidence)

I checked every load-bearing factual claim in the spec/plan against the code:

- **The exclusion mechanism works exactly as claimed.** `_claude_session_log_excludes()` (sandbox.py:258–269) returns `[base / name for name in (*_CLAUDE_SESSION_LOG_DIRS, *_CLAUDE_SESSION_LOG_FILES)]`. The plan splices `*_CLAUDE_CLI_HOUSEKEEPING_FILES` into that tuple. The per-file exclusion in `_fingerprint` (sandbox.py:403–409) is `if name in exclude_names and f.resolve() in excludes: continue`, evaluated on the first `os.walk` iteration where `dirpath == root`. Both `policy-limits.json` and `remote-settings.json` are top-level files directly under `~/.claude`, so they ride the identical path `history.jsonl`/`.last-cleanup` ride today. **Mechanically correct.**
- **`claude_excludes` already consumes the function.** sandbox.py:820 — `claude_excludes = [claude_config_dir, *_claude_session_log_excludes()]`, applied only to `claude_root` in `_snapshot` (sandbox.py:837–838). No new call site needed, as the plan states.
- **The `~/.claude` root is content-hashed.** `_hash_critical` (sandbox.py:365–368) returns True for `~/.claude`; the tuple is `(size, mtime_ns, sha256)` (sandbox.py:414–416). So an mtime-only bump with identical bytes flips the tuple via `mtime_ns` — the spec's stated FP mechanism is accurate.
- **Option-(c) downgrade condition is `changed == [str(claude_root)]`.** sandbox.py:937. The plan does not touch it. Confirmed by reading the whole post-run block (sandbox.py:902–949).
- **Spec's provider citation is substantively correct (minor line drift).** Spec:22–24 cites `provider.py:174 / _subprocess_env respects an upstream value`. Actual: `_subprocess_env` is at provider.py:157; the respecting guard `if "CLAUDE_CONFIG_DIR" not in env:` is at **provider.py:175** (comment at :171). Off by one; substance correct. → **NITPICK-F1.**
- **The option-(c) "Canary CAN write absolute `~/.claude`" claim is TRUE (I verified it in the engine, because the crux rests on it).** provider.py `generate` (:531), `chat` (:671), `chat_stream` (:803) each append `--dangerously-skip-permissions` and **do not** call `_apply_lean_flags`. `_apply_lean_flags` (which adds `--disallowedTools`, provider.py:263–267) is called ONLY at provider.py:862/1216/1391 — the three MCP-tool paths. So on the generate/chat paths the built-in Write/Bash/Edit tools are unrestricted and can write an absolute `~/.claude/...` path. The sandbox.py:928–936 comment is faithful to the engine. This makes the crux (below) a *real* hole, not a hypothetical.

**Factual verdict: no factual error above NITPICK.** One off-by-one line cite (NITPICK-F1).

### Lens 2 — Logical (clean)

- The separate-constant decision (not appending to `_CLAUDE_SESSION_LOG_FILES`) is internally consistent: it keeps `test_af2_exclusion_set_unchanged` (test file:468–483) valid and preserves the semantic split between "session-runtime logs" (F4, justified by the confinement mechanism) and "server-pushed CLI config" (justified by no-write-motive + option-(c) envelope). Sound.
- Sequencing: constant → splat → docstring → tests. No ordering hazard; the change is a pure detection-scope narrowing with no state.
- The oracle-can-fail preface in Test 1 (no-exclude `_fingerprint` before/after an mtime bump differs) is logically valid: with `hash_content=True` the tuple carries `mtime_ns`, so an mtime-only bump changes it → the pre-fix oracle genuinely fires. **No logical flaw found.**

### Lens 3 — Missed opportunity (content-aware pruning) — **MINOR**

The plan rejects content-aware pruning (silence only when `size`+`sha256` match and *only* `mtime_ns` differs) for three reasons (spec:86–93): (i) it adds a special-cased code path; (ii) no companion write-motive for server config; (iii) residual loss is within the option-(c) envelope.

Argued from the code, reason (ii) is the weak link. The two files are not symmetric:
- `remote-settings.json` (observed `{}`, 2 bytes) is inert — negligible write-motive. Fine.
- `policy-limits.json` is described in the spec itself as "**server-pushed enforcement/policy config**." A file that governs *enforcement/policy* on the developer's real orchestrator CLI is a **more** plausible tamper target than `todos/` or `remote-settings.json`, not a less plausible one. "No companion write-motive" is asserted, not established, for exactly the one file where a motive is imaginable.

Content-aware pruning is **strictly tighter** on the owner's own "genuine escape still surfaces" axis: it silences exactly the measured churn (mtime-only) and *preserves the option-(c) warning* on a genuine content change to these two files. It also does **not** touch the raise-vs-warn semantics (it changes only what enters the diff), so it stays inside the owner's hard constraint.

I am **not** calling this a blocker, for two reasons that cut the other way: (a) the option-(c) warning that content-aware would preserve is, by the spec's own framing (spec:32–35), "loud, repeated noise that trains reviewers to ignore" — its marginal safety value is low; (b) the plan explicitly flags this choice for the red-team as a plan-level (not semantics) decision. **Recommendation:** surface content-aware to the owner as the named alternative (it is option (a) in the crux), because it is the clean way to honor "genuine escape still surfaces" for these two files at the cost of ~6 lines. Ranked **MINOR**.

### Lens 4 — Unstated assumptions & risks

- **A-1 (accepted, fail-safe):** the exclusion assumes both files sit *top-level* directly under `~/.claude`. If a future CLI version relocated them into a subdir, the name+resolve-confirm exclusion would simply stop matching → a spurious warning, never a hole. Consistent with the F4 fail-closed design. Not a finding.
- **A-2 (real, → coverage gap CH8-1):** the resolve-confirm (sandbox.py:408) means a same-*basename* file in a subtree — e.g. a hypothetical `~/.claude/plugins/policy-limits.json` — is **still** fingerprinted (its `f.resolve()` is not in `excludes`). This is the correct, desired behavior, but no proposed test pins it for the new constant. Untested narrowing risk. **MINOR** (see CH8).
- **A-3 (acknowledged):** "the churn set is exactly these two files and complete for the observed CLI version" (spec:50) is a manual/advisory completeness claim observable only against a real CLI, same class as the F4 set. Correctly framed as advisory. Not a finding.
- **Position/order sensitivity (conditional lens):** the exclusion is added to a name-keyed set/tuple consumed as an unordered membership test (`exclude_names = {ex.name ...}`, sandbox.py:395; `if name in exclude_names`). Ordering/adjacency is **not** semantic here. G5 pins the tuple as an exact literal only to catch silent narrowing/widening, not because order matters at runtime. **No order-sensitivity issue.**
- **Concurrency (conditional lens):** the change introduces no new accessor and no read-modify-write window over shared mutable state — it adds one module-level constant and one splat into a pure function. `sandbox()`'s existing not-thread-safe/not-nestable contract (sandbox.py:41–43) is unchanged. **No concurrency issue.**

### Lens 5 — Fidelity (the raise-vs-warn check is the headline; it PASSES)

Pinning each loaded term to the concrete mechanism:

- **"known-churn files"** → `_CLAUDE_CLI_HOUSEKEEPING_FILES = ("policy-limits.json", "remote-settings.json")`, the exact pair measured 2026-08-27 (spec:42–45). ✔ implemented as that constant (plan:25–28), pinned by G5/Test 4.
- **"targeted exclusion"** → only the two named files; its own constant, not folded into the session-log set; everything else under `~/.claude` stays guarded. ✔ (plan:7–28, G4 pins a third non-named json still surfaces). No blanket `~/.claude` ignore.
- **"DO NOT change raise-vs-warn semantics"** → **HONORED, and this is the most important check.** The plan touches only `_CLAUDE_CLI_HOUSEKEEPING_FILES` (new) and the `_claude_session_log_excludes()` comprehension (splat) + two docstrings. It does **not** touch the post-run block (sandbox.py:902–949), the option-(c) condition `changed == [str(claude_root)]` (:937), `_guarded_roots`, or `_hash_critical`. The change is a **detection-scope** change (what enters the fingerprint), not a **response-semantics** change (what the guard does with a diff). G7 pins this via diff review + the existing downgrade/scope tests staying green. **Confirmed honored.**
- **"a genuine companion escape still surfaces"** → **partially honored — the one soft spot.** A genuine escape to any *other* `~/.claude` path (→ option-(c) warning) or any non-`~/.claude` root (→ hard raise) still surfaces. But a genuine escape routed through *exactly* these two files no longer surfaces **at all** — not even the option-(c) warning it produces today. This is a detection-scope reduction from "warn" to "silent" for two specific files. It does **not** violate the raise-vs-warn constraint (the post-run logic is untouched), but it does stretch "genuine escape still surfaces." See the crux. Ranked **MINOR**, cross-referenced below.

**Fidelity verdict: honored on the hard constraints (targeted, no-blanket, semantics-unchanged); MINOR soft spot on "genuine escape still surfaces" for the two excluded files.**

---

## 3. The crux — does fully excluding these two files open a detection hole that MATTERS?

**Setup (verified, not assumed).** The sandboxed Canary can write an absolute `~/.claude/...` path: provider.py `generate`/`chat`/`chat_stream` (:531/:671/:803) run `--dangerously-skip-permissions` with **no** `--disallowedTools` (the lean disallow-list is applied only on the MCP paths at :862/:1216/:1391). So the built-in Write/Bash/Edit tools are unrestricted on the generative paths. The F4 "the Canary provably cannot write real `~/.claude`" argument (sandbox.py:218–226) is about the CLI's *own* CLAUDE_CONFIG_DIR-relative session bookkeeping; it does **not** cover an adversarial absolute-path Write. The option-(c) note (sandbox.py:928–936) is the accurate statement of the true capability, and the plan's A1 correctly declines to borrow the F4 argument for these two files.

**The concrete behavior change.** For a Canary write of real content to `~/.claude/policy-limits.json`:
- **Today (pre-change):** the file is fingerprinted → `~/.claude`-only diff → **option-(c) RuntimeWarning.**
- **After the change (name-based full prune):** the file is excluded → **SILENT** (no warning, no raise).

So the change moves these two exact paths from "warn" to "nothing." That is a genuine reduction, and a cold red-team must name it.

**Does it MATTER? — weighing.**
1. **Consistency with existing accepted design.** The F4 dirs (`projects/`, `todos/`, `file-history/`, …) are *already* fully pruned — a Canary absolute-path write into `~/.claude/projects/x` today produces nothing, and the owner accepted F4. So "fully-pruned `~/.claude` paths that produce nothing on a Canary write" is already an accepted category; these two files join it. This is the strongest argument *for* acceptability.
2. **But the owner-accepted option-(c) envelope is literally "warn," not "silent."** The owner accepted (sandbox.py:928–936) that a `~/.claude` escape only *warns*. Name-based pruning takes two specific `~/.claude` files below even that floor — to silent. The F4 precedent makes this *consistent*, but the owner's option-(c) risk-acceptance text is about *warning*, so name-based prune is a small step the owner did not literally ratify for these files.
3. **Practical safety delta is small.** By the spec's own framing the option-(c) warning is ignored-noise; a preserved-but-ignored warning buys little real detection. And `policy-limits.json` is server-overwritten every run, so a Canary write to it is transient.
4. **Plausibility of the escape route.** `remote-settings.json` (inert `{}`): negligible. `policy-limits.json` (enforcement/policy config): the one path with an imaginable tamper-motive — but a low-probability one, and transient.

**Conclusion.** The residual risk is **acceptable at MINOR severity** — it is consistent with the already-accepted F4 full-prune category and the option-(c) envelope, and the preserved warning it forgoes is low-value noise. **However**, because name-based prune drops these two files from "warn" to "silent" — a step the owner's option-(c) text ratified only as "warn" — the owner should be given the explicit choice rather than have it decided in the plan. If the owner wants to keep the floor at "warn" for these files, the clean fix is **(a) content-aware pruning**: silence only the mtime-only bump (`size`+`sha256` identical, only `mtime_ns` differs), preserving the option-(c) warning on a genuine content change. Option (a) does **not** touch raise-vs-warn semantics, so it stays inside the owner's hard constraint — it does not require **(b)** a semantics escalation. My recommendation: proceed with name-based as planned OR adopt content-aware; either is defensible, but **route the "warn → silent" reduction to the owner as a one-line decision** (the plan already gestures at this in Risk/rollback).

---

## 4. Coverage challenge (CH8)

- **CH8-1 (MINOR).** No criterion observes that a same-*basename* file in a `~/.claude` **subtree** (e.g. a hypothetical `~/.claude/plugins/policy-limits.json`) is **still** fingerprinted and not over-pruned by the bare name match. The resolve-confirm at sandbox.py:408 handles it, and `test_fingerprint_exclude_single_subtree_still_prunes` exercises the resolve predicate generally, but nothing pins it *for the new constant*. Scenario: a future/adversarial nested file sharing one of the two names would be wrongly pruned if a later refactor dropped the resolve-confirm. Cheap to add (one assertion). Impact: low-probability over-prune regression, currently uncaught.
- **CH8-2 (MINOR, = the crux, by design).** No criterion observes the *lost* detection on a genuine **content** change to the two files themselves (the warn→silent reduction). This is intentional under name-based pruning, so there is deliberately no test — but it is exactly the behavior the change introduces, and it is "observed" only via prose in the spec/comment. If the owner ratifies the warn→silent step, this is acceptable; if content-aware is adopted, a criterion should assert "a content rewrite of `policy-limits.json` still warns while an mtime-only bump is silent." Flagging so the owner's decision is explicit.
- Otherwise coverage is good: G1 (both mtime-only and content-rewrite FP kill), G2/G4 (non-excluded + third-json still surface), G3 (alongside-escape still raises), G5 (constant pinned), G6 (session-log set unchanged), G7 (semantics unchanged) collectively cover the intended behavior surface. **No further gap found.**

## 5. Label audit (CH9/CH10)

Per gating criterion — is it genuinely gating, and does its verification exercise the path it governs?

- **G1 (negative/core FP).** Genuinely gating (it is the bug). Governed path = `_claude_session_log_excludes` → `claude_excludes` → `_snapshot` → `_fingerprint` exclude branch → post-run diff. Test 1 mutates both files (mtime-only AND content) inside a real `sandbox()` and asserts clean exit + zero DOWNGRADED/guarded-root warning, with a no-exclude oracle-can-fail preface. **Exercises the governed path. Sound.**
- **G2 (non-excluded surfaces).** Gating. Governed path = the non-excluded fingerprint + option-(c) downgrade. Test 2 writes `settings.json` → asserts `pytest.warns(match="DOWNGRADED to a warning")`. **Exercises it. Sound.**
- **G3 (oracle can still hard-raise).** Gating (the load-bearing safety property). Governed path = a non-`claude_root` diff → `changed != [str(claude_root)]` → `raise SandboxLeak`. Test 3 bumps the two files' mtime *and* writes an `extra_guard_roots` file → `pytest.raises(SandboxLeak)`; existing `test_downgrade_is_scoped...` also stays green. **Exercises it. Sound.**
- **G4 (targeted, fail-closed).** Gating. Governed path = a third non-named top-level json is NOT in `exclude_names` → fingerprinted → warning. Test 2's second body writes `some-other.json` → still warns. **Exercises it. Sound.**
- **G5 (constant pinned).** Gating (guards silent narrowing/widening). Governed path = the returned exclude list membership. Test 4 asserts the exact tuple literal AND `all((home/.claude/n) in set(_claude_session_log_excludes()) ...)`. **Exercises it. Sound.**
- **G6 (session-log allowlist unchanged).** Gating. Verified by the *existing* `test_af2_exclusion_set_unchanged` (test file:468–483) staying green, which pins `_CLAUDE_SESSION_LOG_DIRS/FILES` as exact literals. Legitimate gating-via-existing-test; the plan correctly declines to duplicate it.
- **G7 (semantics unchanged).** Gating, and I scrutinized it hardest for being an advisory dodge. Its verification is "diff review + existing downgrade/scope tests staying green." The "diff review" half is not independently falsifiable, **but** it is backstopped by the automated half: `test_concurrent_claude_md_write_downgrades_to_warning`, `test_downgrade_is_scoped_non_claude_root_still_hard_raises`, and `test_non_excluded_claude_write_now_downgrades_to_warning` all exercise the exact `changed == [str(claude_root)]` branch and must stay green. Since the plan does not touch the post-run block, these remain valid and constitute a real gate on the semantics. **Not a dodge — genuinely gating, governed path = post-run block, evidence = the three option-(c) tests.**

**CH11/CH12 (owner ratification).** I checked the change directory: there is **no owner-ratification record** carried into this change (no ratified-decision artifact in `changes/158-sandbox-cli-churn/`; the spec/criteria reference the owner's brief but carry no ratification token). Per the charter, CH11/CH12 therefore **do not apply** — stated explicitly. Note that the crux's "warn → silent" reduction is precisely the kind of step that would benefit from an explicit owner ratification, which is why I route it to the owner in §3.

---

## 6. VERDICT

**MINOR** — the plan is mechanically correct, honors every hard owner constraint (targeted, no-blanket, raise-vs-warn semantics UNCHANGED), and is well-tested; the only substantive issue is that name-based full pruning drops `policy-limits.json`/`remote-settings.json` from the option-(c) "warn" floor to fully silent on a genuine content change — acceptable within the F4/option-(c) envelope but a "warn → silent" step the owner should ratify explicitly, with content-aware pruning available as the semantics-preserving alternative.
