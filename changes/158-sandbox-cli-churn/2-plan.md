# 2 — Plan (#158)

## Change, precisely (content-aware / mtime-insensitive)

### `tests/harness/sandbox.py`

1. **New constant + helper**, placed right after `_CLAUDE_SESSION_LOG_FILES`/
   `_claude_session_log_excludes` (a distinct concept, so a distinct name + its own justification):

   ```python
   # --- claude-code ORCHESTRATOR server-pushed housekeeping files (#158) --------------------------
   # These are NOT session-runtime logs; they are Anthropic-server-pushed CLI CONFIG that the
   # `claude` CLI rewrites under the real ~/.claude on essentially every invocation — including the
   # orchestrator's own and its subprocess calls — bumping mtime even when the bytes are UNCHANGED.
   # (Measured 2026-08-27: one `claude -p` bumped both files' mtime with identical size/content;
   # ~/.claude is content-hashed, but mtime_ns is part of the fingerprint tuple, so the bump alone
   # trips the guard on every run.) They are the ORCHESTRATOR's real CLI config, never companion
   # state: the sandboxed subject's own policy-limits/remote-settings live under the tempdir
   # CLAUDE_CONFIG_DIR.
   #
   # We make ONLY these two top-level files MTIME-INSENSITIVE (drop mtime_ns from the fingerprint
   # entry) while KEEPING their content sha256 — so a benign mtime-only bump is silent, but a
   # genuine CONTENT change still flips the fingerprint and surfaces at the same option-(c) "warn"
   # floor every other ~/.claude path gets. We deliberately do NOT name-EXCLUDE them (unlike the F4
   # session logs): the stage-3 red-team verified the Canary can write an absolute ~/.claude path
   # (generate/chat run --dangerously-skip-permissions with no tool disallow-list), so a full prune
   # would drop these two files below the owner-ratified "warn" floor to silent. Content-aware
   # narrowing fixes the false positive (purely the mtime bump) without that detection loss, and
   # does not touch the raise-vs-warn logic. TARGETED to two exact top-level paths.
   _CLAUDE_CLI_HOUSEKEEPING_FILES = (
       "policy-limits.json",
       "remote-settings.json",
   )


   def _claude_cli_housekeeping_files() -> list[Path]:
       """Absolute ~/.claude paths whose fingerprint entry is mtime-insensitive (content-sensitive).

       Server-pushed CLI config the `claude` CLI rewrites (mtime-bump) every run; we ignore the
       mtime axis for exactly these top-level paths while keeping the content hash, so the benign
       churn is silent but a genuine content change still surfaces (#158). Derived by NAME under the
       REAL Path.home()/".claude" — an absent file is a harmless no-op.
       """
       base = Path.home() / ".claude"
       return [base / name for name in _CLAUDE_CLI_HOUSEKEEPING_FILES]
   ```

2. **Add an `mtime_insensitive=` parameter to `_fingerprint`** (honored only with
   `hash_content=True`, which is where the content hash keeps the entry meaningful). Mirror the
   existing `exclude` name-prefilter (name-match, then `resolve()`-confirm — no per-file syscall on
   the majority):

   ```python
   def _fingerprint(
       root, exclude=None, *, hash_content=False, mtime_insensitive=None
   ) -> dict:
       ...
       mtime_ins = {m.resolve() for m in mtime_insensitive} if mtime_insensitive else set()
       mtime_ins_names = {m.name for m in mtime_ins}
       ...
       for name in filenames:
           f = dp / name
           ...  # existing exclude handling unchanged
           st = f.stat()  # (existing try/except)
           if hash_content:
               if name in mtime_ins_names and f.resolve() in mtime_ins:
                   entry = (st.st_size, None, _content_hash(f))   # mtime slot nulled → mtime-insensitive
               else:
                   entry = (st.st_size, st.st_mtime_ns, _content_hash(f))
           else:
               entry = (st.st_size, st.st_mtime_ns)
   ```

   The nulled mtime slot keeps the 3-tuple arity (unambiguous vs the 2-tuple no-hash entry) and
   makes only `(size, sha256)` load-bearing. `None == None` always, so mtime never enters the
   compare for these two files; size/content still do (a content rewrite → different sha256 →
   diff; a creation absent→present → new key → diff).

3. **Wire it in `_snapshot`** — for the `~/.claude` root only, pass the housekeeping paths:

   ```python
   if gr == claude_root:
       ex.extend(claude_excludes)
   snap[str(gr)] = _fingerprint(
       gr,
       exclude=ex or None,
       hash_content=_hash_critical(gr),
       mtime_insensitive=_claude_cli_housekeeping_files() if gr == claude_root else None,
   )
   ```

That is the whole production-side (harness) change: one constant + one helper + one new
`_fingerprint` param (name-prefiltered) + one `_snapshot` wire-in. No change to the post-run
raise/warn block, `_guarded_roots`, `_hash_critical`, the option-(c) condition, or
`_claude_session_log_excludes()` / the F4 sets.

## Why this mechanism (not the alternative)

- **Keeps the "warn" floor** the owner ratified: a genuine content write to either file still
  produces the option-(c) warning; only the measured mtime-only churn is silenced. This is the
  stage-3 reviewer's own recommended fix (its §3 option (a)).
- **Reuses the fingerprint's existing structure** (per-file name-prefilter, the content hash
  already computed for `~/.claude`) — the only new behavior is nulling one tuple slot for two
  named paths. No new hot-loop pass.
- A **separate constant + helper** (not folded into `_CLAUDE_SESSION_LOG_FILES`/its exclude list)
  keeps `test_af2_exclusion_set_unchanged` valid and the F4 "session-log / Canary-can't-write"
  semantics accurate for their own set.
- Full name-pruning was the original draft and is rejected (spec "Rejected alternative"): it drops
  these paths from "warn" to silent, a loss the reviewer flagged.

## Tests (`tests/unit/harness/test_sandbox_isolation.py`)

Add a focused block. All use `_seed_fake_cred(monkeypatch, tmp_path)` (fake `~/.claude` under
`tmp_path`; real home never touched). Import `_CLAUDE_CLI_HOUSEKEEPING_FILES` and
`_claude_cli_housekeeping_files`.

1. **`test_cli_housekeeping_mtime_bump_is_silent` (G1, negative/core).**
   - Pre-create fake `~/.claude/{policy-limits.json,remote-settings.json}` with initial bytes.
   - Oracle-can-fail preface: an **mtime-sensitive** `_fingerprint(claude, hash_content=True)`
     (no `mtime_insensitive=`) before/after an `os.utime` mtime bump (same bytes) DIFFERS — proves
     the churn would trip pre-fix.
   - Then inside `with sandbox() as sb:` bump BOTH files' mtime via `os.utime(..., ns=...)` with
     identical content. Wrap in `warnings.catch_warnings(record=True)`; assert **zero**
     `RuntimeWarning` mentioning a guarded root / "DOWNGRADED", and clean exit (no `SandboxLeak`).

2. **`test_cli_housekeeping_content_change_still_warns` (G1b, content-aware floor preserved).**
   - Inside `sandbox()`, **rewrite** `policy-limits.json` with different bytes → assert
     `pytest.warns(RuntimeWarning, match="DOWNGRADED to a warning")`. The content axis still
     surfaces; only mtime is silenced. (Also assert, via a direct `_fingerprint(...,
     mtime_insensitive=_claude_cli_housekeeping_files())` before/after a *content* change, that the
     entry DID change — the oracle-can-fail for the content axis.)

3. **`test_non_housekeeping_claude_file_still_surfaces` (G2 + G4, targeted).**
   - Inside `sandbox()`, write a non-housekeeping `~/.claude/settings.json` → warns.
   - Second body: write a **third, non-named** top-level json (`some-other.json`) → still warns
     (fail-closed: only the two named files are mtime-insensitive, not all `*.json`).

4. **`test_same_basename_in_subdir_stays_mtime_guarded` (G4b, top-level-only).**
   - Pre-create `~/.claude/plugins/policy-limits.json`. Inside `sandbox()`, mtime-bump *that*
     nested file (same content) → assert `pytest.warns(match="DOWNGRADED to a warning")` (the
     nested same-basename file is NOT mtime-insensitive; only the exact top-level path is).
   - Direct-`_fingerprint` half: with `mtime_insensitive=_claude_cli_housekeeping_files()`, an
     mtime-only bump of the *top-level* `policy-limits.json` leaves its entry unchanged, while the
     same bump of `plugins/policy-limits.json` changes its entry — pins the resolved-path scope.

5. **`test_housekeeping_churn_alongside_real_escape_still_hard_raises` (G3, safety).**
   - Inside `sandbox(extra_guard_roots=[guarded])`, mtime-bump the two housekeeping files AND
     write `guarded/leaked.txt` → `pytest.raises(SandboxLeak)`. The mtime-insensitivity never
     masks a real non-`~/.claude` escape in the same run.

6. **`test_cli_housekeeping_constant_and_helper_pinned` (G5 + G6).**
   - `assert _CLAUDE_CLI_HOUSEKEEPING_FILES == ("policy-limits.json", "remote-settings.json")`.
   - `assert set(_claude_cli_housekeeping_files()) == {Path.home()/".claude"/n for n in
     _CLAUDE_CLI_HOUSEKEEPING_FILES}` (helper derives exactly those two under `~/.claude`).
   - Assert none of the two housekeeping files are in `_CLAUDE_SESSION_LOG_FILES` nor in
     `_claude_session_log_excludes()` (they ride the mtime-insensitive path, NOT the exclude list —
     guards G6 and the semantic split). `test_af2_exclusion_set_unchanged` already pins the F4
     literals; rely on it staying green rather than duplicating.

## Instrumentation / measurement

No new logging needed — this is a pure detection-scope change, proven by unit tests with an
oracle-can-fail preface (the no-exclude fingerprint diff), matching the existing suite's style.
No regression workload exists for this project (Layer-2 config), so stage 8 is conformance-only:
the criteria above + CI.

## Risk / rollback

- Single-file targeted exclusion; rollback = drop the constant + the splat. No data touched.
- The one genuine risk (detection blind spot for these two exact real paths) is stated in the
  code comment and the spec's rejected-alternative; it is within the owner-accepted option-(c)
  envelope. If the red-team disputes the *severity* of that blind spot enough to want the
  content-aware variant or a semantics change, that routes to the owner.

## Paths validated

- `tests/harness/sandbox.py` — exists (read in full).
- `tests/unit/harness/test_sandbox_isolation.py` — exists (read in full).
- Layer-2 config `~/Desktop/claude-code-skills/Guarded_change/guarded-change.companion.md` — read.
- `redteam_context` paths for the cold reviewer: `~/Desktop/companion-emergence/brain` (the
  clone), and the sandbox source itself. Validate at reviewer spawn.
