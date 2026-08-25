# 3 — Stage-3 red-team (verbatim record)

## Reviewer
- Agent type: `general-purpose`; model: Opus 4.8 (`claude-opus-4-8`); spawned `run_in_background: false` (synchronous — returned in-turn to the loop runner, no main relay needed; this dogfoods item B's own foreground pattern).
- Cold: no shared context with the author.

## Context given (closed set)
Artifacts under review: `changes/run-orchestration-hardening/{1-spec,1.5-criteria,2-plan}.md`.
Source (priority order): `tests/harness/roster_preflight.py`, `tests/harness/dropin.py`,
`tests/harness/__init__.py`, `hunts/symptom-cluster-rootcause/live-run/RUN-INSTRUCTIONS.md`
(read-only input), `changes/harness-drop-in-code-ingest/9-handoff-live-run-callsite-edits.md`.

## Charter given
The red-team charter core (five lenses: factual/logical/missed-opportunity/assumptions-risks/fidelity;
evidence discipline; cite-or-it-doesn't-count; clean factual + clean fidelity must be earned; rank every
finding; flag unverifiable) + stage-3 additions CH8 (coverage challenge, required section), CH9/CH10 (label
audit, required section), CH11/CH12 noted N/A (no carried owner-ruling). Plus task-specific scrutiny:
the no-behavioral-test cap, the em-dash-in-banner argument, whether `boot_and_verify` makes
background-and-await impossible vs inconvenient, and accuracy of reused signatures.

## Reviewer context-file sha256 (reviewer-reported; cross-checked by runner — match)
```
1e1cc956e53ae4b7477e3d038fa750270671ead6af881ab10585411e5431c3f7  1-spec.md
a22be1baa1866c3a99a9084cb78f6b4430ae9572c1e0ed769d2a26a2b54a0469  1.5-criteria.md
671333e9ee46c2e0c1f0801e6bcbed8ff106cfc7723ddd4d230066e46d4322cf  2-plan.md
9a66470cf937599aa5dca15568df6b808296fe6929b71e09cf03f9ce1fdb0aa7  roster_preflight.py
52af3e1c4c0d86306e822e444dbeb6b2873e216e373f7b3dd56a51e107c31c2e  dropin.py
13f32f831d4c50f8d5b514c06b626ff56e0eb00dd64b881fd163474456de9826  __init__.py
d62f05134394a72bdf82e19e68304103f484320632f4c77c13c8cd145f153df5  RUN-INSTRUCTIONS.md
cbf624c4b5dc099102f5677478b4dac2c58a491e5885611aca0548f8e846539d  9-handoff-live-run-callsite-edits.md
```

## Reviewer verbatim output

### HEADLINE — F1 (BLOCKER, Fidelity+Factual+Logical)
`boot_and_verify`'s real default verify seams (`assert_brain_under_dropin`,
`assert_brain_tools_roster`) are process-SELF-checks that validate the *calling* process, but
`boot_and_verify` runs in the ORCHESTRATOR process, not the booted bridge. With the real
defaults, a *healthy* live run raises `DropinMismatch`. No gating criterion catches this because
G1/G2/G3 inject fakes for exactly these seams.
- `boot_and_verify` is orchestrator-side: launches the arm via `boot_cmd` (`2-plan.md:73`), and
  Testing passes `run_arm.sh` as `boot_cmd` (`1-spec.md:78-79`). `run_arm.sh` boots
  `live_server.py` (the bridge). So `boot_and_verify` runs above the bridge, in the session/pytest process.
- `assert_brain_under_dropin` checks `Path(sys.prefix).resolve() == build.venv_root.resolve()`
  (`dropin.py:217-219`), `venv_root == <dropin_copy>/.dropin-venv` (`dropin.py:102`). Orchestrator
  runs under the ce-harness-ingest repo venv, not the drop-in venv → raises `DropinMismatch` on a
  healthy run (`dropin.py:219-225`). Item A wires this into `live_server.py` itself
  (`9-handoff…:121-131`) — the only correct home.
- `assert_brain_tools_roster` first does `_assert_declared_authoritative(build)`
  (`roster_preflight.py:152`) resolving in-process `brain` under `build.repo`
  (`roster_preflight.py:67-68`). In the orchestrator `brain` resolves to
  `/home/zero/Desktop/ce-harness-ingest/brain/`, not the copy → raises `DropinMismatch`
  (`roster_preflight.py:69-73`).
- Ships "green" because `test_foreground.py` injects fakes and never invokes the real defaults.
- Route to spec/plan: either (a) `boot_and_verify` must NOT re-run the process-self asserts (the
  bridge validates itself; the orchestrator confirms via READY-without-hard-fail + transcript
  inspection, per `RUN-INSTRUCTIONS.md:19-21` steps 3-4), or (b) redefine "verify" to something an
  orchestrator-side process can legitimately check, with a criterion exercising that real path.

### Lens 1 Factual
- F1 (above).
- **F2 (MINOR):** `1-spec.md:94` claims the banner text is "already used in Testing's draft" — FALSE; `grep "DRIVE NOW"`/`grep "▶"` over `RUN-INSTRUCTIONS.md` = 0 hits. The charter source is outside the closed set (unverifiable), but the "already used in Testing's draft" clause is falsified.
- Reused signatures CONFIRMED accurate: `assert_brain_tools_roster(build, persona_dir, *, env=None, cwd=None, timeout=15.0)` raising `DropinMismatch` (`roster_preflight.py:139-146,155-161`); `assert_brain_under_dropin(build, *, module="brain")` raising `DropinMismatch` (`dropin.py:199,220,228`); `DropinBuild(repo,python,source)`+`venv_root` (`dropin.py:80-102`); `DropinMismatch(RuntimeError)` (`dropin.py:69`); `__all__` already exports the item-A symbols (`__init__.py:60,82`); item-A tests `test_roster_preflight.py`/`test_dropin.py` exist.

### Lens 2 Logical
- **F3 (MINOR):** boot timeout leaves an orphan server. `run_arm.sh` `nohup … &`s `live_server.py` and blocks until READY; if READY exceeds `ready_timeout`, `subprocess.run` raises `TimeoutExpired` in-turn (no hang) but the nohup'd server keeps running, orphaned — violates the plan's own "one harness server at a time" premise (`RUN-INSTRUCTIONS.md:15`); no cleanup specified.
- **F4 (MINOR, drift):** `tests/harness/README.md:194-233` already has an "### The orchestration protocol" section describing the same orchestrator-driven run + AgentBob pausing. Adding a parallel `RUN-ORCHESTRATION.md` with only a one-line pointer risks the documented two-copies drift. Reconcile / cross-link rather than parallel.

### Lens 3 Missed opportunity
- **F5 (MINOR):** narrow the residual drive seam structurally via `run_arm(spec, drive=callable)` sequencing boot→verify→drive→preserve in one call; the *synchronous AgentBob* drive IS a single callable and could be wrapped (inline Bob cannot). Converts part of the contract-only guarantee into a structural one.
- **F6 (MINOR):** reuse the README section (lower drift) — see F4.

### Lens 4 Assumptions & risks + CH8
- A(i)→F1. A(ii): assumes subprocess-transcript dir is named `"subprocess-transcripts"` (`2-plan.md:96`) — producer is gitignored `live_server.py`, UNVERIFIABLE; a name mismatch → `preserve_artifacts` lists it in `missing` (non-fatal) and returns a green `PreserveResult`, so the most important raw artifact is silently not preserved while the run reports success (`RUN-INSTRUCTIONS.md:27` stresses preserve-raw). A(iii): `ready_timeout=60` + roster 15s bounded, acceptable.
- **CH8 coverage gaps:** (1) [highest] the real default `boot_and_verify` path in its actual process context — no criterion runs the real defaults (F1). (2) silent non-preservation on subprocess-dir name mismatch (A(ii)); G4 only tests a fabricated dir. (3) orphan-server leak on boot timeout (F3). (4) the drive-step park is *deliberately* uncovered per the measurement-apparatus decision — reviewer concurs it should NOT be a behavioral test; noted for completeness only.

### Lens 5 Fidelity (terms pinned)
foreground ✔, blocking ✔, one-turn ◑ (drive is a doc/banner gap between two calls, disclosed), boot ✔ (◑ orphan on timeout F3), **verify ✗ PROXY/BROKEN (F1 — process-self-checks of the caller, not the arm)**, preserve ✔ (◑ dir name unverified), does-not-hang ✔, banner ✔. Verdict: verify fails fidelity (F1); rest clean/honestly-partial.

### CH9/CH10 Label audit
G2, G4–G8 exercise real governed paths (legitimate). G1/G3 use injected fake verify seams — legitimate for *sequence/return/propagation* wiring but a **proxy dodge for the "verify" claim** because the shipped defaults are broken-in-context and no criterion runs them (ties to F1). Advisory A1–A3 are genuine inspection-grade, NOT gates-in-disguise. The no-behavioral-test cap (`1.5-criteria.md:79-95`) is a correct *scope* decision (measurement-apparatus reasoning sound), NOT a mislabel — but the cap must not be used to also skip the *deterministic, mechanical* real verify path (F1), which IS unit-testable.

### CH11/CH12
Not applicable — no carried-forward escalated fidelity finding / owner-ruling.

### Scrutiny of flagged claims
- No-behavioral-test cap: correct for the *agent-behavior* drive guarantee; the residual drive seam is real, inherent, and disclosed — acceptable, NOT a blocker. Do not use the cap to skip F1.
- Em-dash-in-banner: SOUND — the banner is operator-facing (never injected into companion context); README already uses em-dashes/→ freely. (Caveat F2 on the false "already used" justification.)
- Impossible vs inconvenient: merely inconvenient/discouraged; drive sits between two calls as a residual park point, closed by contract (doc+banner), not structure. Adequately disclosed.

### Ranked summary
BLOCKER F1 · MINOR F3 (orphan server) · MINOR F4 (README drift) · MINOR F2 (false "already used") · MINOR F5/F6 (structural narrowing / reuse README) · UNVERIFIABLE A(ii) (subprocess dir name).
