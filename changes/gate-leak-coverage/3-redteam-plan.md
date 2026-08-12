# 3 — Red-team of the plan (verbatim record)

## Reviewer
- Agent type: `general-purpose`; model: `sonnet` (reason: cold adversarial semantic review —
  needs judgment, not a mechanical read).
- Cold: no shared context with the author. Spawned with read access to the change artifacts +
  the worktree source (NOT the ungated main checkout).

## Context given (closed set)
- Change artifacts: `changes/gate-leak-coverage/{1-spec.md, 1.5-criteria.md, 2-plan.md}`
  (as of round 1; revised after this review — see decisions.md gate-4).
- Source (worktree, overriding the config's ungated-main brain path per decisions.md):
  `brain/memory/pending.py`, `brain/engines/consolidation.py`, the 4 GATE loci
  (`self_model/reconcile.py`, `self_model/resolve.py`, `maker/wiring.py`, `files/commit.py`),
  the 4 BYPASS loci (`recovery/engine.py`, `body/events.py`, `grief/breadcrumb.py`,
  `kindled_link/relationship.py`), `chat/extractor.py`, `chat/monologue_capture.py`,
  `monologue/trace.py`, `ingest/commit.py`, `tests/test_consolidation_gate.py`.

## Charter given
The full red-team charter core (five lenses: factual/logical/missed-opportunity/
assumptions-risks/fidelity + evidence discipline + provenance + sha256 reporting), plus the
stage-3 additions (CH8 coverage challenge, CH9/CH10 label audit), plus two task-specific
challenges: (a) coverage completeness — independently enumerate all `.create(` in brain/ and
check the plan's allowlist; (b) does G5 genuinely prove the gate fires end-to-end. Plus explicit
position-lens and concurrency-lens confirm/refute requests.

## Reviewer-reported sha256 of context files read
```
8cb251e29c104547695d48b340a71ddf62f1a6b24853f69d4dab7c3a74870a0b  1-spec.md
51aec884175674ee6a75d8a8e92c54d204d082701c208d7fb030f1ae0f3d2fa9  1.5-criteria.md
47701b5e65dc4d328745afcc1d0c981dc6eb4b0e7dcde352bc70146f0eb85811  2-plan.md
26b19610b84d6a8ee896735e303c2443872f8e0dad76387e3ade2255390c1d1c  brain/memory/pending.py
45f116e5604da17f1ce0cba480513dbb4ebc11c19390e34f9bcad4f2806171cc  brain/engines/consolidation.py
1318ce61209ad7890d726d7fed1a4b999c4abd5db41f60fbd3e782e2160738e7  brain/self_model/reconcile.py
d59c39f5fd3b9e4451cadf3a57d7ef1fe24a5187cc4d4523ea021acc271b08af  brain/self_model/resolve.py
125106501c1f7dc92f0fc89a994a3717f45e9cf4c9db99a3328ba6245ec01683  brain/maker/wiring.py
1803c58c6ca6f34e60b6bc8d764ce56e884934f75741adf0c52ebf8e479c6195  brain/files/commit.py
c67aa102c432e1da537edbd500b3b4aa0b4f28246dbb1117e5bf68a44c75d9c8  brain/recovery/engine.py
98e92fe66a26d3b688e7e96e45f3be0cb87eced144a5687fdfacaa10799dc192  brain/body/events.py
bc152785b9477ca919e402afe35c5de40d5a5f519048cb103bb22b1be54fcdf9  brain/grief/breadcrumb.py
be6abfbcc46aa4f63b2c5f0903afba9053ff2bacd9fd8598dd10d989bbfcbea2  brain/kindled_link/relationship.py
1681b87d3b33b587cdfa560a202e0d60585ee40a5d97003adbdb2520ca88577b  brain/chat/extractor.py
3d86d2c87079f2bb3170dc610d9a0264ff93134ece185e449e74687b92b36d32  brain/chat/monologue_capture.py
2872329f55fc54deea03caa7c3e2561224791c5781abb672ba21b62ec509c194  brain/monologue/trace.py
e83990d27239f42115f1b68a3732c8c14266fc1d49386a53ff9c4b8aecdbc3ec  brain/ingest/commit.py
075976dea151a323f2542f20ad90cedf3e53ea1c8ae03a8a52402911c1ab4b91  tests/test_consolidation_gate.py
(+ propose_write.py, files/pending.py, store.py, making_runner.py, tool_loop.py, dispatch.py,
 file_lock.py, soul_queue.py, soul/review.py — read for the coverage-completeness challenge)
```

## Reviewer verbatim output
Worst severity: **MAJOR**. Two findings:
- **(a) MAJOR** — G6's guard has no specified mechanism to avoid false-positiving on
  `brain/tools/impls/propose_write.py:74,91`, which call `brain.files.pending.create(...)` — a
  DIFFERENT subsystem (file-write-approval queue), not a MemoryStore write. A naive `\.create\(`
  regex catches them. Unaddressed, G6 either (i) fails immediately on those two lines (a
  real-work-blocking false failure), or (ii) the implementer adds an unreviewed ad-hoc scope.
  Recommendation: scope the guard to memory-store receivers, or trace receivers to
  `MemoryStore(`/`SoulStore(` construction. The reviewer independently ran
  `grep -rn '\.create(' brain/` (21 hits) and confirmed **coverage of real memory-store writers
  is complete** — every hit resolves to a routed locus, an owner-BYPASS locus, gate internals,
  explicit-user tools, soul promotion, or migrator/*. No un-allowlisted automatic writer found.
- **(b) MAJOR (CH8 unmeasured blast radius)** — two real, owner-relevant behavior changes with
  no observing criterion: (1) `file_write` audit dedup — two byte-identical `file_write`
  memories in one idle-tick window get one dropped by Pass-1 exact-dup (was impossible before);
  (2) `reconcile.py` marks the gap `acknowledged` + starts cooldown synchronously
  (`reconcile.py:178-182`, keyed on `gap is not None`, not on the write), while the emotion
  memory is now only a queued candidate that may take ticks to land or be deduped/merged.

Minors: `resolve.py` planned `source="self_model_resolve"` vs type `self_model_resolved`
(cosmetic — `_source` is provenance-only, gating keys on `memory_type`); G5 drives private
`_apply_memory_writes` rather than the public `apply_side_effects` wrapper the pass-2 worker
actually calls (minor representativeness gap); G1-G4 could be parametrized (optional).

Lens verdicts: **Factual** — clean, all 8 loci verified line-for-line with citations; the
`route_write` return-value and soul-candidate precedent (`extractor.py:550-561`) confirmed.
**Fidelity** — clean; pinned "automatic writers GATE" / "BYPASS owner rulings" /
"route_write keys on memory_type" / "resolve precedent" and confirmed the plan implements the
literal mechanism, no proxy. **Position lens** — confirmed N/A (same-line, same-position
substitution). **Concurrency lens** — confirmed N/A (enqueue uses the existing `file_lock` on
the same queue file used by extractor/trace/ingest; adds callers of an already-guarded path,
no new lock/file/RMW window; no new `memories.db` write path). Coverage of real memory writers
complete; the gap is G6's mechanism precision, not the allowlist content.

(Full reviewer transcript retained in the run log; agentId af2836e442de2a674.)

---

# 3 — Red-team round 2 (re-review of the revised plan; verbatim record)

## Reviewer
- Agent type: `general-purpose`; model: `sonnet`; cold (agentId a72b0cac09bf27799).
- Context: revised `{1-spec, 1.5-criteria, 2-plan}` + worktree source + the carried-forward
  round-1 findings (a)/(b) to confirm addressed.

## Reviewer-reported sha256 (key files)
1.5-criteria.md 07165c61… ; 2-plan.md 1c4ce034… ; pending.py 26b19610… ;
reconcile.py 1318ce61… ; resolve.py d59c39f5… ; wiring.py 12510650… ; commit.py 1803c58c… ;
propose_write.py 3c5f8c41… ; consolidation.py 45f116e5… ; extractor.py 1681b87d… ;
tool_loop.py ae2408b2… . Independent grep = exactly 21 `.create(` lines = List1(2) ∪ List2(15)
∪ 4 routed loci (matches plan exactly). HEAD confirmed 3e98ee57 on the fix branch.

## Verbatim verdicts
- **(a) coverage-guard false-positive — RESOLVED.** propose_write.py:74/91 confirmed
  `brain.files.pending.create` (non-MemoryStore). G6's subset-of-pinned-lines design fails safe
  on any novel/aliased receiver (`s.create(mem)` can't coincide with a pinned line). List2
  checked entry-by-entry against source — complete and correct.
- **G5 wrapper — constructible, confirmed.** `ExtractorOutput` (extractor.py:139-150) has
  `memory_writes` + `emotion_delta`; `apply_side_effects(out, *, persona_dir)` (extractor.py:312)
  matches; tool_loop.py:78 is the real call site; dispatch.py:186 is the real record_monologue
  handler.
- **(b) A3 — legitimately advisory, VERIFIED.** reconcile.py:174-186 sets `acknowledged`/cooldown
  on `gap is not None` (line 178), NOT on `wrote`; decoupling pre-exists the change. Routing only
  changes DB-visibility latency.
- **NEW MAJOR (round 2): A2's blast-radius description was understated.** `_has_exact_existing`
  (consolidation.py:176-179) checks ALL active `memories.db` rows via `store.search_text` with NO
  time bound; `_wire_memory` content has no differentiator beyond path+outcome. So identical
  file_write memories collide PERMANENTLY across ticks/sessions, not just within one idle window —
  a materially larger, permanent recall-channel drop (full history only in audit.jsonl). Not a
  defect in the routing (G1-G4 unaffected); the risk-acceptance DESCRIPTION misrepresented scope.
  Fix = rewrite A2 to state the true mechanism+severity before owner acceptance.
- **NEW minor: G6 pin by (file, line-text) not line-number** (else brittle to unrelated line
  shifts).
- Position lens N/A; concurrency lens N/A; label audit — A2/A3 legitimately advisory (not a
  gating-criterion dodge), but A2's text needed correction.
- Worst severity: **major** (the A2 description-accuracy finding).

(Full round-2 transcript retained; agentId a72b0cac09bf27799.)
