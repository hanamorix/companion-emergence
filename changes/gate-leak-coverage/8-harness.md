# 8 — Harness (conformance)

**Mode:** conformance-only. No stage-0 baseline exists (this is a correctness/coverage change,
not a telemetry/cost change); the project config's regression metrics (cost/cache/num_turns) are
advisory-only and irrelevant here. So stage 8 = conformance against the frozen 1.5 criteria.

**Freeze verify (FRZ):** `1.5-criteria.md` sha256 = `46c79f2d54423b260a3523d2c081106af03dc4138dfe0e4ee9d6c58f50df85d5`
— matches the value frozen at gate-4 round-2. No post-freeze edit. PASS.
(The gate-7 round-2 minor doc fix touched only `2-plan.md`, which is not frozen.)

**Oracle command (G8):** `uv run ruff check .` (clean) + `uv run pytest -m "not live and not
requires_claude_cli and not integration"` (py3.12) → 4244 passed, 0 failed, exit 0. The migrated
+ new tests all pass; the only observed failure across runs was a pre-existing, order-dependent
flake in unrelated websocket `bridge/test_endpoints.py::*image_shas*` tests (fails a different
test each run; passes on re-run in identical deterministic order; passes in isolation; present on
clean base too) — unattributable to this diff (see decisions.md "G8 CI determination").

## Per-criterion conformance table

| Crit | Gating | Verified | How (governed path exercised) | Shown-able-to-fail |
|---|---|---|---|---|
| G1 reconcile routed | gating | YES | `test_g1_reconcile_routes_to_queue` drives real `_write_self_authored_delta`; queue has `self_model_reconcile`, db=0 | reviewer ran it vs reverted source → FAILS |
| G2 resolve routed + soul-queue intact | gating | YES | `test_g2_resolve_routes_and_still_queues_soul` drives real `_emit_resolution`; queue has `self_model_resolved`, db=0, `soul_candidates.jsonl` exists | FAILS vs reverted source |
| G3 maker routed | gating | YES | `test_g3_maker_routes_to_queue` drives real `write_making_memory`; queue has `making`, db=0 | FAILS vs reverted source |
| G4 file_write routed | gating | YES | `test_g4_file_write_routes_to_queue` drives real `decline_write`→`_wire_memory`; queue has `file_write`, db=0 | FAILS vs reverted source |
| G5 live-path gate fires end-to-end | gating | YES | `test_g5_live_path_gate_fires_end_to_end` drives real `capture_monologue`+public `apply_side_effects`; pending non-empty (monologue+monologue_trace), db=0 gated | inline ungated `store.create` control lands in db (oracle distinguishes) |
| G6 coverage guard | gating | YES | `test_g6_no_ungated_direct_create_writer` enumerates all `brain/` `.create(` ⊆ pinned union (16 unique tuples, independently re-derived exact); routed loci absent | `test_g6_guard_can_fail` drives real `_enumerate_create_sites` on a synthetic novel-receiver file → flagged un-pinned |
| G7 bypass writers stay direct | gating | YES | `test_g7_bypass_writers_remain_direct` asserts grief/breadcrumb + kindled_link still direct `store.create`/`mem_store.create`, no `route_write` | would fail if a writer were routed |
| G8 CI green | gating | YES | ruff clean + pytest selection 4244 passed/0 failed (flake unrelated, see above) | — |
| A1 GATE_BYPASS_TYPES unchanged | advisory | YES | `pending.py:48` still `{journal_entry, initiate_outbound}`; `test_c12` (existing) covers bypass | — |
| A2 file_write permanent exact-dup | advisory | YES (documented) | `test_a2_file_write_identical_content_dropped_cross_tick` proves cross-tick permanent drop (designed gate behavior) | — |
| A3 reconcile visibility latency | advisory | YES (verified in review) | `reconcile.py:177-186` acknowledge/cooldown keyed on `gap is not None`, NOT the write → only the supplementary memory's visibility is delayed; surfaced to owner | — |

**Verdict: PASS (conformance).** All 8 gating criteria empirically verified by exercising the
governed path; every gating oracle shown able to fail. Advisory A1–A3 satisfied/documented and
surfaced to the owner. No baseline → no regression check (conformance-only, as configured).
