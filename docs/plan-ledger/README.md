# docs/plan-ledger/

Working directory for the `plan-ledger` skill (config: `plan-ledger.companion.md` at the repo root;
install: `scripts/install_plan_ledger.sh`).

- `DESIGN-INVARIANTS.md` — tracked. The owners' standing architecture rules; every sheet, spec and
  cold pass reports against it. Edit only with an owner's words and a date.
- `ledgers/` — gitignored. One `<slug>-LEDGER.md` per planning conversation (the state that survives a
  session restart), plus `ledgers/reports/` for cold-pass reports. Ledgers quote the owner verbatim, so
  they stay local like the guarded-change `changes/` scratch.
