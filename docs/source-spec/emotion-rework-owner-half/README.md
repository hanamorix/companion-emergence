# Emotion rework, the owner's half: design appendices

These files are the design appendices of `../2026-10-28-emotion-rework-owner-half-design.md`. The main document says how to read them and which rows of them it changes (its Appendix B).

| File | Level | What it holds |
|---|---|---|
| `A3-engine-draft.md` | 2 | The emotion engine: lines E-1 to E-10, the rulings recorded as rules (R4 to R21), the numbered checks, records of possible later additions |
| `A3-L3-engine-build.md` | 3 (normative) | The engine build detail: module layout, public calls, data shapes, slices S0 to S7, tests |
| `B2-readers-draft.md` | 2 | The readers: what the engine gives them, site rules, checks |
| `B2-L3-readers-build.md` | 3 (normative) | The readers build detail, rows X1 to X39 |
| `A1R-readers-safetynet-draft.md` | 2 | The reader-side safety net: same-answer test, bypass pin, old-branch pins, timings |
| `A1R-L3-build.md` | 3 (normative) | The safety-net build detail, rows Q1 to Q42 |
| `AF-access-fixes-draft.md` | 2 | The access fixes |
| `AF-L3-build.md` | 3 (normative) | The access-fix build detail, rows Q1 to Q38, and the GitHub texts for that part |
| `boundaries-draft.md` | 2 | Boundaries of the owner's parts, and the check against the co-maintainer's plan (the H-numbered interface questions) |

What was done to these copies:

- They are the design sections of the working drafts. The history, fold records, review lists, vote records and citation keys that surrounded them were left out.
- Citations of private working documents (planning record lines, review and vote reports, fact sheets, transcripts) were removed. A provenance bracket now keeps the ids that are defined in these files or in the main document, the confidence tag (H = settled by the cited decision, M = a reading or derived line), and `maintainer ruling <date>` where a maintainer decision stood behind the line. Where removal left a bracket thin or a sentence slightly clipped, the design line itself is unchanged.
- Names were replaced by roles ("the owner", "the co-maintainer", "the planner", "the bug-fix track"). GitHub handles are kept only where a command or an assignee field needs one. No local paths remain, and no time of day: a dated ruling carries its date only.
- Line-number citations in the main document that point into the working drafts (for example `A3-L3:54`) refer to the drafts' own line numbers, which these copies do not keep. Section numbers and row ids (E6.1, X15, Q22, AF-L3 Q5) are authoritative.
- Ids such as TRIO-n, MJn, OBn and BCn name internal review votes and checks. They are kept as labels; the text of the line they annotate states what it needs.
- The sizing runs that fixed the shift rule's numbers (section 9.4, Check 21 of the main document) are not included. The rule itself is specified in A3 E5.20 to E5.26, and the numbers to reproduce are in section 9.4.

## Labels used in these files

- **H / M.** The confidence tag at the end of a provenance bracket. H: the cited decision, text or code settles the line with no reading. M: a reading, a review vote, a judgment call, or a derived line.
- **DERIVED.** The line follows from the cited lines by reasoning and is not stated in any one of them. Always M.
- **INTERP.** The drafter's own reading of a source; no source states it. Always M, and open to the owner's correction.
- **BUILD-FORCED.** The build needs the line and the level-2 design is silent; the line names the level-2 lines it stays inside. M.
- **FACT NEEDED.** A fact the drafter had not yet checked in code when the line was written. Where a later section or the main document states the answer, that answer governs.
- **RULE.** `[RULE:a,b | M]` is a line that two recorded rules or principles agree on, for one reading of the case.
- **Row classes `(D)`, `(S)`, `(N)`.** On the Q rows of `A1R-L3-build.md` and `AF-L3-build.md`: D = defining (delivers a test, timing or edit the owner asked for), S = support (tests or machinery around those), N = scope, order, pointer, fact or wording (not counted).
- **OWNER / "the owner".** A statement the owner made, quoted or paraphrased; in these copies the time of day is dropped and the date kept. `maintainer ruling <date>` is a decision both maintainers made on that date.
- **QD.** The quiet-day ruling (2026-10-14): on an active day on which an emotion that already has an average got no entry, the emotion gets one entry of 0 at weight 1 in both averages, which raises its entry count and lowers its average; an emotion with no average gets nothing. An active day is a calendar day on which the app ran at all. Stated as rules in A3 E5.
- **PU-1 to PU-4.** The four clauses of the simplification proposal the owner accepted as R10: PU-1 the rebuild holds the lock for its whole run, so anything arriving waits; PU-2 the change log is only a backup record, changes go through the normal queue, and a rebuild replays the last 30 days of the log; PU-3 the queue is a plain list drained after the reply; PU-4 active days keep only the essentials (the open day, each feeling's last-entry day, the off-day skip).
- **Chat clause n.** The n-th of the nine clauses of the shift rule (R14) as put to the owner and confirmed on 2026-10-22. The nine clauses are listed under R14 in A3 "Rulings recorded as rules"; the rule itself is specified in E5.20 to E5.26.
- **TRIO-n, MJn, BCn, RLn, Fn.** Ids of internal review votes and checks; labels only. A tally such as "3/3, M" is the vote and the confidence it gave.
- **P / T / D / R followed by a number, `plan:n`, `step0`.** The maintainers' working principles (P), tie-breaks (T), defaults (D) and case rulings (R), and the references to the shared emotion rework plan and the step 0 interface document; see the main document's reading rules.
