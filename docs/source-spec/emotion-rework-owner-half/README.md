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
- Names were replaced by roles ("the owner", "the co-maintainer", "the planner", "the bug-fix track"). No local paths remain.
- Line-number citations in the main document that point into the working drafts (for example `A3-L3:54`) refer to the drafts' own line numbers, which these copies do not keep. Section numbers and row ids (E6.1, X15, Q22, AF-L3 Q5) are authoritative.
- Ids such as TRIO-n, MJn, OBn and BCn name internal review votes and checks. They are kept as labels; the text of the line they annotate states what it needs.
- The sizing runs that fixed the shift rule's numbers (section 9.4, Check 21 of the main document) are not included. The rule itself is specified in A3 E5.20 to E5.26, and the numbers to reproduce are in section 9.4.
