# Behavioral-test harness

A permanent, **sandboxed** framework for behavioral tests that drive the **real**
companion-emergence engine — as opposed to unit tests that mock it. It generalizes the ad-hoc
harness built during the monologue-bleed bug hunt.

The shape of a behavioral test:

> seed a throwaway persona (**Canary**) → stand up the real bridge → drive it with an
> LLM-simulated human (**Bob**) → run a **detector** over the replies → optionally orchestrate
> multiple **arms** that survive usage-limit stalls.

It can exercise ~anything except the GUI.

## Safety first: strong sandbox isolation

The harness is expected to run on developer **laptops** where a real companion lives. Therefore
the #1 design requirement is that **nothing it does may touch anything outside its temporary
sandbox**. Every run goes through the `sandbox()` context manager, which:

- redirects `KINDLED_HOME` (all persona state) and `CLAUDE_CONFIG_DIR` (the `claude` CLI) into a
  fresh temp dir;
- seeds **only** the claude auth credential (never your global `CLAUDE.md` / skills / settings);
- forces the test persona to `notes_enabled=false` + `kindled_relay_url=null` (the only gated
  external write / phone-home paths);
- **fingerprints guarded real-home roots before the run and asserts they are unchanged after** —
  a leak fails the test loudly.

## What it can / can't do

**Can:** drive real multi-turn conversations against a real persona + bridge; build "aged" personas
(a real multiply-folded compaction/incident regime) as fixtures; plug in your own symptom detector;
run multi-arm A/B matrices; survive usage-limit stalls and resume.

**Can't (yet):** test the GUI; run the token-spending behavioral runs in CI (they need real `claude`
auth and cost money — they're marker-gated and run manually/locally); Agent-Bob (a persistent-agent
user-sim) lands in a later phase.

## Usage

_Filled in as Phase 1 lands. The framework's own token-free unit tests live in
`tests/unit/harness/` and run in CI; the worked behavioral example (token-spending) is under
`examples/` and is marked `@pytest.mark.requires_claude_cli` + `@pytest.mark.behavioral`._

## Status

Phase 1 (Dumb-Bob, end-to-end + sandbox isolation) — **in progress.**
