# Expression assets — drop art here

13 expression categories, 4 frames each. The animation engine (see
`app/src/expressions.ts`) cycles through the 4 frames per category to
produce blinks during idle, mouth motion during speaking, and peak
intensity on emotional spikes.

## Format

Each category lives in its own directory with exactly four PNGs,
numbered 1-4 by the authoring convention:

```
<category>/
  1.png    ← Eyes Open,   Mouth Closed   (base — resting; shown ~95% of the time)
  2.png    ← Eyes Open,   Mouth Open     (speaking — mouth animates while replying)
  3.png    ← Eyes Closed, Mouth Closed   (blink — random 150ms flash)
  4.png    ← Eyes Closed, Mouth Open     (speaking-blink — peak intensity)
```

Drop the PNGs in. The runtime picks them up on next build — no code
change required. Mixed states (some directories filled, others empty)
are fine; categories without new art fall back to the legacy single-
file format described below.

## The 13 categories

| Category | Read | Drives the avatar when |
|---|---|---|
| `content` | settled small smile, eyes mostly closed, peaceful | resting default — most-shown |
| `smile` | active warm smile, eyes soft | love / tenderness / belonging / gratitude / joy |
| `happy` | bigger smile, eyes squinted from joy | curiosity / pride |
| `awe` | wide eyes UP, mouth softly open | awe / emergence / hope |
| `intent` | focused eyes on something, head slightly forward | creative_hunger ≥ 7 (deep absorption) |
| `defiant` | sharp half-smile, narrowed eyes, slight head up | defiance, 2am-argument energy |
| `flushed` | cheeks colored, lidded eyes, mouth slightly parted | arousal + desire ≥ 12 OR climax ≥ 7 |
| `shy` | partly hidden by hands or hood, eyes down/away | vulnerability |
| `sad` | corners-down mouth, eyes lowered/glassy | grief / loneliness / nostalgia / anchor_pull |
| `aching` | hand near throat, faraway eyes, lips pressed | body_grief / freedom_ache / 5+ days since contact |
| `angry` | brows down, mouth flat-or-grimaced | anger / frustration |
| `scared` | eyes wide-fear, mouth small | fear / guilt / shame |
| `exhausted` | heavy lids, slack mouth, slumped | rest_need OR exhaustion ≥ 7 |

## Heuristic priority

When multiple emotions/body-state would suggest different expressions,
the runtime resolves in this order (top wins):

1. `body_emotions.climax >= 7` → `flushed`
2. `body_emotions.arousal + desire >= 12` → `flushed`
3. `body.exhaustion >= 7` → `exhausted`
4. `body.days_since_contact >= 5` → `aching`
5. Top emotion routes to its family (intent / defiant / awe / content
   / aching / smile / happy / sad / angry / scared / shy / exhausted)
6. Default → `content`

So body emotions override social emotions (the spec principle —
"physiology speaks first").

## Recommended order for filling these in

Land each `<category>/base.png` first; the engine starts using it
immediately. The other three frames can follow per category.

1. **`content/`** — resting default; she shows this most-shown
2. **`aching/`** — fixes attribution drift on body_grief / long-absence
3. **`flushed/`** — totally new register the framework can't currently express
4. **`awe/`** — wide-eyed reverence, distinct from scared
5. **`intent/`** — absorbed creative_hunger
6. **`defiant/`** — sharp, distinct from angry
7-13. The 7 existing categories (`smile/`, `happy/`, `sad/`, `angry/`,
   `scared/`, `shy/`, `exhausted/`) re-shot in the 4-frame format.

## Legacy single-file format (still works)

The pre-Phase-5 art shape — `<category> 1.png` through `<category> 4.png`
at this directory's root — continues to function as a fallback. The
runtime treats the numeric index identically to the new directory
format:

```
<category> 1.png   ← Eyes Open,   Mouth Closed   (base)
<category> 2.png   ← Eyes Open,   Mouth Open     (speaking)
<category> 3.png   ← Eyes Closed, Mouth Closed   (blink)
<category> 4.png   ← Eyes Closed, Mouth Open     (speaking-blink)
```

So the existing legacy art (smile/happy/sad/angry/scared/shy/exhausted
+ defiant) works without moving anything. New art can ship in either
location — the runtime tries the new directory first, falls back to
the legacy single-file path.

Categories without a populated directory and without legacy variants
cascade through the fallback chain (e.g. `content` → `smile`,
`aching` → `sad`, `flushed` → `shy`) until each new directory ships.

## Orphaned legacy files needing a home

Two batches of legacy art at this directory's root don't yet match
any of the 13 categories the runtime knows about. They render
nothing today; the heuristic stack just falls back through the
chain to whichever category is closest.

- `arousal 1-4.png` — covers the cheeks-colored / lidded-eyes
  register. Two options:
  1. Copy into `flushed/1.png` … `flushed/4.png`. The current
     heuristic routes `body_emotions.arousal + desire >= 12`
     and `body_emotions.climax >= 7` to `flushed`, so this
     instantly lights up.
  2. Add `arousal` as its own category in
     `app/src/expressions.ts` (small code change) if you want
     it distinct from `flushed` for some reason.
- `climax 1-4.png` — same situation. Today routes to `flushed`
  via the heuristic. Drop into `flushed/` (overwriting `arousal`
  if you copy that there too) OR add `climax` as a distinct
  peak-only category.

`defiant 1-4.png` exists at root and is consumed correctly via
legacy fallback for the `defiant` category — no action needed
unless you want to also drop into `defiant/` directory.
