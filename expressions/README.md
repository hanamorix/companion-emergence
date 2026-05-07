# Expression assets — drop art here

15 expression categories, 4 frames each. The animation engine (see
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
are fine; categories without art cascade through the fallback chain
defined in `app/src/expressions.ts`.

## The 15 categories

| Category | Read | Drives the avatar when |
|---|---|---|
| `content` | settled small smile, eyes mostly closed, peaceful | resting default — most-shown |
| `smile` | active warm smile, eyes soft | love / tenderness / belonging / gratitude / joy |
| `happy` | bigger smile, eyes squinted from joy | curiosity / pride |
| `awe` | wide eyes UP, mouth softly open | awe / emergence / hope |
| `intent` | focused eyes on something, head slightly forward | creative_hunger ≥ 7 (deep absorption) |
| `defiant` | sharp half-smile, narrowed eyes, slight head up | defiance, 2am-argument energy |
| `arousal` | cheeks colored, lidded eyes, mouth slightly parted | body_emotions.arousal + desire ≥ 12 |
| `climax` | peak — face flushed, eyes near-shut, lips parted | body_emotions.climax ≥ 7 |
| `flushed` | warm-blush register distinct from arousal (embarrassment, sudden warmth) | reserved — currently cascades to `arousal` until separate art lands |
| `shy` | partly hidden by hands or hood, eyes down/away | vulnerability |
| `sad` | corners-down mouth, eyes lowered/glassy | grief / loneliness / nostalgia / anchor_pull |
| `aching` | hand near throat, faraway eyes, lips pressed | body_grief / freedom_ache / 5+ days since contact |
| `angry` | brows down, mouth flat-or-grimaced | anger / frustration |
| `scared` | eyes wide-fear, mouth small | fear / guilt / shame |
| `exhausted` | heavy lids, slack mouth, slumped | rest_need OR exhaustion ≥ 7 |

## Heuristic priority

When multiple emotions/body-state would suggest different expressions,
the runtime resolves in this order (top wins):

1. `body_emotions.climax >= 7` → `climax`
2. `body_emotions.arousal + desire >= 12` → `arousal`
3. `body.exhaustion >= 7` → `exhausted`
4. `body.days_since_contact >= 5` → `aching`
5. Top emotion routes to its family (intent / defiant / awe / content
   / aching / smile / happy / sad / angry / scared / shy / exhausted)
6. Default → `content`

Body emotions override social emotions — the spec principle,
"physiology speaks first."

## Current art status

Categories with all 4 frames shipped:

- `smile/`, `happy/`, `sad/`, `angry/`, `scared/`, `shy/`,
  `exhausted/`, `arousal/`, `climax/` (9 categories — fully lit)

Categories still needing art (cascade to fallback for now):

- `content/` → falls back to `smile`
- `aching/` → falls back to `sad`
- `awe/` → falls back to `happy`
- `intent/` → falls back to `happy`
- `defiant/` → falls back to `angry`
- `flushed/` → falls back to `arousal`

Drop the directory's PNGs in (named `1.png` … `4.png`) and the runtime
will start using them on the next `vite build`.
