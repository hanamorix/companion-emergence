"""The reranker's memory model (name-recall fix S220, S208): a rerank call's peak memory
growth is predicted from its SHAPE, `pairs` (B) documents padded to the longest pair of
`length` (L) tokens, as

    growth = a + b * T + c * S      T = B * L (padded tokens),  S = B * L * L

fitted per machine by least squares over the measured calls of the last 24 hours (the
samples live in `cost_series`, series `RAM_SHAPE_SERIES`; this module is pure: no I/O, no
clock). Why this form: the fp16 re-scorer runs attention unfused, so the score tensor
[B, heads, L, L] is live, which is the `S` term; the rest of the activations scale with the
padded tokens, the `T` term (`ram-scaling-grid.md`: RMSE 19-27 MB over 22 shapes, against up
to 171 MB for a line through the padded tokens alone).

Derived rules (no tuned constants):

* The two slopes are non-negative: more pairs or longer pairs cannot use less memory. The
  fit is the best non-negative least squares over the term subsets {T, S}, {T}, {S}, {}
  (the intercept is always free; a flat zone at small shapes would otherwise give a negative
  slope and a prediction that falls as the batch grows). A prediction is clamped at 0.
* The fit is usable ("enough distinct shapes") only when the window holds more distinct
  shapes than the formula has parameters (3, so at least 4: the fit has a residual and its
  error is observable), the columns (1, T, S) over the distinct shapes have full rank (exact
  integer determinant), and the window holds a sample taken at the model's FULL input length
  (S200's "worst case per pair": the S193 regression was a long first batch, and shapes that
  are all short say nothing about it). The full-length sample is marked in its packed shape.
* The ladder that makes the samples marks the end of a COMPLETE pass with one extra row (the
  zero shape, `COMPLETE_MARKER`): the single pairs at falling lengths, then the batches of
  increasing size up to the largest a recall sends (or as far as the headroom and the time budget
  allow). The row is recorded only when the window then holds a measured batch of MORE THAN ONE
  PAIR (S245): a pass stopped by the headroom or the time budget before its first batch records no
  marker and runs again at the next lull. A window without that row is a ladder that did not
  finish (a restart, a reply, an error, a starved pass): its fit would extrapolate batch sizes from
  single pairs (it under-predicts them by 17 to 100 percent), so it is not used for sizing until
  the pass is finished.
* The per-dimension cap (`RamFit.within_measured`, S242, S244): while the fit holds no measured
  batch of more than one pair, a real rerank may have no more pairs, and no longer a padded
  length, than the largest shape measured; that call takes the cosine path instead of
  extrapolating batches from single pairs. The cap lifts only once a batch of more than one pair
  is measured AND the fit is usable at all: a `RamFit` exists only when `fit_ram` accepted the
  window (more than three distinct shapes, full rank, a full-length sample), so the lift never
  applies to a window that fails those conditions. With the S245 marker rule a window the ladder
  completes always holds such a batch; the cap is the defence for a marked window that does not
  (one recorded by an earlier build, or by any other writer of the marker).
* Arithmetic is exact (`fractions.Fraction`): the design columns reach 1e9 and the normal
  equations are badly conditioned in floating point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

# A sample's shape is packed into the integer `x` of a cost-series row:
# pairs * SHAPE_RADIX + length, with the full-length marker in bit 19 of the length field.
SHAPE_RADIX = 1 << 20
_FULL_BIT = 1 << 19
MAX_LENGTH = _FULL_BIT - 1  # the model's token limits are far below this

# The formula's parameters (a, b, c). "Enough" needs more distinct shapes than this.
PARAMETERS = 3

# A sample at the zero shape (no pairs) is not a measurement: it marks a COMPLETE ladder pass.
COMPLETE_MARKER = 0


def encode_shape(pairs: int, length: int, *, full: bool = False) -> int | None:
    """The packed integer for a shape, or `None` when it cannot be packed (a non-positive
    count or length, or a length that does not fit the field)."""
    if pairs < 1 or length < 1 or length > MAX_LENGTH:
        return None
    return pairs * SHAPE_RADIX + (length | (_FULL_BIT if full else 0))


def decode_shape(x: int) -> tuple[int, int, bool]:
    """`(pairs, length, full)` of a packed shape."""
    field = x % SHAPE_RADIX
    return x // SHAPE_RADIX, field & MAX_LENGTH, bool(field & _FULL_BIT)


@dataclass(frozen=True)
class ShapeGroup:
    """The window's samples of one shape: how many (`n`) and the sum of their growth in
    bytes (`sum_y`). `full` is True when any of them was taken at the full input length."""

    pairs: int
    length: int
    full: bool
    n: int
    sum_y: float


def groups_from_rows(rows: Iterable[tuple[int, int, float]]) -> list[ShapeGroup]:
    """Fold `(packed x, count, sum of growth)` rows into one group per (pairs, length):
    the same shape with and without the marker is one shape. The ladder-complete marker row is
    not a shape and is skipped (`ladder_complete` reads it)."""
    merged: dict[tuple[int, int], list] = {}
    for x, count, sum_y in rows:
        if int(x) == COMPLETE_MARKER:
            continue
        pairs, length, full = decode_shape(int(x))
        entry = merged.setdefault((pairs, length), [False, 0, 0.0])
        entry[0] = entry[0] or full
        entry[1] += int(count)
        entry[2] += float(sum_y)
    return [
        ShapeGroup(pairs, length, full, n, sum_y)
        for (pairs, length), (full, n, sum_y) in sorted(merged.items())
    ]


def ladder_complete(rows: Iterable[tuple[int, int, float]]) -> bool:
    """Whether the window holds the marker of a finished ladder pass."""
    return any(int(x) == COMPLETE_MARKER for x, _n, _t in rows)


def _columns(pairs: int, length: int) -> tuple[int, int, int]:
    padded = pairs * length
    return 1, padded, padded * length


def distinct_shapes(groups: Iterable[ShapeGroup]) -> set[tuple[int, int]]:
    return {(g.pairs, g.length) for g in groups}


def full_rank(shapes: Iterable[tuple[int, int]]) -> bool:
    """Whether the columns (1, T, S) over these distinct shapes have rank 3: the exact
    determinant of the 3x3 Gram matrix is not zero."""
    gram = [[0] * 3 for _ in range(3)]
    for pairs, length in set(shapes):
        phi = _columns(pairs, length)
        for i in range(3):
            for j in range(3):
                gram[i][j] += phi[i] * phi[j]
    a, b, c = gram
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return det != 0


def enough_shapes(groups: Sequence[ShapeGroup]) -> bool:
    """The window gives a usable fit: more distinct shapes than parameters, full rank, and
    a sample at the model's full input length."""
    shapes = distinct_shapes(groups)
    return len(shapes) > PARAMETERS and full_rank(shapes) and any(g.full for g in groups)


def dominated(pairs: int, length: int, shapes: Iterable[tuple[int, int]]) -> bool:
    """Whether some measured shape has at least this many pairs AND is at least this long:
    memory cannot shrink as either grows, so such a call needs no more than that shape did."""
    return any(p >= pairs and ln >= length for p, ln in shapes)


@dataclass(frozen=True)
class RamFit:
    """`growth = a + b*T + c*S` in bytes, clamped at 0; `shapes` = distinct shapes behind
    it, `samples` = measurements, `max_pairs` / `max_length` = the largest measured."""

    a: float
    b: float
    c: float
    shapes: int
    samples: int
    max_pairs: int
    max_length: int

    def predict(self, pairs: int, length: int) -> float:
        _, padded, squared = _columns(pairs, length)
        return max(0.0, self.a + self.b * padded + self.c * squared)

    def within_measured(self, pairs: int, length: int) -> bool:
        """S242: whether a prospective rerank of `pairs` documents padded to `length` tokens may
        be sized by this fit. While the fit holds no measured batch of more than one pair
        (`max_pairs <= 1`: a marked window without a batch, which the S245 marker rule no longer
        lets a ladder pass produce) it would extrapolate batch sizes from single pairs, which
        under-predicts them by 17 to 100 percent (the module docstring), so a call may have no
        more pairs, and no longer a padded length, than the largest shape measured: each dimension
        is held to its own largest measured value. Once a batch of two or more pairs is measured
        the cap lifts (S244). This method is only ever asked of a `RamFit`, which `fit_ram` returns
        only for a window with more than three distinct shapes, full rank and a full-length
        sample, so the lift always comes with those conditions."""
        return self.max_pairs > 1 or (pairs <= self.max_pairs and length <= self.max_length)


def _solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """Gauss-Jordan over exact fractions; `None` when the matrix is singular."""
    size = len(rhs)
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = next((r for r in range(col, size) if rows[r][col] != 0), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for r in range(size):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [v - factor * w for v, w in zip(rows[r], rows[col], strict=True)]
    return [rows[i][size] for i in range(size)]


# Term subsets, fewest terms first so a tie keeps the simpler model. Column 0 is the intercept.
_SUBSETS: tuple[tuple[int, ...], ...] = ((0,), (0, 1), (0, 2), (0, 1, 2))


def fit_ram(groups: Sequence[ShapeGroup]) -> RamFit | None:
    """The non-negative least-squares fit over the window's samples (every sample weighs
    the same), or `None` unless `enough_shapes` and the fit predicts some growth."""
    if not enough_shapes(groups):
        return None
    best: tuple[Fraction, list[Fraction], tuple[int, ...]] | None = None
    for subset in _SUBSETS:
        matrix = [[Fraction(0)] * len(subset) for _ in subset]
        rhs = [Fraction(0)] * len(subset)
        for g in groups:
            phi = _columns(g.pairs, g.length)
            for i, ci in enumerate(subset):
                rhs[i] += phi[ci] * Fraction(g.sum_y)
                for j, cj in enumerate(subset):
                    matrix[i][j] += g.n * phi[ci] * phi[cj]
        theta = _solve(matrix, rhs)
        if theta is None or any(t < 0 for t in theta[1:]):
            continue
        # At the optimum the residual sum of squares is sum(y^2) - theta . rhs, so the
        # subset with the larger theta . rhs fits better.
        score = sum((t * r for t, r in zip(theta, rhs, strict=True)), Fraction(0))
        if best is None or score > best[0]:
            best = (score, theta, subset)
    if best is None:  # cannot happen: the intercept-only subset is always feasible
        return None
    _, theta, subset = best
    coef = [Fraction(0)] * 3
    for t, ci in zip(theta, subset, strict=True):
        coef[ci] = t
    fit = RamFit(
        a=float(coef[0]),
        b=float(coef[1]),
        c=float(coef[2]),
        shapes=len(distinct_shapes(groups)),
        samples=sum(g.n for g in groups),
        max_pairs=max(g.pairs for g in groups),
        max_length=max(g.length for g in groups),
    )
    # A model that predicts no growth even at the largest shape it was fitted on is a failed
    # measurement (an unreadable or always-zero peak), not a model: it would switch the RAM
    # bound off without anyone noticing (S204: a window with no positive growth is empty).
    return fit if fit.predict(fit.max_pairs, fit.max_length) > 0.0 else None
