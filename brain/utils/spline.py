"""A monotone cubic spline through given points (the PCHIP curve), hand-written with no dependency.

``monotone_spline(points)`` returns a function ``f(x)`` that passes exactly through every given point and never
overshoots between them: on a stretch where the points only rise (or only fall), the curve only rises (or falls).
It is the same curve as scipy's ``PchipInterpolator``, built here so the project needs no scipy.

How the curve is built. With ``h_k = x[k+1] - x[k]`` and ``d_k = (y[k+1] - y[k]) / h_k``, every point gets a slope
``m_k`` and each stretch between two points is the standard cubic Hermite curve for those two points and slopes.
  * An interior point has slope 0 where the stretch before it and the stretch after it do not both rise or both fall
    (``d_{k-1} * d_k <= 0``). Otherwise it has the weighted harmonic mean of the two ``d`` values, with weights
    ``w1 = 2 h_k + h_{k-1}`` and ``w2 = h_k + 2 h_{k-1}``: ``m_k = (w1 + w2) / (w1 / d_{k-1} + w2 / d_k)``.
  * An end point takes the three-point slope ``s = ((2 h_0 + h_1) d_0 - h_0 d_1) / (h_0 + h_1)``, then: 0 if ``s`` and
    ``d_0`` do not point the same way (``s * d_0 <= 0``); ``3 d_0`` if the first two stretches change direction
    (``d_0 * d_1 <= 0``) and ``|s| > 3 |d_0|``; otherwise ``s``. The last point is the same rule mirrored.

Outside the given x range the function holds the nearest end point's y flat. Two points give a straight line.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence


def _end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    """The slope at an end point: ``h0``/``d0`` are the stretch at the end, ``h1``/``d1`` the next one in."""
    s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if s * d0 <= 0.0:
        return 0.0
    if d0 * d1 <= 0.0 and abs(s) > 3.0 * abs(d0):
        return 3.0 * d0
    return s


def monotone_spline(points: Sequence[tuple[float, float]]) -> Callable[[float], float]:
    """Build the monotone cubic (PCHIP) spline through ``points`` and return it as ``f(x) -> y``.

    ``points`` is a sequence of ``(x, y)`` pairs with strictly increasing x. Raises ``ValueError`` for fewer than 2
    points or an x that is not above the one before it. Two points give a straight line. An x below the first point
    or above the last returns that end point's y.
    """
    if len(points) < 2:
        raise ValueError(f"monotone_spline needs at least 2 points, got {len(points)}")
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    for i in range(len(xs) - 1):
        if not xs[i] < xs[i + 1]:
            raise ValueError(
                f"monotone_spline x values must be strictly increasing: {xs[i]} then {xs[i + 1]}"
            )
    n = len(xs)
    h = [xs[k + 1] - xs[k] for k in range(n - 1)]
    d = [(ys[k + 1] - ys[k]) / h[k] for k in range(n - 1)]

    if n == 2:
        m = [d[0], d[0]]
    else:
        m = [0.0] * n
        for k in range(1, n - 1):
            if d[k - 1] * d[k] > 0.0:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        m[0] = _end_slope(h[0], h[1], d[0], d[1])
        m[n - 1] = _end_slope(h[n - 2], h[n - 3], d[n - 2], d[n - 3])

    def f(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        k = bisect_right(xs, x) - 1
        t = (x - xs[k]) / h[k]
        t2 = t * t
        t3 = t2 * t
        return (
            (2.0 * t3 - 3.0 * t2 + 1.0) * ys[k]
            + (t3 - 2.0 * t2 + t) * h[k] * m[k]
            + (-2.0 * t3 + 3.0 * t2) * ys[k + 1]
            + (t3 - t2) * h[k] * m[k + 1]
        )

    return f
