import math
import cmath
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def fast_quadratic_roots(a, b, c):
    """Solves ax^2 + bx + c = 0."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12: return np.empty(0, dtype=np.float64)
        return np.array([-c / b], dtype=np.float64)

    discriminant = (b * b) - (4.0 * a * c)

    if discriminant < -1e-9:
        return np.empty(0, dtype=np.float64)
    elif discriminant < 1e-9:
        return np.array([-b / (2.0 * a)], dtype=np.float64)

    sqrt_d = math.sqrt(discriminant)
    return np.array([(-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a)], dtype=np.float64)


@njit(cache=True, fastmath=True)
def fast_cubic_roots(a, b, c, d):
    """Solves ax^3 + bx^2 + cx + d = 0."""
    if abs(a) < 1e-12:
        return fast_quadratic_roots(b, c, d)

    # Depressed cubic setup: t^3 + pt + q = 0
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)
    discriminant = (q * q / 4.0) + (p * p * p / 27.0)

    shift = b / (3.0 * a)

    if discriminant > 1e-9:
        sqrt_d = math.sqrt(discriminant)
        u = np.cbrt(-q / 2.0 + sqrt_d)
        v = np.cbrt(-q / 2.0 - sqrt_d)
        return np.array([u + v - shift], dtype=np.float64)
    elif discriminant < -1e-9:
        r = math.sqrt(-(p * p * p) / 27.0)

        ratio = -q / (2.0 * r)
        ratio = max(-1.0, min(1.0, ratio))

        phi = math.acos(ratio)
        mag = 2.0 * math.sqrt(-p / 3.0)

        r1 = mag * math.cos(phi / 3.0) - shift
        r2 = mag * math.cos((phi + 2.0 * math.pi) / 3.0) - shift
        r3 = mag * math.cos((phi + 4.0 * math.pi) / 3.0) - shift
        return np.array([r1, r2, r3], dtype=np.float64)
    else:
        if p == 0.0 and q == 0.0:
            return np.array([-shift], dtype=np.float64)
        else:
            u = np.cbrt(-q / 2.0)
            return np.array([2.0 * u - shift, -u - shift], dtype=np.float64)


@njit(cache=True, fastmath=True)
def fast_quartic_roots(A, B, C, D, E):
    """Solves Ax^4 + Bx^3 + Cx^2 + Dx + E = 0 using a robust Descartes/Ferrari Method."""
    if abs(A) < 1e-12:
        return fast_cubic_roots(B, C, D, E)

    # Normalize to x^4 + bx^3 + cx^2 + dx + e = 0
    b = B / A
    c = C / A
    d = D / A
    e = E / A

    # Depressed quartic setup: y^4 + py^2 + qy + r = 0
    p = c - (3.0 * b * b / 8.0)
    q = d - (b * c / 2.0) + (b * b * b / 8.0)
    r = e - (b * d / 4.0) + (b * b * c / 16.0) - (3.0 * b * b * b * b / 256.0)

    if abs(q) < 1e-12:
        # Biquadratic case: y^4 + py^2 + r = 0
        quad_roots = fast_quadratic_roots(1.0, p, r)
        real_roots = []
        for z in quad_roots:
            if z >= 0.0:
                sqrt_z = math.sqrt(z)
                real_roots.append(sqrt_z - (b / 4.0))
                real_roots.append(-sqrt_z - (b / 4.0))
        return np.array(real_roots, dtype=np.float64)

    # Standard Descartes Resolvent cubic: z^3 - p*z^2 - 4*r*z + (4*p*r - q^2) = 0
    cubic_roots = fast_cubic_roots(1.0, -p, -4.0 * r, 4.0 * p * r - q * q)

    z = max(cubic_roots)

    # Float precision safeguard
    if z < p + 1e-12:
        return np.empty(0, dtype=np.float64)

    A_sq = math.sqrt(z - p)

    # Split into two quadratics: y^2 +/- A_sq * y + (z/2 -/+ q / (2*A_sq)) = 0
    quad1 = fast_quadratic_roots(1.0, -A_sq, (z / 2.0) + (q / (2.0 * A_sq)))
    quad2 = fast_quadratic_roots(1.0, A_sq, (z / 2.0) - (q / (2.0 * A_sq)))

    shift = b / 4.0
    real_roots = []
    for root in quad1:
        real_roots.append(root - shift)
    for root in quad2:
        real_roots.append(root - shift)

    return np.array(real_roots, dtype=np.float64)