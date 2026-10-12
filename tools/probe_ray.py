# -*- coding: utf-8 -*-
"""
probe_ray.py —— settle the sign convention against analytic answers, not opinion
===============================================================================

Three separate sign questions have been wrong at least once in this codebase:

  1. `sun_direction()` returns the vector pointing AT the Sun; the cast-shadow ray
     must be marched along it (the shadow itself falls on the OPPOSITE side).
  2. `_shift_zero(a, drow, dcol)` samples `a[i - drow, j - dcol]` (measured, see
     tools/probe_shift.py). Pass the to-Sun direction un-negated and you sample
     AWAY from the Sun.
  3. A pixel is shadowed iff terrain along the to-Sun ray rises ABOVE the ray:
     `z(neighbour) >= z(pixel) + d*tan(el)`. Invert it and flat ground goes black.

Rather than re-argue them, this script tests the shipped implementation against
closed-form answers:

  * a flat DEM must be completely lit
  * a north-south ridge (height 10 m) at column 100, sun due west at 5 deg:
      - horizon angle 25 m east of it must be atan(10/25) = 21.80 deg
      - horizon angle 25 m west of it must be 0
      - the cast shadow must fall EAST of the ridge
  * the same with the sun due east, mirrored
  * the ray march and the horizon map must agree with each other

Run:  python tools/probe_ray.py
Exit: 0 if every analytic check passes.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import lunar_shadow as ls  # noqa: E402

N = 201
RIDGE_COL = 100
RIDGE_H = 10.0
D_NEAR = 25.0
WANT_NEAR = float(np.degrees(np.arctan2(RIDGE_H, D_NEAR)))  # 21.80 deg

failures = 0


def check(name: str, ok: bool, detail: str) -> None:
    global failures
    failures += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  |  {detail}")


def ridge() -> np.ndarray:
    d = np.zeros((N, N))
    d[:, RIDGE_COL] = RIDGE_H
    return d


def main() -> int:
    dem = ridge()
    print(f"analytic: a {RIDGE_H:.0f} m ridge seen from {D_NEAR:.0f} m -> "
          f"{WANT_NEAR:.2f} deg\n")

    print("=== flat DEM must be fully lit ===")
    flat = np.zeros((64, 64))
    for az, el in [(45.0, 5.0), (137.0, 8.0), (270.0, 20.0)]:
        ill = ls.compute_illumination(flat, az, el, cell=1.0, use_slope_term=False)
        check(f"flat DEM, az={az:.0f} el={el:.0f}",
              int((ill == 0).sum()) == 0,
              f"shadowed pixels = {int((ill == 0).sum())}")

    print("\n=== horizon map, sun due WEST (az=270): the ridge is to the EAST ===")
    H = ls.horizon_map(dem, cell=1.0, n_az=72)
    m_w = int(round(270.0 / 5.0)) % 72
    east = float(H[m_w, 100, RIDGE_COL + int(D_NEAR)])   # 25 m east of the ridge
    west = float(H[m_w, 100, RIDGE_COL - int(D_NEAR)])   # 25 m west of it
    check("horizon 25 m east of ridge", abs(east - WANT_NEAR) < 0.5,
          f"{east:.2f} deg (want {WANT_NEAR:.2f})")
    check("horizon 25 m west of ridge", abs(west) < 0.5, f"{west:.2f} deg (want 0.00)")

    print("\n=== horizon map, sun due EAST (az=90): mirrored ===")
    m_e = int(round(90.0 / 5.0)) % 72
    east2 = float(H[m_e, 100, RIDGE_COL + int(D_NEAR)])
    west2 = float(H[m_e, 100, RIDGE_COL - int(D_NEAR)])
    check("horizon 25 m east of ridge (sun east)", abs(east2) < 0.5,
          f"{east2:.2f} deg (want 0.00)")
    check("horizon 25 m west of ridge (sun east)", abs(west2 - WANT_NEAR) < 0.5,
          f"{west2:.2f} deg (want {WANT_NEAR:.2f})")

    print("\n=== cast shadow must fall on the far side from the Sun ===")
    for az, want in [(270.0, "east (cols > 100)"), (90.0, "west (cols < 100)")]:
        ill = ls.compute_illumination(dem, az, 5.0, cell=1.0, use_slope_term=False)
        cols = np.nonzero((ill == 0).any(axis=0))[0]
        if len(cols) == 0:
            check(f"ray march, sun az={az:.0f}", False, "no shadow at all")
            continue
        if az == 270.0:
            ok = int(cols.min()) > RIDGE_COL
        else:
            ok = int(cols.max()) < RIDGE_COL
        check(f"ray march, sun az={az:.0f}", ok,
              f"shadow cols {int(cols.min())}..{int(cols.max())}, want {want}")

    print("\n=== the two independent implementations must agree ===")
    dem2 = ls.make_synthetic_lunar_dem(n=64, seed=20261010, relief=6.0)
    H2 = ls.horizon_map(dem2, cell=1.0, n_az=72)
    for az, el in [(137.0, 6.0), (45.0, 12.0), (270.0, 20.0), (313.0, 3.0)]:
        a = ls.compute_illumination(dem2, az, el, cell=1.0, use_slope_term=False)
        b = ls.illuminate_from_horizon(dem2, H2, az, el, cell=1.0, use_slope_term=False)
        agree = float((a == b).mean())
        check(f"ray vs horizon, az={az:.0f} el={el:.0f}", agree > 0.95,
              f"agreement {agree * 100:.2f}%")

    print(f"\n{'ALL ANALYTIC CHECKS PASS' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
