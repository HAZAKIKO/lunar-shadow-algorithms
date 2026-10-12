# -*- coding: utf-8 -*-
"""
probe_shift.py —— measure, rather than assume, what offset _shift_zero applies
=============================================================================

Two conventions are possible and I flip-flopped between them three times, which
is exactly the failure mode the algorithm guide warns about. So measure it:

    given  shifted = _shift_zero(a, drow, dcol, fill)
    find   (dr, dc)  such that  shifted[i, j] == a[i + dr, j + dc]

The answer is printed for several offsets. Everything else in the codebase must
then be written against THIS measured convention, not against a comment.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import lunar_shadow as ls  # noqa: E402


def measure(drow: int, dcol: int) -> tuple[int | None, int | None]:
    n = 7
    a = np.arange(n * n, dtype=float).reshape(n, n)
    s = ls._shift_zero(a, drow, dcol, fill=-999.0)
    # find (dr, dc) by testing displacements on an interior pixel
    i, j = 3, 3
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            ii, jj = i + dr, j + dc
            if 0 <= ii < n and 0 <= jj < n and s[i, j] == a[ii, jj]:
                return dr, dc
    return None, None


def main() -> None:
    print("measured convention of _shift_zero(a, drow, dcol):")
    print("  offset given      ->  sampled source cell")
    print("  " + "-" * 46)
    for drow, dcol in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 3), (-2, -3)]:
        dr, dc = measure(drow, dcol)
        sign = "a[i + dr, j + dc]" if dr is not None else "?"
        print(f"  drow={drow:+3d} dcol={dcol:+3d}  ->  "
              f"{sign} with dr={dr} dc={dc}")

    print()
    # now state the rule in words, and check it holds for the two cases the
    # forward model depends on
    print("rule check:")
    a = np.arange(49, dtype=float).reshape(7, 7)
    s = ls._shift_zero(a, 1, 1, fill=-999.0)
    print(f"  _shift_zero(a, +1, +1)[3,3] = {s[3,3]:.0f}   "
          f"a[2,2]={a[2,2]:.0f}  a[4,4]={a[4,4]:.0f}")
    s = ls._shift_zero(a, -1, -1, fill=-999.0)
    print(f"  _shift_zero(a, -1, -1)[3,3] = {s[3,3]:.0f}   "
          f"a[2,2]={a[2,2]:.0f}  a[4,4]={a[4,4]:.0f}")


if __name__ == "__main__":
    main()
