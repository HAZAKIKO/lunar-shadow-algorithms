# -*- coding: utf-8 -*-
"""
compare_intermediates.py —— find the FIRST step where the C port diverges
========================================================================

When two implementations of the same geometry disagree, "the mask is wrong" is
useless information. This script walks the pipeline stage by stage and reports
the first quantity that differs:

    DEM -> gx/gy -> cos(incidence) -> illumination mask

Usage:
    python compare_intermediates.py --workdir DIR
where DIR is a writable scratch directory (created by verify_c_vs_python.py or
by hand) containing dem.txt, gx.txt, gy.txt, cos_inc.txt, illum from the C run.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import lunar_shadow as ls  # noqa: E402


def read_txt(path: str) -> np.ndarray:
    with open(path, encoding="ascii") as f:
        rows, cols = (int(x) for x in f.readline().split())
        vals = np.array([float(line) for line in f if line.strip()])
    return vals.reshape(rows, cols)


def report(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    d = np.abs(a - b)
    mx, mean = float(d.max()), float(d.mean())
    n_bad = int((d > tol).sum())
    ok = n_bad == 0
    print(f"  {name:16} max|Δ|={mx:12.6g}  mean|Δ|={mean:12.6g}  "
          f"cells over tol({tol:g})={n_bad:6d}  [{'OK' if ok else 'DIVERGES'}]")
    if not ok:
        # where does it start? report the first differing cell in scan order
        idx = np.argwhere(d > tol)
        r, c = idx[0]
        print(f"       first divergent cell: row={r} col={c}  "
              f"python={a[r, c]:.10g}  C={b[r, c]:.10g}")
        print(f"       (and {len(idx) - 1} more cells)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--az", type=float, default=137.0)
    ap.add_argument("--el", type=float, default=6.0)
    args = ap.parse_args()

    wd = args.workdir
    need = ["dem.txt", "gx.txt", "gy.txt", "cos_inc.txt"]
    missing = [n for n in need if not os.path.exists(os.path.join(wd, n))]
    if missing:
        print("missing from the scratch dir:", missing)
        print("run the C program with --dump-intermediates first")
        return 2

    dem = read_txt(os.path.join(wd, "dem.txt"))
    print(f"DEM {dem.shape}, az={args.az}, el={args.el}")

    # ground truth from the Python reference
    gx_p, gy_p, _ = ls.terrain_gradient(dem, cell=1.0)
    dcol, drow, dz = ls.sun_direction(args.az, args.el)
    cos_p = (-gx_p * dcol - gy_p * drow + dz) / np.sqrt(gx_p ** 2 + gy_p ** 2 + 1.0)
    P = ls.compute_illumination(dem, args.az, args.el, cell=1.0, use_slope_term=False)
    P_slope = ls.compute_illumination(dem, args.az, args.el, cell=1.0, use_slope_term=True)

    print("\nstage-by-stage (each row compares one quantity):")
    ok = True
    ok &= report("gx", gx_p, read_txt(os.path.join(wd, "gx.txt")), 1e-6)
    ok &= report("gy", gy_p, read_txt(os.path.join(wd, "gy.txt")), 1e-6)
    ok &= report("cos_incidence", cos_p, read_txt(os.path.join(wd, "cos_inc.txt")), 1e-6)

    ill_path = os.path.join(wd, "illum.txt")
    if os.path.exists(ill_path):
        C_ill = read_txt(ill_path)
        print()
        print(f"  illumination (no slope): C shadow frac "
              f"{float((C_ill == 0).mean()):.4f} vs Python {(P == 0).mean():.4f}")
        print(f"  illumination (+ slope):  Python shadow frac {(P_slope == 0).mean():.4f}")
        ok &= report("illum (no slope)", P.astype(float), C_ill, 0.0)

    print("\nfirst divergence is reported above; everything before it is correct.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
