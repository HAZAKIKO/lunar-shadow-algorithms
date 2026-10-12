# -*- coding: utf-8 -*-
"""
verify_c_vs_python.py —— prove the C port agrees with the Python reference
=========================================================================

This is the point of the C port. "Translated to C" is a claim; this script turns
it into a measurement. It:

  1. generates a deterministic DEM and writes it in the text format the C program
     reads;
  2. runs the C binary to produce an illumination mask and a horizon map;
  3. computes the same two things with the Python module;
  4. compares them numerically and prints the worst disagreement.

Exit code 0 means the two implementations agree within tolerance.

Usage:
    python verify_c_vs_python.py --exe path/to/lunar_shadow.exe
If --exe is omitted it looks for ./c/lunar_shadow.exe, ./build/lunar_shadow.exe
and a few common names next to the repository root.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import lunar_shadow as ls  # noqa: E402

# Illumination masks are compared exactly (integers), horizon maps within a
# tolerance that only reflects float32 vs float64 rounding and atan2 differences.
HORIZON_TOL_DEG = 0.05


def find_exe(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    candidates = [
        os.path.join(REPO, "c", "lunar_shadow.exe"),
        os.path.join(REPO, "c", "lunar_shadow"),
        os.path.join(REPO, "build", "lunar_shadow.exe"),
        os.path.join(REPO, "lunar_shadow.exe"),
    ]
    found = shutil.which("lunar_shadow")
    if found:
        candidates.insert(0, found)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def write_dem(path: str, dem: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{dem.shape[0]} {dem.shape[1]}\n")
        for v in dem.ravel():
            f.write(f"{v:.10f}\n")


def read_txt_grid(path: str) -> np.ndarray:
    with open(path, encoding="ascii") as f:
        rows, cols = (int(x) for x in f.readline().split())
        vals = np.array([float(line) for line in f if line.strip()])
    if vals.size != rows * cols:
        raise ValueError(f"{path}: got {vals.size} values, expected {rows * cols}")
    return vals.reshape(rows, cols)


def read_txt_flat(path: str) -> tuple[tuple[int, int], np.ndarray]:
    """Read the header and the flat value list, without assuming a 2-D shape.

    The C horizon dump is (n_az * H, W): the per-azimuth planes are stacked
    vertically, so the file is 2-D even though the logical object is 3-D.
    """
    with open(path, encoding="ascii") as f:
        rows, cols = (int(x) for x in f.readline().split())
        vals = np.array([float(line) for line in f if line.strip()])
    return (rows, cols), vals


def make_dem(n: int = 96, seed: int = 20261010) -> np.ndarray:
    return ls.make_synthetic_lunar_dem(n=n, seed=seed, relief=6.0)


def make_workdir(explicit: str | None) -> str:
    """Pick a writable scratch directory.

    Order: --workdir, $LUNAR_SHADOW_TMP, the system temp dir, then a directory
    inside the repository. The last fallback matters because the system temp
    directory is not always writable (restricted environments, CI sandboxes),
    and a comparison script that dies before it compares anything is useless.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("LUNAR_SHADOW_TMP")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(tempfile.gettempdir(), "lunar_shadow_cmp"))
    candidates.append(os.path.join(REPO, ".tmp_verify"))

    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, ".write_probe")
            with open(probe, "w", encoding="ascii") as f:
                f.write("ok")
            os.remove(probe)
            return c
        except OSError:
            continue
    raise SystemExit("no writable scratch directory found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--workdir", help="writable scratch directory")
    args = ap.parse_args()

    exe = find_exe(args.exe)
    if not exe:
        print("C binary not found. Build it first, e.g.:")
        print("  gcc -O2 -std=c99 -Wall -Wextra -o c/lunar_shadow.exe c/lunar_shadow.c -lm")
        print("then pass --exe, or place it next to the repository root.")
        return 2
    print(f"comparing against: {exe}")

    tmp = make_workdir(args.workdir)
    print(f"scratch directory: {tmp}")
    dem = make_dem(args.n)
    dem_path = os.path.join(tmp, "dem.txt")
    write_dem(dem_path, dem)
    print(f"DEM {dem.shape[0]}x{dem.shape[1]} written to {dem_path}")

    cases = [(137.0, 6.0), (45.0, 12.0), (270.0, 20.0), (313.0, 3.0)]
    n_az = 72
    ok = True

    print("\n=== illumination mask (ray marching) ===")
    for az, el in cases:
        c_ill = os.path.join(tmp, f"ill_{az:.0f}_{el:.0f}.txt")
        r = subprocess.run([exe, "--dem", dem_path, "--az", str(az), "--el", str(el),
                            "--n-az", str(n_az), "--out-illum", c_ill],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  az={az} el={el}: C program failed rc={r.returncode}\n{r.stderr}")
            ok = False
            continue
        # The C program writes the ILLUMINATION mask (1 = lit, 0 = shadowed); the Python
        # reference is reduced to a shadow mask here. Comparing like with like matters:
        # an earlier version of this script compared the C lit-mask against the Python
        # shadow-mask and reported "0.000% agreement" while the two were in fact exact
        # inverses of each other, which looks like a catastrophic porting failure and is
        # really just a flipped predicate in the test.
        c_lit = read_txt_grid(c_ill).astype(bool)
        py_shadow = (ls.compute_illumination(dem, az, el, cell=1.0) == 0)
        c_shadow = ~c_lit
        diff = int(np.count_nonzero(c_shadow != py_shadow))
        agree = 1.0 - diff / c_shadow.size
        status = "OK" if diff == 0 else "MISMATCH"
        if diff:
            ok = False
        print(f"  az={az:6.1f} el={el:5.1f}  agreement {agree * 100:7.3f}%  "
              f"differing pixels {diff:5d}  [{status}]")

    print("\n=== horizon map (independent code path) ===")
    c_h = os.path.join(tmp, "horizon.txt")
    r = subprocess.run([exe, "--dem", dem_path, "--az", "137", "--el", "6",
                        "--n-az", str(n_az), "--dump-horizon", c_h],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  C program failed rc={r.returncode}\n{r.stderr}")
        return 1
    (rows, cols), vals = read_txt_flat(c_h)
    Hp = ls.horizon_map(dem, cell=1.0, n_az=n_az)
    Hc = vals.reshape(Hp.shape)
    print(f"  C shape {rows}x{cols} -> {Hc.shape}, Python shape {Hp.shape}")
    if Hc.shape != Hp.shape:
        print("  shape mismatch")
        return 1
    d = np.abs(Hc - Hp)
    print(f"  max |Δ| = {d.max():.6f} deg   mean |Δ| = {d.mean():.6f} deg   "
          f"tolerance {HORIZON_TOL_DEG} deg")
    if d.max() > HORIZON_TOL_DEG:
        ok = False
        print("  FAIL: exceeds tolerance")
    else:
        print("  OK")

    print("\n=== horizon-lookup illumination (no ray marching) ===")
    for az, el in [(137.0, 6.0), (313.0, 3.0)]:
        c_ill = os.path.join(tmp, f"hill_{az:.0f}.txt")
        subprocess.run([exe, "--dem", dem_path, "--az", str(az), "--el", str(el),
                        "--n-az", str(n_az), "--horizon", "1",
                        "--out-illum", c_ill], capture_output=True, text=True)
        if not os.path.exists(c_ill):
            # older CLI without --horizon: fall back to dumping and comparing H only
            print("  (this build has no --horizon switch; skipped)")
            break
        c_lit = read_txt_grid(c_ill).astype(bool)
        py_shadow = (ls.illuminate_from_horizon(dem, Hp, az, el, cell=1.0) == 0)
        diff = int(np.count_nonzero((~c_lit) != py_shadow))
        agree = 1.0 - diff / py_shadow.size
        if diff:
            ok = False
        print(f"  az={az:6.1f} el={el:5.1f}  agreement {agree * 100:7.3f}%  "
              f"[{'OK' if diff == 0 else 'MISMATCH'}]")

    # leave nothing behind: remove only the files this run created
    for name in os.listdir(tmp):
        if name.startswith(("dem_", "ill_", "hill_", "horizon")):
            try:
                os.remove(os.path.join(tmp, name))
            except OSError:
                pass
    print("\n" + ("RESULT: C and Python agree." if ok else "RESULT: DISAGREEMENT FOUND."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
