# Lunar illumination / shadow algorithms in C99

> A dependency-free C port of the numerical core of [`../lunar_shadow.py`](../lunar_shadow.py).
> **The author provides different versions for reference** — this directory is the C one.
> Read the Python module first: it carries the full derivations and the reasoning behind
> every convention. This file is the same geometry, arranged for a flight processor.

---

## Why a C version exists

The Python module is the reference implementation and the one to learn from. It is not,
however, what runs on a rover. Radiation-hardened flight processors run C, and dragging a
Python runtime plus NumPy onto a spacecraft is not an option. So the numerically
meaningful core is reproduced here in plain C99 with no dependencies beyond `libm`.

## What is ported

| Function | Purpose |
|---|---|
| `sun_direction()` | to-Sun unit vector in image coordinates |
| `horizontal_direction()` | in-plane unit vector for a compass bearing |
| `terrain_gradient()` | central differences, NumPy-compatible one-sided edges |
| `cos_incidence()` | incidence cosine against the to-Sun vector |
| `compute_illumination()` | ray marching + local slope-orientation (attached shadow) test |
| `horizon_map()` | per-azimuth-bin maximum elevation angle |
| `illuminate_from_horizon()` | table lookup instead of re-marching |
| `image_shadow_mask()` | soft-scored shadow extraction from an image |
| `iou()`, `f1_with_tolerance()` | evaluation metrics used by the self-test |

Not ported, deliberately: SPICE ephemeris handling and the logistic-regression training
loop from the illumination-generalisation experiment. Both belong on the ground segment.

## Build and run

```bash
gcc -O2 -std=c99 -Wall -Wextra -o lunar_shadow lunar_shadow.c -lm

./lunar_shadow --selftest                      # geometry invariants, no input needed
./lunar_shadow --dem dem.txt --az 137 --el 6 --out-illum illum.txt --out-pgm illum.pgm
./lunar_shadow --dem dem.txt --az 137 --horizon h.txt --n-az 72 --dump-horizon h.txt
```

The DEM input is a whitespace-separated text grid whose first two numbers are the row and
column counts. That format is deliberate: it makes the C program trivially diffable against
the Python reference (see below).

> **Windows / MinGW note.** Building inside a directory whose path contains non-ASCII
> characters (for example a Chinese folder name) makes `ld` fail with a flood of
> `cannot find ...: Invalid argument` errors. The compiler itself is fine; the linker is
> not. This was hit for real during development. Build in an ASCII path.

## Verified against the Python reference

"Translated to C" is a claim; this turns it into a measurement:

```bash
python ../tools/verify_c_vs_python.py --exe ./lunar_shadow.exe
```

It generates a deterministic DEM, runs both implementations, and compares them stage by
stage. Current status (96x96 synthetic DEM, four sun geometries):

| Quantity | Agreement with Python |
|---|---|
| illumination mask, ray marching | **100.000%** (0 differing pixels, all four cases) |
| illumination mask, horizon lookup | **100.000%** |
| horizon map values | max abs difference 134.9 deg on 1.35% of cells |

The horizon-map divergence is confined to cells near the grid border where the sampled
cell leaves the DEM; it does **not** propagate into either illumination path, which match
exactly. It is listed here rather than hidden because a number that disagrees is more
useful than a number that was quietly dropped. `tools/compare_intermediates.py` will
localise it further if you need to chase it.

## The sign conventions, which cost real time

Four separate sign errors were made while writing this port and its Python counterpart.
They are documented here because a C port is exactly where they get reintroduced, and
because none of them announces itself:

1. **`sun_direction()` returns the vector pointing AT the Sun.** The cast-shadow ray must
   be marched *along* it; the shadow itself falls on the *opposite* side. Get this wrong
   and shadows appear on the sunlit side of every ridge — which looks entirely plausible
   in a rendered image.
2. **A pixel is shadowed iff terrain along the to-Sun ray rises ABOVE the ray:**
   `z(neighbour) >= z(pixel) + d*tan(elevation)`. Invert it and flat ground comes out
   completely black, which is at least an obvious symptom.
3. **The offset sign depends on the shift helper's convention, and the two must be flipped
   together.** Here the sampled cell is `dem[r - srow][c - scol]`, so an offset sampling
   towards the Sun is the *negated* to-Sun direction. Equivalent formulation: use the
   bearing `az + 180` with a positive offset. Flipping only one of the two is the bug that
   produces shadows mirrored about a ridge.
4. **Never build a horizontal direction with `sun_direction(az, 90)`** — `cos(90) = 0`
   collapses the vector to zero and every downstream result silently keeps its initial
   value. That is what `horizontal_direction()` is for.

`--selftest` pins all four down against closed-form answers:

```
[PASS] sun at due east -> direction (1, 0, 0)
[PASS] sun due north -> drow negative (row grows southward)
[PASS] flat terrain is fully lit (shadowed=0)
[PASS] ridge analytic: 25 m east=21.80 deg (want 21.80), 25 m west=0.00 deg
[PASS] shadow falls east of the ridge (first shadowed col=51, ridge at 50)
[PASS] ray marching vs horizon lookup agree on 99.61% of pixels
[PASS] shadow length round trip (1 m at 1.5 deg -> 38.19 m -> 1.000000 m)
```

The ridge case is the important one: a 10 m ridge lit from due west, with the analytic
horizon angle `atan(10/25) = 21.80 deg` on its eastern side and zero on its western side.
No amount of code review substitutes for it.

## Debugging aids included

| Flag / script | Purpose |
|---|---|
| `--dump-horizon F` | write the horizon map as text, for diffing against Python |
| `--dump-intermediates` | also write `gx`, `gy`, `cos(incidence)` and per-stage shadow counts |
| `--dump-dir D` | where those files go |
| `../tools/probe_shift.py` | measures (does not assume) the shift helper's convention |
| `../tools/probe_ray.py` | tests the shipped code against the analytic ridge answer |
| `../tools/compare_intermediates.py` | reports the FIRST stage where C and Python diverge |

## Limitations

- Single-precision is not used anywhere; this is `double` throughout, as on the ground.
  A flight build would want a careful float32 assessment.
- No SPICE, no ephemeris, no WCS/georeferencing: this is the algorithm core only.
- The horizon map uses one sample per step (nearest cell). Python does the same, so the two
  agree, but a production version would want a conservative sweep over the segment.
- Not optimised. The loops are written to be readable and to mirror Python line by line;
  `horizon_map()` in particular is O(n_az * steps * rows * cols) and would be the first
  thing to restructure (see the Fourier-compressed horizon maps in the algorithm guide §9.1).

## License

MIT, as for the rest of this repository.
