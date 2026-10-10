# Experiment: illumination generalisation for lunar shadow segmentation

> Does a shadow detector trained under some sun angles still work under others?
> This directory answers that with a fully reproducible, CPU-only, numpy-only
> experiment — no PyTorch, no scikit-learn, no downloaded data.

Everything here is generated from the physics. `run.py` trains and evaluates;
`results.json` / `results.md` hold the numbers; the two figures are the argument.

---

## Why this is worth doing

The companion [algorithm guide](../../docs/algorithm-guide.md) §8.6 says a shadow
mask derived from a DEM must not be used as a per-pixel training label. This
experiment is the constructive version of that warning: it generates labels that
*are* trustworthy (they come from the same physical model that renders the
image), and then measures how much a learned detector actually degrades when
the illumination changes.

The 38× shadow-length scaling from the guide is the reason the question matters:
at the lunar poles the sun sits at 1–2°, so a rover's entire operational
envelope is set by where shadows fall. A detector that only works at the sun
angle it was trained at is not a detector.

## What is generated

For each terrain, the same DEM and the same albedo field are rendered under many
sun geometries, giving paired (image, shadow mask) samples.

```
I = gain * albedo * ( illum * max(cos(incidence), 0) + ambient_field ) + noise
```

Four choices in that line carry the whole experiment:

| Term | Value | Why it matters |
|---|---|---|
| `illum` (cast shadow) | from `lunar_shadow.compute_illumination()` | A pixel can face the sun and still be dark because a ridge blocks it. Omitting this term makes the label and the image contradict each other — see the "two bugs" section below. |
| `ambient_field` | `0.005` (×0.5–3, spatially varying) | On an airless body a shadowed pixel only receives secondary scattering. Very low → the shadow signal is weak. |
| `albedo` | `0.13 … 1.0`, smooth field | Real lunar materials span ~0.1 (dark mare, glassy pyroclastics) to ~1.0. Dark rock is a genuine confounder for "shadow". |
| `gain` | `0.7 … 1.4`, per image | Camera exposure / radiometric calibration uncertainty. **This is what breaks fixed thresholds**, because the absolute brightness of a shadow is no longer a constant. |

Splits are disjoint in **both** terrain and sun elevation:

- train: elevations {8, 15, 25, 40}°, azimuths {0, 90, 180, 270}°, 10 terrains → 160 samples
- test: elevations {4, 12, 20, 30}°, azimuths {45, 135, 225, 315}°, 8 terrains → 128 samples

## Models

| id | model | what it represents |
|---|---|---|
| **B0** | global brightness threshold | the classic "shadow = dark pixel" detector |
| **B0rel** | threshold on `I / mean_local(I)` | the cheapest gain-invariant cue: dividing by a local mean cancels any global multiplicative gain |
| **B1** | per-pixel logistic regression, 4 local features | a small learned model (brightness, local mean, local std, invariant brightness) |
| **B2** | DEM oracle | computes shadows from the DEM with the **true** sun geometry — the ceiling, not a competitor |

Thresholds are fitted by **balanced accuracy**, not F1: with 88% shadow pixels at
4°, "call everything shadow" scores a respectable F1 while being useless.

---

## Results

### Experiment 1 — disjoint elevation bands (train 8/15/25/40°, test 4/12/20/30°)

| model | precision | recall | F1 | IoU |
|---|---|---|---|---|
| B0 global threshold (train fit) | 0.952 | 0.999 | **0.975** | 0.951 |
| B0 global threshold (on test) | 0.971 | 0.997 | **0.984** | 0.968 |
| B0rel local-contrast (train) | — | — | 0.836 | — |
| B0rel local-contrast (test) | — | — | 0.837 | — |
| B1 logistic regression (train) | — | — | 0.954 | — |
| B1 logistic regression (test) | 0.956 | 0.983 | **0.969** | 0.936 |
| B1 with 3 px tolerance (test) | — | — | **0.998** | — |

B1 per test elevation — **higher sun angles are harder**:

| test elevation | F1 | IoU | recall | precision |
|---|---|---|---|---|
| 4° | 0.989 | 0.978 | 0.999 | 0.979 |
| 12° | 0.981 | 0.963 | 0.998 | 0.965 |
| 20° | 0.967 | 0.935 | 0.993 | 0.941 |
| 30° | 0.939 | 0.886 | 0.978 | 0.904 |

### Experiment 2 — train-elevation × test-elevation matrices

Full tables in [`results.md`](results.md); figure in [`fig2_matrices.png`](fig2_matrices.png).

| | matched lighting (diagonal) | shifted lighting (off-diagonal) | degradation |
|---|---|---|---|
| B0 global threshold | 0.980 | 0.981 | **−0.001** (none) |
| B0rel local-contrast | 0.875 | 0.824 | +0.051 |
| B1 logistic regression | 0.956 | 0.931 | **+0.026** |

B1 interpolation (test elevation inside the trained band) 0.938 vs extrapolation
(outside it) 0.878.

**The asymmetry is the single most interesting number here.** A model trained at
low sun is a good model at high sun; the reverse fails badly:

| trained at | F1 at 4° | F1 at 40° |
|---|---|---|
| 8° | 0.99 | **0.87** |
| 15° | 0.99 | 0.88 |
| 25° | 0.97 | 0.89 |
| **40°** | **0.80** | 0.92 |

---

## What the results actually mean

**1. In this setting the task is genuinely easy, and that is a finding, not a
failure.** All three model families clear 0.83 F1 and a fixed global threshold
clears 0.98. The reason is visible in `fig1_pairs.png`: because `ambient` is so
low, a shadowed pixel is always pinned near the sensor floor
(`albedo × gain × ambient ≈ 0.002 … 0.03`), while a lit pixel spans 0 → 1. The
histograms in the bottom row show the two populations essentially never overlap,
at every elevation. So "shadow = dark" survives both the gain variation and the
elevation shift.

**2. The real degradation is elevation transfer, and it is asymmetric.** Training
at 40° loses 0.19 F1 when deployed at 4°, because the elevation change also moves
the shadow *fraction* from 0.31 to 0.88 — the model's decision boundary sees a
prior shift far outside what it was fitted on. Training at 8° costs almost
nothing at 40° (0.87 vs 0.92 matched). Practical rule: **train at the lowest sun
angles you can, even if you care about high ones.**

**3. The "illumination-invariant" cue is worse here, and the reason is
instructive.** B0rel trades an absolute floor for a relative one, and the
relative one drifts with sun elevation (its fitted threshold moves 0.86 → 0.35
as the training elevation goes 8° → 40°), which is why it is the only model with
a real diagonal advantage. Normalising away the gain also normalises away the
signal: when the physics hands you a hard, near-absolute shadow floor, exploiting
it beats being clever.

**4. Tolerance-based F1 matters, again.** B1 scores 0.969 per pixel but 0.998
with a 3 px tolerance. Everything the model gets "wrong" is boundary pixels —
exactly the §8.6 point, now confirmed on generated data rather than asserted.

**5. The oracle is exact (1.000 at every elevation),** which is a sanity check on
the whole pipeline: the labels really are the physics, so every number above is
attributable to the model, not to label noise.

## Limits — do not over-claim from this

- The images are **synthetic**. Real shadows also contend with scattered light
  from the camera optics, non-Lambertian (Hapke) surface photometry, and
  saturated or blooming pixels. `docs/algorithm-guide.md` §5 is explicit that
  lunar photometry needs a Hapke model, and this renderer is Lambertian.
- Labels come from the renderer, so they are exact by construction. That isolates
  the generalisation question but makes the absolute F1 numbers optimistic
  compared with any real orbital-derived label.
- Only sun **elevation** is varied systematically; azimuth is varied at train
  time but the sweep is fixed at 135°, so azimuth transfer is not measured.
- No terrain is shared between splits, so terrain transfer is measured, but the
  synthetic DEM family (fractal noise + rimmed craters) is narrow.

## Reproduce

```bash
python dataset.py        # renders and caches all pairs into data/   (~1 min)
python run.py            # trains and evaluates; writes results.json/.md
python make_figures.py   # needs matplotlib; writes fig1_pairs.png, fig2_matrices.png
```

`dataset.py` prints the brightness-separability report; `results.md` carries the
full matrices.

---

## Two bugs this experiment caught in its own first version

Both were silent — the code ran and produced plausible numbers — and both are
worth knowing if you write a similar renderer.

**Bug 1: the image did not include the cast shadow.** The first renderer computed
brightness from the local surface orientation only, so a pixel facing the sun was
bright even when a distant ridge blocked the sunlight — while its label said
"shadow". The tell-tale sign was a shadow fraction of 0.94 next to a
`cos(incidence) > 0` fraction of 0.67 inside the shadowed set. Every model looked
mediocre for a reason that had nothing to do with the models. Fix: multiply by
the illumination mask, computed once and shared between the renderer and the
labeller.

**Bug 2: the threshold grid was capped at 0.6 and the fit used F1.** For the
low-elevation cases the optimum landed exactly on the grid boundary, so the
"fitted" threshold was really "the largest value I was allowed to try". And with
88% shadow pixels, an F1-maximising search happily returns the degenerate
"everything is shadow" answer. Fix: search the full [0, 1] range and select on
balanced accuracy.
