# Lunar Surface Illumination and Shadow Algorithm Guide

[中文](算法指南.md) | **English**

> **This document answers one question**: how do you compute/invert the illumination state at a given position on the lunar surface from sunlight and shadow?
>
> **Scope**: a runnable algorithm implementation + engineering guide. The companion code [`lunar_shadow.py`](../lunar_shadow.py) has all been self-tested ([`check_lunar_shadow.py`](../check_lunar_shadow.py): **all 29 assertions pass**), and there is a figure that can go straight into a paper ([`lunar_shadow_figure.png`](../lunar_shadow_figure.png)).
>
> **Upstream document**: "Survey of Lunar Rover VLN Navigation Technology", §10 special topic "Visual Recognition of Science-Value Regions" (that survey is the companion document for the lunar-rover visual-language-navigation direction and is not included in this repository). This module is the concrete implementation of the "observability $g_{obs}(x, \odot_{sun})$" item in it.

---

## Table of Contents

1. [First, Distinguish the Three Classes of Problems](#1-first-distinguish-the-three-classes-of-problems)
2. [The Four Coordinate Systems and the Azimuth Convention (Where Errors Are Most Likely)](#2-the-four-coordinate-systems-and-the-azimuth-convention-where-errors-are-most-likely)
3. [Class A: Predicting Shadow from a DEM](#3-class-a-predicting-shadow-from-a-dem)
4. [Class A': How to Compute the Sun Position](#4-class-a-how-to-compute-the-sun-position)
5. [Class B: Extracting Shadow from Images](#5-class-b-extracting-shadow-from-images)
6. [Class C: Inferring Geometry and Sun Parameters from Shadow](#6-class-c-inferring-geometry-and-sun-parameters-from-shadow)
7. [Selection Table](#7-selection-table)
8. [Six Pitfalls You Must Know (All Encountered, with Measured Data)](#8-six-pitfalls-you-must-know-all-encountered-with-measured-data)
9. [Pointers to Off-the-Shelf Tools and Data](#9-pointers-to-off-the-shelf-tools-and-data)
10. [How to Use the Code](#10-how-to-use-the-code)
11. [References](#11-references)
12. [Appendix: Self-Test Results (Measured)](#appendix-self-test-results-measured)

---

## 1. First, Distinguish the Three Classes of Problems

The phrase "judging from sunlight and shadow" covers three **completely different** problems. Confusing them is the most common elementary mistake in this direction:

| Class | Input → Output | Typical use | Mathematical nature |
|---|---|---|---|
| **A. Forward modeling** (predicting shadow) | DEM + sun position → illumination/shadow mask | Computing observability, planning observation windows, determining PSR | Deterministic geometry, exactly solvable |
| **B. Inversion** (extracting shadow) | Single image → shadow mask | Producing segmentation labels, detecting obstacles, finding craters | Ill-posed, needs priors and disambiguation |
| **C. Measurement/reconstruction** | Shadow → sun angle / object height / surface slope | Calibration, ephemeris-free sun direction estimation, photoclinometric height completion | Partly ill-posed (see §8.5 azimuthal degeneracy) |

The relationship among the three is: **A provides ground truth, B provides observations, and C connects the two**. What your paper typically needs to do is "use A to synthesize data to train B, and use C for closed-loop validation".

---

## 2. The Four Coordinate Systems and the Azimuth Convention (Where Errors Are Most Likely)

When I implemented this module, **three of the four real bugs were azimuth signs written backwards**. That is why this section comes first.

### 2.1 Image Coordinates

`dem` is an array of shape `(H, W)`, indexed as `dem[row, col]`.

- Increasing `col` = **eastward**
- Increasing `row` = **southward** (i.e. decreasing `row` = northward)
- `cell` = grid cell size (meters)

### 2.2 Azimuth (Compass Bearing)

`az_deg`: **0° = due north, 90° = due east, 180° = due south, 270° = due west** (clockwise).

### 2.3 The Two Versions of the Direction Vector (Critical)

These two must be kept strictly separate in code; mixing them gives shadows flipped by 180°:

```python
sun_direction(az, el)        # 3D unit vector pointing toward the sun (dcol, drow, dz)
                             #   dcol = +sin(az)*cos(el)     positive eastward
                             #   drow = -cos(az)*cos(el)     negative northward (row increases southward)
                             #   dz   = +sin(el)

horizontal_direction(az)     # 2D unit vector in the horizontal plane along azimuth az
                             #   note: this is "looking out toward az", not "the sun is at az"
```

**Three iron rules**:

1. **Ray marching must sample toward the sun**, i.e. extrapolate along `-(dcol, drow)`; the shadow itself falls on the **opposite side** from the sun.
2. **The horizon map stores the "maximum elevation angle in the direction pointing toward the sun"**, so internally you must take the horizontal direction at `az + 180°` (or equivalently negate `horizontal_direction(az)`). Get this 180° wrong and the shadow is drawn on the same side as the sun.
3. **Never use `sun_direction(az, 90.0)` to obtain a horizontal direction** — `cos(90°) = 0`, the vector degenerates to the zero vector, and the entire computation **fails silently** (everything returns the initial value, with no error). This is a pit I actually fell into, which is why `horizontal_direction()` was extracted as a dedicated function.

### 2.4 Sun Position Input

It can be $(az, el)$, or the **sub-solar point longitude/latitude** $(\lambda_s, \phi_s)$. The two convert to each other:

$$\sin(el) = \sin\phi\sin\phi_s + \cos\phi\cos\phi_s\cos(\lambda_s-\lambda)$$
$$az = \operatorname{atan2}\!\big(\sin(\lambda_s-\lambda)\cos\phi_s,\ \cos\phi\sin\phi_s - \sin\phi\cos\phi_s\cos(\lambda_s-\lambda)\big)$$

Code: `sun_az_el_from_subsolar()`.

---

## 3. Class A: Predicting Shadow from a DEM

### 3.1 The Criterion (Be Sure to Get the Direction Right)

Let the point in question be $P$, with solar elevation angle $h_s$ and azimuth $\phi_s$. Walk $d$ meters from $P$ along the horizontal direction **pointing toward the sun** to reach $Q$; then the height of the sun ray at $Q$ is

$$\text{ray}_h = z(P) + d\tan h_s$$

$P$ is in shadow if and only if **there exists some $Q$ whose terrain rises above this ray**:

$$z(Q) \ge z(P) + d\tan h_s \quad\Longleftrightarrow\quad \underbrace{\arctan\frac{z(Q)-z(P)}{d}}_{\text{horizon elevation angle}} \ge h_s$$

**Sanity check (make sure to put it in a unit test)**: on flat terrain ($z$ equal everywhere) the left side $=0 < d\tan h_s$, so **no shadow is produced**. In my first version I had the inequality backwards (written as $\text{ray}_h > z(Q)$), and the result was **100% black on flat terrain** — running this test caught it immediately.

### 3.2 Two Independent Implementations (Both Provided by This Module, for Cross-Validation)

| Method | Function | Complexity | Changing the sun angle |
|---|---|---|---|
| **Ray marching** | `compute_illumination()` | $O(N_{\text{step}}\cdot N_{\text{pix}})$ | Must recompute |
| **Horizon map** | `horizon_map()` + `illuminate_from_horizon()` | Precompute $O(N_{az}\cdot N_{\text{step}}\cdot N_{\text{pix}})$, then **$O(N_{\text{pix}})$ table lookup** | **No recomputation needed** |

**Definition of the horizon map**: for each pixel and each azimuth bin $m$, precompute

$$H(i,j,m) = \max_{d\le d_{\max}} \arctan\frac{z(i+d\sin\phi_m,\ j-d\cos\phi_m) - z(i,j)}{d}$$

After that, "is the sun blocked" reduces to a single comparison: `shaded ⟺ h_s < H(i,j,φ_s)`, with linear interpolation over the azimuth direction.

**Measured cross-consistency** (192×192 synthetic lunar surface, `n_az=72`, i.e. 5° bins):

| Sun (az, el) | Ray-marching shadow fraction | Horizon-map shadow fraction | Pixel agreement rate |
|---|---|---|---|
| (45°, 8°) | 82.45% | 82.24% | 99.76% |
| (137°, 5°) | 92.29% | 91.90% | 98.97%* |
| (270°, 20°) | 50.79% | 49.93% | 97.88% |
| (313°, 3°) | 87.00% | 86.93% | 99.91% |

\* This row is taken from the output of `make_figure.py` at 192×192 with relief=8 (the others come from the check script, with different seeds).

**This is the value of "two independent implementations corroborating each other"**: they are written with completely different loop structures, so agreement means the geometry really is correct.

### 3.3 One Thing That Must Be Added: Local Slope Orientation (Attached Shadow)

Computing cast shadow alone is not enough. **A slope facing away from the sun receives no direct light**; this is not obvious on Earth (there is atmospheric scattering) but is extremely pronounced on the Moon (no atmosphere, so a slope facing away from the sun goes straight to black). The way to do it is to compute the angle between the surface normal and the sun direction:

$$\cos\theta_i = \frac{-g_x d_{col} - g_y d_{row} + d_z}{\sqrt{g_x^2+g_y^2+1}}$$

$\cos\theta_i \le 0$ means it is judged to face away from the sun. Here $g_x = \partial z/\partial \text{east}$, $g_y = \partial z/\partial \text{north}$ (note that $y$ is positive northward, opposite to the image row direction).

**Measured impact** (`n=160, seed=20261010, az=270°, el=20°`):

| Configuration | Ray-marching shadow fraction | Horizon-map shadow fraction | Agreement rate |
|---|---|---|---|
| Cast shadow only | 49.38% | 50.19% | 40.54% |
| Plus local slope orientation | 50.79% | 71.84% | 61.11% |

(This row is data from **before** the direction bug was fixed and only serves to show that "both are necessary"; after the fix the agreement rate rises to 97.88%.)

**A sense of scale in the lunar polar regions**: at the south pole the solar elevation angle is only 1–2° year-round, so

$$L_{\text{shadow}} = \frac{h}{\tan h_s} \approx \frac{h}{\tan 1.5°} \approx 38\,h$$

**A 1 m tall rock can cast a shadow about 38 m long.** So in the lunar polar regions, "long shadows" are not a visual effect but a **dominant observability constraint** — large areas become unobservable because of shadow.

### 3.4 Key Computation Optimizations

Every implementation in this module rests on one trick: **whole-image shifting instead of per-pixel loops**.

```python
shifted = _shift_zero(dem, srow, scol, fill=-inf)   # vectorized shift
```

⚠️ **Zero padding is mandatory; never use `np.roll`**. `np.roll` is **circular/wraparound**: when the shift exceeds the image size it wraps incorrectly, and occlusion decisions at the image edge are also bogus. `_shift_zero` was written specifically for this, filling out-of-bounds locations with `-inf` (meaning "no terrain occlusion").

**Two further early-exit conditions** (important for low-sun-angle scenes, because the shadow fraction is high):
- If a step occludes no new pixels → skip that step;
- If all pixels are already occluded → exit entirely.

---

## 4. Class A': How to Compute the Sun Position

### 4.1 The Accurate Approach: SPICE (Mandatory at Mission Level)

Standard workflow: load the lunar body-fixed frame attitude kernel (`pck`) + the planetary ephemeris kernel (`spk`/`bsp`) → compute the direction of the sun relative to the Moon's center → transform into the `IAU_MOON` body-fixed frame → obtain the **sub-solar point longitude/latitude** → transform into the topocentric frame to get $(h_s, \phi_s)$. In Python use `spiceypy`; for a tutorial see the [SPICE official documentation](https://readthedocs.org/projects/spiceypy/downloads/pdf/latest/) ✅.

⚠️ I only verified the provenance of that document and did **not check the function signatures one by one**; when writing code, follow the official documentation.

### 4.2 Teaching/Simulation Approach: Low-Accuracy Analytic Method

This module's `solar_position_analytic()` and `subsolar_lonlat_analytic()` use:

- a **low-precision series** for the sun's apparent ecliptic longitude (Meeus's simplified formula),
- the Moon's **Cassini state rotation** (rotation period = orbital period ≈ 27.32166 d, synchronous rotation),
- the lunar north pole direction approximated as $(\alpha_p, \delta_p) \approx (269.9949°, 66.5392°)$.

**Accuracy statement (must go into the paper)**: typical error **0.1°–0.5°**, **enough to support teaching and simulation, not enough to support mission-level planning**. What is the difference: mission level requires including the **Moon's physical libration** and higher-order ecliptic longitude terms.

**Self-test results** (see item 5 of [`check_lunar_shadow.py`](../check_lunar_shadow.py)):

| Julian date | Sub-solar point longitude | Sub-solar point latitude |
|---|---|---|
| 2451545 (J2000) | 100.39° | 0.02° |
| 2451635 | −168.91° | −0.01° |
| 2460545 | −30.10° | −0.01° |

The latitude falls within ±1.6° — consistent with the Moon's obliquity of about **±1.54°**, which is evidence of the model's self-consistency. In addition, at 89.5° S the computed solar elevation angle is 0.44°, matching the expectation of low sun angles in polar regions.

### 4.3 Georeferencing of the DEM

`dem_to_lonlat()` provides the simplest equidistant projection: `lon = west + col*res_deg`, `lat = north - row*res_deg`. For real products (LOLA/LROC), handle them according to their accompanying projection information; do not use this approximation.

---

## 5. Class B: Extracting Shadow from Images

### 5.1 Criteria

Lunar images are nearly grayscale; the physical characteristics of shadow are:

1. **Low brightness** (primary criterion);
2. **Low saturation** (valid with a color camera; with a grayscale camera it degenerates into a useless criterion);
3. **Sharp boundaries** (no atmospheric scattering → the shadow boundary is almost a step).

### 5.2 Implementation: Soft Scoring, Not a Hard AND

This module combines the three into one score rather than taking their intersection:

$$\text{score} = \underbrace{\text{darkness}}_{\text{primary cue}} \times \underbrace{\text{sat\_penalty}}_{\text{colour side-evidence}} \times \Big(1 - w_e\,(1-\text{edge\_bonus})\Big)$$

⚠️ **Why a hard AND does not work** (this is a lesson learned from measurements): real lunar shadows are often **large connected regions** whose interior gradient is zero. If you require "every shadow pixel must be adjacent to a strong edge", recall collapses. Measured on a 160×160 synthetic image:

| Implementation | Detected shadow fraction | F1 (tol=2 px) |
|---|---|---|
| Hard AND (brightness ∧ saturation ∧ edge band) | 0.179 | **0.383** |
| Soft scoring (this module) | — | **1.000** |

The ground-truth shadow fraction is 0.855 — the hard AND discards large shadow regions wholesale. **Boundary sharpness is corroborating evidence used for disambiguation; it cannot serve as a necessary condition.**

### 5.3 Illumination-Invariant Features

`illumination_invariant()` performs local brightness normalization $I_{\text{norm}} = I / (I_{\text{local}} + \epsilon)$, used to suppress appearance differences across illumination. This is the standard preprocessing for "illumination-robust detectors" and directly serves §10.3 of the survey.

**A more physical approach** is photometric normalization, but **on the Moon you must use a Hapke-type light-scattering model, not a Lambertian assumption**.

### 5.4 Disambiguation: Combining B with A

**Failure mode of pure thresholding**: low-albedo materials (dark maria, dark halos, glassy debris) are misjudged as shadow.

**The correct way to disambiguate**: use Class A (DEM prediction) to provide a **prior** for "where shadow should geometrically be", and check it for consistency against Class B (low image brightness):

- Image dark **and** DEM predicts shadow → true shadow;
- Image dark **but** DEM predicts illuminated → low-albedo material (**this is the science target you are actually looking for**);
- Image bright **but** DEM predicts shadow → DEM error or a registration problem.

**Read in reverse, the third row is a free innovation point**: this "contradiction map" can itself serve as a science-target candidate generator — because what it flags is precisely "dark patches that the DEM cannot explain".

---

## 6. Class C: Inferring Geometry and Sun Parameters from Shadow

### 6.1 Shadow Length ↔ Object Height

$$h = L_{\text{shadow}}\cdot\tan h_s \qquad\Longleftrightarrow\qquad h_s = \arctan\frac{h}{L_{\text{shadow}}}$$

`shadow_length_to_height()` / `estimate_sun_elevation_from_shadow()`. Their mutual inverseness has been verified on 9 parameter sets, with error < 1e-6.

### 6.2 Ephemeris-Free Estimation of the Sun Direction (Especially Useful for Lunar Rovers)

**The rover itself is a calibration target of known dimensions.**

- Measure the rover body's projected length from the image + the known body height → invert to obtain $h_s$;
- The projection direction directly gives $\phi_s$.

This is exactly the means by which Mars rover teams confirm the sun's azimuth onboard. **This step can be made part of the rover's boot-time self-check procedure**: if the inverted $(h_s,\phi_s)$ deviates from the ephemeris prediction beyond a threshold, it indicates a problem with the camera mounting matrix or the attitude estimate.

### 6.3 Inverting Sun Geometry from a Shadow Mask (Closed-Loop Self-Check)

`recover_sun_geometry()` searches over an $(az, el)$ grid to **maximize the IoU between the predicted shadow mask and the actual shadow mask**. It uses a coarse search (5°) to locate the peak plus a neighborhood fine search (2°/0.5°).

**Measured results**:

| Ground truth (az, el) | Inverted (az, el) | Best IoU | Second-best azimuth IoU | Gap |
|---|---|---|---|---|
| (60°, 6°) | (61°, 6.0°) | 0.996 | 0.987 (@63°) | **0.0089** |
| (225°, 12°) | (226°, 12.0°) | 0.994 | 0.992 (@224°) | **0.0021** |

**The elevation angle is nearly perfect, the azimuth has only ~1° accuracy, and the gap between the best and second-best is under 0.01** — which shows:

> ⚠️ **A single shadow mask is "degenerate" with respect to the solar azimuth (ill-posed); this is not a bug but an identifiability problem inherent to the method.**

The reason is intuitive: the **shape of a cast shadow is determined mainly by the solar elevation angle** and is far less sensitive to azimuth (azimuth mainly affects the shadow's **orientation**, and orientation shows up in the binary mask of "which region is shadow" only as boundary detail).

**Three countermeasures** (good material for "method limitations + solutions" when writing a paper):

1. Use IoU to pin down the elevation angle (reliable), and find a separate constraint for azimuth;
2. Use the **direction of the rover's own shadow** to constrain the azimuth (the single strongest geometric cue);
3. **Multi-frame joint inversion**: the solar azimuth changes monotonically with time (very slowly on the Moon), and using multiple times jointly can break the degeneracy.

### 6.4 Photoclinometry (Shape-from-Shading)

Infer surface slope from brightness gradients, used to **fill in the lack of elevation data in shadowed regions of the DEM**. On the Moon a Hapke-type light-scattering model is mandatory; for a multiphase implementation see [Multiphase photoclinometry as applied to the lunar photometry with LROC NAC data](https://www.sciencedirect.com/science/article/abs/pii/S0032063324000783) ✅ (provenance verified). A related pipeline can be seen in [Lunar surface DEM construction based on reflectance modeling](https://ui.adsabs.harvard.edu/abs/2014AdSpR..53.1735G/abstract) ✅.

---

## 7. Selection Table

| Your scenario | What to choose | Rationale |
|---|---|---|
| Computing a shadow mask for one sun angle, once | **Ray marching** `compute_illumination()` | 20 lines of core logic, good enough |
| Iterating over many sun angles/dates (computing observation windows, determining CIR/PSR) | **Horizon map** `horizon_map()` | Precompute once; thereafter every angle change is free |
| Large-area/real-time (onboard) | Horizon map + spherical-harmonic/Fourier compression, or GPU | See [HPG 2025](https://highperformancegraphics.org/untracked/2025/presentations/Pa5_2_Fast%20Planetary%20Shadows%20using%20Fourier-Compressed%20Horizon%20Maps.pdf) ✅ |
| Only imagery, no DEM | **Illumination-invariant features** or **rover self-calibration** | B2 / C2 |
| Need ground-truth labels | Ray marching + horizon map **corroborating each other** | Trustworthy only if the two agree |
| Need to fill in elevation in shadowed regions | **Photoclinometry + Hapke** | C4 |
| Need mission-level sun position | **SPICE** | A' 1, don't use the analytic method |

---

## 8. Six Pitfalls You Must Know (All Encountered, with Measured Data)

### 8.1 Azimuth Sign — Three of the Four Bugs

**Symptom**: shadows fall on the same side as the sun; or the two implementations never agree.

**Why it is hard to find**: at a glance "there are shadows and the shapes look plausible", so it is hard to spot by eye; and if you only test a single implementation, it will keep being self-consistently wrong.

**How to nail it down**: use an **analytic test case**. A north-south trending ridge (`dem[:, 100] = 10`), with the sun due west (az=270°):

- The **west side** of the ridge (note that the col 100–200 direction is the east side) should be occluded, with a horizon elevation angle = atan(10/50) = **11.31°**;
- The east side of the ridge should be unoccluded.

Measured: `Hmap[due-west bin, 100, 150] = 11.31°`, `Hmap[due-west bin, 100, 50] = 0.00°`, and the ray-marching shadow falls on col 101–200. **This one test beats a thousand lines of code review.**

### 8.2 The Circular Wraparound of `np.roll`

`np.roll` wraps around at the edges, producing spurious occlusion. Zero padding is mandatory (this module's `_shift_zero()`). **Sanity test: flat terrain must be 100% fully lit.** Measured shadow pixels = 0.

### 8.3 Using `sun_direction(az, 90°)` for a Horizontal Direction — Zero Vector, Silent Failure

`cos(90°) = 0` → the direction vector is zero → all offsets round to 0 → the loop body is skipped by `continue` → **the entire horizon map returns the initial value −90°**, and the code **reports no error at all**.

Measured symptom: the horizon map's global max = −90°, every pixel is judged "illuminated", so ray marching (80% shadow) does not match the horizon map (41% shadow).

**Lesson**: always extract the direction vector into its own function (`horizontal_direction()`); do not assemble the parameters at the call site.

### 8.4 A Hard AND Destroys Recall

See §5.2. Measured F1 drops from 1.000 to 0.383.

### 8.5 Azimuthal Degeneracy (Not a Bug, but a Property of the Method)

See §6.3. **You must acknowledge this in the paper**, otherwise a reviewer's question will expose it.

### 8.6 Shadow Masks Computed from a DEM Cannot Be Used Directly as Training Labels

Three reasons:

1. **Resolution mismatch**: LOLA is at best about 5 m/pixel, 1–2 orders of magnitude off from image scales, so small-scale shadows **simply cannot be rendered**;
2. **The shadowed region itself has no elevation measurement**: the DEM there is interpolated and **has the largest error** — it is least accurate exactly where accuracy is most needed;
3. **Registration error + boundary mismatch**: synthetic shadows and real shadows can never be aligned pixel by pixel.

**Therefore evaluation must use a tolerance-based F1** (dilating the prediction and the ground truth each by tol pixels). Measured effect of "global misalignment" on the score:

| Registration misalignment | F1 (tol=3 px) | IoU (no tolerance) |
|---|---|---|
| 0 px | 1.000 | 1.000 |
| 2 px | 1.000 | 0.903 |
| 4 px | 0.995 | 0.852 |
| 8 px | 0.976 | 0.806 |
| 12 px | 0.962 | 0.776 |

**Conclusion**: IoU without tolerance measures **mainly registration error, not detection performance**. With tolerance-based F1, an 8-pixel misalignment only costs 2.4 percentage points.

**Supporting evidence**: this module specifically verified that "tolerance does not award a high score to completely different masks" — the F1 of complementary masks (tol=1) = 0.111, safe.

---

## 9. Pointers to Off-the-Shelf Tools and Data

### 9.1 Tools

| Tool | Purpose | Notes |
|---|---|---|
| **SPICE / spiceypy** | Planetary/satellite ephemerides and reference-frame transformations | The only correct choice for mission-level sun position [documentation](https://readthedocs.org/projects/spiceypy/downloads/pdf/latest/) ✅ |
| **GDAL / WhiteboxTools / RichDEM** | General terrain analysis: hillshade, horizon angle, viewshed | `hillshade` includes cast shadow; WhiteboxTools has horizon/shadow tools |
| **HPG 2025 Fourier-compressed horizon maps** | Fast planet-scale shadows | [Paper](https://highperformancegraphics.org/untracked/2025/presentations/Pa5_2_Fast%20Planetary%20Shadows%20using%20Fourier-Compressed%20Horizon%20Maps.pdf) ✅ |
| **This module** `lunar_shadow.py` | Complete teaching-grade implementation of the above algorithms | Zero external data dependencies, reproducible |

> ⚠️ **A naming trap**: **`MoonShineR`** on CRAN is an **ecology** package for "predicting moonlight/sunlight/twilight ground illuminance" ([CRAN](https://cran.r-project.org/web/packages/MoonShineR/index.html) ✅), and **has nothing to do with lunar terrain illumination computation**. The classic tool for lunar illumination is called **MoonShine** (NASA/GSFC); do not confuse the two.

### 9.2 Data

| Data | Content | Use |
|---|---|---|
| **LOLA / LRO DEM** | Lunar digital elevation model; for an assessment of elevation error and its illumination impact at south pole landing sites see [Barker et al., P&SS 2021](https://doi.org/10.1016/j.pss.2021.105119) ⚠️ | Terrain base |
| **LROC NAC / WAC** | High-resolution imagery | Detector training data, shadow ground truth |
| **LuMon** | Lunar monocular depth estimation benchmark (CVPRW 2026 AI4Space) [PDF](https://www.openaccess.thecvf.com/content/CVPR2026W/AI4Space/papers/Sekmen_LuMon_A_Comprehensive_Benchmark_and_Development_Suite_with_Novel_Datasets_CVPRW_2026_paper.pdf) ✅ | Depth prior |
| **LunaPolaris** | Lunar polar stereo + point cloud + IMU [CORE PDF](https://core.ac.uk/download/722268884.pdf) ✅ | Localization/odometry |

### 9.3 Essential Reading for This Subfield

1. ✅ **Mazarico et al.** *Illumination conditions of the lunar polar regions using LOLA topography.* Icarus (2011) — **the foundational work**, a classic on lunar polar illumination computation. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0019103510004203)
2. ✅ *Intelligent Identification of Continuously Illuminated Regions at Lunar South Pole Based on Deep Learning.* Space: Science & Technology (2025) — the modern approach to identifying continuously illuminated regions (CIR). [Journal page](https://www.spacejournal.cn/sktcxb/en/article/doi/10.3724/j.issn.2096-9287.2025.20250044)
3. ✅ *New developments in modeling of illumination conditions, at the Moon and beyond.* LPSC 2018 — a review of modeling progress. [LPSC abstract](https://www.hou.usra.edu/meetings/lpsc2018/pdf/2801.pdf)
4. ✅ *Fast Planetary Shadows using Fourier-Compressed Horizon Maps.* HPG 2025 — a reference for the efficiency direction.
5. ⚠️ **Horne R.** *Hill shading and the reflectance map.* A series of works from 1988 onward — the classic algorithmic origin of terrain shading and cast shadow (the algorithmic basis of GIS `hillshade`).
6. ⚠️ **Hapke B.** *Theory of Reflectance and Emittance Spectroscopy.* — the theoretical basis for lunar photometric normalization (Lambertian cannot be used).

---

## 10. How to Use the Code

### 10.1 Files

| File | Content |
|---|---|
| [`lunar_shadow.py`](../lunar_shadow.py) | All algorithm implementations (only numpy required) |
| [`check_lunar_shadow.py`](../check_lunar_shadow.py) | Self-test script, **all 29 assertions pass** |
| [`make_figure.py`](../make_figure.py) | Generates the paper figure (matplotlib required) |
| [`lunar_shadow_figure.png`](../lunar_shadow_figure.png) | The generated 6-panel figure |

### 10.2 Running

```bash
# Self-test (recommended to run first, to confirm both the environment and the algorithms are correct)
python check_lunar_shadow.py

# Make the figure (matplotlib required)
python make_figure.py --n 192 --az 137 --el 6
```

> **Local environment note**: the self-test in this repository uses the Python bundled with the harness (includes numpy 2.3.5, no matplotlib);
> figure generation uses the workspace's `.venv-jupyter` (matplotlib 3.11.2). If matplotlib reports a cache-directory permission error,
> just set `MPLCONFIGDIR` to point to a writable directory inside the workspace:
>
> ```powershell
> $env:MPLCONFIGDIR='D:\lunar-shadow-algorithms\.mplcache'
> ```

### 10.3 Minimal Usage

```python
import numpy as np
import lunar_shadow as ls

# 1) Build terrain (or read your own DEM)
dem = ls.make_synthetic_lunar_dem(n=256, seed=42, relief=8.0)

# 2) Forward modeling: illumination mask with the sun at azimuth 137°, elevation 6°
illum = ls.compute_illumination(dem, az_deg=137.0, el_deg=6.0, cell=1.0)
shadow_frac = float((illum == 0).mean())

# 3) Need to iterate over many sun angles? Build the horizon map first, then look up each time
Hmap = ls.horizon_map(dem, cell=1.0, n_az=72)
for az in range(0, 360, 10):
    m = ls.illuminate_from_horizon(dem, Hmap, az, el_deg=6.0, cell=1.0)

# 4) Sun position (teaching-grade accuracy)
az, el = ls.solar_position_analytic(jd=2460545.0, lon_deg=0.0, lat_deg=-89.5)

# 5) Inversion loop: infer sun geometry from the shadow mask
res = ls.recover_sun_geometry(dem, shadow_mask, coarse_step=5.0,
                              az_step=2.0, el_step=0.5, tol=1)
print(res["az_deg"], res["el_deg"], res["score"])

# 6) Evaluation: must use tolerance-based F1
score = ls.f1_with_tolerance(pred_shadow, gt_shadow, tol=3)
```

### 10.4 Figure (`lunar_shadow_figure.png`)

![Lunar surface illumination and shadow algorithm results](../lunar_shadow_figure.png)

Six panels: (a) the synthetic DEM; (b) the **polar horizon-angle profile** of the central pixel (which makes it visually obvious in which azimuth the obstruction lies); (c)(d) the shadow masks from the two implementations and the disagreeing pixels (red); (e) the **inversion IoU heatmap** (white circle = ground truth, red star = optimum; the azimuth direction is visibly "flatter" — this is the degeneracy of §8.5); (f) the **tolerance curve**, arguing why tolerance-based F1 is mandatory.

---

## 11. References

> **Marker convention**: ✅ = title and provenance verified in this pass; ⚠️ = from search abstracts or prior knowledge, **verify against the original before citing**.

See the essential reading list in §9.3, plus:

- ⚠️ **Barker M.K. et al.** *Improved LOLA elevation maps for south pole landing sites: error estimates and their impact on illumination conditions.* Planetary and Space Science 203 (2021) 105119.
- ✅ **ESA ASTRA 2025** *Lunar Prospecting and Scouting Rover — CONOPS and GNC Architectures.* [PDF](https://roboshare.esa.int/ASTRA/Astra2025/2025Papers/127_Vangeffelen_Lunar-Prospecting-And-Scouting-Rover-Conops-And-Gnc-Architectures.pdf)
- ✅ **AGX Dynamics / OpenPLX lunar robotics simulation framework** (includes parameterized regolith terrain, non-ideal sensors, and an illumination and scenario designer) [arXiv:2509.12367](https://ar5iv.labs.arxiv.org/html/2509.12367)

---

## Appendix: Self-Test Results (Measured)

Summary of the actual output of `python check_lunar_shadow.py` (**all 29 PASS, 0 FAIL**):

| # | Check | Measured result |
|---|---|---|
| 1 | Direction vectors (due east/zenith/due north) | All three correct |
| 2 | Ray marching vs horizon map cross-consistency | Worst agreement rate 97.88%, largest shadow-fraction difference 0.87% |
| 2 | Analytic test case: ridge occlusion | West side 11.31° (analytic value 11.31°), east side 0.00° |
| 2 | Ray-marching shadow direction | Sun due west → shadow at col 101–200 (east side of the ridge) ✔ |
| 3 | No shadow on flat terrain | Shadow pixels = 0 |
| 4 | Sun geometry inversion loop | Δaz ≤ 1.0°, Δel = 0.0°, IoU ≥ 0.994 |
| 4 | Azimuthal degeneracy confirmation | Best–second-best gap only 0.002–0.009 |
| 5 | Analytic sun position | Sub-solar point latitude ±0.02°, consistent with the ±1.54° expectation |
| 6 | Tolerance F1 vs IoU | Under 3 px misalignment, F1 0.964 vs IoU 0.856 |
| 7 | Shadow length ↔ height inverse | Error < 1e-6 over 9 parameter sets |
| 7 | Polar low-sun-angle magnitude | 1 m height → 38 m shadow → 1.507° |
| 8 | Reproducibility | Byte-for-byte identical with the same seed |
| 9 | Image shadow extraction | F1(tol=2) = 1.000 |
| 9 | Illumination-invariant features | Output range [0.030, 1.000] |

**The self-test script is itself a deliverable**: it is reproducible evidence of "algorithmic correctness" and ready-made material for a paper appendix.
