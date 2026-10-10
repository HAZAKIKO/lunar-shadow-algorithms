# -*- coding: utf-8 -*-
"""
check_lunar_shadow.py —— self-check script for lunar_shadow.py
==================================================

Run: python check_lunar_shadow.py

Verification, item by item (no external data required):
  1. Correctness of the utility functions
  2. Ray marching vs horizon map: cross-consistency of two independent implementations
  3. Ray marching: zero padding has no wrap-around bug (border pixels are not wrongly occluded)
  4. Closed-loop solar-position inversion: known (az, el) -> shadow -> inversion -> error
  5. Analytic solar position: sub-solar point range sanity + polar solar-elevation magnitude
  6. How tolerance-based F1 differs from per-pixel IoU (why tolerance is mandatory)
  7. Reciprocity between shadow length <-> object height
  8. Reproducibility: the same seed generates an identical DEM twice

If everything passes, it prints PASS.
"""

from __future__ import annotations

import sys
import numpy as np

from lunar_shadow import (
    compute_illumination,
    dem_to_lonlat,
    estimate_sun_elevation_from_shadow,
    f1_with_tolerance,
    horizon_map,
    illuminate_from_horizon,
    image_shadow_mask,
    illumination_invariant,
    iou,
    make_synthetic_lunar_dem,
    recover_sun_geometry,
    shadow_length_to_height,
    solar_position_analytic,
    subsolar_lonlat_analytic,
    sun_direction,
)

PASS, FAIL = 0, 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "[PASS]" if ok else "[FAIL]"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"{mark} {name}" + (f"  |  {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("lunar_shadow self-check")
    print("=" * 78)

    # ---------------------------------------------------------------- 1
    print("\n[1] Utility-function correctness")
    dcol, drow, dz = sun_direction(90.0, 0.0)      # sun due east, on the horizon
    report("due-east direction vector", abs(dcol - 1) < 1e-9 and abs(drow) < 1e-9 and abs(dz) < 1e-9,
           f"(dcol,drow,dz)=({dcol:.3f},{drow:.3f},{dz:.3f})")
    dcol, drow, dz = sun_direction(0.0, 90.0)      # sun at the zenith
    report("zenith direction vector", abs(dz - 1) < 1e-9, f"dz={dz:.3f}")
    dcol, drow, dz = sun_direction(0.0, 0.0)       # sun due north
    report("due-north direction vector (drow should be negative)", drow < -0.99, f"drow={drow:.3f}")

    dem = make_synthetic_lunar_dem(n=192, seed=7)
    report("DEM shape and magnitude", dem.shape == (192, 192) and 1.0 < dem.std() < 50.0,
           f"std={dem.std():.2f} m")

    # ---------------------------------------------------------------- 2
    print("\n[2] Ray marching vs horizon map: cross-consistency")
    dem = make_synthetic_lunar_dem(n=160, seed=20261010)
    # ⚠️ We must include azimuths that are **not bin-aligned** (e.g. 137°); otherwise the
    #    72 bins (5° steps) land exactly on bin centres, interpolation becomes the identity
    #    operation, and the effect of resolution cannot be measured.
    cases = [(45.0, 8.0), (137.0, 5.0), (270.0, 20.0), (313.0, 3.0)]

    def cross_check(n_az: int) -> tuple[float, float]:
        Hmap = horizon_map(dem, cell=1.0, n_az=n_az)
        worst_agree, worst_frac = 1.0, 0.0
        for az, el in cases:
            m1 = compute_illumination(dem, az, el, cell=1.0)
            m2 = illuminate_from_horizon(dem, Hmap, az, el, cell=1.0)
            worst_agree = min(worst_agree, float((m1 == m2).mean()))
            worst_frac = max(worst_frac,
                             abs(float((m1 == 0).mean()) - float((m2 == 0).mean())))
        return worst_agree, worst_frac

    agree72, frac72 = cross_check(72)
    agree360, frac360 = cross_check(360)
    report("Horizon-map shape", horizon_map(dem, n_az=72).shape == (72, 160, 160))
    report("Both implementations give close shadow area fractions (<2%)", frac72 < 0.02,
           f"max fraction difference={frac72*100:.2f}%")
    report("Pixel agreement between the two implementations > 95% (n_az=72, 5° bins)", agree72 > 0.95,
           f"worst agreement={agree72*100:.2f}%")
    report("Higher azimuth resolution further improves agreement (n_az=360)", agree360 >= agree72,
           f"{agree72*100:.2f}% -> {agree360*100:.2f}%")
    print("       Note: the horizon map's precision loss comes mainly from **azimuth binning + "
          "linear interpolation**, and it only shows up when the azimuth does not fall on a "
          "bin centre.")
    print("           Engineering advice: use the horizon map for fast sweeps / coarse screening, "
          "and re-verify critical cases with ray marching.")

    # Analytic case: north–south ridge + sun due east/west (this is what pins down the azimuth convention)
    ridge = np.zeros((201, 201))
    ridge[:, 100] = 10.0
    Hr = horizon_map(ridge, cell=1.0, n_az=72)
    ang = float(np.rad2deg(np.arctan2(10.0, 50.0)))
    m_w = int(round(270.0 / 5.0)) % 72      # sun due west
    ok = (abs(Hr[m_w, 100, 150] - ang) < 0.1 and Hr[m_w, 100, 50] < 0.1)
    report("Analytic case: west of the ridge is occluded, east of it is not (10/50 -> 11.31°)", ok,
           f"Hmap west={Hr[m_w,100,150]:.2f}° east={Hr[m_w,100,50]:.2f}° analytic={ang:.2f}°")
    sh_w = compute_illumination(ridge, 270.0, 5.0, cell=1.0) == 0
    cols = np.nonzero(sh_w.any(axis=0))[0]
    report("Ray marching: with the sun due west the shadow falls east of the ridge",
           len(cols) > 0 and cols.min() > 100,
           f"shadow column range [{cols.min()}, {cols.max()}], ridge at col=100")

    # ---------------------------------------------------------------- 3
    print("\n[3] Zero padding: no phantom occlusion at the borders")
    # On flat terrain no solar elevation angle should produce a shadow
    # 2) Flat ground produces no shadow: this is the sanity check on the predicate's direction
    flat = np.zeros((64, 64))
    m = compute_illumination(flat, az_deg=45.0, el_deg=5.0, cell=1.0)
    report("Flat terrain has no shadow (no wrap-around bug, predicate direction correct)", (m == 1).all(),
           f"shadow pixels={int((m == 0).sum())}")
    # Single peak: the shadow should appear on the side opposite the Sun
    peak = np.zeros((64, 64))
    peak[32, 32] = 20.0
    m = compute_illumination(peak, az_deg=90.0, el_deg=5.0, cell=1.0)   # sun due east
    shadow = (m == 0)
    ys, xs = np.nonzero(shadow)
    if len(xs):
        report("Single-peak shadow falls opposite the Sun (west side)",
               float(xs.mean()) < 32.0 and float(xs.max()) < 32,
               f"shadow {len(xs)} px, centroid col={xs.mean():.1f}, col range [{xs.min()},{xs.max()}]")
    else:
        report("Single-peak shadow falls opposite the Sun (west side)", False, "no shadow produced")

    # ---------------------------------------------------------------- 4
    print("\n[4] Closed-loop solar-position inversion (the core self-check)")
    dem = make_synthetic_lunar_dem(n=160, seed=1234, relief=8.0)
    for az_true, el_true in [(60.0, 6.0), (225.0, 12.0)]:
        gt = (compute_illumination(dem, az_true, el_true, cell=1.0) == 0)
        res = recover_sun_geometry(
            dem, gt,
            az_range=(0, 360), el_range=(1, 40),
            az_step=2.0, el_step=0.5, coarse_step=5.0, tol=1, metric="iou",
        )
        d_az = min(abs(res["az_deg"] - az_true), 360 - abs(res["az_deg"] - az_true))
        d_el = abs(res["el_deg"] - el_true)
        # Azimuth degeneracy: peek at the runner-up azimuth to assess "identifiability"
        z_az = res["z_az"]
        order = np.argsort(z_az)[::-1]
        azs = res["table"]["az"]
        second = float(z_az[order[1]]) if len(order) > 1 else 0.0
        margin = res["score"] - second
        print(f"       truth az={az_true:5.1f}° el={el_true:4.1f}°  ->  "
              f"recovered az={res['az_deg']:5.1f}° el={res['el_deg']:4.1f}°  "
              f"IoU={res['score']:.3f}  Δaz={d_az:.1f}° Δel={d_el:.1f}°")
        print(f"             runner-up azimuth az={azs[order[1]]:5.1f}° (IoU={second:.3f}), "
              f"best-to-runner-up margin={margin:.4f}")
        report(f"Recovery is usable (truth az={az_true:.0f}°)",
               d_az <= 8.0 and d_el <= 2.5 and res["score"] > 0.80,
               f"Δaz={d_az:.1f}° Δel={d_el:.1f}° IoU={res['score']:.3f}")
        report(f"Azimuth is degenerate (a single shadow mask is weakly sensitive to az)",
               margin < 0.05,
               f"best-to-runner-up margin is only {margin:.4f}")
    print("       Note: this is an **inherent identifiability problem of the method**, not a bug: "
          "the shape of a cast shadow")
    print("           is governed mainly by the solar elevation angle and is far less sensitive "
          "to azimuth.")
    print("           Countermeasures: lock the geometry with elevation, constrain azimuth with the "
          "rover's own shadow direction, or invert jointly over several frames.")

    # ---------------------------------------------------------------- 5
    print("\n[5] Analytic solar position (accuracy self-check of the teaching model)")
    for jd in [2451545.0, 2451545.0 + 90, 2451545.0 + 9000]:
        lon_s, lat_s = subsolar_lonlat_analytic(jd)
        ok = -180 <= lon_s <= 180 and -1.6 <= lat_s <= 1.6
        report(f"jd={jd:.0f} sub-solar point", ok, f"lon={lon_s:7.2f}° lat={lat_s:5.2f}°"
               "  (lunar obliquity about ±1.54°)")
    # The solar elevation angle near the south pole should be of a very low magnitude
    jd = 2451545.0 + 9000
    az, el = solar_position_analytic(jd, lon_deg=0.0, lat_deg=-89.5)
    report("Solar elevation near the south pole is in the low-angle range", -2.0 <= el <= 3.0,
           f"az={az:.1f}° el={el:.2f}°")
    lon, lat = dem_to_lonlat(np.array([0, 100]), np.array([0, 100]), 256,
                             west=-10.0, north=-88.0, res_deg=0.01)
    report("Row/col -> lon/lat projection", abs(lon[1] - (-10.0 + 1.0)) < 1e-9
           and abs(lat[1] - (-88.0 - 1.0)) < 1e-9, f"lon={lon[1]:.3f} lat={lat[1]:.3f}")

    # ---------------------------------------------------------------- 6
    print("\n[6] Tolerance-based F1 vs per-pixel IoU (choosing the evaluation)")
    gt = (compute_illumination(dem, 60.0, 6.0, cell=1.0) == 0)
    # Simulate a real registration error between DEM and image with a "global 3-pixel offset"
    from lunar_shadow import _shift_zero as _sz
    pred = _sz(gt.astype(np.uint8), 0, 3, 0).astype(bool)
    i0 = iou(pred, gt)
    f0 = f1_with_tolerance(pred, gt, tol=0)
    f1 = f1_with_tolerance(pred, gt, tol=1)
    f3 = f1_with_tolerance(pred, gt, tol=3)
    print(f"       global 3 px offset: IoU={i0:.4f}  F1(tol=0)={f0:.4f}  "
          f"F1(tol=1)={f1:.4f}  F1(tol=3)={f3:.4f}")
    report("Tolerance makes the evaluation insensitive to registration error", f3 > f1 > f0,
           f"F1: {f0:.3f} -> {f1:.3f} -> {f3:.3f}")
    report("Tolerance does not score completely different masks highly",
           f1_with_tolerance(~gt, gt, tol=1) < 0.35,
           f"complementary-mask F1={f1_with_tolerance(~gt, gt, tol=1):.3f}")

    # ---------------------------------------------------------------- 7
    print("\n[7] Shadow length <-> object height reciprocity")
    for h in [0.5, 2.0, 10.0]:
        for el in [2.0, 15.0, 45.0]:
            L_m = h / np.tan(np.deg2rad(el))
            L_px = L_m / 0.01                       # GSD = 1 cm/px
            el_hat = estimate_sun_elevation_from_shadow(h, L_px, 0.01)
            h_hat = shadow_length_to_height(L_px, 0.01, el)
            if abs(el_hat - el) > 1e-6 or abs(h_hat - h) > 1e-6:
                report(f"h={h} el={el}", False, f"el_hat={el_hat:.6f} h_hat={h_hat:.6f}")
                break
    else:
        report("Reciprocity (9 parameter sets)", True, "error < 1e-6")
    el_low = estimate_sun_elevation_from_shadow(1.0, 3800.0, 0.01)   # 38 m shadow
    report("Polar low-sun-angle magnitude is correct (1 m tall -> ~38 m shadow -> ~1.5°)",
           abs(el_low - 1.508) < 0.1, f"el={el_low:.3f}°")

    # ---------------------------------------------------------------- 8
    print("\n[8] Reproducibility")
    a = make_synthetic_lunar_dem(n=128, seed=42)
    b = make_synthetic_lunar_dem(n=128, seed=42)
    c = make_synthetic_lunar_dem(n=128, seed=43)
    report("Same seed is byte-for-byte identical", np.array_equal(a, b))
    report("Different seeds really do differ", not np.array_equal(a, c),
           f"differing-pixel fraction={float((a != c).mean())*100:.1f}%")

    # ---------------------------------------------------------------- 9
    print("\n[9] Image shadow extraction (self-check on a synthetic image)")
    # Render an "image" from the DEM: low brightness inside shadow, medium brightness elsewhere
    illum = compute_illumination(dem, 60.0, 6.0, cell=1.0)
    img = np.where(illum > 0, 0.6, 0.05).astype(np.float64)
    gt_shadow = (illum == 0)
    pred_shadow = image_shadow_mask(img, brightness_thr=0.3, saturation_thr=1.0,
                                    edge_thr=0.05, morph=0).astype(bool)
    f1 = f1_with_tolerance(pred_shadow, gt_shadow, tol=2)
    report("Threshold-based shadow extraction F1 > 0.9", f1 > 0.9, f"F1(tol=2)={f1:.4f}")
    inv = illumination_invariant(img, sigma=3)
    report("Illumination-invariant feature output lies in [0,1] and is not constant",
           inv.min() >= 0 and inv.max() <= 1.0 and inv.std() > 1e-4,
           f"range=[{inv.min():.3f},{inv.max():.3f}] std={inv.std():.4f}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
