# -*- coding: utf-8 -*-
"""
dataset.py —— Lunar paired-illumination dataset generator
=========================================================

Renders the SAME terrain under MANY sun geometries, producing paired
(grayscale image, shadow mask) samples. This is the data layer for the
illumination-generalisation experiment: a model is trained at some sun
elevations and tested at others, and the resulting matrix shows how badly
perception degrades when the lighting changes.

Why pairs matter
----------------
On the Moon there is no atmospheric scattering, so a shadowed pixel receives
(almost) no direct sunlight and is essentially black, while a lit pixel varies
smoothly with the local incidence angle. That gives a very clean generative
model for the image:

    I = clip( albedo * max(cos(incidence), 0) + ambient * albedo + noise )

and the shadow mask is exactly the set of pixels with cos(incidence) == 0.
Because we render from a DEM we KNOW the mask, so labels are free and exact --
which is precisely why this synthetic setup is useful for a course project:
no annotation cost, and the ground truth is not a noisy orbital-derived guess.

The catch (documented deliberately): the resulting pairs are synthetic. A real
system must additionally handle real sensor noise, albedo-dominated contrast,
and DEM/image registration error.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Import the repository's own module rather than keeping a copy here: a duplicated
# lunar_shadow.py would silently diverge from the shipped one.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import lunar_shadow as ls

# ----------------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------------
CFG = {
    # terrain
    "size": 128,
    "cell": 1.0,
    "relief": 6.0,
    # sun geometries are expressed as elevation bands (degrees)
    "train_elevations": [8.0, 15.0, 25.0, 40.0],
    "test_elevations": [4.0, 12.0, 20.0, 30.0],
    # azimuths sampled per elevation (training only; test uses a fixed set)
    "train_azimuths": [0.0, 90.0, 180.0, 270.0],
    "test_azimuths": [45.0, 135.0, 225.0, 315.0],
    # the sweep used to build the full train-elevation x test-elevation matrix
    "sweep_elevations": [4.0, 6.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    "sweep_azimuth": 135.0,
    # rendering
    #
    # These two numbers decide whether the whole experiment is interesting, so
    # they are set to the physics rather than to convenience:
    #
    #   ambient = 0.005   the Moon has no atmospheric scattering, so the only
    #                     light reaching a shadowed pixel is secondary scattering
    #                     off nearby terrain. It is very dark -- which means the
    #                     SIGNAL for "shadow" is weak.
    #   albedo_lo = 0.13  real lunar materials span roughly 0.1 (dark mare, glassy
    #                     pyroclastics) to 1.0 (fresh anorthositic highlands).
    #
    # Together these create the ambiguity that makes shadow detection a real
    # problem: an ordinary-albedo pixel in shadow (0.5*0.005 = 0.0025) and a dark
    # pixel in full sunlight (0.13*~1.0 = 0.13) are separated, but a dark pixel at
    # grazing incidence is genuinely darker than a shadowed pixel of average
    # albedo. "Dark" alone is therefore NOT a valid shadow cue, and a detector
    # that treats it as one will fail -- differently at different sun angles.
    "ambient": 0.005,
    "noise_sigma": 0.012,
    "albedo_range": (0.13, 1.00),
    "albedo_scale": 18.0,     # correlation length of the albedo field (pixels)
    #
    # Camera gain / exposure uncertainty.
    #
    # This is the mechanism that turns "shadow segmentation" from a solved
    # thresholding exercise into a real problem. Once a global unknown scale
    # multiplies the image, the absolute brightness of a shadow is no longer a
    # constant, so ANY fixed threshold is wrong by that factor. A detector must
    # then rely on RELATIVE (illumination-invariant) structure instead.
    #
    # Physically this stands for: exposure settings, camera radiometric
    # calibration drift, dust on the optics, and the fact that a rover rarely
    # carries an absolute radiance reference.
    "gain_range": (0.7, 1.4),
    #
    # Secondary scattering off nearby terrain, which is the only light a shadowed
    # pixel receives on an airless body. Modelled as a smooth spatial field so
    # that "how dark is dark" also varies across the scene.
    "interreflection_range": (0.5, 3.0),   # multiplier on `ambient`
    # counts
    "n_train_terrains": 10,
    "n_test_terrains": 8,
    "seed0": 1000,
}


# ----------------------------------------------------------------------------
# Image formation model
# ----------------------------------------------------------------------------
def render_image(dem: np.ndarray, albedo: np.ndarray, az_deg: float, el_deg: float,
                 cell: float, ambient: float, noise_sigma: float,
                 rng: np.random.Generator | None = None,
                 illum: np.ndarray | None = None,
                 interreflection_range: tuple[float, float] = (1.0, 1.0),
                 gain: float = 1.0) -> np.ndarray:
    """Render a grayscale image of the terrain under the given sun geometry.

    Returns a float32 array in [0, 1].

    The image formation model must include the CAST shadow, not just the local
    surface orientation:

        I = gain * albedo * ( illum * max(cos_incidence, 0) + ambient_field ) + noise

    Two parts of this are easy to get wrong, and getting either wrong silently
    destroys the whole experiment:

    1. The CAST shadow must be in the image, not just the local surface
       orientation. A pixel can face the sun (cos_incidence > 0) and still be
       dark because a distant ridge blocks the sunlight. If the renderer only
       used cos_incidence, the image and the shadow label would disagree over a
       large fraction of pixels and every model would look bad for the wrong
       reason.

    2. `gain` is a global unknown scale. Without it, "shadow = dark" is exactly
       true and the task degenerates into a single fixed threshold, with nothing
       to generalise across sun angles.
    """
    dem = np.asarray(dem, dtype=np.float64)
    gx, gy, _ = ls.terrain_gradient(dem, cell)
    dcol, drow, dz = ls.sun_direction(az_deg, el_deg)

    # local surface orientation: cos(incidence) with the to-sun vector
    cos_inc = (-gx * dcol - gy * drow + dz) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)

    if illum is None:
        illum = ls.compute_illumination(dem, az_deg, el_deg, cell=cell, use_slope_term=True)
    illum = np.asarray(illum, dtype=np.float64)

    # spatially varying ambient (secondary scattering); `gain` was drawn by the caller
    n = dem.shape[0]
    if rng is not None and interreflection_range != (1.0, 1.0):
        k = max(3, n // 24)
        raw = ls._bilinear_upsample(rng.random((k, k)), n)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        lo, hi = interreflection_range
        amb_field = ambient * (lo + (hi - lo) * raw)
    else:
        amb_field = np.full(dem.shape, ambient, dtype=np.float64)

    shading = illum * np.maximum(cos_inc, 0.0) + amb_field
    img = gain * albedo * shading
    if noise_sigma > 0 and rng is not None:
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_albedo(n: int, rng: np.random.Generator, lo: float, hi: float,
                scale: float) -> np.ndarray:
    """Smooth random albedo field in [lo, hi].

    A realistic touch that also matters: albedo variation is a *confounder* for
    shadow detection, because dark rock looks like shadow.
    """
    k = max(4, int(n / scale) + 2)
    coarse = rng.random((k, k))
    up = ls._bilinear_upsample(coarse, n)
    up = (up - up.min()) / (up.max() - up.min() + 1e-9)
    return (lo + (hi - lo) * up).astype(np.float64)


def make_sample(dem: np.ndarray, albedo: np.ndarray, az_deg: float, el_deg: float,
                cell: float, ambient: float, noise_sigma: float,
                rng: np.random.Generator,
                gain_range: tuple[float, float] = (1.0, 1.0),
                interreflection_range: tuple[float, float] = (1.0, 1.0)) -> dict:
    """Render one (image, shadow mask) pair.

    The illumination mask is computed once and handed to the renderer, so the
    image and the label are guaranteed to come from the SAME physical model.
    """
    illum = ls.compute_illumination(dem, az_deg, el_deg, cell=cell, use_slope_term=True)
    shadow = (illum == 0)
    gain = float(rng.uniform(*gain_range))
    img = render_image(dem, albedo, az_deg, el_deg, cell, ambient, noise_sigma, rng,
                       illum=illum, interreflection_range=interreflection_range,
                       gain=gain)
    return {
        "image": img,
        "shadow": shadow.astype(np.uint8),
        "dem": dem.astype(np.float32),
        "az": float(az_deg),
        "el": float(el_deg),
        "gain": gain,
        "shadow_fraction": float(shadow.mean()),
    }


# ----------------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------------
def build_terrain(seed: int, cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ls.make_synthetic_lunar_dem(
        n=cfg["size"], cell=cfg["cell"], seed=seed,
        n_craters=int(rng.integers(10, 18)), relief=cfg["relief"],
    )


def build_split(cfg: dict, split: str, verbose: bool = True) -> list[dict]:
    """Build the train or test split.

    Terrains are disjoint between splits AND sun elevations come from disjoint
    bands, so a model cannot win by memorising the lighting.
    """
    if split == "train":
        n_terr, eels, azs = cfg["n_train_terrains"], cfg["train_elevations"], cfg["train_azimuths"]
    elif split == "test":
        n_terr, eels, azs = cfg["n_test_terrains"], cfg["test_elevations"], cfg["test_azimuths"]
    else:
        raise ValueError(split)

    base = cfg["seed0"] + (0 if split == "train" else 5000)
    samples: list[dict] = []
    for t in range(n_terr):
        tseed = base + t
        dem = build_terrain(tseed, cfg)
        rng_a = np.random.default_rng(tseed + 77)
        albedo = make_albedo(cfg["size"], rng_a, *cfg["albedo_range"], cfg["albedo_scale"])
        for el in eels:
            for az in azs:
                rng = np.random.default_rng(tseed * 1000 + int(el * 10) + int(az))
                s = make_sample(dem, albedo, az, el, cfg["cell"], cfg["ambient"],
                                cfg["noise_sigma"], rng,
                                gain_range=cfg["gain_range"],
                                interreflection_range=cfg["interreflection_range"])
                s["terrain_id"] = tseed
                samples.append(s)
        if verbose:
            print(f"  [{split}] terrain {t + 1}/{n_terr} (seed {tseed}) -> "
                  f"{len(eels) * len(azs)} samples", flush=True)
    return samples


def build_sweep(cfg: dict, verbose: bool = True) -> dict[float, list[dict]]:
    """Render the SAME test terrains at every elevation in the sweep.

    This is what makes the train-elevation x test-elevation matrix meaningful:
    for a fixed terrain and fixed azimuth, only the sun elevation changes.
    """
    out: dict[float, list[dict]] = {}
    base = cfg["seed0"] + 5000
    terrains = []
    for t in range(cfg["n_test_terrains"]):
        tseed = base + t
        dem = build_terrain(tseed, cfg)
        rng_a = np.random.default_rng(tseed + 77)
        albedo = make_albedo(cfg["size"], rng_a, *cfg["albedo_range"], cfg["albedo_scale"])
        terrains.append((tseed, dem, albedo))

    for el in cfg["sweep_elevations"]:
        rows = []
        for tseed, dem, albedo in terrains:
            rng = np.random.default_rng(tseed * 1000 + int(el * 10) + int(cfg["sweep_azimuth"]))
            s = make_sample(dem, albedo, cfg["sweep_azimuth"], el, cfg["cell"],
                            cfg["ambient"], cfg["noise_sigma"], rng,
                            gain_range=cfg["gain_range"],
                            interreflection_range=cfg["interreflection_range"])
            s["terrain_id"] = tseed
            rows.append(s)
        out[float(el)] = rows
        if verbose:
            print(f"  [sweep] el={el:5.1f}° -> {len(rows)} samples, "
                  f"mean shadow fraction {np.mean([r['shadow_fraction'] for r in rows]):.3f}",
                  flush=True)
    return out


def save_npz(path: str, samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        image=np.stack([s["image"] for s in samples]),
        shadow=np.stack([s["shadow"] for s in samples]),
        az=np.array([s["az"] for s in samples], dtype=np.float32),
        el=np.array([s["el"] for s in samples], dtype=np.float32),
        gain=np.array([s.get("gain", 1.0) for s in samples], dtype=np.float32),
        terrain_id=np.array([s["terrain_id"] for s in samples], dtype=np.int64),
    )
    print(f"  saved {len(samples)} samples -> {path}")


def albedo_overlap_report(cfg: dict, samples: list[dict], verbose: bool = True) -> dict:
    """Quantify how much the shadow and non-shadow brightness distributions overlap.

    This number is the reason the experiment exists. If the two distributions are
    cleanly separated, "shadow detection" degenerates into a fixed threshold and
    there is nothing to generalise. The overlap fraction below is reported
    alongside every result so the difficulty of the task is explicit.
    """
    b_sh = np.concatenate([s["image"][s["shadow"] == 1] for s in samples])
    b_lit = np.concatenate([s["image"][s["shadow"] == 0] for s in samples])
    # a lit pixel is "confusable" if it is darker than the 95th percentile of shadowed pixels
    thr = float(np.percentile(b_sh, 95))
    confusable = float((b_lit < thr).mean())
    out = {
        "shadow_brightness_p50": float(np.percentile(b_sh, 50)),
        "shadow_brightness_p95": float(np.percentile(b_sh, 95)),
        "lit_brightness_p05": float(np.percentile(b_lit, 5)),
        "lit_brightness_p50": float(np.percentile(b_lit, 50)),
        "confusable_lit_fraction": confusable,
    }
    if verbose:
        print(f"    shadow brightness p50={out['shadow_brightness_p50']:.4f} "
              f"p95={out['shadow_brightness_p95']:.4f}")
        print(f"    lit    brightness p05={out['lit_brightness_p05']:.4f} "
              f"p50={out['lit_brightness_p50']:.4f}")
        print(f"    lit pixels darker than the shadow p95: {out['confusable_lit_fraction']*100:.1f}% "
              f"(this is the ambiguity the model must resolve)")
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "data")
    os.makedirs(out, exist_ok=True)
    print("Lunar paired-illumination dataset")
    print(f"  terrain {CFG['size']}x{CFG['size']}, cell {CFG['cell']} m, relief {CFG['relief']} m")
    print(f"  train elevations {CFG['train_elevations']}")
    print(f"  test  elevations {CFG['test_elevations']}  (disjoint from train)")
    print(f"  sweep elevations {CFG['sweep_elevations']}\n")

    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    tr = build_split(CFG, "train")
    te = build_split(CFG, "test")
    sw = build_sweep(CFG)

    save_npz(os.path.join(out, "train.npz"), tr)
    save_npz(os.path.join(out, "test.npz"), te)
    for el, rows in sw.items():
        save_npz(os.path.join(out, f"sweep_el{el:04.1f}.npz"), rows)

    print("\nDataset summary")
    for name, rows in (("train", tr), ("test", te)):
        els = sorted({r["el"] for r in rows})
        print(f"  {name}: {len(rows)} samples, elevations {els}, "
              f"mean shadow fraction {np.mean([r['shadow_fraction'] for r in rows]):.3f}")
    print("  sweep shadow fraction by elevation:")
    for el in sorted(sw):
        print(f"    el={el:5.1f}°  shadow {np.mean([r['shadow_fraction'] for r in sw[el]]):.3f}")

    print("\nBrightness separability (train split):")
    overlap = albedo_overlap_report(CFG, tr)
    with open(os.path.join(out, "separability.json"), "w", encoding="utf-8") as f:
        json.dump(overlap, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
