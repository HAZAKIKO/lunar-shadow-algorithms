# -*- coding: utf-8 -*-
"""
make_figures.py —— Figures for the illumination-generalisation experiment
========================================================================

Two figures:

  fig1_pairs.png    the SAME terrain rendered at four sun elevations, with the
                    corresponding shadow masks beneath. This is the core visual:
                    only the lighting changed.

  fig2_matrices.png three heatmaps, one per model family, of
                    train-elevation x test-elevation F1. Reading them side by
                    side is the whole argument of the experiment.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataset as ds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def fig_pairs() -> None:
    # rebuild one terrain deterministically, then render it at several elevations
    cfg = ds.CFG
    dem = ds.build_terrain(cfg["seed0"] + 5000, cfg)
    rng_a = np.random.default_rng(cfg["seed0"] + 5000 + 77)
    albedo = ds.make_albedo(cfg["size"], rng_a, *cfg["albedo_range"], cfg["albedo_scale"])

    elevations = [4.0, 8.0, 20.0, 40.0]
    az = cfg["sweep_azimuth"]

    fig, axes = plt.subplots(3, len(elevations), figsize=(4.2 * len(elevations), 11))
    fig.suptitle(
        "Same terrain, same azimuth (%.0f°), four sun elevations — only the lighting changed"
        % az, fontsize=14)

    for j, el in enumerate(elevations):
        rng = np.random.default_rng(int(el * 1000) + 7)
        s = ds.make_sample(dem, albedo, az, el, cfg["cell"], cfg["ambient"],
                           cfg["noise_sigma"], rng,
                           gain_range=cfg["gain_range"],
                           interreflection_range=cfg["interreflection_range"])
        img = s["image"]
        sh = s["shadow"].astype(bool)

        ax = axes[0, j]
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"sun elevation {el:.0f}°\nrendered image (gain {s['gain']:.2f})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, j]
        ax.imshow(sh, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"shadow mask\n{sh.mean() * 100:.1f}% shadowed", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2, j]
        b_sh, b_lit = img[sh], img[~sh]
        ax.hist(b_sh, bins=60, range=(0, 1), color="tab:red", alpha=0.75, label="shadow")
        ax.hist(b_lit, bins=60, range=(0, 1), color="tab:blue", alpha=0.55, label="lit")
        ax.set_yscale("log")
        ax.set_xlabel("brightness")
        ax.set_title("brightness distribution", fontsize=11)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "fig1_pairs.png")
    fig.savefig(out, dpi=140)
    print("saved", out)


def fig_matrices() -> None:
    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    m = res["matrix"]
    train_els = m["train_elevations"]
    sweep_els = m["sweep_elevations"]

    panels = [
        ("B0  global brightness threshold", m["B0_f1"]),
        ("B0rel  local-contrast (gain-invariant) threshold", m["B0rel_f1"]),
        ("B1  per-pixel logistic regression (4 features)", m["B1_f1"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for ax, (title, mat) in zip(axes, panels):
        arr = np.array(mat)
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0.3, vmax=1.0)
        ax.set_xticks(range(len(sweep_els)))
        ax.set_xticklabels([f"{e:.0f}°" for e in sweep_els])
        ax.set_yticks(range(len(train_els)))
        ax.set_yticklabels([f"{e:.0f}°" for e in train_els])
        ax.set_xlabel("test sun elevation")
        ax.set_ylabel("train sun elevation")
        ax.set_title(title, fontsize=11)
        # annotate
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                matched = abs(train_els[i] - sweep_els[j]) < 1e-6
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if arr[i, j] < 0.7 else "black",
                        fontweight="bold" if matched else "normal")
        # mark the matched-lighting diagonal cells
        for i, e_tr in enumerate(train_els):
            if e_tr in sweep_els:
                j = sweep_els.index(e_tr)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="red", lw=2))
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("F1 by train elevation (rows) x test elevation (columns) — "
                 "red boxes mark matched lighting", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(HERE, "fig2_matrices.png")
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    fig_pairs()
    fig_matrices()
