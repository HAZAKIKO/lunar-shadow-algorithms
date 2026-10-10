# -*- coding: utf-8 -*-
"""
baselines.py —— Shadow-segmentation baselines and a numpy logistic-regression model
==================================================================================

Three families, chosen so that the illumination-generalisation question is
isolated from model capacity:

  B0  GlobalThreshold   a single global brightness threshold, fitted on the
                        training split. This is the classic "shadow = dark pixel"
                        detector, and it is the baseline that fails most
                        spectacularly under illumination shift.

  B1  LogRegPixel       per-pixel logistic regression over 4 local features
                        (brightness, local mean, local std, illumination-invariant
                        normalised brightness). Trained by full-batch gradient
                        descent. Small enough to read, strong enough to be a
                        fair comparison.

  B2  DEMOracle         computes the shadow mask from the DEM and the TRUE sun
                        geometry (an upper bound, not a learnable model). This is
                        the ceiling the learned models are trying to reach.

Deliberately no PyTorch / scikit-learn: the whole experiment then runs on
CPU-only numpy, and every number is reproducible from the source.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import lunar_shadow as ls


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def _neighbourhood(img: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std over a square window, via zero-padded shifts."""
    acc = np.zeros_like(img, dtype=np.float64)
    acc2 = np.zeros_like(img, dtype=np.float64)
    cnt = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            shifted = ls._shift_zero(img, dr, dc, 0.0)
            acc += shifted
            acc2 += shifted * shifted
            cnt += 1
    mean = acc / cnt
    var = np.maximum(acc2 / cnt - mean * mean, 0.0)
    return mean, np.sqrt(var)


FEATURE_NAMES = [
    "brightness",
    "local_mean",
    "local_std",
    "invariant_brightness",
]


def pixel_features(img: np.ndarray, radius: int = 4) -> np.ndarray:
    """Build the (H, W, 4) feature stack for one image."""
    img = np.asarray(img, dtype=np.float64)
    mean, std = _neighbourhood(img, radius)
    inv = img / (mean + 1e-3)
    inv = np.clip(inv, 0.0, 3.0)
    return np.stack([img, mean, std, inv], axis=-1)


def features_to_matrix(samples: list[dict], radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Stack features and labels for a list of samples."""
    xs, ys = [], []
    for s in samples:
        xs.append(pixel_features(s["image"], radius).reshape(-1, len(FEATURE_NAMES)))
        ys.append(s["shadow"].reshape(-1).astype(np.float64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ----------------------------------------------------------------------------
# B0: global brightness threshold
# ----------------------------------------------------------------------------
class GlobalThreshold:
    """Shadow = brightness below one global threshold, chosen on the training split.

    Two details that matter and are easy to get wrong:

    * The search grid must cover the FULL brightness range. The first version of
      this class searched [0, 0.6], so for low sun elevations (where almost
      everything is shadowed) the optimum landed on the grid boundary and the
      "fitted" threshold was really just "the largest value I was allowed to try".
    * The selection criterion is **balanced accuracy**, not F1. With 90%+ shadow
      pixels, "call everything shadow" scores a respectable F1 while being
      useless, so an F1-maximising fit happily chooses that degenerate answer.
    """

    def __init__(self) -> None:
        self.thr = 0.5
        self.train_balanced_acc = float("nan")

    def fit(self, samples: list[dict], n_grid: int = 400) -> "GlobalThreshold":
        xs, ys = [], []
        for s in samples:
            xs.append(s["image"].reshape(-1).astype(np.float64))
            ys.append(s["shadow"].reshape(-1))
        x = np.concatenate(xs)
        y = np.concatenate(ys).astype(bool)

        grid = np.linspace(0.0, 1.0, n_grid)
        pos = max(int(y.sum()), 1)
        neg = max(int((~y).sum()), 1)
        best, best_score = 0.5, -1.0
        for t in grid:
            pred = x < t
            tp = int(np.logical_and(pred, y).sum())
            tn = int(np.logical_and(~pred, ~y).sum())
            tpr = tp / pos
            tnr = tn / neg
            score = 0.5 * (tpr + tnr)          # balanced accuracy
            if score > best_score:
                best, best_score = float(t), float(score)
        self.thr = best
        self.train_balanced_acc = best_score
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        return (img < self.thr).astype(np.uint8)


# ----------------------------------------------------------------------------
# B1: per-pixel logistic regression
# ----------------------------------------------------------------------------
class LogRegPixel:
    """Logistic regression on per-pixel local features, trained with Adam."""

    def __init__(self, l2: float = 1e-4, lr: float = 5e-2, iters: int = 400,
                 radius: int = 4, seed: int = 0) -> None:
        self.l2, self.lr, self.iters, self.radius, self.seed = l2, lr, iters, radius, seed
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[float] = []

    # -- training -----------------------------------------------------------
    def _standardise_fit(self, x: np.ndarray) -> np.ndarray:
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-6
        return (x - self.mu) / self.sd

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sd is not None
        return (x - self.mu) / self.sd

    def fit(self, samples: list[dict], verbose: bool = True) -> "LogRegPixel":
        x, y = features_to_matrix(samples, self.radius)
        # subsample pixels for speed; keep the class balance
        rng = np.random.default_rng(self.seed)
        n_max = 400_000
        if x.shape[0] > n_max:
            idx = rng.choice(x.shape[0], n_max, replace=False)
            x, y = x[idx], y[idx]
        x = self._standardise_fit(x)

        n, d = x.shape
        self.w = np.zeros(d)
        self.b = 0.0
        mw = np.zeros(d)
        vw = np.zeros(d)
        mb = vb = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8

        for it in range(1, self.iters + 1):
            z = x @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad_w = x.T @ (p - y) / n + self.l2 * self.w
            grad_b = float((p - y).mean())

            mw = b1 * mw + (1 - b1) * grad_w
            vw = b2 * vw + (1 - b2) * grad_w * grad_w
            mb = b1 * mb + (1 - b1) * grad_b
            vb = b2 * vb + (1 - b2) * grad_b * grad_b
            mw_hat = mw / (1 - b1 ** it)
            vw_hat = vw / (1 - b2 ** it)
            mb_hat = mb / (1 - b1 ** it)
            vb_hat = vb / (1 - b2 ** it)

            self.w -= self.lr * mw_hat / (np.sqrt(vw_hat) + eps)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if it % 50 == 0 or it == 1:
                loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
                loss += 0.5 * self.l2 * float(self.w @ self.w)
                self.history.append(float(loss))
                if verbose:
                    acc = float(((p > 0.5) == (y > 0.5)).mean())
                    print(f"    iter {it:4d}  loss {loss:.5f}  pixel-acc {acc:.4f}", flush=True)
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, img: np.ndarray) -> np.ndarray:
        f = pixel_features(img, self.radius)
        flat = self._standardise(f.reshape(-1, f.shape[-1]))
        z = flat @ self.w + self.b
        return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).reshape(img.shape)

    def predict(self, img: np.ndarray) -> np.ndarray:
        return (self.predict_proba(img) > 0.5).astype(np.uint8)


class RelativeThreshold:
    """Shadow = local contrast below a threshold, i.e. a calibration-free cue.

    The point of this baseline is to isolate WHY an absolute threshold fails.

    A global threshold on raw brightness assumes the sensor's absolute scale is
    known. When the exposure/gain is an unknown per-image factor `g`, the shadow
    brightness is `g * albedo * ambient`, so a fixed threshold is simply wrong by
    a factor of `g` -- and the more the gain varies, the worse it gets.

    Dividing by a local mean removes `g` (and most of the albedo and shading
    variation) because both numerator and denominator scale with it:

        r = I / mean_local(I)      is invariant to any global multiplicative gain

    So a threshold on `r` survives what a threshold on `I` cannot. This is the
    cheapest possible "illumination-invariant feature", and it is the honest
    baseline that a paper should report before claiming a learned model helps.
    """

    def __init__(self, radius: int = 6) -> None:
        self.radius = radius
        self.thr = 0.5
        self.train_balanced_acc = float("nan")

    def _ratio(self, img: np.ndarray) -> np.ndarray:
        mean, _ = _neighbourhood(img, self.radius)
        return img / (mean + 1e-4)

    def fit(self, samples: list[dict], n_grid: int = 400) -> "RelativeThreshold":
        xs, ys = [], []
        for s in samples:
            xs.append(self._ratio(s["image"]).reshape(-1))
            ys.append(s["shadow"].reshape(-1))
        x = np.concatenate(xs)
        y = np.concatenate(ys).astype(bool)
        grid = np.linspace(0.0, 1.0, n_grid)
        pos, neg = max(int(y.sum()), 1), max(int((~y).sum()), 1)
        best, best_score = 0.5, -1.0
        for t in grid:
            pred = x < t
            tp = int(np.logical_and(pred, y).sum())
            tn = int(np.logical_and(~pred, ~y).sum())
            score = 0.5 * (tp / pos + tn / neg)
            if score > best_score:
                best, best_score = float(t), float(score)
        self.thr = best
        self.train_balanced_acc = best_score
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        return (self._ratio(img) < self.thr).astype(np.uint8)


# ----------------------------------------------------------------------------
# B2: DEM oracle
# ----------------------------------------------------------------------------
class DEMOracle:
    """Upper bound: shadow computed from the DEM with the true sun geometry.

    It is handed `el` at test time, so it is an oracle in the sense that it never
    has to *infer* the lighting -- it is told. No image is used at all.
    """

    def __init__(self, cell: float = 1.0) -> None:
        self.cell = cell

    def predict(self, sample: dict) -> np.ndarray:
        illum = ls.compute_illumination(sample["dem"], sample["az"], sample["el"],
                                        cell=self.cell, use_slope_term=True)
        return (illum == 0).astype(np.uint8)
