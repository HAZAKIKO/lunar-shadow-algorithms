# -*- coding: utf-8 -*-
"""
run.py —— Illumination-generalisation experiment
===============================================

Two evaluations, both using ground truth that comes free from the renderer:

  Experiment 1: DISJOINT ELEVATION BANDS
      train on elevations {8, 15, 25, 40}°, test on {4, 12, 20, 30}°.
      Terrains are also disjoint between splits. This answers the practical
      question: "I trained at some sun angles -- how badly do I degrade at the
      angles I never saw?"

  Experiment 2: THE GENERALISATION MATRIX
      for every train elevation e_tr in the sweep, fit a model using ONLY that
      elevation, then evaluate the SAME fixed test terrains (same azimuth) at
      every test elevation e_te. The result is a |sweep| x |sweep| table whose
      diagonal is "lighting matched" and whose off-diagonal is the domain shift.
      A model that generalises would have a flat table; a model that memorised
      the lighting has a bright diagonal and a dark everywhere-else.

Metrics: precision / recall / F1 / IoU, plus a tolerance-based (3 px) F1 for the
learned models -- because the guide's §8.6 point is that per-pixel scores mostly
measure registration error, and we want to show that this matters here too.

Outputs: results.json + results.md (tables) in this directory.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import dataset as ds
import lunar_shadow as ls
from baselines import DEMOracle, GlobalThreshold, LogRegPixel, RelativeThreshold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def pixel_metrics(pred: np.ndarray, gt: np.ndarray, tol: int = 0) -> dict:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if tol > 0:
        f1_tol = ls.f1_with_tolerance(p, g, tol=tol)
    else:
        f1_tol = float("nan")
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "f1_tol3": float(f1_tol),
        "iou": float(iou), "acc": float((tp + tn) / max(p.size, 1)),
    }


def agg(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "f1_tol3"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    f1t = [r["f1_tol3"] for r in rows if not np.isnan(r["f1_tol3"])]
    out["f1_tol3"] = float(np.mean(f1t)) if f1t else float("nan")
    return out


def evaluate_on_samples(model, samples: list[dict], tol: int = 3, is_oracle: bool = False) -> dict:
    rows = []
    for s in samples:
        pred = model.predict(s) if is_oracle else model.predict(s["image"])
        rows.append(pixel_metrics(pred, s["shadow"], tol=tol))
    return agg(rows)


def _rows_from_npz(path: str) -> list[dict]:
    z = np.load(path)
    gains = z["gain"] if "gain" in z.files else np.ones(z["image"].shape[0], dtype=np.float32)
    return [
        {"image": z["image"][i], "shadow": z["shadow"][i], "dem": None,
         "az": float(z["az"][i]), "el": float(z["el"][i]),
         "gain": float(gains[i]), "terrain_id": int(z["terrain_id"][i])}
        for i in range(z["image"].shape[0])
    ]


def load_split(name: str) -> list[dict]:
    return _rows_from_npz(os.path.join(DATA, f"{name}.npz"))


def load_sweep() -> dict[float, list[dict]]:
    out = {}
    for f in sorted(os.listdir(DATA)):
        if not f.startswith("sweep_el"):
            continue
        el = float(f[len("sweep_el"):-4])
        out[el] = _rows_from_npz(os.path.join(DATA, f))
    return out


# ----------------------------------------------------------------------------
# Experiment 1
# ----------------------------------------------------------------------------
def experiment_bands(train: list[dict], test: list[dict]) -> dict:
    print("\n" + "=" * 78)
    print("Experiment 1: disjoint elevation bands")
    print("=" * 78)

    results = {}

    print("\n[B0] GlobalThreshold -- fitted on the training split")
    t0 = time.time()
    b0 = GlobalThreshold().fit(train)
    print(f"  fitted threshold = {b0.thr:.4f}  ({time.time() - t0:.1f}s)")
    results["B0_trainfit_on_train"] = evaluate_on_samples(b0, train)
    results["B0_trainfit_on_test"] = evaluate_on_samples(b0, test)
    print(f"  train {results['B0_trainfit_on_train']['f1']:.4f} F1 | "
          f"test {results['B0_trainfit_on_test']['f1']:.4f} F1")

    print("\n[B0m] GlobalThreshold -- fitted on ONE matched elevation (8 deg)")
    tr_matched = [s for s in train if s["el"] == 8.0]
    b0m = GlobalThreshold().fit(tr_matched)
    print(f"  fitted threshold = {b0m.thr:.4f} on {len(tr_matched)} samples")
    results["B0m_el8_on_el8"] = evaluate_on_samples(
        b0m, [s for s in test if s["el"] == 4.0] + tr_matched)
    results["B0m_el8_on_alltest"] = evaluate_on_samples(b0m, test)
    print(f"  on its own elevation {results['B0m_el8_on_el8']['f1']:.4f} F1 | "
          f"on all test elevations {results['B0m_el8_on_alltest']['f1']:.4f} F1")

    print("\n[B0rel] RelativeThreshold -- local-contrast cue (gain-invariant by construction)")
    b0r = RelativeThreshold(radius=6).fit(train)
    print(f"  fitted threshold = {b0r.thr:.4f}  "
          f"(train balanced acc {b0r.train_balanced_acc:.4f})")
    results["B0rel_on_train"] = evaluate_on_samples(b0r, train)
    results["B0rel_on_test"] = evaluate_on_samples(b0r, test)
    print(f"  train {results['B0rel_on_train']['f1']:.4f} F1 | "
          f"test {results['B0rel_on_test']['f1']:.4f} F1")

    print("\n[B1] LogRegPixel -- trained on the full training split")
    t0 = time.time()
    b1 = LogRegPixel(iters=400)
    b1.fit(train)
    print(f"  trained in {time.time() - t0:.1f}s")
    print("  learned weights (standardised feature space):")
    for name, w in zip(["brightness", "local_mean", "local_std", "inv_bright"], b1.w):
        print(f"    {name:14} {w:+.3f}")
    results["B1_on_train"] = evaluate_on_samples(b1, train)
    results["B1_on_test"] = evaluate_on_samples(b1, test)
    print(f"  train {results['B1_on_train']['f1']:.4f} F1 | "
          f"test {results['B1_on_test']['f1']:.4f} F1 "
          f"(tol3 {results['B1_on_test']['f1_tol3']:.4f})")

    # per-elevation breakdown of the learned model
    print("\n  B1 per test elevation:")
    per_el = {}
    for el in sorted({s["el"] for s in test}):
        sub = [s for s in test if s["el"] == el]
        m = evaluate_on_samples(b1, sub)
        per_el[el] = m
        print(f"    el={el:5.1f}°  F1 {m['f1']:.4f}  IoU {m['iou']:.4f}  "
              f"recall {m['recall']:.4f}  precision {m['precision']:.4f}")
    results["B1_per_test_elevation"] = per_el
    return results


# ----------------------------------------------------------------------------
# Experiment 2: the matrix
# ----------------------------------------------------------------------------
def experiment_matrix(train: list[dict], sweep: dict[float, list[dict]]) -> dict:
    print("\n" + "=" * 78)
    print("Experiment 2: train-elevation x test-elevation matrix")
    print("=" * 78)

    sweep_els = sorted(sweep)
    train_els = sorted({s["el"] for s in train})

    # ---- B0: one global threshold per training elevation ----
    print("\n[B0] GlobalThreshold, one per train elevation")
    b0_matrix, b0_thr = [], {}
    for e_tr in train_els:
        sub = [s for s in train if s["el"] == e_tr]
        b0 = GlobalThreshold().fit(sub)
        b0_thr[e_tr] = b0.thr
        row = [evaluate_on_samples(b0, sweep[e_te])["f1"] for e_te in sweep_els]
        b0_matrix.append(row)
        print(f"  e_tr={e_tr:5.1f}°  thr={b0.thr:.3f}  ->  " +
              " ".join(f"{v:.2f}" for v in row))

    # ---- B0rel: the gain-invariant variant, same protocol ----
    print("\n[B0rel] RelativeThreshold, one per train elevation")
    b0r_matrix = []
    for e_tr in train_els:
        sub = [s for s in train if s["el"] == e_tr]
        m = RelativeThreshold(radius=6).fit(sub)
        row = [evaluate_on_samples(m, sweep[e_te])["f1"] for e_te in sweep_els]
        b0r_matrix.append(row)
        print(f"  e_tr={e_tr:5.1f}°  thr={m.thr:.3f}  ->  " +
              " ".join(f"{v:.2f}" for v in row))

    # ---- B1: one logistic model per training elevation ----
    print("\n[B1] LogRegPixel, one per train elevation (this is the matrix that matters)")
    b1_matrix, b1_iou, b1_models = [], [], {}
    for e_tr in train_els:
        sub = [s for s in train if s["el"] == e_tr]
        t0 = time.time()
        m = LogRegPixel(iters=300)
        m.fit(sub, verbose=False)
        b1_models[e_tr] = m
        row = [evaluate_on_samples(m, sweep[e_te])["f1"] for e_te in sweep_els]
        row_iou = [evaluate_on_samples(m, sweep[e_te])["iou"] for e_te in sweep_els]
        b1_matrix.append(row)
        b1_iou.append(row_iou)
        print(f"  e_tr={e_tr:5.1f}°  ({time.time() - t0:4.1f}s)  ->  " +
              " ".join(f"{v:.2f}" for v in row))

    diag = [b1_matrix[i][sweep_els.index(e)] for i, e in enumerate(train_els) if e in sweep_els]
    off = [v for i, r in enumerate(b1_matrix) for j, v in enumerate(r)
           if train_els[i] != sweep_els[j]]
    print(f"\n  diagonal (matched lighting) mean F1 = {np.mean(diag):.4f}")
    print(f"  off-diagonal (shifted lighting) mean F1 = {np.mean(off):.4f}")
    print(f"  --> degradation = {np.mean(diag) - np.mean(off):+.4f} F1")

    return {
        "train_elevations": train_els,
        "sweep_elevations": sweep_els,
        "B0_thresholds": {str(k): v for k, v in b0_thr.items()},
        "B0_f1": b0_matrix,
        "B0rel_f1": b0r_matrix,
        "B1_f1": b1_matrix,
        "B1_iou": b1_iou,
        "diag_mean_f1": float(np.mean(diag)),
        "offdiag_mean_f1": float(np.mean(off)),
        "degradation_f1": float(np.mean(diag) - np.mean(off)),
        "diag_mean_f1_B0": float(np.mean(
            [b0_matrix[i][sweep_els.index(e)] for i, e in enumerate(train_els) if e in sweep_els])),
        "diag_mean_f1_B0rel": float(np.mean(
            [b0r_matrix[i][sweep_els.index(e)] for i, e in enumerate(train_els) if e in sweep_els])),
        "extrapolation_B1": _extrapolation_note(b1_matrix, train_els, sweep_els),
    }


def _extrapolation_note(matrix: list[list[float]], train_els: list[float],
                        sweep_els: list[float]) -> dict:
    """Summarise how B1 behaves when the test elevation is OUTSIDE the trained band.

    Interpolation (test elevation inside the trained range) and extrapolation
    (outside it) are separated, because a real rover does not get to choose the
    sun angle it must operate under.
    """
    inter, extra = [], []
    for i, e_tr in enumerate(train_els):
        for j, e_te in enumerate(sweep_els):
            if e_te == e_tr:
                continue
            (inter if e_te <= max(train_els) else extra).append(matrix[i][j])
    return {
        "interpolation_mean_f1": float(np.mean(inter)) if inter else float("nan"),
        "extrapolation_mean_f1": float(np.mean(extra)) if extra else float("nan"),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def markdown_table(name: str, matrix: list[list[float]],
                   train_els: list[float], sweep_els: list[float]) -> str:
    head = "| train \\ test | " + " | ".join(f"{e:.0f}°" for e in sweep_els) + " |"
    sep = "|---|" + "---|" * len(sweep_els)
    lines = [f"### {name}", "", head, sep]
    for i, e_tr in enumerate(train_els):
        cells = []
        for j, e_te in enumerate(sweep_els):
            v = matrix[i][j]
            mark = "**" if abs(e_tr - e_te) < 1e-6 else ""
            cells.append(f"{mark}{v:.3f}{mark}")
        lines.append(f"| **{e_tr:.0f}°** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    if not os.path.exists(os.path.join(DATA, "train.npz")):
        raise SystemExit("dataset missing -- run: python dataset.py")

    print("loading data ...")
    train = load_split("train")
    test = load_split("test")
    sweep = load_sweep()
    print(f"  train {len(train)} | test {len(test)} | "
          f"sweep {len(sweep)} elevations x {len(next(iter(sweep.values())))} samples")

    results = {
        "config": ds.CFG,
        "n_train": len(train), "n_test": len(test),
    }
    results.update(experiment_bands(train, test))
    mat = experiment_matrix(train, sweep)
    results["matrix"] = mat

    # DEM oracle on the sweep (upper bound)
    print("\n[B2] DEMOracle (upper bound, uses the true sun geometry)")
    oracle = DEMOracle(cell=ds.CFG["cell"])
    oracle_rows = []
    for el in sorted(sweep):
        rows = []
        for s in sweep[el]:
            s2 = dict(s)
            s2["dem"] = ds.build_terrain(s["terrain_id"], ds.CFG).astype(np.float64)
            rows.append(pixel_metrics(oracle.predict(s2), s["shadow"], tol=3))
        m = agg(rows)
        oracle_rows.append(m["f1"])
        print(f"  el={el:5.1f}°  F1 {m['f1']:.4f}  IoU {m['iou']:.4f}")
    results["B2_oracle_f1_by_elevation"] = dict(zip([str(e) for e in sorted(sweep)], oracle_rows))

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    md = []
    md.append("# Illumination-generalisation results\n")
    md.append("Generated by `run.py`; raw numbers in `results.json`.\n")
    md.append("## Experiment 1 — disjoint elevation bands\n")
    md.append("| model | split | precision | recall | F1 | IoU | F1 @3px tol |")
    md.append("|---|---|---|---|---|---|---|")
    for key, label in [
        ("B0_trainfit_on_train", "B0 global threshold (train-fitted)"),
        ("B0_trainfit_on_test", "B0 global threshold (train-fitted)"),
        ("B0m_el8_on_el8", "B0 threshold @8° only"),
        ("B0m_el8_on_alltest", "B0 threshold @8° only"),
        ("B1_on_train", "B1 logistic regression"),
        ("B1_on_test", "B1 logistic regression"),
    ]:
        m = results[key]
        split = "train" if "train" in key or "el8_on_el8" in key else "test"
        md.append(f"| {label} | {split} | {m['precision']:.3f} | {m['recall']:.3f} | "
                  f"{m['f1']:.3f} | {m['iou']:.3f} | {m['f1_tol3']:.3f} |")
    md.append("")
    md.append("### B1 per test elevation\n")
    md.append("| test elevation | F1 | IoU | recall | precision |")
    md.append("|---|---|---|---|---|")
    for el, m in results["B1_per_test_elevation"].items():
        md.append(f"| {float(el):.1f}° | {m['f1']:.3f} | {m['iou']:.3f} | "
                  f"{m['recall']:.3f} | {m['precision']:.3f} |")
    md.append("")
    md.append("## Experiment 2 — generalisation matrices\n")
    md.append("Rows are the elevation the model was trained at; columns are the elevation it "
              "was tested at. **Bold** marks matched lighting.\n")
    md.append(markdown_table("B0 — global brightness threshold, F1",
                             mat["B0_f1"], mat["train_elevations"], mat["sweep_elevations"]))
    md.append("")
    md.append(markdown_table("B0rel — local-contrast (gain-invariant) threshold, F1",
                             mat["B0rel_f1"], mat["train_elevations"], mat["sweep_elevations"]))
    md.append("")
    md.append(markdown_table("B1 — per-pixel logistic regression, F1",
                             mat["B1_f1"], mat["train_elevations"], mat["sweep_elevations"]))
    md.append("")
    md.append(markdown_table("B1 — per-pixel logistic regression, IoU",
                             mat["B1_iou"], mat["train_elevations"], mat["sweep_elevations"]))
    md.append("")
    md.append(f"Matched-lighting mean F1: **{mat['diag_mean_f1']:.4f}**  \n"
              f"Shifted-lighting mean F1: **{mat['offdiag_mean_f1']:.4f}**  \n"
              f"Degradation: **{mat['degradation_f1']:+.4f}**\n")
    md.append("### Interpolation vs extrapolation (B1)\n")
    ex = mat["extrapolation_B1"]
    md.append(f"- Test elevation inside the trained band (interpolation): "
              f"**{ex['interpolation_mean_f1']:.4f}** F1\n"
              f"- Test elevation outside the trained band (extrapolation): "
              f"**{ex['extrapolation_mean_f1']:.4f}** F1\n")
    md.append("### Matched-lighting F1 by model family\n")
    md.append(f"| model | matched-lighting mean F1 |\n|---|---|\n"
              f"| B0 global brightness threshold | {mat['diag_mean_f1_B0']:.4f} |\n"
              f"| B0rel local-contrast threshold | {mat['diag_mean_f1_B0rel']:.4f} |\n"
              f"| B1 logistic regression (4 features) | {mat['diag_mean_f1']:.4f} |\n")
    md.append("## B2 — DEM oracle (upper bound)\n")
    md.append("| elevation | F1 |")
    md.append("|---|---|")
    for el, v in results["B2_oracle_f1_by_elevation"].items():
        md.append(f"| {float(el):.1f}° | {v:.3f} |")
    md.append("")

    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")
    print(f"wrote {os.path.join(HERE, 'results.md')}")


if __name__ == "__main__":
    main()
