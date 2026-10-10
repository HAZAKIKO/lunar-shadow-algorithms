# -*- coding: utf-8 -*-
"""
make_figure.py —— 可视化月面光照/阴影算法结果
============================================

运行：python make_figure.py [--n 192] [--az 137] [--el 6] [--out figure.png]

产出 6 幅子图：
  (a) 合成月面 DEM + 等高线
  (b) 地平线图在给定太阳方位角下的极坐标切片（度）
  (c) 射线行进法的阴影掩膜
  (d) 地平线图查表法的阴影掩膜 + 两法不一致像素（红）
  (e) 由阴影掩膜反演太阳几何的 IoU 热力图
  (f) 容差 F1 / IoU 随太阳角扰动的衰减曲线

图表标题用英文，避免中文字体缺失告警。所有数值都会同时打印到终端。
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lunar_shadow as ls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--az", type=float, default=137.0)
    ap.add_argument("--el", type=float, default=6.0)
    ap.add_argument("--out", type=str, default="lunar_shadow_figure.png")
    args = ap.parse_args()

    dem = ls.make_synthetic_lunar_dem(n=args.n, seed=args.seed, relief=8.0)
    az, el = args.az, args.el
    print(f"DEM {dem.shape}, relief std = {dem.std():.3f} m")
    print(f"sun az = {az}°, el = {el}°")

    # ---- 三种光照结果 -------------------------------------------------------
    m_ray = ls.compute_illumination(dem, az, el, cell=1.0, use_slope_term=True)
    Hmap = ls.horizon_map(dem, cell=1.0, n_az=72)
    m_hor = ls.illuminate_from_horizon(dem, Hmap, az, el, cell=1.0, use_slope_term=True)
    agree = float((m_ray == m_hor).mean())
    f_shadow_ray = float((m_ray == 0).mean())
    f_shadow_hor = float((m_hor == 0).mean())
    print(f"shadow fraction: ray = {f_shadow_ray:.4f}, horizon = {f_shadow_hor:.4f}")
    print(f"agreement = {agree:.4f}")

    # ---- 方位角切片 ---------------------------------------------------------
    n_az = Hmap.shape[0]
    az_bins = np.arange(n_az) * (360.0 / n_az)
    i0, j0 = args.n // 2, args.n // 2
    az_profile = Hmap[:, i0, j0]

    # ---- 太阳几何反演 -------------------------------------------------------
    gt = (m_ray == 0)
    print("recovering sun geometry (coarse->fine search) ...")
    res = ls.recover_sun_geometry(
        dem, gt, az_range=(0, 360), el_range=(1, 40),
        az_step=2.0, el_step=0.5, coarse_step=5.0, tol=1, metric="iou",
    )
    d_az = min(abs(res["az_deg"] - az), 360 - abs(res["az_deg"] - az))
    print(f"recovered az = {res['az_deg']:.1f}° (Δ={d_az:.1f}°), "
          f"el = {res['el_deg']:.1f}° (Δ={abs(res['el_deg']-el):.1f}°), "
          f"IoU = {res['score']:.4f}")

    # ---- 容差衰减曲线 -------------------------------------------------------
    shifts = [0, 1, 2, 3, 4, 6, 8, 12]
    ious, f1s = [], []
    for s in shifts:
        shifted = ls._shift_zero(gt.astype(np.uint8), 0, s, 0).astype(bool)
        ious.append(ls.iou(shifted, gt))
        f1s.append(ls.f1_with_tolerance(shifted, gt, tol=3))

    # ---- 绘图 ---------------------------------------------------------------
    fig = plt.figure(figsize=(16.5, 10))
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    fig.suptitle(
        f"Lunar illumination / shadow algorithms   "
        f"(sun az={az:.0f}°, el={el:.0f}°, DEM {args.n}×{args.n})",
        fontsize=14,
    )

    ax = ax_a
    im = ax.imshow(dem, cmap="terrain")
    ax.contour(dem, levels=8, colors="k", linewidths=0.3, alpha=0.5)
    ax.set_title("(a) Synthetic lunar DEM [m]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = ax_b
    theta = np.deg2rad(az_bins)
    ax.plot(theta, az_profile, lw=1.0, color="tab:orange")
    ax.axvline(np.deg2rad(az), color="k", ls="--", lw=1.2, label=f"sun az={az:.0f}°")
    ax.axhline(el, color="tab:blue", ls=":", lw=1.2, label=f"sun el={el:.0f}°")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"(b) Horizon angle at centre pixel\n(max={az_profile.max():.1f}°)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)

    ax = ax_c
    ax.imshow(dem, cmap="gray", alpha=0.45)
    ax.imshow(np.ma.masked_where(m_ray > 0, m_ray), cmap="autumn", vmin=0, vmax=1)
    ax.set_title(f"(c) Ray-marching shadow   ({f_shadow_ray*100:.1f}% shaded)")

    ax = ax_d
    ax.imshow(dem, cmap="gray", alpha=0.45)
    ax.imshow(np.ma.masked_where(m_hor > 0, m_hor), cmap="autumn", vmin=0, vmax=1)
    diff = (m_ray != m_hor)
    ys, xs = np.nonzero(diff)
    ax.scatter(xs, ys, s=1.5, c="red", marker="s", label=f"disagreement ({diff.sum()} px)")
    ax.set_title(f"(d) Horizon-map shadow   ({f_shadow_hor*100:.1f}% shaded), "
                 f"agreement {agree*100:.2f}%", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)

    ax = ax_e
    azs = res["table"]["az"]
    els = res["table"]["el"]
    im = ax.imshow(res["table"]["scores"], aspect="auto", origin="lower", cmap="viridis",
                   extent=[azs.min(), azs.max(), els.min(), els.max()])
    ax.plot(res["az_deg"], res["el_deg"], "r*", ms=14, label="best fit")
    ax.plot(az, el, "wo", ms=8, mfc="none", mew=2, label="ground truth")
    ax.set_xlabel("sun azimuth [deg]")
    ax.set_ylabel("sun elevation [deg]")
    ax.set_title(f"(e) Shadow-mask inversion IoU\n"
                 f"best az={res['az_deg']:.0f}° el={res['el_deg']:.0f}°", fontsize=11)
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = ax_f
    ax.plot(shifts, f1s, "o-", label="F1 (tol=3 px)")
    ax.plot(shifts, ious, "s--", label="IoU (no tolerance)")
    ax.set_xlabel("registration shift [px]")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("(f) Why tolerance-based F1 is required", fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")

    # 终端数值表（供论文直接抄）
    print("\n--- numeric summary ---")
    print(f"shadow fraction ray / horizon : {f_shadow_ray:.4f} / {f_shadow_hor:.4f}")
    print(f"cross-check agreement         : {agree:.4f}")
    print(f"recovered sun az / el         : {res['az_deg']:.1f} / {res['el_deg']:.1f} "
          f"(truth {az:.1f} / {el:.1f})")
    print(f"best-fit IoU                  : {res['score']:.4f}")
    print("shift -> F1(tol=3) / IoU      : "
          + ", ".join(f"{s}:{f:.3f}/{i:.3f}" for s, f, i in zip(shifts, f1s, ious)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
