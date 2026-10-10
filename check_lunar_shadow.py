# -*- coding: utf-8 -*-
"""
check_lunar_shadow.py —— lunar_shadow.py 的自检脚本
==================================================

运行：python check_lunar_shadow.py

逐项验证（无需任何外部数据）：
  1. 工具函数正确性
  2. 射线行进 vs 地平线图：两种独立实现的交叉一致性
  3. 射线行进：零填充无回绕 bug（边界像素不被错误遮挡）
  4. 太阳位置反演闭环：已知 (az, el) -> 阴影 -> 反演 -> 误差
  5. 解析太阳位置：亚太阳点范围合理性 + 极区太阳高度角量级
  6. 容差 F1 与逐像素 IoU 的差异（说明为什么必须用容差）
  7. 阴影长度 <-> 物体高度 的互逆性
  8. 可复现性：同种子两次生成 DEM 完全一致

全部通过则打印 PASS。
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
    print("lunar_shadow 自检")
    print("=" * 78)

    # ---------------------------------------------------------------- 1
    print("\n[1] 工具函数正确性")
    dcol, drow, dz = sun_direction(90.0, 0.0)      # 太阳在正东、地平线上
    report("正东方向向量", abs(dcol - 1) < 1e-9 and abs(drow) < 1e-9 and abs(dz) < 1e-9,
           f"(dcol,drow,dz)=({dcol:.3f},{drow:.3f},{dz:.3f})")
    dcol, drow, dz = sun_direction(0.0, 90.0)      # 太阳在天顶
    report("天顶方向向量", abs(dz - 1) < 1e-9, f"dz={dz:.3f}")
    dcol, drow, dz = sun_direction(0.0, 0.0)       # 太阳在正北
    report("正北方向向量（drow应为负）", drow < -0.99, f"drow={drow:.3f}")

    dem = make_synthetic_lunar_dem(n=192, seed=7)
    report("DEM 形状与量级", dem.shape == (192, 192) and 1.0 < dem.std() < 50.0,
           f"std={dem.std():.2f} m")

    # ---------------------------------------------------------------- 2
    print("\n[2] 射线行进 vs 地平线图 交叉一致性")
    dem = make_synthetic_lunar_dem(n=160, seed=20261010)
    # ⚠️ 必须包含**非分箱对齐**的方位角（如 137°），否则 72 分箱（5° 步长）会正好落在
    #    分箱中心，插值变成恒等操作，测不出分辨率的影响。
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
    report("地平线图形状", horizon_map(dem, n_az=72).shape == (72, 160, 160))
    report("两实现阴影面积占比接近（<2%）", frac72 < 0.02,
           f"最大占比差={frac72*100:.2f}%")
    report("两实现像素一致率 > 95%（n_az=72, 5°分箱）", agree72 > 0.95,
           f"最差一致率={agree72*100:.2f}%")
    report("提高方位分辨率进一步改善一致性（n_az=360）", agree360 >= agree72,
           f"{agree72*100:.2f}% -> {agree360*100:.2f}%")
    print("       注：地平线图的精度损失主要来自**方位角分箱 + 线性插值**，"
          "且只在方位角不落在分箱中心时显现。")
    print("           工程建议：地平线图用于快速遍历/粗筛，关键场景再用射线行进复核。")

    # 解析算例：南北向山脊 + 太阳在正东/正西（这一条能钉死方位角约定）
    ridge = np.zeros((201, 201))
    ridge[:, 100] = 10.0
    Hr = horizon_map(ridge, cell=1.0, n_az=72)
    ang = float(np.rad2deg(np.arctan2(10.0, 50.0)))
    m_w = int(round(270.0 / 5.0)) % 72      # 太阳在正西
    ok = (abs(Hr[m_w, 100, 150] - ang) < 0.1 and Hr[m_w, 100, 50] < 0.1)
    report("解析算例：山脊西侧被遮挡、东侧无遮挡（10/50 -> 11.31°）", ok,
           f"Hmap西侧={Hr[m_w,100,150]:.2f}° 东侧={Hr[m_w,100,50]:.2f}° 解析值={ang:.2f}°")
    sh_w = compute_illumination(ridge, 270.0, 5.0, cell=1.0) == 0
    cols = np.nonzero(sh_w.any(axis=0))[0]
    report("射线行进：太阳在正西时阴影落在山脊东侧",
           len(cols) > 0 and cols.min() > 100,
           f"阴影列范围 [{cols.min()}, {cols.max()}]，山脊在 col=100")

    # ---------------------------------------------------------------- 3
    print("\n[3] 零填充：边界不产生虚假遮挡")
    # 平坦地形上，任何太阳高度角都不应产生阴影
    # 2) 平地不产生阴影：这是判别式方向的健全性检查
    flat = np.zeros((64, 64))
    m = compute_illumination(flat, az_deg=45.0, el_deg=5.0, cell=1.0)
    report("平坦地形无阴影（无回绕 bug、判别式方向正确）", (m == 1).all(),
           f"阴影像素={int((m == 0).sum())}")
    # 单峰地形：阴影应出现在太阳的反方向一侧
    peak = np.zeros((64, 64))
    peak[32, 32] = 20.0
    m = compute_illumination(peak, az_deg=90.0, el_deg=5.0, cell=1.0)   # 太阳在正东
    shadow = (m == 0)
    ys, xs = np.nonzero(shadow)
    if len(xs):
        report("单峰阴影落在太阳反方向（西侧）",
               float(xs.mean()) < 32.0 and float(xs.max()) < 32,
               f"阴影 {len(xs)} px，质心 col={xs.mean():.1f}，col范围[{xs.min()},{xs.max()}]")
    else:
        report("单峰阴影落在太阳反方向（西侧）", False, "未产生阴影")

    # ---------------------------------------------------------------- 4
    print("\n[4] 太阳位置反演闭环（核心自检）")
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
        # 方位简并：再看一眼次优方位角，评估"可辨识性"
        z_az = res["z_az"]
        order = np.argsort(z_az)[::-1]
        azs = res["table"]["az"]
        second = float(z_az[order[1]]) if len(order) > 1 else 0.0
        margin = res["score"] - second
        print(f"       真值 az={az_true:5.1f}° el={el_true:4.1f}°  ->  "
              f"反演 az={res['az_deg']:5.1f}° el={res['el_deg']:4.1f}°  "
              f"IoU={res['score']:.3f}  Δaz={d_az:.1f}° Δel={d_el:.1f}°")
        print(f"             次优方位 az={azs[order[1]]:5.1f}° (IoU={second:.3f})，"
              f"最优-次优间隔={margin:.4f}")
        report(f"反演可用（真值 az={az_true:.0f}°）",
               d_az <= 8.0 and d_el <= 2.5 and res["score"] > 0.80,
               f"Δaz={d_az:.1f}° Δel={d_el:.1f}° IoU={res['score']:.3f}")
        report(f"方位角存在简并（单张阴影掩膜对 az 敏感度低）",
               margin < 0.05,
               f"最优-次优间隔仅 {margin:.4f}")
    print("       注：这是**方法固有的可辨识性问题**，不是 bug：cast shadow 的形状")
    print("           主要由太阳高度角决定，对方位角的敏感度低得多。")
    print("           对策：用高度角锁定几何，用巡视器自身阴影方向约束方位角，或多帧联合反演。")

    # ---------------------------------------------------------------- 5
    print("\n[5] 解析太阳位置（教学模型精度自查）")
    for jd in [2451545.0, 2451545.0 + 90, 2451545.0 + 9000]:
        lon_s, lat_s = subsolar_lonlat_analytic(jd)
        ok = -180 <= lon_s <= 180 and -1.6 <= lat_s <= 1.6
        report(f"jd={jd:.0f} 亚太阳点", ok, f"lon={lon_s:7.2f}° lat={lat_s:5.2f}°"
               "  (月球黄赤交角约 ±1.54°)")
    # 南极附近的太阳高度角应在极低量级
    jd = 2451545.0 + 9000
    az, el = solar_position_analytic(jd, lon_deg=0.0, lat_deg=-89.5)
    report("南极附近太阳高度角处于低角度区间", -2.0 <= el <= 3.0,
           f"az={az:.1f}° el={el:.2f}°")
    lon, lat = dem_to_lonlat(np.array([0, 100]), np.array([0, 100]), 256,
                             west=-10.0, north=-88.0, res_deg=0.01)
    report("行列 -> 经纬度投影", abs(lon[1] - (-10.0 + 1.0)) < 1e-9
           and abs(lat[1] - (-88.0 - 1.0)) < 1e-9, f"lon={lon[1]:.3f} lat={lat[1]:.3f}")

    # ---------------------------------------------------------------- 6
    print("\n[6] 容差 F1 vs 逐像素 IoU（评估方式的选择）")
    gt = (compute_illumination(dem, 60.0, 6.0, cell=1.0) == 0)
    # 用"整体错位 3 像素"模拟真实 DEM 与影像之间的配准误差
    from lunar_shadow import _shift_zero as _sz
    pred = _sz(gt.astype(np.uint8), 0, 3, 0).astype(bool)
    i0 = iou(pred, gt)
    f0 = f1_with_tolerance(pred, gt, tol=0)
    f1 = f1_with_tolerance(pred, gt, tol=1)
    f3 = f1_with_tolerance(pred, gt, tol=3)
    print(f"       整体错位 3 px：IoU={i0:.4f}  F1(tol=0)={f0:.4f}  "
          f"F1(tol=1)={f1:.4f}  F1(tol=3)={f3:.4f}")
    report("容差使评估对配准误差不敏感", f3 > f1 > f0,
           f"F1: {f0:.3f} -> {f1:.3f} -> {f3:.3f}")
    report("容差不会把完全不同掩膜判为高分",
           f1_with_tolerance(~gt, gt, tol=1) < 0.35,
           f"互补掩膜 F1={f1_with_tolerance(~gt, gt, tol=1):.3f}")

    # ---------------------------------------------------------------- 7
    print("\n[7] 阴影长度 <-> 物体高度 互逆性")
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
        report("互逆性（9 组参数）", True, "误差 < 1e-6")
    el_low = estimate_sun_elevation_from_shadow(1.0, 3800.0, 0.01)   # 38 m 阴影
    report("极区低太阳角量级正确（1m 高 -> 约 38m 影 -> ~1.5°）",
           abs(el_low - 1.508) < 0.1, f"el={el_low:.3f}°")

    # ---------------------------------------------------------------- 8
    print("\n[8] 可复现性")
    a = make_synthetic_lunar_dem(n=128, seed=42)
    b = make_synthetic_lunar_dem(n=128, seed=42)
    c = make_synthetic_lunar_dem(n=128, seed=43)
    report("同种子逐字节一致", np.array_equal(a, b))
    report("不同种子确实不同", not np.array_equal(a, c),
           f"差异像素占比={float((a != c).mean())*100:.1f}%")

    # ---------------------------------------------------------------- 9
    print("\n[9] 图像阴影提取（合成图自检）")
    # 用 DEM 渲染一张"图"，阴影区给低亮度，其余给中等亮度
    illum = compute_illumination(dem, 60.0, 6.0, cell=1.0)
    img = np.where(illum > 0, 0.6, 0.05).astype(np.float64)
    gt_shadow = (illum == 0)
    pred_shadow = image_shadow_mask(img, brightness_thr=0.3, saturation_thr=1.0,
                                    edge_thr=0.05, morph=0).astype(bool)
    f1 = f1_with_tolerance(pred_shadow, gt_shadow, tol=2)
    report("阈值法提取阴影 F1 > 0.9", f1 > 0.9, f"F1(tol=2)={f1:.4f}")
    inv = illumination_invariant(img, sigma=3)
    report("光照不变特征输出在 [0,1] 且非恒定",
           inv.min() >= 0 and inv.max() <= 1.0 and inv.std() > 1e-4,
           f"range=[{inv.min():.3f},{inv.max():.3f}] std={inv.std():.4f}")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 78)
    print(f"结果：{PASS} 项通过，{FAIL} 项失败")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
