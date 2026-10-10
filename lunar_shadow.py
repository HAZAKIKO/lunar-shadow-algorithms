# -*- coding: utf-8 -*-
"""
lunar_shadow.py —— 月面光照/阴影算法的可运行实现
=================================================

对应文档《月面光照与阴影算法指南》的代码部分。包含三类算法：

  A. 正演：DEM + 太阳位置 -> 光照/阴影掩膜
     - compute_illumination()   射线行进（ray-marching），含局部坡面朝向判定
     - horizon_map()            地平线图，预计算一次即可遍历任意太阳方位
     - illuminate_from_horizon() 用地平线图查表得到光照掩膜（O(1) 每次）
     - solar_position_analytic() 月球太阳位置解析近似（教学/仿真用）
     - dem_to_lonlat()          行列号 <-> 经纬度的简单等距投影（教学用）
     - subsolar_lonlat_analytic() 亚太阳点经纬度（教学用）

  B. 反演：图像 -> 阴影掩膜
     - image_shadow_mask()      亮度 + 饱和度双阈值 + 形态学清理
     - illumination_invariant() 局部亮度归一化，压制光照变化

  C. 测量/重建：阴影 -> 几何 / 光照参数
     - estimate_sun_elevation_from_shadow()  阴影长度比 -> 太阳高度角
     - recover_sun_geometry()   由阴影掩膜反演 (方位角, 高度角)
     - shadow_length_to_height() 阴影长度 -> 物体高度

  评估：iou / f1_with_tolerance / match_shadow_masks

依赖：numpy（必需）、scipy（可选，仅用于更快的形态学与滤波）
运行自检：python check_lunar_shadow.py

坐标约定（全文统一，务必记住）
-----------------------------
  * dem 形状 (H, W) 的 numpy 数组，单位米，dem[row, col]。
  * 图像坐标系：col 增大 = 向东，row 减小 = 向北。
  * 方位角 az_deg 使用罗盘方位：0°=正北，90°=正东，180°=正南，270°=正西。
  * 太阳方位角 = 从该点看太阳的罗盘方位；太阳高度角 el_deg = 0° 在地平线，90° 在天顶。
  * 方向向量（图像坐标）：dcol = +sin(az)*cos(el)，drow = -cos(az)*cos(el)。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sun_direction",
    "horizontal_direction",
    "terrain_gradient",
    "compute_illumination",
    "horizon_map",
    "illuminate_from_horizon",
    "solar_position_analytic",
    "subsolar_lonlat_analytic",
    "sun_az_el_from_subsolar",
    "dem_to_lonlat",
    "sun_az_el_from_dem",
    "to_luminance",
    "image_shadow_mask",
    "illumination_invariant",
    "estimate_sun_elevation_from_shadow",
    "shadow_length_to_height",
    "recover_sun_geometry",
    "iou",
    "f1_with_tolerance",
    "match_shadow_masks",
    "make_synthetic_lunar_dem",
]

EPS = 1e-9


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def sun_direction(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """**指向太阳**的单位方向向量（图像坐标 + 高度）。

    返回 (dcol, drow, dz)，其中 dcol 向东为正、drow 向北为负（因为 row 向下增大）。
    """
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return (np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el))


def horizontal_direction(az_deg: float) -> tuple[float, float]:
    """水平面内沿方位角 az 的**单位**方向 (dcol, drow)。

    单独抽出来是为了防止一个真实踩过的坑：用 sun_direction(az, 90.0) 想取"水平方向"，
    结果 cos(90°)=0 让向量退化成零向量，整个地平线计算静默失效（全部返回初始值）。
    """
    az = np.deg2rad(az_deg)
    return (float(np.sin(az)), float(-np.cos(az)))


def terrain_gradient(dem: np.ndarray, cell: float = 1.0):
    """返回 (gx, gy, slope_rad)。
    gx = d z / d 东（米/米），gy = d z / d 北（米/米）。
    """
    drow, dcol = np.gradient(dem, cell)   # np.gradient 返回 (axis0, axis1)
    gx, gy = dcol, -drow                  # row 向南增大 -> 向北取负
    slope = np.arctan(np.hypot(gx, gy))
    return gx, gy, slope


def _shift_zero(a: np.ndarray, drow: int, dcol: int, fill: float) -> np.ndarray:
    """零填充平移：结果[i, j] = a[i - drow, j - dcol]，越界处填 fill。

    这是本模块的关键工具：**必须用零填充，不能用 np.roll**。
    np.roll 是环形回绕，会在图像边缘产生虚假遮挡。
    """
    H, W = a.shape

    def _sl(start: int, shift: int, n: int) -> slice:
        """区间 [start, start+n) 减去 shift 后的切片（用于目标或源）。"""
        return slice(start - shift, start - shift + n)

    rows_dst = slice(max(0, drow), H + min(0, drow))
    cols_dst = slice(max(0, dcol), W + min(0, dcol))
    rows_src = _sl(max(0, drow), drow, rows_dst.stop - rows_dst.start)
    cols_src = _sl(max(0, dcol), dcol, cols_dst.stop - cols_dst.start)

    out = np.full((H, W), fill, dtype=np.float64)
    out[rows_dst, cols_dst] = a[rows_src, cols_src]
    return out


# ----------------------------------------------------------------------------
# A. 正演：DEM -> 光照/阴影
# ----------------------------------------------------------------------------
def compute_illumination(
    dem: np.ndarray,
    az_deg: float,
    el_deg: float,
    cell: float = 1.0,
    max_steps: int | None = None,
    use_slope_term: bool = True,
    min_cos_incidence: float = 0.0,
) -> np.ndarray:
    """射线行进法计算光照掩膜。

    对每个像素沿太阳方向的**反方向**逐步外推，判断地形是否挡住太阳。
    判定条件（第 k 步）：
        z(邻域) > z(本点) + k * cell * tan(el)
    等价于该邻域点的地平高度角 > 太阳高度角。

    Parameters
    ----------
    dem : (H, W) 高程，米
    az_deg, el_deg : 太阳罗盘方位角 / 高度角（度）
    cell : 网格边长，米
    max_steps : 最大外推步数，默认取 2*max(H, W)
    use_slope_term : 是否加入"局部坡面背向太阳"判定（attached shadow）。
        月球极区太阳高度角极低，背阳坡面积占比很大，强烈建议开启。
    min_cos_incidence : 局部入射角余弦阈值，低于此值判为背阳（默认 0，即 >=90° 才判背阳）

    Returns
    -------
    illum : (H, W) uint8，1 = 被太阳照亮，0 = 在阴影中
    """
    dem = np.asarray(dem, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError("dem 必须是二维数组")
    H, W = dem.shape
    if max_steps is None:
        max_steps = int(2 * max(H, W))

    dcol, drow, dz = sun_direction(az_deg, el_deg)
    tan_el = dz / max(np.hypot(dcol, drow), EPS)   # = tan(el)

    illum = np.ones((H, W), dtype=bool)

    # 1) 局部坡面朝向（attached shadow）：背阳坡收不到直射光
    if use_slope_term:
        gx, gy, _ = terrain_gradient(dem, cell)
        # 地表法向 n ∝ (-gx, -gy, 1)；与太阳方向的余弦
        cos_inc = (-gx * dcol - gy * drow + dz) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        illum &= cos_inc > min_cos_incidence

    # 2) 地形投射阴影（cast shadow）
    #
    # 判别式（方向务必注意！）：
    #   "谁挡住太阳" —— 遮挡物在**太阳那一侧**。sun_direction() 返回的是**指向太阳**的
    #   单位向量，所以沿 -(dcol, drow) 方向外推去采样地形；而阴影本身落在太阳的**反侧**。
    #
    #   第 k 步（水平距离 d = k*cell）处，太阳光线相对本点的高度为
    #       ray_h = z(本点) + d * tan(el)
    #   该处地形挡住太阳，当且仅当地形高于这条光线：
    #       z(太阳侧邻域) >= ray_h
    #   等价于"邻域相对本点的地平高度角 >= 太阳高度角"。
    #   平坦地形时左边为 0 < d*tan(el)，不产生阴影（健全性检查）。
    alive = illum.copy()
    for k in range(1, max_steps + 1):
        srow = int(round(-k * drow / cell))      # 注意负号：向太阳方向采样
        scol = int(round(-k * dcol / cell))      # 注意负号：向太阳方向采样
        if abs(srow) >= H or abs(scol) >= W:
            break
        if srow == 0 and scol == 0:
            continue
        shifted = _shift_zero(dem, srow, scol, fill=-np.inf)
        # 关键：在第 k 步处，不能只用 k 这一个距离去比，否则当 srow/scol 因取整重复时
        # （例如 k 从 2 到 40 都映射到同一格），该格会一直用同一个 k 判角度，导致
        # 系统性漏检阴影。正确做法是把该格覆盖的**整个距离区间**都算进去：
        # 只要存在 d ∈ [k-0.5, k+0.5] 使地形挡住光线，就判为阴影。取 k-0.5 最保守，
        # 因为光线最低、最容易被挡。
        if k == 1:
            k_lo = 1.0
        else:
            k_lo = k - 0.5
        ray_h_lo = dem + k_lo * cell * tan_el
        newly = alive & (shifted >= ray_h_lo - EPS)
        if not newly.any():
            if not alive.any():
                break
            continue
        illum[newly] = False
        alive &= ~newly
        if not alive.any():
            break

    return illum.astype(np.uint8)


def horizon_map(
    dem: np.ndarray,
    cell: float = 1.0,
    n_az: int = 72,
    max_dist: float | None = None,
) -> np.ndarray:
    """预计算地平线图（horizon map）。

    H(i, j, m) = 在像素 (i, j) 沿方位角 m 指向**太阳**的方向上，地形相对该点的
    **最大仰角**（度）。有了它，"太阳是否被挡住"退化为一次查表：

        shaded  <=>  el_sun < H(i, j, az)

    换太阳方位角不需要重算，这是它相对射线行进的最大优势。

    ⚠️ 方位角约定：这里的 az 是**太阳的罗盘方位角**（与 compute_illumination 一致），
    因此内部采样方向是 -(sin az, -cos az)（即指向太阳）。早期版本误用了"沿 az 看出去"
    的方向（差 180°），会让阴影落在太阳同侧——单靠代码审查很难发现，
    必须用"南北向山脊 + 太阳在正东/正西"这类解析算例把它钉死。

    Parameters
    ----------
    n_az : 方位角分箱数，360/n_az 为角分辨率。默认 72 -> 5°
    max_dist : 最大探测距离（米），默认 2*max(H, W)*cell

    Returns
    -------
    Hmap : (n_az, H, W) float32，单位度
    """
    dem = np.asarray(dem, dtype=np.float64)
    H, W = dem.shape
    if max_dist is None:
        max_dist = 2.0 * max(H, W) * cell
    max_steps = int(max_dist / cell) + 1

    az_list = np.arange(n_az) * (360.0 / n_az)
    Hmap = np.zeros((n_az, H, W), dtype=np.float32)

    for m, az_deg in enumerate(az_list):
        # 指向太阳的单位水平方向（不能写成 horizontal_direction(az_deg)，差 180°）
        dcol, drow = horizontal_direction(az_deg + 180.0)
        best = np.full((H, W), -90.0, dtype=np.float64)
        for k in range(1, max_steps + 1):
            srow = int(round(k * drow))
            scol = int(round(k * dcol))
            if srow == 0 and scol == 0:
                continue
            if abs(srow) >= H or abs(scol) >= W:
                break
            shifted = _shift_zero(dem, srow, scol, fill=-np.inf)
            ang = np.rad2deg(np.arctan2(shifted - dem, k * cell))
            best = np.maximum(best, ang)
        Hmap[m] = best.astype(np.float32)

    return Hmap


def illuminate_from_horizon(
    dem: np.ndarray,
    Hmap: np.ndarray,
    az_deg: float,
    el_deg: float,
    cell: float = 1.0,
    use_slope_term: bool = True,
) -> np.ndarray:
    """用地平线图 + 线性插值查表得到光照掩膜（不重新射线行进）。"""
    dem = np.asarray(dem, dtype=np.float64)
    n_az = Hmap.shape[0]
    step = 360.0 / n_az
    # 方位角 -> 分箱（循环插值）
    t = (az_deg % 360.0) / step
    m0 = int(np.floor(t)) % n_az
    m1 = (m0 + 1) % n_az
    w = t - np.floor(t)
    H_interp = (1.0 - w) * Hmap[m0] + w * Hmap[m1]

    illum = (el_deg >= H_interp)

    if use_slope_term:
        gx, gy, _ = terrain_gradient(dem, cell)
        dcol, drow, dz = sun_direction(az_deg, el_deg)
        # 地表法向 n ∝ (-gx, -gy, 1)，与**指向太阳**的方向做内积即为入射角余弦
        cos_inc = (-gx * dcol - gy * drow + dz) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        illum &= cos_inc > 0.0

    return illum.astype(np.uint8)


# ----------------------------------------------------------------------------
# A'. 太阳位置与投影（教学/仿真用，高精度请用 SPICE）
# ----------------------------------------------------------------------------
def solar_position_analytic(
    jd: float,
    lon_deg: float,
    lat_deg: float,
    body: str = "moon",
) -> tuple[float, float]:
    """低精度解析法给出太阳的罗盘方位角与高度角（度）。

    ⚠️ 精度说明：这里用的是"太阳视黄经低精度级数 + 月球卡西尼态自转"的简化模型，
    典型误差在 0.1°~0.5° 量级，**足以支撑教学与仿真**，但**不足以支撑任务级规划**。
    任务级请使用 SPICE（spiceypy）：加载月体固连系姿态内核 + 行星星历内核，
    求太阳相对月球中心的方向，转到 IAU_MOON 固连系，再转到站心系。

    月球自转：卡西尼态，自转周期 = 公转周期 ≈ 27.32166 d（同步自转）。
    """
    # 儒略世纪数
    T = (jd - 2451545.0) / 36525.0
    # 太阳几何平黄经与平近点角（Meeus 低精度）
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    Mr = np.deg2rad(M % 360.0)
    # 中心差
    C = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * np.sin(Mr)
         + (0.019993 - 0.000101 * T) * np.sin(2 * Mr)
         + 0.000289 * np.sin(3 * Mr))
    true_lon = L0 + C
    # 黄赤交角
    eps = 23.439291 - 0.0130042 * T
    lam = np.deg2rad(true_lon % 360.0)
    epsr = np.deg2rad(eps)
    # 太阳赤经赤纬（地心赤道系）
    ra = np.arctan2(np.cos(epsr) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(epsr) * np.sin(lam))

    if body.lower() != "moon":
        raise NotImplementedError("目前仅实现月球")

    # 月体固连系：Z 轴指向月球北极；本初子午线用卡西尼态近似
    # 月球北极在 ICRF 中的方向（近似，J2000）：ra_pole ≈ 269.9949°, dec_pole ≈ 66.5392°
    ra_p = np.deg2rad(269.9949)
    dec_p = np.deg2rad(66.5392)
    # 太阳方向单位向量（ICRF）
    s_vec = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    z_axis = np.array([np.cos(dec_p) * np.cos(ra_p), np.cos(dec_p) * np.sin(ra_p), np.sin(dec_p)])
    x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    s_body = np.array([s_vec @ x_axis, s_vec @ y_axis, s_vec @ z_axis])

    # 亚太阳点经纬度
    sub_lat = np.rad2deg(np.arcsin(np.clip(s_body[2], -1, 1)))
    sub_lon = np.rad2deg(np.arctan2(s_body[1], s_body[0]))

    return _sun_az_el_from_subsolar(lon_deg, lat_deg, sub_lon, sub_lat)


def _sun_az_el_from_subsolar(
    lon_deg: float, lat_deg: float, sub_lon_deg: float, sub_lat_deg: float
) -> tuple[float, float]:
    """由亚太阳点与站点经纬度求太阳的罗盘方位角与高度角（度）。"""
    lam = np.deg2rad(lon_deg)
    phi = np.deg2rad(lat_deg)
    lams = np.deg2rad(sub_lon_deg)
    phis = np.deg2rad(sub_lat_deg)
    dlam = lams - lam

    sin_el = np.sin(phi) * np.sin(phis) + np.cos(phi) * np.cos(phis) * np.cos(dlam)
    el = np.arcsin(np.clip(sin_el, -1, 1))
    # 方位角（自北向东）
    num = np.sin(dlam) * np.cos(phis)
    den = np.cos(phi) * np.sin(phis) - np.sin(phi) * np.cos(phis) * np.cos(dlam)
    az = np.rad2deg(np.arctan2(num, den)) % 360.0
    return float(az), float(np.rad2deg(el))


def sun_az_el_from_subsolar(
    lon_deg: float, lat_deg: float, sub_lon_deg: float, sub_lat_deg: float
) -> tuple[float, float]:
    """公开别名：由亚太阳点与站点经纬度求太阳的罗盘方位角与高度角（度）。"""
    return _sun_az_el_from_subsolar(lon_deg, lat_deg, sub_lon_deg, sub_lat_deg)


def subsolar_lonlat_analytic(jd: float) -> tuple[float, float]:
    """低精度亚太阳点经纬度（度）。⚠️ 同样仅适用于教学/仿真。"""
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    M = np.deg2rad((357.52911 + 35999.05029 * T) % 360.0)
    C = (1.914602 - 0.004817 * T) * np.sin(M) + 0.019993 * np.sin(2 * M)
    true_lon = L0 + C
    eps = np.deg2rad(23.439291 - 0.0130042 * T)
    lam = np.deg2rad(true_lon % 360.0)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))

    ra_p = np.deg2rad(269.9949)
    dec_p = np.deg2rad(66.5392)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    s_vec = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    z_axis = np.array([np.cos(dec_p) * np.cos(ra_p), np.cos(dec_p) * np.sin(ra_p), np.sin(dec_p)])
    x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    s_body = np.array([s_vec @ x_axis, s_vec @ y_axis, s_vec @ z_axis])
    return (float(np.rad2deg(np.arctan2(s_body[1], s_body[0]))),
            float(np.rad2deg(np.arcsin(np.clip(s_body[2], -1, 1)))))


def dem_to_lonlat(
    rows: np.ndarray,
    cols: np.ndarray,
    size: int,
    west: float,
    north: float,
    res_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """行列号 -> 经纬度（简单等距投影，教学用）。"""
    lon = west + np.asarray(cols) * res_deg
    lat = north - np.asarray(rows) * res_deg
    return lon, lat


def sun_az_el_from_dem(
    dem: np.ndarray,
    cell: float = 1.0,
    west: float = 0.0,
    north: float = 0.0,
    res_deg: float = 0.001,
    jd: float | None = None,
    point_rc: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """给定 DEM 的地理参考与时间，返回该点太阳的 (方位角, 高度角)。

    jd 为儒略日；为 None 时用固定示例值（仅演示）。
    """
    H, W = dem.shape
    if point_rc is None:
        point_rc = (H // 2, W // 2)
    lon, lat = dem_to_lonlat(
        np.array([point_rc[0]]), np.array([point_rc[1]]), max(H, W),
        west, north, res_deg,
    )
    if jd is None:
        jd = 2451545.0 + 9000.0
    return solar_position_analytic(jd, float(lon[0]), float(lat[0]))


# ----------------------------------------------------------------------------
# B. 反演：图像 -> 阴影
# ----------------------------------------------------------------------------
def to_luminance(img: np.ndarray, normalize: bool = False) -> np.ndarray:
    """RGB/RGBA/灰度 -> 浮点亮度。

    normalize=False（默认）保持绝对亮度尺度 [0,1]——阈值法必须这样，
    因为"暗"是绝对概念。normalize=True 会把最大值拉到 1，仅用于展示。
    """
    a = np.asarray(img).astype(np.float64)
    if a.ndim == 2:
        y = a
    elif a.shape[-1] >= 3:
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    else:
        y = a[..., 0]
    if normalize:
        mx = np.nanmax(y)
        if mx > 1.0:
            y = y / mx
    elif y.max() > 1.0:
        y = y / 255.0
    return y


def _saturation(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img).astype(np.float64)
    if a.ndim == 2 or a.shape[-1] < 3:
        return np.zeros(a.shape[:2])
    mx = a[..., :3].max(axis=-1)
    mn = a[..., :3].min(axis=-1)
    return (mx - mn) / np.maximum(mx, EPS)


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(1, radius)):
        p = out
        out = (p
               | _shift_zero(p.astype(np.uint8), 1, 0, 0).astype(bool)
               | _shift_zero(p.astype(np.uint8), -1, 0, 0).astype(bool)
               | _shift_zero(p.astype(np.uint8), 0, 1, 0).astype(bool)
               | _shift_zero(p.astype(np.uint8), 0, -1, 0).astype(bool))
    return out


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    inv = ~mask
    return ~_binary_dilate(inv, radius)


def image_shadow_mask(
    img: np.ndarray,
    brightness_thr: float = 0.35,
    saturation_thr: float = 0.40,
    edge_thr: float = 0.05,
    edge_weight: float = 0.35,
    threshold: float = 0.5,
    morph: int = 1,
) -> np.ndarray:
    """从月面图像中提取阴影掩膜（亮度低 + 饱和度低 + 边界锐利）。

    实现为**软打分**而非硬与（AND）：

        score = darkness * (1 - edge_weight * edge_bonus)

      darkness     = 1 - y / brightness_thr               （越暗越接近 1）
      sat_penalty  = 1 - s / saturation_thr               （饱和度越高越惩罚）
      edge_bonus   = 局部存在强梯度的证据（阴影响边界锐利）

    ⚠️ 为什么不用硬 AND：真实/合成的月面阴影常常是**大片连通区域**，其内部梯度为零。
    若要求"每个阴影像素都必须紧邻强边缘"，召回率会崩掉（实测 F1 从 0.99 掉到 0.38）。
    边界锐利度是用来**消歧**的旁证，不能作为必要条件。

    ⚠️ 另一个必须知道的失效模式：低反照率物质（暗色月海、暗色晕、玻璃质碎屑）会被
    误判为阴影。生产环境必须结合几何一致性（与 DEM 预测阴影比对）消歧——这就是
    check 脚本里"用真值掩膜做 F1 评估"要保留的意义。

    Returns: (H, W) uint8，1 = 判为阴影
    """
    y = to_luminance(img)
    s = _saturation(img)

    darkness = np.clip(1.0 - y / max(brightness_thr, EPS), 0.0, 1.0)
    sat_penalty = np.clip(1.0 - s / max(saturation_thr, EPS), 0.0, 1.0)

    gx = _shift_zero(y, 0, 1, 0.0) - _shift_zero(y, 0, -1, 0.0)
    gy = _shift_zero(y, 1, 0, 0.0) - _shift_zero(y, -1, 0, 0.0)
    grad = np.hypot(gx, gy)
    edge_bonus = _binary_dilate(grad > edge_thr, 2).astype(np.float64)

    score = darkness * sat_penalty * (1.0 - edge_weight * (1.0 - edge_bonus))
    shadow = score >= threshold

    if morph > 0:
        shadow = _binary_erode(_binary_dilate(shadow, morph), morph)
    return shadow.astype(np.uint8)


def illumination_invariant(img: np.ndarray, sigma: int = 5) -> np.ndarray:
    """局部亮度归一化（光照不变特征，用于压制跨光照的外观差异）。

    做法：I_norm = I / (I_local_mean + eps)，再裁剪到 [0, 1]。
    用滑动平均实现（O(H*W*sigma)，够用；追求速度可换 scipy.ndimage.uniform_filter）。
    """
    y = to_luminance(img)
    acc = np.zeros_like(y)
    cnt = 0
    for drow in range(-sigma, sigma + 1):
        for dcol in range(-sigma, sigma + 1):
            acc += _shift_zero(y, drow, dcol, 0.0)
            cnt += 1
    local = acc / max(cnt, 1)
    out = y / (local + 1e-3)
    out = np.clip(out / (np.percentile(out, 99) + EPS), 0.0, 1.0)
    return out


# ----------------------------------------------------------------------------
# C. 测量/重建
# ----------------------------------------------------------------------------
def estimate_sun_elevation_from_shadow(
    object_height_m: float,
    shadow_length_px: float,
    gsd_m_per_px: float,
) -> float:
    """由物体高度与其投影长度反推太阳高度角（度）。

        tan(el) = h / L   ->   el = atan(h / L)

    巡视器自身是已知尺寸的标定物，因此可在**无星历**条件下做星上自检。
    """
    L = shadow_length_px * gsd_m_per_px
    if L <= 0:
        raise ValueError("阴影长度必须为正")
    return float(np.rad2deg(np.arctan(object_height_m / L)))


def shadow_length_to_height(
    shadow_length_px: float, gsd_m_per_px: float, el_deg: float
) -> float:
    """由阴影长度与太阳高度角反推物体高度（米）：h = L * tan(el)。"""
    return float(shadow_length_px * gsd_m_per_px * np.tan(np.deg2rad(el_deg)))


def recover_sun_geometry(
    dem: np.ndarray,
    observed_shadow: np.ndarray,
    az_range: tuple[float, float] = (0.0, 360.0),
    el_range: tuple[float, float] = (1.0, 60.0),
    az_step: float = 5.0,
    el_step: float = 1.0,
    cell: float = 1.0,
    tol: int = 1,
    metric: str = "iou",
    coarse_step: float = 3.0,
) -> dict:
    """由阴影掩膜反演太阳的 (方位角, 高度角)。

    这是**闭环自检**的核心：已知真值 -> 生成阴影 -> 反演 -> 看误差。
    做法：粗搜（coarse_step）定位峰值，再在峰值邻域细搜（az_step/el_step）。

    Returns dict: az_deg, el_deg, score, table (az_list, el_list, scores)
    """
    observed = np.asarray(observed_shadow).astype(bool)

    def _scan(azs, els):
        best = {"score": -1.0, "az": None, "el": None}
        scores = np.zeros((len(els), len(azs)))
        for i, el in enumerate(els):
            for j, az in enumerate(azs):
                pred = compute_illumination(dem, az, el, cell=cell) == 0
                sc = match_shadow_masks(pred, observed, tol=tol, metric=metric)
                scores[i, j] = sc
                if sc > best["score"]:
                    best = {"score": sc, "az": float(az), "el": float(el)}
        return best, scores

    azs_c = np.arange(az_range[0], az_range[1], coarse_step)
    els_c = np.arange(el_range[0], el_range[1], coarse_step)
    best_c, _ = _scan(azs_c, els_c)

    az_lo, az_hi = best_c["az"] - coarse_step, best_c["az"] + coarse_step
    el_lo, el_hi = best_c["el"] - coarse_step, best_c["el"] + coarse_step
    azs_f = np.arange(az_lo, az_hi + 1e-9, az_step)
    els_f = np.arange(max(el_range[0], el_lo), min(el_range[1], el_hi) + 1e-9, el_step)
    if len(azs_f) == 0:
        azs_f = np.array([best_c["az"]])
    if len(els_f) == 0:
        els_f = np.array([best_c["el"]])
    best_f, scores_f = _scan(azs_f, els_f)

    return {
        "az_deg": best_f["az"],
        "el_deg": best_f["el"],
        "score": best_f["score"],
        "coarse": best_c,
        "table": {"az": azs_f, "el": els_f, "scores": scores_f},
        "z_az": scores_f.max(axis=0),      # 每个方位角上的最优得分（用于看方位简并）
        "z_el": scores_f.max(axis=1),      # 每个高度角上的最优得分
    }


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def f1_with_tolerance(pred: np.ndarray, gt: np.ndarray, tol: int = 1) -> float:
    """容差 F1：预测与真值各自膨胀 tol 像素后再计算精确率/召回率。

    **这是本任务唯一正确的评估方式**。原因：DEM 合成阴影与真实阴影必然存在
    配准误差与边界不一致，逐像素严格比较测到的是配准误差而不是检测性能。
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if tol > 0:
        p_d, g_d = _binary_dilate(p, tol), _binary_dilate(g, tol)
    else:
        p_d, g_d = p, g
    tp = np.logical_and(p, g_d).sum()
    fp = np.logical_and(p, ~g_d).sum()
    fn = np.logical_and(~p_d, g).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def match_shadow_masks(pred: np.ndarray, gt: np.ndarray, tol: int = 1, metric: str = "iou") -> float:
    if metric == "f1":
        return f1_with_tolerance(pred, gt, tol=tol)
    if metric == "iou":
        p = np.asarray(pred).astype(bool)
        g = np.asarray(gt).astype(bool)
        if tol > 0:
            p, g = _binary_dilate(p, tol), _binary_dilate(g, tol)
        return iou(p, g)
    raise ValueError(f"未知 metric: {metric}")


# ----------------------------------------------------------------------------
# 合成月面地形（自检与演示用）
# ----------------------------------------------------------------------------
def make_synthetic_lunar_dem(
    n: int = 256,
    cell: float = 1.0,
    seed: int = 20261010,
    n_craters: int = 14,
    relief: float = 6.0,
) -> np.ndarray:
    """生成合成月面 DEM：分形噪声底 + 若干带坑缘的陨石坑。

    **固定 seed 保证可复现**——这是要求 5（可复现性）的实现基础。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]

    # 分形噪声（多尺度平滑随机场叠加）
    h = np.zeros((n, n))
    amp, scale = 1.0, 32.0
    for _ in range(6):
        k = max(3, int(scale))
        base = rng.normal(size=(max(4, n // k + 2), max(4, n // k + 2)))
        # 双线性上采样到 n x n
        up = _bilinear_upsample(base, n)
        h += amp * up
        amp *= 0.5
        scale /= 2.0
    h = (h - h.mean()) / (h.std() + EPS) * relief

    # 陨石坑：碗形 + 抬升坑缘
    for _ in range(n_craters):
        cx, cy = rng.uniform(0.15, 0.85, 2) * n
        r = rng.uniform(0.03, 0.12) * n
        d = rng.uniform(0.25, 0.5) * r
        dist = np.hypot(xx - cx, yy - cy)
        bowl = np.clip(1.0 - (dist / r) ** 2, 0.0, 1.0) * d
        rim = np.exp(-((dist - r) ** 2) / (2 * (0.18 * r) ** 2)) * 0.45 * d
        h += -bowl + rim

    return (h * cell).astype(np.float64)


def _bilinear_upsample(a: np.ndarray, n: int) -> np.ndarray:
    h, w = a.shape
    ri = np.linspace(0, h - 1, n)
    ci = np.linspace(0, w - 1, n)
    r0 = np.clip(np.floor(ri).astype(int), 0, h - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c0 = np.clip(np.floor(ci).astype(int), 0, w - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    wr = (ri - r0)[:, None]
    wc = (ci - c0)[None, :]
    top = a[r0][:, c0] * (1 - wc) + a[r0][:, c1] * wc
    bot = a[r1][:, c0] * (1 - wc) + a[r1][:, c1] * wc
    return top * (1 - wr) + bot * wr
