# -*- coding: utf-8 -*-
"""
lunar_shadow.py —— a runnable implementation of lunar-surface illumination/shadow algorithms
=================================================

The code companion to the document "Lunar Surface Illumination and Shadow Algorithms: A Guide".
It contains three families of algorithms:

  A. Forward: DEM + sun position -> illumination/shadow mask
     - compute_illumination()   ray marching, including the local slope-orientation test
     - horizon_map()            horizon map; precompute once and sweep any sun azimuth
     - illuminate_from_horizon() horizon-map lookup for the illumination mask (O(1) per query)
     - solar_position_analytic() analytic approximation of the lunar solar position (teaching/simulation)
     - dem_to_lonlat()          simple equidistant row/col <-> lon/lat projection (teaching)
     - subsolar_lonlat_analytic() sub-solar point lon/lat (teaching)

  B. Inverse: image -> shadow mask
     - image_shadow_mask()      brightness + saturation dual threshold + morphological cleanup
     - illumination_invariant() local brightness normalisation; suppresses illumination changes

  C. Measurement/reconstruction: shadow -> geometry / illumination parameters
     - estimate_sun_elevation_from_shadow()  shadow length ratio -> solar elevation angle
     - recover_sun_geometry()   invert a shadow mask for (azimuth, elevation)
     - shadow_length_to_height() shadow length -> object height

  Evaluation: iou / f1_with_tolerance / match_shadow_masks

Dependencies: numpy (required), scipy (optional, only for faster morphology and filtering)
Run the self-check: python check_lunar_shadow.py

Coordinate conventions (uniform throughout — memorise these)
-----------------------------
  * dem is a (H, W) numpy array in metres, dem[row, col].
  * Image coordinate system: increasing col = east, decreasing row = north.
  * The azimuth az_deg uses compass bearings: 0°=north, 90°=east, 180°=south, 270°=west.
  * Solar azimuth = the compass bearing of the Sun as seen from that point; solar elevation el_deg = 0° at the horizon, 90° at the zenith.
  * Direction vector (image coordinates): dcol = +sin(az)*cos(el), drow = -cos(az)*cos(el).

Easter egg (HARUHIKAGE)
-----------------------
  The Moon has no air, so the terminator of a shadow there is not a gradient but an edge.
  Everything below exists to answer one question: where exactly does that edge fall?
  Written while listening to the opening of "Haruhikage" (春日影, CRYCHIC) on repeat.
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
# Basic utilities
# ----------------------------------------------------------------------------
def sun_direction(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Unit direction vector **pointing at the Sun** (image coordinates + height).

    Returns (dcol, drow, dz), where dcol is positive towards the east and drow is negative
    towards the north (because row increases downwards).
    """
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return (np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el))


def horizontal_direction(az_deg: float) -> tuple[float, float]:
    """**Unit** direction (dcol, drow) in the horizontal plane along azimuth az.

    This is factored out to guard against a pitfall we really did hit: using
    sun_direction(az, 90.0) to get a "horizontal direction" makes cos(90°)=0 collapse the
    vector to zero, so the whole horizon computation fails silently (everything keeps its
    initial value).
    """
    az = np.deg2rad(az_deg)
    return (float(np.sin(az)), float(-np.cos(az)))


def terrain_gradient(dem: np.ndarray, cell: float = 1.0):
    """Returns (gx, gy, slope_rad).
    gx = d z / d east (m/m), gy = d z / d north (m/m).
    """
    drow, dcol = np.gradient(dem, cell)   # np.gradient returns (axis0, axis1)
    gx, gy = dcol, -drow                  # row increases southward -> negate for north
    slope = np.arctan(np.hypot(gx, gy))
    return gx, gy, slope


def _shift_zero(a: np.ndarray, drow: int, dcol: int, fill: float) -> np.ndarray:
    """Zero-padded shift: out[i, j] = a[i - drow, j - dcol], out-of-bounds filled with fill.

    This is the key utility of this module: **zero padding is mandatory; np.roll is not allowed**.
    np.roll is a cyclic wrap-around, which produces phantom occlusion at image borders.
    """
    H, W = a.shape

    def _sl(start: int, shift: int, n: int) -> slice:
        """Slice of the interval [start, start+n) after subtracting shift (for destination or source)."""
        return slice(start - shift, start - shift + n)

    rows_dst = slice(max(0, drow), H + min(0, drow))
    cols_dst = slice(max(0, dcol), W + min(0, dcol))
    rows_src = _sl(max(0, drow), drow, rows_dst.stop - rows_dst.start)
    cols_src = _sl(max(0, dcol), dcol, cols_dst.stop - cols_dst.start)

    out = np.full((H, W), fill, dtype=np.float64)
    out[rows_dst, cols_dst] = a[rows_src, cols_src]
    return out


# ----------------------------------------------------------------------------
# A. Forward: DEM -> illumination/shadow
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
    """Compute an illumination mask by ray marching.

    For every pixel, step outward along the direction **opposite** to the Sun and test whether
    the terrain blocks the Sun. The predicate at step k is:
        z(neighbour) > z(this point) + k * cell * tan(el)
    which is equivalent to "the horizon elevation angle of that neighbour exceeds the solar
    elevation angle".

    Parameters
    ----------
    dem : (H, W) elevation, metres
    az_deg, el_deg : solar compass azimuth / elevation angle (degrees)
    cell : grid cell size, metres
    max_steps : maximum number of outward steps; defaults to 2*max(H, W)
    use_slope_term : whether to add the "local slope faces away from the Sun" test
        (attached shadow). Solar elevation in the lunar polar regions is extremely low and
        the area fraction of slopes facing away from the Sun is large, so turning this on is
        strongly recommended.
    min_cos_incidence : cosine threshold for the local incidence angle; below it the surface
        counts as facing away (default 0, i.e. only >=90° counts as facing away)

    Returns
    -------
    illum : (H, W) uint8, 1 = lit by the Sun, 0 = in shadow
    """
    dem = np.asarray(dem, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError("dem must be a 2-D array")
    H, W = dem.shape
    if max_steps is None:
        max_steps = int(2 * max(H, W))

    dcol, drow, dz = sun_direction(az_deg, el_deg)
    tan_el = dz / max(np.hypot(dcol, drow), EPS)   # = tan(el)

    illum = np.ones((H, W), dtype=bool)

    # 1) Local slope orientation (attached shadow): slopes facing away receive no direct sunlight
    if use_slope_term:
        gx, gy, _ = terrain_gradient(dem, cell)
        # surface normal n ∝ (-gx, -gy, 1); cosine with the sun direction
        cos_inc = (-gx * dcol - gy * drow + dz) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        illum &= cos_inc > min_cos_incidence

    # 2) Terrain cast shadow
    #
    # The predicate (mind the direction!):
    #   "who blocks the Sun" — the occluder is on the **Sun's side**. sun_direction() returns
    #   the unit vector **pointing at the Sun**, so we step outward along -(dcol, drow) to
    #   sample the terrain; the shadow itself falls on the **opposite side** from the Sun.
    #
    #   At step k (horizontal distance d = k*cell), the height of the sun ray relative to this
    #   point is
    #       ray_h = z(this point) + d * tan(el)
    #   The terrain there blocks the Sun if and only if it rises above that ray:
    #       z(sun-side neighbour) >= ray_h
    #   which is equivalent to "the horizon elevation angle of that neighbour relative to this
    #   point >= the solar elevation angle".
    #   On flat terrain the left-hand side is 0 < d*tan(el), so no shadow is produced
    #   (a sanity check).
    alive = illum.copy()
    for k in range(1, max_steps + 1):
        srow = int(round(-k * drow / cell))      # note the minus sign: sample towards the Sun
        scol = int(round(-k * dcol / cell))      # note the minus sign: sample towards the Sun
        if abs(srow) >= H or abs(scol) >= W:
            break
        if srow == 0 and scol == 0:
            continue
        shifted = _shift_zero(dem, srow, scol, fill=-np.inf)
        # Key point: at step k we must not test the single distance k, because when srow/scol
        # repeat due to rounding (e.g. k from 2 to 40 all map to the same cell) that cell keeps
        # being tested at the same angle, which systematically misses shadows. The correct
        # approach is to cover the **entire distance interval** that cell spans: if there exists
        # d ∈ [k-0.5, k+0.5] where the terrain blocks the ray, it counts as shadow. k-0.5 is the
        # most conservative choice, because that ray is the lowest and easiest to block.
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
    """Precompute the horizon map.

    H(i, j, m) = the **maximum elevation angle** (degrees) of the terrain relative to pixel
    (i, j), along the direction of azimuth m **pointing at the Sun**. With it, "is the Sun
    blocked?" degenerates into a single lookup:

        shaded  <=>  el_sun < H(i, j, az)

    Changing the sun azimuth requires no recomputation — its biggest advantage over ray
    marching.

    ⚠️ Azimuth convention: az here is the **Sun's compass azimuth** (consistent with
    compute_illumination), so the internal sampling direction is -(sin az, -cos az) (i.e.
    pointing at the Sun). An early version mistakenly used the "look outward along az"
    direction (off by 180°), which put shadows on the same side as the Sun — very hard to
    catch by code review alone; you have to pin it down with an analytic case such as
    "north–south ridge + sun due east/west".

    Parameters
    ----------
    n_az : number of azimuth bins; 360/n_az is the angular resolution. Default 72 -> 5°
    max_dist : maximum probing distance (metres); defaults to 2*max(H, W)*cell

    Returns
    -------
    Hmap : (n_az, H, W) float32, in degrees
    """
    dem = np.asarray(dem, dtype=np.float64)
    H, W = dem.shape
    if max_dist is None:
        max_dist = 2.0 * max(H, W) * cell
    max_steps = int(max_dist / cell) + 1

    az_list = np.arange(n_az) * (360.0 / n_az)
    Hmap = np.zeros((n_az, H, W), dtype=np.float32)

    for m, az_deg in enumerate(az_list):
        # unit horizontal direction pointing at the Sun (must NOT be horizontal_direction(az_deg): off by 180°)
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
    """Get an illumination mask by horizon-map lookup with linear interpolation (no re-marching)."""
    dem = np.asarray(dem, dtype=np.float64)
    n_az = Hmap.shape[0]
    step = 360.0 / n_az
    # azimuth -> bin (cyclic interpolation)
    t = (az_deg % 360.0) / step
    m0 = int(np.floor(t)) % n_az
    m1 = (m0 + 1) % n_az
    w = t - np.floor(t)
    H_interp = (1.0 - w) * Hmap[m0] + w * Hmap[m1]

    illum = (el_deg >= H_interp)

    if use_slope_term:
        gx, gy, _ = terrain_gradient(dem, cell)
        dcol, drow, dz = sun_direction(az_deg, el_deg)
        # surface normal n ∝ (-gx, -gy, 1); its dot product with the **Sun-pointing** direction is the incidence cosine
        cos_inc = (-gx * dcol - gy * drow + dz) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        illum &= cos_inc > 0.0

    return illum.astype(np.uint8)


# ----------------------------------------------------------------------------
# A'. Solar position and projection (teaching/simulation; use SPICE for high precision)
# ----------------------------------------------------------------------------
def solar_position_analytic(
    jd: float,
    lon_deg: float,
    lat_deg: float,
    body: str = "moon",
) -> tuple[float, float]:
    """Low-precision analytic solar compass azimuth and elevation angle (degrees).

    ⚠️ Accuracy note: this uses a simplified model — a low-precision series for the Sun's
    apparent ecliptic longitude plus the Moon's Cassini-state rotation — with typical errors
    on the order of 0.1°–0.5°. That is **enough to support teaching and simulation**, but
    **not enough to support mission-level planning**. For mission level use SPICE (spiceypy):
    load a lunar body-fixed orientation kernel + a planetary ephemeris kernel, compute the
    Sun's direction relative to the Moon's centre, rotate it into the IAU_MOON body-fixed
    frame, then into the local topocentric frame.

    Lunar rotation: Cassini state, rotation period = orbital period ≈ 27.32166 d
    (synchronous rotation).
    """
    # Julian centuries
    T = (jd - 2451545.0) / 36525.0
    # geometric mean longitude and mean anomaly of the Sun (Meeus, low precision)
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    Mr = np.deg2rad(M % 360.0)
    # equation of the centre
    C = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * np.sin(Mr)
         + (0.019993 - 0.000101 * T) * np.sin(2 * Mr)
         + 0.000289 * np.sin(3 * Mr))
    true_lon = L0 + C
    # obliquity of the ecliptic
    eps = 23.439291 - 0.0130042 * T
    lam = np.deg2rad(true_lon % 360.0)
    epsr = np.deg2rad(eps)
    # solar right ascension and declination (geocentric equatorial frame)
    ra = np.arctan2(np.cos(epsr) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(epsr) * np.sin(lam))

    if body.lower() != "moon":
        raise NotImplementedError("only the Moon is implemented so far")

    # Moon body-fixed frame: the Z axis points at the lunar north pole; the prime meridian is
    # approximated by the Cassini state. Direction of the lunar north pole in ICRF
    # (approximate, J2000): ra_pole ≈ 269.9949°, dec_pole ≈ 66.5392°
    ra_p = np.deg2rad(269.9949)
    dec_p = np.deg2rad(66.5392)
    # unit vector towards the Sun (ICRF)
    s_vec = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    z_axis = np.array([np.cos(dec_p) * np.cos(ra_p), np.cos(dec_p) * np.sin(ra_p), np.sin(dec_p)])
    x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    s_body = np.array([s_vec @ x_axis, s_vec @ y_axis, s_vec @ z_axis])

    # sub-solar point lon/lat
    sub_lat = np.rad2deg(np.arcsin(np.clip(s_body[2], -1, 1)))
    sub_lon = np.rad2deg(np.arctan2(s_body[1], s_body[0]))

    return _sun_az_el_from_subsolar(lon_deg, lat_deg, sub_lon, sub_lat)


def _sun_az_el_from_subsolar(
    lon_deg: float, lat_deg: float, sub_lon_deg: float, sub_lat_deg: float
) -> tuple[float, float]:
    """Solar compass azimuth and elevation angle (degrees) from the sub-solar point and the site lon/lat."""
    lam = np.deg2rad(lon_deg)
    phi = np.deg2rad(lat_deg)
    lams = np.deg2rad(sub_lon_deg)
    phis = np.deg2rad(sub_lat_deg)
    dlam = lams - lam

    sin_el = np.sin(phi) * np.sin(phis) + np.cos(phi) * np.cos(phis) * np.cos(dlam)
    el = np.arcsin(np.clip(sin_el, -1, 1))
    # azimuth (from north towards east)
    num = np.sin(dlam) * np.cos(phis)
    den = np.cos(phi) * np.sin(phis) - np.sin(phi) * np.cos(phis) * np.cos(dlam)
    az = np.rad2deg(np.arctan2(num, den)) % 360.0
    return float(az), float(np.rad2deg(el))


def sun_az_el_from_subsolar(
    lon_deg: float, lat_deg: float, sub_lon_deg: float, sub_lat_deg: float
) -> tuple[float, float]:
    """Public alias: solar compass azimuth and elevation angle (degrees) from the sub-solar point and the site lon/lat."""
    return _sun_az_el_from_subsolar(lon_deg, lat_deg, sub_lon_deg, sub_lat_deg)


def subsolar_lonlat_analytic(jd: float) -> tuple[float, float]:
    """Low-precision sub-solar point lon/lat (degrees). ⚠️ Likewise only suitable for teaching/simulation."""
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
    """Row/col -> lon/lat (simple equidistant projection, for teaching)."""
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
    """Given the DEM's georeference and a time, return the Sun's (azimuth, elevation) at that point.

    jd is the Julian date; when None, a fixed example value is used (demonstration only).
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
# B. Inverse: image -> shadow
# ----------------------------------------------------------------------------
def to_luminance(img: np.ndarray, normalize: bool = False) -> np.ndarray:
    """RGB/RGBA/grayscale -> floating-point luminance.

    normalize=False (the default) keeps the absolute brightness scale [0,1] — a thresholding
    method must do this, because "dark" is an absolute notion. normalize=True rescales the
    maximum to 1 and is for display only.
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
    """Extract a shadow mask from a lunar-surface image (low brightness + low saturation +
    sharp boundary).

    Implemented as **soft scoring** rather than a hard AND:

        score = darkness * (1 - edge_weight * edge_bonus)

      darkness     = 1 - y / brightness_thr               (the darker, the closer to 1)
      sat_penalty  = 1 - s / saturation_thr               (the more saturated, the heavier the penalty)
      edge_bonus   = evidence of a strong local gradient (shadow boundaries are sharp)

    ⚠️ Why not a hard AND: real/synthetic lunar shadows are often **large connected regions**
    whose interior gradient is zero. Demanding that "every shadow pixel must sit next to a
    strong edge" destroys recall (measured F1 drops from 0.99 to 0.38). Boundary sharpness is
    **disambiguating** circumstantial evidence, never a necessary condition.

    ⚠️ Another failure mode you must know about: low-albedo materials (dark mare, dark halos,
    glassy fragments) get misclassified as shadow. Production use must disambiguate with
    geometric consistency (comparing against DEM-predicted shadows) — that is exactly why the
    "F1 evaluation against a ground-truth mask" in the check script has to stay.

    Returns: (H, W) uint8, 1 = classified as shadow
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
    """Local brightness normalisation (an illumination-invariant feature, to suppress
    appearance differences across illumination).

    Recipe: I_norm = I / (I_local_mean + eps), then clip to [0, 1].
    Implemented with a moving average (O(H*W*sigma), good enough; swap in
    scipy.ndimage.uniform_filter if you want speed).
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
# C. Measurement/reconstruction
# ----------------------------------------------------------------------------
def estimate_sun_elevation_from_shadow(
    object_height_m: float,
    shadow_length_px: float,
    gsd_m_per_px: float,
) -> float:
    """Invert the solar elevation angle (degrees) from an object's height and its projected length.

        tan(el) = h / L   ->   el = atan(h / L)

    The rover itself is a calibration target of known size, so this enables an on-board
    self-check with **no ephemeris**.
    """
    L = shadow_length_px * gsd_m_per_px
    if L <= 0:
        raise ValueError("shadow length must be positive")
    return float(np.rad2deg(np.arctan(object_height_m / L)))


def shadow_length_to_height(
    shadow_length_px: float, gsd_m_per_px: float, el_deg: float
) -> float:
    """Invert object height (metres) from the shadow length and the solar elevation angle: h = L * tan(el)."""
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
    """Invert the Sun's (azimuth, elevation) from a shadow mask.

    This is the core of the **closed-loop self-check**: ground truth -> generate shadow ->
    invert -> inspect the error.
    Method: a coarse search (coarse_step) localises the peak, then a fine search
    (az_step/el_step) runs in the peak's neighbourhood.

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
        "z_az": scores_f.max(axis=0),      # best score at each azimuth (to inspect azimuth degeneracy)
        "z_el": scores_f.max(axis=1),      # best score at each elevation
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def f1_with_tolerance(pred: np.ndarray, gt: np.ndarray, tol: int = 1) -> float:
    """Tolerance-based F1: dilate prediction and ground truth by tol pixels each, then compute
    precision/recall.

    **This is the only correct way to evaluate this task.** Reason: DEM-synthesised shadows and
    real shadows inevitably differ by registration error and boundary mismatch, so a strict
    per-pixel comparison measures registration error rather than detection performance.
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
    raise ValueError(f"unknown metric: {metric}")


# ----------------------------------------------------------------------------
# Synthetic lunar terrain (for the self-check and demos)
# ----------------------------------------------------------------------------
def make_synthetic_lunar_dem(
    n: int = 256,
    cell: float = 1.0,
    seed: int = 20261010,
    n_craters: int = 14,
    relief: float = 6.0,
) -> np.ndarray:
    """Generate a synthetic lunar DEM: a fractal-noise base + several craters with rims.

    **A fixed seed guarantees reproducibility** — the basis for requirement 5 (reproducibility).

    The default seed is the date this file was written (2026-10-10). Change it if you like:
    the terrain is not the point, the shadow it casts is. Every crater rim here exists to
    throw one hard edge across the ground — the way the opening bars of "Haruhikage"
    (春日影) throw their first chord and then wait to see what you do with it.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]

    # fractal noise (a sum of multi-scale smooth random fields)
    h = np.zeros((n, n))
    amp, scale = 1.0, 32.0
    for _ in range(6):
        k = max(3, int(scale))
        base = rng.normal(size=(max(4, n // k + 2), max(4, n // k + 2)))
        # bilinear upsample to n x n
        up = _bilinear_upsample(base, n)
        h += amp * up
        amp *= 0.5
        scale /= 2.0
    h = (h - h.mean()) / (h.std() + EPS) * relief

    # craters: bowl shape + raised rim
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
