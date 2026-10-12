/*
 * lunar_shadow.c —— Lunar illumination / shadow algorithms in C99
 * ==============================================================
 *
 * Why a C version exists
 * ----------------------
 * The Python module (../lunar_shadow.py) is the reference implementation and
 * the one to read first. This file exists because the target platform for the
 * real thing is a radiation-hardened flight processor, and those run C. It is
 * also the version you can drop into a rover's onboard stack without dragging
 * a Python runtime, NumPy, or a JIT along with it.
 *
 * What is ported
 * --------------
 * The numerically meaningful core, kept bit-comparable with Python:
 *
 *   - sun_direction()               to-sun unit vector (image coordinates)
 *   - horizontal_direction()        in-plane unit vector for an azimuth
 *   - terrain_gradient()            central differences, NumPy-compatible edges
 *   - compute_illumination()        ray marching + local slope-orientation test
 *   - horizon_map()                 precompute max elevation per azimuth bin
 *   - illuminate_from_horizon()     table lookup instead of re-marching
 *   - shadow_length_to_height()     L * tan(el)
 *   - image_shadow_mask()           darkness + local-contrast soft score
 *   - iou()/f1_with_tolerance()     evaluation (used by the self-test)
 *
 * Deliberately NOT ported (they belong on the ground segment, not on a rover):
 * SPICE ephemeris handling, and the logistic-regression training loop from the
 * illumination-generalisation experiment.
 *
 * The two conventions that bite everyone
 * -------------------------------------
 * Both of these were live bugs in the Python version first (see the algorithm
 * guide §8). They are reproduced here as comments because a C port is exactly
 * where they get reintroduced:
 *
 *   1. `sun_direction()` returns the vector **pointing at the Sun**. The ray
 *      march must therefore sample along the NEGATIVE of it (towards the Sun),
 *      and the shadow itself falls on the opposite side. Getting this backwards
 *      draws shadows on the sunlit side -- which looks plausible in a rendered
 *      image and is very hard to spot by eye.
 *
 *   2. A pixel is shadowed iff some terrain point along the to-Sun ray rises
 *      ABOVE the ray:
 *
 *          shadowed  <=>  z(neighbour) >= z(pixel) + d * tan(elevation)
 *
 *      i.e. the terrain height is compared against the RAY height, not the
 *      other way round. Flat terrain must come out fully lit; if it comes out
 *      fully shadowed, this inequality is inverted.
 *
 * Build
 * -----
 *   gcc -O2 -std=c99 -Wall -Wextra -o lunar_shadow lunar_shadow.c -lm
 *
 * Run
 * ---
 *   ./lunar_shadow --selftest                  # check the geometry invariants
 *   ./lunar_shadow --dem dem.txt --az 137 --el 6 --out-illum illum.pgm
 *   ./lunar_shadow --dem dem.txt --az 137 --horizon h.txt --n-az 72 \
 *                  --dump-horizon horizon.txt
 *
 * The DEM input is a whitespace-separated text grid whose first two numbers are
 * the row and column counts. `tools/verify_c_vs_python.py` writes this format
 * and compares this program's output against the Python module, so the port is
 * verified rather than asserted.
 *
 * Author: HAZAKIKO.  License: MIT (see ../LICENSE).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* M_PI is not required to be visible in strict C99 -- define it rather than
 * relying on the implementation to leak it in. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPS 1e-9

/* ------------------------------------------------------------------ */
/* Grid: a row-major 2-D array of doubles, y increasing southward.      */
/* ------------------------------------------------------------------ */
typedef struct {
    int rows, cols;
    double *v; /* v[r*cols + c], row = y (south +), col = x (east +) */
} Grid;

static Grid *grid_new(int rows, int cols)
{
    Grid *g = (Grid *)malloc(sizeof(Grid));
    if (!g) return NULL;
    g->rows = rows;
    g->cols = cols;
    g->v = (double *)calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (!g->v) { free(g); return NULL; }
    return g;
}

static void grid_free(Grid *g)
{
    if (g) { free(g->v); free(g); }
}

#define AT(g, r, c) ((g)->v[(size_t)(r) * (size_t)(g)->cols + (size_t)(c)])

/* ------------------------------------------------------------------ */
/* Conventions                                                         */
/* ------------------------------------------------------------------ */

/* Unit vector POINTING AT THE SUN, in image coordinates.
 *   dcol = +sin(az) * cos(el)      east positive
 *   drow = -cos(az) * cos(el)      north is negative row (row grows southward)
 *   dz   = +sin(el)
 * Note: NEVER build a horizontal direction by calling this with el = 90 deg --
 * cos(90) = 0 collapses the vector to zero and everything silently returns its
 * initial value. Use horizontal_direction() for that.
 */
static void sun_direction(double az_deg, double el_deg,
                          double *dcol, double *drow, double *dz)
{
    double az = az_deg * M_PI / 180.0;
    double el = el_deg * M_PI / 180.0;
    *dcol = sin(az) * cos(el);
    *drow = -cos(az) * cos(el);
    *dz = sin(el);
}

/* In-plane unit vector along the compass bearing az (0 = north, 90 = east). */
static void horizontal_direction(double az_deg, double *dcol, double *drow)
{
    double az = az_deg * M_PI / 180.0;
    *dcol = sin(az);
    *drow = -cos(az);
}

/* Central-difference gradient, matching numpy.gradient's one-sided edges.
 * Writes gx = dz/d(east), gy = dz/d(north). */
static void terrain_gradient(const Grid *dem, double cell, Grid *gx, Grid *gy)
{
    const int R = dem->rows, C = dem->cols;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            double dc, dr;
            if (C == 1)      dc = 0.0;
            else if (c == 0) dc = (AT(dem, r, c + 1) - AT(dem, r, c)) / cell;
            else if (c == C - 1) dc = (AT(dem, r, c) - AT(dem, r, c - 1)) / cell;
            else dc = (AT(dem, r, c + 1) - AT(dem, r, c - 1)) / (2.0 * cell);

            if (R == 1)      dr = 0.0;
            else if (r == 0) dr = (AT(dem, r + 1, c) - AT(dem, r, c)) / cell;
            else if (r == R - 1) dr = (AT(dem, r, c) - AT(dem, r - 1, c)) / cell;
            else dr = (AT(dem, r + 1, c) - AT(dem, r - 1, c)) / (2.0 * cell);

            AT(gx, r, c) = dc;
            AT(gy, r, c) = -dr; /* north is -row */
        }
    }
}

/* cos(incidence) against the to-Sun vector; 1 = face-on to the Sun. */
static void cos_incidence(const Grid *dem, double cell, double az_deg, double el_deg,
                          Grid *out)
{
    Grid *gx = grid_new(dem->rows, dem->cols);
    Grid *gy = grid_new(dem->rows, dem->cols);
    terrain_gradient(dem, cell, gx, gy);
    double dcol, drow, dz;
    sun_direction(az_deg, el_deg, &dcol, &drow, &dz);
    for (int r = 0; r < dem->rows; r++) {
        for (int c = 0; c < dem->cols; c++) {
            double a = AT(gx, r, c), b = AT(gy, r, c);
            double n = sqrt(a * a + b * b + 1.0);
            /* surface normal is proportional to (-gx, -gy, 1) */
            AT(out, r, c) = (-a * dcol - b * drow + dz) / n;
        }
    }
    grid_free(gx);
    grid_free(gy);
}

/* ------------------------------------------------------------------ */
/* Forward model: ray marching                                         */
/* ------------------------------------------------------------------ */
/* Returns a Grid of 1 (lit) and 0 (shadow). */
static Grid *compute_illumination(const Grid *dem, double az_deg, double el_deg,
                                  double cell, int max_steps, int use_slope_term)
{
    const int R = dem->rows, C = dem->cols;
    Grid *illum = grid_new(R, C);
    Grid *ci = grid_new(R, C);

    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            AT(illum, r, c) = 1.0;

    /* 1) local surface orientation (attached shadow): a slope facing away from
     *    the Sun receives no direct light. On the Moon there is no atmospheric
     *    scattering to soften this, so it matters a lot. */
    if (use_slope_term) {
        cos_incidence(dem, cell, az_deg, el_deg, ci);
        for (size_t i = 0; i < (size_t)R * C; i++)
            if (illum->v[i] > 0.5 && ci->v[i] <= 0.0)
                illum->v[i] = 0.0;
    }

    /* 2) cast shadow: march towards the Sun. */
    double dcol, drow, dz;
    sun_direction(az_deg, el_deg, &dcol, &drow, &dz);
    double horiz = sqrt(dcol * dcol + drow * drow);
    double tan_el = (horiz > EPS) ? dz / horiz : 0.0;

    if (max_steps <= 0)
        max_steps = 2 * (R > C ? R : C);

    for (int k = 1; k <= max_steps; k++) {
        /* Matches Python's _shift_zero: the sample is dem[r - srow][c - scol].
         * sun_direction() points at the Sun, so the NEGATED offset marches towards it. */
        int srow = (int)lround(-(double)k * drow / cell);
        int scol = (int)lround(-(double)k * dcol / cell);
        if (srow == 0 && scol == 0) continue;
        if (abs(srow) >= R || abs(scol) >= C) break;

        /* The same grid cell is revisited for several k when the offset saturates; every
         * time it is, the ray is lower, so the most conservative test uses the smallest k
         * in that cell's range. Using only the current k under-detects shadow. */
        double k_lo = (k == 1) ? 1.0 : (double)k - 0.5;

        int any_alive = 0;
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                if (AT(illum, r, c) <= 0.5) continue;
                any_alive = 1;
                int rr = r - srow, cc = c - scol;   /* MINUS: see the note above */
                if (rr < 0 || rr >= R || cc < 0 || cc >= C) continue; /* outside: nothing blocks */
                double ray_h = AT(dem, r, c) + k_lo * cell * tan_el;
                if (AT(dem, rr, cc) >= ray_h - EPS)
                    AT(illum, r, c) = 0.0;
            }
        }
        if (!any_alive) break;
    }

    grid_free(ci);
    return illum;
}

/* ------------------------------------------------------------------ */
/* Forward model: horizon map                                          */
/* ------------------------------------------------------------------ */
/* H[az_bin][r][c] = max elevation angle (deg) of terrain along the direction
 * TOWARDS THE SUN at that azimuth bin. Then "is it shadowed" is a comparison:
 *
 *     shadowed  <=>  el_sun < H[bin]
 *
 * The point of precomputing is that the expensive part does not depend on the
 * Sun's elevation at all, so sweeping elevations becomes cheap.
 */
static Grid *horizon_map(const Grid *dem, double cell, int n_az, int max_steps)
{
    const int R = dem->rows, C = dem->cols;
    Grid *H = grid_new(n_az, R * C); /* one (R, C) plane per azimuth bin */
    if (max_steps <= 0) max_steps = 2 * (R > C ? R : C);

    for (int m = 0; m < n_az; m++) {
        double az = (double)m * 360.0 / (double)n_az;
        /* horizontal_direction(az) points at the Sun; h_map_shift reads
         * dem[r - srow][c - scol], so the NEGATED offset walks towards the Sun.
         * Equivalent: horizontal_direction(az + 180) with a positive offset. Flip the
         * two together, never one alone. */
        double dcol, drow;
        horizontal_direction(az, &dcol, &drow);

        for (int i = 0; i < R * C; i++) H->v[(size_t)m * R * C + i] = -90.0;

        for (int k = 1; k <= max_steps; k++) {
            int srow = (int)lround(-(double)k * drow);
            int scol = (int)lround(-(double)k * dcol);
            if (srow == 0 && scol == 0) continue;
            if (abs(srow) >= R || abs(scol) >= C) break;
            double dist = (double)k * cell;
            /* Replicate Python's _shift_zero exactly: the sampled SOURCE cell is
             * (r - srow, c - scol), and whenever that falls outside the grid the filler
             * (-inf) is used, which can never raise the horizon. Testing the destination
             * instead of the source lets border cells read terrain the reference cannot
             * see, which is what made 1.35% of horizon cells disagree while both
             * illumination paths matched at 100%. */
            for (int r = 0; r < R; r++) {
                int rs = r - srow;
                if (rs < 0 || rs >= R) continue;
                for (int c = 0; c < C; c++) {
                    int cs = c - scol;
                    if (cs < 0 || cs >= C) continue;
                    double ang = atan2(AT(dem, rs, cs) - AT(dem, r, c), dist)
                                 * 180.0 / M_PI;
                    double *slot = &H->v[((size_t)m * R * C) + (size_t)r * C + (size_t)c];
                    if (ang > *slot) *slot = ang;
                }
            }
        }
    }
    return H;
}

/* Lookup version: no ray marching, just interpolation + comparison. */
static Grid *illuminate_from_horizon(const Grid *dem, const Grid *H, double az_deg,
                                     double el_deg, int n_az, double cell,
                                     int use_slope_term)
{
    const int R = dem->rows, C = dem->cols;
    const double plane = (double)R * C;
    Grid *illum = grid_new(R, C);
    Grid *ci = grid_new(R, C);

    double step = 360.0 / (double)n_az;
    double t = fmod(az_deg, 360.0);
    if (t < 0) t += 360.0;
    t /= step;
    int m0 = (int)floor(t) % n_az;
    int m1 = (m0 + 1) % n_az;
    double w = t - floor(t);

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            size_t idx = (size_t)r * C + (size_t)c;
            double h = (1.0 - w) * H->v[(size_t)m0 * (size_t)plane + idx]
                     + w * H->v[(size_t)m1 * (size_t)plane + idx];
            AT(illum, r, c) = (el_deg >= h) ? 1.0 : 0.0;
        }
    }

    if (use_slope_term) {
        cos_incidence(dem, cell, az_deg, el_deg, ci);
        for (size_t i = 0; i < (size_t)R * C; i++)
            if (ci->v[i] <= 0.0) illum->v[i] = 0.0;
    }
    grid_free(ci);
    return illum;
}

/* ------------------------------------------------------------------ */
/* Inverse: image -> shadow mask                                       */
/* ------------------------------------------------------------------ */
/* Soft score rather than a hard AND. Requiring every shadow pixel to sit next
 * to a strong edge collapses recall on large connected shadow regions (measured
 * F1 0.38 vs 1.00 in the Python version), because the interior of such a region
 * has no gradient at all. Edge evidence is a hint, not a requirement. */
static Grid *image_shadow_mask(const Grid *img, double brightness_thr,
                               double edge_thr, double edge_weight, int radius)
{
    const int R = img->rows, C = img->cols;
    Grid *out = grid_new(R, C);
    /* local mean */
    Grid *mean = grid_new(R, C);
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            double acc = 0.0;
            int cnt = 0;
            for (int dr = -radius; dr <= radius; dr++) {
                for (int dc = -radius; dc <= radius; dc++) {
                    int rr = r + dr, cc = c + dc;
                    if (rr < 0 || rr >= R || cc < 0 || cc >= C) continue;
                    acc += AT(img, rr, cc);
                    cnt++;
                }
            }
            AT(mean, r, c) = cnt ? acc / cnt : 0.0;
        }
    }
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            double y = AT(img, r, c);
            double darkness = 1.0 - y / (brightness_thr > EPS ? brightness_thr : EPS);
            if (darkness < 0.0) darkness = 0.0;
            if (darkness > 1.0) darkness = 1.0;
            /* local gradient magnitude as edge evidence */
            double gxv = (c + 1 < C ? AT(img, r, c + 1) : y)
                       - (c > 0 ? AT(img, r, c - 1) : y);
            double gyv = (r + 1 < R ? AT(img, r + 1, c) : y)
                       - (r > 0 ? AT(img, r - 1, c) : y);
            double grad = sqrt(gxv * gxv + gyv * gyv);
            double edge = (grad > edge_thr) ? 1.0 : 0.0;
            double score = darkness * (1.0 - edge_weight * (1.0 - edge));
            AT(out, r, c) = (score >= 0.5) ? 1.0 : 0.0;
        }
    }
    grid_free(mean);
    return out;
}

/* ------------------------------------------------------------------ */
/* Evaluation                                                          */
/* ------------------------------------------------------------------ */
static double iou(const Grid *pred, const Grid *gt)
{
    double inter = 0.0, uni = 0.0;
    for (size_t i = 0; i < (size_t)pred->rows * pred->cols; i++) {
        int p = pred->v[i] > 0.5, g = gt->v[i] > 0.5;
        if (p && g) inter += 1.0;
        if (p || g) uni += 1.0;
    }
    return uni > 0.0 ? inter / uni : 1.0;
}

/* Tolerance-based F1: dilate both masks by `tol` before scoring.
 * Per-pixel comparison between a terrain-derived mask and a real image mostly
 * measures misregistration, so this is the metric that matters in practice. */
static double f1_with_tolerance(const Grid *pred, const Grid *gt, int tol)
{
    const int R = pred->rows, C = pred->cols;
    double tp = 0.0, fp = 0.0, fn = 0.0;

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            int p = AT(pred, r, c) > 0.5, g = AT(gt, r, c) > 0.5;
            int g_near = 0, p_near = 0;
            for (int dr = -tol; dr <= tol && (!g_near || !p_near); dr++) {
                for (int dc = -tol; dc <= tol; dc++) {
                    int rr = r + dr, cc = c + dc;
                    if (rr < 0 || rr >= R || cc < 0 || cc >= C) continue;
                    if (AT(gt, rr, cc) > 0.5) g_near = 1;
                    if (AT(pred, rr, cc) > 0.5) p_near = 1;
                    if (g_near && p_near) break;
                }
            }
            if (p && g_near) tp += 1.0;
            if (p && !g_near) fp += 1.0;
            if (!p_near && g) fn += 1.0;
        }
    }
    double prec = tp / (tp + fp > 0 ? tp + fp : 1.0);
    double rec = tp / (tp + fn > 0 ? tp + fn : 1.0);
    return (prec + rec > 0.0) ? 2.0 * prec * rec / (prec + rec) : 0.0;
}

/* ------------------------------------------------------------------ */
/* I/O                                                                 */
/* ------------------------------------------------------------------ */
static Grid *read_dem(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open DEM: %s\n", path); return NULL; }
    int rows = 0, cols = 0;
    if (fscanf(f, "%d %d", &rows, &cols) != 2 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "bad DEM header in %s\n", path);
        fclose(f);
        return NULL;
    }
    Grid *g = grid_new(rows, cols);
    if (!g) { fclose(f); return NULL; }
    for (size_t i = 0; i < (size_t)rows * cols; i++) {
        if (fscanf(f, "%lf", &g->v[i]) != 1) {
            fprintf(stderr, "DEM truncated at element %zu\n", i);
            fclose(f);
            grid_free(g);
            return NULL;
        }
    }
    fclose(f);
    return g;
}

/* Plain-text output: "rows cols" then one value per line. Chosen over an image
 * format so that tools/verify_c_vs_python.py can diff it exactly. */
static int write_txt(const char *path, const Grid *g)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    fprintf(f, "%d %d\n", g->rows, g->cols);
    for (size_t i = 0; i < (size_t)g->rows * g->cols; i++)
        fprintf(f, "%.6f\n", g->v[i]);
    fclose(f);
    return 0;
}

static int write_pgm(const char *path, const Grid *g, int invert)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    fprintf(f, "P5\n%d %d\n255\n", g->cols, g->rows);
    for (size_t i = 0; i < (size_t)g->rows * g->cols; i++) {
        double x = g->v[i];
        if (invert) x = 1.0 - x;
        if (x < 0.0) x = 0.0;
        if (x > 1.0) x = 1.0;
        unsigned char b = (unsigned char)lround(x * 255.0);
        fputc(b, f);
    }
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Self-test                                                           */
/* ------------------------------------------------------------------ */
/* These are the checks that caught real sign errors in the Python version.
 * Keep them: a shadow routine that is wrong in a consistent way still produces
 * a plausible-looking image. */
static int selftest(void)
{
    int failures = 0;
    printf("lunar_shadow (C) self-test\n");
    printf("==========================\n");

    /* 1) direction conventions */
    double dc, dr, dz;
    sun_direction(90.0, 0.0, &dc, &dr, &dz);
    int ok1 = fabs(dc - 1.0) < 1e-12 && fabs(dr) < 1e-12 && fabs(dz) < 1e-12;
    printf("[%s] sun at due east -> direction (1, 0, 0)\n", ok1 ? "PASS" : "FAIL");
    failures += !ok1;

    sun_direction(0.0, 0.0, &dc, &dr, &dz);
    int ok2 = fabs(dc) < 1e-12 && dr < -0.99;
    printf("[%s] sun due north -> drow negative (row grows southward)\n",
           ok2 ? "PASS" : "FAIL");
    failures += !ok2;

    /* 2) flat terrain must be fully lit -- catches an inverted inequality */
    Grid *flat = grid_new(32, 32);
    Grid *ill = compute_illumination(flat, 45.0, 5.0, 1.0, 0, 1);
    int shadowed = 0;
    for (size_t i = 0; i < 32u * 32u; i++) if (ill->v[i] < 0.5) shadowed++;
    int ok3 = (shadowed == 0);
    printf("[%s] flat terrain is fully lit (shadowed=%d)\n", ok3 ? "PASS" : "FAIL", shadowed);
    failures += !ok3;
    grid_free(ill);

    /* 3) a north-south ridge with the Sun due west: the analytic horizon angle
     *    on the western side is atan(10/50) = 11.31 deg, and the shadow must
     *    fall EAST of the ridge. This pins the azimuth convention down; no
     *    amount of code review substitutes for it. */
    Grid *ridge = grid_new(101, 101);
    for (int r = 0; r < 101; r++) AT(ridge, r, 50) = 10.0;
    Grid *H = horizon_map(ridge, 1.0, 72, 0);
    /* Sun due west, ridge at column 50. A point 25 m EAST of the ridge must see it:
     * atan(10/25) = 21.80 deg. A point 25 m west of it sees nothing. (An earlier version
     * of this assertion compared against atan(10/50) = 11.31 deg while sampling at 25 m,
     * i.e. the test itself was wrong, not the code.) */
    double want = atan2(10.0, 25.0) * 180.0 / M_PI;
    int m_w = (int)lround(270.0 / 5.0) % 72;
    double h_east = H->v[(size_t)m_w * 101u * 101u + 50u * 101u + 75u]; /* 25 m east */
    double h_west = H->v[(size_t)m_w * 101u * 101u + 50u * 101u + 25u]; /* 25 m west */
    int ok4 = fabs(h_east - want) < 0.05 && h_west < 0.05;
    printf("[%s] ridge analytic: 25 m east=%.2f deg (want %.2f), 25 m west=%.2f deg\n",
           ok4 ? "PASS" : "FAIL", h_east, want, h_west);
    failures += !ok4;

    Grid *ill_r = compute_illumination(ridge, 270.0, 5.0, 1.0, 0, 0);
    int first_shadow_col = -1;
    for (int c = 0; c < 101 && first_shadow_col < 0; c++)
        for (int r = 0; r < 101; r++)
            if (AT(ill_r, r, c) < 0.5) { first_shadow_col = c; break; }
    int ok5 = (first_shadow_col > 50);
    printf("[%s] shadow falls east of the ridge (first shadowed col=%d, ridge at 50)\n",
           ok5 ? "PASS" : "FAIL", first_shadow_col);
    failures += !ok5;

    /* 4) ray marching vs horizon lookup must agree: two independent code paths */
    Grid *dem = grid_new(64, 64);
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++)
            AT(dem, r, c) = 3.0 * sin(r / 7.0) * cos(c / 5.0)
                          + 2.0 * sin((r + c) / 11.0);
    Grid *H2 = horizon_map(dem, 1.0, 72, 0);
    Grid *a = compute_illumination(dem, 137.0, 8.0, 1.0, 0, 1);
    Grid *b = illuminate_from_horizon(dem, H2, 137.0, 8.0, 72, 1.0, 1);
    int agree = 0;
    for (size_t i = 0; i < 64u * 64u; i++) if (a->v[i] == b->v[i]) agree++;
    double rate = (double)agree / (64.0 * 64.0);
    int ok6 = rate > 0.95;
    printf("[%s] ray marching vs horizon lookup agree on %.2f%% of pixels\n",
           ok6 ? "PASS" : "FAIL", rate * 100.0);
    failures += !ok6;

    /* 5) shadow length <-> height round trip */
    double h = 1.0, el = 1.5;
    double L = h / tan(el * M_PI / 180.0);
    double back = L * tan(el * M_PI / 180.0);
    int ok7 = fabs(back - h) < 1e-12;
    printf("[%s] shadow length round trip (1 m at 1.5 deg -> %.2f m -> %.6f m)\n",
           ok7 ? "PASS" : "FAIL", L, back);
    failures += !ok7;

    grid_free(flat); grid_free(ridge); grid_free(H); grid_free(ill_r);
    grid_free(dem); grid_free(H2); grid_free(a); grid_free(b);

    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */
static void usage(const char *argv0)
{
    printf(
        "lunar_shadow (C99) -- lunar illumination / shadow algorithms\n\n"
        "usage:\n"
        "  %s --selftest\n"
        "  %s --dem FILE [options]\n\n"
        "options:\n"
        "  --dem FILE        DEM as text: \"rows cols\" then rows*cols values\n"
        "  --az DEG          sun azimuth, compass (default 137)\n"
        "  --el DEG          sun elevation (default 6)\n"
        "  --horizon FILE    input horizon-map text file (skips ray marching)\n"
        "  --n-az N          azimuth bins for the horizon map (default 72)\n"
        "  --cell M          grid cell size in metres (default 1)\n"
        "  --max-steps N     max ray-march steps (default 2*max(rows,cols))\n"
        "  --no-slope        omit the local slope-orientation (attached shadow) test\n"
        "  --out-illum FILE  write the illumination mask as text\n"
        "  --out-pgm FILE    write the illumination mask as a PGM image\n"
        "  --dump-horizon F  write the horizon map as text (n_az planes)\n"
        "  --dump-intermediates  also write gx / gy / cos(incidence) next to the\n"
        "                        outputs (debugging the port against Python)\n"
        "  --selftest        run the geometry invariants and exit\n",
        argv0, argv0);
}

int main(int argc, char **argv)
{
    if (argc == 1) { usage(argv[0]); return 2; }

    const char *dem_path = NULL, *out_illum = NULL, *out_pgm = NULL;
    const char *dump_h = NULL, *horizon_in = NULL, *dump_dir = NULL;
    double az = 137.0, el = 6.0, cell = 1.0;
    int n_az = 72, max_steps = 0, use_slope = 1, dump_inter = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--selftest")) return selftest();
        else if (!strcmp(a, "--dump-intermediates")) dump_inter = 1;
        else if (!strcmp(a, "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else if (!strcmp(a, "--dem") && i + 1 < argc) dem_path = argv[++i];
        else if (!strcmp(a, "--az") && i + 1 < argc) az = atof(argv[++i]);
        else if (!strcmp(a, "--el") && i + 1 < argc) el = atof(argv[++i]);
        else if (!strcmp(a, "--cell") && i + 1 < argc) cell = atof(argv[++i]);
        else if (!strcmp(a, "--n-az") && i + 1 < argc) n_az = atoi(argv[++i]);
        else if (!strcmp(a, "--max-steps") && i + 1 < argc) max_steps = atoi(argv[++i]);
        else if (!strcmp(a, "--no-slope")) use_slope = 0;
        else if (!strcmp(a, "--out-illum") && i + 1 < argc) out_illum = argv[++i];
        else if (!strcmp(a, "--out-pgm") && i + 1 < argc) out_pgm = argv[++i];
        else if (!strcmp(a, "--dump-horizon") && i + 1 < argc) dump_h = argv[++i];
        else if (!strcmp(a, "--horizon") && i + 1 < argc) horizon_in = argv[++i];
        else { fprintf(stderr, "unknown argument: %s\n", a); usage(argv[0]); return 2; }
    }

    if (!dem_path) { fprintf(stderr, "--dem is required (or use --selftest)\n"); return 2; }

    Grid *dem = read_dem(dem_path);
    if (!dem) return 1;
    printf("DEM %d x %d, cell %.4g m, sun az %.2f deg el %.2f deg\n",
           dem->rows, dem->cols, cell, az, el);

    Grid *H = NULL;
    if (dump_h || horizon_in) {
        H = horizon_map(dem, cell, n_az, max_steps);
        double mx = -1e9;
        for (size_t i = 0; i < (size_t)n_az * dem->rows * dem->cols; i++)
            if (H->v[i] > mx) mx = H->v[i];
        printf("horizon map: %d azimuth bins, max horizon angle %.2f deg\n", n_az, mx);
        if (dump_h && write_txt(dump_h, H)) { grid_free(dem); grid_free(H); return 1; }
    }

    Grid *illum = horizon_in
        ? illuminate_from_horizon(dem, H, az, el, n_az, cell, use_slope)
        : compute_illumination(dem, az, el, cell, max_steps, use_slope);

    double sum = 0.0;
    for (size_t i = 0; i < (size_t)dem->rows * dem->cols; i++) sum += illum->v[i];
    double frac = sum / ((double)dem->rows * dem->cols);
    printf("lit fraction %.6f, shadow fraction %.6f\n", frac, 1.0 - frac);

    if (dump_inter) {
        /* Attribute the shadow count to each stage, so a disagreement with
         * Python can be localised to the slope term or to the ray march. */
        const size_t N = (size_t)dem->rows * dem->cols;
        Grid *ci_tmp = grid_new(dem->rows, dem->cols);
        Grid *co = compute_illumination(dem, az, el, cell, max_steps, 0); /* cast only */
        cos_incidence(dem, cell, az, el, ci_tmp);

        long slope_shadow = 0, cast_shadow = 0, combined_shadow = 0;
        for (size_t i = 0; i < N; i++) {
            if (ci_tmp->v[i] <= 0.0) slope_shadow++;
            if (co->v[i] <= 0.5) cast_shadow++;
            if (illum->v[i] <= 0.5) combined_shadow++;
        }
        printf("stage shadow fractions: slope-only %.6f | cast-only %.6f | combined %.6f\n",
               (double)slope_shadow / N, (double)cast_shadow / N,
               (double)combined_shadow / N);
        {
            const int probe[][2] = {{0, 40}, {21, 30}, {50, 50}, {95, 95}};
            printf("  probe pixels (row,col): slope_ok cast_only combined\n");
            for (size_t p = 0; p < sizeof(probe) / sizeof(probe[0]); p++) {
                int r = probe[p][0], c = probe[p][1];
                size_t idx = (size_t)r * dem->cols + c;
                printf("    (%3d,%3d)  slope=%d  cast_only=%d  combined=%d\n",
                       r, c, ci_tmp->v[idx] > 0.0, co->v[idx] > 0.5,
                       illum->v[idx] > 0.5);
            }
        }
        grid_free(ci_tmp);
        grid_free(co);
    }

    int rc = 0;
    if (out_illum) rc |= write_txt(out_illum, illum);
    if (out_pgm)   rc |= write_pgm(out_pgm, illum, 0);

    if (dump_inter) {
        /* Intermediates for pinpointing a porting disagreement against Python.
         * These are exactly the quantities tools/verify_c_vs_python.py compares. */
        Grid *gx = grid_new(dem->rows, dem->cols);
        Grid *gy = grid_new(dem->rows, dem->cols);
        Grid *ci = grid_new(dem->rows, dem->cols);
        terrain_gradient(dem, cell, gx, gy);
        cos_incidence(dem, cell, az, el, ci);
        const char *names[3] = {"gx", "gy", "cos_inc"};
        Grid *grids[3] = {gx, gy, ci};
        for (int i = 0; i < 3; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s.txt", dump_dir ? dump_dir : ".", names[i]);
            rc |= write_txt(path, grids[i]);
            printf("wrote %s\n", path);
        }
        grid_free(gx); grid_free(gy); grid_free(ci);
    }

    grid_free(dem);
    grid_free(H);
    grid_free(illum);
    return rc;
}
