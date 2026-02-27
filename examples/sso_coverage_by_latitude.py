"""
SSO coverage by latitude band
==============================
Computes 30-day coverage statistics for a 550 km sun-synchronous orbit
(LTAN 10:30, descending node) equipped with a 10° half-angle nadir sensor.

Results are reported per 5° latitude band:
  - Number of sample points in the band
  - Cumulative coverage fraction (% of points seen ≥ once)
  - Mean revisit time (hours)
  - Max revisit time (hours)
"""

import time
import numpy as np

from missiontools.orbit.propagation import sun_synchronous_orbit
from missiontools.coverage import sample_region, coverage_fraction, revisit_time

# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------
EPOCH = np.datetime64('2025-01-01T00:00:00', 'us')

params = sun_synchronous_orbit(
    altitude           = 550_000.0,    # m
    local_time_at_node = '10:30',
    node_type          = 'descending',
    epoch              = EPOCH,
)

# ---------------------------------------------------------------------------
# Analysis window: 30 days
# ---------------------------------------------------------------------------
T_START = EPOCH
T_END   = EPOCH + np.timedelta64(30 * 86400, 's')

# ---------------------------------------------------------------------------
# Sensor and propagation settings
# ---------------------------------------------------------------------------
# Nadir direction in LVLH = −R̂ (negative radial)
NADIR_LVLH    = np.array([-1.0, 0.0, 0.0])
FOV_HALF_ANGLE = np.radians(10.0)          # 10° half-angle → 20° full cone

MAX_STEP      = np.timedelta64(20, 's')    # 20 s step resolves the ~25 s dwell time
POINT_DENSITY = 2e11                       # m² per sample point (~450 km² each)

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------
T_DAYS = int((T_END - T_START) / np.timedelta64(1, 's')) / 86400

print("=" * 72)
print("  SSO Coverage Analysis — 5° Latitude Bands")
print("=" * 72)
print(f"  Orbit          : SSO  550 km  LTAN 10:30  (descending)")
print(f"  Inclination    : {np.degrees(params['i']):.3f}°")
print(f"  Sensor         : Nadir-pointing  ±{np.degrees(FOV_HALF_ANGLE):.0f}° FOV")
print(f"  Propagator     : J2")
print(f"  Window         : {T_START}  →  {T_END}  ({T_DAYS:.0f} days)")
print(f"  Time step      : {int(MAX_STEP / np.timedelta64(1, 's'))} s")
print(f"  Point density  : {POINT_DENSITY:.0e} m²/point"
      f"  (~{POINT_DENSITY/1e6:.0f} km²/point)")
print("=" * 72)

# ---------------------------------------------------------------------------
# Per-band computation
# ---------------------------------------------------------------------------
# Latitude bands from −90° to +90° in 5° steps
LAT_EDGES_DEG = np.arange(-90, 91, 5)   # 37 edges → 36 bands

rows = []   # (band_label, n_pts, cov_pct, mean_rev_h, max_rev_h)

t0 = time.perf_counter()

for lo_deg, hi_deg in zip(LAT_EDGES_DEG[:-1], LAT_EDGES_DEG[1:]):
    lo = np.radians(float(lo_deg))
    hi = np.radians(float(hi_deg))

    lat, lon = sample_region(lat_min=lo, lat_max=hi, point_density=POINT_DENSITY)
    n = len(lat)

    label = (f"{abs(lo_deg):2.0f}°{'S' if lo_deg < 0 else 'N'}"
             f" – {abs(hi_deg):2.0f}°{'S' if hi_deg <= 0 else 'N'}")

    cf = coverage_fraction(
        lat, lon, params, T_START, T_END,
        fov_pointing_lvlh = NADIR_LVLH,
        fov_half_angle    = FOV_HALF_ANGLE,
        propagator_type   = 'j2',
        max_step          = MAX_STEP,
    )
    rt = revisit_time(
        lat, lon, params, T_START, T_END,
        fov_pointing_lvlh = NADIR_LVLH,
        fov_half_angle    = FOV_HALF_ANGLE,
        propagator_type   = 'j2',
        max_step          = MAX_STEP,
    )

    cov_pct     = cf['final_cumulative'] * 100.0
    mean_rev_h  = rt['global_mean'] / 3600.0 if not np.isnan(rt['global_mean']) else float('nan')
    max_rev_h   = rt['global_max']  / 3600.0 if not np.isnan(rt['global_max'])  else float('nan')

    rows.append((label, n, cov_pct, mean_rev_h, max_rev_h))

elapsed = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n  {'Band':>13}  {'Pts':>5}  {'Coverage':>10}  {'Mean Rev':>10}  {'Max Rev':>10}")
print(f"  {'─'*13}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}")

for label, n, cov_pct, mean_rev_h, max_rev_h in rows:
    mean_s = f"{mean_rev_h:8.2f} h" if not np.isnan(mean_rev_h) else "      — "
    max_s  = f"{max_rev_h:8.2f} h"  if not np.isnan(max_rev_h)  else "      — "
    print(f"  {label:>13}  {n:>5}  {cov_pct:>9.1f}%  {mean_s}  {max_s}")

print(f"\n  Calculation time : {elapsed:.1f} s")
print("=" * 72)
