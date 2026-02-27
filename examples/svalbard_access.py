"""
Svalbard access example
=======================
Computes all access windows to the Svalbard Satellite Station (SvalSat)
for a sun-synchronous satellite at 550 km with an LTDN of 10:30,
over the period 2025-03-01 to 2025-03-07 (inclusive).
"""

import time
import numpy as np

from missiontools.orbit.propagation import sun_synchronous_orbit
from missiontools.orbit.access import earth_access_intervals

# ---------------------------------------------------------------------------
# Ground station: SvalSat, Svalbard, Norway
# Coordinates: 78.229°N, 15.407°E, ~500 m altitude
# ---------------------------------------------------------------------------
SVALSAT_LAT = np.radians(78.229)
SVALSAT_LON = np.radians(15.407)
SVALSAT_ALT = 500.0   # m

# ---------------------------------------------------------------------------
# Sun-synchronous orbit: 550 km, LTDN 10:30
# Epoch set to the start of the analysis window
# ---------------------------------------------------------------------------
EPOCH = np.datetime64('2025-03-01T00:00:00', 'us')

params = sun_synchronous_orbit(
    altitude            = 550_000.0,     # m
    local_time_at_node  = '10:30',
    node_type           = 'descending',
    epoch               = EPOCH,
)

# ---------------------------------------------------------------------------
# Analysis window: 2025-03-01 00:00 UTC → 2025-03-08 00:00 UTC (7 full days)
# ---------------------------------------------------------------------------
T_START = np.datetime64('2025-03-01T00:00:00', 'us')
T_END   = np.datetime64('2025-03-08T00:00:00', 'us')

EL_MIN   = np.radians(5.0)                  # minimum elevation (5°)
MAX_STEP = np.timedelta64(20, 's')          # coarse scan step
REFINE   = np.timedelta64(1, 's')           # edge refinement tolerance

# ---------------------------------------------------------------------------
# Print orbit summary
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Svalbard Satellite Station — Access Report")
print("=" * 65)
print(f"  Ground station : SvalSat  ({np.degrees(SVALSAT_LAT):.3f}°N, "
      f"{np.degrees(SVALSAT_LON):.3f}°E,  {SVALSAT_ALT:.0f} m)")
print(f"  Orbit          : SSO  550 km  LTDN 10:30")
print(f"  Inclination    : {np.degrees(params['i']):.3f}°")
print(f"  RAAN at epoch  : {np.degrees(params['raan']):.3f}°")
print(f"  Period         : {2*np.pi*np.sqrt(params['a']**3/params['central_body_mu'])/60:.2f} min")
print(f"  Propagator     : J2")
print(f"  Min elevation  : {np.degrees(EL_MIN):.0f}°")
print(f"  Window         : {T_START}  →  {T_END}")
print("=" * 65)

# ---------------------------------------------------------------------------
# Compute access intervals (J2 propagator for physical fidelity)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()

intervals = earth_access_intervals(
    T_START, T_END, params,
    lat             = SVALSAT_LAT,
    lon             = SVALSAT_LON,
    alt             = SVALSAT_ALT,
    el_min          = EL_MIN,
    propagator_type = 'j2',
    max_step        = MAX_STEP,
    refine_tol      = REFINE,
)

elapsed_ms = (time.perf_counter() - t0) * 1e3

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n  {'#':>3}  {'AOS (UTC)':^26}  {'LOS (UTC)':^26}  {'Duration':>9}")
print(f"  {'─'*3}  {'─'*26}  {'─'*26}  {'─'*9}")

total_s = 0
for idx, (start, end) in enumerate(intervals, 1):
    dur_s = int((end - start) / np.timedelta64(1, 's'))
    total_s += dur_s
    print(f"  {idx:>3}  {str(start):^26}  {str(end):^26}  {dur_s//60:3d}m {dur_s%60:02d}s")

print(f"\n  Total passes       : {len(intervals)}")
print(f"  Total contact time : {total_s // 3600}h {(total_s % 3600) // 60}m {total_s % 60:02d}s"
      f"  ({total_s / (7*86400) * 100:.2f}% duty cycle)")
print(f"  Calculation time   : {elapsed_ms:.1f} ms")
print("=" * 65)
