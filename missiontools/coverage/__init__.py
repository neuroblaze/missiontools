"""
missiontools.coverage
=====================
Coverage and access analysis, and geographic area sampling.

Sampling
--------
:func:`sample_aoi`
    Sample points inside an arbitrary lat/lon polygon using the Fibonacci
    sphere lattice filtered by a point-in-polygon test.
:func:`sample_region`
    Fibonacci-sphere sample points inside a rectangular lat/lon region.
    All parameters are optional; omitting both latitude bounds gives the
    full sphere.
:func:`sample_shapefile`
    Fibonacci-sphere sample points inside an ESRI Shapefile polygon,
    including antimeridian-crossing geometries.
:func:`sample_geography`
    Fibonacci-sphere sample points for a Natural Earth geography, resolved
    by country/subdivision name, ISO 3166 code, or ``'Country/Region'``
    slash pattern.

Coverage
--------
:func:`coverage_fraction`
    Fraction of the AoI visible at each timestep plus the cumulative
    (ever-seen) fraction over the analysis window.
:func:`revisit_time`
    Per-point maximum and mean gap between consecutive accesses over the
    analysis window.

Pointwise access
----------------
:func:`pointwise_coverage`
    Raw ``(T × M)`` boolean visibility matrix — which ground points are
    in view at each timestep.
:func:`access_pointwise`
    AOS/LOS access intervals for every ground point.
:func:`revisit_pointwise`
    LOS-to-AOS gap arrays for every ground point.

Notes
-----
All sampling functions return latitudes and longitudes in **radians**.
Coverage and access functions accept angles in radians and distances in
metres, consistent with the package-wide SI convention.
"""

from .coverage import (sample_aoi, sample_region, sample_shapefile,
                       sample_geography,
                       coverage_fraction, revisit_time, pointwise_coverage,
                       access_pointwise, revisit_pointwise)

__all__ = ['sample_aoi', 'sample_region', 'sample_shapefile',
           'sample_geography',
           'coverage_fraction', 'revisit_time', 'pointwise_coverage',
           'access_pointwise', 'revisit_pointwise']
