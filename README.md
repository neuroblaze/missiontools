# `missiontools`: Space Mission Analysis in Python

`missiontools` is an MIT-licensed framework for space mission analysis tasks in Python. It is currently focused on Earth-orbiting missions, earth observation (EO) in particular.

**Check out the [examples folder](examples/) to see what you can do with `missiontools`**

## Features

`missiontools` currently supports:
 - Analytical propagation using either Keplerian two-body or J2 with secular perturbations.
 - Convenience functions for generating sun-synchronous, geostationary, and highly elliptical orbits.
 - Access interval computation (spacecraft to ground station, spacecraft to spacecraft)
 - Creation of areas of interest (AoIs) for coverage analysis: global, latitude/longitude bounding box, ESRI shapefile.
 - Convenience functions for creation of AoIs by country or state/province (eg: `AoI.from_geography('Canada/British Columbia')`). Uses the [Natural Earth](https://www.naturalearthdata.com) 1:50m dataset.
 - Computation of space-to-ground coverage including constraints on spacecraft elevation, solar zenith angle (SZA), and field-of-view.
 - Flexible condition system using boolean time-domain predicates (currently: sunlight, region, visibility), composable via boolean operations.
 - Conic and rectangular (pyramidal) sensor models with optional condition-based gating of coverage activity.
 - Selectable spacecraft attitude laws: fixed (LVLH, ECI, ECEF), targeted (spacecraft or ground station), limb (grazing-tangent), custom (user callback).
 - Condition-based attitude switching: route between attitude laws using conditions
 - Interference analysis between victim and interfering constellations with cumulative interference percentage and ITU threshold overlay.
 - Solar panel definition from panel normals and areas
 - Solar generation and orbit average power
 - Yaw steering for maximum solar generation
 - Thermal analysis: faces/normals connected to lumped element thermal model
 - Antenna modeling (isotropic, radially symmetric), including antenna pointing modes and ITU-R S.465 reference pattern.
 - Dynamic link budget computation including optional ITU-R P.618 weather effects (P.618 requires [ITU-RPy](https://itu-rpy.readthedocs.io/en/latest/)).
 
 Possible future features (open an issue if you'd like to see one of these implemented):
 - CAD import for solar & thermal
 - Self-shadowing for solar/thermal imported from CAD
 - Radiative thermal coupling between surface elements imported from CAD
 - Variable-rate communications
 - Radiation environment definition

## Dependencies
 - Python 3.13 or later
 - numpy
 - scipy
 - matplotlib
 - skyfield
 - pyshp
 - shapely
 - itur

## Validation

This project is in early days and is not yet well validated. Do not depend on the outputs for any critical design decisions. If an output looks off, it probably is - please open an issue!

## Vibe Coding Disclaimer

This project is my first foray into agentic development. I make the architecture decisions myself and use Claude Code to implement them - generating functions, classes, and tests from my specifications. I review all output and redirect when needed, but I do not generally write the implementation code directly.

## Acknowledgements
 - [Natural Earth](https://www.naturalearthdata.com) for the country/state/province shapefiles
 - Claude Sonnet 4.6 & Opus 4.6 for doing the implementation gruntwork
