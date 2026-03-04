# `missiontools`: Space Mission Analysis in Python

`missiontools` is an MIT-licensed framework for space mission analysis tasks in Python. It is currently focused on Earth-orbiting missions, earth observation (EO) in particular.

**If you are arriving here from a web search:** this package is under very active development and is probably not useful to you yet. APIs are subject to change without notice.

**Check out the [examples folder](examples/) to see what you can do with `missiontools`**

It currently supports:
 - Analytical propagation using either Keplerian two-body or J2 with secular perturbations.
 - Convenience functions for generating sun-synchronous, geostationary, and highly elliptical orbits.
 - Access interval computation (spacecraft to ground station, spacecraft to spacecraft)
 - Creation of areas of interest (AoIs) for coverage analysis: global, latitude/longitude bounding box, ESRI shapefile.
 - Convenience functions for creation of AoIs by country or state/province (eg: `AoI.from_geography('Canada/British Columbia')`). Uses the [Natural Earth](https://www.naturalearthdata.com) 1:50m dataset.
 - Computation of space-to-ground coverage including constraints on spacecraft elevation, solar zenith angle (SZA), and field-of-view.
 - Selectable spacecraft attitude laws: fixed (choice of frame: LVLH, ECI, ECEF), targeted
 - Solar generation and orbit average power
 - Solar panel definition from panel normals and areas
 
 Planned Features:
 - Basic thermal analysis: faces/normals connected to lumped element thermal model
 - CAD import for solar & thermal
 - Self-shadowing for solar/thermal imported from CAD
 - Communications system modeling including statistical weather effects and variable rate modulation

Possible features:
 - Radiation environment definition

## Vibe Coding Disclaimer

This project is my first foray into agentic development. I make the architecture decisions myself and use Claude Code to implement them - generating functions, classes, and tests from my specifications. I review all output and redirect when needed, but I do not generally write the implementation code directly.

## Acknowledgements
 - [Natural Earth](https://www.naturalearthdata.com) for the country/state/province shapefiles
 - Claude Sonnet 4.6 & Opus 4.6 for doing the implementation gruntwork