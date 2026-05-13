# 3D Spacecraft Models

This directory contains a curated collection of NASA spacecraft 3D models
sourced from [NASA-3D-Resources](https://github.com/nasa/NASA-3D-Resources).
All models are in glTF binary (`.glb`) format for use with the
`missiontools` Cesium visualisation submodule.

## Selection criteria

* **Spacecraft only** — satellites, orbiters, observatories, crewed vehicles,
  and generic bus designs. Ground hardware, rockets, rovers, landers, suits,
  celestial bodies, and nebulae were excluded.
* **One per design** — where NASA provides multiple variants/quality levels
  for the same spacecraft, only the single largest file under 4 MB was kept.
* **4 MB ceiling** — no model exceeds 4 MB, keeping Cesium startup fast and
  memory usage reasonable.
* **Normalised filenames** — all names are lower-case, spaces are replaced
  with underscores, and parenthetical acronyms are collapsed to the acronym.
  Example: `Mars Odyssey.glb` → `mars_odyssey.glb`;
  `Advanced Technology Large-Aperture Space Telescope (ATLAST).glb` → `atlast.glb`.

## Available models

| Model | Filename | Size |
|-------|----------|------|
| AcrimSAT | `acrimsat.glb` | 2.02 MB |
| Advanced Composition Explorer | `advanced_composition_explorer.glb` | 1.94 MB |
| Advanced Crew Escape Suit | `advanced_crew_escape_suit.glb` | 0.82 MB |
| Aeronomy of Ice in the Mesosphere | `aeronomy_of_ice_in_the_mesosphere.glb` | 0.31 MB |
| Agena Target Vehicle | `agena_target_vehicle.glb` | 1.10 MB |
| Apollo Lunar Module | `apollo_lunar_module.glb` | 0.68 MB |
| Apollo-Soyuz Test Project | `apollo_soyuz.glb` | 0.91 MB |
| Aqua | `aqua.glb` | 1.90 MB |
| Aquarius | `aquarius.glb` | 3.62 MB |
| Argo | `argo.glb` | 0.04 MB |
| ASTRE | `astre.glb` | 0.56 MB |
| ATLAST | `atlast.glb` | 0.44 MB |
| Aura | `aura.glb` | 1.35 MB |
| CALIPSO | `calipso.glb` | 0.51 MB |
| Cassini Assembly | `cassini_assembly.glb` | 2.28 MB |
| Cassini-Huygens | `cassini_huygens.glb` | 1.58 MB |
| Chandra X-ray Observatory | `chandra_x_ray_observatory.glb` | 0.95 MB |
| Clementine | `clementine.glb` | 2.89 MB |
| CloudSat | `cloudsat.glb` | 0.67 MB |
| Cluster | `cluster.glb` | 0.14 MB |
| Cluster II | `cluster_ii.glb` | 0.80 MB |
| CNOFS | `cnofs.glb` | 1.42 MB |
| Constellation-X | `constellation_x.glb` | 0.24 MB |
| Cosmic Origins Spectrograph | `cosmic_origins_spectrograph.glb` | 0.42 MB |
| CubeSat 1 RU Generic | `cubesat_1_ru_generic.glb` | 0.14 MB |
| CubeSat 2 RU Generic | `cubesat_2_ru_generic.glb` | 0.20 MB |
| CubeSat ICECube | `cubesat_icecube.glb` | 0.30 MB |
| CubeSat MiRaTa | `cubesat_mirata.glb` | 0.42 MB |
| CYGNSS | `cygnss.glb` | 1.55 MB |
| Dawn | `dawn.glb` | 0.88 MB |
| Deep Space 1 | `deep_space_1.glb` | 0.89 MB |
| DESDynI | `desdyni.glb` | 0.43 MB |
| DSCOVR (Triana) | `dscovr.glb` | 0.11 MB |
| Earth Observing-1 (EO-1) | `eo_1.glb` | 0.21 MB |
| EPOXI | `epoxi.glb` | 2.74 MB |
| ESAS Crew Module | `esas_crew_module.glb` | 0.01 MB |
| Far Ultraviolet Spectroscopic Explorer | `far_ultraviolet_spectroscopic_explorer.glb` | 0.23 MB |
| Fermi Gamma-ray Large Area Space Telescope | `fermi_gamma_ray_large_area_space_telescope.glb` | 1.32 MB |
| Firefly | `firefly.glb` | 0.28 MB |
| Galileo | `galileo.glb` | 0.24 MB |
| Gamma Ray Observatory | `gamma_ray_observatory.glb` | 0.06 MB |
| Gemini | `gemini.glb` | 1.32 MB |
| Geostationary Operational Environmental Satellites | `geostationary_operational_environmental_satellites.glb` | 0.32 MB |
| GeoTailSAT | `geotailsat.glb` | 0.09 MB |
| Global Hawk | `global_hawk.glb` | 1.13 MB |
| Global Precipitation Measurement | `global_precipitation_measurement.glb` | 2.15 MB |
| GRACE | `grace.glb` | 1.70 MB |
| HESSI-RHESSI | `hessi_rhessi.glb` | 0.22 MB |
| High Energy Transient Explorer | `high_energy_transient_explorer.glb` | 0.06 MB |
| Hubble Space Telescope | `hubble_space_telescope.glb` | 1.62 MB |
| IBEX | `ibex.glb` | 1.25 MB |
| ICESat | `icesat.glb` | 2.72 MB |
| ICESat-2 | `icesat_2.glb` | 1.91 MB |
| ICON | `icon.glb` | 2.50 MB |
| International X-ray Observatory | `international_x_ray_observatory.glb` | 1.17 MB |
| International Space Station | `iss.glb` | 0.45 MB |
| James Webb Space Telescope | `james_webb_space_telescope.glb` | 0.96 MB |
| Jason-1 | `jason_1.glb` | 0.57 MB |
| Juno | `juno.glb` | 0.25 MB |
| Kepler | `kepler.glb` | 1.91 MB |
| LADEE | `ladee.glb` | 0.22 MB |
| Landsat 7 | `landsat_7.glb` | 0.15 MB |
| Landsat 8 | `landsat_8.glb` | 0.74 MB |
| LISA (2006) | `lisa.glb` | 0.29 MB |
| LLCD | `llcd.glb` | 1.00 MB |
| Lunar Reconnaissance Orbiter | `lunar_reconnaissance_orbiter.glb` | 2.77 MB |
| Magellan | `magellan.glb` | 3.01 MB |
| Mars Global Surveyor | `mars_global_surveyor.glb` | 1.83 MB |
| Mars Exploration Rover Opportunity (MER-B) | `mer_b.glb` | 0.03 MB |
| MESSENGER | `messenger.glb` | 2.55 MB |
| Mir | `mir.glb` | 0.02 MB |
| MMS | `mms.glb` | 1.13 MB |
| MRO | `mro.glb` | 2.53 MB |
| Nancy Grace Roman Space Telescope | `nancy_grace_roman_space_telescope.glb` | 2.55 MB |
| NEAR Shoemaker | `near_shoemaker.glb` | 0.67 MB |
| Ocean Surface Topography Mission | `ocean_surface_topography_mission.glb` | 0.88 MB |
| Orbiting Carbon Observatory (OCO-2) | `oco.glb` | 0.27 MB |
| OSIRIS-REx | `osiris_rex.glb` | 1.22 MB |
| Parker Solar Probe | `parker_solar_probe.glb` | 0.42 MB |
| Pioneer 10 | `pioneer_10.glb` | 1.95 MB |
| POES | `poes.glb` | 1.11 MB |
| Polar | `polar.glb` | 0.09 MB |
| Rosetta | `rosetta.glb` | 2.16 MB |
| SAC-C | `sac_c.glb` | 0.07 MB |
| SeaStar | `seastar.glb` | 0.02 MB |
| Sentinel-6 | `sentinel_6.glb` | 2.01 MB |
| SEXTANT | `sextant.glb` | 2.51 MB |
| Skylab | `skylab.glb` | 0.09 MB |
| SOHO | `solar_and_heliospheric_observatory.glb` | 0.08 MB |
| Hinode (Solar-B) | `solar_b.glb` | 0.26 MB |
| Solar Dynamics Observatory | `solar_dynamics_observatory.glb` | 0.47 MB |
| Solar Sail Concept | `solar_sail_concept.glb` | 0.22 MB |
| SORCE | `sorce.glb` | 0.35 MB |
| Space Shuttle | `space_shuttle.glb` | 2.36 MB |
| Spartan-201 | `spartan_201.glb` | 0.13 MB |
| Spitzer Space Telescope | `spitzer_space_telescope.glb` | 0.24 MB |
| SSL-1300 | `ssl_1300.glb` | 0.49 MB |
| Stardust | `stardust.glb` | 3.62 MB |
| STEREO | `stereo.glb` | 0.34 MB |
| Suomi NPP | `suomi_national_polar_orbiting_partnership.glb` | 2.18 MB |
| Super Lightweight Interchangeable | `super_lightweight_interchangeable.glb` | 1.08 MB |
| Suzaku | `suzaku.glb` | 0.37 MB |
| SWAS | `swas.glb` | 0.14 MB |
| Swift | `swift.glb` | 0.20 MB |
| TDRS | `tdrs.glb` | 2.78 MB |
| Terra | `terra.glb` | 2.05 MB |
| TESS | `tess.glb` | 1.62 MB |
| Tether | `tether.glb` | 0.47 MB |
| THEMIS | `themis.glb` | 0.65 MB |
| TOMS | `toms.glb` | 0.09 MB |
| TOPEX/Poseidon | `topex_poseidon.glb` | 0.20 MB |
| TRDS | `trds.glb` | 0.21 MB |
| TRMM | `trmm.glb` | 0.19 MB |
| Tselina-2 | `tselina_2.glb` | 0.53 MB |
| Ulysses | `ulysses.glb` | 2.35 MB |
| Van Allen Probes | `van_allen_probes.glb` | 0.89 MB |
| Voyager Probe | `voyager_probe.glb` | 1.64 MB |
| Wide Field Planetary Camera | `wide_field_planetary_camera.glb` | 0.28 MB |
| Wind | `wind.glb` | 0.22 MB |
| WIRE | `wire.glb` | 0.36 MB |
| WMAP | `wmap.glb` | 1.13 MB |

## Usage in `CesiumViewer`

The `model` argument to `CesiumViewer.add_spacecraft` now resolves bare
filenames against the bundled `_static/Models` directory automatically. You
can therefore pass any filename from the table above directly:

```python
from missiontools.cesium import CesiumViewer

viewer = CesiumViewer()
viewer.add_spacecraft(sc, t_start, t_end, model="hubble_space_telescope.glb", scale=1000.0)
viewer.show()
```

If the file does not exist in the built-in models directory, the value is
interpreted relative to the current working directory (or as an absolute path)
as before.

### Correcting model frame alignment

Many of the NASA models were authored in different tools with inconsistent
up/front/right axes. The `model_rotation` parameter lets you apply a one-time
Euler correction (rx, ry, rz in radians) so the model points correctly without
editing the glTF file:

```python
viewer.add_spacecraft(
    sc, t_start, t_end,
    model="hubble_space_telescope.glb",
    scale=1000.0,
    model_rotation=(np.pi, 0, 0),  # flip up if the model is upside-down
)
```

## Source

Models are provided by NASA and are in the public domain. Original collection:
<https://github.com/nasa/NASA-3D-Resources>
