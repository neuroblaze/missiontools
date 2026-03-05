"""Thermal surface configuration classes.

A thermal config models the radiative interaction of a spacecraft's external
surfaces with the space environment (direct solar absorption, Earth albedo,
Earth IR, and infrared emission to space).  Attach it to a
:class:`~missiontools.Spacecraft` via
:meth:`~missiontools.Spacecraft.add_thermal_config`, then couple its faces
to a :class:`ThermalCircuit` via :meth:`attach`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..orbit.frames import sun_vec_eci
from ..orbit.shadow import in_sunlight

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)


class AbstractThermalConfig(ABC):
    """Base class for thermal surface configurations.

    Stores per-face areas, emissivities, and absorptivities.  Subclasses
    implement :meth:`_compute_absorbed_solar` and
    :meth:`_compute_earth_loads` to define how incident solar and Earth
    fluxes are projected onto each face.  The concrete :meth:`attach`
    method pre-computes all environmental heat loads over a time span and
    creates load functions (combining them with T⁴ emission) that are
    added to a :class:`ThermalCircuit`.

    Parameters
    ----------
    areas : array_like, shape (M,)
        Face areas (m²).  Must be positive.
    emissivities : array_like, shape (M,)
        IR emissivity of each face, in [0, 1].
    absorptivities : array_like, shape (M,)
        Solar absorptivity of each face, in [0, 1].
    irradiance : float
        Solar irradiance (W m⁻²).  Defaults to AM0 constant, 1366 W m⁻².
    earth_ir : float
        Average Earth IR flux (W m⁻²).  Defaults to 240 W m⁻².
    albedo : float
        Earth albedo coefficient, in [0, 1].  Defaults to 0.3.
    """

    def __init__(
        self,
        areas: npt.ArrayLike,
        emissivities: npt.ArrayLike,
        absorptivities: npt.ArrayLike,
        irradiance: float = 1366.0,
        earth_ir: float = 240.0,
        albedo: float = 0.3,
    ) -> None:
        areas_arr = np.asarray(areas, dtype=np.float64)
        emissivities_arr = np.asarray(emissivities, dtype=np.float64)
        absorptivities_arr = np.asarray(absorptivities, dtype=np.float64)

        if areas_arr.ndim != 1:
            raise ValueError(
                f"areas must be 1-D, got shape {areas_arr.shape}"
            )
        m = len(areas_arr)
        if emissivities_arr.shape != (m,):
            raise ValueError(
                f"emissivities must have shape ({m},), "
                f"got {emissivities_arr.shape}"
            )
        if absorptivities_arr.shape != (m,):
            raise ValueError(
                f"absorptivities must have shape ({m},), "
                f"got {absorptivities_arr.shape}"
            )
        if np.any(areas_arr <= 0):
            raise ValueError("All face areas must be positive.")
        if np.any((emissivities_arr < 0) | (emissivities_arr > 1)):
            raise ValueError("Emissivities must be in [0, 1].")
        if np.any((absorptivities_arr < 0) | (absorptivities_arr > 1)):
            raise ValueError("Absorptivities must be in [0, 1].")

        irradiance = float(irradiance)
        if irradiance <= 0:
            raise ValueError(
                f"irradiance must be positive, got {irradiance}."
            )

        earth_ir = float(earth_ir)
        if earth_ir < 0:
            raise ValueError(
                f"earth_ir must be non-negative, got {earth_ir}."
            )

        albedo = float(albedo)
        if not 0 <= albedo <= 1:
            raise ValueError(
                f"albedo must be in [0, 1], got {albedo}."
            )

        self._areas = areas_arr
        self._emissivities = emissivities_arr
        self._absorptivities = absorptivities_arr
        self._irradiance = irradiance
        self._earth_ir = earth_ir
        self._albedo = albedo
        self._spacecraft = None

    # --- properties ---

    @property
    def areas(self) -> npt.NDArray[np.floating]:
        """Face areas (m²), shape ``(M,)``."""
        return self._areas.copy()

    @property
    def emissivities(self) -> npt.NDArray[np.floating]:
        """IR emissivity per face, shape ``(M,)``."""
        return self._emissivities.copy()

    @property
    def absorptivities(self) -> npt.NDArray[np.floating]:
        """Solar absorptivity per face, shape ``(M,)``."""
        return self._absorptivities.copy()

    @property
    def irradiance(self) -> float:
        """Solar irradiance (W m⁻²)."""
        return self._irradiance

    @property
    def earth_ir(self) -> float:
        """Average Earth IR flux (W m⁻²)."""
        return self._earth_ir

    @property
    def albedo(self) -> float:
        """Earth albedo coefficient."""
        return self._albedo

    @property
    def num_faces(self) -> int:
        """Number of faces."""
        return len(self._areas)

    @property
    def spacecraft(self):
        """Spacecraft this config is attached to, or ``None``."""
        return self._spacecraft

    def _require_spacecraft(self) -> None:
        if self._spacecraft is None:
            raise RuntimeError(
                "Thermal config must be attached to a Spacecraft via "
                "add_thermal_config() before calling this method."
            )

    # --- abstract interface ---

    @abstractmethod
    def _compute_absorbed_solar(
        self,
        r: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        sun_eci: np.ndarray,
        lit: np.ndarray,
    ) -> np.ndarray:
        """Compute absorbed solar power for each face at each timestep.

        Parameters
        ----------
        r : ndarray, shape (N, 3)
            ECI position (m).
        v : ndarray, shape (N, 3)
            ECI velocity (m/s).
        t : ndarray, shape (N,)
            Timestamps (datetime64[us]).
        sun_eci : ndarray, shape (N, 3)
            Unit vectors toward the Sun in ECI.
        lit : ndarray, shape (N,)
            Boolean mask: True where spacecraft is in sunlight.

        Returns
        -------
        ndarray, shape (N, M)
            Absorbed solar power (W) per face per timestep.
        """

    def _compute_earth_loads(
        self,
        r: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        sun_eci: np.ndarray,
        lit: np.ndarray,
    ) -> np.ndarray:
        """Compute absorbed Earth IR and albedo power per face per timestep.

        The default implementation returns zeros.  Subclasses that know the
        face geometry (e.g. normal vectors) should override this to compute
        view factors and the resulting Earth heat loads.

        Parameters
        ----------
        r : ndarray, shape (N, 3)
            ECI position (m).
        v : ndarray, shape (N, 3)
            ECI velocity (m/s).
        t : ndarray, shape (N,)
            Timestamps (datetime64[us]).
        sun_eci : ndarray, shape (N, 3)
            Unit vectors toward the Sun in ECI.
        lit : ndarray, shape (N,)
            Boolean mask: True where spacecraft is in sunlight.

        Returns
        -------
        ndarray, shape (N, M)
            Combined Earth IR + albedo absorbed power (W) per face.
        """
        return np.zeros((len(t), self.num_faces), dtype=np.float64)

    # --- circuit coupling ---

    def attach(
        self,
        circuit,
        face_nodes: list[str],
        t_start: np.datetime64,
        t_end: np.datetime64,
        step: np.timedelta64,
        *,
        prefix: str = 'thermal',
    ) -> float:
        """Couple surface faces to a :class:`ThermalCircuit`.

        Pre-computes environmental heat loads (direct solar, Earth albedo,
        Earth IR) over the time span, then creates load functions for each
        face that combine interpolated absorption with Stefan-Boltzmann
        emission (``-ε σ A T⁴``).  Each load is registered on the circuit
        via :meth:`~ThermalCircuit.add_load`.

        Parameters
        ----------
        circuit : ThermalCircuit
            The thermal circuit to attach loads to.
        face_nodes : list of str
            Capacitance node name for each face.  Length must equal
            :attr:`num_faces`.
        t_start : datetime64
            Start of the simulation window.
        t_end : datetime64
            End of the simulation window.
        step : timedelta64
            Time step for orbital propagation (used to pre-compute
            solar absorption).
        prefix : str
            Name prefix for the load elements added to the circuit.
            Defaults to ``'thermal'``.

        Returns
        -------
        float
            Simulation duration in seconds, for passing to
            ``circuit.solve()``.
        """
        self._require_spacecraft()
        sc = self._spacecraft

        if len(face_nodes) != self.num_faces:
            raise ValueError(
                f"face_nodes length ({len(face_nodes)}) must match "
                f"num_faces ({self.num_faces})."
            )

        # Propagate orbit
        state = sc.propagate(t_start, t_end, step)
        t = state['t']
        r = state['r']
        v = state['v']

        if len(t) == 0:
            return 0.0

        # Compute orbital environment
        sun = sun_vec_eci(t)
        lit = in_sunlight(r, t, body_radius=sc.central_body_radius)

        # Pre-compute environmental heat loads: (N, M)
        absorbed = self._compute_absorbed_solar(r, v, t, sun, lit)
        absorbed += self._compute_earth_loads(r, v, t, sun, lit)

        # Convert timestamps to seconds from zero
        t_sec = (t - t[0]) / np.timedelta64(1, 'us') * 1e-6
        duration = float(t_sec[-1])

        # Create load functions and register on circuit
        for m in range(self.num_faces):
            absorbed_m = absorbed[:, m].copy()
            t_sec_m = t_sec.copy()
            eps_m = float(self._emissivities[m])
            area_m = float(self._areas[m])

            def _make_load_fn(t_arr, q_arr, eps, area):
                def load_fn(t, T):
                    q_solar = np.interp(t, t_arr, q_arr)
                    q_emit = eps * STEFAN_BOLTZMANN * area * T ** 4
                    return q_solar - q_emit
                return load_fn

            fn = _make_load_fn(t_sec_m, absorbed_m, eps_m, area_m)
            circuit.add_load(f'{prefix}_face_{m}', face_nodes[m], fn)

        return duration


class NormalVectorThermalConfig(AbstractThermalConfig):
    """Thermal config defined by face normal vectors.

    Each face is characterised by an outward-facing normal vector in the
    spacecraft body frame, an area, an IR emissivity, and a solar
    absorptivity.

    **Direct solar** absorption on face *m*:

    .. math::

        Q_{\\mathrm{solar},m} = \\alpha_m \\, A_m \\, \\max(\\hat{n}_m
        \\cdot \\hat{s},\\; 0) \\, S

    Power is zero in eclipse.

    **Earth IR** absorbed on face *m*:

    .. math::

        Q_{\\mathrm{EIR},m} = \\varepsilon_m \\, A_m \\, F_m \\, q_{\\mathrm{EIR}}

    where :math:`F_m = \\max(\\hat{n}_m \\cdot \\hat{r}_{\\mathrm{nadir}}, 0)
    \\, (R_E / r)^2` is the flat-plate view factor to Earth, and
    :math:`q_{\\mathrm{EIR}}` is the average Earth IR flux (240 W m⁻²).
    IR absorptivity equals emissivity (Kirchhoff's law).

    **Earth albedo** absorbed on face *m*:

    .. math::

        Q_{\\mathrm{alb},m} = \\alpha_m \\, A_m \\, F_m \\, a \\, S \\,
        \\max(\\hat{s} \\cdot \\hat{r}_{\\mathrm{zenith}}, 0)

    where *a* is the albedo coefficient (default 0.3) and the final
    cosine term accounts for the solar illumination of the sub-satellite
    point.

    Parameters
    ----------
    normal_vecs : array_like, shape (M, 3)
        Outward-facing normal vectors in the spacecraft body frame.
        Normalised internally.
    areas : array_like, shape (M,)
        Face areas (m²).
    emissivities : array_like, shape (M,)
        IR emissivity of each face, in [0, 1].
    absorptivities : array_like, shape (M,)
        Solar absorptivity of each face, in [0, 1].
    irradiance : float
        Solar irradiance (W m⁻²).  Defaults to 1366 W m⁻².
    earth_ir : float
        Average Earth IR flux (W m⁻²).  Defaults to 240 W m⁻².
    albedo : float
        Earth albedo coefficient, in [0, 1].  Defaults to 0.3.
    """

    def __init__(
        self,
        normal_vecs: npt.ArrayLike,
        areas: npt.ArrayLike,
        emissivities: npt.ArrayLike,
        absorptivities: npt.ArrayLike,
        irradiance: float = 1366.0,
        earth_ir: float = 240.0,
        albedo: float = 0.3,
    ) -> None:
        super().__init__(
            areas, emissivities, absorptivities, irradiance, earth_ir, albedo,
        )

        normals = np.asarray(normal_vecs, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(
                f"normal_vecs must have shape (M, 3), got {normals.shape}"
            )
        if len(normals) != self.num_faces:
            raise ValueError(
                f"normal_vecs length ({len(normals)}) must match "
                f"areas length ({self.num_faces})."
            )

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Normal vectors must be non-zero.")

        self._normals = normals / norms

    @property
    def normals(self) -> npt.NDArray[np.floating]:
        """Face unit normals in the body frame, shape ``(M, 3)``."""
        return self._normals.copy()

    def _compute_absorbed_solar(
        self,
        r: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        sun_eci: np.ndarray,
        lit: np.ndarray,
    ) -> np.ndarray:
        sc = self._spacecraft
        n = len(t)
        m = self.num_faces
        absorbed = np.zeros((n, m), dtype=np.float64)

        sun_2d = np.atleast_2d(sun_eci)

        for k in range(m):
            n_eci = sc.attitude_law.rotate_from_body(
                self._normals[k], r, v, t,
            )
            n_eci_2d = np.atleast_2d(n_eci)
            cos_angle = np.einsum('ij,ij->i', n_eci_2d, sun_2d)
            absorbed[:, k] = (
                self._absorptivities[k]
                * self._areas[k]
                * np.maximum(cos_angle, 0.0)
                * self._irradiance
            )
            absorbed[~lit, k] = 0.0

        return absorbed

    def _compute_earth_loads(
        self,
        r: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        sun_eci: np.ndarray,
        lit: np.ndarray,
    ) -> np.ndarray:
        """Compute Earth IR and albedo loads for each face.

        View factor for face *m* is approximated as

        .. math::

            F_m = \\max(\\hat{n}_m \\cdot \\hat{r}_{\\mathrm{nadir}}, 0)
            \\, (R_E / r)^2

        Earth IR uses emissivity (Kirchhoff's law: IR absorptivity =
        emissivity).  Albedo uses solar absorptivity.
        """
        sc = self._spacecraft
        n_steps = len(t)
        m = self.num_faces
        earth_loads = np.zeros((n_steps, m), dtype=np.float64)

        # Nadir direction (unit vector from spacecraft toward Earth centre)
        r_2d = np.atleast_2d(r)
        r_mag = np.linalg.norm(r_2d, axis=1, keepdims=True)
        nadir = -r_2d / r_mag  # (N, 3)

        # Geometric factor: (R_e / r)^2
        R_e = sc.central_body_radius
        geo_factor = (R_e / r_mag.ravel()) ** 2  # (N,)

        # Solar illumination of sub-satellite point:
        # cos(sun zenith at sub-sat point) = max(ŝ · r̂_zenith, 0)
        # r̂_zenith = -nadir = r̂
        sun_2d = np.atleast_2d(sun_eci)
        zenith = r_2d / r_mag  # (N, 3)
        cos_sun_subsatellite = np.maximum(
            np.einsum('ij,ij->i', sun_2d, zenith), 0.0,
        )  # (N,)

        for k in range(m):
            n_eci = sc.attitude_law.rotate_from_body(
                self._normals[k], r_2d, v, t,
            )
            n_eci_2d = np.atleast_2d(n_eci)

            # View factor: max(n̂ · nadir, 0) × (R_e / r)²
            cos_nadir = np.einsum('ij,ij->i', n_eci_2d, nadir)
            view_factor = np.maximum(cos_nadir, 0.0) * geo_factor  # (N,)

            # Earth IR: ε_m × A_m × F_m × q_earth_ir
            q_ir = (
                self._emissivities[k]
                * self._areas[k]
                * view_factor
                * self._earth_ir
            )

            # Earth albedo: α_m × A_m × F_m × a × S × cos(sun_zenith)
            q_alb = (
                self._absorptivities[k]
                * self._areas[k]
                * view_factor
                * self._albedo
                * self._irradiance
                * cos_sun_subsatellite
            )

            earth_loads[:, k] = q_ir + q_alb

        return earth_loads
