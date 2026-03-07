"""Lumped-parameter thermal network solver.

Build a network of thermal capacitances connected by thermal resistances,
with optional heat sources and active coolers.  Solve the transient thermal
response using scipy's ODE integrators.

All units are SI: temperatures in kelvin, thermal capacitance in J/K,
thermal resistance in K/W, power in watts, time in seconds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Internal element dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _Capacitance:
    capacity: float      # J/K
    initial_temp: float  # K


@dataclass
class _HeatSource:
    power: float  # W
    target: str   # capacitance node name


@dataclass
class _Cooler:
    cold_node: str
    hot_node: str
    power: float       # input electrical work (W)
    efficiency: float  # fraction of Carnot COP (0, 1]
    cop_max: float     # clips effective COP


@dataclass
class _Load:
    node: str                                    # capacitance node name
    load_fn: Callable[[float, float], float]     # (t_seconds, T_node_K) -> watts


@dataclass
class _Connection:
    node_a: str
    node_b: str
    resistance: float  # K/W


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ThermalResult:
    """Result of a thermal circuit simulation.

    Attributes
    ----------
    t : ndarray, shape (M,)
        Time points (s).
    temperatures : dict[str, ndarray]
        Mapping from node name to temperature history array, shape (M,).
    success : bool
        Whether the ODE solver converged.
    message : str
        Solver status message.
    """

    def __init__(
        self,
        t: np.ndarray,
        temperatures: dict[str, np.ndarray],
        success: bool,
        message: str,
    ) -> None:
        self.t = t
        self.temperatures = temperatures
        self.success = success
        self.message = message

    def __repr__(self) -> str:
        return (
            f"ThermalResult(nodes={list(self.temperatures.keys())}, "
            f"steps={len(self.t)}, success={self.success})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ThermalCircuit:
    """Lumped-parameter thermal network.

    Example
    -------
    >>> circuit = ThermalCircuit()
    >>> circuit.add_capacitance('bench', 50.0, initial_temp=300.0)
    >>> circuit.add_capacitance('detector', 5.0, initial_temp=300.0)
    >>> circuit.connect('bench', 'detector', 0.5)
    >>> circuit.add_heat_source('electronics', 10.0, target='bench')
    >>> result = circuit.solve(3600.0)
    """

    def __init__(self) -> None:
        self._capacitances: dict[str, _Capacitance] = {}
        self._heat_sources: dict[str, _HeatSource] = {}
        self._coolers: dict[str, _Cooler] = {}
        self._loads: dict[str, _Load] = {}
        self._connections: list[_Connection] = []
        self._all_names: set[str] = set()

    # --- helpers ---

    def _check_name_available(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Element name must be a non-empty string.")
        if name in self._all_names:
            raise ValueError(f"Name '{name}' is already in use.")

    def _require_capacitance(self, name: str) -> None:
        if name not in self._capacitances:
            raise ValueError(
                f"'{name}' is not a capacitance node. "
                f"Existing capacitances: {list(self._capacitances.keys())}"
            )

    # --- build network ---

    def add_capacitance(
        self, name: str, capacity: float, initial_temp: float = 293.15,
    ) -> None:
        """Add a thermal mass node.

        Parameters
        ----------
        name : str
            Unique identifier for this element.
        capacity : float
            Thermal capacitance (J/K).  Must be positive.
        initial_temp : float
            Initial temperature (K).  Defaults to 293.15 K (20 °C).
        """
        self._check_name_available(name)
        capacity = float(capacity)
        initial_temp = float(initial_temp)
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}.")
        if initial_temp <= 0:
            raise ValueError(
                f"Initial temperature must be positive (K), got {initial_temp}."
            )
        self._capacitances[name] = _Capacitance(capacity, initial_temp)
        self._all_names.add(name)

    def add_heat_source(
        self, name: str, power: float, target: str,
    ) -> None:
        """Add a constant heat source injecting into a capacitance node.

        Parameters
        ----------
        name : str
            Unique identifier for this element.
        power : float
            Heat dissipation (W).  Must be non-negative.
        target : str
            Name of the capacitance node receiving the heat.
        """
        self._check_name_available(name)
        power = float(power)
        if power < 0:
            raise ValueError(f"Power must be non-negative, got {power}.")
        self._require_capacitance(target)
        self._heat_sources[name] = _HeatSource(power, target)
        self._all_names.add(name)

    def add_cooler(
        self,
        name: str,
        cold_node: str,
        hot_node: str,
        power: float,
        efficiency: float,
        cop_max: float = 20.0,
    ) -> None:
        """Add an active cooler (heat pump) between two capacitance nodes.

        The cooler extracts heat from *cold_node* and rejects it to
        *hot_node*.  Its coefficient of performance (COP) is computed as::

            COP = min(efficiency * T_cold / (T_hot - T_cold), cop_max)

        where *efficiency* is the fraction of the Carnot COP (not the COP
        itself).  An efficiency of 1.0 means the cooler operates at its
        Carnot limit; 0.5 means half the Carnot COP.

        Parameters
        ----------
        name : str
            Unique identifier for this element.
        cold_node : str
            Capacitance node from which heat is extracted.
        hot_node : str
            Capacitance node to which heat is rejected.
        power : float
            Electrical input power to the cooler (W).  Must be positive.
        efficiency : float
            Fraction of Carnot COP, in (0, 1].
        cop_max : float
            Maximum COP to prevent divergence when T_hot ≈ T_cold.
            Defaults to 20.0.
        """
        self._check_name_available(name)
        self._require_capacitance(cold_node)
        self._require_capacitance(hot_node)
        power = float(power)
        efficiency = float(efficiency)
        cop_max = float(cop_max)
        if cold_node == hot_node:
            raise ValueError("cold_node and hot_node must be different.")
        if power <= 0:
            raise ValueError(f"Power must be positive, got {power}.")
        if not (0 < efficiency <= 1):
            raise ValueError(
                f"Efficiency must be in (0, 1], got {efficiency}."
            )
        if cop_max <= 0:
            raise ValueError(f"cop_max must be positive, got {cop_max}.")
        self._coolers[name] = _Cooler(
            cold_node, hot_node, power, efficiency, cop_max,
        )
        self._all_names.add(name)

    def add_load(
        self,
        name: str,
        node: str,
        load_fn: Callable[[float, float], float],
    ) -> None:
        """Add a time- and temperature-dependent heat load to a node.

        This is the general mechanism for coupling external physics (e.g.
        orbital heat fluxes, radiative emission) into the thermal network.

        Parameters
        ----------
        name : str
            Unique identifier for this element.
        node : str
            Target capacitance node.
        load_fn : callable
            Signature ``(t, T) -> Q`` where *t* is time (s), *T* is the
            node temperature (K), and *Q* is the heat load (W).
            Positive values heat the node; negative values cool it.
        """
        self._check_name_available(name)
        self._require_capacitance(node)
        if not callable(load_fn):
            raise TypeError("load_fn must be callable.")
        self._loads[name] = _Load(node, load_fn)
        self._all_names.add(name)

    def connect(
        self, node_a: str, node_b: str, resistance: float,
    ) -> None:
        """Connect two capacitance nodes with a thermal resistance.

        Parameters
        ----------
        node_a, node_b : str
            Capacitance node names.
        resistance : float
            Thermal resistance (K/W).  Must be positive.
        """
        self._require_capacitance(node_a)
        self._require_capacitance(node_b)
        if node_a == node_b:
            raise ValueError("Cannot connect a node to itself.")
        resistance = float(resistance)
        if resistance <= 0:
            raise ValueError(
                f"Resistance must be positive, got {resistance}."
            )
        pair = frozenset((node_a, node_b))
        for conn in self._connections:
            if frozenset((conn.node_a, conn.node_b)) == pair:
                raise ValueError(
                    f"Connection between '{node_a}' and '{node_b}' "
                    f"already exists."
                )
        self._connections.append(_Connection(node_a, node_b, resistance))

    def set_initial_temp(self, name: str, temp: float) -> None:
        """Override the initial temperature of a capacitance node.

        Parameters
        ----------
        name : str
            Capacitance node name.
        temp : float
            Temperature (K).  Must be positive.
        """
        self._require_capacitance(name)
        temp = float(temp)
        if temp <= 0:
            raise ValueError(
                f"Temperature must be positive (K), got {temp}."
            )
        self._capacitances[name].initial_temp = temp

    # --- system assembly ---

    def _build_system(self) -> tuple[
        list[str],       # node_names
        np.ndarray,      # C_vec (N,)
        np.ndarray,      # G (N, N) conductance matrix
        np.ndarray,      # Q_src (N,) source power vector
        list[tuple[int, int, float, float, float]],  # cooler specs
        list[tuple[int, Callable[[float, float], float]]],  # load specs
        np.ndarray,      # T0 (N,) initial temperatures
    ]:
        node_names = list(self._capacitances.keys())
        n = len(node_names)
        idx = {name: i for i, name in enumerate(node_names)}

        C_vec = np.array([self._capacitances[n].capacity for n in node_names])
        T0 = np.array(
            [self._capacitances[n].initial_temp for n in node_names],
        )

        G = np.zeros((n, n))
        for conn in self._connections:
            i, j = idx[conn.node_a], idx[conn.node_b]
            g = 1.0 / conn.resistance
            G[i, j] += g
            G[j, i] += g

        Q_src = np.zeros(n)
        for hs in self._heat_sources.values():
            Q_src[idx[hs.target]] += hs.power

        cooler_specs = []
        for cooler in self._coolers.values():
            cooler_specs.append((
                idx[cooler.cold_node],
                idx[cooler.hot_node],
                cooler.power,
                cooler.efficiency,
                cooler.cop_max,
            ))

        load_specs = [
            (idx[load.node], load.load_fn)
            for load in self._loads.values()
        ]

        return node_names, C_vec, G, Q_src, cooler_specs, load_specs, T0

    # --- ODE right-hand side ---

    @staticmethod
    def _rhs(
        t: float,
        T: np.ndarray,
        C_vec: np.ndarray,
        G: np.ndarray,
        G_sum: np.ndarray,
        Q_src: np.ndarray,
        cooler_specs: list[tuple[int, int, float, float, float]],
        load_specs: list[tuple[int, Callable[[float, float], float]]],
    ) -> np.ndarray:
        # conductive flows
        Q = G @ T - G_sum * T + Q_src

        # cooler contributions
        for cold_idx, hot_idx, W, eff, cop_max in cooler_specs:
            delta_T = T[hot_idx] - T[cold_idx]
            if delta_T > 0:
                carnot_cop = T[cold_idx] / delta_T
                cop = min(eff * carnot_cop, cop_max)
            else:
                cop = cop_max
            Q_cold = cop * W
            Q_hot = Q_cold + W
            Q[cold_idx] -= Q_cold
            Q[hot_idx] += Q_hot

        # callable load contributions
        for node_idx, load_fn in load_specs:
            Q[node_idx] += load_fn(t, T[node_idx])

        return Q / C_vec

    # --- solve ---

    def solve(
        self,
        duration: float,
        *,
        method: str = 'Radau',
        max_step: float | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        t_eval: npt.ArrayLike | None = None,
    ) -> ThermalResult:
        """Solve the transient thermal response.

        Parameters
        ----------
        duration : float
            Simulation duration (s).  Must be positive.
        method : str
            ``scipy.integrate.solve_ivp`` method.  Defaults to ``'Radau'``
            (implicit, suitable for stiff thermal systems).
        max_step : float, optional
            Maximum integration step (s).
        rtol, atol : float
            Relative and absolute tolerances for the ODE solver.
        t_eval : array_like, optional
            Times (s) at which to store the solution.  If *None*, the
            solver chooses its own output times.

        Returns
        -------
        ThermalResult
        """
        if not self._capacitances:
            raise RuntimeError("Circuit has no capacitance nodes.")
        duration = float(duration)
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}.")

        (node_names, C_vec, G, Q_src, cooler_specs,
         load_specs, T0) = self._build_system()
        G_sum = G.sum(axis=1)

        kwargs: dict = {
            'method': method,
            'rtol': rtol,
            'atol': atol,
            'dense_output': False,
        }
        if max_step is not None:
            kwargs['max_step'] = float(max_step)
        if t_eval is not None:
            kwargs['t_eval'] = np.asarray(t_eval, dtype=float)

        sol = solve_ivp(
            self._rhs,
            (0.0, duration),
            T0,
            args=(C_vec, G, G_sum, Q_src, cooler_specs, load_specs),
            **kwargs,
        )

        temperatures = {
            name: sol.y[i] for i, name in enumerate(node_names)
        }

        result = ThermalResult(
            t=sol.t,
            temperatures=temperatures,
            success=sol.success,
            message=sol.message,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        return result

    # --- steady state ---

    def steady_state(self) -> dict[str, float]:
        """Compute steady-state temperatures (linear systems only).

        For circuits with only capacitances, resistances, and heat sources
        (no coolers), solves the linear system directly.

        Returns
        -------
        dict[str, float]
            Node name to steady-state temperature (K).

        Raises
        ------
        RuntimeError
            If the circuit contains coolers.
        """
        if self._coolers:
            raise RuntimeError(
                "steady_state() does not support coolers. "
                "Use solve() with a long duration instead."
            )
        if self._loads:
            raise RuntimeError(
                "steady_state() does not support callable loads. "
                "Use solve() with a long duration instead."
            )
        if not self._capacitances:
            raise RuntimeError("Circuit has no capacitance nodes.")

        node_names, C_vec, G, Q_src, _, _, T0 = self._build_system()
        Q_src = Q_src.copy()
        n = len(node_names)
        G_sum = G.sum(axis=1)

        # At steady state: G @ T - diag(G_sum) @ T + Q_src = 0
        # => (G - diag(G_sum)) @ T = -Q_src
        # This is a graph Laplacian (rows sum to zero) so it is always
        # singular for each connected component.  We resolve this by
        # finding connected components and, for each one:
        #  - If net heat input is nonzero → no finite steady state exists
        #  - If net heat input is zero → add energy conservation constraint
        #    (sum C_i T_i = sum C_i T_i_0) by replacing one row
        A = G - np.diag(G_sum)

        # Find connected components via adjacency
        adj = G > 0
        visited = np.zeros(n, dtype=bool)
        components: list[list[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            comp = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp.append(node)
                for nb in range(n):
                    if adj[node, nb] and not visited[nb]:
                        stack.append(nb)
            components.append(comp)

        for comp in components:
            net_q = sum(Q_src[i] for i in comp)
            if abs(net_q) > 1e-12:
                names = [node_names[i] for i in comp]
                raise RuntimeError(
                    f"No finite steady state: nodes {names} have net "
                    f"heat input {net_q:.4g} W with no heat sink."
                )
            # Replace one row with the energy conservation constraint:
            # sum(C_i * T_i) = sum(C_i * T_i_0)
            pivot = comp[0]
            A[pivot, :] = 0.0
            for i in comp:
                A[pivot, i] = C_vec[i]
            Q_src[pivot] = -sum(C_vec[i] * T0[i] for i in comp)

        T_ss = np.linalg.solve(A, -Q_src)
        return {name: float(T_ss[i]) for i, name in enumerate(node_names)}

    # --- inspection ---

    @property
    def nodes(self) -> list[str]:
        """Names of all capacitance nodes."""
        return list(self._capacitances.keys())

    @property
    def num_nodes(self) -> int:
        """Number of capacitance nodes."""
        return len(self._capacitances)

    def __repr__(self) -> str:
        parts = [f"ThermalCircuit(nodes={self.num_nodes}"]
        if self._heat_sources:
            parts.append(f"sources={len(self._heat_sources)}")
        if self._coolers:
            parts.append(f"coolers={len(self._coolers)}")
        if self._loads:
            parts.append(f"loads={len(self._loads)}")
        parts.append(f"connections={len(self._connections)}")
        return ", ".join(parts) + ")"
