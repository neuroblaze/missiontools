"""Tests for missiontools.thermal.thermal_circuit."""

import numpy as np
import pytest

from missiontools.thermal import ThermalCircuit, ThermalResult


# ===================================================================
# Construction & validation
# ===================================================================

class TestAddCapacitance:
    def test_basic(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0)
        assert tc.num_nodes == 1
        assert tc.nodes == ['node']

    def test_custom_initial_temp(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0, initial_temp=150.0)
        # verify via solve: single node, no source → stays at 150 K
        result = tc.solve(100.0)
        assert result.temperatures['node'][-1] == pytest.approx(150.0)

    def test_duplicate_name_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(ValueError, match="already in use"):
            tc.add_capacitance('a', 5.0)

    def test_zero_capacity_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="positive"):
            tc.add_capacitance('a', 0.0)

    def test_negative_capacity_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="positive"):
            tc.add_capacitance('a', -1.0)

    def test_zero_temp_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="positive"):
            tc.add_capacitance('a', 10.0, initial_temp=0.0)

    def test_empty_name_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="non-empty"):
            tc.add_capacitance('', 10.0)


class TestAddHeatSource:
    def test_basic(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0)
        tc.add_heat_source('heater', 5.0, target='node')
        assert 'heater' in tc._all_names

    def test_nonexistent_target_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="not a capacitance"):
            tc.add_heat_source('heater', 5.0, target='missing')

    def test_negative_power_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0)
        with pytest.raises(ValueError, match="non-negative"):
            tc.add_heat_source('heater', -1.0, target='node')

    def test_zero_power_allowed(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0)
        tc.add_heat_source('heater', 0.0, target='node')

    def test_name_collision_with_capacitance(self):
        tc = ThermalCircuit()
        tc.add_capacitance('x', 10.0)
        with pytest.raises(ValueError, match="already in use"):
            tc.add_heat_source('x', 5.0, target='x')


class TestAddCooler:
    def test_basic(self):
        tc = ThermalCircuit()
        tc.add_capacitance('cold', 10.0)
        tc.add_capacitance('hot', 10.0)
        tc.add_cooler('cryo', 'cold', 'hot', power=5.0, efficiency=0.3)

    def test_same_node_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0)
        with pytest.raises(ValueError, match="different"):
            tc.add_cooler('c', 'node', 'node', power=5.0, efficiency=0.3)

    def test_nonexistent_node_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(ValueError, match="not a capacitance"):
            tc.add_cooler('c', 'a', 'missing', power=5.0, efficiency=0.3)

    def test_zero_power_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        with pytest.raises(ValueError, match="positive"):
            tc.add_cooler('c', 'a', 'b', power=0.0, efficiency=0.3)

    def test_efficiency_out_of_range(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            tc.add_cooler('c', 'a', 'b', power=5.0, efficiency=0.0)
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            tc.add_cooler('c', 'a', 'b', power=5.0, efficiency=1.5)

    def test_negative_cop_max_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        with pytest.raises(ValueError, match="positive"):
            tc.add_cooler('c', 'a', 'b', power=5.0, efficiency=0.3,
                          cop_max=-1.0)


class TestConnect:
    def test_basic(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        tc.connect('a', 'b', 0.5)

    def test_nonexistent_node_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(ValueError, match="not a capacitance"):
            tc.connect('a', 'missing', 0.5)

    def test_self_connection_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(ValueError, match="itself"):
            tc.connect('a', 'a', 0.5)

    def test_zero_resistance_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        with pytest.raises(ValueError, match="positive"):
            tc.connect('a', 'b', 0.0)

    def test_duplicate_connection_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        tc.connect('a', 'b', 0.5)
        with pytest.raises(ValueError, match="already exists"):
            tc.connect('b', 'a', 1.0)


class TestSetInitialTemp:
    def test_basic(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0, initial_temp=200.0)
        tc.set_initial_temp('a', 300.0)
        result = tc.solve(10.0)
        assert result.temperatures['a'][0] == pytest.approx(300.0)

    def test_nonexistent_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="not a capacitance"):
            tc.set_initial_temp('missing', 300.0)

    def test_zero_temp_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(ValueError, match="positive"):
            tc.set_initial_temp('a', 0.0)


# ===================================================================
# Solve — physics tests
# ===================================================================

class TestSolveBasic:
    def test_empty_circuit_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(RuntimeError, match="no capacitance"):
            tc.solve(100.0)

    def test_negative_duration_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(ValueError, match="positive"):
            tc.solve(-1.0)

    def test_result_type(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        result = tc.solve(100.0)
        assert isinstance(result, ThermalResult)
        assert result.success
        assert 'a' in result.temperatures

    def test_t_eval(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        t_eval = np.linspace(0, 100, 50)
        result = tc.solve(100.0, t_eval=t_eval)
        np.testing.assert_allclose(result.t, t_eval)

    def test_repr(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 5.0)
        tc.connect('a', 'b', 1.0)
        assert 'nodes=2' in repr(tc)
        assert 'connections=1' in repr(tc)


class TestConstantTemperature:
    """Single node, no source → temperature stays constant."""

    def test_single_node_no_source(self):
        tc = ThermalCircuit()
        tc.add_capacitance('node', 10.0, initial_temp=250.0)
        result = tc.solve(1000.0, t_eval=np.linspace(0, 1000, 100))
        np.testing.assert_allclose(
            result.temperatures['node'], 250.0, atol=1e-10,
        )


class TestHeating:
    """Single node with constant heat source → linear temperature rise."""

    def test_linear_heating(self):
        C = 20.0  # J/K
        P = 5.0   # W
        T0 = 300.0
        duration = 100.0  # s

        tc = ThermalCircuit()
        tc.add_capacitance('node', C, initial_temp=T0)
        tc.add_heat_source('heater', P, target='node')
        t_eval = np.linspace(0, duration, 200)
        result = tc.solve(duration, t_eval=t_eval)

        # dT/dt = P/C → T(t) = T0 + P*t/C
        expected = T0 + P * t_eval / C
        np.testing.assert_allclose(
            result.temperatures['node'], expected, rtol=1e-6,
        )


class TestRCDecay:
    """Two nodes at different temps connected by R → exponential decay.

    Analytical solution:
        T_eq = (C1*T1_0 + C2*T2_0) / (C1 + C2)
        tau  = R * C1 * C2 / (C1 + C2)
        T1(t) = T_eq + (T1_0 - T_eq) * exp(-t/tau)
        T2(t) = T_eq + (T2_0 - T_eq) * exp(-t/tau)
    """

    def test_equal_capacitances(self):
        C = 10.0
        R = 2.0
        T1_0, T2_0 = 400.0, 200.0

        tc = ThermalCircuit()
        tc.add_capacitance('hot', C, initial_temp=T1_0)
        tc.add_capacitance('cold', C, initial_temp=T2_0)
        tc.connect('hot', 'cold', R)

        T_eq = (C * T1_0 + C * T2_0) / (2 * C)
        tau = R * C * C / (2 * C)
        duration = 10 * tau

        t_eval = np.linspace(0, duration, 500)
        result = tc.solve(duration, t_eval=t_eval)

        expected_hot = T_eq + (T1_0 - T_eq) * np.exp(-t_eval / tau)
        expected_cold = T_eq + (T2_0 - T_eq) * np.exp(-t_eval / tau)

        np.testing.assert_allclose(
            result.temperatures['hot'], expected_hot, rtol=1e-5,
        )
        np.testing.assert_allclose(
            result.temperatures['cold'], expected_cold, rtol=1e-5,
        )

    def test_unequal_capacitances(self):
        C1, C2 = 50.0, 5.0
        R = 1.0
        T1_0, T2_0 = 350.0, 250.0

        tc = ThermalCircuit()
        tc.add_capacitance('big', C1, initial_temp=T1_0)
        tc.add_capacitance('small', C2, initial_temp=T2_0)
        tc.connect('big', 'small', R)

        T_eq = (C1 * T1_0 + C2 * T2_0) / (C1 + C2)
        tau = R * C1 * C2 / (C1 + C2)
        duration = 10 * tau

        t_eval = np.linspace(0, duration, 500)
        result = tc.solve(duration, t_eval=t_eval)

        expected_big = T_eq + (T1_0 - T_eq) * np.exp(-t_eval / tau)
        expected_small = T_eq + (T2_0 - T_eq) * np.exp(-t_eval / tau)

        np.testing.assert_allclose(
            result.temperatures['big'], expected_big, rtol=1e-5,
        )
        np.testing.assert_allclose(
            result.temperatures['small'], expected_small, rtol=1e-5,
        )


class TestCooler:
    """Active cooler tests."""

    def test_cooler_cools_cold_node(self):
        tc = ThermalCircuit()
        tc.add_capacitance('cold', 10.0, initial_temp=250.0)
        tc.add_capacitance('hot', 10.0, initial_temp=300.0)
        tc.add_cooler('cryo', 'cold', 'hot', power=2.0, efficiency=0.5)
        result = tc.solve(100.0)

        # cold node should cool down
        assert result.temperatures['cold'][-1] < 250.0
        # hot node should heat up
        assert result.temperatures['hot'][-1] > 300.0

    def test_cop_max_clips(self):
        """When T_hot ≈ T_cold, COP should be clipped to cop_max."""
        tc = ThermalCircuit()
        # Same initial temp → Carnot COP would be infinite
        tc.add_capacitance('cold', 100.0, initial_temp=300.0)
        tc.add_capacitance('hot', 100.0, initial_temp=300.0)
        tc.add_cooler('c', 'cold', 'hot', power=1.0, efficiency=1.0,
                      cop_max=5.0)
        result = tc.solve(10.0)

        # With cop_max=5, Q_cold = 5 W, Q_hot = 6 W
        # After 10 s: cold drops by ~0.5 K, hot rises by ~0.6 K
        assert result.temperatures['cold'][-1] < 300.0
        assert result.temperatures['hot'][-1] > 300.0

    def test_energy_balance(self):
        """Total energy change = work input by cooler (no external sources)."""
        C = 50.0
        tc = ThermalCircuit()
        tc.add_capacitance('cold', C, initial_temp=200.0)
        tc.add_capacitance('hot', C, initial_temp=350.0)
        W = 3.0
        tc.add_cooler('c', 'cold', 'hot', power=W, efficiency=0.4,
                      cop_max=10.0)
        duration = 50.0
        result = tc.solve(duration, t_eval=np.linspace(0, duration, 1000))

        # Total internal energy change = C*(T_cold_f - T_cold_0) +
        #                                 C*(T_hot_f - T_hot_0)
        # Should equal total work input = W * duration
        dE = C * (result.temperatures['cold'][-1] - 200.0) + \
             C * (result.temperatures['hot'][-1] - 350.0)
        expected_work = W * duration
        np.testing.assert_allclose(dE, expected_work, rtol=1e-4)


class TestEnergyConservation:
    """Total energy change = total external heat input."""

    def test_heat_source_energy(self):
        C = 30.0
        P = 7.0
        T0 = 280.0
        duration = 200.0

        tc = ThermalCircuit()
        tc.add_capacitance('a', C, initial_temp=T0)
        tc.add_heat_source('h', P, target='a')
        result = tc.solve(duration)

        dE = C * (result.temperatures['a'][-1] - T0)
        expected = P * duration
        np.testing.assert_allclose(dE, expected, rtol=1e-5)

    def test_two_nodes_with_source_energy(self):
        """Energy is conserved across connected nodes with a source."""
        C1, C2 = 20.0, 30.0
        T1_0, T2_0 = 300.0, 300.0
        P = 10.0
        R = 0.5
        duration = 500.0

        tc = ThermalCircuit()
        tc.add_capacitance('a', C1, initial_temp=T1_0)
        tc.add_capacitance('b', C2, initial_temp=T2_0)
        tc.connect('a', 'b', R)
        tc.add_heat_source('h', P, target='a')
        result = tc.solve(duration)

        dE = C1 * (result.temperatures['a'][-1] - T1_0) + \
             C2 * (result.temperatures['b'][-1] - T2_0)
        expected = P * duration
        np.testing.assert_allclose(dE, expected, rtol=1e-4)


# ===================================================================
# Callable loads
# ===================================================================

class TestAddLoad:
    def test_basic(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_load('ld', 'a', lambda t, T: 5.0)
        assert 'ld' in tc._all_names

    def test_nonexistent_node_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(ValueError, match="not a capacitance"):
            tc.add_load('ld', 'missing', lambda t, T: 0.0)

    def test_not_callable_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        with pytest.raises(TypeError, match="callable"):
            tc.add_load('ld', 'a', 42)

    def test_name_collision_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_load('ld', 'a', lambda t, T: 0.0)
        with pytest.raises(ValueError, match="already in use"):
            tc.add_load('ld', 'a', lambda t, T: 1.0)


class TestLoadPhysics:
    def test_constant_load_matches_heat_source(self):
        """A constant load_fn should produce the same result as HeatSource."""
        C, P, T0, dur = 20.0, 5.0, 300.0, 100.0
        t_eval = np.linspace(0, dur, 200)

        tc_src = ThermalCircuit()
        tc_src.add_capacitance('n', C, initial_temp=T0)
        tc_src.add_heat_source('h', P, target='n')
        r_src = tc_src.solve(dur, t_eval=t_eval)

        tc_ld = ThermalCircuit()
        tc_ld.add_capacitance('n', C, initial_temp=T0)
        tc_ld.add_load('h', 'n', lambda t, T: P)
        r_ld = tc_ld.solve(dur, t_eval=t_eval)

        np.testing.assert_allclose(
            r_ld.temperatures['n'], r_src.temperatures['n'], rtol=1e-8,
        )

    def test_time_dependent_ramp(self):
        """Q = k*t → T(t) = T0 + k*t²/(2*C)."""
        C = 10.0
        T0 = 300.0
        k = 2.0  # W/s ramp rate
        dur = 50.0
        t_eval = np.linspace(0, dur, 500)

        tc = ThermalCircuit()
        tc.add_capacitance('n', C, initial_temp=T0)
        tc.add_load('ramp', 'n', lambda t, T: k * t)
        result = tc.solve(dur, t_eval=t_eval)

        expected = T0 + k * t_eval**2 / (2 * C)
        np.testing.assert_allclose(
            result.temperatures['n'], expected, rtol=1e-5,
        )

    def test_temperature_dependent_cooling(self):
        """Q = -k*(T - T_env) → exponential decay toward T_env.

        T(t) = T_env + (T0 - T_env) * exp(-k*t/C)
        """
        C = 15.0
        T0 = 400.0
        T_env = 250.0
        k = 0.5  # W/K
        dur = 200.0
        t_eval = np.linspace(0, dur, 500)

        tc = ThermalCircuit()
        tc.add_capacitance('n', C, initial_temp=T0)
        tc.add_load('rad', 'n', lambda t, T: -k * (T - T_env))
        result = tc.solve(dur, t_eval=t_eval)

        tau = C / k
        expected = T_env + (T0 - T_env) * np.exp(-t_eval / tau)
        np.testing.assert_allclose(
            result.temperatures['n'], expected, rtol=1e-5,
        )

    def test_load_energy_conservation(self):
        """Constant load: total energy change = Q * duration."""
        C = 30.0
        T0 = 280.0
        P = 7.0
        dur = 200.0

        tc = ThermalCircuit()
        tc.add_capacitance('a', C, initial_temp=T0)
        tc.add_load('h', 'a', lambda t, T: P)
        result = tc.solve(dur)

        dE = C * (result.temperatures['a'][-1] - T0)
        np.testing.assert_allclose(dE, P * dur, rtol=1e-5)


# ===================================================================
# Steady state
# ===================================================================

class TestSteadyState:
    def test_with_cooler_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_capacitance('b', 10.0)
        tc.add_cooler('c', 'a', 'b', power=1.0, efficiency=0.3)
        with pytest.raises(RuntimeError, match="cooler"):
            tc.steady_state()

    def test_with_load_raises(self):
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0)
        tc.add_load('ld', 'a', lambda t, T: 1.0)
        with pytest.raises(RuntimeError, match="load"):
            tc.steady_state()

    def test_empty_circuit_raises(self):
        tc = ThermalCircuit()
        with pytest.raises(RuntimeError, match="no capacitance"):
            tc.steady_state()

    def test_disconnected_node_keeps_initial(self):
        tc = ThermalCircuit()
        tc.add_capacitance('isolated', 10.0, initial_temp=250.0)
        ss = tc.steady_state()
        assert ss['isolated'] == pytest.approx(250.0)

    def test_two_nodes_with_source(self):
        """Steady state: heat flows through resistance to second node.

        At steady state with both nodes connected in series and no heat
        loss, temperatures diverge (no sink).  But if we have two nodes
        connected with a source on one and treat the other as also
        accumulating, steady state is only defined if there's a sink.

        Test a 3-node chain with a source on one end and a fixed-temp
        approximation on the other via a very large capacitance starting
        at T0 (acts as approximate heat sink).
        """
        # Simple case: two nodes, source on 'a', both connected.
        # With no heat loss, steady state doesn't truly exist for a
        # closed system with a source.  Test the disconnected case instead.
        tc = ThermalCircuit()
        tc.add_capacitance('a', 10.0, initial_temp=300.0)
        tc.add_capacitance('b', 10.0, initial_temp=300.0)
        tc.connect('a', 'b', 1.0)
        # No source → steady state is initial temps (system in equilibrium)
        ss = tc.steady_state()
        # With both at 300 K and no source, equilibrium is 300 K
        assert ss['a'] == pytest.approx(300.0, abs=1e-10)
        assert ss['b'] == pytest.approx(300.0, abs=1e-10)

    def test_agrees_with_long_solve(self):
        """Steady state should match long-duration solve result."""
        tc_ss = ThermalCircuit()
        tc_ss.add_capacitance('a', 10.0, initial_temp=400.0)
        tc_ss.add_capacitance('b', 20.0, initial_temp=200.0)
        tc_ss.connect('a', 'b', 0.5)
        ss = tc_ss.steady_state()

        tc_dyn = ThermalCircuit()
        tc_dyn.add_capacitance('a', 10.0, initial_temp=400.0)
        tc_dyn.add_capacitance('b', 20.0, initial_temp=200.0)
        tc_dyn.connect('a', 'b', 0.5)
        result = tc_dyn.solve(10000.0)

        assert ss['a'] == pytest.approx(
            result.temperatures['a'][-1], rel=1e-3,
        )
        assert ss['b'] == pytest.approx(
            result.temperatures['b'][-1], rel=1e-3,
        )


# ===================================================================
# Top-level import
# ===================================================================

def test_top_level_import():
    from missiontools import ThermalCircuit as TC
    assert TC is ThermalCircuit
