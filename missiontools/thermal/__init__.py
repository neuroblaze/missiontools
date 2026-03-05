"""
missiontools.thermal
====================
Thermal analysis.

Lumped-parameter thermal network
--------------------------------
:class:`ThermalCircuit`
    Build a network of thermal capacitances, heat sources, and active
    coolers connected by thermal resistances.  Solve transient or
    steady-state thermal response.

:class:`ThermalResult`
    Container for simulation results (time history of node temperatures).
"""

from .thermal_circuit import ThermalCircuit, ThermalResult

__all__ = ['ThermalCircuit', 'ThermalResult']
