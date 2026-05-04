"""
missiontools.comm
=================
Link budget analysis and interference analysis.

Antenna classes
---------------
:class:`AbstractAntenna`
    Base class for antennas attachable to Spacecraft or GroundStation.
:class:`IsotropicAntenna`
    Constant-gain antenna (direction-independent).
:class:`SymmetricAntenna`
    Axially symmetric antenna defined by a gain-vs-angle table.

Link budget
-----------
:class:`Link`
    RF link between two antennas.  Computes link margin via
    :meth:`~Link.link_margin`.

Interference analysis
---------------------
:class:`InterferenceAnalysis`
    Analyse interference risk between space networks.
"""

from .antenna import AbstractAntenna, IsotropicAntenna, SymmetricAntenna
from .interference import InterferenceAnalysis
from .link import Link

__all__ = [
    "AbstractAntenna",
    "InterferenceAnalysis",
    "IsotropicAntenna",
    "Link",
    "SymmetricAntenna",
]
