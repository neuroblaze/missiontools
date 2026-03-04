"""
missiontools.power
==================
Power budget and eclipse analysis.

Planned functionality
---------------------
- Solar array sizing
- Battery sizing and depth of discharge
- Eclipse duration calculation
- End-of-life power degradation
- Power mode / duty cycle analysis
"""

from .solar_config import AbstractSolarConfig, NormalVectorSolarConfig

__all__ = ['AbstractSolarConfig', 'NormalVectorSolarConfig']
