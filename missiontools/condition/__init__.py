"""
missiontools.condition
======================
Boolean time-domain conditions for use with control logic.

A :class:`Condition` evaluates to a boolean array over time, and is the
primitive building block for control logic such as
:class:`~missiontools.attitude.ConditionAttitudeLaw` (which routes between
attitude laws based on a chain of conditions).

Hierarchy
---------
:class:`AbstractCondition` (ABC)
└── :class:`SpaceGroundAccessCondition` — true when a spacecraft is
    visible from a ground station above a minimum elevation angle.
"""

from .condition import AbstractCondition, SpaceGroundAccessCondition

__all__ = ['AbstractCondition', 'SpaceGroundAccessCondition']
