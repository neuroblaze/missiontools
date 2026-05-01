"""
missiontools.sensor
===================
Instrument sensor classes for spacecraft field-of-view modelling.

The base class is :class:`AbstractSensor`, with concrete subclasses for
each sensor geometry:

- :class:`ConicSensor` — sensor with a conical field of view, defined by a
  half-angle.  The boresight can be driven by an independent
  :class:`~missiontools.AbstractAttitudeLaw` or fixed in the spacecraft body
  frame.
- :class:`RectangularSensor` — sensor with a rectangular (pyramidal) field
  of view, defined by two half-angles in orthogonal planes.
"""

from .sensor_law import AbstractSensor, ConicSensor, RectangularSensor

__all__ = ["AbstractSensor", "ConicSensor", "RectangularSensor"]
