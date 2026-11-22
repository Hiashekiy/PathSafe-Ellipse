"""PathSafe-Ellipse: path-conditioned safe ellipse supervision dataset.

Coordinate conventions (used everywhere in this package)
-------------------------------------------------------
* Grid arrays are indexed ``occ[row, col]`` with ``row == y`` and ``col == x``.
* A grid cell ``(row=i, col=j)`` is the unit square centred on the integer
  point ``(x=j, y=i)``, i.e. it spans ``[j-0.5, j+0.5] x [i-0.5, i+0.5]``.
* Every geometric quantity in this package is expressed in ``(x, y)`` order.
* Units: distances in this package are in **cells** unless the name says ``_m``.
* A local patch is ``128x128`` centred on cell ``(64, 64)``; the path point is
  always the continuous point ``(64.0, 64.0)``.
"""

__version__ = "0.1.0"
