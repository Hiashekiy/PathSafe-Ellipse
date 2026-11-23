"""PathSafe-Ellipse V3: whole-scene sampling + distance-field centreline +
fast fixed-centre 2x2 max-det ellipse labels, collected **per trajectory**.

V3 keeps the V2/V2.1 geometry (A*, route-constrained centre line, ellipse solver)
and changes only the sampling unit: one NPZ = one complete trajectory, with a
256x256 inflated occupancy map, the resampled A* path, the optimised centre line,
per-point LiDAR ranges and per-point ellipse labels.  No 64x64 patch, no clearance
feature, no single-point samples.
"""

__version__ = "3.0.0"
