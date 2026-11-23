"""PathSafe-EllipseNet: path-level ellipse prediction with differentiable
geometric safety training.

Structure (implementation plan V1.0):

    datasets/   trajectory-level V3 NPZ -> padded batches + path geometry features
    models/     map encoder + coordinate query + LiDAR circular encoder +
                path-geometry encoder + path-level backbone + three heads
    geometry/   torch ellipse algebra, exact box quadratic minimum, SDF,
                differentiable safety loss
    losses/     regression / continuity / total loss
    evaluation/ metrics + exact (NumPy) safety checks
"""

__version__ = "1.0.0"
