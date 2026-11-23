"""Geometry package: torch ellipse algebra, exact box quadratic, SDF, safety loss."""
from .box_quadratic_torch import (box_quadratic_min_torch, ellipse_quadratic,
                                  invert_shape, midpoint_metric, rotation,
                                  shape_axes_angle, shape_matrix)
from .ellipse import (decode_angle, decode_axes, decode_center, decode_ellipse,
                      ray_directions, ray_limits)
from .safety_loss import (SafetyCriterion, SafetyParts, border_loss,
                          center_free_loss, ellipse_safe_loss)
from .sdf import sample_sdf, signed_distance_field

__all__ = ["box_quadratic_min_torch", "ellipse_quadratic", "invert_shape",
           "midpoint_metric", "rotation", "shape_axes_angle", "shape_matrix",
           "decode_angle", "decode_axes", "decode_center", "decode_ellipse",
           "ray_directions", "ray_limits", "SafetyCriterion", "SafetyParts",
           "border_loss", "center_free_loss", "ellipse_safe_loss", "sample_sdf",
           "signed_distance_field"]
