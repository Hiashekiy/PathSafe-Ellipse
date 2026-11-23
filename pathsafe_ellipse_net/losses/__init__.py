"""Loss package."""
from .continuity_loss import continuity_loss
from .regression_loss import (angle_loss, axes_loss, center_loss, double_angle_vector,
                             regression_losses)
from .total_loss import TotalLoss

__all__ = ["TotalLoss", "continuity_loss", "center_loss", "axes_loss", "angle_loss",
           "double_angle_vector", "regression_losses"]
