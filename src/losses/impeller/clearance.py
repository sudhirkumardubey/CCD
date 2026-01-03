"""
Tip clearance loss models
"""

import numpy as np
from math import pi, atan, tan
from losses.registry import LossModelRegistry, LossContext


@LossModelRegistry.register("impeller", "jansen_clearance")
def clearance_jansen(context: LossContext) -> float:
    """
    Tip clearance losses according to Jansen
    (RadComp implementation)
    """
    geom = context.geometry
    c4t = context.velocity_triangle["v_t4"]
    n_rot = context.operating_condition.n_rot
    tp4 = context.outlet_state
    
    # Clearance loss (Jansen correlation)
    # Includes leakage flow effects
    dh_cl = 2 * tp4.D * c4t**2 * geom.t_cl / geom.b4
    
    return dh_cl


@LossModelRegistry.register("impeller", "brasz_clearance")
def clearance_brasz(context: LossContext) -> float:
    """
    Tip clearance losses according to Brasz
    """
    geom = context.geometry
    v_t4 = context.velocity_triangle["v_t4"]
    n_rot = context. operating_condition.n_rot
    
    # Brasz correlation (simpler form)
    u4 = geom.r4 * n_rot
    dh_cl = 0.6 * (geom.t_cl / geom.b4) * v_t4 * u4
    
    return dh_cl