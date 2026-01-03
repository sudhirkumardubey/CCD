"""
Recirculation loss models
"""

import numpy as np
from math import exp, radians, tan
from losses.registry import LossModelRegistry, LossContext


@LossModelRegistry.register("impeller", "rodgers_recirculation")
def recirculation_rodgers(context: LossContext) -> float:
    """
    Recirculation losses according to Rodgers
    (RadComp implementation)
    
    Returns
    -------
    float
        Recirculation loss (J/kg)
    """
    geom = context.geometry
    Df = context.velocity_triangle. get("diffusion_factor", 0.0)
    alpha = context.velocity_triangle.get("alpha4", 0.0)
    n_rot = context.operating_condition.n_rot
    
    # Recirculation factor
    K_rc = 0.02 if Df < 0.4 else 0.02 * exp(3 * (Df - 0.4))
    
    # Recirculation loss
    u4 = geom.r4 * n_rot
    v_t4 = u4 * tan(radians(alpha))
    dh_rc = K_rc * v_t4**2
    
    return dh_rc


@LossModelRegistry.register("impeller", "aungier_recirculation")
def recirculation_aungier(context: LossContext) -> float:
    """
    Recirculation losses according to Aungier
    """
    geom = context.geometry
    phi = context.velocity_triangle.get("flow_coefficient", 0.0)
    n_rot = context.operating_condition.n_rot
    
    # Critical flow coefficient
    phi_crit = 0.3  # Typical value, should be calibrated
    
    if phi < phi_crit: 
        # Recirculation occurs
        K_rc = 0.1 * (phi_crit - phi) / phi_crit
        u4 = geom.r4 * n_rot
        dh_rc = K_rc * u4**2
    else: 
        dh_rc = 0.0
    
    return dh_rc