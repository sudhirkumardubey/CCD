"""
Blade loading loss models
"""

import numpy as np
from losses.registry import LossModelRegistry, LossContext


@LossModelRegistry.register("impeller", "rodgers_blade_loading")
def blade_loading_rodgers(context:  LossContext) -> float:
    """
    Blade loading losses according to Rodgers
    (RadComp implementation)
    
    Returns
    -------
    float
        Blade loading loss (J/kg)
    """
    geom = context.geometry
    Df = context.velocity_triangle. get("diffusion_factor", 0.0)
    n_rot = context.operating_condition.n_rot
    
    # Rodgers correlation
    dh_bl = 0.05 * Df**2 * (geom.r4 * n_rot)**2
    
    return dh_bl


@LossModelRegistry.register("impeller", "aungier_blade_loading")
def blade_loading_aungier(context: LossContext) -> float:
    """
    Blade loading losses according to Aungier
    
    Returns
    -------
    float
        Blade loading loss (J/kg)
    """
    geom = context.geometry
    v_t4 = context.velocity_triangle["v_t4"]
    n_rot = context.operating_condition.n_rot
    
    # Blade loading parameter
    L_b = np.pi / 8 * (2 * geom.r4 - geom.r2rms - geom.b4)
    delta_W = 2 * np.pi * 2 * geom.r4 * v_t4 / (geom.n_blades * L_b)
    
    # Aungier correlation
    dh_bl = delta_W**2 / 48
    
    return dh_bl