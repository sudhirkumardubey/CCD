"""
Skin friction loss models for impeller
Combining RadComp implementations with TurboFlow registry
"""

import numpy as np
from math import pi, cos, sin, atan, tan, radians
from losses.registry import LossModelRegistry, LossContext


@LossModelRegistry.register("impeller", "jansen_skin_friction")
def skin_friction_jansen(context: LossContext) -> float:
    """
    Skin friction losses according to Jansen
    (RadComp implementation)
    
    Returns
    -------
    float
        Skin friction loss (J/kg)
    """
    geom = context.geometry
    w4 = context.velocity_triangle["w4"]
    tp4 = context.outlet_state
    
    # Hydraulic diameter
    D_h = 4 * geom.r4 * geom.b4 / (2 * geom.r4 + geom.b4)
    
    # Average meridional length
    L_m = pi * (geom.r4 + geom.r2rms) / 2
    
    # Reynolds number
    Re = tp4.D * w4 * D_h / tp4.mu
    
    # Friction coefficient (Blasius correlation)
    if Re > 2300:
        Cf = 0.079 / (Re ** 0.25)
    else:
        Cf = 16 / Re
    
    # Skin friction loss
    dh_sf = 4 * Cf * L_m / D_h * w4**2 / 2
    
    return dh_sf


@LossModelRegistry.register("impeller", "coppage_galvas_skin_friction")
def skin_friction_coppage_galvas(context: LossContext) -> float:
    """
    Skin friction losses according to Coppage & Galvas
    
    Returns
    -------
    float
        Skin friction loss (J/kg)
    """
    geom = context.geometry
    w4 = context.velocity_triangle["w4"]
    w2 = context.velocity_triangle["w2"]
    tp4 = context.outlet_state
    
    # Mean relative velocity
    w_mean = (w2 + w4) / 2
    
    # Blade surface area (approximate)
    L_blade = pi * (geom.r4 + geom. r2rms) / 2
    A_blade = 2 * geom.n_blades * L_blade * geom.b4
    
    # Reynolds number
    D_h = 4 * geom.r4 * geom.b4 / (2 * geom.r4 + geom.b4)
    Re = tp4.D * w_mean * D_h / tp4.mu
    
    # Friction coefficient
    Cf = 0.0412 / (Re ** 0.1925)
    
    # Skin friction loss
    dh_sf = Cf * A_blade / (geom.A4_eff) * w_mean**2
    
    return dh_sf