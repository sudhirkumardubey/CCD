"""
Disc friction loss models
"""

import numpy as np
from math import pi
from losses.registry import LossModelRegistry, LossContext


@LossModelRegistry.register("impeller", "daily_nece_disc_friction")
def disc_friction_daily_nece(context: LossContext) -> float:
    """
    Disc friction losses according to Daily & Nece
    (RadComp implementation)
    
    Returns
    -------
    float
        Disc friction loss (J/kg)
    """
    geom = context.geometry
    tp4 = context.outlet_state
    m = context.operating_condition.m
    n_rot = context.operating_condition.n_rot
    
    # Reynolds number at disc
    Re_disc = tp4.D * (geom.r4 * n_rot) * geom.r4 / tp4.mu
    
    # Moment coefficient (Daily & Nece correlation)
    if Re_disc < 3e5:
        Cm = 3.7 / (Re_disc ** 0.5)
    else:
        Cm = 0.102 / (Re_disc ** 0.2)
    
    # Disc friction power
    P_disc = Cm * tp4.D * n_rot**3 * geom.r4**5 * pi / 2
    
    # Convert to specific enthalpy loss
    dh_df = P_disc / m
    
    return dh_df