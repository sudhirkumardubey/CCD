"""Wake mixing loss correlations (author-named)."""

import numpy as np
from math import tan, radians
from ..registry import LossModelRegistry, LossContext


@LossModelRegistry.register("impeller", "johnston_dean_mixing")
def mixing_johnston_dean(context: LossContext) -> float:
    """Johnston & Dean wake mixing (Oh)."""
    alpha_out = context.velocity_triangle["alpha_out"]  # degrees
    v_out = context.velocity_triangle["v_out"]
    wake_width = context.geometry.wake_width  # factor, e.g. 0.366
    b_out = context.geometry.b_out
    width_diffuser = b_out  # Assume no change
    b_star = width_diffuser / b_out
    
    dh_mix = (1 / (1 + tan(radians(alpha_out))**2) * 
              ((1 - wake_width - b_star) / (1 - wake_width))**2 * 
              0.5 * v_out**2)
    return dh_mix


@LossModelRegistry.register("impeller", "aungier_mixing")
def mixing_aungier(context: LossContext) -> float:
    """Aungier/Zhang wake mixing."""
    geom = context.geometry
    v_m_out = context.velocity_triangle["v_m_out"]
    w_t_out = context.velocity_triangle["w_t_out"]
    w_in = context.velocity_triangle["w_in"]
    w_out = context.velocity_triangle["w_out"]
    delta_W = context.velocity_triangle.get("delta_W", 0.0)
    
    W_out = np.sqrt((v_m_out * geom.A_out / (2 * np.pi * geom.r_out * geom.b_out))**2 + w_t_out**2)
    W_max = (w_in + w_out + delta_W) / 2
    D_eq = W_max / w_out
    W_sep = w_out if D_eq <= 2.0 else w_out * D_eq / 2
    
    dh_mix = 0.5 * (W_sep - W_out)**2
    return dh_mix