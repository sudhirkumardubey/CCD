"""
Vaneless diffuser loss models
"""

from ..registry import LossModelRegistry, LossContext


@LossModelRegistry.register("vaneless_diffuser", "stanitz")
def diffuser_stanitz(context: LossContext) -> float:
    """
    Vaneless diffuser losses according to Stanitz
    """
    T0_in = context. inlet_state.T0
    p0_in = context.inlet_state.p0
    v_in = context.velocity_triangle["v_in"]
    cp = context.inlet_state.cp
    cv = context.inlet_state.cv
    p_out = context.outlet_state.p
    p0_out = context.outlet_state.p0
    
    gamma = cp / cv
    alpha = (gamma - 1) / gamma
    Y_tot = cp * T0_in * ((p_out / p0_out)**alpha - (p_out / p0_in)**alpha)
    
    # Return scaled loss
    scale = 0.5 * v_in**2
    return Y_tot / scale